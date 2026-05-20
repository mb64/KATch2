//! SPP learner with existential boolean variables, using Z3.
//!
//! Like [`super::clause_learner::ClauseLearner`], this learner accumulates
//! abstract clauses and extracts an SPP that satisfies all of them.  Abstract
//! packet bits here may be:
//!
//! * [`AbstractBit::Concrete`] — a fixed `false`/`true`, or
//! * [`AbstractBit::Exist`] — a globally scoped existential boolean variable
//!   (same `id` shares the same Z3 const across all clauses).
//!
//! There are **no universally quantified bits**.  Universal quantifiers cause
//! Z3 to choose interpretations where the uninterpreted function `f` collapses
//! to a constant via the `else` value of its function interpretation; that
//! interpretation hands the inductive bias to Z3 rather than to
//! [`super::learner::Learner`], which is the wrong shape.  Adding universals
//! back is a future MBQI-style outer CEGIS loop.
//!
//! Internally:
//! 1. Declare one uninterpreted function `f : Bool^{2n} -> Bool` meaning
//!    "this concrete packet pair is in the SPP".
//! 2. Each clause becomes a disjunction `l_0 ∨ l_1 ∨ ...` where each literal
//!    is `f(...)` (or its negation when polarity is false).  Arguments come
//!    from the literal's abstract bits, with existentials shared by id.
//! 3. After Z3 solves, [`z3::FuncInterp::get_entries`] gives a list of
//!    concrete I/O pairs, fed to [`super::learner::Learner`].
//!
//! `model.compact = false` is set globally so `f`'s function interpretation
//! enumerates every pair that the constraints touch; the `else` default is
//! discarded.  This is sound because in the quantifier-free encoding every
//! application of `f` is at a concrete (or Z3-decided existential) bit
//! pattern, so each constrained pair appears as an explicit entry.

use z3::{
    FuncDecl, SatResult, Solver, Sort,
    ast::{Ast, Bool},
    set_global_param,
};

use super::learner::{AbstractPacket as ConcretePacket, Learner};
use super::{SPP, SPPstore, Var};
use crate::sp::{SP, SPstore};

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Opaque handle to an existential boolean variable owned by the learner.
/// Obtain one via [`ExistentialLearner::fresh_existential`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Existential(u32);

/// Opaque handle to a learnable SPP slot owned by the learner.
/// Obtain one via [`ExistentialLearner::fresh_spp`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SppVar(u32);

/// One bit of an abstract packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractBit {
    /// A fixed value (`false` or `true`).
    Concrete(bool),
    /// An existential boolean variable allocated by the learner.
    Exist(Existential),
}

/// An abstract packet: one [`AbstractBit`] per packet field.
pub type AbstractPacket = Vec<AbstractBit>;

/// A literal — an abstract packet pair plus a polarity, targeting a
/// specific learnable SPP.
///
/// `polarity = true` means "the pair is in `spp`"; `false` means "the pair
/// is not in `spp`".
pub struct Literal {
    /// Which learnable SPP this literal constrains.
    pub spp: SppVar,
    pub ap1: AbstractPacket,
    pub ap2: AbstractPacket,
    pub polarity: bool,
}

/// A disjunction of literals.
pub struct AbstractClause {
    pub literals: Vec<Literal>,
}

/// Returned by [`ExistentialLearner::extract`] when the clauses are mutually
/// inconsistent (Z3 returned UNSAT).
#[derive(Debug, Clone)]
pub struct InconsistentError;

impl std::fmt::Display for InconsistentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inconsistent clauses")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Learner
// ────────────────────────────────────────────────────────────────────────────

pub struct ExistentialLearner {
    num_vars: Var,
    solver: Solver,
    /// One uninterpreted function `Bool^{2n} -> Bool` per learnable SPP,
    /// indexed by [`SppVar::0`].
    spps: Vec<FuncDecl>,
    existentials: Vec<Bool>,
}

impl ExistentialLearner {
    /// Create a new learner for packets with `num_vars` fields.  No SPP
    /// slots or existentials exist yet; allocate them with
    /// [`Self::fresh_spp`] and [`Self::fresh_existential`].
    pub fn new(num_vars: Var) -> Self {
        set_global_param("model.compact", "false");
        Self {
            num_vars,
            solver: Solver::new(),
            spps: Vec::new(),
            existentials: Vec::new(),
        }
    }

    /// Allocate a fresh learnable SPP slot.  The returned [`SppVar`] can be
    /// referenced in [`Literal::spp`].
    pub fn fresh_spp(&mut self) -> SppVar {
        let id = self.spps.len() as u32;
        let bool_sort = Sort::bool();
        let domain_refs: Vec<&Sort> = (0..2 * self.num_vars).map(|_| &bool_sort).collect();
        let f = FuncDecl::new(format!("spp_mem_{}", id), &domain_refs, &Sort::bool());
        self.spps.push(f);
        SppVar(id)
    }

    /// Allocate a fresh existential boolean variable.  The returned
    /// [`Existential`] can be embedded in [`AbstractBit::Exist`] or referenced
    /// in [`Self::add_existential_clause`].
    pub fn fresh_existential(&mut self) -> Existential {
        let id = self.existentials.len() as u32;
        self.existentials.push(Bool::new_const(format!("e_{}", id)));
        Existential(id)
    }

    fn exist(&self, e: Existential) -> Bool {
        self.existentials[e.0 as usize].clone()
    }

    /// Add a pure-existential clause: a disjunction of literals, each
    /// `(e, polarity)` meaning `e` (if `polarity`) or `¬e`.
    pub fn add_existential_clause(&mut self, literals: &[(Existential, bool)]) {
        let lits: Vec<Bool> = literals
            .iter()
            .map(|&(e, pol)| {
                let b = self.exist(e);
                if pol { b } else { b.not() }
            })
            .collect();
        let body = match lits.len() {
            0 => Bool::from_bool(false),
            1 => lits.into_iter().next().unwrap(),
            _ => Bool::or(&lits),
        };
        self.solver.assert(&body);
    }

    /// Constrain the bit-vector formed by `vars[0..n]` (where `n` matches the
    /// SP's depth) to represent a packet that lies in `sp`.
    ///
    /// Implemented by recursively walking the SP BDD, building one Z3 Bool
    /// AST per node (`ite(vars[depth], rec(x1), rec(x0))`), and asserting the
    /// result.  Hash-consing on the SP plus a memo table over the recursion
    /// keep the work proportional to the BDD size, not exponential.
    ///
    /// Asserting membership in an empty SP makes the solver immediately UNSAT;
    /// membership in the full SP is a no-op.
    pub fn add_sp_membership(&mut self, sp: SP, vars: &[Existential], sp_store: &SPstore) {
        let mut memo: HashMap<SP, Bool> = HashMap::new();
        let body = self.build_sp_bool(sp, vars, 0, sp_store, &mut memo);
        self.solver.assert(&body);
    }

    fn build_sp_bool(
        &self,
        sp: SP,
        vars: &[Existential],
        depth: usize,
        sp_store: &SPstore,
        memo: &mut HashMap<SP, Bool>,
    ) -> Bool {
        if sp == SP::new(0) {
            return Bool::from_bool(false);
        }
        if sp == SP::new(1) {
            return Bool::from_bool(true);
        }
        if let Some(b) = memo.get(&sp) {
            return b.clone();
        }
        let node = sp_store.get(sp);
        let x0_b = self.build_sp_bool(node.x0, vars, depth + 1, sp_store, memo);
        let x1_b = self.build_sp_bool(node.x1, vars, depth + 1, sp_store, memo);
        let var_b = self.exist(vars[depth]);
        let result = var_b.ite(&x1_b, &x0_b);
        memo.insert(sp, result.clone());
        result
    }

    /// Add an abstract clause.  The corresponding Z3 assertion is added
    /// immediately.
    pub fn add_clause(&mut self, clause: AbstractClause) {
        let n = self.num_vars as usize;

        let mut lit_asts: Vec<Bool> = Vec::with_capacity(clause.literals.len());
        for lit in &clause.literals {
            assert_eq!(lit.ap1.len(), n, "ap1 length mismatch");
            assert_eq!(lit.ap2.len(), n, "ap2 length mismatch");
            let mut args: Vec<Bool> = Vec::with_capacity(2 * n);
            for side in [&lit.ap1, &lit.ap2] {
                for b in side {
                    args.push(match b {
                        AbstractBit::Concrete(v) => Bool::from_bool(*v),
                        AbstractBit::Exist(id) => self.exist(*id),
                    });
                }
            }
            let arg_refs: Vec<&dyn Ast> = args.iter().map(|a| a as &dyn Ast).collect();
            let f = &self.spps[lit.spp.0 as usize];
            let applied = f.apply(&arg_refs).as_bool().expect("f has Bool range");
            lit_asts.push(if lit.polarity { applied } else { applied.not() });
        }

        let body = match lit_asts.len() {
            0 => Bool::from_bool(false),
            1 => lit_asts.into_iter().next().unwrap(),
            _ => Bool::or(&lit_asts),
        };
        self.solver.assert(&body);
    }

    /// Solve once and extract a concrete SPP for each allocated [`SppVar`].
    ///
    /// Returns a map from every [`SppVar`] allocated via [`Self::fresh_spp`]
    /// to a learned SPP.  Each SPP is the result of feeding its UF's
    /// [`z3::FuncInterp`] entries into an independent
    /// [`super::learner::Learner`], inheriting that learner's BDD-style
    /// inductive bias.
    pub fn extract(
        &mut self,
        spp_store: &mut SPPstore,
    ) -> Result<HashMap<SppVar, SPP>, InconsistentError> {
        match self.solver.check() {
            SatResult::Unsat => return Err(InconsistentError),
            SatResult::Unknown => panic!("Z3 returned unknown"),
            SatResult::Sat => {}
        }
        let model = self.solver.get_model().expect("model after Sat");
        let n = self.num_vars as usize;

        let mut result = HashMap::with_capacity(self.spps.len());
        for (idx, f) in self.spps.iter().enumerate() {
            let mut learner = Learner::new(self.num_vars);

            // If this UF was never constrained, there is no func interp;
            // we feed no examples and the learner extracts the zero SPP.
            if let Some(interp) = model.get_func_interp(f) {
                for entry in interp.get_entries() {
                    let args = entry.get_args();
                    let polarity = entry
                        .get_value()
                        .as_bool()
                        .and_then(|b| b.as_bool())
                        .expect("entry value is a concrete Bool");
                    let bits: Vec<bool> = args
                        .iter()
                        .map(|a| {
                            a.as_bool()
                                .and_then(|b| b.as_bool())
                                .expect("entry arg is a concrete Bool")
                        })
                        .collect();
                    assert_eq!(bits.len(), 2 * n);
                    let ap1: ConcretePacket = bits[..n].iter().map(|&b| Some(b)).collect();
                    let ap2: ConcretePacket = bits[n..].iter().map(|&b| Some(b)).collect();
                    learner
                        .add_example(ap1, ap2, polarity)
                        .expect("Z3 model is internally consistent");
                }
            }

            result.insert(SppVar(idx as u32), learner.extract(spp_store));
        }

        Ok(result)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: Var = 3;

    fn spp_accepts(store: &SPPstore, spp: SPP, p1: &[bool], p2: &[bool]) -> bool {
        let mut cur = spp;
        for (&b1, &b2) in p1.iter().zip(p2.iter()) {
            if cur == SPP::new(0) {
                return false;
            }
            if cur == SPP::new(1) {
                return true;
            }
            let node = store.get(cur);
            cur = match (b1, b2) {
                (false, false) => node.x00,
                (false, true) => node.x01,
                (true, false) => node.x10,
                (true, true) => node.x11,
            };
        }
        cur == SPP::new(1)
    }

    fn c(b: bool) -> AbstractBit {
        AbstractBit::Concrete(b)
    }

    fn ap_concrete(bits: &[bool]) -> AbstractPacket {
        bits.iter().map(|&b| c(b)).collect()
    }

    #[test]
    fn test_empty() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        let result = learner.extract(&mut store).unwrap();
        assert_eq!(result[&s], store.zero);
    }

    #[test]
    fn test_empty_no_spps() {
        // No SPP slots allocated: extract returns an empty map.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let result = learner.extract(&mut store).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_positive_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: ap_concrete(&[false, false, false]),
                ap2: ap_concrete(&[true, true, true]),
                polarity: true,
            }],
        });
        let spp = learner.extract(&mut store).unwrap()[&s];
        assert!(spp_accepts(
            &store,
            spp,
            &[false, false, false],
            &[true, true, true],
        ));
    }

    #[test]
    fn test_single_negative_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: ap_concrete(&[true, true, true]),
                ap2: ap_concrete(&[false, false, false]),
                polarity: false,
            }],
        });
        let spp = learner.extract(&mut store).unwrap()[&s];
        assert!(!spp_accepts(
            &store,
            spp,
            &[true, true, true],
            &[false, false, false],
        ));
    }

    #[test]
    fn test_conflicting_unit_clauses() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        let ap1 = ap_concrete(&[false, false, false]);
        let ap2 = ap_concrete(&[true, false, false]);
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: ap1.clone(),
                ap2: ap2.clone(),
                polarity: true,
            }],
        });
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1,
                ap2,
                polarity: false,
            }],
        });
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn test_disjunctive_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        // (0,0,0)->(0,0,0) is in SPP  OR  (1,1,1)->(1,1,1) is not in SPP
        learner.add_clause(AbstractClause {
            literals: vec![
                Literal {
                    spp: s,
                    ap1: ap_concrete(&[false, false, false]),
                    ap2: ap_concrete(&[false, false, false]),
                    polarity: true,
                },
                Literal {
                    spp: s,
                    ap1: ap_concrete(&[true, true, true]),
                    ap2: ap_concrete(&[true, true, true]),
                    polarity: false,
                },
            ],
        });
        let spp = learner.extract(&mut store).unwrap()[&s];
        let a = spp_accepts(&store, spp, &[false, false, false], &[false, false, false]);
        let b = spp_accepts(&store, spp, &[true, true, true], &[true, true, true]);
        assert!(a || !b, "clause disjunction not satisfied");
    }

    #[test]
    fn test_existential_only() {
        // (0, 0, e0) -> (1, 1, e0) is in the SPP, for some choice of e0.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        let e0 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: vec![c(false), c(false), AbstractBit::Exist(e0)],
                ap2: vec![c(true), c(true), AbstractBit::Exist(e0)],
                polarity: true,
            }],
        });
        let spp = learner.extract(&mut store).unwrap()[&s];
        let accepts_false = spp_accepts(&store, spp, &[false, false, false], &[true, true, false]);
        let accepts_true = spp_accepts(&store, spp, &[false, false, true], &[true, true, true]);
        assert!(
            accepts_false || accepts_true,
            "neither e0 choice was realized"
        );
    }

    #[test]
    fn test_existential_clause_constrains_choice() {
        // (e0, e1, 0) -> (0, 0, 0) is in SPP, AND e0 ∨ e1, AND ¬e0.
        // Forces e0=false, e1=true, so (0,1,0)->(0,0,0) must be accepted.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        let e0 = learner.fresh_existential();
        let e1 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: vec![AbstractBit::Exist(e0), AbstractBit::Exist(e1), c(false)],
                ap2: ap_concrete(&[false, false, false]),
                polarity: true,
            }],
        });
        learner.add_existential_clause(&[(e0, true), (e1, true)]);
        learner.add_existential_clause(&[(e0, false)]);
        let spp = learner.extract(&mut store).unwrap()[&s];
        assert!(spp_accepts(
            &store,
            spp,
            &[false, true, false],
            &[false, false, false],
        ));
    }

    #[test]
    fn test_existential_clause_unsat() {
        // e0 AND ¬e0  → inconsistent.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let e0 = learner.fresh_existential();
        learner.add_existential_clause(&[(e0, true)]);
        learner.add_existential_clause(&[(e0, false)]);
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn test_existential_shared_across_clauses() {
        // Clause 1: (e0, 0, 0) -> (0, 0, 0) is in SPP
        // Clause 2: (1, 0, 0) -> (0, 0, 0) is NOT in SPP
        // Only e0 = false makes both consistent.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        let e0 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: vec![AbstractBit::Exist(e0), c(false), c(false)],
                ap2: ap_concrete(&[false, false, false]),
                polarity: true,
            }],
        });
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: ap_concrete(&[true, false, false]),
                ap2: ap_concrete(&[false, false, false]),
                polarity: false,
            }],
        });
        let spp = learner.extract(&mut store).unwrap()[&s];
        assert!(spp_accepts(
            &store,
            spp,
            &[false, false, false],
            &[false, false, false],
        ));
        assert!(!spp_accepts(
            &store,
            spp,
            &[true, false, false],
            &[false, false, false],
        ));
    }

    #[test]
    fn test_two_spps_independent() {
        // Two SPP slots constrained independently: s1 must accept (0,0,0)→(1,1,1),
        // s2 must NOT accept (0,0,0)→(0,0,0).  Both constraints satisfiable
        // in isolation; extract returns one SPP per slot.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s1 = learner.fresh_spp();
        let s2 = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s1,
                ap1: ap_concrete(&[false, false, false]),
                ap2: ap_concrete(&[true, true, true]),
                polarity: true,
            }],
        });
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s2,
                ap1: ap_concrete(&[false, false, false]),
                ap2: ap_concrete(&[false, false, false]),
                polarity: false,
            }],
        });
        let result = learner.extract(&mut store).unwrap();
        assert!(spp_accepts(
            &store,
            result[&s1],
            &[false, false, false],
            &[true, true, true],
        ));
        assert!(!spp_accepts(
            &store,
            result[&s2],
            &[false, false, false],
            &[false, false, false],
        ));
    }

    /// Singleton SP for a concrete packet (test helper, mirrors aut::singleton_sp).
    fn singleton_sp_for_test(store: &mut SPPstore, pkt: &[bool]) -> SP {
        let mut sp_val = SP::new(1);
        let mut zero = SP::new(0);
        for &b in pkt.iter().rev() {
            let next = if b {
                store.sp.mk(zero, sp_val)
            } else {
                store.sp.mk(sp_val, zero)
            };
            zero = store.sp.mk(zero, zero);
            sp_val = next;
        }
        sp_val
    }

    #[test]
    fn test_sp_membership_full_set_is_noop() {
        // sp = full set; adding membership shouldn't change satisfiability.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let _s = learner.fresh_spp();
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(store.sp.one, &e, &store.sp);
        assert!(learner.extract(&mut store).is_ok());
    }

    #[test]
    fn test_sp_membership_empty_set_is_unsat() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let _s = learner.fresh_spp();
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(store.sp.zero, &e, &store.sp);
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn test_sp_membership_singleton_pins_existentials() {
        // Constrain existentials to be a specific packet, then assert a
        // literal that uses them; the resulting SPP must accept that pair.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s = learner.fresh_spp();
        let target = vec![true, false, true];
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        let pkt_sp = singleton_sp_for_test(&mut store, &target);
        learner.add_sp_membership(pkt_sp, &e, &store.sp);

        // Literal: s accepts (existentials, [false, true, false]).  With
        // existentials pinned to `target`, this forces (target, [false,true,false]) ∈ s.
        let out = vec![false, true, false];
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                spp: s,
                ap1: e.iter().map(|&ev| AbstractBit::Exist(ev)).collect(),
                ap2: out.iter().map(|&b| AbstractBit::Concrete(b)).collect(),
                polarity: true,
            }],
        });

        let result = learner.extract(&mut store).unwrap();
        assert!(spp_accepts(&store, result[&s], &target, &out));
    }

    #[test]
    fn test_clause_spanning_two_spps() {
        // s1 accepts (0,0,0)→(0,0,0)  OR  s2 does NOT accept (1,1,1)→(1,1,1).
        // A disjunction across distinct SPPs.  The learner must satisfy *one*.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let s1 = learner.fresh_spp();
        let s2 = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![
                Literal {
                    spp: s1,
                    ap1: ap_concrete(&[false, false, false]),
                    ap2: ap_concrete(&[false, false, false]),
                    polarity: true,
                },
                Literal {
                    spp: s2,
                    ap1: ap_concrete(&[true, true, true]),
                    ap2: ap_concrete(&[true, true, true]),
                    polarity: false,
                },
            ],
        });
        let result = learner.extract(&mut store).unwrap();
        let a = spp_accepts(
            &store,
            result[&s1],
            &[false, false, false],
            &[false, false, false],
        );
        let b = spp_accepts(
            &store,
            result[&s2],
            &[true, true, true],
            &[true, true, true],
        );
        assert!(a || !b, "no disjunct satisfied across two SPPs");
    }
}
