//! SMT-backed learner for SPPs and [`Cand`]s, using Z3.
//!
//! The learner accumulates abstract clauses and extracts, for each allocated
//! slot, a concrete decision diagram consistent with all of them.  Slots come
//! in two flavours:
//!
//! * **SPP slots** ([`SmtLearner::fresh_spp`]) — learned via
//!   [`crate::spp::learner::Learner`].
//! * **Cand slots** ([`SmtLearner::fresh_cand`]) — learned via
//!   [`Cand::from_examples`]; each is tied to the [`ExplicitDFA`] it predicates
//!   over.
//!
//! Abstract packet bits may be:
//!
//! * [`AbstractBit::Concrete`] — a fixed `false`/`true`, or
//! * [`AbstractBit::Exist`] — a globally scoped existential boolean variable
//!   (same `id` shares the same Z3 const across all clauses).
//!
//! DFA state values in a [`Literal::Cand`] are always concrete (`usize`).
//!
//! There are **no universally quantified bits**.  Universal quantifiers cause
//! Z3 to choose interpretations where the uninterpreted function collapses to a
//! constant via the `else` value of its function interpretation; that hands the
//! inductive bias to Z3 rather than to the BDD-style learners, which is the
//! wrong shape.  Adding universals back is a future MBQI-style outer CEGIS loop.
//!
//! Internally:
//! 1. Declare one uninterpreted function per slot meaning "this concrete input
//!    is a member".  SPP slots have signature `Bool^{2n} -> Bool`; Cand slots
//!    have `Bool^{n} (pkt_in) x Bool^{n} (pkt_start) x Int^{N} (states) x
//!    Bool^{n} (pkt_end) x Bool^{n} (pkt_out) -> Bool`, where `N` is the DFA's
//!    state count.
//! 2. Each clause becomes a disjunction of literals, each an application of the
//!    slot's function (or its negation when polarity is false).
//! 3. After Z3 solves, [`z3::FuncInterp::get_entries`] gives a list of concrete
//!    inputs, fed to the appropriate BDD-style learner.
//!
//! `model.compact = false` is set globally so each function interpretation
//! enumerates every input that the constraints touch; the `else` default is
//! discarded.  This is sound because in the quantifier-free encoding every
//! application is at a concrete (or Z3-decided existential) pattern, so each
//! constrained input appears as an explicit entry.

use z3::{
    FuncDecl, SatResult, Solver, Sort,
    ast::{Ast, Bool, Dynamic, Int},
    set_global_param,
};

use crate::holes::aut::ExplicitDFA;
use crate::holes::cand::{Cand, Input};
use crate::sp::{SP, SPstore};
use crate::spp::learner::{AbstractPacket as ConcretePacket, Learner};
use crate::spp::{SPP, SPPstore, Var};

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Opaque handle to an existential boolean variable owned by the learner.
/// Obtain one via [`SmtLearner::fresh_existential`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Existential(u32);

/// Opaque handle to a learnable SPP slot owned by the learner.
/// Obtain one via [`SmtLearner::fresh_spp`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SppVar(u32);

/// Opaque handle to a learnable [`Cand`] slot owned by the learner.
/// Obtain one via [`SmtLearner::fresh_cand`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandVar(u32);

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

/// A literal: an abstract membership assertion about one slot, plus a polarity.
///
/// `polarity = true` means "the input is a member"; `false` means "not a
/// member".
pub enum Literal {
    /// A membership literal over an [`SppVar`]: is the pair `(ap1, ap2)` in the
    /// SPP?
    Spp {
        spp: SppVar,
        ap1: AbstractPacket,
        ap2: AbstractPacket,
        polarity: bool,
    },
    /// A membership literal over a [`CandVar`]: does the candidate accept this
    /// (abstract) [`Input`]?  `states` is concrete.
    Cand {
        cand: CandVar,
        pkt_in: AbstractPacket,
        pkt_start: AbstractPacket,
        states: Vec<usize>,
        pkt_end: AbstractPacket,
        pkt_out: AbstractPacket,
        polarity: bool,
    },
}

/// A disjunction of literals.  Literals may target a mix of SPP and Cand slots.
pub struct AbstractClause {
    pub literals: Vec<Literal>,
}

/// The result of a successful [`SmtLearner::extract`]: one learned diagram per
/// allocated slot.
pub struct Solution<'a> {
    pub spps: HashMap<SppVar, SPP>,
    pub cands: HashMap<CandVar, Cand<'a>>,
}

/// Returned by [`SmtLearner::extract`] when the clauses are mutually
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

pub struct SmtLearner<'a> {
    num_vars: Var,
    solver: Solver,
    /// One uninterpreted function `Bool^{2n} -> Bool` per learnable SPP,
    /// indexed by [`SppVar::0`].
    spps: Vec<FuncDecl>,
    /// One uninterpreted function per learnable [`Cand`], paired with the DFA
    /// it predicates over, indexed by [`CandVar::0`].
    cands: Vec<(FuncDecl, &'a ExplicitDFA)>,
    existentials: Vec<Bool>,
}

impl<'a> SmtLearner<'a> {
    /// Create a new learner for packets with `num_vars` fields.  No slots or
    /// existentials exist yet; allocate them with [`Self::fresh_spp`],
    /// [`Self::fresh_cand`], and [`Self::fresh_existential`].
    pub fn new(num_vars: Var) -> Self {
        set_global_param("model.compact", "false");
        Self {
            num_vars,
            solver: Solver::new(),
            spps: Vec::new(),
            cands: Vec::new(),
            existentials: Vec::new(),
        }
    }

    /// Allocate a fresh learnable SPP slot.  The returned [`SppVar`] can be
    /// referenced in [`Literal::Spp`].
    pub fn fresh_spp(&mut self) -> SppVar {
        let id = self.spps.len() as u32;
        let bool_sort = Sort::bool();
        let domain_refs: Vec<&Sort> = (0..2 * self.num_vars).map(|_| &bool_sort).collect();
        let f = FuncDecl::new(format!("spp_mem_{}", id), &domain_refs, &Sort::bool());
        self.spps.push(f);
        SppVar(id)
    }

    /// Allocate a fresh learnable [`Cand`] slot predicating over `dfa`.  The
    /// returned [`CandVar`] can be referenced in [`Literal::Cand`].
    pub fn fresh_cand(&mut self, dfa: &'a ExplicitDFA) -> CandVar {
        let id = self.cands.len() as u32;
        let n = self.num_vars;
        let ns = dfa.num_states() as u32;
        let bool_sort = Sort::bool();
        let int_sort = Sort::int();
        // pkt_in, pkt_start (bools), states (ints), pkt_end, pkt_out (bools).
        let mut domain_refs: Vec<&Sort> = Vec::with_capacity((4 * n + ns) as usize);
        domain_refs.extend((0..2 * n).map(|_| &bool_sort));
        domain_refs.extend((0..ns).map(|_| &int_sort));
        domain_refs.extend((0..2 * n).map(|_| &bool_sort));
        let f = FuncDecl::new(format!("cand_mem_{}", id), &domain_refs, &Sort::bool());
        self.cands.push((f, dfa));
        CandVar(id)
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

    /// Z3 argument for one abstract bit.
    fn bit_arg(&self, b: &AbstractBit) -> Dynamic {
        match b {
            AbstractBit::Concrete(v) => Dynamic::from_ast(&Bool::from_bool(*v)),
            AbstractBit::Exist(e) => Dynamic::from_ast(&self.exist(*e)),
        }
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
            let applied = match lit {
                Literal::Spp {
                    spp,
                    ap1,
                    ap2,
                    polarity,
                } => {
                    assert_eq!(ap1.len(), n, "ap1 length mismatch");
                    assert_eq!(ap2.len(), n, "ap2 length mismatch");
                    let mut args: Vec<Dynamic> = Vec::with_capacity(2 * n);
                    for side in [ap1, ap2] {
                        for b in side {
                            args.push(self.bit_arg(b));
                        }
                    }
                    let arg_refs: Vec<&dyn Ast> = args.iter().map(|a| a as &dyn Ast).collect();
                    let f = &self.spps[spp.0 as usize];
                    let a = f.apply(&arg_refs).as_bool().expect("f has Bool range");
                    if *polarity { a } else { a.not() }
                }
                Literal::Cand {
                    cand,
                    pkt_in,
                    pkt_start,
                    states,
                    pkt_end,
                    pkt_out,
                    polarity,
                } => {
                    let (f, dfa) = &self.cands[cand.0 as usize];
                    let ns = dfa.num_states();
                    assert_eq!(pkt_in.len(), n, "pkt_in length mismatch");
                    assert_eq!(pkt_start.len(), n, "pkt_start length mismatch");
                    assert_eq!(pkt_end.len(), n, "pkt_end length mismatch");
                    assert_eq!(pkt_out.len(), n, "pkt_out length mismatch");
                    assert_eq!(states.len(), ns, "states length mismatch");
                    let mut args: Vec<Dynamic> = Vec::with_capacity(4 * n + ns);
                    for b in pkt_in {
                        args.push(self.bit_arg(b));
                    }
                    for b in pkt_start {
                        args.push(self.bit_arg(b));
                    }
                    for &s in states {
                        // A state value may be the `CompletedDfa` sink (`ns`),
                        // one past the real states `0..ns`.
                        assert!(s <= ns, "state value out of range");
                        args.push(Dynamic::from_ast(&Int::from_u64(s as u64)));
                    }
                    for b in pkt_end {
                        args.push(self.bit_arg(b));
                    }
                    for b in pkt_out {
                        args.push(self.bit_arg(b));
                    }
                    let arg_refs: Vec<&dyn Ast> = args.iter().map(|a| a as &dyn Ast).collect();
                    let a = f.apply(&arg_refs).as_bool().expect("f has Bool range");
                    if *polarity { a } else { a.not() }
                }
            };
            lit_asts.push(applied);
        }

        let body = match lit_asts.len() {
            0 => Bool::from_bool(false),
            1 => lit_asts.into_iter().next().unwrap(),
            _ => Bool::or(&lit_asts),
        };
        self.solver.assert(&body);
    }

    /// Solve once and extract a concrete diagram for each allocated slot.
    ///
    /// Each SPP comes from feeding its function's [`z3::FuncInterp`] entries
    /// into an independent [`crate::spp::learner::Learner`]; each [`Cand`] from
    /// feeding its entries into [`Cand::from_examples`].  Both inherit the
    /// BDD-style inductive bias of those learners.
    pub fn extract(&mut self, spp_store: &mut SPPstore) -> Result<Solution<'a>, InconsistentError> {
        match self.solver.check() {
            SatResult::Unsat => return Err(InconsistentError),
            SatResult::Unknown => panic!("Z3 returned unknown"),
            SatResult::Sat => {}
        }
        let model = self.solver.get_model().expect("model after Sat");
        let n = self.num_vars as usize;

        // SPP slots.
        let mut spps = HashMap::with_capacity(self.spps.len());
        for (idx, f) in self.spps.iter().enumerate() {
            let mut learner = Learner::new(self.num_vars);

            // If this UF was never constrained, there is no func interp;
            // we feed no examples and the learner extracts the zero SPP.
            if let Some(interp) = model.get_func_interp(f) {
                for entry in interp.get_entries() {
                    let args = entry.get_args();
                    let polarity = read_polarity(&entry.get_value());
                    let bits: Vec<bool> = args.iter().map(read_bit).collect();
                    assert_eq!(bits.len(), 2 * n);
                    let ap1: ConcretePacket = bits[..n].iter().map(|&b| Some(b)).collect();
                    let ap2: ConcretePacket = bits[n..].iter().map(|&b| Some(b)).collect();
                    learner
                        .add_example(ap1, ap2, polarity)
                        .expect("Z3 model is internally consistent");
                }
            }

            spps.insert(SppVar(idx as u32), learner.extract(spp_store));
        }

        // Cand slots.
        let mut cands = HashMap::with_capacity(self.cands.len());
        for (idx, (f, dfa)) in self.cands.iter().enumerate() {
            let ns = dfa.num_states();
            let mut examples: Vec<(Input, bool)> = Vec::new();

            if let Some(interp) = model.get_func_interp(f) {
                for entry in interp.get_entries() {
                    let args = entry.get_args();
                    assert_eq!(args.len(), 4 * n + ns);
                    let polarity = read_polarity(&entry.get_value());
                    let pkt_in = args[..n].iter().map(read_bit).collect();
                    let pkt_start = args[n..2 * n].iter().map(read_bit).collect();
                    let states: Vec<usize> =
                        args[2 * n..2 * n + ns].iter().map(read_state).collect();
                    let pkt_end = args[2 * n + ns..3 * n + ns].iter().map(read_bit).collect();
                    let pkt_out = args[3 * n + ns..4 * n + ns].iter().map(read_bit).collect();
                    examples.push((
                        Input {
                            pkt_in,
                            pkt_start,
                            states,
                            pkt_end,
                            pkt_out,
                        },
                        polarity,
                    ));
                }
            }

            let cand = Cand::from_examples(spp_store, dfa, &examples)
                .expect("Z3 model is internally consistent");
            cands.insert(CandVar(idx as u32), cand);
        }

        Ok(Solution { spps, cands })
    }
}

/// Read a concrete `bool` out of a func-interp argument/value.
fn read_bit(d: &Dynamic) -> bool {
    d.as_bool()
        .and_then(|b| b.as_bool())
        .expect("entry arg is a concrete Bool")
}

/// Read a concrete state index out of a func-interp argument.
fn read_state(d: &Dynamic) -> usize {
    d.as_int()
        .and_then(|i| i.as_u64())
        .expect("entry arg is a concrete Int") as usize
}

/// Read a concrete polarity out of a func-interp entry value.
fn read_polarity(d: &Dynamic) -> bool {
    d.as_bool()
        .and_then(|b| b.as_bool())
        .expect("entry value is a concrete Bool")
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

    fn spp_lit(spp: SppVar, ap1: AbstractPacket, ap2: AbstractPacket, polarity: bool) -> Literal {
        Literal::Spp {
            spp,
            ap1,
            ap2,
            polarity,
        }
    }

    #[test]
    fn test_empty() {
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        let result = learner.extract(&mut store).unwrap();
        assert_eq!(result.spps[&s], store.zero);
    }

    #[test]
    fn test_empty_no_spps() {
        // No SPP slots allocated: extract returns an empty map.
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
        let result = learner.extract(&mut store).unwrap();
        assert!(result.spps.is_empty());
    }

    #[test]
    fn test_single_positive_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                ap_concrete(&[false, false, false]),
                ap_concrete(&[true, true, true]),
                true,
            )],
        });
        let spp = learner.extract(&mut store).unwrap().spps[&s];
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
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                ap_concrete(&[true, true, true]),
                ap_concrete(&[false, false, false]),
                false,
            )],
        });
        let spp = learner.extract(&mut store).unwrap().spps[&s];
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
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        let ap1 = ap_concrete(&[false, false, false]);
        let ap2 = ap_concrete(&[true, false, false]);
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(s, ap1.clone(), ap2.clone(), true)],
        });
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(s, ap1, ap2, false)],
        });
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn test_disjunctive_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        // (0,0,0)->(0,0,0) is in SPP  OR  (1,1,1)->(1,1,1) is not in SPP
        learner.add_clause(AbstractClause {
            literals: vec![
                spp_lit(
                    s,
                    ap_concrete(&[false, false, false]),
                    ap_concrete(&[false, false, false]),
                    true,
                ),
                spp_lit(
                    s,
                    ap_concrete(&[true, true, true]),
                    ap_concrete(&[true, true, true]),
                    false,
                ),
            ],
        });
        let spp = learner.extract(&mut store).unwrap().spps[&s];
        let a = spp_accepts(&store, spp, &[false, false, false], &[false, false, false]);
        let b = spp_accepts(&store, spp, &[true, true, true], &[true, true, true]);
        assert!(a || !b, "clause disjunction not satisfied");
    }

    #[test]
    fn test_existential_only() {
        // (0, 0, e0) -> (1, 1, e0) is in the SPP, for some choice of e0.
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        let e0 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                vec![c(false), c(false), AbstractBit::Exist(e0)],
                vec![c(true), c(true), AbstractBit::Exist(e0)],
                true,
            )],
        });
        let spp = learner.extract(&mut store).unwrap().spps[&s];
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
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        let e0 = learner.fresh_existential();
        let e1 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                vec![AbstractBit::Exist(e0), AbstractBit::Exist(e1), c(false)],
                ap_concrete(&[false, false, false]),
                true,
            )],
        });
        learner.add_existential_clause(&[(e0, true), (e1, true)]);
        learner.add_existential_clause(&[(e0, false)]);
        let spp = learner.extract(&mut store).unwrap().spps[&s];
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
        let mut learner = SmtLearner::new(N);
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
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        let e0 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                vec![AbstractBit::Exist(e0), c(false), c(false)],
                ap_concrete(&[false, false, false]),
                true,
            )],
        });
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                ap_concrete(&[true, false, false]),
                ap_concrete(&[false, false, false]),
                false,
            )],
        });
        let spp = learner.extract(&mut store).unwrap().spps[&s];
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
        let mut learner = SmtLearner::new(N);
        let s1 = learner.fresh_spp();
        let s2 = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s1,
                ap_concrete(&[false, false, false]),
                ap_concrete(&[true, true, true]),
                true,
            )],
        });
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s2,
                ap_concrete(&[false, false, false]),
                ap_concrete(&[false, false, false]),
                false,
            )],
        });
        let result = learner.extract(&mut store).unwrap();
        assert!(spp_accepts(
            &store,
            result.spps[&s1],
            &[false, false, false],
            &[true, true, true],
        ));
        assert!(!spp_accepts(
            &store,
            result.spps[&s2],
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
        let mut learner = SmtLearner::new(N);
        let _s = learner.fresh_spp();
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(store.sp.one, &e, &store.sp);
        assert!(learner.extract(&mut store).is_ok());
    }

    #[test]
    fn test_sp_membership_empty_set_is_unsat() {
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
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
        let mut learner = SmtLearner::new(N);
        let s = learner.fresh_spp();
        let target = vec![true, false, true];
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        let pkt_sp = singleton_sp_for_test(&mut store, &target);
        learner.add_sp_membership(pkt_sp, &e, &store.sp);

        // Literal: s accepts (existentials, [false, true, false]).  With
        // existentials pinned to `target`, this forces (target, [false,true,false]) ∈ s.
        let out = vec![false, true, false];
        learner.add_clause(AbstractClause {
            literals: vec![spp_lit(
                s,
                e.iter().map(|&ev| AbstractBit::Exist(ev)).collect(),
                out.iter().map(|&b| AbstractBit::Concrete(b)).collect(),
                true,
            )],
        });

        let result = learner.extract(&mut store).unwrap();
        assert!(spp_accepts(&store, result.spps[&s], &target, &out));
    }

    #[test]
    fn test_clause_spanning_two_spps() {
        // s1 accepts (0,0,0)→(0,0,0)  OR  s2 does NOT accept (1,1,1)→(1,1,1).
        // A disjunction across distinct SPPs.  The learner must satisfy *one*.
        let mut store = SPPstore::new(N);
        let mut learner = SmtLearner::new(N);
        let s1 = learner.fresh_spp();
        let s2 = learner.fresh_spp();
        learner.add_clause(AbstractClause {
            literals: vec![
                spp_lit(
                    s1,
                    ap_concrete(&[false, false, false]),
                    ap_concrete(&[false, false, false]),
                    true,
                ),
                spp_lit(
                    s2,
                    ap_concrete(&[true, true, true]),
                    ap_concrete(&[true, true, true]),
                    false,
                ),
            ],
        });
        let result = learner.extract(&mut store).unwrap();
        let a = spp_accepts(
            &store,
            result.spps[&s1],
            &[false, false, false],
            &[false, false, false],
        );
        let b = spp_accepts(
            &store,
            result.spps[&s2],
            &[true, true, true],
            &[true, true, true],
        );
        assert!(a || !b, "no disjunct satisfied across two SPPs");
    }

    #[test]
    fn test_cand_single_positive() {
        // A single positive Cand literal must be reproduced by the learned Cand.
        let mut store = SPPstore::new(N);
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.zero],
        };
        let mut learner = SmtLearner::new(N);
        let cv = learner.fresh_cand(&dfa);
        learner.add_clause(AbstractClause {
            literals: vec![Literal::Cand {
                cand: cv,
                pkt_in: ap_concrete(&[false, false, false]),
                pkt_start: ap_concrete(&[true, false, true]),
                states: vec![0],
                pkt_end: ap_concrete(&[true, true, true]),
                pkt_out: ap_concrete(&[false, false, false]),
                polarity: true,
            }],
        });
        let solution = learner.extract(&mut store).unwrap();
        let cand = &solution.cands[&cv];
        assert!(cand.accepts_input(
            &mut store,
            &Input {
                pkt_in: vec![false, false, false],
                pkt_start: vec![true, false, true],
                states: vec![0],
                pkt_end: vec![true, true, true],
                pkt_out: vec![false, false, false],
            },
        ));
    }
}
