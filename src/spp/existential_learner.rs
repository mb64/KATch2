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

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Opaque handle to an existential boolean variable owned by the learner.
/// Obtain one via [`ExistentialLearner::fresh_existential`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Existential(u32);

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

/// A literal — an abstract packet pair plus a polarity.
///
/// `polarity = true` means "the pair is in the SPP"; `false` means "the pair
/// is not in the SPP".
pub struct Literal {
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
    f: FuncDecl,
    existentials: Vec<Bool>,
}

impl ExistentialLearner {
    /// Create a new learner for packets with `num_vars` fields.
    pub fn new(num_vars: Var) -> Self {
        set_global_param("model.compact", "false");
        let bool_sort = Sort::bool();
        let domain_refs: Vec<&Sort> = (0..2 * num_vars).map(|_| &bool_sort).collect();
        let f = FuncDecl::new("spp_mem", &domain_refs, &Sort::bool());
        Self {
            num_vars,
            solver: Solver::new(),
            f,
            existentials: Vec::new(),
        }
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
            let applied = self.f.apply(&arg_refs).as_bool().expect("f has Bool range");
            lit_asts.push(if lit.polarity { applied } else { applied.not() });
        }

        let body = match lit_asts.len() {
            0 => Bool::from_bool(false),
            1 => lit_asts.into_iter().next().unwrap(),
            _ => Bool::or(&lit_asts),
        };
        self.solver.assert(&body);
    }

    /// Solve and extract an SPP consistent with every added clause.
    pub fn extract(&mut self, spp_store: &mut SPPstore) -> Result<SPP, InconsistentError> {
        match self.solver.check() {
            SatResult::Unsat => return Err(InconsistentError),
            SatResult::Unknown => panic!("Z3 returned unknown"),
            SatResult::Sat => {}
        }
        let model = self.solver.get_model().expect("model after Sat");
        let n = self.num_vars as usize;
        let mut learner = Learner::new(self.num_vars);

        // If `f` is never constrained (e.g. no clauses), there is no func interp.
        if let Some(interp) = model.get_func_interp(&self.f) {
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

        Ok(learner.extract(spp_store))
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
        let spp = learner.extract(&mut store).unwrap();
        assert_eq!(spp, store.zero);
    }

    #[test]
    fn test_single_positive_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: ap_concrete(&[false, false, false]),
                ap2: ap_concrete(&[true, true, true]),
                polarity: true,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();
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
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: ap_concrete(&[true, true, true]),
                ap2: ap_concrete(&[false, false, false]),
                polarity: false,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();
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
        let ap1 = ap_concrete(&[false, false, false]);
        let ap2 = ap_concrete(&[true, false, false]);
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: ap1.clone(),
                ap2: ap2.clone(),
                polarity: true,
            }],
        });
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
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
        // (0,0,0)->(0,0,0) is in SPP  OR  (1,1,1)->(1,1,1) is not in SPP
        learner.add_clause(AbstractClause {
            literals: vec![
                Literal {
                    ap1: ap_concrete(&[false, false, false]),
                    ap2: ap_concrete(&[false, false, false]),
                    polarity: true,
                },
                Literal {
                    ap1: ap_concrete(&[true, true, true]),
                    ap2: ap_concrete(&[true, true, true]),
                    polarity: false,
                },
            ],
        });
        let spp = learner.extract(&mut store).unwrap();
        let a = spp_accepts(&store, spp, &[false, false, false], &[false, false, false]);
        let b = spp_accepts(&store, spp, &[true, true, true], &[true, true, true]);
        assert!(a || !b, "clause disjunction not satisfied");
    }

    #[test]
    fn test_existential_only() {
        // (0, 0, e0) -> (1, 1, e0) is in the SPP, for some choice of e0.
        let mut store = SPPstore::new(N);
        let mut learner = ExistentialLearner::new(N);
        let e0 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: vec![c(false), c(false), AbstractBit::Exist(e0)],
                ap2: vec![c(true), c(true), AbstractBit::Exist(e0)],
                polarity: true,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();
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
        let e0 = learner.fresh_existential();
        let e1 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: vec![AbstractBit::Exist(e0), AbstractBit::Exist(e1), c(false)],
                ap2: ap_concrete(&[false, false, false]),
                polarity: true,
            }],
        });
        learner.add_existential_clause(&[(e0, true), (e1, true)]);
        learner.add_existential_clause(&[(e0, false)]);
        let spp = learner.extract(&mut store).unwrap();
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
        let e0 = learner.fresh_existential();
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: vec![AbstractBit::Exist(e0), c(false), c(false)],
                ap2: ap_concrete(&[false, false, false]),
                polarity: true,
            }],
        });
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: ap_concrete(&[true, false, false]),
                ap2: ap_concrete(&[false, false, false]),
                polarity: false,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();
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
}
