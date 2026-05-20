//! CEGIS loop: synthesize a concrete SPP for a single hole such that
//! `lower_bound ⊆ expr[hole] ⊆ upper_bound`.
//!
//! # Algorithm
//!
//! 1. Ask the [`ExistentialLearner`] for a candidate SPP.
//! 2. Plug it into the [`Instantiate`] for the hole-bearing expression.
//! 3. Check `expr[candidate] ⊆ upper_bound` via
//!    [`Instantiate::check_less_than`].  If it fails, the witness is a list
//!    of `(hole, (in, out))` pairs from a single counterexample trace; we
//!    convert it into a negative-literal clause for the learner: "the hole
//!    cannot accept all of these pairs simultaneously".
//! 4. Check `lower_bound ⊆ expr[candidate]` via
//!    [`Instantiate::check_greater_than`].  If it fails, we'd need to add
//!    clauses involving existential variables — that processing is stubbed.
//! 5. If both checks pass, return the candidate.
//! 6. Otherwise, ask the learner for a refined candidate and loop.  If the
//!    learner returns `InconsistentError`, the original problem is
//!    infeasible.
//!
//! # Limitations
//!
//! * Single-hole only — the [`ExistentialLearner`] currently learns one SPP.
//! * Lower-bound counterexample processing is a stub
//!   ([`Instantiate::check_greater_than`] is `todo!()`); callers that
//!   exercise it will panic.

use std::collections::HashMap;

use crate::holes::aut::{DFA, NFA};
use crate::holes::inst::Instantiate;
use crate::holes::nk_with_holes::{AutWithHoles, Hole, State};
use crate::spp;
use crate::spp::existential_learner::{AbstractBit, AbstractClause, ExistentialLearner, Literal};

/// Why the CEGIS loop gave up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CegisError {
    /// The accumulated clauses became unsatisfiable: no SPP value for the
    /// hole can simultaneously satisfy the lower and upper bounds.
    Infeasible,
}

/// Solve `lower_bound ⊆ expr[hole] ⊆ upper_bound` for a single hole.
///
/// `aut` and `start` describe the hole-bearing expression as a
/// [`crate::holes::nk_with_holes::AutWithHoles`] state machine; `hole` is the
/// (single) hole label that appears in it.  Returns the synthesized SPP on
/// success.
pub fn run<L: NFA, U: DFA>(
    aut: AutWithHoles,
    start: State,
    hole: Hole,
    lower_bound: &L,
    upper_bound: &U,
    store: &mut spp::SPPstore,
) -> Result<spp::SPP, CegisError> {
    let mut learner = ExistentialLearner::new(store.num_vars());

    let mut candidate = match learner.extract(store) {
        Ok(c) => c,
        Err(_) => return Err(CegisError::Infeasible),
    };

    let mut holes_map = HashMap::new();
    holes_map.insert(hole, candidate);
    let mut inst = Instantiate::new(aut, start, holes_map);

    loop {
        match inst.check_less_than(store, upper_bound) {
            Ok(()) => match inst.check_greater_than(store, lower_bound) {
                Ok(()) => return Ok(candidate),
                Err(cex) => add_lower_bound_clauses(cex, &mut learner),
            },
            Err(witnesses) => add_upper_bound_clause(witnesses, &mut learner),
        }

        candidate = match learner.extract(store) {
            Ok(c) => c,
            Err(_) => return Err(CegisError::Infeasible),
        };
        inst.set_hole(hole, candidate);
    }
}

/// Convert an upper-bound counterexample (a list of `(hole, (in, out))` pairs
/// recorded along a single violating trace) into a clause for the learner.
///
/// Semantics: each pair `(in_i, out_i)` is a place the trace relied on the
/// hole accepting that pair.  To kill the counterexample, the hole's SPP
/// must *not* accept at least one of them.  That's a disjunction of
/// negative literals — exactly one [`AbstractClause`].
///
/// An empty witness vec means the violation was purely concrete (no hole
/// involvement).  Adding an empty clause makes the learner immediately
/// UNSAT, which is correct: no choice of hole can resolve a concrete
/// violation.
fn add_upper_bound_clause(
    witnesses: Vec<(Hole, (Vec<bool>, Vec<bool>))>,
    learner: &mut ExistentialLearner,
) {
    let literals = witnesses
        .into_iter()
        .map(|(_h, (p1, p2))| Literal {
            ap1: p1.into_iter().map(AbstractBit::Concrete).collect(),
            ap2: p2.into_iter().map(AbstractBit::Concrete).collect(),
            polarity: false,
        })
        .collect();
    learner.add_clause(AbstractClause { literals });
}

/// Convert a lower-bound counterexample into clauses for the learner.
///
/// **Stub**: turns the counterexample into existential-variable-bearing
/// clauses.  Will panic until implemented.
fn add_lower_bound_clauses(
    _cex: crate::holes::inst::LowerBoundCounterexample,
    _learner: &mut ExistentialLearner,
) {
    todo!("lower-bound clause generation not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holes::aut::ExplicitDFA;
    use crate::holes::nk_with_holes::Expr;

    fn mk_store() -> spp::SPPstore {
        spp::SPPstore::new(3)
    }

    fn zero_dfa(store: &spp::SPPstore) -> ExplicitDFA {
        ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.zero],
        }
    }

    fn top_dfa(store: &spp::SPPstore) -> ExplicitDFA {
        ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.top],
        }
    }

    /// 0 ⊆ Hole ⊆ top: trivially solvable.  The learner returns the empty
    /// SPP (its default when unconstrained), the upper-bound check passes
    /// vacuously, the lower-bound check passes vacuously, done.
    #[test]
    fn trivial_bounds_converge_immediately() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &Expr::hole(Hole(0)));
        let lb = zero_dfa(&store);
        let ub = top_dfa(&store);
        let spp = run(aut, start, Hole(0), &lb, &ub, &mut store).unwrap();
        assert_eq!(spp, store.zero);
    }

    /// Expression is just a concrete `top`, no hole used.  With ub = zero,
    /// the upper-bound check finds a *concrete* violation: witness is
    /// empty, which produces an empty clause, which makes the learner
    /// immediately UNSAT → `Infeasible`.
    #[test]
    fn concrete_violation_is_infeasible() {
        let mut store = mk_store();
        let top = store.top;
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &Expr::spp(top));
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let err = run(aut, start, Hole(0), &lb, &ub, &mut store).unwrap_err();
        assert_eq!(err, CegisError::Infeasible);
    }

    /// `Hole(0)` with ub = zero: the only valid value is `store.zero`.  The
    /// learner's empty extraction is already zero, so we converge on the
    /// first iteration without ever needing to refine.
    #[test]
    fn hole_bounded_above_by_zero_yields_zero() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &Expr::hole(Hole(0)));
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let spp = run(aut, start, Hole(0), &lb, &ub, &mut store).unwrap();
        assert_eq!(spp, store.zero);
    }
}
