//! NetKAT with holes — synthesizing concrete SPPs for unknown sub-expressions.
//!
//! # Overview
//!
//! A [`nk_with_holes::Expr`] is a NetKAT expression whose primitives may be
//! either a concrete SPP or an opaque [`nk_with_holes::Hole`].  Given upper
//! and lower bounds on the overall expression's behaviour, the goal is to
//! synthesize one SPP per hole that lands the expression inside the
//! `lower_bound ⊆ expr[holes] ⊆ upper_bound` interval.
//!
//! The submodules cover the pieces:
//!
//! * [`nk_with_holes`] — hole-bearing NetKAT expressions and their Antimirov
//!   automaton ([`nk_with_holes::AutWithHoles`]); states are pairs of an
//!   ε-derivative class and a visibility flag, and edges/output summands
//!   are labelled either `Concrete(spp)` or `Abstract(Hole)`.
//! * [`inst`] — a concrete SPP assignment for each hole, wrapped to expose
//!   the resulting ENFA.  Surfaces both [`inst::Instantiate::check_less_than`]
//!   (upper-bound counterexamples) and [`inst::Instantiate::check_greater_than`]
//!   (lower-bound counterexamples).
//! * [`aut`] — generic ENFA / NFA / DFA infrastructure with the forward
//!   ([`aut::forward_reachable`]) and backward ([`aut::backward_reachable`])
//!   reachability used to mine hole sites.
//! * [`cegis`] — the synthesis loop, driving the
//!   [`crate::spp::existential_learner::ExistentialLearner`] with clauses
//!   derived from each bound check.
//!
//! # Convenience entry point
//!
//! [`solve_holes`] is the high-level helper most callers want: hand it an
//! [`nk_with_holes::Expr`], the hole labels appearing in it, both bounds,
//! and an SPP store; receive the synthesized assignment per hole.
//!
//! # Example: `expr[hole] == target`
//!
//! When the lower and upper bound are the same target automaton, the CEGIS
//! loop solves for an exact equation.  Here we ask: for what `Hole(0)` does
//! `dup ; Hole(0)` equal the target `dup ; (x0 := 1)`?
//!
//! ```
//! # use katch2::expr::Expr;
//! # use katch2::holes::aut::expr_to_dfa;
//! # use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
//! # use katch2::holes::solve_holes;
//! # use katch2::spp::SPPstore;
//! let mut store = SPPstore::new(2);
//! let target = Expr::sequence(Expr::dup(), Expr::assign(0, true));
//! let target_dfa = expr_to_dfa(&target, &mut store);
//! let hole_expr = HExpr::sequence(HExpr::dup(), HExpr::hole(Hole(0)));
//! let result = solve_holes(
//!     &hole_expr, &[Hole(0)], &target_dfa, &target_dfa, &mut store,
//! ).unwrap();
//! let h0 = result[&Hole(0)];
//!
//! // The synthesized Hole(0) is `x0 := 1`: any input maps to itself with bit 0 set.
//! for b1 in [false, true] {
//!     for b0 in [false, true] {
//!         assert!(store.accepts(h0, &[b0, b1], &[true, b1]));
//!     }
//! }
//! assert!(!store.accepts(h0, &[true, true], &[false, true]));
//! ```
//!
//! # Example: infeasible synthesis
//!
//! When the bounds force a behaviour the hole cannot reach, the loop
//! converges to `Infeasible`.  Here we ask for `Hole(0) == dup` — but a
//! hole is filled with an *SPP*, which contributes no trace step, while
//! `dup` is the operation that produces length-2 traces.  No SPP value can
//! bridge this structural mismatch.
//!
//! The lower-bound check fires on the first iteration; site collection
//! finds zero viable hole sites (the only candidate site is `Hole(0)`'s
//! output summand, but `forward_candidate[(start, 1)]` is empty since the
//! expression has no way to advance the trace position), so an empty
//! clause is added and the next extraction is immediately UNSAT.
//!
//! ```
//! # use katch2::expr::Expr;
//! # use katch2::holes::aut::expr_to_dfa;
//! # use katch2::holes::cegis::CegisError;
//! # use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
//! # use katch2::holes::solve_holes;
//! # use katch2::spp::SPPstore;
//! let mut store = SPPstore::new(2);
//! let target = expr_to_dfa(&Expr::dup(), &mut store);
//! let hole_expr = HExpr::hole(Hole(0));   // no `dup` — length-1 traces only
//! let result = solve_holes(
//!     &hole_expr, &[Hole(0)], &target, &target, &mut store,
//! );
//! assert!(matches!(result, Err(CegisError::Infeasible)));
//! ```

pub mod aut;
pub mod cegis;
pub mod inst;
pub mod nk_with_holes;

use std::collections::HashMap;

use crate::holes::aut::{DFA, NFA};
use crate::holes::cegis::{CegisError, run};
use crate::holes::nk_with_holes::{AutWithHoles, Expr, Hole};
use crate::spp;

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound` for the holes appearing
/// in `expr`, returning the synthesized SPP per hole on success.
///
/// Convenience wrapper around [`cegis::run`] that compiles the hole-bearing
/// [`Expr`] into an [`AutWithHoles`] internally.  Every hole label the
/// expression can reach must be listed in `holes` (the loop panics on
/// encountering an unmapped hole when it instantiates the expression).
///
/// Returns [`CegisError::Infeasible`] if the constraints are mutually
/// unsatisfiable (no assignment can land inside the interval).
pub fn solve_holes<L: NFA, U: DFA>(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &L,
    upper_bound: &U,
    store: &mut spp::SPPstore,
) -> Result<HashMap<Hole, spp::SPP>, CegisError> {
    let mut aut = AutWithHoles::new();
    let start = aut.expr_to_state(store, expr);
    run(aut, start, holes, lower_bound, upper_bound, store)
}

#[cfg(test)]
mod test {
    use katch2::expr::Expr;
    use katch2::holes::aut::expr_to_dfa;
    use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
    use katch2::holes::solve_holes;
    use katch2::spp::SPPstore;

    #[test]
    fn two_holes() {
        let mut store = SPPstore::new(1);
        let target_dfa = expr_to_dfa(&Expr::one(), &mut store);
        let spp_0 = store.test(0, true);
        let spp_1 = store.test(0, false);
        let spp_01 = store.union(spp_0, spp_1);
        let hole_expr = HExpr::sequence(
            HExpr::hole(Hole(0)),
            HExpr::sequence(HExpr::spp(spp_01), HExpr::hole(Hole(1))),
        );
        let result = solve_holes(
            &hole_expr,
            &[Hole(0), Hole(1)],
            &target_dfa,
            &target_dfa,
            &mut store,
        );
        assert!(result.is_ok());
    }

    /// The "slow" parameterized test asks:
    ///
    /// ```text
    /// Hole0; (f_0 == 1 || ... || f_n == 1); Hole1 == id
    /// ```
    ///
    /// That is, Hole0 must be an injective function from 2^`n` pigeons into 2^`n` - 1 holes.
    ///
    /// This is of course UNSAT, but figuring that out is slow.
    fn slow(n_fields: u32) {
        let mut store = SPPstore::new(n_fields);
        let target_dfa = expr_to_dfa(&Expr::one(), &mut store);
        let mut spp = store.zero;
        for i in 0..n_fields {
            let spp_i = store.test(i, true);
            spp = store.union(spp, spp_i);
        }
        let hole_expr = HExpr::sequence(
            HExpr::hole(Hole(0)),
            HExpr::sequence(HExpr::spp(spp), HExpr::hole(Hole(1))),
        );
        let result = solve_holes(
            &hole_expr,
            &[Hole(0), Hole(1)],
            &target_dfa,
            &target_dfa,
            &mut store,
        );
        assert!(result.is_err());
    }

    /// 4 pigeons: essentially instant
    #[test]
    fn slightly_slow_test() {
        slow(2);
    }

    /// 8 pigeons: a couple seconds
    #[test]
    fn slow_test() {
        slow(3);
    }

    /// 16 pigeons: impossibly long
    #[test]
    #[ignore]
    fn slower_test() {
        slow(4);
    }
}
