//! NetKAT with holes — synthesizing NetKAT values for unknown sub-expressions.
//!
//! # Overview
//!
//! A [`nk_with_holes::Expr`] is a NetKAT expression whose primitives may be
//! either a concrete SPP or an opaque [`nk_with_holes::Hole`].  Given upper
//! and lower bounds on the overall expression's behaviour, the goal is to
//! synthesize one candidate per hole that lands the expression inside the
//! `lower_bound ⊆ expr[holes] ⊆ upper_bound` interval.
//!
//! What a candidate may be is configurable: at one extreme a candidate is a
//! single dup-free [`spp::SPP`], at the other it is an arbitrary, possibly
//! dup-ful sub-program.  See [`candidate::Candidate`].
//!
//! The submodules cover the pieces:
//!
//! * [`nk_with_holes`] — hole-bearing NetKAT expressions and their Antimirov
//!   automaton ([`nk_with_holes::AutWithHoles`]); states are pairs of an
//!   ε-derivative class and a visibility flag, and edges/output summands
//!   are labelled either `Concrete(spp)` or `Abstract(Hole)`.
//! * [`inst`] — a concrete NFA assignment for each hole, wrapped to expose
//!   the resulting ENFA.  Surfaces both [`inst::Instantiate::check_less_than`]
//!   (upper-bound counterexamples) and [`inst::Instantiate::check_greater_than`]
//!   (lower-bound counterexamples).
//! * [`aut`] — generic ENFA / NFA / DFA infrastructure with the forward
//!   ([`aut::forward_reachable`]) and backward ([`aut::backward_reachable`])
//!   reachability used to mine hole sites.
//! * [`cegis`] — the synthesis loop, driving the
//!   [`crate::holes::smt::SmtLearner`] with clauses
//!   derived from each bound check.
//!
//! # Convenience entry points
//!
//! Both helpers take the same arguments — an [`nk_with_holes::Expr`], the hole
//! labels appearing in it, both bounds, and an SPP store — and return the
//! synthesized assignment per hole.  They differ only in the search space they
//! consider for each hole:
//!
//! * [`solve_holes`] fills every hole with a single dup-free [`spp::SPP`].  It
//!   is the right choice when the holes are meant to be plain
//!   packet-transformations, and it is what most callers want.
//! * [`solve_holes_full`] fills every hole with an arbitrary, possibly dup-ful
//!   sub-program (a union of an [`spp::SPP`] and a richer
//!   [`cand::Cand`]).  Use it when a hole may need to emit `dup`s — that is,
//!   produce traces longer than one step.
//!
//! Because [`solve_holes_full`] searches a strictly larger space, it can solve
//! problems [`solve_holes`] reports as [`cegis::CegisError::Infeasible`] (see
//! the second example below).  Both are thin wrappers over
//! [`solve_holes_general`], which is generic over the candidate kind.
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
//! # Example: `Hole(0) == dup`
//!
//! Here we ask for `Hole(0) == dup`.
//!
//! * Filling holes with [`spp::SPP`]'s, this is, of course, infeasible, since they are `dup`-free.
//!   So [`solve_holes`] gives an error, after just a couple CEGIS iterations.
//!
//! * On the other hand, if you allow `dup`s is is very easily satisfied by just putting `dup` in the
//!   hole. So [`solve_holes_full`] gives a successfull assignment.
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
//!
//! [`solve_holes_full`], whose candidates may be dup-ful, can fill `Hole(0)`
//! with `dup` itself, so the same query succeeds:
//!
//! ```
//! # use katch2::expr::Expr;
//! # use katch2::holes::aut::expr_to_dfa;
//! # use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
//! # use katch2::holes::solve_holes_full;
//! # use katch2::spp::SPPstore;
//! let mut store = SPPstore::new(2);
//! let target = expr_to_dfa(&Expr::dup(), &mut store);
//! let hole_expr = HExpr::hole(Hole(0));
//! let result = solve_holes_full(
//!     &hole_expr, &[Hole(0)], &target, &target, &mut store,
//! );
//! assert!(result.is_ok());
//! ```

pub mod aut;
pub mod cand;
pub mod candidate;
pub mod cegis;
pub mod inst;
pub mod nk_with_holes;
pub mod smt;

use std::collections::HashMap;

use crate::spp;
use aut::{ExplicitDFA, NFA, ops};
use candidate::Candidate;
use cegis::{CegisError, run};
use nk_with_holes::{AutWithHoles, Expr, Hole};

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound` for the holes appearing
/// in `expr`, filling each hole with a candidate of kind `C`.
///
/// This is the generic core behind [`solve_holes`] (where `C` is a dup-free
/// [`spp::SPP`]) and [`solve_holes_full`] (where `C` may be dup-ful); the two
/// public helpers just pin `C`.
///
/// Convenience wrapper around [`cegis::run`] that compiles the hole-bearing
/// [`Expr`] into an [`AutWithHoles`] internally.  Every hole label the
/// expression can reach must be listed in `holes` (the loop panics on
/// encountering an unmapped hole when it instantiates the expression).
///
/// Returns [`CegisError::Infeasible`] when no assignment of kind `C` lands the
/// expression inside the interval — note this is relative to the candidate
/// kind, so a problem infeasible for one `C` may be solvable for a richer one.
pub fn solve_holes_general<'a, L: NFA, C: Candidate<'a>>(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &L,
    upper_bound: &'a ExplicitDFA,
    store: &mut spp::SPPstore,
) -> Result<HashMap<Hole, C>, CegisError> {
    let mut aut = AutWithHoles::new();
    let start = aut.expr_to_state(store, expr);
    run::<C, L>(aut, start, holes, lower_bound, upper_bound, store)
}

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound`, filling each hole with a
/// single dup-free [`spp::SPP`].
///
/// This is the dup-free specialization of [`solve_holes_general`].
pub fn solve_holes<L: NFA>(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &L,
    upper_bound: &ExplicitDFA,
    store: &mut spp::SPPstore,
) -> Result<HashMap<Hole, spp::SPP>, CegisError> {
    solve_holes_general(expr, holes, lower_bound, upper_bound, store)
}

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound`, filling each hole with an
/// arbitrary, possibly dup-ful candidate: a union of a dup-free [`spp::SPP`]
/// and a richer [`cand::Cand`] that pulls its states from `upper_bound`.
///
/// This searches a strictly larger space than [`solve_holes`], so it can solve
/// problems that solver reports as [`CegisError::Infeasible`] (e.g. a hole that
/// must equal `dup`).
pub fn solve_holes_full<'a, L: NFA>(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &L,
    upper_bound: &'a ExplicitDFA,
    store: &mut spp::SPPstore,
) -> Result<HashMap<Hole, ops::Union<spp::SPP, cand::Cand<'a>>>, CegisError> {
    solve_holes_general(expr, holes, lower_bound, upper_bound, store)
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

    use katch2::holes::cegis::CegisError;
    use katch2::holes::solve_holes_full;

    /// `solve_holes_full` should solve a problem that `solve_holes` already
    /// handles with a plain dup-free SPP: `dup ; Hole(0) == dup ; (x0 := 1)`.
    /// The synthesized `Hole(0)` is the assignment `x0 := 1`.
    #[test]
    fn full_solves_spp_problem() {
        let mut store = SPPstore::new(2);
        let target = Expr::sequence(Expr::dup(), Expr::assign(0, true));
        let target_dfa = expr_to_dfa(&target, &mut store);
        let hole_expr = HExpr::sequence(HExpr::dup(), HExpr::hole(Hole(0)));
        let result = solve_holes_full(&hole_expr, &[Hole(0)], &target_dfa, &target_dfa, &mut store);
        assert!(result.is_ok());
    }

    /// `Hole(0) == dup` is infeasible for `solve_holes` (an SPP contributes no
    /// trace step, so it can never produce the length-2 trace of `dup`), but
    /// `solve_holes_full` may fill the hole with a dup-ful candidate, so it
    /// must succeed.  This is the case that exercises the extra power of the
    /// full solver over the SPP-only one.
    #[test]
    fn full_solves_dup_where_spp_cannot() {
        let mut store = SPPstore::new(2);
        let target = expr_to_dfa(&Expr::dup(), &mut store);
        let hole_expr = HExpr::hole(Hole(0));

        // SPP-only is infeasible.
        let spp_result = solve_holes(&hole_expr, &[Hole(0)], &target, &target, &mut store);
        assert!(matches!(spp_result, Err(CegisError::Infeasible)));

        // The full solver can realize `dup` with a dup-ful candidate.
        let full_result = solve_holes_full(&hole_expr, &[Hole(0)], &target, &target, &mut store);
        assert!(full_result.is_ok());
    }

    /// When the bounds themselves are unsatisfiable independent of the hole
    /// (`one ⊆ Hole(0) ⊆ zero`, but `one ⊄ zero`), even the full solver must
    /// report `Infeasible`.
    #[test]
    fn full_infeasible_bounds() {
        let mut store = SPPstore::new(1);
        let lower = expr_to_dfa(&Expr::one(), &mut store);
        let upper = expr_to_dfa(&Expr::zero(), &mut store);
        let hole_expr = HExpr::hole(Hole(0));
        let result = solve_holes_full(&hole_expr, &[Hole(0)], &lower, &upper, &mut store);
        assert!(matches!(result, Err(CegisError::Infeasible)));
    }
}
