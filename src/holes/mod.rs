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
//! use katch2::aut::Aut;
//! use katch2::expr::Expr;
//! use katch2::holes::aut::aut_to_dfa;
//! use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
//! use katch2::holes::solve_holes;
//!
//! // Build the target as a concrete DFA via the holeless `Aut`.
//! let mut aut = Aut::new(2);
//! let target = Expr::sequence(Expr::dup(), Expr::assign(0, true));
//! let target_state = aut.expr_to_state(&target);
//! let target_dfa = aut_to_dfa(&mut aut, target_state);
//!
//! // Hole-bearing expression sharing the same SPP store.
//! let store = aut.spp_store_mut();
//! let hole_expr = HExpr::sequence(HExpr::dup(), HExpr::hole(Hole(0)));
//!
//! // Lower-bound = upper-bound = target  →  forces equation.
//! let result = solve_holes(
//!     &hole_expr, &[Hole(0)], &target_dfa, &target_dfa, store,
//! ).unwrap();
//! let h0 = result[&Hole(0)];
//!
//! // The unique Hole(0) is `x0 := 1`: any input maps to itself with bit 0 = 1.
//! for in_b1 in [false, true] {
//!     for in_b0 in [false, true] {
//!         let inp  = [in_b0, in_b1];
//!         let outp = [true,  in_b1];
//!         assert!(
//!             store.accepts(h0, &inp, &outp),
//!             "expected ({:?})→({:?}) in synthesized hole", inp, outp,
//!         );
//!     }
//! }
//! // And no extension that flips bit 0 the wrong way.
//! assert!(!store.accepts(h0, &[true, true], &[false, true]));
//! assert!(!store.accepts(h0, &[false, false], &[false, true]));
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
//! use katch2::aut::Aut;
//! use katch2::expr::Expr;
//! use katch2::holes::aut::aut_to_dfa;
//! use katch2::holes::cegis::CegisError;
//! use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
//! use katch2::holes::solve_holes;
//!
//! let mut aut = Aut::new(2);
//! let target = Expr::dup();
//! let target_state = aut.expr_to_state(&target);
//! let target_dfa = aut_to_dfa(&mut aut, target_state);
//!
//! let store = aut.spp_store_mut();
//! let hole_expr = HExpr::hole(Hole(0));   // no `dup` — length-1 traces only
//!
//! let result = solve_holes(
//!     &hole_expr, &[Hole(0)], &target_dfa, &target_dfa, store,
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
