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
//! * [`cegis`] — the synthesis loop ([`cegis::run`]).  It takes a list of
//!   [`cegis::Constraint`]s — each relating a hole-bearing automaton to a
//!   concrete DFA as a lower bound, upper bound, or equality — plus a
//!   freestanding `reference_dfa` to ground the candidates, and drives the
//!   [`crate::holes::smt::SmtLearner`] with clauses derived from each
//!   constraint check until one assignment satisfies them all.
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
//!     &hole_expr, &[Hole(0)], &target_dfa, &target_dfa, &mut store, None,
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
//!     &hole_expr, &[Hole(0)], &target, &target, &mut store, None,
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
//! # use katch2::holes::{full_reference_dfa, solve_holes_full};
//! # use katch2::spp::SPPstore;
//! let mut store = SPPstore::new(2);
//! let target = expr_to_dfa(&Expr::dup(), &mut store);
//! let hole_expr = HExpr::hole(Hole(0));
//! // The reference DFA grounds the Cand candidates; the caller owns it.
//! let rdfa = full_reference_dfa(&mut store, &hole_expr, &target, &target);
//! let result = solve_holes_full(
//!     &hole_expr, &[Hole(0)], &target, &target, &rdfa, &mut store, None,
//! );
//! assert!(result.is_ok());
//! ```

pub mod aut;
pub mod cand;
pub mod candidate;
pub mod cegis;
pub mod inst;
pub mod nk_with_holes;
pub mod parser;
pub mod smt;

use std::collections::HashMap;

use crate::spp;
use aut::{ExplicitDFA, ops};
use candidate::Candidate;
use cegis::{CegisError, Constraint, run};
use nk_with_holes::{Expr, Hole};

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
pub fn solve_holes_general<'a, C: Candidate<'a>>(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &ExplicitDFA,
    upper_bound: &ExplicitDFA,
    reference_dfa: &'a ExplicitDFA,
    store: &mut spp::SPPstore,
    max_iters: Option<usize>,
) -> Result<HashMap<Hole, C>, CegisError> {
    // `expr` must satisfy two constraints at once: contained in the upper
    // bound and containing the lower bound.  Candidates are grounded on the
    // caller-owned `reference_dfa` (see [`bounds_reference_dfa`]), which they
    // may borrow, so it must outlive the returned candidates.
    let constraints = bounds_constraints(store, expr, lower_bound, upper_bound);
    run::<C>(&constraints, holes, reference_dfa, store, max_iters)
}

/// The upper/lower bound constraint pair for `lower ⊆ expr[holes] ⊆ upper`.
fn bounds_constraints(
    store: &mut spp::SPPstore,
    expr: &Expr,
    lower_bound: &ExplicitDFA,
    upper_bound: &ExplicitDFA,
) -> Vec<Constraint> {
    vec![
        Constraint::upper_bound(store, expr, upper_bound.clone()),
        Constraint::lower_bound(store, expr, lower_bound.clone()),
    ]
}

/// Build the reference DFA that grounds candidates of kind `C` for the
/// `lower ⊆ expr[holes] ⊆ upper` problem (see [`Candidate::make_reference_dfa`]).
///
/// The caller must own the result for as long as the synthesized candidates
/// are alive, since [`cand::Cand`] candidates borrow it.
pub fn bounds_reference_dfa<'a, C: Candidate<'a>>(
    store: &mut spp::SPPstore,
    expr: &Expr,
    lower_bound: &ExplicitDFA,
    upper_bound: &ExplicitDFA,
) -> ExplicitDFA {
    let constraints = bounds_constraints(store, expr, lower_bound, upper_bound);
    C::make_reference_dfa(store, &constraints)
}

/// Reference DFA for [`solve_holes_full`]: the [`bounds_reference_dfa`]
/// specialized to the full (`SPP ∪ Cand`) candidate kind.
pub fn full_reference_dfa(
    store: &mut spp::SPPstore,
    expr: &Expr,
    lower_bound: &ExplicitDFA,
    upper_bound: &ExplicitDFA,
) -> ExplicitDFA {
    bounds_reference_dfa::<ops::Union<spp::SPP, cand::Cand<'_>>>(
        store,
        expr,
        lower_bound,
        upper_bound,
    )
}

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound`, filling each hole with a
/// single dup-free [`spp::SPP`].
///
/// This is the dup-free specialization of [`solve_holes_general`].
pub fn solve_holes(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &ExplicitDFA,
    upper_bound: &ExplicitDFA,
    store: &mut spp::SPPstore,
    max_iters: Option<usize>,
) -> Result<HashMap<Hole, spp::SPP>, CegisError> {
    // SPP candidates don't borrow the reference DFA, so we can own it locally.
    let reference_dfa = bounds_reference_dfa::<spp::SPP>(store, expr, lower_bound, upper_bound);
    solve_holes_general(
        expr,
        holes,
        lower_bound,
        upper_bound,
        &reference_dfa,
        store,
        max_iters,
    )
}

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound`, filling each hole with an
/// arbitrary, possibly dup-ful candidate: a union of a dup-free [`spp::SPP`]
/// and a richer [`cand::Cand`] that pulls its states from `reference_dfa`.
///
/// This searches a strictly larger space than [`solve_holes`], so it can solve
/// problems that solver reports as [`CegisError::Infeasible`] (e.g. a hole that
/// must equal `dup`).
///
/// `reference_dfa` is the caller-owned DFA the [`cand::Cand`] candidates draw
/// their states from; build it with [`full_reference_dfa`].  It must outlive
/// the returned candidates (they borrow it).
pub fn solve_holes_full<'a>(
    expr: &Expr,
    holes: &[Hole],
    lower_bound: &ExplicitDFA,
    upper_bound: &ExplicitDFA,
    reference_dfa: &'a ExplicitDFA,
    store: &mut spp::SPPstore,
    max_iters: Option<usize>,
) -> Result<HashMap<Hole, ops::Union<spp::SPP, cand::Cand<'a>>>, CegisError> {
    solve_holes_general(
        expr,
        holes,
        lower_bound,
        upper_bound,
        reference_dfa,
        store,
        max_iters,
    )
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
            Some(FUZZ_MAX_ITERS),
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
            Some(FUZZ_MAX_ITERS),
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
    use katch2::holes::{full_reference_dfa, solve_holes_full};

    /// `solve_holes_full` should solve a problem that `solve_holes` already
    /// handles with a plain dup-free SPP: `dup ; Hole(0) == dup ; (x0 := 1)`.
    /// The synthesized `Hole(0)` is the assignment `x0 := 1`.
    #[test]
    fn full_solves_spp_problem() {
        let mut store = SPPstore::new(2);
        let target = Expr::sequence(Expr::dup(), Expr::assign(0, true));
        let target_dfa = expr_to_dfa(&target, &mut store);
        let hole_expr = HExpr::sequence(HExpr::dup(), HExpr::hole(Hole(0)));
        let rdfa = full_reference_dfa(&mut store, &hole_expr, &target_dfa, &target_dfa);
        let result = solve_holes_full(
            &hole_expr,
            &[Hole(0)],
            &target_dfa,
            &target_dfa,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
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
        let spp_result = solve_holes(
            &hole_expr,
            &[Hole(0)],
            &target,
            &target,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
        assert!(matches!(spp_result, Err(CegisError::Infeasible)));

        // The full solver can realize `dup` with a dup-ful candidate.
        let rdfa = full_reference_dfa(&mut store, &hole_expr, &target, &target);
        let full_result = solve_holes_full(
            &hole_expr,
            &[Hole(0)],
            &target,
            &target,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
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
        let rdfa = full_reference_dfa(&mut store, &hole_expr, &lower, &upper);
        let result = solve_holes_full(
            &hole_expr,
            &[Hole(0)],
            &lower,
            &upper,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
        assert!(matches!(result, Err(CegisError::Infeasible)));
    }

    // ---- randomized round-trip fuzzing of the hole solvers ------------
    //
    // We never hand a solver a problem we don't already know is satisfiable.
    // The recipe:
    //
    //   1. Pick a random program for each hole, drawn from the candidate kind
    //      the solver searches: *dup-free* (a single SPP) for `solve_holes`,
    //      possibly *dup-ful* (a multi-step sub-program) for `solve_holes_full`.
    //   2. Build a random hole-bearing expression and, in lock-step, the
    //      concrete target obtained by substituting each hole's program for the
    //      hole.  (The hole-bearing side may contain `dup`; that combinator
    //      lives identically in both worlds.)
    //   3. Ask the solver to solve `hole_expr == target`, under an iteration
    //      cap so a slow/diverging instance can't hang the suite.
    //
    // Because the per-hole programs we substituted are themselves a witness,
    // the equation is satisfiable, so the solver must *never* report
    // `Infeasible`.  It may return `Ok` (found an assignment — any assignment
    // inside `target ⊆ . ⊆ target` is correct) or, if the search is slow, hit
    // the cap (`IterationLimit`).  Both are acceptable; only `Infeasible` (or a
    // panic) is a bug.  Capping low keeps the suite fast while still catching
    // those bugs — the search is known to blow up on some instances
    // (see `roundtrip_three_hole_chain_blowup`), and the cap turns that into
    // a tolerated `IterationLimit` rather than a (very long) wait.

    use katch2::expr::Exp;
    use std::collections::BTreeSet;

    /// Field count for the fuzzer.  Small keeps the SPPs (and DFAs) tiny.
    const FUZZ_FIELDS: u32 = 2;

    /// Iteration cap for fuzz solves.  Low enough to stay fast even when the
    /// solver diverges, high enough that genuinely-converging instances (which
    /// finish in a few dozen rounds) still converge.
    const FUZZ_MAX_ITERS: usize = 64;

    /// A random dup-free expression: leaves are `0`/`1`/`test`/`assign`, joined
    /// by `union`/`sequence`/`star`.  No `dup`, so it denotes a single SPP and
    /// is a valid `solve_holes` candidate.
    fn random_dup_free(depth: u32) -> Exp {
        if depth == 0 || rand::random::<f64>() < 0.4 {
            return match rand::random_range(0..4u32) {
                0 => Expr::one(),
                1 => Expr::zero(),
                2 => Expr::test(rand::random_range(0..FUZZ_FIELDS), rand::random()),
                _ => Expr::assign(rand::random_range(0..FUZZ_FIELDS), rand::random()),
            };
        }
        match rand::random_range(0..3u32) {
            0 => Expr::union(random_dup_free(depth - 1), random_dup_free(depth - 1)),
            1 => Expr::sequence(random_dup_free(depth - 1), random_dup_free(depth - 1)),
            _ => Expr::star(random_dup_free(depth - 1)),
        }
    }

    /// A random expression that *may* contain `dup`, so it can denote a
    /// multi-step, dup-ful program.  Leaves add `dup` to the `random_dup_free`
    /// repertoire; combinators are the same `union`/`sequence`/`star`.  This is
    /// the candidate kind `solve_holes_full` searches.
    fn random_maybe_dupful(depth: u32) -> Exp {
        if depth == 0 || rand::random::<f64>() < 0.4 {
            return match rand::random_range(0..5u32) {
                0 => Expr::one(),
                1 => Expr::zero(),
                2 => Expr::test(rand::random_range(0..FUZZ_FIELDS), rand::random()),
                3 => Expr::assign(rand::random_range(0..FUZZ_FIELDS), rand::random()),
                _ => Expr::dup(),
            };
        }
        match rand::random_range(0..3u32) {
            0 => Expr::union(
                random_maybe_dupful(depth - 1),
                random_maybe_dupful(depth - 1),
            ),
            1 => Expr::sequence(
                random_maybe_dupful(depth - 1),
                random_maybe_dupful(depth - 1),
            ),
            _ => Expr::star(random_maybe_dupful(depth - 1)),
        }
    }

    /// Build a random hole-bearing expression together with the concrete target
    /// it becomes once each `Hole(i)` is replaced by `insts[i]`.  Every
    /// primitive leaf is either a hole or `dup`, so all concrete packet-shuffling
    /// in the target flows in through the instantiations.  Records which holes
    /// actually appear in `used`.
    fn random_template(depth: u32, insts: &[Exp], used: &mut BTreeSet<Hole>) -> (HExpr, Exp) {
        if depth == 0 || rand::random::<f64>() < 0.4 {
            // A leaf: usually a hole, occasionally a `dup`.
            if rand::random::<f64>() < 0.85 {
                let i = rand::random_range(0..insts.len());
                used.insert(Hole(i as u32));
                return (HExpr::hole(Hole(i as u32)), insts[i].clone());
            }
            return (HExpr::dup(), Expr::dup());
        }
        match rand::random_range(0..3u32) {
            0 => {
                let (a1, a2) = random_template(depth - 1, insts, used);
                let (b1, b2) = random_template(depth - 1, insts, used);
                (HExpr::union(a1, b1), Expr::union(a2, b2))
            }
            1 => {
                let (a1, a2) = random_template(depth - 1, insts, used);
                let (b1, b2) = random_template(depth - 1, insts, used);
                (HExpr::sequence(a1, b1), Expr::sequence(a2, b2))
            }
            _ => {
                let (a1, a2) = random_template(depth - 1, insts, used);
                (HExpr::star(a1), Expr::star(a2))
            }
        }
    }

    /// Build a random satisfiable instance: a hole-bearing expression, the
    /// substituted target, the holes that actually appear, and the per-hole
    /// instantiations (kept for failure messages).  `inst_gen` chooses the
    /// candidate kind — `random_dup_free` or `random_maybe_dupful`.
    fn random_instance(
        template_depth: u32,
        inst_depth: u32,
        inst_gen: fn(u32) -> Exp,
    ) -> (HExpr, Exp, Vec<Hole>, Vec<Exp>) {
        let num_holes = 1 + rand::random_range(0..3usize); // 1..=3 holes
        let insts: Vec<Exp> = (0..num_holes).map(|_| inst_gen(inst_depth)).collect();
        let mut used = BTreeSet::new();
        let (hole_expr, target_expr) = random_template(template_depth, &insts, &mut used);
        let holes: Vec<Hole> = used.into_iter().collect();
        (hole_expr, target_expr, holes, insts)
    }

    /// Shared assertion: on a known-satisfiable instance the solver must never
    /// report [`CegisError::Infeasible`] (a soundness bug) and must not panic.
    /// `Ok` (found a solution) and `Err(IterationLimit)` (hit the cap — slow or
    /// diverging) are both acceptable.  `result` is the solver's outcome with
    /// its assignment dropped (the two solvers return different assignment
    /// types, so we compare on the `Err` shape only).
    fn assert_not_infeasible(
        trial: usize,
        solver: &str,
        result: Result<(), CegisError>,
        hole_expr: &HExpr,
        target_expr: &Exp,
        holes: &[Hole],
        insts: &[Exp],
    ) {
        assert!(
            !matches!(result, Err(CegisError::Infeasible)),
            "trial {trial}: {solver} wrongly reported Infeasible on a known-satisfiable instance\n  \
             hole_expr = {hole_expr:?}\n  target    = {target_expr:?}\n  \
             insts     = {insts:?}\n  holes     = {holes:?}\n  result    = {result:?}"
        );
    }

    /// Fast `solve_holes` round-trip: small dup-free fillings, many trials.
    ///
    /// Asserts `solve_holes` never wrongly reports `Infeasible` on a
    /// satisfiable-by-construction instance (a cap-hit `IterationLimit` is
    /// tolerated as merely slow).  This used to fail intermittently — a dup-free
    /// hole inside `dup`/`dup*` structure could be wrongly rejected — but that
    /// bug is now fixed, so the fuzzer runs by default again.
    #[test]
    fn fuzz_solve_holes_roundtrip() {
        for trial in 0..40 {
            let mut store = SPPstore::new(FUZZ_FIELDS);
            let (hole_expr, target_expr, holes, insts) = random_instance(3, 2, random_dup_free);
            let target_dfa = expr_to_dfa(&target_expr, &mut store);
            let result = solve_holes(
                &hole_expr,
                &holes,
                &target_dfa,
                &target_dfa,
                &mut store,
                Some(FUZZ_MAX_ITERS),
            )
            .map(|_| ());
            assert_not_infeasible(
                trial,
                "solve_holes",
                result,
                &hole_expr,
                &target_expr,
                &holes,
                &insts,
            );
        }
    }

    /// Heavier `solve_holes` round-trip: deeper expressions.  Slower, so opt-in.
    #[test]
    #[ignore]
    fn fuzz_solve_holes_roundtrip_deep() {
        for trial in 0..40 {
            let mut store = SPPstore::new(FUZZ_FIELDS);
            let (hole_expr, target_expr, holes, insts) = random_instance(4, 3, random_dup_free);
            let target_dfa = expr_to_dfa(&target_expr, &mut store);
            let result = solve_holes(
                &hole_expr,
                &holes,
                &target_dfa,
                &target_dfa,
                &mut store,
                Some(FUZZ_MAX_ITERS),
            )
            .map(|_| ());
            assert_not_infeasible(
                trial,
                "solve_holes",
                result,
                &hole_expr,
                &target_expr,
                &holes,
                &insts,
            );
        }
    }

    /// Randomized `solve_holes_full` round-trip: fillings may be *dup-ful*
    /// (multi-step), exercising the `Union<SPP, Cand>` candidate space.
    ///
    /// The iteration cap means a slow/diverging instance surfaces as a tolerated
    /// `IterationLimit` rather than a hang.  Asserts `solve_holes_full` never
    /// wrongly reports `Infeasible` on a satisfiable-by-construction instance.
    /// The known wrong-`Infeasible` bugs are now fixed (the `Cand` ENFA stall,
    /// resolved by grounding candidates on a *complete* reference DFA, and the
    /// lower-bound hole-site trace misalignment in `cegis::collect_hole_sites`),
    /// so this runs by default again.
    #[test]
    fn fuzz_solve_holes_full_roundtrip() {
        for trial in 0..30 {
            let mut store = SPPstore::new(FUZZ_FIELDS);
            let (hole_expr, target_expr, holes, insts) = random_instance(3, 2, random_maybe_dupful);
            let target_dfa = expr_to_dfa(&target_expr, &mut store);
            let rdfa = full_reference_dfa(&mut store, &hole_expr, &target_dfa, &target_dfa);
            let result = solve_holes_full(
                &hole_expr,
                &holes,
                &target_dfa,
                &target_dfa,
                &rdfa,
                &mut store,
                Some(FUZZ_MAX_ITERS),
            )
            .map(|_| ());
            assert_not_infeasible(
                trial,
                "solve_holes_full",
                result,
                &hole_expr,
                &target_expr,
                &holes,
                &insts,
            );
        }
    }

    /// Heavier `solve_holes_full` round-trip: deeper expressions.  No longer
    /// finds wrong-`Infeasible`s, but stays `#[ignore]`d (opt-in) because the
    /// deeper, larger generated DFAs make it slow and can occasionally trip the
    /// `crate::aut` derivative-size limit — environmental, not a solver bug.
    #[test]
    #[ignore]
    fn fuzz_solve_holes_full_roundtrip_deep() {
        for trial in 0..30 {
            let mut store = SPPstore::new(FUZZ_FIELDS);
            let (hole_expr, target_expr, holes, insts) = random_instance(4, 3, random_maybe_dupful);
            let target_dfa = expr_to_dfa(&target_expr, &mut store);
            let rdfa = full_reference_dfa(&mut store, &hole_expr, &target_dfa, &target_dfa);
            let result = solve_holes_full(
                &hole_expr,
                &holes,
                &target_dfa,
                &target_dfa,
                &rdfa,
                &mut store,
                Some(FUZZ_MAX_ITERS),
            )
            .map(|_| ());
            assert_not_infeasible(
                trial,
                "solve_holes_full",
                result,
                &hole_expr,
                &target_expr,
                &holes,
                &insts,
            );
        }
    }

    /// Regression test for a case the fuzzer first found: a hole filled with a
    /// *nondeterministic* SPP (`(0:=false) + (1==true)`, whose summands overlap
    /// when field 1 is true).  Counterexample-trace elaboration used to
    /// forward-simulate greedily and could follow the wrong transition, tripping
    /// an internal assertion in `aut::elaborate_step`/`elaborate_tail`.  The
    /// equation is satisfiable by construction, so `solve_holes` must succeed.
    #[test]
    fn roundtrip_nondeterministic_hole_fill() {
        let mut store = SPPstore::new(2);
        // hole_expr = (Hole(1)* ; Hole(2)*)*
        let hole_expr = HExpr::star(HExpr::sequence(
            HExpr::star(HExpr::hole(Hole(1))),
            HExpr::star(HExpr::hole(Hole(2))),
        ));
        // Hole(1) := (1==true ; 1==false)*  (denotes identity),
        // Hole(2) := (0:=false) + (1==true)  (a nondeterministic SPP).
        let h1 = Expr::star(Expr::sequence(Expr::test(1, true), Expr::test(1, false)));
        let h2 = Expr::union(Expr::assign(0, false), Expr::test(1, true));
        let target = Expr::star(Expr::sequence(Expr::star(h1), Expr::star(h2)));
        let target_dfa = expr_to_dfa(&target, &mut store);
        let result = solve_holes(
            &hole_expr,
            &[Hole(1), Hole(2)],
            &target_dfa,
            &target_dfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
        assert!(result.is_ok(), "{result:?}");
    }

    /// CEGIS **blowup** on a chain of three sequenced holes (2 fields, target as
    /// both bounds). The search blows up combinatorially (like a pigeonhole
    /// instance for a SAT solver): it terminates in principle, but in practice
    /// the cap usually surfaces it as `IterationLimit` rather than
    /// `Ok`/`Infeasible`:
    ///
    /// - `H0;H2;H2 == dup` (`solve_holes_full`) — sat; slow
    /// - `H0;H1;H2 == 1` (`solve_holes`) — sat; slow
    /// - `H0;H0;H0 == 1` (`solve_holes`) — sat; fast
    /// - `H0;H0;H0 == dup` (`solve_holes_full`) — unsat; slow
    /// - `H0;H0;H0 == dup;dup;dup` (`solve_holes_full`) — sat; slow
    ///
    /// The exact round count is sensitive to which counterexample each check
    /// happens to return (per-process `HashMap` ordering) and to the SMT
    /// model choices, so a "slow" satisfiable instance may occasionally
    /// converge within the cap.  We therefore only require that the
    /// satisfiable cases never report `Infeasible` (returning either `Ok` or a
    /// cap-hit `IterationLimit`), and that the unsatisfiable case never reports
    /// `Ok`.
    #[test]
    fn roundtrip_three_hole_chain_blowup() {
        // `Ha ; Hb ; Hc`
        let chain = |a: Hole, b: Hole, c: Hole| {
            HExpr::sequence(
                HExpr::sequence(HExpr::hole(a), HExpr::hole(b)),
                HExpr::hole(c),
            )
        };
        let dup = Expr::dup();
        let dup2 = Expr::sequence(dup.clone(), dup.clone());
        let dup3 = Expr::sequence(dup2.clone(), dup.clone());
        // Run `chain == target` (target as both bounds) under the chosen solver.
        let run = |full: bool, he: &HExpr, holes: &[Hole], target: &Expr| {
            let mut store = SPPstore::new(2);
            let dfa = expr_to_dfa(target, &mut store);
            if full {
                let rdfa = full_reference_dfa(&mut store, he, &dfa, &dfa);
                solve_holes_full(
                    he,
                    holes,
                    &dfa,
                    &dfa,
                    &rdfa,
                    &mut store,
                    Some(FUZZ_MAX_ITERS),
                )
                .map(|_| ())
            } else {
                solve_holes(he, holes, &dfa, &dfa, &mut store, Some(FUZZ_MAX_ITERS)).map(|_| ())
            }
        };

        // A satisfiable instance must never be reported `Infeasible`; `Ok` and a
        // cap-hit `IterationLimit` are both acceptable.
        let assert_sat = |r: Result<(), CegisError>| {
            assert!(
                matches!(r, Ok(()) | Err(CegisError::IterationLimit)),
                "satisfiable instance wrongly reported {r:?}"
            );
        };

        let [h0, h1, h2] = [Hole(0), Hole(1), Hole(2)];
        assert_sat(run(true, &chain(h0, h2, h2), &[h0, h2], &dup));
        assert_sat(run(false, &chain(h0, h1, h2), &[h0, h1, h2], &Expr::one()));
        assert_sat(run(false, &chain(h0, h0, h0), &[h0], &Expr::one()));
        // Unsatisfiable: must never converge to `Ok` (cap-hit or `Infeasible`).
        assert!(run(true, &chain(h0, h0, h0), &[h0], &dup).is_err());
        assert_sat(run(true, &chain(h0, h0, h0), &[h0], &dup3));
    }

    /// This is unsatisfiable:
    ///
    /// ```text
    /// Hole(0); Hole(0) == dup             (one field)
    /// ```
    #[test]
    fn sqrt_of_dup() {
        let mut store = SPPstore::new(1);
        // hole_expr = Hole(0) ; Hole(0)
        let hole_expr = HExpr::sequence(HExpr::hole(Hole(0)), HExpr::hole(Hole(0)));
        let target = expr_to_dfa(&Expr::dup(), &mut store);

        let rdfa = full_reference_dfa(&mut store, &hole_expr, &target, &target);
        let result = solve_holes_full(
            &hole_expr,
            &[Hole(0)],
            &target,
            &target,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
        assert!(matches!(result, Err(CegisError::Infeasible)));
    }

    /// Regression test: `solve_holes_full` synthesizes a hole that must emit
    /// **two chained `dup`s**:
    ///
    /// ```text
    /// Hole(0)  ==  dup ; dup
    /// ```
    ///
    /// This once wrongly returned `Infeasible` (the `Cand` ENFA stalled and
    /// couldn't realize a length-≥3 hole; see
    /// [`crate::holes::cand`]'s `top_cand_over_two_dups_reaches_length_three`).
    /// Fixed by grounding `Cand` on a *complete* reference DFA (the upper bound
    /// completed via [`crate::holes::aut::ops::WithSinkState`]); it now solves
    /// (witness `dup; dup`).
    #[test]
    fn full_solves_chained_dup() {
        let mut store = SPPstore::new(1);
        let target = Expr::sequence(Expr::dup(), Expr::dup());
        let target_dfa = expr_to_dfa(&target, &mut store);
        let hole_expr = HExpr::hole(Hole(0));
        let rdfa = full_reference_dfa(&mut store, &hole_expr, &target_dfa, &target_dfa);
        let result = solve_holes_full(
            &hole_expr,
            &[Hole(0)],
            &target_dfa,
            &target_dfa,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
        assert!(result.is_ok(), "{:?}", result.map(|_| ()));
    }

    /// Regression test for a bug found by `fuzz_solve_holes_roundtrip`.
    ///
    /// ```text
    /// (dup* ; (Hole(0) ; dup))*  ==  (dup* ; (0:=false ; dup))*       (2 fields)
    /// ```
    ///
    /// This was caused by a bug in [`crate::holes::aut::elaborate_step`], which
    /// has been fixed.
    #[test]
    fn dup_in_a_loop() {
        let mut store = SPPstore::new(2);
        // (dup* ; (Hole(0) ; dup))*
        let hole_expr = HExpr::star(HExpr::sequence(
            HExpr::star(HExpr::dup()),
            HExpr::sequence(HExpr::hole(Hole(0)), HExpr::dup()),
        ));
        // target with Hole(0) := 0:=false
        let target = Expr::star(Expr::sequence(
            Expr::star(Expr::dup()),
            Expr::sequence(Expr::assign(0, false), Expr::dup()),
        ));
        let target_dfa = expr_to_dfa(&target, &mut store);
        let result = solve_holes(
            &hole_expr,
            &[Hole(0)],
            &target_dfa,
            &target_dfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        )
        .map(|_| ());
        result.expect("should have a solution");
    }

    /// Regression test for a bug found by `fuzz_solve_holes_full_roundtrip`.
    ///
    /// ```text
    /// Hole(0)*  ==  (0:=true; dup)*        (1 field)
    /// ```
    ///
    /// Caused by a bug in [`cegis::collect_hole_sites`], which has been fixed.
    #[test]
    fn full_solves_assign_dup_under_star() {
        let mut store = SPPstore::new(1);
        // Hole(0)*  ==  (0:=true; dup)*
        let hole_expr = HExpr::star(HExpr::hole(Hole(0)));
        let target = Expr::star(Expr::sequence(Expr::assign(0, true), Expr::dup()));
        let target_dfa = expr_to_dfa(&target, &mut store);
        let rdfa = full_reference_dfa(&mut store, &hole_expr, &target_dfa, &target_dfa);
        let result = solve_holes_full(
            &hole_expr,
            &[Hole(0)],
            &target_dfa,
            &target_dfa,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        );
        result.expect("should have a solution");
    }

    /// Regression test for a bug found by `fuzz_solve_holes_full_roundtrip`.
    ///
    /// ```text
    /// ((Hole(1) ∪ Hole(0)) ; Hole(0)) ∪ Hole(0)  ==  target        (2 fields)
    /// Hole(0) := dup;0:=false,   Hole(1) := dup   (target is that substitution)
    /// ```
    ///
    /// Caused by a bug in [`crate::holes::aut::elaborate_step`], which has been
    /// fixed.
    #[test]
    fn repro_fuzz_wrong_infeasible() {
        // Hole programs (the fuzzer's, simplified: dup∪dup = dup, dup;1 = dup).
        let h0: Exp = Expr::sequence(Expr::dup(), Expr::assign(0, false)); // dup ; 0:=false
        let h1: Exp = Expr::dup();

        // hole_expr = ((H1 ∪ H0) ; H0) ∪ H0
        let hole_expr = HExpr::union(
            HExpr::sequence(
                HExpr::union(HExpr::hole(Hole(1)), HExpr::hole(Hole(0))),
                HExpr::hole(Hole(0)),
            ),
            HExpr::hole(Hole(0)),
        );
        // target = hole_expr with H0 := h0, H1 := h1.
        let target = Expr::union(
            Expr::sequence(Expr::union(h1.clone(), h0.clone()), h0.clone()),
            h0.clone(),
        );

        let mut store = SPPstore::new(2);
        let target_dfa = expr_to_dfa(&target, &mut store);
        let rdfa = full_reference_dfa(&mut store, &hole_expr, &target_dfa, &target_dfa);
        let result = solve_holes_full(
            &hole_expr,
            &[Hole(0), Hole(1)],
            &target_dfa,
            &target_dfa,
            &rdfa,
            &mut store,
            Some(FUZZ_MAX_ITERS),
        )
        .map(|_| ());
        assert_ne!(result, Err(CegisError::Infeasible), "wrong Infeasible");
    }
}
