//! CEGIS loop: synthesize a concrete SPP for a single hole such that
//! `lower_bound ⊆ expr[hole] ⊆ upper_bound`.
//!
//! # Algorithm
//!
//! 1. Ask the [`SmtLearner`] for a candidate SPP.
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
//! * Single-hole only — the [`SmtLearner`] currently learns one SPP.
//! * Lower-bound counterexample processing is a stub
//!   ([`Instantiate::check_greater_than`] is `todo!()`); callers that
//!   exercise it will panic.

use std::collections::{HashMap, HashSet};

use crate::holes::aut::{DFA, NFA, backward_reachable, forward_reachable};
use crate::holes::inst::{self, Instantiate, LowerBoundCounterexample};
use crate::holes::nk_with_holes::{AutWithHoles, EdgeLabel, Hole, State};
use crate::holes::smt::{AbstractBit, AbstractClause, Existential, Literal, SmtLearner, SppVar};
use crate::sp;
use crate::spp;

/// Why the CEGIS loop gave up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CegisError {
    /// The accumulated clauses became unsatisfiable: no SPP value for the
    /// hole can simultaneously satisfy the lower and upper bounds.
    Infeasible,
}

/// Solve `lower_bound ⊆ expr[holes] ⊆ upper_bound` for multiple holes.
///
/// `aut` and `start` describe the hole-bearing expression as a
/// [`crate::holes::nk_with_holes::AutWithHoles`] state machine; `holes` is
/// the set of hole labels that may appear in it (every hole the expression
/// reaches must be listed, otherwise [`Instantiate`] will panic when it
/// hits an unmapped one).  On success, returns one synthesized SPP per
/// hole.
pub fn run<L: NFA, U: DFA>(
    aut: AutWithHoles,
    start: State,
    holes: &[Hole],
    lower_bound: &L,
    upper_bound: &U,
    store: &mut spp::SPPstore,
) -> Result<HashMap<Hole, spp::SPP>, CegisError> {
    let mut learner = SmtLearner::new(store.num_vars());

    let mut hole_to_var: HashMap<Hole, SppVar> = HashMap::new();
    for &h in holes {
        hole_to_var.insert(h, learner.fresh_spp());
    }

    let mut cands = learner
        .extract(store)
        .expect("haven't added any constraints yet")
        .spps;
    let mut inst = Instantiate::new(aut, start, holes_map(&hole_to_var, &cands));

    loop {
        match inst.check_less_than(store, upper_bound) {
            Ok(()) => match inst.check_greater_than(store, lower_bound) {
                Ok(()) => return Ok(holes_map(&hole_to_var, &cands)),
                Err(cex) => {
                    println!("New lower bound cex: {cex:?}");
                    add_lower_bound_clauses(cex, &mut inst, &hole_to_var, &mut learner, store);
                }
            },
            Err(witnesses) => {
                let witnesses: Vec<_> = witnesses
                    .into_iter()
                    .map(|(hole, (start, _, end))| (hole, (start, end)))
                    .collect();
                println!("New upper bound cex: {witnesses:?}");
                add_upper_bound_clause(witnesses, &hole_to_var, &mut learner);
            }
        }

        cands = match learner.extract(store) {
            Ok(sol) => sol.spps,
            Err(_) => return Err(CegisError::Infeasible),
        };
        for (h, v) in &hole_to_var {
            inst.set_hole(*h, cands[v]);
        }
    }
}

/// Project a `SppVar → SPP` candidate map back through `hole_to_var` to get
/// a `Hole → SPP` map suitable for [`Instantiate`].
fn holes_map(
    hole_to_var: &HashMap<Hole, SppVar>,
    cands: &HashMap<SppVar, spp::SPP>,
) -> HashMap<Hole, spp::SPP> {
    hole_to_var.iter().map(|(&h, &v)| (h, cands[&v])).collect()
}

/// Convert an upper-bound counterexample (a list of `(hole, (in, out))` pairs
/// recorded along a single violating trace) into a clause for the learner.
///
/// Semantics: each pair `(in_i, out_i)` is a place the trace relied on the
/// hole accepting that pair.  To kill the counterexample, the hole's SPP
/// must *not* accept at least one of them.  That's a disjunction of
/// negative literals — exactly one [`AbstractClause`], each literal targeting
/// the hole's [`SppVar`].
///
/// An empty witness vec means the violation was purely concrete (no hole
/// involvement).  Adding an empty clause makes the learner immediately
/// UNSAT, which is correct: no choice of hole can resolve a concrete
/// violation.
fn add_upper_bound_clause(
    witnesses: Vec<(Hole, (Vec<bool>, Vec<bool>))>,
    hole_to_var: &HashMap<Hole, SppVar>,
    learner: &mut SmtLearner,
) {
    let literals = witnesses
        .into_iter()
        .map(|(h, (p1, p2))| Literal::Spp {
            spp: hole_to_var[&h],
            ap1: p1.into_iter().map(AbstractBit::Concrete).collect(),
            ap2: p2.into_iter().map(AbstractBit::Concrete).collect(),
            polarity: false,
        })
        .collect();
    learner.add_clause(AbstractClause { literals });
}

/// A single hole site discovered while walking the product of the
/// hole-bearing automaton and the trace.
struct HoleSite {
    hole: Hole,
    /// SP of carry packets at the in-side that *are* forward-reachable
    /// under the current candidate (set (1) in the design notes).
    in_sp: sp::SP,
    /// SP of carry packets at the out-side that (a) plausibly let the rest
    /// of the trace continue under the all-top instantiation (set (3)) and
    /// (b) are *not* already forward-reachable under the current candidate
    /// (complement of set (2)).  Adding an entry whose out-packet falls in
    /// `out_sp` genuinely expands the candidate's behaviour towards
    /// satisfying the trace.
    out_sp: sp::SP,
}

/// Convert a lower-bound counterexample into an existential disjunctive
/// clause for the learner.
///
/// For each hole-bearing edge or output summand in the automaton, we compute
/// three SPs (see the design notes / module docstring): (1) packets that
/// actually reach the in-side under the *current* candidate, (2) packets
/// that actually reach the out-side under the same, and (3) packets that
/// plausibly let the rest of the trace continue under the *all-top*
/// instantiation.  The site is viable iff `(1)` and `(3) ∩ ¬(2)` are both
/// non-empty.  We then allocate `2 * num_vars` fresh existentials per
/// viable site, constrain `ap1 ∈ (1)` and `ap2 ∈ (3) ∩ ¬(2)`, and emit a
/// positive [`Literal`] over the hole's [`SppVar`].
///
/// All literals are joined into one [`AbstractClause`]: "at least one of
/// these hole sites must accept a fresh (ap1, ap2) of the appropriate shape".
/// Empty sites list → empty clause → instant UNSAT → `CegisError::Infeasible`
/// on the next learner extraction (correct: no extension at any site can fix
/// this counterexample).
fn add_lower_bound_clauses(
    cex: LowerBoundCounterexample,
    inst: &mut Instantiate<spp::SPP>,
    hole_to_var: &HashMap<Hole, SppVar>,
    learner: &mut SmtLearner,
    store: &mut spp::SPPstore,
) {
    // Forward reach under the *current* candidate.
    let forward_candidate = forward_reachable(&*inst, store, &cex.trace);

    // Backward reach under all-top: temporarily swap, compute, restore.
    let saved: HashMap<Hole, spp::SPP> = inst.holes().clone();
    let top = store.top;
    for &h in hole_to_var.keys() {
        inst.set_hole(h, top);
    }
    let backward_top = backward_reachable(&*inst, store, &cex.trace, &cex.output);
    for (&h, &spp) in &saved {
        inst.set_hole(h, spp);
    }

    // Ignore non-outer states
    let forward_candidate: HashMap<(State, usize), sp::SP> = forward_candidate
        .into_iter()
        .flat_map(|((q, n), sp)| match q {
            inst::State::Outer(q) => Some(((q, n), sp)),
            _ => None,
        })
        .collect();
    let backward_top: HashMap<(State, usize), sp::SP> = backward_top
        .into_iter()
        .flat_map(|((q, n), sp)| match q {
            inst::State::Outer(q) => Some(((q, n), sp)),
            _ => None,
        })
        .collect();

    // Walk the raw AutWithHoles to enumerate hole sites.
    let n = cex.trace.len();
    let start = inst.start_state();
    let sites = {
        let mut aut_ref = inst.aut();
        collect_hole_sites(
            &mut aut_ref,
            start,
            n,
            &forward_candidate,
            &backward_top,
            store,
        )
    };

    // For each site, allocate existentials, constrain to (in_sp, out_sp),
    // emit a positive literal over the hole's SPP.
    let num_vars = store.num_vars() as usize;
    let literals: Vec<Literal> = sites
        .into_iter()
        .map(|site| {
            let in_vars: Vec<Existential> =
                (0..num_vars).map(|_| learner.fresh_existential()).collect();
            let out_vars: Vec<Existential> =
                (0..num_vars).map(|_| learner.fresh_existential()).collect();
            learner.add_sp_membership(site.in_sp, &in_vars, &store.sp);
            learner.add_sp_membership(site.out_sp, &out_vars, &store.sp);
            Literal::Spp {
                spp: hole_to_var[&site.hole],
                ap1: in_vars.into_iter().map(AbstractBit::Exist).collect(),
                ap2: out_vars.into_iter().map(AbstractBit::Exist).collect(),
                polarity: true,
            }
        })
        .collect();
    learner.add_clause(AbstractClause { literals });
}

/// Walk every state forward-reachable from `start` in `aut` and collect a
/// [`HoleSite`] for each hole-bearing edge or output summand that has
/// non-empty `(in_sp, out_sp)` under the supplied reachability maps.
fn collect_hole_sites(
    aut: &mut AutWithHoles,
    start: State,
    n: usize,
    forward: &HashMap<(State, usize), sp::SP>,
    backward: &HashMap<(State, usize), sp::SP>,
    store: &mut spp::SPPstore,
) -> Vec<HoleSite> {
    let mut sites = Vec::new();
    let mut seen: HashSet<State> = HashSet::new();
    let mut stack = vec![start];
    let sp_zero = store.sp.zero;

    while let Some(q) = stack.pop() {
        if !seen.insert(q) {
            continue;
        }

        let trans = aut.transitions(store, q);
        for (label, qp) in &trans {
            if !seen.contains(qp) {
                stack.push(*qp);
            }
            if let EdgeLabel::Abstract(hole) = label {
                let qp_visible = aut.is_visible(*qp);
                for i in 0..n {
                    let i_target = if qp_visible {
                        if i + 1 >= n {
                            continue;
                        }
                        i + 1
                    } else {
                        i
                    };
                    let in_sp = forward.get(&(q, i)).copied().unwrap_or(sp_zero);
                    if store.sp.is_zero(in_sp) {
                        continue;
                    }
                    let already = forward.get(&(*qp, i_target)).copied().unwrap_or(sp_zero);
                    let plausible = backward.get(&(*qp, i_target)).copied().unwrap_or(sp_zero);
                    let not_already = store.sp.complement(already);
                    let out_sp = store.sp.intersect(plausible, not_already);
                    if store.sp.is_zero(out_sp) {
                        continue;
                    }
                    sites.push(HoleSite {
                        hole: *hole,
                        in_sp,
                        out_sp,
                    });
                }
            }
        }
    }

    sites
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
        let result = run(aut, start, &[Hole(0)], &lb, &ub, &mut store).unwrap();
        assert_eq!(result[&Hole(0)], store.zero);
    }

    /// Expression is just a concrete `top`, no holes used.  With ub = zero,
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
        let err = run(aut, start, &[], &lb, &ub, &mut store).unwrap_err();
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
        let result = run(aut, start, &[Hole(0)], &lb, &ub, &mut store).unwrap();
        assert_eq!(result[&Hole(0)], store.zero);
    }

    /// Two holes in `Hole(0) ∪ Hole(1)` with trivial bounds: the empty
    /// candidate for both passes vacuously, returns a map with one entry
    /// per hole.
    #[test]
    fn two_holes_trivial_bounds() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(
            &mut store,
            &Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
        );
        let lb = zero_dfa(&store);
        let ub = top_dfa(&store);
        let result = run(aut, start, &[Hole(0), Hole(1)], &lb, &ub, &mut store).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[&Hole(0)], store.zero);
        assert_eq!(result[&Hole(1)], store.zero);
    }

    /// Two holes with ub = zero: again the only valid assignment is
    /// `zero, zero`.  Converges on the first iteration.
    #[test]
    fn two_holes_bounded_above_by_zero() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(
            &mut store,
            &Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
        );
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let result = run(aut, start, &[Hole(0), Hole(1)], &lb, &ub, &mut store).unwrap();
        assert_eq!(result[&Hole(0)], store.zero);
        assert_eq!(result[&Hole(1)], store.zero);
    }

    /// Build the SPP that accepts only the single pair `(p_in, p_out)`.
    fn singleton_pair_spp(store: &mut spp::SPPstore, p_in: &[bool], p_out: &[bool]) -> spp::SPP {
        let mut spp_acc = spp::SPP::new(1);
        let mut zero = spp::SPP::new(0);
        for (&bi, &bo) in p_in.iter().rev().zip(p_out.iter().rev()) {
            let next = match (bi, bo) {
                (false, false) => store.mk(spp_acc, zero, zero, zero),
                (false, true) => store.mk(zero, spp_acc, zero, zero),
                (true, false) => store.mk(zero, zero, spp_acc, zero),
                (true, true) => store.mk(zero, zero, zero, spp_acc),
            };
            zero = store.mk(zero, zero, zero, zero);
            spp_acc = next;
        }
        spp_acc
    }

    /// Lower-bound failure forces refinement.  Expression `Hole(0)` alone;
    /// upper bound = top; lower bound is the one-state DFA that accepts
    /// exactly the trace `([false,false])` with output `[true,true]`.  The
    /// initial candidate (zero) doesn't accept that, so the lower-bound
    /// check fails — `add_lower_bound_clauses` is exercised end-to-end.
    /// After refinement, the returned SPP must accept the pair.
    #[test]
    fn lower_bound_drives_refinement() {
        let n_vars: spp::Var = 2;
        let mut store = spp::SPPstore::new(n_vars);
        let trace0 = vec![false, false];
        let output_pkt = vec![true, true];
        let singleton = singleton_pair_spp(&mut store, &trace0, &output_pkt);
        let lb = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![singleton],
        };
        let ub = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.top],
        };

        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &Expr::hole(Hole(0)));

        let result = run(aut, start, &[Hole(0)], &lb, &ub, &mut store).unwrap();
        let h0 = result[&Hole(0)];
        assert!(
            store.accepts(h0, &trace0, &output_pkt),
            "refined hole SPP must accept the lower-bound's witness pair"
        );
    }
}
