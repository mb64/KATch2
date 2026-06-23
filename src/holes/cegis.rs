//! CEGIS loop: synthesize a concrete candidate per hole that satisfies a
//! list of [`Constraint`]s simultaneously.
//!
//! Each [`Constraint`] relates a hole-bearing automaton
//! ([`AutWithHoles`]) to a concrete [`ExplicitDFA`], asking that the
//! instantiated automaton be contained in the DFA ([`Constraint::UpperBound`]),
//! contain it ([`Constraint::LowerBound`]), or equal it
//! ([`Constraint::Equality`]).  The synthesized candidates are shared across
//! every constraint, so [`run`] looks for one assignment that satisfies them
//! all.
//!
//! Generic over the candidate kind `C: Candidate` — either [`spp::SPP`] or
//! [`crate::holes::cand::Cand`].  Each candidate-specific step (slot allocation,
//! reading the solution back, the all-top fallback, and the two
//! counterexample → literal conversions) is a [`Candidate`] trait method.
//!
//! # Algorithm
//!
//! 1. Ask the [`SmtLearner`] for a candidate per hole, allocating each slot
//!    against the freestanding `reference_dfa` (this is the DFA whose states
//!    [`crate::holes::cand::Cand`] candidates draw on).
//! 2. Plug the candidates into one [`Instantiate`] per constraint.
//! 3. Visit the constraints in order and ask each whether it is satisfied via
//!    [`Constraint::check`]:
//!    * an upper-bound check (`aut[candidate] ⊆ dfa`) uses
//!      [`Instantiate::check_less_than`]; on failure each hole's witness
//!      becomes a negative literal ([`Candidate::reject_literal`]): "the hole
//!      cannot accept all of these simultaneously".
//!    * a lower-bound check (`dfa ⊆ aut[candidate]`) uses
//!      [`Instantiate::check_greater_than`]; on failure each viable hole site
//!      becomes a positive literal ([`Candidate::accept_literal`]).
//!
//!    The first failing constraint adds its clause and stops the pass.
//! 4. If every constraint is satisfied, return the candidates.
//! 5. Otherwise, ask the learner for refined candidates and loop.  If the
//!    learner returns an inconsistency, the original problem is infeasible.
//!
//! # Limitations
//!
//! * Lower-bound (`accept_literal`) is implemented for [`spp::SPP`] only;
//!   [`crate::holes::cand::Cand`] panics if a lower-bound counterexample arises.

use std::collections::{HashMap, HashSet};

use crate::holes::aut::{ENFA, ExplicitDFA, backward_reachable, forward_reachable};
use crate::holes::candidate::Candidate;
use crate::holes::inst::{self, Instantiate, LowerBoundCounterexample};
use crate::holes::nk_with_holes::{AutWithHoles, EdgeLabel, Expr, Hole, State};
use crate::holes::smt::{AbstractClause, SmtLearner};
use crate::sp;
use crate::spp;

/// Why the CEGIS loop gave up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CegisError {
    /// The accumulated clauses became unsatisfiable: no SPP value for the
    /// hole can simultaneously satisfy the lower and upper bounds.
    Infeasible,
    /// The loop hit the caller's iteration cap (`max_iters`) before either
    /// converging or proving infeasibility.  Unlike [`CegisError::Infeasible`]
    /// this is *inconclusive*: a solution may still exist (the loop may just be
    /// converging slowly, or diverging).  Raising the cap may resolve it.
    IterationLimit,
}

/// A single constraint relating a hole-bearing automaton to a concrete DFA.
///
/// Each variant pairs an [`AutWithHoles`] (with its pinned start [`State`])
/// against an [`ExplicitDFA`].  Once the holes are filled with concrete
/// candidates, the constraint asserts a containment between the resulting
/// instantiated automaton and the DFA; see the variant docs for the
/// direction.
///
/// Build one with [`Constraint::upper_bound`], [`Constraint::lower_bound`], or
/// [`Constraint::equality`], which compile a [`nk_with_holes::Expr`] into the
/// underlying automaton for you.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// `dfa ⊆ aut[holes]`: the DFA is a lower bound on the instantiated
    /// automaton (every triple the DFA accepts must also be accepted).
    LowerBound {
        aut: AutWithHoles,
        start: State,
        dfa: ExplicitDFA,
    },
    /// `aut[holes] ⊆ dfa`: the DFA is an upper bound on the instantiated
    /// automaton (every triple the automaton accepts must also be accepted by
    /// the DFA).
    UpperBound {
        aut: AutWithHoles,
        start: State,
        dfa: ExplicitDFA,
    },
    /// `aut[holes] == dfa`: the instantiated automaton must accept exactly the
    /// DFA's language — both an upper and a lower bound at once.
    Equality {
        aut: AutWithHoles,
        start: State,
        dfa: ExplicitDFA,
    },
}

impl Constraint {
    /// Upper-bound constraint `expr[holes] ⊆ dfa`, compiling `expr` into the
    /// hole-bearing automaton.
    pub fn upper_bound(store: &mut spp::SPPstore, expr: &Expr, dfa: ExplicitDFA) -> Self {
        let (aut, start) = compile(store, expr);
        Constraint::UpperBound { aut, start, dfa }
    }

    /// Lower-bound constraint `dfa ⊆ expr[holes]`, compiling `expr` into the
    /// hole-bearing automaton.
    pub fn lower_bound(store: &mut spp::SPPstore, expr: &Expr, dfa: ExplicitDFA) -> Self {
        let (aut, start) = compile(store, expr);
        Constraint::LowerBound { aut, start, dfa }
    }

    /// Equality constraint `expr[holes] == dfa`, compiling `expr` into the
    /// hole-bearing automaton.
    pub fn equality(store: &mut spp::SPPstore, expr: &Expr, dfa: ExplicitDFA) -> Self {
        let (aut, start) = compile(store, expr);
        Constraint::Equality { aut, start, dfa }
    }

    /// The hole-bearing automaton this constraint is over.
    pub fn aut(&self) -> &AutWithHoles {
        match self {
            Constraint::LowerBound { aut, .. }
            | Constraint::UpperBound { aut, .. }
            | Constraint::Equality { aut, .. } => aut,
        }
    }

    /// The pinned start state of [`Constraint::aut`].
    pub fn start(&self) -> State {
        match self {
            Constraint::LowerBound { start, .. }
            | Constraint::UpperBound { start, .. }
            | Constraint::Equality { start, .. } => *start,
        }
    }

    /// The concrete DFA this constraint compares against.
    pub fn dfa(&self) -> &ExplicitDFA {
        match self {
            Constraint::LowerBound { dfa, .. }
            | Constraint::UpperBound { dfa, .. }
            | Constraint::Equality { dfa, .. } => dfa,
        }
    }

    /// Check whether `inst` (the constraint's automaton instantiated with the
    /// current candidates) satisfies this constraint.
    ///
    /// Returns `true` if satisfied.  Otherwise records a refining clause on
    /// `learner` — a [`Candidate::reject_literal`] disjunction for an
    /// upper-bound violation, or a [`Candidate::accept_literal`] disjunction
    /// for a lower-bound one — and returns `false`.  At most one clause is
    /// added per call: an [`Constraint::Equality`] that fails its upper-bound
    /// check stops before checking the lower bound.
    fn check<'a, C: Candidate<'a>>(
        &self,
        inst: &mut Instantiate<C>,
        hole_to_var: &HashMap<Hole, C::Var>,
        learner: &mut SmtLearner<'a>,
        store: &mut spp::SPPstore,
        reference_dfa: &'a ExplicitDFA,
    ) -> bool {
        let dfa = self.dfa();

        // Upper-bound half: `aut[candidate] ⊆ dfa`.
        if matches!(
            self,
            Constraint::UpperBound { .. } | Constraint::Equality { .. }
        ) && let Err(witnesses) = inst.check_less_than(store, dfa)
        {
            add_upper_bound_clause::<C>(witnesses, hole_to_var, learner);
            return false;
        }

        // Lower-bound half: `dfa ⊆ aut[candidate]`.
        if matches!(
            self,
            Constraint::LowerBound { .. } | Constraint::Equality { .. }
        ) && let Err(cex) = inst.check_greater_than(store, dfa)
        {
            add_lower_bound_clauses(cex, inst, hole_to_var, learner, store, reference_dfa);
            return false;
        }

        true
    }
}

/// Compile a hole-bearing [`Expr`] into its automaton and visible start state.
fn compile(store: &mut spp::SPPstore, expr: &Expr) -> (AutWithHoles, State) {
    let mut aut = AutWithHoles::new();
    let start = aut.expr_to_state(store, expr);
    (aut, start)
}

/// Synthesize a candidate per hole satisfying every constraint in
/// `constraints` simultaneously.
///
/// `holes` is the set of hole labels that may appear in the constraints'
/// automata (every hole the automata reach must be listed, otherwise
/// [`Instantiate`] will panic when it hits an unmapped one).  On success,
/// returns one synthesized candidate per hole.
///
/// `C` is the candidate kind ([`spp::SPP`] or [`crate::holes::cand::Cand`]).
/// `reference_dfa` is a freestanding DFA used only to allocate and ground the
/// candidates — it is the DFA that [`crate::holes::cand::Cand`] candidates pull
/// their states from, and the all-top fallback used in lower-bound processing.
/// It is independent of any constraint's own DFA.
///
/// `max_iters` caps the number of refinement rounds: pass `Some(n)` to return
/// [`CegisError::IterationLimit`] after `n` rounds without convergence, or
/// `None` to loop unboundedly (the historical behaviour).
pub fn run<'a, C: Candidate<'a>>(
    constraints: &[Constraint],
    holes: &[Hole],
    reference_dfa: &'a ExplicitDFA,
    store: &mut spp::SPPstore,
    max_iters: Option<usize>,
) -> Result<HashMap<Hole, C>, CegisError> {
    let mut learner = SmtLearner::new(store.num_vars());

    let mut hole_to_var: HashMap<Hole, C::Var> = HashMap::new();
    for &h in holes {
        hole_to_var.insert(h, C::fresh_var(&mut learner, reference_dfa));
    }

    let sol = learner
        .extract(store)
        .expect("haven't added any constraints yet");
    let mut candidates: HashMap<Hole, C> = hole_to_var
        .iter()
        .map(|(&h, &v)| (h, C::from_solution(&sol, v)))
        .collect();

    // One instantiation per constraint, all sharing the same hole assignment.
    let mut insts: Vec<Instantiate<C>> = constraints
        .iter()
        .map(|c| Instantiate::new(c.aut().clone(), c.start(), candidates.clone()))
        .collect();

    let mut iters: usize = 0;
    loop {
        if let Some(cap) = max_iters
            && iters >= cap
        {
            return Err(CegisError::IterationLimit);
        }
        iters += 1;

        // Visit constraints in order; the first failure adds its clause and
        // ends the pass so we re-extract before re-checking.
        let mut satisfied = true;
        for (c, inst) in constraints.iter().zip(insts.iter_mut()) {
            if !c.check(inst, &hole_to_var, &mut learner, store, reference_dfa) {
                satisfied = false;
                break;
            }
        }
        if satisfied {
            return Ok(candidates);
        }

        let sol = match learner.extract(store) {
            Ok(sol) => sol,
            Err(_) => return Err(CegisError::Infeasible),
        };
        candidates = hole_to_var
            .iter()
            .map(|(&h, &v)| (h, C::from_solution(&sol, v)))
            .collect();
        for inst in insts.iter_mut() {
            for (&h, &v) in &hole_to_var {
                inst.set_hole(h, C::from_solution(&sol, v));
            }
        }
    }
}

/// Convert an upper-bound counterexample (per-hole `(in, inner_trace, out)`
/// witnesses recorded along a single violating trace) into a clause for the
/// learner.
///
/// Semantics: each witness is a place the trace relied on the hole accepting
/// that traversal.  To kill the counterexample, at least one hole must *not*
/// accept its witness — a disjunction of negative literals, exactly one
/// [`AbstractClause`].  Each [`Candidate::reject_literal`] decides how to encode
/// its own witness.
///
/// An empty witness vec means the violation was purely concrete (no hole
/// involvement).  Adding an empty clause makes the learner immediately
/// UNSAT, which is correct: no choice of hole can resolve a concrete
/// violation.
fn add_upper_bound_clause<'a, C: Candidate<'a>>(
    witnesses: Vec<(
        Hole,
        (Vec<bool>, Vec<(<C as ENFA>::State, Vec<bool>)>, Vec<bool>),
    )>,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut SmtLearner<'a>,
) {
    let literals = witnesses
        .into_iter()
        .map(|(h, (start, inner, end))| C::reject_literal(hole_to_var[&h], &start, &inner, &end))
        .collect();
    learner.add_clause(AbstractClause { literals });
}

/// A single hole site discovered while walking the product of the
/// hole-bearing automaton and the trace.
struct HoleSite<'a> {
    hole: Hole,

    /// SP of carry-on packets at the in-side that *are* forward-reachable
    /// under the current candidate (set (1) in the design notes).
    in_sp: sp::SP,

    /// Trace accumulated by the hole
    trace: &'a [Vec<bool>],

    /// SP of carry-on packets at the out-side that (a) plausibly let the rest
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
fn add_lower_bound_clauses<'a, C: Candidate<'a>>(
    cex: LowerBoundCounterexample,
    inst: &mut Instantiate<C>,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut SmtLearner<'a>,
    store: &mut spp::SPPstore,
    reference_dfa: &'a ExplicitDFA,
) where
    <C as ENFA>::State: Ord,
{
    // Forward reach under the *current* candidate.
    let forward_candidate = forward_reachable(&*inst, store, &cex.trace);

    // Backward reach under all-top: temporarily swap, compute, restore.
    let saved: HashMap<Hole, C> = inst.holes().clone();
    let top = C::top(store, reference_dfa);
    for &h in hole_to_var.keys() {
        inst.set_hole(h, top.clone());
    }
    let backward_top = backward_reachable(&*inst, store, &cex.trace, &cex.output);
    for (h, candidate) in saved {
        inst.set_hole(h, candidate);
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
    let start = inst.start_state();
    let sites = {
        let mut aut_ref = inst.aut();
        collect_hole_sites(
            &mut aut_ref,
            start,
            &cex.trace,
            &forward_candidate,
            &backward_top,
            store,
        )
    };

    // For each site, ask the candidate to emit a positive literal constraining
    // it to accept some `(ap1 ∈ in_sp, ap2 ∈ out_sp)`.
    let literals = sites
        .into_iter()
        .flat_map(|site| {
            C::accept_literal(
                hole_to_var[&site.hole],
                learner,
                store,
                reference_dfa,
                site.in_sp,
                site.out_sp,
                site.trace,
            )
        })
        .collect();
    learner.add_clause(AbstractClause { literals });
}

/// Walk every state forward-reachable from `start` in `aut` and collect a
/// [`HoleSite`] for each hole-bearing edge or output summand that has
/// non-empty `(in_sp, out_sp)` under the supplied reachability maps.
fn collect_hole_sites<'a>(
    aut: &mut AutWithHoles,
    start: State,
    trace: &'a [Vec<bool>],
    forward: &HashMap<(State, usize), sp::SP>,
    backward: &HashMap<(State, usize), sp::SP>,
    store: &mut spp::SPPstore,
) -> Vec<HoleSite<'a>> {
    let n = trace.len();
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
                    for j in i..n {
                        let j_target = if qp_visible {
                            if j + 1 >= n {
                                continue;
                            }
                            j + 1
                        } else {
                            j
                        };
                        let in_sp = forward.get(&(q, i)).copied().unwrap_or(sp_zero);
                        if store.sp.is_zero(in_sp) {
                            continue;
                        }
                        let already = forward.get(&(*qp, j_target)).copied().unwrap_or(sp_zero);
                        let plausible = backward.get(&(*qp, j_target)).copied().unwrap_or(sp_zero);
                        let not_already = store.sp.complement(already);
                        let out_sp = store.sp.intersect(plausible, not_already);
                        if store.sp.is_zero(out_sp) {
                            continue;
                        }
                        sites.push(HoleSite {
                            hole: *hole,
                            in_sp,
                            // The hole's internal sub-trace is its *dup'd*
                            // packets, at positions `i+1 ..= j_target`.
                            // The carry-in packet is at position `i.
                            trace: &trace[i + 1..j_target + 1],
                            out_sp,
                        });
                    }
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
    use crate::holes::cand::{Cand, Input};
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

    /// Solve `lb ⊆ expr[holes] ⊆ ub` via a pair of [`Constraint`]s (an upper
    /// and a lower bound over `expr`), grounding the candidates on `ub`.
    fn run_bounds<'a, C: Candidate<'a>>(
        store: &mut spp::SPPstore,
        expr: &Expr,
        holes: &[Hole],
        lb: &ExplicitDFA,
        ub: &'a ExplicitDFA,
        max_iters: Option<usize>,
    ) -> Result<HashMap<Hole, C>, CegisError> {
        let constraints = vec![
            Constraint::upper_bound(store, expr, ub.clone()),
            Constraint::lower_bound(store, expr, lb.clone()),
        ];
        run::<C>(&constraints, holes, ub, store, max_iters)
    }

    /// 0 ⊆ Hole ⊆ top: trivially solvable.  The learner returns the empty
    /// SPP (its default when unconstrained), the upper-bound check passes
    /// vacuously, the lower-bound check passes vacuously, done.
    #[test]
    fn trivial_bounds_converge_immediately() {
        let mut store = mk_store();
        let lb = zero_dfa(&store);
        let ub = top_dfa(&store);
        let result =
            run_bounds::<spp::SPP>(&mut store, &Expr::hole(Hole(0)), &[Hole(0)], &lb, &ub, None)
                .unwrap();
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
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let err =
            run_bounds::<spp::SPP>(&mut store, &Expr::spp(top), &[], &lb, &ub, None).unwrap_err();
        assert_eq!(err, CegisError::Infeasible);
    }

    /// `Hole(0)` with ub = zero: the only valid value is `store.zero`.  The
    /// learner's empty extraction is already zero, so we converge on the
    /// first iteration without ever needing to refine.
    #[test]
    fn hole_bounded_above_by_zero_yields_zero() {
        let mut store = mk_store();
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let result =
            run_bounds::<spp::SPP>(&mut store, &Expr::hole(Hole(0)), &[Hole(0)], &lb, &ub, None)
                .unwrap();
        assert_eq!(result[&Hole(0)], store.zero);
    }

    /// Two holes in `Hole(0) ∪ Hole(1)` with trivial bounds: the empty
    /// candidate for both passes vacuously, returns a map with one entry
    /// per hole.
    #[test]
    fn two_holes_trivial_bounds() {
        let mut store = mk_store();
        let expr = Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1)));
        let lb = zero_dfa(&store);
        let ub = top_dfa(&store);
        let result =
            run_bounds::<spp::SPP>(&mut store, &expr, &[Hole(0), Hole(1)], &lb, &ub, None).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[&Hole(0)], store.zero);
        assert_eq!(result[&Hole(1)], store.zero);
    }

    /// Two holes with ub = zero: again the only valid assignment is
    /// `zero, zero`.  Converges on the first iteration.
    #[test]
    fn two_holes_bounded_above_by_zero() {
        let mut store = mk_store();
        let expr = Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1)));
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let result =
            run_bounds::<spp::SPP>(&mut store, &expr, &[Hole(0), Hole(1)], &lb, &ub, None).unwrap();
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

        let result =
            run_bounds::<spp::SPP>(&mut store, &Expr::hole(Hole(0)), &[Hole(0)], &lb, &ub, None)
                .unwrap();
        let h0 = result[&Hole(0)];
        assert!(
            store.accepts(h0, &trace0, &output_pkt),
            "refined hole SPP must accept the lower-bound's witness pair"
        );
    }

    /// `0 ⊆ Hole ⊆ top` with `Cand` candidates: converges on the first
    /// iteration (both checks vacuous), returning the unconstrained Cand, which
    /// accepts everything.  Exercises the whole `run::<Cand, _>` pipeline —
    /// fresh_cand, from_solution, and the `Instantiate<Cand>` embedding.
    #[test]
    fn cand_trivial_bounds_converge() {
        let mut store = mk_store();
        let lb = zero_dfa(&store);
        let ub = top_dfa(&store); // one state ⇒ Cand has num_states == 1
        let result =
            run_bounds::<Cand>(&mut store, &Expr::hole(Hole(0)), &[Hole(0)], &lb, &ub, None)
                .unwrap();
        let cand = &result[&Hole(0)];
        assert!(cand.accepts_input(
            &mut store,
            &Input {
                pkt_in: vec![false, false, false],
                pkt_start: vec![true, false, true],
                states: vec![0],
                pkt_end: vec![false, true, false],
                pkt_out: vec![true, true, false],
            },
        ));
    }

    /// `0 ⊆ Hole ⊆ 0` with `Cand` candidates: the upper bound rejects
    /// everything, so the accept-all default must be refined (via reject
    /// literals) down to a Cand that makes the expression empty.  Exercises the
    /// upper-bound refinement loop and `Cand::reject_literal`.
    #[test]
    fn cand_upper_bound_refinement() {
        let mut store = spp::SPPstore::new(1);
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let result =
            run_bounds::<Cand>(&mut store, &Expr::hole(Hole(0)), &[Hole(0)], &lb, &ub, None)
                .unwrap();
        let cand = &result[&Hole(0)];
        assert!(!cand.accepts_input(
            &mut store,
            &Input {
                pkt_in: vec![false],
                pkt_start: vec![false],
                states: vec![0],
                pkt_end: vec![false],
                pkt_out: vec![false],
            },
        ));
    }
}
