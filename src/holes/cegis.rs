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

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::Direction;
use petgraph::algo::dinics;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::flags;
use crate::holes::aut::{ENFA, ExplicitDFA, backward_reachable, forward_reachable};
use crate::holes::candidate::Candidate;
use crate::holes::inst::{self, Instantiate, LowerBoundCounterexample};
use crate::holes::nk_with_holes::{AutWithHoles, EdgeLabel, Hole, State};
use crate::holes::problem::Constraint;
use crate::holes::smt::{AbstractClause, Literal, SmtLearner};
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

impl Constraint {
    /// Check whether `inst` (the constraint's automaton instantiated with the
    /// current candidates) satisfies this constraint.
    ///
    /// Returns `true` if satisfied.  Otherwise records a refining clause on
    /// `learner` — a [`Candidate::reject_literal`] disjunction for an
    /// upper-bound violation, or a [`Candidate::accept_literal`] disjunction
    /// for a lower-bound one — and returns `false`.  At most one clause is
    /// added per call: an [`Constraint::Equality`] that fails its upper-bound
    /// check stops before checking the lower bound.
    fn check<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
        &self,
        inst: &mut Instantiate<C>,
        hole_to_var: &HashMap<Hole, C::Var>,
        learner: &mut L,
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
            // println!("Upper bound cex: {witnesses:?}");
            add_upper_bound_clause::<C, L>(witnesses, hole_to_var, learner, &mut store.sp);
            return false;
        }

        // Lower-bound half: `dfa ⊆ aut[candidate]`.
        if matches!(
            self,
            Constraint::LowerBound { .. } | Constraint::Equality { .. }
        ) && let Err(cex) = inst.check_greater_than(store, dfa)
        {
            // println!("Lower bound cex: {cex:?}");
            add_lower_bound_clause(cex, inst, hole_to_var, learner, store, reference_dfa);
            return false;
        }

        true
    }
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
pub fn run<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    constraints: &[Constraint],
    holes: &[Hole],
    reference_dfa: &'a ExplicitDFA,
    store: &mut spp::SPPstore,
    max_iters: Option<usize>,
) -> Result<HashMap<Hole, C>, CegisError> {
    let mut learner = L::new(store.num_vars());

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
            // println!("CEGIS: finished in {iters} iterations");
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
fn add_upper_bound_clause<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    witnesses: Vec<(
        Hole,
        (Vec<bool>, Vec<(<C as ENFA>::State, Vec<bool>)>, Vec<bool>),
    )>,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut L,
    sp_store: &mut sp::SPstore,
) {
    let literals = witnesses
        .into_iter()
        .map(|(h, (start, inner, end))| C::reject_literal(hole_to_var[&h], &start, &inner, &end))
        .collect();
    learner.add_clause(AbstractClause { literals }, sp_store);
}

/// One of the three packet regions an outer `(state, position)` node is split
/// into in the abstract cut graph.
///
/// The regions partition packet space at the node, all computed as *actual*
/// reachability under the current candidate:
///
/// * `Fwd` — packets that forward-reach the node from the start.
/// * `Bwd` — packets from which the node co-reaches the final output.
/// * `Int` — everything else (`¬(Fwd ∪ Bwd)`).
///
/// `Fwd` and `Bwd` are disjoint: a packet in both would witness an accepting
/// run the instantiation already realizes, contradicting the counterexample.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Region {
    Fwd,
    Int,
    Bwd,
}

/// Drop every non-`Outer` `(state, position)` key, projecting an
/// [`Instantiate`] reachability map onto the underlying [`AutWithHoles`] states.
fn outer_only<S>(map: HashMap<(inst::State<S>, usize), sp::SP>) -> HashMap<(State, usize), sp::SP> {
    map.into_iter()
        .filter_map(|((q, n), sp)| match q {
            inst::State::Outer(q) => Some(((q, n), sp)),
            _ => None,
        })
        .collect()
}

/// The SP of packets in `region` at outer node `(q, i)`.
fn region_sp(
    store: &mut spp::SPPstore,
    forward: &HashMap<(State, usize), sp::SP>,
    backward: &HashMap<(State, usize), sp::SP>,
    q: State,
    i: usize,
    region: Region,
) -> sp::SP {
    let zero = store.sp.zero;
    let fwd = forward.get(&(q, i)).copied().unwrap_or(zero);
    let bwd = backward.get(&(q, i)).copied().unwrap_or(zero);
    match region {
        Region::Fwd => fwd,
        Region::Bwd => bwd,
        Region::Int => {
            let live = store.sp.union(fwd, bwd);
            store.sp.complement(live)
        }
    }
}

/// Resolve a `(region, q, i)` to its graph node, allocating an `Int` node on
/// demand.  Every `Fwd` region collapses into the single `source`, every `Bwd`
/// into the single `sink` (the infinite-capacity connectors of the original
/// formulation).
fn node_for(
    region: Region,
    q: State,
    i: usize,
    source: NodeIndex,
    sink: NodeIndex,
    graph: &mut DiGraph<(), u32>,
    int_nodes: &mut HashMap<(State, usize), NodeIndex>,
) -> NodeIndex {
    match region {
        Region::Fwd => source,
        Region::Bwd => sink,
        Region::Int => *int_nodes
            .entry((q, i))
            .or_insert_with(|| graph.add_node(())),
    }
}

/// Edges crossing the minimum `source`→`sink` cut.
///
/// Runs Dinic's algorithm for the max flow, then recovers the cut by finding
/// the residual-reachable set `R` from `source` (forward edges with spare
/// capacity, backward edges carrying flow) and returning every original edge
/// from `R` into its complement.  Those edges are exactly saturated, and — by
/// max-flow/min-cut — their total capacity is the max flow.
fn min_cut_edges(graph: &DiGraph<(), u32>, source: NodeIndex, sink: NodeIndex) -> Vec<EdgeIndex> {
    let (_max_flow, flows) = dinics(graph, source, sink);

    let mut reachable = vec![false; graph.node_count()];
    reachable[source.index()] = true;
    let mut queue = VecDeque::from([source]);
    while let Some(u) = queue.pop_front() {
        // Forward edges with residual capacity.
        for e in graph.edges_directed(u, Direction::Outgoing) {
            if *e.weight() - flows[e.id().index()] > 0 && !reachable[e.target().index()] {
                reachable[e.target().index()] = true;
                queue.push_back(e.target());
            }
        }
        // Backward edges that carry flow we could cancel.
        for e in graph.edges_directed(u, Direction::Incoming) {
            if flows[e.id().index()] > 0 && !reachable[e.source().index()] {
                reachable[e.source().index()] = true;
                queue.push_back(e.source());
            }
        }
    }

    graph
        .edge_indices()
        .filter(|&e| {
            let (u, v) = graph.edge_endpoints(e).unwrap();
            reachable[u.index()] && !reachable[v.index()]
        })
        .collect()
}

/// Convert a lower-bound counterexample into an existential disjunctive clause
/// for the learner, dispatching on [`flags::lb_mincut`]: the min-cut clause
/// ([`add_lower_bound_clause_mincut`]) when on, the frontier clause
/// ([`add_lower_bound_clause_frontier`]) when off.
fn add_lower_bound_clause<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    cex: LowerBoundCounterexample,
    inst: &mut Instantiate<C>,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut L,
    store: &mut spp::SPPstore,
    reference_dfa: &'a ExplicitDFA,
) where
    <C as ENFA>::State: Ord,
{
    if flags::lb_mincut() {
        add_lower_bound_clause_mincut(cex, inst, hole_to_var, learner, store, reference_dfa)
    } else {
        add_lower_bound_clause_frontier(cex, inst, hole_to_var, learner, store, reference_dfa)
    }
}

/// Convert a lower-bound counterexample into an existential disjunctive clause
/// for the learner, using a **min-cut** over an abstract reachability graph to
/// keep the clause small.
///
/// The really-big graph has vertices `(i, q, pkt)`; an edge exists when a
/// (potential, all-top) transition relates the packets.  We collapse the packet
/// dimension into three regions per outer node `(q, i)` — [`Region::Fwd`],
/// [`Region::Bwd`], [`Region::Int`] — computed from *symmetric actual*
/// reachability under the current candidate:
///
/// * `forward` = [`forward_reachable`] and `backward` = [`backward_reachable`],
///   both run on the **current** instantiation (no all-top plausibility).
///
/// Edges come from the automaton's transitions taken under the all-top
/// instantiation:
///
/// * **concrete** edges never cross regions (forward/backward closure), so only
///   their `Int → Int` part is added, with infinite capacity (uncuttable
///   backbone);
/// * **hole** edges are the cuttable, capacity-1 edges.  For each span we emit
///   one edge per relevant `(source region, target region)` pair, each carrying
///   the positive [`Literal`] that [`Candidate::accept_literal`] would add (a
///   `None` literal means this candidate kind cannot realize the span, so no
///   edge is created).
///
/// `Fwd` collapses into a single source, `Bwd` into a single sink.  The min cut
/// is a set of hole edges separating "reachable now" from "co-reaches the
/// output"; the disjunction of their literals is the clause.  An empty cut →
/// empty clause → instant UNSAT (no extension can fix the counterexample).
fn add_lower_bound_clause_mincut<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    cex: LowerBoundCounterexample,
    inst: &mut Instantiate<C>,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut L,
    store: &mut spp::SPPstore,
    reference_dfa: &'a ExplicitDFA,
) where
    <C as ENFA>::State: Ord,
{
    let n = cex.trace.len();

    // Enumerate every structurally-reachable outer state and its transitions in
    // a single walk of the raw `AutWithHoles` (visibility is read off each
    // target's `visible` flag, so we never need the automaton again).
    let mut edges: Vec<(State, EdgeLabel, State)> = Vec::new();
    let outer_states: Vec<State> = {
        let start = inst.start_state();
        let mut aut = inst.aut();
        let mut seen: HashSet<State> = HashSet::new();
        let mut stack = vec![start];
        while let Some(q) = stack.pop() {
            if !seen.insert(q) {
                continue;
            }
            for (label, qp) in aut.transitions(store, q) {
                if !seen.contains(&qp) {
                    stack.push(qp);
                }
                edges.push((q, label, qp));
            }
        }
        seen.into_iter().collect()
    };

    // Symmetric *actual* reachability under the current candidate.  `backward`
    // is seeded with every outer state: the backward region lives off the
    // forward-reachable path (behind the holes we still have to fill), so a
    // start-only enumeration would miss it.
    let roots: Vec<inst::State<<C as ENFA>::State>> =
        outer_states.into_iter().map(inst::State::Outer).collect();
    let forward = outer_only(forward_reachable(&*inst, store, &cex.trace));
    let backward = outer_only(backward_reachable(
        &*inst,
        store,
        &cex.trace,
        &cex.output,
        &roots,
    ));

    // Backbone (concrete / connector) edges are uncuttable: larger than any
    // possible cut, which is bounded by the number of capacity-1 hole edges.
    const INF: u32 = 1 << 30;

    let mut graph: DiGraph<(), u32> = DiGraph::new();
    let source = graph.add_node(());
    let sink = graph.add_node(());
    let mut int_nodes: HashMap<(State, usize), NodeIndex> = HashMap::new();
    let mut edge_literals: HashMap<EdgeIndex, Literal> = HashMap::new();

    for (q, label, qp) in &edges {
        let (q, qp) = (*q, *qp);
        let qp_visible = qp.visible;
        match label {
            EdgeLabel::Concrete(spp) => {
                // Single step; only the `Int → Int` part can matter.
                for i in 0..n {
                    let it = if qp_visible {
                        if i + 1 >= n {
                            continue;
                        }
                        i + 1
                    } else {
                        i
                    };
                    let src = region_sp(store, &forward, &backward, q, i, Region::Int);
                    let tgt = region_sp(store, &forward, &backward, qp, it, Region::Int);
                    if store.sp.is_zero(src) || store.sp.is_zero(tgt) {
                        continue;
                    }
                    let pushed = store.push(src, *spp);
                    let img = store.sp.intersect(pushed, tgt);
                    if store.sp.is_zero(img) {
                        continue;
                    }
                    let u = node_for(Region::Int, q, i, source, sink, &mut graph, &mut int_nodes);
                    let v = node_for(
                        Region::Int,
                        qp,
                        it,
                        source,
                        sink,
                        &mut graph,
                        &mut int_nodes,
                    );
                    if u != v {
                        graph.add_edge(u, v, INF);
                    }
                }
            }
            EdgeLabel::Abstract(hole) => {
                let var = hole_to_var[hole];
                // A hole may span a sub-trace (`i ..= jt`); under all-top it
                // relates every packet, so each region's full SP carries over.
                for i in 0..n {
                    for j in i..n {
                        let jt = if qp_visible {
                            if j + 1 >= n {
                                continue;
                            }
                            j + 1
                        } else {
                            j
                        };
                        let trace_slice = &cex.trace[i + 1..jt + 1];
                        // Edges out of `Bwd`/into `Fwd` only re-enter source
                        // or leave sink, so they are useless for the cut.
                        for (src_region, tgt_region) in [
                            (Region::Fwd, Region::Int),
                            (Region::Fwd, Region::Bwd),
                            (Region::Int, Region::Int),
                            (Region::Int, Region::Bwd),
                        ] {
                            let in_sp = region_sp(store, &forward, &backward, q, i, src_region);
                            let out_sp = region_sp(store, &forward, &backward, qp, jt, tgt_region);
                            if store.sp.is_zero(in_sp) || store.sp.is_zero(out_sp) {
                                continue;
                            }
                            // Only realizable spans become edges, so the cut
                            // always maps back to a genuine literal.
                            let Some(lit) = C::accept_literal(
                                var,
                                learner,
                                store,
                                reference_dfa,
                                in_sp,
                                out_sp,
                                trace_slice,
                            ) else {
                                continue;
                            };
                            let u = node_for(
                                src_region,
                                q,
                                i,
                                source,
                                sink,
                                &mut graph,
                                &mut int_nodes,
                            );
                            let v = node_for(
                                tgt_region,
                                qp,
                                jt,
                                source,
                                sink,
                                &mut graph,
                                &mut int_nodes,
                            );
                            if u == v {
                                continue;
                            }
                            let e = graph.add_edge(u, v, 1);
                            edge_literals.insert(e, lit);
                        }
                    }
                }
            }
        }
    }

    let literals = min_cut_edges(&graph, source, sink)
        .into_iter()
        .filter_map(|e| edge_literals.remove(&e))
        .collect();
    learner.add_clause(AbstractClause { literals }, &mut store.sp);
}

/// Convert a lower-bound counterexample into an existential disjunctive
/// clause for the learner — the **frontier** clause used when
/// [`flags::lb_mincut`] is off.
///
/// For each hole-bearing edge or output summand in the automaton, we compute
/// three SPs: (1) packets that actually reach the in-side under the *current*
/// candidate, (2) packets that actually reach the out-side under the same, and
/// (3) packets that plausibly let the rest of the trace continue under the
/// *all-top* instantiation.  The site is viable iff `(1)` and `(3) ∩ ¬(2)` are
/// both non-empty; every viable site becomes a positive literal, all joined
/// into one clause.  Empty sites → empty clause → instant UNSAT.
fn add_lower_bound_clause_frontier<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    cex: LowerBoundCounterexample,
    inst: &mut Instantiate<C>,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut L,
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
    let backward_top = backward_reachable(&*inst, store, &cex.trace, &cex.output, &[]);
    for (h, candidate) in saved {
        inst.set_hole(h, candidate);
    }

    // Ignore non-outer states.
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
    learner.add_clause(AbstractClause { literals }, &mut store.sp);
}

/// A single hole site discovered while walking the hole-bearing automaton along
/// the counterexample trace (frontier clause; see [`add_lower_bound_clause`]).
struct HoleSite<'a> {
    hole: Hole,
    /// SP of carry-in packets that *are* forward-reachable under the current
    /// candidate.
    in_sp: sp::SP,
    /// Sub-trace the hole consumes internally.
    trace: &'a [Vec<bool>],
    /// SP of carry-out packets that plausibly continue the trace under all-top
    /// but are not already forward-reachable.
    out_sp: sp::SP,
}

/// Walk every state reachable from `start` and collect a [`HoleSite`] for each
/// hole edge whose `(in_sp, out_sp)` are both non-empty.
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
                            // The hole's internal sub-trace is its dup'd packets,
                            // at positions `i+1 ..= j_target`; the carry-in packet
                            // is at position `i`.
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
    use crate::holes::smt::Z3;

    /// Iteration cap for these unit tests.  Every case here converges (or
    /// proves infeasible) in a couple of rounds, so this is purely a guard
    /// against an unexpected divergence hanging the suite.
    const MAX_ITERS: usize = 64;

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

    /// The upper/lower bound constraints for `lb ⊆ expr[holes] ⊆ ub`.
    fn bounds_constraints(
        store: &mut spp::SPPstore,
        expr: &Expr,
        lb: &ExplicitDFA,
        ub: &ExplicitDFA,
    ) -> Vec<Constraint> {
        vec![
            Constraint::upper_bound(store, expr, ub.clone()),
            Constraint::lower_bound(store, expr, lb.clone()),
        ]
    }

    /// The caller-owned reference DFA grounding candidates of kind `C` for
    /// `lb ⊆ expr[holes] ⊆ ub`.  Must outlive the candidates `run_bounds`
    /// returns (a [`Cand`] borrows it).
    fn bounds_reference_dfa<'a, C: Candidate<'a>>(
        store: &mut spp::SPPstore,
        expr: &Expr,
        lb: &ExplicitDFA,
        ub: &ExplicitDFA,
    ) -> ExplicitDFA {
        let constraints = bounds_constraints(store, expr, lb, ub);
        C::make_reference_dfa(store, &constraints)
    }

    /// Solve `lb ⊆ expr[holes] ⊆ ub` via a pair of [`Constraint`]s (an upper
    /// and a lower bound over `expr`), grounding the candidates on the
    /// caller-owned `reference_dfa` (see [`bounds_reference_dfa`]).
    fn run_bounds<'a, C: Candidate<'a>>(
        store: &mut spp::SPPstore,
        expr: &Expr,
        holes: &[Hole],
        lb: &ExplicitDFA,
        ub: &ExplicitDFA,
        reference_dfa: &'a ExplicitDFA,
        max_iters: Option<usize>,
    ) -> Result<HashMap<Hole, C>, CegisError> {
        let constraints = bounds_constraints(store, expr, lb, ub);
        run::<C, Z3>(&constraints, holes, reference_dfa, store, max_iters)
    }

    /// 0 ⊆ Hole ⊆ top: trivially solvable.  The learner returns the empty
    /// SPP (its default when unconstrained), the upper-bound check passes
    /// vacuously, the lower-bound check passes vacuously, done.
    #[test]
    fn trivial_bounds_converge_immediately() {
        let mut store = mk_store();
        let lb = zero_dfa(&store);
        let ub = top_dfa(&store);
        let expr = Expr::hole(Hole(0));
        let rdfa = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<spp::SPP>(
            &mut store,
            &expr,
            &[Hole(0)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
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
        let expr = Expr::spp(top);
        let rdfa = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lb, &ub);
        let err = run_bounds::<spp::SPP>(&mut store, &expr, &[], &lb, &ub, &rdfa, Some(MAX_ITERS))
            .unwrap_err();
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
        let expr = Expr::hole(Hole(0));
        let rdfa = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<spp::SPP>(
            &mut store,
            &expr,
            &[Hole(0)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
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
        let rdfa = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<spp::SPP>(
            &mut store,
            &expr,
            &[Hole(0), Hole(1)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
        .unwrap();
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
        let rdfa = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<spp::SPP>(
            &mut store,
            &expr,
            &[Hole(0), Hole(1)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
        .unwrap();
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
    /// check fails — `add_lower_bound_clause` is exercised end-to-end.
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

        let expr = Expr::hole(Hole(0));
        let rdfa = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<spp::SPP>(
            &mut store,
            &expr,
            &[Hole(0)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
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
        let expr = Expr::hole(Hole(0));
        let rdfa = bounds_reference_dfa::<Cand>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<Cand>(
            &mut store,
            &expr,
            &[Hole(0)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
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
    ///
    /// FIXME: depends too strongly on number of states
    #[test]
    #[ignore]
    fn cand_upper_bound_refinement() {
        let mut store = spp::SPPstore::new(1);
        let lb = zero_dfa(&store);
        let ub = zero_dfa(&store);
        let expr = Expr::hole(Hole(0));
        let rdfa = bounds_reference_dfa::<Cand>(&mut store, &expr, &lb, &ub);
        let result = run_bounds::<Cand>(
            &mut store,
            &expr,
            &[Hole(0)],
            &lb,
            &ub,
            &rdfa,
            Some(MAX_ITERS),
        )
        .unwrap();
        let cand = &result[&Hole(0)];
        assert!(!cand.accepts_input(
            &mut store,
            &Input {
                pkt_in: vec![false],
                pkt_start: vec![false],
                // The reference DFA is the completed (sink-augmented) upper
                // bound, so `zero` (one dead state) grounds a 2-state Cand.
                states: vec![0, 0],
                pkt_end: vec![false],
                pkt_out: vec![false],
            },
        ));
    }

    // ---- min-cut helper ------------------------------------------------

    use petgraph::graph::DiGraph;

    /// Two capacity-1 (hole) edges in series, joined by an infinite (concrete)
    /// backbone edge: the min cut is the single bottleneck, not both holes.
    #[test]
    fn min_cut_series_picks_one() {
        const INF: u32 = 1 << 30;
        let mut g: DiGraph<(), u32> = DiGraph::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let b = g.add_node(());
        let t = g.add_node(());
        let e1 = g.add_edge(s, a, 1); // hole
        g.add_edge(a, b, INF); // concrete backbone
        let _e2 = g.add_edge(b, t, 1); // hole
        let cut = min_cut_edges(&g, s, t);
        // Exactly one hole edge, and it is the one nearest the source.
        assert_eq!(cut, vec![e1]);
    }

    /// Two capacity-1 edges in parallel: both must be cut (no smaller
    /// separator exists), so shrinking would be unsound.
    #[test]
    fn min_cut_parallel_keeps_both() {
        let mut g: DiGraph<(), u32> = DiGraph::new();
        let s = g.add_node(());
        let t = g.add_node(());
        let e1 = g.add_edge(s, t, 1);
        let e2 = g.add_edge(s, t, 1);
        let mut cut = min_cut_edges(&g, s, t);
        cut.sort();
        let mut expected = vec![e1, e2];
        expected.sort();
        assert_eq!(cut, expected);
    }

    /// A diamond: one hole into a fork, then two holes out.  The single
    /// upstream hole is the bottleneck, so the cut is just that one edge.
    #[test]
    fn min_cut_diamond_bottleneck() {
        const INF: u32 = 1 << 30;
        let mut g: DiGraph<(), u32> = DiGraph::new();
        let s = g.add_node(());
        let mid = g.add_node(());
        let x = g.add_node(());
        let y = g.add_node(());
        let t = g.add_node(());
        let bottleneck = g.add_edge(s, mid, 1); // hole
        g.add_edge(mid, x, INF);
        g.add_edge(mid, y, INF);
        g.add_edge(x, t, 1); // hole
        g.add_edge(y, t, 1); // hole
        assert_eq!(min_cut_edges(&g, s, t), vec![bottleneck]);
    }

    /// No source→sink path → empty cut (an unfixable, purely-concrete gap).
    #[test]
    fn min_cut_disconnected_is_empty() {
        let mut g: DiGraph<(), u32> = DiGraph::new();
        let s = g.add_node(());
        let t = g.add_node(());
        let isolated = g.add_node(());
        g.add_edge(s, isolated, 1);
        assert!(min_cut_edges(&g, s, t).is_empty());
    }
}
