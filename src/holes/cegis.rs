//! CEGIS loop: synthesize a concrete candidate per hole that satisfies a
//! list of [`Constraint`]s simultaneously.
//!
//! Each [`Constraint`] relates a hole-bearing automaton
//! ([`crate::holes::nk_with_holes::AutWithHoles`]) to a concrete [`ExplicitDFA`],
//! asking that the instantiated automaton be contained in the DFA ([`Constraint::UpperBound`]),
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
//!      cannot accept all of these simultaneously". A path with a single SPP
//!      traversal instead supplies a symbolic forbidden relation.
//!    * a lower-bound check (`dfa ⊆ aut[candidate]`) uses
//!      [`Instantiate::check_greater_than`]; on failure each viable hole site
//!      becomes a positive literal ([`Candidate::accept_literal`]).
//!
//!    The first failing constraint adds its refining clauses and stops the pass.
//! 4. If every constraint is satisfied, return the candidates.
//! 5. Otherwise, ask the learner for refined candidates and loop.  If the
//!    learner returns an inconsistency, the original problem is infeasible.
//!
//! # Limitations
//!
//! * Symbolic upper-bound generalization requires a single SPP traversal.
//!   Multiple traversals and [`crate::holes::cand::Cand`] use concrete clauses.

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::Direction;
use petgraph::algo::dinics;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::flags;
use crate::holes::aut::{ENFA, ExplicitDFA, backward_reachable, forward_reachable};
use crate::holes::candidate::Candidate;
use crate::holes::inst::{self, Instantiate, LowerBoundCounterexample};
use crate::holes::nk_with_holes::{EdgeLabel, Hole, State};
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
    /// Returns `true` if satisfied. Otherwise records refining clauses on
    /// `learner` — a [`Candidate::reject_literal`] disjunction for an
    /// upper-bound violation (or a forbidden relation for a single SPP
    /// traversal), or a [`Candidate::accept_literal`] disjunction
    /// for a lower-bound one — and returns `false`. Lower-bound violations
    /// learn the forward frontier, plus the backward frontier and min-cut
    /// clauses when [`flags::lb_backward`] and [`flags::lb_mincut`] are enabled.
    /// Equality checks randomize which bound is checked first and stop at
    /// the first counterexample.
    fn check<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
        &self,
        inst: &mut Instantiate<C>,
        hole_to_var: &HashMap<Hole, C::Var>,
        learner: &mut L,
        store: &mut spp::SPPstore,
        reference_dfa: &'a ExplicitDFA,
        metrics: &mut crate::holes::metrics::Metrics,
    ) -> bool {
        let dfa = self.dfa();
        let lower_first =
            matches!(self, Constraint::Equality { .. }) && crate::experiment::positive_first();
        if matches!(self, Constraint::Equality { .. }) {
            metrics.add(
                if lower_first {
                    "positive_first_rounds"
                } else {
                    "negative_first_rounds"
                },
                1,
            );
        }
        for upper in if lower_first {
            [false, true]
        } else {
            [true, false]
        } {
            if upper
                && matches!(
                    self,
                    Constraint::UpperBound { .. } | Constraint::Equality { .. }
                )
            {
                metrics.add("negative_checks", 1);
                let started = metrics.start();
                let result = inst.check_upper_bound(store, dfa);
                metrics.elapsed("check_upper_ns", started);
                if let Err(cex) = result {
                    metrics.add("negative_counterexamples", 1);
                    if crate::experiment::symbolic_upper() && cex.witnesses.len() == 1 {
                        let (hole, (input, inner, output)) = &cex.witnesses[0];
                        if let Literal::Spp { spp, .. } =
                            C::reject_literal(hole_to_var[hole], input, inner, output)
                            && let Some(forbidden) =
                                inst::upper_bound_forbidden(inst, &cex, dfa, store)
                        {
                            debug_assert!(store.accepts(forbidden, input, output));
                            learner.forbid_spp(spp, forbidden, store);
                            return false;
                        }
                    }
                    add_upper_bound_clause::<C, L>(
                        cex.witnesses,
                        hole_to_var,
                        learner,
                        &mut store.sp,
                    );
                    return false;
                }
            }
            if !upper
                && matches!(
                    self,
                    Constraint::LowerBound { .. } | Constraint::Equality { .. }
                )
            {
                metrics.add("positive_checks", 1);
                let started = metrics.start();
                let result = inst.check_greater_than(store, dfa);
                metrics.elapsed("check_lower_ns", started);
                if let Err(cex) = result {
                    metrics.add("positive_counterexamples", 1);
                    add_lower_bound_clause(cex, inst, hole_to_var, learner, store, reference_dfa);
                    return false;
                }
            }
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
    let mut metrics = super::metrics::Metrics::new("cegis");
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
        metrics.add("iterations", 1);
        let started = metrics.start();

        // Visit constraints in order; the first failure adds its clauses and
        // ends the pass so we re-extract before re-checking.
        let mut satisfied = true;
        for (c, inst) in constraints.iter().zip(insts.iter_mut()) {
            if !c.check(
                inst,
                &hole_to_var,
                &mut learner,
                store,
                reference_dfa,
                &mut metrics,
            ) {
                satisfied = false;
                break;
            }
        }
        metrics.elapsed("check_and_learn_ns", started);
        if satisfied {
            // println!("CEGIS: finished in {iters} iterations");
            return Ok(candidates);
        }

        let started = metrics.start();
        let sol = match learner.extract(store) {
            Ok(sol) => sol,
            Err(_) => {
                metrics.elapsed("extract_ns", started);
                return Err(CegisError::Infeasible);
            }
        };
        metrics.elapsed("extract_ns", started);
        let started = metrics.start();
        candidates = hole_to_var
            .iter()
            .map(|(&h, &v)| (h, C::from_solution(&sol, v)))
            .collect();
        for inst in insts.iter_mut() {
            for (&h, &v) in &hole_to_var {
                inst.set_hole(h, C::from_solution(&sol, v));
            }
        }
        metrics.elapsed("install_candidates_ns", started);
        let interval = crate::experiment::stats_interval();
        if metrics.enabled() && interval > 0 && iters.is_multiple_of(interval) {
            eprintln!("KATCH2_CHECKPOINT round={iters}");
            learner.checkpoint();
            metrics.checkpoint();
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
/// [`Instantiate`] reachability map onto the underlying [`crate::holes::nk_with_holes::AutWithHoles`] states.
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

/// Learn a forward-frontier clause from each lower-bound counterexample, plus
/// a backward frontier when [`flags::lb_backward`] is on and a min-cut clause
/// when [`flags::lb_mincut`] is on.
/// All three are necessary conditions for repairing that counterexample.
/// Deduplicate within this batch before membership existentialization.
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
    let reach = lower_bound_reachability(
        &cex,
        inst,
        hole_to_var,
        store,
        reference_dfa,
        flags::lb_backward(),
    );
    let mut clauses = Vec::with_capacity(3);
    if flags::lb_mincut() {
        clauses.push(lower_bound_clause_mincut::<C, L>(
            &cex,
            hole_to_var,
            learner,
            store,
            reference_dfa,
            &reach,
        ));
    }
    let directions: &[FrontierDirection] = if flags::lb_backward() {
        &[FrontierDirection::Forward, FrontierDirection::Backward]
    } else {
        &[FrontierDirection::Forward]
    };
    for &direction in directions {
        let clause = lower_bound_clause_frontier::<C, L>(
            &cex,
            hole_to_var,
            learner,
            store,
            reference_dfa,
            &reach,
            direction,
        );
        if !clauses
            .iter()
            .any(|prior| prior.same_literals(&clause, &mut store.sp))
        {
            clauses.push(clause);
        }
    }
    for clause in clauses {
        learner.add_clause(clause, &mut store.sp);
    }
}

/// All clause builders use the same current instantiation and counterexample.
/// The two frontier directions also share the all-top reachability passes.
struct LowerBoundReachability {
    edges: Vec<(State, EdgeLabel, State)>,
    forward: HashMap<(State, usize), sp::SP>,
    backward: HashMap<(State, usize), sp::SP>,
    forward_top: HashMap<(State, usize), sp::SP>,
    backward_top: HashMap<(State, usize), sp::SP>,
}

fn lower_bound_reachability<'a, C: Candidate<'a>>(
    cex: &LowerBoundCounterexample,
    inst: &mut Instantiate<C>,
    hole_to_var: &HashMap<Hole, C::Var>,
    store: &mut spp::SPPstore,
    reference_dfa: &'a ExplicitDFA,
    backward_frontier: bool,
) -> LowerBoundReachability {
    // Enumerate raw outer states, including suffixes behind empty holes. A
    // backward pass rooted only at the instantiated start would miss those.
    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![inst.start_state()];
    {
        let mut aut = inst.aut();
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
    }
    let roots: Vec<_> = seen.into_iter().map(inst::State::Outer).collect();
    let forward = outer_only(forward_reachable(&*inst, store, &cex.trace));
    let backward = if flags::lb_mincut() || backward_frontier {
        outer_only(backward_reachable(
            &*inst,
            store,
            &cex.trace,
            &cex.output,
            &roots,
        ))
    } else {
        HashMap::new()
    };

    let saved = inst.holes().clone();
    let top = C::top(store, reference_dfa);
    for &hole in hole_to_var.keys() {
        inst.set_hole(hole, top.clone());
    }
    let forward_top = if backward_frontier {
        outer_only(forward_reachable(&*inst, store, &cex.trace))
    } else {
        HashMap::new()
    };
    let backward_top = outer_only(backward_reachable(
        &*inst,
        store,
        &cex.trace,
        &cex.output,
        &roots,
    ));
    for (hole, candidate) in saved {
        inst.set_hole(hole, candidate);
    }

    LowerBoundReachability {
        edges,
        forward,
        backward,
        forward_top,
        backward_top,
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
fn lower_bound_clause_mincut<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    cex: &LowerBoundCounterexample,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut L,
    store: &mut spp::SPPstore,
    reference_dfa: &'a ExplicitDFA,
    reach: &LowerBoundReachability,
) -> AbstractClause
where
    <C as ENFA>::State: Ord,
{
    let n = cex.trace.len();
    let forward = &reach.forward;
    let backward = &reach.backward;

    // Backbone (concrete / connector) edges are uncuttable: larger than any
    // possible cut, which is bounded by the number of capacity-1 hole edges.
    const INF: u32 = 1 << 30;

    let mut graph: DiGraph<(), u32> = DiGraph::new();
    let source = graph.add_node(());
    let sink = graph.add_node(());
    let mut int_nodes: HashMap<(State, usize), NodeIndex> = HashMap::new();
    let mut edge_literals: HashMap<EdgeIndex, Literal> = HashMap::new();

    for (q, label, qp) in &reach.edges {
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
                    let src = region_sp(store, forward, backward, q, i, Region::Int);
                    let tgt = region_sp(store, forward, backward, qp, it, Region::Int);
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
                            let in_sp = region_sp(store, forward, backward, q, i, src_region);
                            let out_sp = region_sp(store, forward, backward, qp, jt, tgt_region);
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
    AbstractClause { literals }
}

#[derive(Clone, Copy)]
enum FrontierDirection {
    Forward,
    Backward,
}

/// Forward frontier: `F_current(source) -> B_top(target) \ F_current(target)`.
/// Backward frontier: `F_top(source) \ B_current(source) -> B_current(target)`.
/// Any repaired accepting path must cross each frontier on a changed hole
/// traversal; concrete transitions are closed under current reachability.
fn lower_bound_clause_frontier<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    cex: &LowerBoundCounterexample,
    hole_to_var: &HashMap<Hole, C::Var>,
    learner: &mut L,
    store: &mut spp::SPPstore,
    reference_dfa: &'a ExplicitDFA,
    reach: &LowerBoundReachability,
    direction: FrontierDirection,
) -> AbstractClause
where
    <C as ENFA>::State: Ord,
{
    let (forward, backward) = match direction {
        FrontierDirection::Forward => (&reach.forward, &reach.backward_top),
        FrontierDirection::Backward => (&reach.forward_top, &reach.backward),
    };

    let sites = collect_hole_sites(
        &reach.edges,
        &cex.trace,
        forward,
        backward,
        direction,
        store,
    );

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
    AbstractClause { literals }
}

/// A single hole site discovered while walking the hole-bearing automaton along
/// the counterexample trace (frontier clause; see [`add_lower_bound_clause`]).
struct HoleSite<'a> {
    hole: Hole,
    /// Carry-in packet set on the source side of the selected frontier.
    in_sp: sp::SP,
    /// Sub-trace the hole consumes internally.
    trace: &'a [Vec<bool>],
    /// Carry-out packet set on the target side of the selected frontier.
    out_sp: sp::SP,
}

/// Inspect the shared raw graph and collect a [`HoleSite`] for each hole edge
/// crossing the selected frontier with non-empty packet sets.
fn collect_hole_sites<'a>(
    edges: &[(State, EdgeLabel, State)],
    trace: &'a [Vec<bool>],
    forward: &HashMap<(State, usize), sp::SP>,
    backward: &HashMap<(State, usize), sp::SP>,
    direction: FrontierDirection,
    store: &mut spp::SPPstore,
) -> Vec<HoleSite<'a>> {
    let n = trace.len();
    let mut sites = Vec::new();
    let sp_zero = store.sp.zero;

    for &(q, ref label, qp) in edges {
        if let EdgeLabel::Abstract(hole) = label {
            let qp_visible = qp.visible;
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
                    let source = forward.get(&(q, i)).copied().unwrap_or(sp_zero);
                    let target = backward.get(&(qp, j_target)).copied().unwrap_or(sp_zero);
                    let (in_sp, out_sp) = match direction {
                        FrontierDirection::Forward => {
                            let already = forward.get(&(qp, j_target)).copied().unwrap_or(sp_zero);
                            let not_already = store.sp.complement(already);
                            (source, store.sp.intersect(target, not_already))
                        }
                        FrontierDirection::Backward => {
                            let already = backward.get(&(q, i)).copied().unwrap_or(sp_zero);
                            let not_already = store.sp.complement(already);
                            (store.sp.intersect(source, not_already), target)
                        }
                    };
                    if store.sp.is_zero(in_sp) || store.sp.is_zero(out_sp) {
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

    sites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holes::aut::ExplicitDFA;
    use crate::holes::cand::{Cand, Input};
    use crate::holes::nk_with_holes::Expr;
    use crate::holes::smt::Native;

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
        run::<C, Native>(&constraints, holes, reference_dfa, store, max_iters)
    }

    #[test]
    fn wide_upper_bound_generalizes_in_one_refinement() {
        let mut store = spp::SPPstore::new(64);
        let packet = vec![false; 64];
        let singleton = singleton_pair_spp(&mut store, &packet, &packet);
        let allowed = store.test(0, false);
        let lower = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![singleton],
        };
        let upper = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![allowed],
        };
        let expr = Expr::hole(Hole(0));
        let reference = bounds_reference_dfa::<spp::SPP>(&mut store, &expr, &lower, &upper);
        // Empty candidate -> lower-bound example -> identity candidate ->
        // symbolic upper-bound region -> candidate satisfying both bounds.
        let solution = run_bounds::<spp::SPP>(
            &mut store,
            &expr,
            &[Hole(0)],
            &lower,
            &upper,
            &reference,
            Some(3),
        )
        .unwrap();
        let learned = solution[&Hole(0)];
        assert!(store.accepts(learned, &packet, &packet));
        let excess = store.difference(learned, allowed);
        assert!(store.is_zero(excess));
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

    #[test]
    fn lower_bound_deduplicates_matching_clauses_before_reextracting() {
        use crate::holes::smt::{CandVar, Existential, InconsistentError, Solution, SppVar};

        struct RecordingLearner<'a> {
            inner: Native<'a>,
            clauses: Vec<AbstractClause>,
        }
        impl<'a> SmtLearner<'a> for RecordingLearner<'a> {
            fn new(num_vars: spp::Var) -> Self {
                Self {
                    inner: Native::new(num_vars),
                    clauses: Vec::new(),
                }
            }
            fn fresh_spp(&mut self) -> SppVar {
                self.inner.fresh_spp()
            }
            fn fresh_cand(&mut self, dfa: &'a ExplicitDFA) -> CandVar {
                self.inner.fresh_cand(dfa)
            }
            fn fresh_existential(&mut self) -> Existential {
                self.inner.fresh_existential()
            }
            fn add_sp_membership(&mut self, sp: sp::SP, vars: &[Existential], store: &sp::SPstore) {
                self.inner.add_sp_membership(sp, vars, store);
            }
            fn add_clause(&mut self, clause: AbstractClause, store: &mut sp::SPstore) {
                self.clauses.push(clause.clone());
                self.inner.add_clause(clause, store);
            }
            fn forbid_spp(&mut self, spp: SppVar, relation: spp::SPP, store: &mut spp::SPPstore) {
                self.inner.forbid_spp(spp, relation, store);
            }
            fn extract(
                &mut self,
                store: &mut spp::SPPstore,
            ) -> Result<Solution<'a>, InconsistentError> {
                self.inner.extract(store)
            }
        }

        for shape in 0..4 {
            let mut store = spp::SPPstore::new(2);
            let input = vec![false, true];
            let output = vec![true, false];
            let singleton = singleton_pair_spp(&mut store, &input, &output);
            let lower = ExplicitDFA {
                start: 0,
                transitions: vec![vec![]],
                outputs: vec![singleton],
            };
            let (expr, holes) = match shape {
                0 => (Expr::hole(Hole(0)), vec![Hole(0)]),
                1 => (
                    Expr::sequence(
                        Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
                        Expr::hole(Hole(2)),
                    ),
                    vec![Hole(0), Hole(1), Hole(2)],
                ),
                2 => (
                    Expr::sequence(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
                    vec![Hole(0), Hole(1)],
                ),
                3 => (
                    Expr::sequence(
                        Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
                        Expr::sequence(
                            Expr::hole(Hole(2)),
                            Expr::union(Expr::hole(Hole(3)), Expr::hole(Hole(4))),
                        ),
                    ),
                    (0..5).map(Hole).collect(),
                ),
                _ => unreachable!(),
            };
            let constraint = Constraint::lower_bound(&mut store, &expr, lower.clone());
            let reference = top_dfa(&store);
            let mut learner = RecordingLearner::new(2);
            let hole_to_var: HashMap<_, _> =
                holes.iter().map(|&h| (h, learner.fresh_spp())).collect();
            let mut inst = Instantiate::new(
                constraint.aut().clone(),
                constraint.start(),
                holes.iter().map(|&h| (h, store.zero)).collect(),
            );

            assert!(!constraint.check(
                &mut inst,
                &hole_to_var,
                &mut learner,
                &mut store,
                &reference,
                &mut crate::holes::metrics::Metrics::new("test")
            ));
            let expected = match shape {
                0 => 1,
                1 => {
                    if flags::lb_mincut() || flags::lb_backward() {
                        2
                    } else {
                        1
                    }
                }
                2 => {
                    if flags::lb_backward() {
                        2
                    } else {
                        1
                    }
                }
                3 => 1 + flags::lb_backward() as usize + flags::lb_mincut() as usize,
                _ => unreachable!(),
            };
            assert_eq!(learner.clauses.len(), expected, "shape={shape}");
            for (idx, clause) in learner.clauses.iter().enumerate() {
                for other in &learner.clauses[..idx] {
                    assert!(!clause.same_literals(other, &mut store.sp));
                }
            }
            // The frontier's all-top pass restores the original candidates.
            assert!(inst.holes().values().all(|&spp| spp == store.zero));
            let solution = learner.extract(&mut store).unwrap();
            // Every retained clause is realized by the trained diagrams.
            for clause in &learner.clauses {
                assert!(clause.literals.iter().any(|literal| {
                    let Literal::SppMember { spp, in_sp, out_sp } = literal else {
                        panic!("expected membership");
                    };
                    (0..4).any(|p| {
                        (0..4).any(|q| {
                            let input = [p & 1 != 0, p & 2 != 0];
                            let output = [q & 1 != 0, q & 2 != 0];
                            store.sp.accepts(*in_sp, &input)
                                && store.sp.accepts(*out_sp, &output)
                                && store.accepts(solution.spps[spp], &input, &output)
                        })
                    })
                }));
            }
            if shape == 0 {
                inst.set_hole(Hole(0), solution.spps[&hole_to_var[&Hole(0)]]);
                assert!(inst.check_greater_than(&mut store, &lower).is_ok());
            }
        }
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

    #[test]
    fn backward_frontier_is_sound_for_all_one_bit_hole_relations() {
        use crate::holes::nk_with_holes::AutWithHoles;
        let mut store = spp::SPPstore::new(1);
        let reference = top_dfa(&store);
        let mut learner = Native::new(1);
        let slots = [learner.fresh_spp(), learner.fresh_spp()];
        let hole_to_var = HashMap::from([(Hole(0), slots[0]), (Hole(1), slots[1])]);
        let expr = Expr::sequence(Expr::hole(Hole(0)), Expr::hole(Hole(1)));
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &expr);
        let relations: Vec<_> = (0..16u32)
            .map(|mask| {
                store.mk(
                    spp::SPP::new(mask & 1),
                    spp::SPP::new((mask >> 1) & 1),
                    spp::SPP::new((mask >> 2) & 1),
                    spp::SPP::new((mask >> 3) & 1),
                )
            })
            .collect();
        let accepts = |mask: u32, input: u32, output: u32| mask & (1 << (2 * input + output)) != 0;
        let composed = |a, b, input, output| {
            (0..2).any(|middle| accepts(a, input, middle) && accepts(b, middle, output))
        };
        for a in 0..16u32 {
            for b in 0..16u32 {
                for input in 0..2 {
                    for output in 0..2 {
                        if composed(a, b, input, output) {
                            continue;
                        }
                        let mut inst = Instantiate::new(
                            aut.clone(),
                            start,
                            HashMap::from([
                                (Hole(0), relations[a as usize]),
                                (Hole(1), relations[b as usize]),
                            ]),
                        );
                        let cex = LowerBoundCounterexample {
                            trace: vec![vec![input != 0]],
                            output: vec![output != 0],
                        };
                        // The backward builder needs both maps even when the
                        // process was configured to omit it from CEGIS.
                        let reach = lower_bound_reachability(
                            &cex,
                            &mut inst,
                            &hole_to_var,
                            &mut store,
                            &reference,
                            true,
                        );

                        let clause = lower_bound_clause_frontier::<spp::SPP, Native>(
                            &cex,
                            &hole_to_var,
                            &mut learner,
                            &mut store,
                            &reference,
                            &reach,
                            FrontierDirection::Backward,
                        );
                        let realizes = |masks: [u32; 2]| {
                            clause.literals.iter().any(|literal| {
                                let Literal::SppMember { spp, in_sp, out_sp } = literal else {
                                    panic!("expected SPP membership");
                                };
                                let idx = if *spp == slots[0] {
                                    0
                                } else {
                                    assert_eq!(*spp, slots[1]);
                                    1
                                };
                                (0..2).any(|p| {
                                    (0..2).any(|q| {
                                        store.sp.accepts(*in_sp, &[p != 0])
                                            && store.sp.accepts(*out_sp, &[q != 0])
                                            && accepts(masks[idx], p, q)
                                    })
                                })
                            })
                        };
                        assert!(
                            !realizes([a, b]),
                            "the clause must rule out the current candidate"
                        );
                        for new_a in 0..16u32 {
                            for new_b in 0..16u32 {
                                if composed(new_a, new_b, input, output) {
                                    assert!(
                                        realizes([new_a, new_b]),
                                        "clause excludes a repair: current ({a}, {b}), repaired ({new_a}, {new_b}), witness ({input}, {output})"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn backward_frontier_preserves_visible_trace_positions() {
        use crate::holes::nk_with_holes::AutWithHoles;
        let mut store = spp::SPPstore::new(1);
        let reference = top_dfa(&store);
        let mut learner = Native::new(1);
        let first = learner.fresh_spp();
        let last = learner.fresh_spp();
        let hole_to_var = HashMap::from([(Hole(0), first), (Hole(1), last)]);
        let expr = Expr::sequence(
            Expr::hole(Hole(0)),
            Expr::sequence(Expr::dup(), Expr::hole(Hole(1))),
        );
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &expr);
        let mut inst = Instantiate::new(
            aut,
            start,
            HashMap::from([(Hole(0), store.zero), (Hole(1), store.zero)]),
        );
        let cex = LowerBoundCounterexample {
            trace: vec![vec![false], vec![true]],
            output: vec![false],
        };
        // Request the shared maps even when this builder is disabled in CEGIS.
        let reach =
            lower_bound_reachability(&cex, &mut inst, &hole_to_var, &mut store, &reference, true);

        let clause = lower_bound_clause_frontier::<spp::SPP, Native>(
            &cex,
            &hole_to_var,
            &mut learner,
            &mut store,
            &reference,
            &reach,
            FrontierDirection::Backward,
        );
        assert_eq!(clause.literals.len(), 1);
        let Literal::SppMember { spp, in_sp, out_sp } = clause.literals[0] else {
            panic!("expected SPP membership");
        };
        assert_eq!(
            spp, last,
            "the backward frontier must find the suffix behind the empty first hole"
        );
        assert!(store.sp.accepts(in_sp, &[true]));
        assert!(!store.sp.accepts(in_sp, &[false]));
        assert!(store.sp.accepts(out_sp, &[false]));
        assert!(!store.sp.accepts(out_sp, &[true]));
        assert!(inst.holes().values().all(|&spp| spp == store.zero));
    }
}
