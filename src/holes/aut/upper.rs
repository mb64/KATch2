//! Direct ENFA inclusion checking, without materializing epsilon closures.
//!
//! A zero/one-cost worklist propagates packet relations while preserving the
//! preceding visible packet as their input. Invisible and visible steps share
//! the worklist, so hole costs order complete traces. At visible boundaries,
//! intersect with the upper DFA transition before projecting to packet sets.

use super::{DFA, ENFA, ops};
use crate::{
    sp::SP,
    spp::{SPP, SPPstore},
};
use std::collections::HashMap;

struct EpsilonStep<S> {
    state: S,
    relation: SPP,
    predecessors: Vec<(usize, SPP)>,
}

struct Boundary {
    source: usize,
    epsilon: usize,
    label: SPP,
    relation: SPP,
}

struct VisibleStep<S, U> {
    state: (S, U),
    packets: SP,
    predecessors: Vec<Boundary>,
}

/// Pick a middle packet witnessing `prefix(anchor, middle); edge(middle, end)`.
fn preceding_packet(
    store: &mut SPPstore,
    prefix: SPP,
    edge: SPP,
    anchor: &[bool],
    end: &[bool],
) -> Vec<bool> {
    let anchor = store.sp.singleton(anchor);
    let end = store.sp.singleton(end);
    let reached = store.push(anchor, prefix);
    let predecessors = store.pull(edge, end);
    let candidates = store.sp.intersect(reached, predecessors);
    debug_assert!(!store.sp.is_zero(candidates));
    store.sp.any_packet(candidates)
}

fn epsilon_path_back<S: Clone>(
    steps: &[EpsilonStep<S>],
    mut idx: usize,
    anchor: &[bool],
    mut packet: Vec<bool>,
    store: &mut SPPstore,
) -> Vec<(S, Vec<bool>)> {
    let mut path = Vec::new();
    loop {
        let step = &steps[idx];
        path.push((step.state.clone(), packet.clone()));
        if step.predecessors.is_empty() {
            debug_assert_eq!(packet, anchor);
            return path;
        }
        let source = store.sp.singleton(anchor);
        let target = store.sp.singleton(&packet);
        let mut found = None;
        for &(predecessor, label) in &step.predecessors {
            let reached = store.push(source, steps[predecessor].relation);
            let possible = store.pull(label, target);
            let possible = store.sp.intersect(reached, possible);
            if !store.sp.is_zero(possible) {
                found = Some((predecessor, store.sp.any_packet(possible)));
                break;
            }
        }
        (idx, packet) = found.expect("reachable epsilon pair has provenance");
    }
}

/// Find a full counterexample trace using zero/one-cost symbolic reachability.
/// The returned path includes invisible states and the final output packet.
/// The deque saturates free transitions
/// before advancing to another paid traversal, across visible boundaries too.
/// Each epsilon relation retains its own visible anchor; projecting earlier
/// would lose correlation with the upper DFA's transition relation.
pub(crate) fn get_weighted_upper_bound_counterexample<A: ENFA, U: DFA>(
    aut: &A,
    upper: &U,
    store: &mut SPPstore,
    mut split: impl FnMut(&A::State, SPP, &A::State, &mut SPPstore) -> Vec<(SPP, bool)>,
) -> Option<(Vec<(A::State, Vec<bool>)>, Vec<bool>)> {
    use std::collections::VecDeque;
    enum Work<S, U> {
        Visible((S, U), SP, Vec<Boundary>),
        Epsilon(usize, S, SPP, Vec<(usize, SPP)>),
    }
    let complement = ops::complement(upper);
    let start = (aut.start(store), complement.start(store));
    let mut queue = VecDeque::from([Work::Visible(start, store.sp.one, vec![])]);
    let mut done = HashMap::new();
    let mut epsilon_done = HashMap::new();
    let mut visible_steps: Vec<VisibleStep<_, _>> = Vec::new();
    let mut epsilon_steps = Vec::new();
    let mut enfa_nodes = HashMap::new();
    let mut dfa_nodes = HashMap::new();
    while let Some(work) = queue.pop_front() {
        let (visible, q, pending, incoming) = match work {
            Work::Visible(state, pending, incoming) => {
                let prior = done.get(&state).copied().unwrap_or(store.sp.zero);
                let packets = store.sp.difference(pending, prior);
                if store.sp.is_zero(packets) {
                    continue;
                }
                done.insert(state.clone(), store.sp.union(prior, packets));
                let visible = visible_steps.len();
                visible_steps.push(VisibleStep {
                    state: state.clone(),
                    packets,
                    predecessors: incoming,
                });
                let inputs = store.ibwd(packets);
                let seed = store.intersect(store.one, inputs);
                (visible, state.0, seed, vec![])
            }
            Work::Epsilon(visible, q, relation, incoming) => (visible, q, relation, incoming),
        };
        let prior = epsilon_done
            .get(&(visible, q.clone()))
            .copied()
            .unwrap_or(store.zero);
        let relation = store.difference(pending, prior);
        if store.is_zero(relation) {
            continue;
        }
        epsilon_done.insert((visible, q.clone()), store.union(prior, relation));
        let epsilon = epsilon_steps.len();
        epsilon_steps.push(EpsilonStep {
            state: q.clone(),
            relation,
            predecessors: incoming,
        });
        let (edges, output) = enfa_nodes
            .entry(q.clone())
            .or_insert_with(|| (aut.transitions(store, &q), aut.output(store, &q)))
            .clone();
        let state = &visible_steps[visible].state;
        let (dfa_edges, dfa_output) = dfa_nodes
            .entry(state.1.clone())
            .or_insert_with(|| {
                (
                    complement.transitions(store, &state.1),
                    complement.output(store, &state.1),
                )
            })
            .clone();
        let output_relation = store.sequence(relation, output);
        let bad_output = store.intersect(output_relation, dfa_output);
        if !store.is_zero(bad_output) {
            let inputs = store.bwd(bad_output);
            let anchor = store.sp.any_packet(inputs);
            let final_packet = store.any_output_packet_from_input(bad_output, &anchor);
            let current = preceding_packet(store, relation, output, &anchor, &final_packet);
            let mut path_back = epsilon_path_back(&epsilon_steps, epsilon, &anchor, current, store);
            let mut visible_idx = visible;
            let mut packet = anchor;
            while !visible_steps[visible_idx].predecessors.is_empty() {
                let target = store.sp.singleton(&packet);
                let mut found = None;
                for boundary in &visible_steps[visible_idx].predecessors {
                    let possible = store.pull(boundary.relation, target);
                    let possible = store
                        .sp
                        .intersect(possible, visible_steps[boundary.source].packets);
                    if !store.sp.is_zero(possible) {
                        found = Some((boundary, store.sp.any_packet(possible)));
                        break;
                    }
                }
                let (boundary, anchor) = found.expect("reachable visible packet has provenance");
                let step = &epsilon_steps[boundary.epsilon];
                let current =
                    preceding_packet(store, step.relation, boundary.label, &anchor, &packet);
                path_back.extend(epsilon_path_back(
                    &epsilon_steps,
                    boundary.epsilon,
                    &anchor,
                    current,
                    store,
                ));
                visible_idx = boundary.source;
                packet = anchor;
            }
            debug_assert_eq!(
                path_back.last().unwrap().0,
                visible_steps[visible_idx].state.0
            );
            path_back.reverse();
            return Some((path_back, final_packet));
        }
        for (label, target) in edges {
            for (label, paid) in split(&q, label, &target, store) {
                let next = store.sequence(relation, label);
                if store.is_zero(next) {
                    continue;
                }
                if aut.is_visible(store, &target) {
                    for (dfa_label, dfa_target) in &dfa_edges {
                        let valid = store.intersect(next, *dfa_label);
                        if store.is_zero(valid) {
                            continue;
                        }
                        let packets = store.fwd(valid);
                        let work = Work::Visible(
                            (target.clone(), dfa_target.clone()),
                            packets,
                            vec![Boundary {
                                source: visible,
                                epsilon,
                                label,
                                relation: valid,
                            }],
                        );
                        if paid {
                            queue.push_back(work);
                        } else {
                            queue.push_front(work);
                        }
                    }
                } else {
                    let work = Work::Epsilon(visible, target.clone(), next, vec![(epsilon, label)]);
                    if paid {
                        queue.push_back(work);
                    } else {
                        queue.push_front(work);
                    }
                }
            }
        }
    }
    None
}
