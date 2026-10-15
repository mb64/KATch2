//! Direct ENFA inclusion checking, without materializing epsilon closures.
//!
//! The outer worklist propagates packet sets at visible product states. For
//! each newly reached set, an inner worklist propagates SPPs through invisible
//! states, keeping the preceding visible packet as the relation's input. On
//! entering a visible state, intersect that relation with a DFA transition
//! *before* projecting to packets. Forgetting the preceding visible packet
//! would incorrectly combine incompatible epsilon paths and DFA transitions.

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

/// Find a trace accepted by `aut` and rejected by `upper`. The returned path
/// contains every ENFA state, including invisible intermediates, and the
/// final output packet. No closure NFA or trace elaboration is needed.
pub fn get_upper_bound_counterexample<A: ENFA, U: DFA>(
    aut: &A,
    upper: &U,
    store: &mut SPPstore,
) -> Option<(Vec<(A::State, Vec<bool>)>, Vec<bool>)> {
    let complement = ops::complement(upper);
    let start = (aut.start(store), complement.start(store));
    let mut todo = HashMap::from([(start.clone(), store.sp.one)]);
    let mut done = HashMap::<_, SP>::new();
    let mut worklist = vec![start];
    let mut predecessors = HashMap::<_, Vec<Boundary>>::new();
    let mut visible_steps = Vec::new();
    let mut epsilon_steps = Vec::new();
    // These caches live for one immutable candidate check. Provenance retains
    // the actual edge labels, so reconstruction never queries transitions.
    let mut enfa_nodes = HashMap::new();
    let mut dfa_nodes = HashMap::new();

    while let Some(state) = worklist.pop() {
        let Some(pending) = todo.remove(&state) else {
            continue;
        };
        let prior = done.get(&state).copied().unwrap_or(store.sp.zero);
        let packets = store.sp.difference(pending, prior);
        let incoming = predecessors.remove(&state).unwrap_or_default();
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
        let (dfa_edges, dfa_output) = dfa_nodes
            .entry(state.1.clone())
            .or_insert_with(|| {
                (
                    complement.transitions(store, &state.1),
                    complement.output(store, &state.1),
                )
            })
            .clone();

        let inputs = store.ibwd(packets);
        let seed = store.intersect(store.one, inputs);
        let mut epsilon_todo = HashMap::from([(state.0.clone(), seed)]);
        let mut epsilon_done = HashMap::<_, SPP>::new();
        let mut epsilon_worklist = vec![state.0.clone()];
        let mut epsilon_predecessors = HashMap::<_, Vec<(usize, SPP)>>::new();

        while let Some(q) = epsilon_worklist.pop() {
            let Some(pending) = epsilon_todo.remove(&q) else {
                continue;
            };
            let prior = epsilon_done.get(&q).copied().unwrap_or(store.zero);
            let relation = store.difference(pending, prior);
            let incoming = epsilon_predecessors.remove(&q).unwrap_or_default();
            if store.is_zero(relation) {
                continue;
            }
            epsilon_done.insert(q.clone(), store.union(prior, relation));
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

            let output_relation = store.sequence(relation, output);
            let bad_output = store.intersect(output_relation, dfa_output);
            if !store.is_zero(bad_output) {
                let inputs = store.bwd(bad_output);
                let anchor = store.sp.any_packet(inputs);
                let final_packet = store.any_output_packet_from_input(bad_output, &anchor);
                let current = preceding_packet(store, relation, output, &anchor, &final_packet);
                let mut path_back =
                    epsilon_path_back(&epsilon_steps, epsilon, &anchor, current, store);
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
                    let (boundary, anchor) =
                        found.expect("reachable visible packet has provenance");
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
                        let next_packets = store.fwd(valid);
                        let target = (target.clone(), dfa_target.clone());
                        let processed = done.get(&target).copied().unwrap_or(store.sp.zero);
                        let next_packets = store.sp.difference(next_packets, processed);
                        let pending = todo.get(&target).copied().unwrap_or(store.sp.zero);
                        let merged = store.sp.union(pending, next_packets);
                        if merged != pending {
                            todo.insert(target.clone(), merged);
                            predecessors
                                .entry(target.clone())
                                .or_default()
                                .push(Boundary {
                                    source: visible,
                                    epsilon,
                                    label,
                                    relation: valid,
                                });
                            worklist.push(target);
                        }
                    }
                } else {
                    let processed = epsilon_done.get(&target).copied().unwrap_or(store.zero);
                    let next = store.difference(next, processed);
                    let pending = epsilon_todo.get(&target).copied().unwrap_or(store.zero);
                    let merged = store.union(pending, next);
                    if merged != pending {
                        epsilon_todo.insert(target.clone(), merged);
                        epsilon_predecessors
                            .entry(target.clone())
                            .or_default()
                            .push((epsilon, label));
                        epsilon_worklist.push(target);
                    }
                }
            }
        }
    }
    None
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
