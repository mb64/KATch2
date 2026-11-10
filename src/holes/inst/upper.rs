//! Generalize a single hole traversal on an upper-bound counterexample path.
//!
//! Keep the path's state sequence and all other trace packets fixed, but let
//! the two visible endpoints surrounding the hole vary. Forward/backward
//! product reachability supplies packet regions at these endpoints. The
//! complement DFA's intervening relation retains correlations between them.
//! If `bad` relates those endpoints, and the concrete context is `A; P; B`,
//! the forbidden hole relation is `flip(A); bad; flip(B)`.

use super::{Instantiate, State, UpperBoundCounterexample};
use crate::holes::aut::{ENFA, ExplicitDFA, NFA, ops};
use crate::spp::{SPP, SPPstore};

#[derive(Clone, Copy)]
enum Segment {
    Concrete(SPP),
    Hole { before: SPP, after: SPP },
}

/// Recover the concrete SPP segments between visible states on this path.
/// The hole must be traversed exactly once. Its internal packets are retained
/// in the witness rather than treated as concrete context segments.
fn path_segments<C: NFA>(
    inst: &Instantiate<C>,
    cex: &UpperBoundCounterexample<C::State>,
    store: &mut SPPstore,
) -> Option<(Vec<Segment>, Vec<Vec<bool>>)> {
    if cex.witnesses.len() != 1 {
        return None;
    }
    let mut segments = Vec::new();
    let mut packets = vec![cex.path.first()?.1.clone()];
    let mut before = store.one;
    let mut after = store.one;
    let mut has_hole = false;
    for (idx, (source, input)) in cex.path.iter().enumerate() {
        let (relation, boundary) = if let Some((target, output)) = cex.path.get(idx + 1) {
            if matches!(source, State::Entering(_, _)) {
                if has_hole {
                    return None;
                }
                has_hole = true;
                (
                    None,
                    (matches!(target, State::Outer(_)) && inst.is_visible(store, target))
                        .then_some(output),
                )
            } else if matches!(source, State::Inner(_, _, _)) {
                (
                    None,
                    (matches!(target, State::Outer(_)) && inst.is_visible(store, target))
                        .then_some(output),
                )
            } else {
                // Only the chosen state path is explored, but all concrete
                // labels between these same states may safely be unioned.
                let mut relation = store.zero;
                for (label, next) in inst.transitions(store, source) {
                    if next == *target {
                        relation = store.union(relation, label);
                    }
                }
                if !store.accepts(relation, input, output) {
                    return None;
                }
                (
                    Some(relation),
                    inst.is_visible(store, target).then_some(output),
                )
            }
        } else {
            let relation = inst.output(store, source);
            if !store.accepts(relation, input, &cex.output) {
                return None;
            }
            (Some(relation), Some(&cex.output))
        };
        if let Some(relation) = relation {
            if has_hole {
                after = store.sequence(after, relation);
            } else {
                before = store.sequence(before, relation);
            }
        }
        if let Some(packet) = boundary {
            segments.push(if has_hole {
                Segment::Hole { before, after }
            } else {
                Segment::Concrete(before)
            });
            packets.push(packet.clone());
            before = store.one;
            after = store.one;
            has_hole = false;
        }
    }
    Some((segments, packets))
}

pub(crate) fn upper_bound_forbidden<C: NFA>(
    inst: &Instantiate<C>,
    cex: &UpperBoundCounterexample<C::State>,
    upper: &ExplicitDFA,
    store: &mut SPPstore,
) -> Option<SPP> {
    match upper_bound_region(inst, cex, upper, store)? {
        Forbidden::Spp(r) => Some(r),
        Forbidden::Cand(_) => None,
    }
}

/// Correlated regions are a union of products over the incoming and outgoing
/// packet pairs. The concrete internal trace (and thus its state vector) stays
/// fixed; the packets outside the hole vary symbolically.
pub(crate) fn upper_bound_forbidden_cand<C: NFA>(
    inst: &Instantiate<C>,
    cex: &UpperBoundCounterexample<C::State>,
    upper: &ExplicitDFA,
    store: &mut SPPstore,
) -> Option<Vec<(SPP, SPP)>> {
    match upper_bound_region(inst, cex, upper, store)? {
        Forbidden::Cand(regions) => Some(regions),
        Forbidden::Spp(_) => None,
    }
}

enum Forbidden {
    Spp(SPP),
    Cand(Vec<(SPP, SPP)>),
}

fn upper_bound_region<C: NFA>(
    inst: &Instantiate<C>,
    cex: &UpperBoundCounterexample<C::State>,
    upper: &ExplicitDFA,
    store: &mut SPPstore,
) -> Option<Forbidden> {
    let (segments, packets) = path_segments(inst, cex, store)?;
    let mut holes = segments.iter().enumerate().filter_map(|(i, s)| match s {
        Segment::Hole { before, after } => Some((i, *before, *after)),
        _ => None,
    });
    let (hole, before, after) = holes.next()?;
    if holes.next().is_some() {
        return None;
    }

    // Complete the complement, including its rejecting-bound sink. Missing
    // upper-bound transitions must contribute forbidden pairs as well.
    let complement = ops::complement(upper);
    let size = upper.num_states() + 1;
    let mut transitions = Vec::with_capacity(size);
    let mut outputs = Vec::with_capacity(size);
    for idx in 0..size {
        let state = if idx == size - 1 {
            ops::CompState::Sink
        } else {
            ops::CompState::Inner(idx)
        };
        transitions.push(
            complement
                .transitions(store, &state)
                .into_iter()
                .map(|(r, q)| {
                    (
                        r,
                        match q {
                            ops::CompState::Inner(idx) => idx,
                            ops::CompState::Sink => size - 1,
                        },
                    )
                })
                .collect::<Vec<_>>(),
        );
        outputs.push(complement.output(store, &state));
    }
    let fixed: Vec<_> = packets.iter().map(|p| store.sp.singleton(p)).collect();
    let mut forward = vec![store.sp.zero; size];
    forward[upper.start] = if hole == 0 { store.sp.one } else { fixed[0] };
    for (idx, segment) in segments.iter().enumerate().take(hole) {
        let Segment::Concrete(path) = *segment else {
            return None;
        };
        let mut next = vec![store.sp.zero; size];
        for q in 0..size {
            if store.sp.is_zero(forward[q]) {
                continue;
            }
            for &(relation, target) in &transitions[q] {
                let relation = store.intersect(path, relation);
                let mut reachable = store.push(forward[q], relation);
                if idx + 1 != hole {
                    reachable = store.sp.intersect(reachable, fixed[idx + 1]);
                }
                next[target] = store.sp.union(next[target], reachable);
            }
        }
        forward = next;
    }

    // Backward product reachability for the concrete suffix. The first
    // packet after the hole-containing segment is symbolic; later packets
    // retain their original trace values, including the final output.
    let mut backward = vec![store.sp.zero; size];
    for idx in (hole + 1..segments.len()).rev() {
        let Segment::Concrete(path) = segments[idx] else {
            return None;
        };
        let mut previous = vec![store.sp.zero; size];
        for q in 0..size {
            if idx + 1 == segments.len() {
                let relation = store.intersect(path, outputs[q]);
                previous[q] = store.pull(relation, fixed[idx + 1]);
            } else {
                for &(relation, target) in &transitions[q] {
                    let relation = store.intersect(path, relation);
                    let reachable = store.pull(relation, backward[target]);
                    previous[q] = store.sp.union(previous[q], reachable);
                }
            }
            if idx != hole + 1 {
                previous[q] = store.sp.intersect(previous[q], fixed[idx]);
            }
        }
        backward = previous;
    }

    let inner = &cex.witnesses[0].1.1;
    if !inner.is_empty() {
        let first = store.sp.singleton(&inner.first()?.1);
        let last = store.sp.singleton(&inner.last()?.1);
        let mut origins = vec![store.sp.zero; size];
        for q in 0..size {
            for &(relation, target) in &transitions[q] {
                let pre = store.pull(relation, first);
                let pre = store.sp.intersect(pre, forward[q]);
                origins[target] = store.sp.union(origins[target], pre);
            }
        }
        for pair in inner.windows(2) {
            let mut next = vec![store.sp.zero; size];
            for q in 0..size {
                for &(relation, target) in &transitions[q] {
                    if store.accepts(relation, &pair[0].1, &pair[1].1) {
                        next[target] = store.sp.union(next[target], origins[q]);
                    }
                }
            }
            origins = next;
        }
        let mut regions = Vec::new();
        for q in 0..size {
            if store.sp.is_zero(origins[q]) {
                continue;
            }
            let mut post = store.sp.zero;
            if hole + 1 == segments.len() {
                post = store.push(last, outputs[q]);
            } else {
                for &(relation, target) in &transitions[q] {
                    let image = store.push(last, relation);
                    let image = store.sp.intersect(image, backward[target]);
                    post = store.sp.union(post, image);
                }
            }
            let pre = store.push(origins[q], before);
            let post = store.pull(after, post);
            if store.sp.is_zero(pre) || store.sp.is_zero(post) {
                continue;
            }
            let pre = store.ibwd(pre);
            let first = store.ifwd(first);
            let incoming = store.intersect(pre, first);
            let last = store.ibwd(last);
            let post = store.ifwd(post);
            let outgoing = store.intersect(last, post);
            regions.push((incoming, outgoing));
        }
        return Some(Forbidden::Cand(regions));
    }

    let mut bad = store.zero;
    for q in 0..size {
        if store.sp.is_zero(forward[q]) {
            continue;
        }
        let source = store.ibwd(forward[q]);
        if hole + 1 == segments.len() {
            let contribution = store.intersect(source, outputs[q]);
            bad = store.union(bad, contribution);
        } else {
            for &(relation, target) in &transitions[q] {
                if store.sp.is_zero(backward[target]) {
                    continue;
                }
                let target = store.ifwd(backward[target]);
                let contribution = store.intersect(source, relation);
                let contribution = store.intersect(contribution, target);
                bad = store.union(bad, contribution);
            }
        }
    }
    let before = store.flip(before);
    let after = store.flip(after);
    let bad = store.sequence(before, bad);
    Some(Forbidden::Spp(store.sequence(bad, after)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{HashMap, HashMapExt as _};
    use crate::holes::aut::expr_to_dfa;
    use crate::holes::nk_with_holes::{AutWithHoles, Expr, Hole};

    #[test]
    fn cand_regions_are_sound_for_every_one_bit_concrete_context() {
        use crate::holes::cand::{Cand, Input};
        let mut store = SPPstore::new(1);
        let reference = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.zero],
        };
        let upper = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.one],
        };
        for a in 0..16 {
            for b in 0..16 {
                let a = relation(a, &mut store);
                let b = relation(b, &mut store);
                let expr = Expr::sequence(
                    Expr::spp(a),
                    Expr::sequence(Expr::hole(Hole(0)), Expr::spp(b)),
                );
                let mut aut = AutWithHoles::new();
                let start = aut.expr_to_state(&mut store, &expr);
                let top = Cand::constant(&mut store, &reference, true);
                let mut inst = Instantiate::new(aut, start, HashMap::from_iter([(Hole(0), top)]));
                let Err(cex) = inst.check_upper_bound(&mut store, &upper, &HashMap::new()) else {
                    continue;
                };
                let regions = upper_bound_forbidden_cand(&inst, &cex, &upper, &mut store).unwrap();
                let inner = &cex.witnesses[0].1.1;
                let first = inner.first().unwrap().1.clone();
                let last = inner.last().unwrap().1.clone();
                assert!(!regions.is_empty());
                for x in [false, true] {
                    for y in [false, true] {
                        let forbidden = regions.iter().any(|&(incoming, outgoing)| {
                            store.accepts(incoming, &[x], &first)
                                && store.accepts(outgoing, &last, &[y])
                        });
                        if !forbidden {
                            continue;
                        }
                        let input = Input {
                            pkt_in: vec![x],
                            pkt_start: first.clone(),
                            states: vec![0],
                            pkt_end: last.clone(),
                            pkt_out: vec![y],
                        };
                        let singleton = Cand::singleton(&mut store, &reference, &input);
                        inst.set_hole(Hole(0), singleton);
                        assert!(
                            inst.check_upper_bound(&mut store, &upper, &HashMap::new())
                                .is_err(),
                            "A={a:?} B={b:?} input={x} output={y}"
                        );
                    }
                }
                let (_, (x, _, y)) = &cex.witnesses[0];
                assert!(
                    regions
                        .iter()
                        .any(|&(a, b)| store.accepts(a, x, &first) && store.accepts(b, &last, y))
                );
            }
        }
    }

    #[test]
    fn cand_regions_cross_visible_context_and_multiple_internal_packets() {
        use crate::holes::cand::Cand;
        let mut store = SPPstore::new(1);
        let reference = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 1)], vec![(store.top, 0)]],
            outputs: vec![store.zero; 2],
        };
        let upper = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.one],
        };
        let top = store.top;
        let expr = Expr::sequence(
            Expr::dup(),
            Expr::sequence(
                Expr::hole(Hole(0)),
                Expr::sequence(Expr::dup(), Expr::spp(top)),
            ),
        );
        let candidate = Cand::product(&mut store, &reference, top, Some(&[1, 0]), top);
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(&mut store, &expr);
        let inst = Instantiate::new(aut, start, HashMap::from_iter([(Hole(0), candidate)]));
        let cex = inst
            .check_upper_bound(&mut store, &upper, &HashMap::new())
            .unwrap_err();
        let inner = &cex.witnesses[0].1.1;
        assert!(inner.len() >= 2);
        let regions = upper_bound_forbidden_cand(&inst, &cex, &upper, &mut store).unwrap();
        let (_, (input, _, output)) = &cex.witnesses[0];
        assert!(
            regions
                .iter()
                .any(|&(a, b)| store.accepts(a, input, &inner[0].1)
                    && store.accepts(b, &inner.last().unwrap().1, output))
        );
    }

    fn instantiate(store: &mut SPPstore, expr: &Expr, candidate: SPP) -> Instantiate<SPP> {
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(store, expr);
        Instantiate::new(aut, start, HashMap::from_iter([(Hole(0), candidate)]))
    }

    fn relation(mask: u32, store: &mut SPPstore) -> SPP {
        let bit = |i: u32| SPP::new((mask >> i) & 1u32);
        store.mk(bit(0), bit(1), bit(2), bit(3))
    }

    #[test]
    fn one_hole_preserves_input_output_correlations() {
        let mut store = SPPstore::new(3);
        let upper = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.one],
        };
        let top = store.top;
        let inst = instantiate(&mut store, &Expr::hole(Hole(0)), top);
        let cex = inst
            .check_upper_bound(&mut store, &upper, &HashMap::new())
            .unwrap_err();
        let forbidden = upper_bound_forbidden(&inst, &cex, &upper, &mut store).unwrap();
        assert_eq!(forbidden, store.complement(store.one));
    }

    #[test]
    fn concrete_sandwich_matches_exhaustive_relational_abduction() {
        let mut store = SPPstore::new(1);
        for a_mask in 0..16 {
            for b_mask in 0..16 {
                for upper_mask in 0..16 {
                    let a = relation(a_mask, &mut store);
                    let b = relation(b_mask, &mut store);
                    let upper_rel = relation(upper_mask, &mut store);
                    let upper = ExplicitDFA {
                        start: 0,
                        transitions: vec![vec![]],
                        outputs: vec![upper_rel],
                    };
                    let expr = Expr::sequence(
                        Expr::spp(a),
                        Expr::sequence(Expr::hole(Hole(0)), Expr::spp(b)),
                    );
                    let top = store.top;
                    let inst = instantiate(&mut store, &expr, top);
                    let forbidden =
                        match inst.check_upper_bound(&mut store, &upper, &HashMap::new()) {
                            Ok(()) => store.zero,
                            Err(cex) => {
                                assert_eq!(cex.witnesses.len(), 1);
                                upper_bound_forbidden(&inst, &cex, &upper, &mut store).unwrap()
                            }
                        };
                    for x in [false, true] {
                        for y in [false, true] {
                            let expected = [false, true].into_iter().any(|p0| {
                                [false, true].into_iter().any(|p3| {
                                    store.accepts(a, &[p0], &[x])
                                        && store.accepts(b, &[y], &[p3])
                                        && !store.accepts(upper_rel, &[p0], &[p3])
                                })
                            });
                            assert_eq!(
                                store.accepts(forbidden, &[x], &[y]),
                                expected,
                                "A={a_mask}, B={b_mask}, upper={upper_mask}, x={x}, y={y}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn generalized_trace_context_is_sound_before_between_and_after_dups() {
        let arena = &bumpalo::Bump::new();
        let mut store = SPPstore::new(1);
        // Explore hole-containing transition segments and final output
        // segments, as well as prefix/suffix products with multiple DFA states.
        for position in 0..3 {
            for a_mask in 1..16 {
                for b_mask in [3, 6, 9, 12, 15] {
                    let a = relation(a_mask, &mut store);
                    let b = relation(b_mask, &mut store);
                    let middle = Expr::sequence(
                        Expr::spp(a),
                        Expr::sequence(Expr::hole(Hole(0)), Expr::spp(b)),
                    );
                    let identity = Expr::spp(store.one);
                    let segments = match position {
                        0 => [middle, identity.clone(), identity],
                        1 => [identity.clone(), middle, identity],
                        _ => [identity.clone(), identity, middle],
                    };
                    let expr = Expr::sequence(
                        segments[0].clone(),
                        Expr::sequence(
                            Expr::dup(),
                            Expr::sequence(
                                segments[1].clone(),
                                Expr::sequence(Expr::dup(), segments[2].clone()),
                            ),
                        ),
                    );
                    let upper_expr = crate::expr::Expr::sequence(
                        arena,
                        crate::expr::Expr::one(arena),
                        crate::expr::Expr::sequence(
                            arena,
                            crate::expr::Expr::dup(arena),
                            crate::expr::Expr::sequence(
                                arena,
                                crate::expr::Expr::one(arena),
                                crate::expr::Expr::sequence(
                                    arena,
                                    crate::expr::Expr::dup(arena),
                                    crate::expr::Expr::one(arena),
                                ),
                            ),
                        ),
                    );
                    let upper = expr_to_dfa(upper_expr, &mut store);
                    let top = store.top;
                    let inst = instantiate(&mut store, &expr, top);
                    let Err(cex) = inst.check_upper_bound(&mut store, &upper, &HashMap::new())
                    else {
                        continue;
                    };
                    let forbidden = upper_bound_forbidden(&inst, &cex, &upper, &mut store).unwrap();
                    let witness = &cex.witnesses[0].1;
                    assert!(store.accepts(forbidden, &witness.0, &witness.2));
                    for pair in 0..4 {
                        let singleton = relation(1 << pair, &mut store);
                        if store.accepts(forbidden, &[pair & 2 != 0], &[pair & 1 != 0]) {
                            let candidate = instantiate(&mut store, &expr, singleton);
                            assert!(candidate.check_less_than(&mut store, &upper).is_err());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn multiple_hole_occurrences_keep_the_original_clause() {
        let mut store = SPPstore::new(1);
        let upper = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.zero],
        };
        let expr = Expr::sequence(Expr::hole(Hole(0)), Expr::hole(Hole(0)));
        let top = store.top;
        let inst = instantiate(&mut store, &expr, top);
        let cex = inst
            .check_upper_bound(&mut store, &upper, &HashMap::new())
            .unwrap_err();
        assert_eq!(cex.witnesses.len(), 2);
        assert!(upper_bound_forbidden(&inst, &cex, &upper, &mut store).is_none());
    }
}
