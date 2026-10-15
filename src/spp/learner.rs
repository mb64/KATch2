//! One-shot passive learning of an [`SPP`] from concrete examples.
//!
//! [`learn_spp`] takes a set of [`Example`]s — concrete `(input, output)` packet
//! pairs, each tagged in (`in_spp = true`) or out (`false`) of the SPP — and
//! returns an [`SPP`] consistent with all of them, or [`ConflictError`] if two
//! examples give the same pair opposite labels.
//!
//! # Method
//!
//! Finite examples are compiled into shared positive and negative SPPs. Symbolic
//! forbidden regions join the negative evidence. The same DAG learner handles
//! both modes without expanding relations into a tree of packet pairs.
//!
//! At each field level, greedily merge compatible evidence cells and fold the
//! resulting graph bottom-up. Configured experiments shuffle incoming cells
//! and merge-target order. Compatibility preserves all positive and negative
//! requirements; final unspecified branches use the existing heuristic.

use super::{SPP, SPPstore};
use std::collections::HashMap;

/// A single concrete training example for [`learn_spp`].
///
/// The packet pair `(input, output)` is in the learned SPP when `in_spp` is
/// `true`, and excluded when `false`.  Both packets must have the `spp_store`'s
/// field width.
pub struct Example {
    pub input: Vec<bool>,
    pub output: Vec<bool>,
    pub in_spp: bool,
}

/// Two examples gave the same concrete `(input, output)` pair opposite labels.
#[derive(Debug, Clone)]
pub struct ConflictError;

impl std::fmt::Display for ConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "conflicting examples")
    }
}

/// Learn an [`SPP`] consistent with `examples`, in one shot.
///
/// The result **accepts** every `in_spp = true` pair and **rejects** every
/// `in_spp = false` pair; the field width is taken from `spp_store`.  Returns
/// [`ConflictError`] if two examples contradict (same concrete pair, both
/// labels).  With no examples the result is the empty SPP.
pub fn learn_spp(
    examples: impl IntoIterator<Item = Example>,
    store: &mut SPPstore,
) -> Result<SPP, ConflictError> {
    learn_spp_with_forbidden(examples, store.zero, store)
}

/// Learn from concrete examples and an entire symbolic region of negative
/// examples. The forbidden relation is never expanded into packet pairs.
pub fn learn_spp_with_forbidden(
    examples: impl IntoIterator<Item = Example>,
    forbidden: SPP,
    store: &mut SPPstore,
) -> Result<SPP, ConflictError> {
    let mut positive = store.zero;
    let mut negative = forbidden;
    for ex in examples {
        assert_eq!(ex.input.len(), store.num_vars() as usize);
        assert_eq!(ex.output.len(), store.num_vars() as usize);
        let input = store.sp.singleton(&ex.input);
        let output = store.sp.singleton(&ex.output);
        let input = store.ibwd(input);
        let output = store.ifwd(output);
        let pair = store.intersect(input, output);
        if ex.in_spp {
            positive = store.union(positive, pair);
        } else {
            negative = store.union(negative, pair);
        }
    }
    learn_spp_from_relations(positive, negative, store)
}

/// RPNI-style learning from shared positive and negative decision diagrams.
/// Every positive pair is accepted and every negative pair is rejected.
/// Diagram operations and duplicate suffix cells are memoized; we walk shared
/// evidence nodes, rather than unfolding the exponentially many prefix paths.
pub fn learn_spp_from_relations(
    positive: SPP,
    negative: SPP,
    store: &mut SPPstore,
) -> Result<SPP, ConflictError> {
    measure_relations(positive, negative, store);
    let overlap = store.intersect(positive, negative);
    if !store.is_zero(overlap) {
        return Err(ConflictError);
    }
    if store.is_zero(positive) {
        return Ok(store.zero);
    }
    let nv = store.num_vars() as usize;
    if nv == 0 {
        return Ok(positive);
    }

    // Evidence is immutable below the current level. Pooling a class uses
    // hash-consed relation union; it never mutates a shared child's evidence.
    let mut evidence = vec![(positive, negative)];
    let mut trie = vec![TrieNode::Branch([None; 4])];
    let mut levels = vec![vec![0u32]];
    for depth in 0..nv {
        let mut classes = Vec::<u32>::new();
        let mut assigned = HashMap::<(SPP, SPP), u32>::new();
        let mut slots = Vec::new();
        for &parent in &levels[depth] {
            let (pos, neg) = evidence[parent as usize];
            let p = store.get(pos);
            let n = store.get(neg);
            for (branch, (pos, neg)) in [p.x00, p.x01, p.x10, p.x11]
                .into_iter()
                .zip([n.x00, n.x01, n.x10, n.x11])
                .enumerate()
            {
                if store.is_zero(pos) && store.is_zero(neg) {
                    continue;
                }
                slots.push((parent, branch, pos, neg));
            }
        }
        crate::experiment::shuffle(&mut slots);
        for (parent, branch, pos, neg) in slots {
            let child = if let Some(&id) = assigned.get(&(pos, neg)) {
                id
            } else {
                let mut targets = classes.clone();
                crate::experiment::shuffle(&mut targets);
                let compatible = targets.into_iter().find(|&id| {
                    let (p, n) = evidence[id as usize];
                    let pn = store.intersect(p, neg);
                    let np = store.intersect(n, pos);
                    store.is_zero(pn) && store.is_zero(np)
                });
                let id = if let Some(id) = compatible {
                    let (p, n) = evidence[id as usize];
                    evidence[id as usize] = (store.union(p, pos), store.union(n, neg));
                    id
                } else {
                    let id = trie.len() as u32;
                    evidence.push((pos, neg));
                    trie.push(if depth + 1 == nv {
                        TrieNode::Leaf(pos == SPP::new(1))
                    } else {
                        TrieNode::Branch([None; 4])
                    });
                    classes.push(id);
                    id
                };
                assigned.insert((pos, neg), id);
                id
            };
            if let TrieNode::Branch(children) = &mut trie[parent as usize] {
                children[branch] = Some(child);
            }
        }
        levels.push(classes);
    }
    INPUT_SIZES.with(|s| {
        let mut sizes = s.get();
        sizes[4] = levels.iter().map(Vec::len).sum();
        s.set(sizes);
    });
    Ok(build_spp(&trie, &levels, store))
}

/// A node of the decision-path trie: an internal 4-way branch (indexed by
/// `2*input + output`) or an accept/reject leaf.
#[derive(Clone, Copy)]
enum TrieNode {
    Branch([Option<u32>; 4]),
    Leaf(bool),
}

/// Fold the merged, layered trie into an [`SPP`], one whole level at a time
/// from the leaves up.
///
/// A leaf class becomes its accept/reject terminal.  A branch class becomes
/// [`SPPstore::mk`] of its four children (already folded, one level below),
/// with don't-care branches filled by [`heuristic_fill`].  Each class is folded
/// once, however many parent edges point at it.
fn build_spp(trie: &[TrieNode], levels: &[Vec<u32>], store: &mut SPPstore) -> SPP {
    assert_eq!(levels.len(), 1 + store.num_vars() as usize);

    let mut spp_of: Vec<Option<SPP>> = vec![None; trie.len()];
    let mut zero = SPP::new(0);
    for (i, level) in levels.iter().enumerate().rev() {
        for &node in level {
            let spp = match trie[node as usize] {
                TrieNode::Leaf(label) => SPP::new(label as u32),
                TrieNode::Branch(children) => {
                    let spps = children.map(|c| c.map(|id| spp_of[id as usize].unwrap()));
                    heuristic_fill(spps, zero, store)
                }
            };
            spp_of[node as usize] = Some(spp);
        }

        // Except at the first iteration:
        if i != store.num_vars() as usize {
            zero = store.mk(zero, zero, zero, zero);
        }
    }
    spp_of[0].unwrap()
}

fn heuristic_fill(spps: [Option<SPP>; 4], zero: SPP, store: &mut SPPstore) -> SPP {
    // Every branch has ≥1 child to fill the don't-cares with.
    let fill = spps
        .iter()
        .flatten()
        .next()
        .copied()
        .expect("trie branch has at least one child");

    // Heuristic: don't change a field if we don't observe it changing
    let [x00, mut x01, mut x10, x11] = spps;
    if x01.is_none() && x10.is_none() {
        x01 = Some(zero);
        x10 = Some(zero);
    }

    store.mk(
        x00.unwrap_or(fill),
        x01.unwrap_or(fill),
        x10.unwrap_or(fill),
        x11.unwrap_or(fill),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::HashMap;

    const N: u32 = 2;

    fn ex(input: &[bool], output: &[bool], in_spp: bool) -> Example {
        Example {
            input: input.to_vec(),
            output: output.to_vec(),
            in_spp,
        }
    }

    #[test]
    fn empty_is_zero() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp(std::iter::empty(), &mut store).unwrap();
        assert_eq!(spp, store.zero);
    }

    #[test]
    fn single_positive_accepted() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp([ex(&[false, true], &[true, false], true)], &mut store).unwrap();
        assert!(store.accepts(spp, &[false, true], &[true, false]));
    }

    #[test]
    fn single_negative_rejected() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp([ex(&[true, true], &[true, true], false)], &mut store).unwrap();
        assert!(!store.accepts(spp, &[true, true], &[true, true]));
    }

    /// A single negative example generalizes to "reject all"
    #[test]
    fn single_negative_generalizes_to_zero() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp([ex(&[false, false], &[false, false], false)], &mut store).unwrap();
        assert_eq!(spp, store.zero);
    }

    #[test]
    fn conflict_detected() {
        let mut store = SPPstore::new(N);
        let result = learn_spp(
            [
                ex(&[false, false], &[false, false], true),
                ex(&[false, false], &[false, false], false),
            ],
            &mut store,
        );
        assert!(result.is_err(), "contradictory examples must conflict");
    }

    #[test]
    fn duplicate_same_label_ok() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp(
            [
                ex(&[false, true], &[true, false], true),
                ex(&[false, true], &[true, false], true),
            ],
            &mut store,
        )
        .unwrap();
        assert!(store.accepts(spp, &[false, true], &[true, false]));
    }

    /// A mix of positives and negatives over the same field: the result must be
    /// consistent with every training example (accept all positives, reject all
    /// negatives).
    #[test]
    fn consistent_with_all_examples() {
        let mut store = SPPstore::new(N);
        let pos: &[([bool; 2], [bool; 2])] = &[
            ([false, false], [false, false]),
            ([true, false], [true, false]),
        ];
        let neg: &[([bool; 2], [bool; 2])] = &[
            ([false, true], [true, false]),
            ([true, true], [false, true]),
        ];
        let examples: Vec<Example> = pos
            .iter()
            .map(|(i, o)| ex(i, o, true))
            .chain(neg.iter().map(|(i, o)| ex(i, o, false)))
            .collect();
        let spp = learn_spp(examples, &mut store).unwrap();
        for (i, o) in pos {
            assert!(store.accepts(spp, i, o), "positive ({i:?},{o:?}) rejected");
        }
        for (i, o) in neg {
            assert!(!store.accepts(spp, i, o), "negative ({i:?},{o:?}) accepted");
        }
    }

    /// Two prefix cells with non-conflicting suffix evidence must be merged, so
    /// positive evidence seen under one prefix generalizes to the other.  Here
    /// the field-0 cells (0,0) and (1,1) exercise disjoint field-1 branches;
    /// after merging, the accept observed under (0,0) at field 1 (branch (0,0))
    /// carries over to the (1,1) prefix — without merging that cell holds only
    /// the lone negative and collapses to reject-all.
    #[test]
    fn merges_compatible_prefixes() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp(
            [
                ex(&[false, false], &[false, false], true),
                ex(&[true, true], &[true, false], false),
            ],
            &mut store,
        )
        .unwrap();
        assert!(store.accepts(spp, &[true, false], &[true, false]));
    }

    /// The core guarantee, fuzzed: for any non-conflicting example set, the
    /// learned SPP reproduces every training label exactly.
    #[test]
    fn random_consistent() {
        const NV: u32 = 5;
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);

        for _ in 0..200 {
            let mut store = SPPstore::new(NV);

            // Collect non-conflicting examples: the first label wins for a pair.
            let mut labels: HashMap<(Vec<bool>, Vec<bool>), bool> = HashMap::new();
            for _ in 0..50 {
                let input: Vec<bool> = (0..NV).map(|_| rng.random()).collect();
                let output: Vec<bool> = (0..NV).map(|_| rng.random()).collect();
                labels.entry((input, output)).or_insert(rng.random());
            }

            let examples: Vec<Example> = labels.iter().map(|((i, o), &l)| ex(i, o, l)).collect();
            let spp = learn_spp(examples, &mut store).unwrap();

            for ((i, o), &l) in &labels {
                assert_eq!(
                    store.accepts(spp, i, o),
                    l,
                    "example ({i:?},{o:?}) should have label {l}"
                );
            }
        }
    }

    #[test]
    fn generalized_evidence_matches_exhaustive_packet_oracle() {
        let mut rng = StdRng::seed_from_u64(0xBDDAB);
        for _ in 0..200 {
            let mut store = SPPstore::new(2);
            let mut positive = store.zero;
            let mut negative = store.zero;
            let mut labels = Vec::new();
            for input in 0..4 {
                for output in 0..4 {
                    let i = [input & 1 != 0, input & 2 != 0];
                    let o = [output & 1 != 0, output & 2 != 0];
                    let in_sp = store.sp.singleton(&i);
                    let out_sp = store.sp.singleton(&o);
                    let in_rel = store.ibwd(in_sp);
                    let out_rel = store.ifwd(out_sp);
                    let pair = store.intersect(in_rel, out_rel);
                    match rng.random_range(0..3) {
                        0 => {}
                        1 => {
                            positive = store.union(positive, pair);
                            labels.push((i, o, true));
                        }
                        _ => {
                            negative = store.union(negative, pair);
                            labels.push((i, o, false));
                        }
                    }
                }
            }
            let learned = learn_spp_from_relations(positive, negative, &mut store).unwrap();
            for (input, output, label) in labels {
                assert_eq!(store.accepts(learned, &input, &output), label);
            }
        }
    }

    #[test]
    fn generalized_evidence_keeps_wide_shared_diagrams_compact() {
        // Each relation represents exponentially many pairs, with only O(n)
        // shared diagram nodes. Unfolding a trie here cannot finish.
        let mut store = SPPstore::new(96);
        let positive = store.one;
        let negative = store.complement(positive);
        let learned = learn_spp_from_relations(positive, negative, &mut store).unwrap();
        assert_eq!(learned, positive);

        let zero_packet = vec![false; 96];
        let learned =
            learn_spp_with_forbidden([ex(&zero_packet, &zero_packet, true)], negative, &mut store)
                .unwrap();
        assert!(store.accepts(learned, &zero_packet, &zero_packet));
        let overlap = store.intersect(learned, negative);
        assert!(store.is_zero(overlap));
    }

    #[test]
    fn generalized_conflicts_and_zero_width() {
        let mut store = SPPstore::new(2);
        let top = store.top;
        assert!(
            learn_spp_with_forbidden([ex(&[false, true], &[true, false], true)], top, &mut store,)
                .is_err()
        );
        let mut store = SPPstore::new(0);
        let one = store.one;
        let zero = store.zero;
        assert!(learn_spp_from_relations(one, one, &mut store).is_err());
        assert_eq!(
            learn_spp_from_relations(one, zero, &mut store).unwrap(),
            one
        );
        assert_eq!(
            learn_spp_from_relations(zero, one, &mut store).unwrap(),
            zero
        );
    }
}

// Opt-in experiment measurements of actual learner evidence.
thread_local! {
    static INPUT_SIZES: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) };
}
pub(crate) fn input_sizes() -> [usize; 5] {
    INPUT_SIZES.with(|s| s.get())
}
fn measure_relations(positive: SPP, negative: SPP, store: &SPPstore) {
    if !crate::experiment::candidate_stats_enabled() {
        return;
    }
    fn visit(root: SPP, store: &SPPstore, seen: &mut std::collections::BTreeSet<SPP>) {
        if !seen.insert(root) || root.as_u32() <= 1 {
            return;
        }
        let n = store.get(root);
        for c in [n.x00, n.x01, n.x10, n.x11] {
            visit(c, store, seen);
        }
    }
    let mut p = std::collections::BTreeSet::new();
    let mut n = std::collections::BTreeSet::new();
    visit(positive, store, &mut p);
    visit(negative, store, &mut n);
    let sizes = [p.len(), n.len(), p.union(&n).count(), 0, 0];
    INPUT_SIZES.with(|s| s.set(sizes));
}

#[cfg(test)]
mod randomized_tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    #[test]
    fn random_merge_orders_preserve_finite_requirements() {
        for seed in 0..16 {
            crate::experiment::with_seed(seed, || {
                let mut rng = rand::rngs::StdRng::seed_from_u64(123);
                for _ in 0..24 {
                    let mut store = SPPstore::new(3);
                    let mut examples = Vec::new();
                    for i in 0..8 {
                        for o in 0..8 {
                            let label = rng.random_range(0..4);
                            if label < 2 {
                                examples.push(Example {
                                    input: (0..3).map(|b| i & (1 << b) != 0).collect(),
                                    output: (0..3).map(|b| o & (1 << b) != 0).collect(),
                                    in_spp: label == 1,
                                });
                            }
                        }
                    }
                    let result = learn_spp(
                        examples.iter().map(|e| Example {
                            input: e.input.clone(),
                            output: e.output.clone(),
                            in_spp: e.in_spp,
                        }),
                        &mut store,
                    )
                    .unwrap();
                    for e in examples {
                        assert_eq!(
                            store.accepts(result, &e.input, &e.output),
                            e.in_spp,
                            "seed={seed}"
                        );
                    }
                }
            });
        }
    }
}
