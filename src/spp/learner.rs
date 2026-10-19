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
//! Each parent first groups its children to prefer unchanged fields, tests,
//! case analysis, and constant assignments, in that order. Then greedily merge
//! compatible groups across the whole field level and fold the graph bottom-up.
//! Configured experiments shuffle equally preferred choices, incoming groups,
//! and merge-target order. All positive and negative requirements are preserved.

use super::{SPP, SPPstore};
use ahash::{HashMap, HashMapExt as _};

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
    learn_spp_with_bounds(examples, store.zero, forbidden, store)
}

/// Learn from concrete examples plus entire required and forbidden relations.
pub fn learn_spp_with_bounds(
    examples: impl IntoIterator<Item = Example>,
    required: SPP,
    forbidden: SPP,
    store: &mut SPPstore,
) -> Result<SPP, ConflictError> {
    let mut positive = required;
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
    let mut evidence = vec![Evidence::new(positive, negative, store)];
    let mut trie = vec![TrieNode::Branch([None; 4])];
    let mut levels = vec![vec![0u32]];
    for depth in 0..nv {
        let mut classes = Vec::<u32>::new();
        let mut assigned = HashMap::<(SPP, SPP), u32>::new();
        let mut slots = Vec::new();
        for &parent in &levels[depth] {
            let cell = evidence[parent as usize];
            let p = store.get(cell.positive);
            let n = store.get(cell.negative);
            let children = std::array::from_fn(|branch| {
                Evidence::new(
                    [p.x00, p.x01, p.x10, p.x11][branch],
                    [n.x00, n.x01, n.x10, n.x11][branch],
                    store,
                )
            });
            for group in preferred_groups(children, store).into_iter().flatten() {
                slots.push((parent, group));
            }
        }
        crate::experiment::shuffle(&mut slots);
        for (parent, group) in slots {
            let cell = group.evidence;
            let key = (cell.positive, cell.negative);
            let child = if let Some(&id) = assigned.get(&key) {
                id
            } else {
                let mut targets = classes.clone();
                crate::experiment::shuffle(&mut targets);
                let compatible = targets
                    .into_iter()
                    .find(|&id| evidence[id as usize].compatible(cell, store));
                let id = if let Some(id) = compatible {
                    evidence[id as usize] = evidence[id as usize].merge(cell, store);
                    id
                } else {
                    let id = trie.len() as u32;
                    evidence.push(cell);
                    trie.push(if depth + 1 == nv {
                        TrieNode::Leaf(cell.has_positive)
                    } else {
                        TrieNode::Branch([None; 4])
                    });
                    classes.push(id);
                    id
                };
                assigned.insert(key, id);
                id
            };
            if let TrieNode::Branch(children) = &mut trie[parent as usize] {
                for (branch, edge) in children.iter_mut().enumerate() {
                    if group.branches & (1 << branch) != 0 {
                        *edge = Some(child);
                    }
                }
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

/// Immutable suffix requirements, pooled only by union. A cell is compatible
/// with the zero continuation exactly when it has no positive examples. Cache
/// that fact here so testing a preferred shape never traverses the evidence DAG.
#[derive(Clone, Copy)]
struct Evidence {
    positive: SPP,
    negative: SPP,
    has_positive: bool,
}

impl Evidence {
    fn new(positive: SPP, negative: SPP, store: &mut SPPstore) -> Self {
        Self {
            positive,
            negative,
            has_positive: !store.is_zero(positive),
        }
    }

    fn compatible(self, other: Self, store: &mut SPPstore) -> bool {
        if self.has_positive {
            let overlap = store.intersect(self.positive, other.negative);
            if !store.is_zero(overlap) {
                return false;
            }
        }
        if other.has_positive {
            let overlap = store.intersect(other.positive, self.negative);
            if !store.is_zero(overlap) {
                return false;
            }
        }
        true
    }

    fn merge(self, other: Self, store: &mut SPPstore) -> Self {
        Self {
            positive: store.union(self.positive, other.positive),
            negative: store.union(self.negative, other.negative),
            has_positive: self.has_positive || other.has_positive,
        }
    }
}

struct ChildGroup {
    branches: u8,
    evidence: Evidence,
}

/// Four branch labels in (input, output) order: 00, 01, 10, 11.
/// Equal labels share a continuation; Z is forced to reject all suffixes.
const Z: u8 = 4;
const PREFERRED_SHAPES: &[&[[u8; 4]]] = &[
    &[[0, Z, Z, 0]],               // Unchanged field.
    &[[0, Z, Z, Z], [Z, Z, Z, 0]], // Test false / true.
    &[[0, Z, Z, 1]],               // Case analysis, preserving the field.
    &[[0, Z, 0, Z], [Z, 0, Z, 0]], // Assign false / true.
    &[[0, 1, 2, 3]],               // General relation.
];

fn shape_fits(shape: [u8; 4], children: &[Evidence; 4], store: &mut SPPstore) -> bool {
    for branch in 0..4 {
        if shape[branch] == Z {
            if children[branch].has_positive {
                return false;
            }
        } else {
            for previous in 0..branch {
                if shape[previous] == shape[branch]
                    && !children[previous].compatible(children[branch], store)
                {
                    return false;
                }
            }
        }
    }
    true
}

/// Make the preferred merges within one parent before considering any other
/// parent. Pairwise compatibility suffices: pooled positives and negatives are
/// unions, so any conflict would already occur between a pair of input cells.
fn preferred_groups(children: [Evidence; 4], store: &mut SPPstore) -> [Option<ChildGroup>; 4] {
    let shape = PREFERRED_SHAPES
        .iter()
        .find_map(|tier| {
            // Each tier has at most two Boolean polarities. Keep temporary
            // shape choices on the stack, even for a large shared DAG.
            let mut order = [0, 1];
            let order = &mut order[..tier.len()];
            crate::experiment::shuffle(order);
            order
                .iter()
                .map(|&i| tier[i])
                .find(|&shape| shape_fits(shape, &children, store))
        })
        .expect("the general four-way shape always fits consistent evidence");
    std::array::from_fn(|label| {
        let mut evidence: Option<Evidence> = None;
        let mut branches = 0;
        for branch in 0..4 {
            if shape[branch] == label as u8 {
                branches |= 1 << branch;
                evidence = Some(match evidence {
                    Some(cell) => cell.merge(children[branch], store),
                    None => children[branch],
                });
            }
        }
        // A purely negative group can be zero immediately. It must not later
        // merge into a positive class and reintroduce an unwanted assignment.
        evidence
            .filter(|cell| cell.has_positive)
            .map(|evidence| ChildGroup { branches, evidence })
    })
}

/// A node of the decision-path trie: an internal 4-way branch (indexed by
/// `2*input + output`) or an accept/reject leaf. Absent children are forced zero,
/// rather than don't-cares, so later cross-parent merges preserve local choices.
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
/// with absent branches set to zero. Each class is folded once, however many
/// parent edges point at it.
fn build_spp(trie: &[TrieNode], levels: &[Vec<u32>], store: &mut SPPstore) -> SPP {
    assert_eq!(levels.len(), 1 + store.num_vars() as usize);

    let mut spp_of: Vec<Option<SPP>> = vec![None; trie.len()];
    let mut zero = SPP::new(0);
    for (i, level) in levels.iter().enumerate().rev() {
        for &node in level {
            let spp = match trie[node as usize] {
                TrieNode::Leaf(label) => SPP::new(label as u32),
                TrieNode::Branch(children) => {
                    let [x00, x01, x10, x11] =
                        children.map(|c| c.map_or(zero, |id| spp_of[id as usize].unwrap()));
                    store.mk(x00, x01, x10, x11)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

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

    #[test]
    fn preferred_one_field_shapes_across_random_seeds() {
        // Branch indices are 00, 01, 10, 11. Negative-only branches must not
        // count as evidence that an assignment is needed.
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[0], &[], &[0, 3]),     // Unchanged, not a test.
            (&[0], &[1, 2], &[0, 3]), // Negative changes still permit identity.
            (&[0], &[3], &[0]),       // Test false.
            (&[3], &[0], &[3]),       // Test true.
            (&[2], &[], &[0, 2]),     // Blind assignment to false.
            (&[1], &[], &[1, 3]),     // Blind assignment to true.
            (&[1], &[3], &[1]),       // Assignment conflicts: general relation.
            (&[0, 1], &[], &[0, 1]),  // Both outputs are genuinely required.
        ];
        for seed in 0..16 {
            crate::experiment::with_seed(seed, || {
                for &(positive, negative, accepted) in cases {
                    let mut store = SPPstore::new(1);
                    let examples = positive
                        .iter()
                        .map(|&b| ex(&[b & 2 != 0], &[b & 1 != 0], true))
                        .chain(
                            negative
                                .iter()
                                .map(|&b| ex(&[b & 2 != 0], &[b & 1 != 0], false)),
                        );
                    let learned = learn_spp(examples, &mut store).unwrap();
                    for branch in 0..4 {
                        assert_eq!(
                            store.accepts(learned, &[branch & 2 != 0], &[branch & 1 != 0]),
                            accepted.contains(&branch),
                            "seed={seed}, positive={positive:?}, negative={negative:?}, branch={branch}"
                        );
                    }
                }
            });
        }
    }

    #[test]
    fn case_analysis_precedes_unnecessary_assignments() {
        for seed in 0..16 {
            crate::experiment::with_seed(seed, || {
                let mut store = SPPstore::new(2);
                // When field 0 is false, preserve field 1. When field 0 is
                // true, assign field 1 to true. The suffix requirements
                // conflict, so the root's two continuations cannot be shared.
                let learned = learn_spp(
                    [
                        ex(&[false, false], &[false, false], true),
                        ex(&[false, false], &[false, true], false),
                        ex(&[true, false], &[true, true], true),
                        ex(&[true, false], &[true, false], false),
                    ],
                    &mut store,
                )
                .unwrap();
                for input in 0..4 {
                    let i = [input & 1 != 0, input & 2 != 0];
                    for output in 0..4 {
                        let o = [output & 1 != 0, output & 2 != 0];
                        assert_eq!(
                            store.accepts(learned, &i, &o),
                            o[0] == i[0] && o[1] == (i[0] || i[1]),
                            "seed={seed}, input={i:?}, output={o:?}"
                        );
                    }
                }
            });
        }
    }

    #[test]
    fn local_groups_still_merge_across_parents() {
        for seed in 0..16 {
            crate::experiment::with_seed(seed, || {
                let mut store = SPPstore::new(3);
                // Both first-level continuations require case analysis, giving
                // four distinct suffix cells at the next level. Some of those
                // cells from different parents are compatible and must still
                // share a class in the subsequent randomized merge pass.
                let examples = [
                    ex(&[false, false, false], &[false, false, false], true),
                    ex(&[false, false, false], &[false, false, true], false),
                    ex(&[false, true, false], &[false, true, true], true),
                    ex(&[false, true, false], &[false, true, false], false),
                    ex(&[true, false, true], &[true, false, true], true),
                    ex(&[true, false, true], &[true, false, false], false),
                    ex(&[true, true, true], &[true, true, false], true),
                    ex(&[true, true, false], &[true, true, true], false),
                ];
                let learned = learn_spp(
                    examples.iter().map(|e| ex(&e.input, &e.output, e.in_spp)),
                    &mut store,
                )
                .unwrap();
                for e in examples {
                    assert_eq!(store.accepts(learned, &e.input, &e.output), e.in_spp);
                }
                // Root + two parents + at most three suffix classes + accept.
                // Without merging across parents, there would be eight nodes.
                assert!(input_sizes()[4] <= 7, "seed={seed}");
            });
        }
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
