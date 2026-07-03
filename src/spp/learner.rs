//! One-shot passive learning of an [`SPP`] from concrete examples.
//!
//! [`learn_spp`] takes a set of [`Example`]s — concrete `(input, output)` packet
//! pairs, each tagged in (`in_spp = true`) or out (`false`) of the SPP — and
//! returns an [`SPP`] consistent with all of them, or [`ConflictError`] if two
//! examples give the same pair opposite labels.
//!
//! # Method (trie + RPNI-style per-level merging)
//!
//! An [`SPP`] over `n` fields is an order-`n` decision diagram: level `i` is a
//! 4-way branch on the `(input[i], output[i])` bit pair, terminating in
//! accept/reject.  We build it in three steps:
//!
//! 1. **Trie.** Insert every example's length-`n` *decision path* —
//!    `2*input[i] + output[i]` at field `i` — into a 4-ary trie, leaves carrying
//!    the accept/reject label.  This computes *all* the prefix partitions in one
//!    pass: a node at depth `d` is exactly the set of examples sharing that
//!    length-`d` prefix, and a shared prefix is walked only once.  A leaf reached
//!    with two different labels is a [`ConflictError`].
//! 2. **Per-level merging (RPNI-style).** Sweep the trie top-down, one level at
//!    a time.  The nodes at depth `d` are the length-`d` prefix cells of the
//!    examples; two cells are *compatible* when their suffix evidence never
//!    conflicts, i.e. no decision path both exercise ends in different labels.
//!    Each node is greedily merged into the first compatible class at its
//!    level, unioning the two subtrees, so evidence pooled at one level narrows
//!    the don't-cares at every level below.  This is the state merging the
//!    pre-trie learner did with explicit example partitions.
//! 3. **Bottom-up fold.** Walk the merged levels leaves-first: a leaf class
//!    becomes its accept (`1`) / reject (`0`) terminal; a branch class becomes
//!    [`SPPstore::mk`] of its four children, with don't-care branches (those no
//!    example exercises) filled heuristically — the eager generalization.
//!    Hash-consing in `mk` collapses any identical classes the greedy merge
//!    missed.
//!
//! Building the trie first means each partition is computed once instead of being
//! re-derived at every level.  Merging only unions non-conflicting evidence and
//! fills only touch branches no example exercises, so an example's own path is
//! never redirected: the result accepts every positive example and rejects every
//! negative one, and the generalization only touches inputs no example pins down.

use super::{SPP, SPPstore};

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
    spp_store: &mut SPPstore,
) -> Result<SPP, ConflictError> {
    let nv = spp_store.num_vars() as usize;
    let examples: Vec<Example> = examples.into_iter().collect();

    // No evidence anywhere → reject everything.
    if examples.is_empty() {
        return Ok(spp_store.zero);
    }

    // Zero-field packets: every example is the single empty pair, so they must
    // all agree (there is no trie level to disambiguate them).
    if nv == 0 {
        let label = examples[0].in_spp;
        if examples.iter().any(|ex| ex.in_spp != label) {
            return Err(ConflictError);
        }
        return Ok(SPP::new(label as u32));
    }

    // Step 1: build the decision-path trie (node 0 is the root).  Each example's
    // path is `2*input[i] + output[i]` at field `i`; this computes every prefix
    // partition in one pass.  A leaf reached with two labels is a conflict.
    let mut trie: Vec<TrieNode> = vec![TrieNode::Branch([None; 4])];
    for ex in &examples {
        assert_eq!(ex.input.len(), nv, "example input has wrong width");
        assert_eq!(ex.output.len(), nv, "example output has wrong width");
        let mut cur = 0usize;
        for (d, (&inp, &outp)) in ex.input.iter().zip(&ex.output).enumerate() {
            let b = 2 * inp as usize + outp as usize;
            let is_last = d + 1 == nv;
            let next = match trie[cur] {
                TrieNode::Branch(children) => children[b],
                TrieNode::Leaf(_) => unreachable!("path longer than trie depth"),
            };
            match next {
                Some(nx) => {
                    if is_last
                        && let TrieNode::Leaf(prev) = trie[nx as usize]
                        && prev != ex.in_spp
                    {
                        return Err(ConflictError);
                    }
                    cur = nx as usize;
                }
                None => {
                    let id = trie.len() as u32;
                    trie.push(if is_last {
                        TrieNode::Leaf(ex.in_spp)
                    } else {
                        TrieNode::Branch([None; 4])
                    });
                    // Re-borrow after the push (which may have reallocated).
                    if let TrieNode::Branch(children) = &mut trie[cur] {
                        children[b] = Some(id);
                    }
                    cur = id as usize;
                }
            }
        }
    }

    // Step 2: RPNI-style merging of compatible prefix cells, level by level.
    let levels = merge_levels(&mut trie, nv);

    // Step 3: fold the merged levels bottom-up into an SPP.
    Ok(build_spp(&trie, &levels, spp_store))
}

/// A node of the decision-path trie: an internal 4-way branch (indexed by
/// `2*input + output`) or an accept/reject leaf.
#[derive(Clone, Copy)]
enum TrieNode {
    Branch([Option<u32>; 4]),
    Leaf(bool),
}

/// Merge compatible trie nodes level by level, top-down (RPNI-style).
///
/// The nodes at depth `d` are the prefix cells of the examples; each level is
/// swept in order, greedily merging every node into the first [`compatible`]
/// class found at that level (unioning the subtrees and redirecting the parent
/// edge), or starting a new class.  Pooled evidence at one level narrows the
/// don't-cares at every level below.  The result is a layered DAG; returns the
/// surviving class nodes per level, root (depth 0) first.
fn merge_levels(trie: &mut [TrieNode], nv: usize) -> Vec<Vec<u32>> {
    let mut levels: Vec<Vec<u32>> = Vec::with_capacity(nv + 1);
    levels.push(vec![0]);
    for d in 0..nv {
        // Gather the populated (parent, branch, child) slots feeding depth d+1.
        let mut slots: Vec<(u32, usize, u32)> = Vec::new();
        for &p in &levels[d] {
            if let TrieNode::Branch(children) = trie[p as usize] {
                for (b, child) in children.iter().enumerate() {
                    if let Some(n) = child {
                        slots.push((p, b, *n));
                    }
                }
            }
        }
        let mut classes: Vec<u32> = Vec::new();
        for (p, b, n) in slots {
            match classes
                .iter()
                .copied()
                .find(|&c| compatible(trie, c as usize, n as usize))
            {
                Some(c) => {
                    merge(trie, c as usize, n as usize);
                    if let TrieNode::Branch(children) = &mut trie[p as usize] {
                        children[b] = Some(c);
                    }
                }
                None => classes.push(n),
            }
        }
        levels.push(classes);
    }
    levels
}

/// Do two same-depth subtrees agree on every decision path they both exercise?
fn compatible(trie: &[TrieNode], a: usize, b: usize) -> bool {
    match (trie[a], trie[b]) {
        (TrieNode::Leaf(la), TrieNode::Leaf(lb)) => la == lb,
        (TrieNode::Branch(ca), TrieNode::Branch(cb)) => {
            ca.iter().zip(&cb).all(|(x, y)| match (x, y) {
                (Some(x), Some(y)) => compatible(trie, *x as usize, *y as usize),
                _ => true,
            })
        }
        _ => unreachable!("compatibility check across depths"),
    }
}

/// Union subtree `b` into `a` (same depth, already checked [`compatible`]).
/// `b` becomes unreachable afterwards.
fn merge(trie: &mut [TrieNode], a: usize, b: usize) {
    let (ca, cb) = match (trie[a], trie[b]) {
        (TrieNode::Branch(ca), TrieNode::Branch(cb)) => (ca, cb),
        _ => return, // leaves: labels already known to agree
    };
    let mut merged = ca;
    for (slot, y) in merged.iter_mut().zip(cb) {
        match (*slot, y) {
            (Some(x), Some(y)) => merge(trie, x as usize, y as usize),
            (None, Some(y)) => *slot = Some(y),
            _ => {}
        }
    }
    trie[a] = TrieNode::Branch(merged);
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
    for (i,level) in levels.iter().enumerate().rev() {
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
}
