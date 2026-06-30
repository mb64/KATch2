//! One-shot passive learning of an [`SPP`] from concrete examples.
//!
//! [`learn_spp`] takes a set of [`Example`]s — concrete `(input, output)` packet
//! pairs, each tagged in (`in_spp = true`) or out (`false`) of the SPP — and
//! returns an [`SPP`] consistent with all of them, or [`ConflictError`] if two
//! examples give the same pair opposite labels.
//!
//! # Method (trie + RPNI-style eager merging)
//!
//! An [`SPP`] over `n` fields is an order-`n` decision diagram: level `i` is a
//! 4-way branch on the `(input[i], output[i])` bit pair, terminating in
//! accept/reject.  We build it in two steps:
//!
//! 1. **Trie.** Insert every example's length-`n` *decision path* —
//!    `2*input[i] + output[i]` at field `i` — into a 4-ary trie, leaves carrying
//!    the accept/reject label.  This computes *all* the prefix partitions in one
//!    pass: a node at depth `d` is exactly the set of examples sharing that
//!    length-`d` prefix, and a shared prefix is walked only once.  A leaf reached
//!    with two different labels is a [`ConflictError`].
//! 2. **Bottom-up fold.** Post-order over the trie: a leaf becomes its accept
//!    (`1`) / reject (`0`) terminal; a branch becomes [`SPPstore::mk`] of its
//!    four children, with don't-care branches (those no example exercises) taking
//!    a sibling — the eager generalization.  Hash-consing in `mk` collapses nodes
//!    with identical suffix behaviour — the state merge.
//!
//! Building the trie first means each partition is computed once instead of being
//! re-derived at every level.  An example's own path is never redirected by a
//! fill, so the result accepts every positive example and rejects every negative
//! one; the generalization only touches inputs no example pins down.

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

    // Step 2: fold the trie bottom-up into an SPP.
    Ok(build_spp(&trie, 0, spp_store))
}

/// A node of the decision-path trie: an internal 4-way branch (indexed by
/// `2*input + output`) or an accept/reject leaf.
#[derive(Clone, Copy)]
enum TrieNode {
    Branch([Option<u32>; 4]),
    Leaf(bool),
}

/// Fold the trie rooted at `node` into an [`SPP`], bottom-up.
///
/// A leaf becomes its accept/reject terminal.  A branch becomes [`SPPstore::mk`]
/// of its four children, with don't-care branches (those no example exercises)
/// taking a sibling — the eager generalization.  Each trie node has a single
/// parent, so this visits every node once; `mk` hash-conses, merging nodes with
/// identical suffix behaviour.
fn build_spp(trie: &[TrieNode], node: usize, store: &mut SPPstore) -> SPP {
    let children = match trie[node] {
        TrieNode::Leaf(label) => return SPP::new(label as u32),
        TrieNode::Branch(children) => children,
    };
    let mut spps: [Option<SPP>; 4] = [None; 4];
    for (b, &child) in children.iter().enumerate() {
        if let Some(id) = child {
            spps[b] = Some(build_spp(trie, id as usize, store));
        }
    }
    // Every branch has ≥1 child to fill the don't-cares with.
    let fill = spps
        .iter()
        .flatten()
        .next()
        .copied()
        .expect("trie branch has at least one child");
    store.mk(
        spps[0].unwrap_or(fill),
        spps[1].unwrap_or(fill),
        spps[2].unwrap_or(fill),
        spps[3].unwrap_or(fill),
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

    /// One positive example and nothing else: eager merging fills every
    /// don't-care with the accepting branch, generalizing to "accept all".
    #[test]
    fn single_positive_generalizes_to_top() {
        let mut store = SPPstore::new(N);
        let spp = learn_spp([ex(&[false, false], &[false, false], true)], &mut store).unwrap();
        assert_eq!(spp, store.top);
    }

    /// Symmetrically, a single negative example generalizes to "reject all".
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
