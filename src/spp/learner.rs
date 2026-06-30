//! One-shot passive learning of an [`SPP`] from concrete examples.
//!
//! [`learn_spp`] takes a set of [`Example`]s — concrete `(input, output)` packet
//! pairs, each tagged in (`in_spp = true`) or out (`false`) of the SPP — and
//! returns an [`SPP`] consistent with all of them, or [`ConflictError`] if two
//! examples give the same pair opposite labels.
//!
//! # Method (RPNI-style eager merging)
//!
//! An [`SPP`] over `n` fields is an order-`n` decision diagram: level `i` is a
//! 4-way branch on the `(input[i], output[i])` bit pair, terminating in
//! accept/reject.  We build that diagram **bottom-up**, one field level at a
//! time:
//!
//! * Each example becomes a length-`n` *decision path* — `2*input[i] + output[i]`
//!   at level `i` — ending on its accept (`1`) or reject (`0`) terminal.
//! * At each level, examples are grouped by their shared prefix (everything that
//!   reaches the same node), and each group's four branches are populated from
//!   the examples that exercise them.  Branches no example exercises
//!   (don't-cares) take a sibling branch — the eager generalization.
//! * [`SPPstore::mk`] hash-conses, so nodes with identical suffix behaviour
//!   collapse into one — the state merge.
//!
//! An example's own path is never redirected by a fill, so the result accepts
//! every positive example and rejects every negative one; the generalization
//! only touches inputs no example pins down.

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
    spp_store: &mut SPPstore,
) -> Result<SPP, ConflictError> {
    let nv = spp_store.num_vars() as usize;
    let examples: Vec<Example> = examples.into_iter().collect();

    // Each example → a length-`nv` decision path (`2*input + output` per field).
    let decisions: Vec<Vec<usize>> = examples
        .iter()
        .map(|ex| {
            assert_eq!(ex.input.len(), nv, "example input has wrong width");
            assert_eq!(ex.output.len(), nv, "example output has wrong width");
            (0..nv)
                .map(|i| 2 * ex.input[i] as usize + ex.output[i] as usize)
                .collect()
        })
        .collect();

    // The full path identifies the concrete pair, so a repeated path with
    // opposite labels is a genuine conflict.
    let mut seen: HashMap<&[usize], bool> = HashMap::new();
    for (path, ex) in decisions.iter().zip(&examples) {
        match seen.insert(path.as_slice(), ex.in_spp) {
            Some(prev) if prev != ex.in_spp => return Err(ConflictError),
            _ => {}
        }
    }

    // No evidence anywhere → reject everything.
    if examples.is_empty() {
        return Ok(spp_store.zero);
    }

    // Bottom: each example sits on its accept/reject terminal.
    let mut layer: HashMap<usize, SPP> = (0..examples.len())
        .map(|e| (e, SPP::new(examples[e].in_spp as u32)))
        .collect();

    // Build the field levels bottom-up; hash-consing merges equivalent nodes.
    for depth in (0..nv).rev() {
        layer = build_field_layer(depth, &decisions, &layer, spp_store);
    }

    // At depth 0 every example shares the empty prefix, so they all land on the
    // single root node.
    Ok(layer[&0])
}

/// Build one field level of the diagram, bottom-up.
///
/// `below` maps each example to the node it reaches just below this level.
/// Examples are grouped by their length-`depth` prefix; each group's four
/// `(input, output)` branches are filled from the examples that exercise them,
/// with don't-cares taking a sibling branch.  [`SPPstore::mk`] hash-conses, so
/// identical nodes merge.  Returns the map from each example to its node here.
fn build_field_layer(
    depth: usize,
    decisions: &[Vec<usize>],
    below: &HashMap<usize, SPP>,
    store: &mut SPPstore,
) -> HashMap<usize, SPP> {
    // Group examples by the prefix that brought them to this level.
    let mut groups: HashMap<&[usize], Vec<usize>> = HashMap::new();
    for &e in below.keys() {
        groups.entry(&decisions[e][..depth]).or_default().push(e);
    }

    let mut result = HashMap::new();
    for (_prefix, members) in groups {
        let mut row: [Option<SPP>; 4] = [None; 4];
        for &e in &members {
            row[decisions[e][depth]] = Some(below[&e]);
        }
        // Every group has ≥1 member, hence ≥1 defined branch to fill with.
        let fill = row
            .iter()
            .flatten()
            .next()
            .copied()
            .expect("non-empty group");
        let node = store.mk(
            row[0].unwrap_or(fill),
            row[1].unwrap_or(fill),
            row[2].unwrap_or(fill),
            row[3].unwrap_or(fill),
        );
        for e in members {
            result.insert(e, node);
        }
    }
    result
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
