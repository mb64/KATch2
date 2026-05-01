//! Passive learning of SPPs from abstract examples.
//!
//! # Overview
//!
//! An [`AbstractPacket`] is a packet where each field is either `Some(bit)` or
//! `None` (wildcard).  An abstract example is a pair of abstract packets plus a
//! polarity: *positive* means the pair is in the SPP, *negative* means it is not.
//!
//! **Correlated wildcards**: when the same field is `None` in *both* packets of a
//! pair, it expands only to `(0, 0)` and `(1, 1)` — the input and output bits
//! take the same value.  If only one side is `None`, the two bits vary
//! independently.
//!
//! [`Learner`] accumulates abstract examples in a compact, hash-consed evidence
//! DAG and can extract an [`SPP`] consistent with all of them:
//!
//! ```ignore
//! let mut learner = Learner::new(num_vars);
//! learner.add_example(ap1, ap2, true)?;   // positive example
//! learner.add_example(ap3, ap4, false)?;  // negative example
//! let spp = learner.extract(&mut spp_store);
//! ```
//!
//! # Guarantees
//!
//! * The extracted SPP **accepts** every concrete packet pair covered by a
//!   positive example.
//! * The extracted SPP **rejects** every concrete packet pair covered by a
//!   negative example.
//! * [`add_example`](Learner::add_example) returns [`ConflictError`] if the new
//!   example contradicts an existing one (same concrete pair with opposite
//!   polarity).

use super::{SPP, SPPstore, Var};
use std::collections::HashMap;

/// An abstract packet: each field is `Some(bit)` or `None` (wildcard meaning either 0 or 1).
pub type AbstractPacket = Vec<Option<bool>>;

/// Error returned when two examples conflict (same packet pair given both positive and negative polarity).
#[derive(Debug, Clone)]
pub struct ConflictError;

impl std::fmt::Display for ConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "conflicting examples")
    }
}

/// Reference into the evidence DAG arena.
///
/// - `EvidRef(0)` = UNSET: no evidence anywhere in this subtree
/// - `EvidRef(1)` = POS:   positive leaf (only valid at depth == num_vars)
/// - `EvidRef(2)` = NEG:   negative leaf (only valid at depth == num_vars)
/// - `EvidRef(n >= 3)` = internal node at index `n - 3`
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct EvidRef(u32);

const UNSET: EvidRef = EvidRef(0);
const POS: EvidRef = EvidRef(1);
const NEG: EvidRef = EvidRef(2);

/// An internal evidence DAG node.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct EvidNode {
    x00: EvidRef,
    x01: EvidRef,
    x10: EvidRef,
    x11: EvidRef,
}

/// Arena + hash-consing store for the evidence DAG.
struct EvidStore {
    num_vars: Var,
    nodes: Vec<EvidNode>,
    hc: HashMap<EvidNode, EvidRef>,
}

impl EvidStore {
    fn new(num_vars: Var) -> Self {
        Self {
            num_vars,
            nodes: Vec::new(),
            hc: HashMap::new(),
        }
    }

    fn get(&self, r: EvidRef) -> EvidNode {
        debug_assert!(r.0 >= 3, "cannot get base EvidRef");
        self.nodes[(r.0 - 3) as usize]
    }

    fn mk(&mut self, x00: EvidRef, x01: EvidRef, x10: EvidRef, x11: EvidRef) -> EvidRef {
        let node = EvidNode { x00, x01, x10, x11 };
        if let Some(&r) = self.hc.get(&node) {
            return r;
        }
        let r = EvidRef(self.nodes.len() as u32 + 3);
        self.nodes.push(node);
        self.hc.insert(node, r);
        r
    }

    /// Public-facing entry point: update the subtree rooted at `node` with one
    /// abstract example, returning the new root.
    fn add(
        &mut self,
        node: EvidRef,
        ap1: &AbstractPacket,
        ap2: &AbstractPacket,
        polarity: bool,
    ) -> Result<EvidRef, ConflictError> {
        let mut memo = HashMap::new();
        self.add_rec(node, 0, ap1, ap2, polarity, &mut memo)
    }

    /// Recursive helper for `add`.  `memo` caches `(node, depth) -> new_node` so
    /// that shared DAG nodes are updated only once regardless of how many wildcard
    /// branches reach them, keeping work proportional to the DAG size.
    fn add_rec(
        &mut self,
        node: EvidRef,
        depth: Var,
        ap1: &AbstractPacket,
        ap2: &AbstractPacket,
        polarity: bool,
        memo: &mut HashMap<(EvidRef, Var), EvidRef>,
    ) -> Result<EvidRef, ConflictError> {
        if depth == self.num_vars {
            // Leaf level: check/set polarity. Base cases are cheap; no memo needed.
            return match node {
                UNSET => Ok(if polarity { POS } else { NEG }),
                POS => {
                    if polarity {
                        Ok(POS)
                    } else {
                        Err(ConflictError)
                    }
                }
                NEG => {
                    if !polarity {
                        Ok(NEG)
                    } else {
                        Err(ConflictError)
                    }
                }
                _ => unreachable!("internal node at leaf depth"),
            };
        }

        if let Some(&cached) = memo.get(&(node, depth)) {
            return Ok(cached);
        }

        let (mut c00, mut c01, mut c10, mut c11) = if node == UNSET {
            (UNSET, UNSET, UNSET, UNSET)
        } else {
            let n = self.get(node);
            (n.x00, n.x01, n.x10, n.x11)
        };

        let b1 = ap1[depth as usize];
        let b2 = ap2[depth as usize];

        let pairs: &[(bool, bool)] = match (b1, b2) {
            (Some(false), Some(false)) => &[(false, false)],
            (Some(false), Some(true)) => &[(false, true)],
            (Some(true), Some(false)) => &[(true, false)],
            (Some(true), Some(true)) => &[(true, true)],
            (Some(false), None) => &[(false, false), (false, true)],
            (Some(true), None) => &[(true, false), (true, true)],
            (None, Some(false)) => &[(false, false), (true, false)],
            (None, Some(true)) => &[(false, true), (true, true)],
            (None, None) => &[(false, false), (true, true)],
        };

        for &(v1, v2) in pairs {
            let child = match (v1, v2) {
                (false, false) => c00,
                (false, true) => c01,
                (true, false) => c10,
                (true, true) => c11,
            };
            let new_child = self.add_rec(child, depth + 1, ap1, ap2, polarity, memo)?;
            match (v1, v2) {
                (false, false) => c00 = new_child,
                (false, true) => c01 = new_child,
                (true, false) => c10 = new_child,
                (true, true) => c11 = new_child,
            }
        }

        let result = self.mk(c00, c01, c10, c11);
        memo.insert((node, depth), result);
        Ok(result)
    }

    /// Public-facing wrapper: extract an SPP consistent with all evidence under `root`.
    fn extract_spp(&self, root: EvidRef, spp_store: &mut SPPstore) -> SPP {
        let map = self.extract_layer(&[root], 0, spp_store);
        map[&root]
    }

    /// Bottom-up, layer-by-layer extractor.
    ///
    /// Takes a **deduplicated** slice of `EvidRef`s all at the same `depth` and
    /// returns a map from each input node to its extracted `SPP`.
    ///
    /// The four steps per layer:
    /// 1. Collect all unique children and recurse — many will map to the *same* SPP,
    ///    naturally collapsing shared DAG structure.
    /// 2. Build a *partial* SPP node for each evidence node: UNSET children become
    ///    `None` (free), evidence children become `Some(spp)` (constrained).
    /// 3. Greedily merge compatible partial nodes — two are compatible when every
    ///    edge where *both* are constrained carries the same SPP.  Merged groups
    ///    share one SPP node, biasing the result towards simpler SPPs.
    /// 4. Fill remaining free edges with the depth-appropriate zero SPP, then call
    ///    `spp_store.mk` once per group and record the result for every node in it.
    fn extract_layer(
        &self,
        nodes: &[EvidRef],
        depth: Var,
        spp_store: &mut SPPstore,
    ) -> HashMap<EvidRef, SPP> {
        // Base case: leaf level.
        if depth == self.num_vars {
            return nodes
                .iter()
                .map(|&node| {
                    let spp = if node == POS {
                        SPP::new(1)
                    } else {
                        SPP::new(0)
                    };
                    (node, spp)
                })
                .collect();
        }

        // Step 1: collect unique children.
        // Always include UNSET so child_map always contains the depth-appropriate
        // zero SPP, which we use as the fill default for free edges.
        let mut children_set: HashMap<EvidRef, ()> = HashMap::new();
        children_set.insert(UNSET, ());
        for &node in nodes {
            let (c00, c01, c10, c11) = if node == UNSET {
                (UNSET, UNSET, UNSET, UNSET)
            } else {
                let n = self.get(node);
                (n.x00, n.x01, n.x10, n.x11)
            };
            children_set.insert(c00, ());
            children_set.insert(c01, ());
            children_set.insert(c10, ());
            children_set.insert(c11, ());
        }
        let children: Vec<EvidRef> = children_set.into_keys().collect();

        // Step 2: recurse on children layer (all unique, so no duplicated work).
        let child_spp = self.extract_layer(&children, depth + 1, spp_store);
        let default_spp = child_spp[&UNSET]; // zero SPP at depth+1

        // Step 3: build partial SPP nodes.
        // UNSET children → None (free to fill); evidence children → Some(spp).
        let partial: Vec<(EvidRef, [Option<SPP>; 4])> = nodes
            .iter()
            .map(|&node| {
                let (c00, c01, c10, c11) = if node == UNSET {
                    (UNSET, UNSET, UNSET, UNSET)
                } else {
                    let n = self.get(node);
                    (n.x00, n.x01, n.x10, n.x11)
                };
                let f = |c: EvidRef| {
                    if c == UNSET {
                        None
                    } else {
                        Some(child_spp[&c])
                    }
                };
                (node, [f(c00), f(c01), f(c10), f(c11)])
            })
            .collect();

        // Step 4: greedy merge.
        // Each group: (merged partial node, list of EvidRefs that belong to it).
        let mut groups: Vec<([Option<SPP>; 4], Vec<EvidRef>)> = Vec::new();

        'outer: for &(node, p) in &partial {
            for (gp, gv) in groups.iter_mut() {
                let compat = (0..4usize).all(|i| match (gp[i], p[i]) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                });
                if compat {
                    for i in 0..4usize {
                        if gp[i].is_none() {
                            gp[i] = p[i];
                        }
                    }
                    gv.push(node);
                    continue 'outer;
                }
            }
            groups.push((p, vec![node]));
        }

        // Step 5: complete each group to a full SPP node and record results.
        let mut result = HashMap::new();
        for (gp, gv) in groups {
            let s = |opt: Option<SPP>| opt.unwrap_or(default_spp);
            let spp = spp_store.mk(s(gp[0]), s(gp[1]), s(gp[2]), s(gp[3]));
            for node in gv {
                result.insert(node, spp);
            }
        }
        result
    }
}

/// Passive learner for SPPs.
///
/// Stores abstract examples in a hash-consed evidence DAG, then extracts
/// a consistent SPP on demand.
pub struct Learner {
    evid: EvidStore,
    root: EvidRef,
}

impl Learner {
    /// Creates a new learner for packets with `num_vars` fields.
    pub fn new(num_vars: Var) -> Self {
        Self {
            evid: EvidStore::new(num_vars),
            root: UNSET,
        }
    }

    /// Adds an abstract example.
    ///
    /// `polarity = true`  → the pair `(ap1, ap2)` is in the SPP.
    /// `polarity = false` → the pair `(ap1, ap2)` is not in the SPP.
    ///
    /// Returns `ConflictError` if this contradicts a previously added example.
    pub fn add_example(
        &mut self,
        ap1: AbstractPacket,
        ap2: AbstractPacket,
        polarity: bool,
    ) -> Result<(), ConflictError> {
        self.root = self.evid.add(self.root, &ap1, &ap2, polarity)?;
        Ok(())
    }

    /// Extracts an SPP consistent with all stored examples.
    ///
    /// The returned SPP accepts every positive example and rejects every negative example.
    pub fn extract(&self, spp_store: &mut SPPstore) -> SPP {
        self.evid.extract_spp(self.root, spp_store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check whether a concrete packet pair (pk1, pk2) is accepted by an SPP.
    /// Traverses the BDD, following the (input_bit, output_bit) path.
    fn spp_accepts(spp_store: &SPPstore, spp: SPP, pk1: &[bool], pk2: &[bool]) -> bool {
        let mut cur = spp;
        for (&b1, &b2) in pk1.iter().zip(pk2.iter()) {
            if cur == SPP::new(0) {
                return false;
            }
            if cur == SPP::new(1) {
                return true;
            }
            let node = spp_store.get(cur);
            cur = match (b1, b2) {
                (false, false) => node.x00,
                (false, true) => node.x01,
                (true, false) => node.x10,
                (true, true) => node.x11,
            };
        }
        cur == SPP::new(1)
    }

    /// Expand a pair of abstract packets into all concrete pairs they represent,
    /// respecting correlated-wildcard semantics: when both fields are `None` at
    /// the same position, they expand only to `(false, false)` and `(true, true)`.
    fn expand_pair(ap1: &AbstractPacket, ap2: &AbstractPacket) -> Vec<(Vec<bool>, Vec<bool>)> {
        let mut result: Vec<(Vec<bool>, Vec<bool>)> = vec![(vec![], vec![])];
        for (&b1, &b2) in ap1.iter().zip(ap2.iter()) {
            let pairs: Vec<(bool, bool)> = match (b1, b2) {
                (Some(v1), Some(v2)) => vec![(v1, v2)],
                (Some(v1), None) => vec![(v1, false), (v1, true)],
                (None, Some(v2)) => vec![(false, v2), (true, v2)],
                (None, None) => vec![(false, false), (true, true)],
            };
            result = result
                .into_iter()
                .flat_map(|(p1, p2)| {
                    pairs.iter().map(move |&(v1, v2)| {
                        let mut np1 = p1.clone();
                        let mut np2 = p2.clone();
                        np1.push(v1);
                        np2.push(v2);
                        (np1, np2)
                    })
                })
                .collect();
        }
        result
    }

    const N: Var = 2;

    #[test]
    fn test_empty_learner() {
        let mut store = SPPstore::new(N);
        let learner = Learner::new(N);
        // Should not panic; result is a valid SPP (equal to store.zero since no positive evidence)
        let spp = learner.extract(&mut store);
        assert_eq!(spp, store.zero);
    }

    #[test]
    fn test_single_positive_example() {
        let mut store = SPPstore::new(N);
        let mut learner = Learner::new(N);
        let pk1 = vec![Some(false), Some(true)];
        let pk2 = vec![Some(true), Some(false)];
        learner.add_example(pk1.clone(), pk2.clone(), true).unwrap();
        let spp = learner.extract(&mut store);
        // The extracted SPP must accept the positive example
        let c1 = vec![false, true];
        let c2 = vec![true, false];
        assert!(spp_accepts(&store, spp, &c1, &c2));
    }

    #[test]
    fn test_single_negative_example() {
        let mut store = SPPstore::new(N);
        let mut learner = Learner::new(N);
        let pk1 = vec![Some(false), Some(false)];
        let pk2 = vec![Some(false), Some(false)];
        learner
            .add_example(pk1.clone(), pk2.clone(), false)
            .unwrap();
        let spp = learner.extract(&mut store);
        // The extracted SPP must reject the negative example
        let c1 = vec![false, false];
        let c2 = vec![false, false];
        assert!(!spp_accepts(&store, spp, &c1, &c2));
    }

    #[test]
    fn test_wildcard_positive_example() {
        let mut store = SPPstore::new(N);
        let mut learner = Learner::new(N);
        // (*, 0) -> (1, *) : any (b, 0) input maps to (1, b') output for any b'
        let ap1 = vec![None, Some(false)];
        let ap2 = vec![Some(true), None];
        learner.add_example(ap1.clone(), ap2.clone(), true).unwrap();
        let spp = learner.extract(&mut store);
        // All concrete instances must be accepted
        for (c1, c2) in expand_pair(&ap1, &ap2) {
            assert!(
                spp_accepts(&store, spp, &c1, &c2),
                "SPP should accept ({:?}, {:?})",
                c1,
                c2
            );
        }
    }

    #[test]
    fn test_wildcard_negative_example() {
        let mut store = SPPstore::new(N);
        let mut learner = Learner::new(N);
        // All pairs where both packets are (1, 1) are negative
        let ap1 = vec![Some(true), Some(true)];
        let ap2 = vec![Some(true), Some(true)];
        learner
            .add_example(ap1.clone(), ap2.clone(), false)
            .unwrap();
        let spp = learner.extract(&mut store);
        assert!(!spp_accepts(&store, spp, &[true, true], &[true, true]));
    }

    #[test]
    fn test_conflict_error() {
        let mut learner = Learner::new(N);
        let pk1 = vec![Some(false), Some(false)];
        let pk2 = vec![Some(false), Some(false)];
        learner.add_example(pk1.clone(), pk2.clone(), true).unwrap();
        let result = learner.add_example(pk1.clone(), pk2.clone(), false);
        assert!(result.is_err(), "should return ConflictError");
    }

    #[test]
    fn test_multiple_examples_consistent() {
        let mut store = SPPstore::new(N);
        let mut learner = Learner::new(N);
        let positives: &[(&[Option<bool>], &[Option<bool>])] = &[
            (&[Some(false), Some(false)], &[Some(false), Some(false)]),
            (&[Some(true), Some(false)], &[Some(true), Some(false)]),
        ];
        let negatives: &[(&[Option<bool>], &[Option<bool>])] = &[
            (&[Some(false), Some(true)], &[Some(true), Some(false)]),
            (&[Some(true), Some(true)], &[Some(false), Some(true)]),
        ];
        for &(ap1, ap2) in positives {
            learner
                .add_example(ap1.to_vec(), ap2.to_vec(), true)
                .unwrap();
        }
        for &(ap1, ap2) in negatives {
            learner
                .add_example(ap1.to_vec(), ap2.to_vec(), false)
                .unwrap();
        }
        let spp = learner.extract(&mut store);
        for &(ap1, ap2) in positives {
            for (c1, c2) in expand_pair(&ap1.to_vec(), &ap2.to_vec()) {
                assert!(
                    spp_accepts(&store, spp, &c1, &c2),
                    "positive example ({:?},{:?}) should be accepted",
                    c1,
                    c2
                );
            }
        }
        for &(ap1, ap2) in negatives {
            for (c1, c2) in expand_pair(&ap1.to_vec(), &ap2.to_vec()) {
                assert!(
                    !spp_accepts(&store, spp, &c1, &c2),
                    "negative example ({:?},{:?}) should be rejected",
                    c1,
                    c2
                );
            }
        }
    }

    #[test]
    fn test_random() {
        const N_RAND: Var = 8;

        for _ in 0..100 {
            let mut store = SPPstore::new(N_RAND);
            let mut learner = Learner::new(N_RAND);
            let mut positives: Vec<(AbstractPacket, AbstractPacket)> = Vec::new();
            let mut negatives: Vec<(AbstractPacket, AbstractPacket)> = Vec::new();

            // Try to collect 100 non-conflicting examples; skip any that conflict.
            let mut added = 0;
            let mut attempts = 0;
            while added < 100 && attempts < 10_000 {
                attempts += 1;
                let ap1: AbstractPacket = (0..N_RAND)
                    .map(|_| match rand::random_range(0u32..3) {
                        0 => Some(false),
                        1 => Some(true),
                        _ => None,
                    })
                    .collect();
                let ap2: AbstractPacket = (0..N_RAND)
                    .map(|_| match rand::random_range(0u32..3) {
                        0 => Some(false),
                        1 => Some(true),
                        _ => None,
                    })
                    .collect();
                let polarity = rand::random::<bool>();
                if learner
                    .add_example(ap1.clone(), ap2.clone(), polarity)
                    .is_ok()
                {
                    if polarity {
                        positives.push((ap1, ap2));
                    } else {
                        negatives.push((ap1, ap2));
                    }
                    added += 1;
                }
            }

            let spp = learner.extract(&mut store);

            for (ap1, ap2) in &positives {
                for (c1, c2) in expand_pair(ap1, ap2) {
                    assert!(
                        spp_accepts(&store, spp, &c1, &c2),
                        "positive ({:?},{:?}) rejected",
                        c1,
                        c2
                    );
                }
            }
            for (ap1, ap2) in &negatives {
                for (c1, c2) in expand_pair(ap1, ap2) {
                    assert!(
                        !spp_accepts(&store, spp, &c1, &c2),
                        "negative ({:?},{:?}) accepted",
                        c1,
                        c2
                    );
                }
            }
        }
    }
}
