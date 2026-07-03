//! # Solution candidates
//!
//! A solution candidate [`Cand`] for a DFA `aut` is a predicate on:
//!
//! * an input packet;
//! * a "first" packet;
//! * a list of N states from `aut`;
//! * a "last" packet; and
//! * an output packet.
//!
//! It's essentially an OBDD, like an SP or SPP, but the implementation is slightly different.

use crate::holes::aut::{DFA, ENFA, ExplicitDFA, NFA, ops};
use crate::spp;

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Cand<'a> {
    nodes: Vec<CandNode>,
    head: CandIdx,
    /// The DFA this candidate is built over.
    ///
    /// **Invariant:** `dfa` must be *complete* — its transition function must be
    /// total (every state has, for every `(in, out)` packet pair, an outgoing
    /// edge, e.g. via a dedicated sink state).  Stepping the [`ops::Exponential`]
    /// product assumes a component never runs out of transitions; an incomplete
    /// DFA would let a component on a dead state stall the whole product.  Wrap
    /// incomplete DFAs in [`ops::WithSinkState`] (then materialize) before use.
    dfa: &'a ExplicitDFA,
}

type CandIdx = u32;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum CandNode {
    /// Examine the next pair of fields in the input packet and "first" packet.
    NextField {
        b00: CandIdx,
        b01: CandIdx,
        b10: CandIdx,
        b11: CandIdx,
    },
    /// Examine the next state
    NextState(Vec<CandIdx>),
    /// A predicate on the "last" packet and the output packet.
    Root(spp::SPP),
}

pub struct Input {
    pub pkt_in: Vec<bool>,
    pub pkt_start: Vec<bool>,
    pub states: Vec<usize>,
    pub pkt_end: Vec<bool>,
    pub pkt_out: Vec<bool>,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum State {
    Start,
    Middle(CandIdx, Vec<usize>),
}

impl<'a> ENFA for Cand<'a> {
    type State = State;

    fn start(&self, _store: &mut spp::SPPstore) -> State {
        State::Start
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &State) -> bool {
        true
    }

    fn transitions(&self, store: &mut spp::SPPstore, q: &State) -> Vec<(spp::SPP, Self::State)> {
        match *q {
            State::Start => {
                // First, collect all the nodes at the appropriate level
                let mut nodes = vec![self.head];
                for _ in 0..store.num_vars() as usize {
                    let new_nodes = nodes
                        .into_iter()
                        .flat_map(|n| {
                            let CandNode::NextField { b00, b01, b10, b11 } = *self.get_node(n)
                            else {
                                unreachable!("malformed")
                            };
                            [b00, b01, b10, b11].into_iter()
                        })
                        .collect();
                    nodes = new_nodes;
                    nodes.sort();
                    nodes.dedup();
                }

                // Each field-bottom node is entered via the SPP routing
                // `(pkt_in, pkt_start)` to it, landing in `Middle` with the
                // identity state vector (one token per DFA state).
                let init: Vec<usize> = (0..self.dfa.num_states()).collect();
                nodes
                    .into_iter()
                    .map(|n| (self.spp_to(store, n), State::Middle(n, init.clone())))
                    .collect()
            }
            State::Middle(node, ref states) => {
                debug_assert_eq!(states.len(), self.dfa.num_states());
                // Step the product over `self.dfa` directly.  This relies on
                // `self.dfa` being *complete* (total transition function, e.g.
                // via a sink state): otherwise a component sitting on a dead
                // state would have no outgoing edge and stall the whole product
                // rather than flowing into a sink.  See [`Cand`]'s contract.
                let exp = ops::Exponential { inner: self.dfa };
                exp.transitions(store, states)
                    .into_iter()
                    .map(|(spp, target)| (spp, State::Middle(node, target)))
                    .collect()
            }
        }
    }

    fn output(&self, store: &mut spp::SPPstore, q: &State) -> spp::SPP {
        let State::Middle(mut node, ref states) = *q else {
            return store.zero;
        };

        assert_eq!(states.len(), self.dfa.num_states());
        for &state in states {
            let CandNode::NextState(children) = self.get_node(node) else {
                unreachable!("malformed")
            };
            node = children[state];
        }

        let CandNode::Root(spp) = *self.get_node(node) else {
            unreachable!("malformed")
        };

        spp
    }
}

// A `Cand`'s transitions are disjoint by construction: the `Start` transitions
// partition `(pkt_in, pkt_start)` (each pair routes to exactly one field-bottom
// node), and the `Exponential` of a DFA keeps per-component transitions
// disjoint.  So a `Cand` is a deterministic automaton.
impl<'a> NFA for Cand<'a> {}
impl<'a> DFA for Cand<'a> {}

/// Error returned when two examples conflict: the same concrete [`Input`] is
/// supplied with both `true` and `false`.
#[derive(Debug, Clone)]
pub struct ConflictError;

impl std::fmt::Display for ConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "conflicting examples")
    }
}

impl<'a> Cand<'a> {
    /// The SPP relating `(pkt_in, pkt_start)` pairs that route from [`Self::head`]
    /// to `target` through the field region.
    ///
    /// The field region (the `NextField` nodes) is shaped exactly like an SPP —
    /// each level is a 4-way branch on a `(pkt_in[i], pkt_start[i])` bit pair —
    /// so this just rebuilds that sub-diagram as an [`spp::SPP`], with `target`
    /// as the sole accepting field-bottom node.
    fn spp_to(&self, store: &mut spp::SPPstore, target: CandIdx) -> spp::SPP {
        let mut memo: HashMap<CandIdx, spp::SPP> = HashMap::new();
        self.spp_to_rec(store, self.head, target, &mut memo)
    }

    fn spp_to_rec(
        &self,
        store: &mut spp::SPPstore,
        node: CandIdx,
        target: CandIdx,
        memo: &mut HashMap<CandIdx, spp::SPP>,
    ) -> spp::SPP {
        if let Some(&spp) = memo.get(&node) {
            return spp;
        }
        let spp = match *self.get_node(node) {
            CandNode::NextField { b00, b01, b10, b11 } => {
                let s00 = self.spp_to_rec(store, b00, target, memo);
                let s01 = self.spp_to_rec(store, b01, target, memo);
                let s10 = self.spp_to_rec(store, b10, target, memo);
                let s11 = self.spp_to_rec(store, b11, target, memo);
                store.mk(s00, s01, s10, s11)
            }
            // Field-bottom node (a `NextState` or `Root`): a depth-0 SPP that
            // accepts iff we landed on `target`.
            _ => spp::SPP::new((node == target) as u32),
        };
        memo.insert(node, spp);
        spp
    }

    /// Generalize a `Cand` from examples, in one shot.
    ///
    /// The packet width `nv` is taken from `store` (length of every `pkt_*`);
    /// the state-vector width `ns` from `dfa.num_states()` (every `states`
    /// vector has length `ns`, and — since each state level is a predicate over
    /// the same state set, also the width of each `NextState` node — every
    /// `state` value must lie in `0..ns`).
    ///
    /// Guarantee: let `out` be the output. Then for each `(input, b)` pair in
    /// the input, `out.accepts_input(input) == b`.
    ///
    /// Returns [`ConflictError`] if two examples give the same concrete `Input`
    /// opposite labels.
    ///
    /// # Method (trie + RPNI-style per-level merging)
    ///
    /// Same algorithm as [`crate::spp::learner::learn_spp`], over a mixed-arity
    /// decision diagram.  An example's *decision path* has length
    /// `nv + ns + nv`: `2*pkt_in[i] + pkt_start[i]` at the `nv` field levels
    /// (width 4), then `states[j]` at the `ns` state levels (width `ns`), then
    /// `2*pkt_end[i] + pkt_out[i]` at the `nv` output levels (width 4).
    ///
    /// 1. **Trie.** Insert every path into a mixed-arity trie, leaves carrying
    ///    the label.  This computes *all* the prefix partitions in one pass: a
    ///    node at depth `d` is exactly the set of examples sharing that
    ///    length-`d` prefix, and a shared prefix is walked only once.  A leaf
    ///    reached with two different labels is a [`ConflictError`].
    /// 2. **Per-level merging (RPNI-style).** Sweep the trie top-down, one
    ///    level at a time; two prefix cells are *compatible* when their suffix
    ///    evidence never conflicts.  Each node is greedily merged into the
    ///    first compatible class at its level (unioning the subtrees), so
    ///    evidence pooled at one level narrows the don't-cares at every level
    ///    below.
    /// 3. **Bottom-up fold.** Fold the merged levels leaves-first, in the same
    ///    three regions as the path: a leaf class becomes its accept/reject
    ///    terminal and each output-level class an [`spp::SPPstore::mk`] node;
    ///    at the boundary, each depth-`nv + ns` class's SPP is wrapped in a
    ///    [`CandNode::Root`]; state-level classes become
    ///    [`CandNode::NextState`] and field-level classes
    ///    [`CandNode::NextField`], hash-consed by [`Builder::mk`].  Don't-care
    ///    branches (those no example exercises) are filled with the first
    ///    defined sibling child.
    pub fn from_examples(
        store: &mut spp::SPPstore,
        dfa: &'a ExplicitDFA,
        examples: &[(Input, bool)],
    ) -> Result<Self, ConflictError> {
        let nv = store.num_vars() as usize;
        let ns = dfa.num_states();
        let total = nv + ns + nv;
        // Branch width at each trie depth: field and output levels are 4-way,
        // state levels are `ns`-way.
        let width = |d: usize| if d < nv || d >= nv + ns { 4 } else { ns };

        let mut builder = Builder::default();

        // Empty input: we just accept everything
        if examples.is_empty() {
            let mut node = builder.mk(CandNode::Root(store.top));
            // `ns` state levels, each a `NextState` with one child per state.
            for _ in 0..ns {
                node = builder.mk(CandNode::NextState(vec![node; ns]));
            }
            for _ in 0..nv {
                node = builder.mk(CandNode::NextField {
                    b00: node,
                    b01: node,
                    b10: node,
                    b11: node,
                });
            }
            return Ok(Cand {
                nodes: builder.nodes,
                head: node,
                dfa,
            });
        }

        // Flatten every example into its decision path (and check well-formedness).
        let decisions: Vec<Vec<usize>> = examples
            .iter()
            .map(|(input, _)| {
                assert_eq!(input.pkt_in.len(), nv);
                assert_eq!(input.pkt_start.len(), nv);
                assert_eq!(input.pkt_end.len(), nv);
                assert_eq!(input.pkt_out.len(), nv);
                assert_eq!(input.states.len(), ns);
                let mut path = Vec::with_capacity(total);
                for i in 0..nv {
                    path.push(2 * input.pkt_in[i] as usize + input.pkt_start[i] as usize);
                }
                for &s in &input.states {
                    // The DFA is complete (any sink is a real state), so every
                    // state value lies in `0..ns`.
                    assert!(s < ns, "state value out of range");
                    path.push(s);
                }
                for i in 0..nv {
                    path.push(2 * input.pkt_end[i] as usize + input.pkt_out[i] as usize);
                }
                path
            })
            .collect();

        // Zero-depth diagram (`nv == 0` and `ns == 0`): every example is the
        // single empty path, so they must all agree (there is no trie level to
        // disambiguate them).
        if total == 0 {
            let label = examples[0].1;
            if examples.iter().any(|(_, l)| *l != label) {
                return Err(ConflictError);
            }
            let head = builder.mk(CandNode::Root(spp::SPP::new(label as u32)));
            return Ok(Cand {
                nodes: builder.nodes,
                head,
                dfa,
            });
        }

        // Step 1: build the decision-path trie (node 0 is the root).  This
        // computes every prefix partition in one pass.  A leaf reached with two
        // labels is a conflict (the full path identifies the input).
        let mut trie: Vec<TrieNode> = vec![TrieNode::Branch(vec![None; width(0)])];
        for (path, &(_, label)) in decisions.iter().zip(examples) {
            let mut cur = 0usize;
            for (d, &b) in path.iter().enumerate() {
                let is_last = d + 1 == total;
                let next = match &trie[cur] {
                    TrieNode::Branch(children) => children[b],
                    TrieNode::Leaf(_) => unreachable!("path longer than trie depth"),
                };
                match next {
                    Some(nx) => {
                        if is_last
                            && let TrieNode::Leaf(prev) = trie[nx as usize]
                            && prev != label
                        {
                            return Err(ConflictError);
                        }
                        cur = nx as usize;
                    }
                    None => {
                        let id = trie.len() as u32;
                        trie.push(if is_last {
                            TrieNode::Leaf(label)
                        } else {
                            TrieNode::Branch(vec![None; width(d + 1)])
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
        let levels = merge_levels(&mut trie, total);

        // Step 3: fold the merged levels bottom-up.  Each class is folded once,
        // however many parent edges point at it, memoized by trie node id.

        // Output levels (and the leaves): fold into SPPs.
        let mut spp_of: Vec<Option<spp::SPP>> = vec![None; trie.len()];
        for d in (nv + ns..=total).rev() {
            for &node in &levels[d] {
                let s = match &trie[node as usize] {
                    TrieNode::Leaf(label) => spp::SPP::new(*label as u32),
                    TrieNode::Branch(children) => {
                        let row = fill_row(children, &spp_of);
                        store.mk(row[0], row[1], row[2], row[3])
                    }
                };
                spp_of[node as usize] = Some(s);
            }
        }

        // Boundary: wrap each depth-`nv + ns` class's SPP in a `Root` node.
        let mut cand_of: Vec<Option<CandIdx>> = vec![None; trie.len()];
        for &node in &levels[nv + ns] {
            let root = builder.mk(CandNode::Root(spp_of[node as usize].unwrap()));
            cand_of[node as usize] = Some(root);
        }

        // State levels: `ns` levels, each a `NextState` of width `ns` (one child
        // per state of the complete DFA).
        for d in (nv..nv + ns).rev() {
            for &node in &levels[d] {
                let TrieNode::Branch(children) = &trie[node as usize] else {
                    unreachable!("leaf above the bottom level");
                };
                let row = fill_row(children, &cand_of);
                cand_of[node as usize] = Some(builder.mk(CandNode::NextState(row)));
            }
        }

        // Field levels (width 4).
        for d in (0..nv).rev() {
            for &node in &levels[d] {
                let TrieNode::Branch(children) = &trie[node as usize] else {
                    unreachable!("leaf above the bottom level");
                };
                let row = fill_row(children, &cand_of);
                cand_of[node as usize] = Some(builder.mk(CandNode::NextField {
                    b00: row[0],
                    b01: row[1],
                    b10: row[2],
                    b11: row[3],
                }));
            }
        }

        // The root (node 0) is the sole depth-0 class.
        let head = cand_of[0].unwrap();
        Ok(Cand {
            nodes: builder.nodes,
            head,
            dfa,
        })
    }

    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    fn get_node(&self, idx: CandIdx) -> &CandNode {
        &self.nodes[idx as usize]
    }

    pub fn accepts_input(&self, store: &mut spp::SPPstore, input: &Input) -> bool {
        // Check that the input is well-formed
        let num_vars = store.num_vars() as usize;
        assert_eq!(input.pkt_in.len(), num_vars);
        assert_eq!(input.pkt_start.len(), num_vars);
        assert_eq!(input.pkt_end.len(), num_vars);
        assert_eq!(input.pkt_out.len(), num_vars);
        assert_eq!(input.states.len(), self.num_states());

        let mut node = self.head;

        for (x, y) in input
            .pkt_in
            .iter()
            .copied()
            .zip(input.pkt_start.iter().copied())
        {
            let CandNode::NextField { b00, b01, b10, b11 } = *self.get_node(node) else {
                unreachable!("malformed cand");
            };
            node = match (x, y) {
                (false, false) => b00,
                (false, true) => b01,
                (true, false) => b10,
                (true, true) => b11,
            }
        }

        for &state in &input.states {
            let CandNode::NextState(children) = self.get_node(node) else {
                unreachable!("malformed cand");
            };
            node = children[state];
        }

        let CandNode::Root(spp) = *self.get_node(node) else {
            unreachable!("malformed cand");
        };
        store.accepts(spp, &input.pkt_end, &input.pkt_out)
    }
}

/// Hash-consed arena for `CandNode`s used while building a [`Cand`].
#[derive(Default)]
struct Builder {
    nodes: Vec<CandNode>,
    hc: HashMap<CandNode, CandIdx>,
}

impl Builder {
    fn mk(&mut self, node: CandNode) -> CandIdx {
        if let Some(&idx) = self.hc.get(&node) {
            return idx;
        }
        let idx = self.nodes.len() as CandIdx;
        self.nodes.push(node.clone());
        self.hc.insert(node, idx);
        idx
    }
}

/// A node of the decision-path trie: an internal branch (width 4 at field and
/// output levels, `ns` at state levels — merging and compatibility only ever
/// compare same-depth nodes, so the width is implicit) or an accept/reject
/// leaf.
#[derive(Clone)]
enum TrieNode {
    Branch(Vec<Option<u32>>),
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
fn merge_levels(trie: &mut [TrieNode], total: usize) -> Vec<Vec<u32>> {
    let mut levels: Vec<Vec<u32>> = Vec::with_capacity(total + 1);
    levels.push(vec![0]);
    for d in 0..total {
        // Gather the populated (parent, branch, child) slots feeding depth d+1.
        let mut slots: Vec<(u32, usize, u32)> = Vec::new();
        for &p in &levels[d] {
            if let TrieNode::Branch(children) = &trie[p as usize] {
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
    match (&trie[a], &trie[b]) {
        (TrieNode::Leaf(la), TrieNode::Leaf(lb)) => la == lb,
        (TrieNode::Branch(ca), TrieNode::Branch(cb)) => {
            ca.iter().zip(cb).all(|(x, y)| match (x, y) {
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
    let (mut merged, cb) = match (&trie[a], &trie[b]) {
        (TrieNode::Branch(ca), TrieNode::Branch(cb)) => (ca.clone(), cb.clone()),
        _ => return, // leaves: labels already known to agree
    };
    for (slot, y) in merged.iter_mut().zip(cb) {
        match (*slot, y) {
            (Some(x), Some(y)) => merge(trie, x as usize, y as usize),
            (None, Some(y)) => *slot = Some(y),
            _ => {}
        }
    }
    trie[a] = TrieNode::Branch(merged);
}

/// Look up the already-folded results of a branch class's children in `done`
/// and fill the don't-care slots (branches no example exercises) with the
/// first defined sibling child.
fn fill_row<T: Copy>(children: &[Option<u32>], done: &[Option<T>]) -> Vec<T> {
    let row: Vec<Option<T>> = children
        .iter()
        .map(|c| c.map(|id| done[id as usize].unwrap()))
        .collect();
    let fill = row
        .iter()
        .flatten()
        .next()
        .copied()
        .expect("trie branch has at least one child");
    row.into_iter().map(|c| c.unwrap_or(fill)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `Input` from concrete fields.
    fn input(
        pkt_in: &[bool],
        pkt_start: &[bool],
        states: &[usize],
        pkt_end: &[bool],
        pkt_out: &[bool],
    ) -> Input {
        Input {
            pkt_in: pkt_in.to_vec(),
            pkt_start: pkt_start.to_vec(),
            states: states.to_vec(),
            pkt_end: pkt_end.to_vec(),
            pkt_out: pkt_out.to_vec(),
        }
    }

    /// Assert the core guarantee: the `Cand` reproduces every example's label.
    fn assert_consistent(cand: &Cand<'_>, store: &mut spp::SPPstore, examples: &[(Input, bool)]) {
        for (inp, label) in examples {
            assert_eq!(cand.accepts_input(store, inp), *label, "example mislabeled");
        }
    }

    /// A *complete* DFA with `num_states` states: every state has a `top`
    /// self-loop, so its transition function is total (the invariant [`Cand`]
    /// requires) without changing the state count.
    fn example_dfa(store: &mut spp::SPPstore, num_states: usize) -> ExplicitDFA {
        ExplicitDFA {
            start: 0,
            transitions: (0..num_states).map(|i| vec![(store.top, i)]).collect(),
            outputs: vec![store.zero; num_states],
        }
    }

    #[test]
    fn empty_accepts_everything() {
        let mut store = spp::SPPstore::new(2);
        let dfa = example_dfa(&mut store, 2);
        let cand = Cand::from_examples(&mut store, &dfa, &[]).unwrap();
        let inp = input(
            &[true, false],
            &[false, true],
            &[0, 0],
            &[true, true],
            &[false, false],
        );
        assert!(cand.accepts_input(&mut store, &inp));
    }

    #[test]
    fn single_positive() {
        let mut store = spp::SPPstore::new(2);
        let examples = vec![(
            input(
                &[true, false],
                &[false, true],
                &[1, 0],
                &[true, true],
                &[false, true],
            ),
            true,
        )];
        let dfa = example_dfa(&mut store, 2);
        let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();
        assert_consistent(&cand, &mut store, &examples);
    }

    #[test]
    fn single_negative() {
        let mut store = spp::SPPstore::new(2);
        let examples = vec![(
            input(
                &[false, false],
                &[false, false],
                &[0, 0],
                &[false, false],
                &[false, false],
            ),
            false,
        )];
        let dfa = example_dfa(&mut store, 2);
        let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();
        assert_consistent(&cand, &mut store, &examples);
    }

    #[test]
    fn conflict_detected() {
        let mut store = spp::SPPstore::new(1);
        let mk = || input(&[true], &[false], &[0], &[true], &[false]);
        let examples = vec![(mk(), true), (mk(), false)];
        let dfa = example_dfa(&mut store, 1);
        assert!(Cand::from_examples(&mut store, &dfa, &examples).is_err());
    }

    #[test]
    fn duplicate_same_label_ok() {
        let mut store = spp::SPPstore::new(1);
        let mk = || input(&[true], &[false], &[0], &[true], &[false]);
        let examples = vec![(mk(), true), (mk(), true)];
        let dfa = example_dfa(&mut store, 1);
        let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();
        assert_consistent(&cand, &mut store, &examples);
    }

    #[test]
    fn multiple_consistent() {
        let mut store = spp::SPPstore::new(2);
        let examples = vec![
            (
                input(
                    &[false, false],
                    &[false, false],
                    &[0, 0],
                    &[false, false],
                    &[false, false],
                ),
                true,
            ),
            (
                input(
                    &[true, false],
                    &[true, false],
                    &[1, 0],
                    &[true, false],
                    &[true, false],
                ),
                true,
            ),
            (
                input(
                    &[false, true],
                    &[true, false],
                    &[0, 1],
                    &[true, false],
                    &[false, true],
                ),
                false,
            ),
            (
                input(
                    &[true, true],
                    &[true, true],
                    &[1, 1],
                    &[false, true],
                    &[true, true],
                ),
                false,
            ),
        ];
        let dfa = example_dfa(&mut store, 2);
        let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();
        assert_consistent(&cand, &mut store, &examples);
    }

    /// Two prefix cells with non-conflicting suffix evidence must be merged, so
    /// evidence observed under one prefix generalizes to the other (the analogue
    /// of `merges_compatible_prefixes` in `spp::learner`).
    ///
    /// With `nv = 1`, `ns = 1` a decision path is `[field, state, output]`:
    ///
    /// * A: `(in=0, start=0)`, state 0, `(end=0, out=0)`, positive → `[0, 0, 0] ↦ 1`
    /// * B: `(in=1, start=1)`, state 0, `(end=1, out=0)`, negative → `[3, 0, 2] ↦ 0`
    ///
    /// The field-level cells at branches 0 and 3 exercise *disjoint* output
    /// branches (0 vs 2), so they are compatible and merge into one class; the
    /// merged output node has row `[1, ·, 0, ·]`, and the fill policy (first
    /// defined sibling child) completes it to `mk(1, 1, 0, 1)`.  So the query
    /// `(in=1, start=1)`, state 0, `(end=0, out=0)` — B's prefix with A's
    /// suffix — routes through field branch 3 to the merged class and lands on
    /// output branch 0, an accept.  Without merging, B's prefix cell holds only
    /// the lone negative and collapses to reject-all, so the query is rejected.
    #[test]
    fn merges_compatible_prefixes() {
        let mut store = spp::SPPstore::new(1);
        let dfa = example_dfa(&mut store, 1);
        let examples = vec![
            (input(&[false], &[false], &[0], &[false], &[false]), true),
            (input(&[true], &[true], &[0], &[true], &[false]), false),
        ];
        let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();
        assert_consistent(&cand, &mut store, &examples);
        assert!(
            cand.accepts_input(
                &mut store,
                &input(&[true], &[true], &[0], &[false], &[false])
            ),
            "positive evidence under prefix (0,0) must generalize to the merged prefix (1,1)"
        );
    }

    /// The `ENFA` view must agree with `accepts_input`.
    ///
    /// `example_dfa` has no transitions, so (for `num_states >= 1`) a `Middle`
    /// node is a dead end and every accepted trace has length 2:
    /// `[pkt_in, pkt_start]`.  After the `Start` step the automaton sits in
    /// `Middle(n, identity)` with the last packet equal to `pkt_start`, and its
    /// `output` is the `Root` SPP reached by walking `n` through the identity
    /// state vector.  That is exactly `accepts_input` on the input whose
    /// `states` is the identity and whose `pkt_end` is `pkt_start`.
    #[test]
    fn enfa_matches_accepts_input() {
        const NV: u32 = 2;
        const NS: usize = 2;

        let bits = |n: u32| {
            (0..n)
                .map(|_| rand::random::<bool>())
                .collect::<Vec<bool>>()
        };
        let identity: Vec<usize> = (0..NS).collect();

        for _ in 0..50 {
            let mut store = spp::SPPstore::new(NV);
            let dfa = example_dfa(&mut store, NS);

            // Build from random (deduplicated) examples.
            let mut seen: HashMap<Vec<usize>, bool> = HashMap::new();
            let mut examples: Vec<(Input, bool)> = Vec::new();
            for _ in 0..20 {
                let states: Vec<usize> = (0..NS).map(|_| rand::random_range(0..NS)).collect();
                let inp = input(&bits(NV), &bits(NV), &states, &bits(NV), &bits(NV));
                let label = rand::random::<bool>();
                let mut key: Vec<usize> = Vec::new();
                key.extend(inp.pkt_in.iter().map(|&b| b as usize));
                key.extend(inp.pkt_start.iter().map(|&b| b as usize));
                key.extend(&inp.states);
                key.extend(inp.pkt_end.iter().map(|&b| b as usize));
                key.extend(inp.pkt_out.iter().map(|&b| b as usize));
                if seen.insert(key, label).is_none() {
                    examples.push((inp, label));
                }
            }
            let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();

            // The two evaluation paths must agree on arbitrary queries.
            for _ in 0..20 {
                let pkt_in = bits(NV);
                let pkt_start = bits(NV);
                let pkt_out = bits(NV);

                let via_input = cand.accepts_input(
                    &mut store,
                    &input(&pkt_in, &pkt_start, &identity, &pkt_start, &pkt_out),
                );
                let trace = vec![pkt_in, pkt_start];
                let via_enfa = cand.dfa_accepts(&mut store, &trace, &pkt_out);

                assert_eq!(via_input, via_enfa, "ENFA disagrees with accepts_input");
            }
        }
    }

    #[test]
    fn random_consistent() {
        const NV: u32 = 3;
        const NS: u32 = 2;

        for _ in 0..100 {
            let mut store = spp::SPPstore::new(NV);

            // Collect non-conflicting random examples (dedup by input).
            let mut seen: HashMap<Vec<usize>, bool> = HashMap::new();
            let mut examples: Vec<(Input, bool)> = Vec::new();
            for _ in 0..40 {
                let bits = |n: u32| (0..n).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
                let states: Vec<usize> = (0..NS)
                    .map(|_| rand::random_range(0..NS as usize))
                    .collect();
                let inp = input(&bits(NV), &bits(NV), &states, &bits(NV), &bits(NV));
                let label = rand::random::<bool>();
                let mut key: Vec<usize> = Vec::new();
                key.extend(inp.pkt_in.iter().map(|&b| b as usize));
                key.extend(inp.pkt_start.iter().map(|&b| b as usize));
                key.extend(&inp.states);
                key.extend(inp.pkt_end.iter().map(|&b| b as usize));
                key.extend(inp.pkt_out.iter().map(|&b| b as usize));
                if seen.insert(key, label).is_none() {
                    examples.push((inp, label));
                }
            }

            let dfa = example_dfa(&mut store, NS as usize);
            let cand = Cand::from_examples(&mut store, &dfa, &examples).unwrap();
            assert_consistent(&cand, &mut store, &examples);
        }
    }

    // ---- Cand ENFA over a transition-bearing DFA (`dup; dup`) ----------
    //
    // Every test above uses `example_dfa`, which has *no* transitions, so a
    // `Middle` state is always a dead end and the ENFA only ever accepts
    // length-2 traces.  These tests use a real DFA whose states have
    // transitions: `dup; dup`, whose DFA is `0 -one-> 1 -one-> 2` with state 2
    // the only (dead) accepting state, accepting exactly `[p, p, p] -> p`.

    use crate::expr::Expr;
    use crate::holes::aut::expr_to_dfa;

    fn two_dup_dfa(store: &mut spp::SPPstore) -> ExplicitDFA {
        expr_to_dfa(&Expr::sequence(Expr::dup(), Expr::dup()), store)
    }

    /// Totalize `dfa` (add a sink state) and materialize it, satisfying the
    /// completeness invariant `Cand` requires of its DFA.
    fn complete(store: &mut spp::SPPstore, dfa: &ExplicitDFA) -> ExplicitDFA {
        ExplicitDFA::from_dfa(store, ops::WithSinkState(dfa))
    }

    /// Sanity: `dup; dup` accepts exactly the length-3 trace `[p, p, p] -> p`
    /// (two dup-crossings), and nothing of length 2 or 4.
    #[test]
    fn two_dup_dfa_accepts_only_length_three() {
        let mut store = spp::SPPstore::new(1);
        let dfa = two_dup_dfa(&mut store);
        let p = vec![false];
        assert!(!dfa.dfa_accepts(&mut store, &[p.clone(), p.clone()], &p));
        assert!(dfa.dfa_accepts(&mut store, &[p.clone(), p.clone(), p.clone()], &p));
        assert!(!dfa.dfa_accepts(
            &mut store,
            &[p.clone(), p.clone(), p.clone(), p.clone()],
            &p
        ));
    }

    /// Contrast / control: over the (completed) *single*-`dup` DFA, the
    /// most-permissive `Cand` accepts `dup`'s length-2 trace `[p, p] -> p` via
    /// the `Start -> Middle -> output` path (which needs no `Middle` step).
    #[test]
    fn top_cand_over_one_dup_accepts_its_length_two_trace() {
        let mut store = spp::SPPstore::new(1);
        let raw = expr_to_dfa(&Expr::dup(), &mut store);
        let dfa = complete(&mut store, &raw);
        let top = Cand::from_examples(&mut store, &dfa, &[]).unwrap();
        let p = vec![false];
        assert!(top.dfa_accepts(&mut store, &[p.clone(), p.clone()], &p));
    }

    /// Regression test for the `Cand` ENFA stall over a transition-bearing DFA.
    ///
    /// `Cand::from_examples(.., &[])` is the most-permissive candidate
    /// (`Cand::top`, "accepts everything"), so over the (completed) `dup; dup`
    /// DFA it must accept the length-3 trace `[p, p, p] -> p` that `dup; dup`
    /// produces.  Stepping over an *incomplete* DFA would stall when a component
    /// hits a dead state; because the DFA is completed (every dead component
    /// flows into the sink), the product keeps stepping and reaches length 3.
    #[test]
    fn top_cand_over_two_dups_reaches_length_three() {
        let mut store = spp::SPPstore::new(1);
        let raw = two_dup_dfa(&mut store);
        let dfa = complete(&mut store, &raw);
        let top = Cand::from_examples(&mut store, &dfa, &[]).unwrap();
        let p = vec![false];

        // The first `Middle` (identity vector over every state of the completed
        // DFA) steps instead of stalling.
        let start = top.start(&mut store);
        let edges = top.transitions(&mut store, &start);
        assert_eq!(edges.len(), 1, "Start has a single routing edge");
        let middle = edges[0].1.clone();
        let identity: Vec<usize> = (0..dfa.num_states()).collect();
        assert!(
            matches!(middle, State::Middle(_, ref v) if *v == identity),
            "Start lands in Middle with the identity state vector"
        );
        assert!(
            !top.transitions(&mut store, &middle).is_empty(),
            "Middle has outgoing transitions (no component stalls on a dead state)"
        );

        // The most-permissive Cand accepts every trace `dup; dup` does — in
        // particular the length-3 `[p, p, p] -> p` (and, being permissive,
        // longer traces too).
        assert!(top.dfa_accepts(&mut store, &[p.clone(), p.clone()], &p));
        assert!(top.dfa_accepts(&mut store, &[p.clone(), p.clone(), p.clone()], &p));
    }

    /// [`ops::WithSinkState`] totalizes the transition function: the dead state
    /// 2 of `dup; dup` steps to the sink, the sink absorbs everything, and a
    /// live state both keeps its real edge and routes the uncovered packet pairs
    /// to the sink.
    #[test]
    fn with_sink_state_totalizes_dead_state() {
        use ops::SinkOr::{Inner, Sink};
        let mut store = spp::SPPstore::new(1);
        let dfa = two_dup_dfa(&mut store); // states 0,1,2 with 2 dead
        let comp = ops::WithSinkState(&dfa);

        // Dead state 2 now steps to the sink on every packet pair.
        assert_eq!(
            comp.transitions(&mut store, &Inner(2)),
            vec![(store.top, Sink)]
        );
        // The sink absorbs everything.
        assert_eq!(comp.transitions(&mut store, &Sink), vec![(store.top, Sink)]);
        // A live state keeps its real successor and routes the rest to the sink.
        let t0 = comp.transitions(&mut store, &Inner(0));
        assert!(t0.iter().any(|(_, tgt)| *tgt == Inner(1)));
        assert!(t0.iter().any(|(_, tgt)| *tgt == Sink));
    }
}
