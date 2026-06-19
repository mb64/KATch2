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

    /// Generalize a `Cand` from examples.
    ///
    /// `num_vars` is the packet width (length of every `pkt_*`); `num_states`
    /// is the length of every `states` vector (and, since each state level is a
    /// predicate over the same state set, also the width of each `NextState`
    /// node, i.e. every `state` value must lie in `0..num_states`).
    ///
    /// Guarantee: let `out` be the output. Then for each `(input, b)` pair in
    /// the input, `out.accepts_input(input) == b`.
    ///
    /// Returns [`ConflictError`] if two examples give the same concrete `Input`
    /// opposite labels.
    pub fn from_examples(
        store: &mut spp::SPPstore,
        dfa: &'a ExplicitDFA,
        examples: &[(Input, bool)],
    ) -> Result<Self, ConflictError> {
        let nv = store.num_vars() as usize;
        let ns = dfa.num_states();

        // Flatten every example into its decision path (and check well-formedness).
        let decisions: Vec<Vec<usize>> = examples
            .iter()
            .map(|(input, _)| {
                assert_eq!(input.pkt_in.len(), nv);
                assert_eq!(input.pkt_start.len(), nv);
                assert_eq!(input.pkt_end.len(), nv);
                assert_eq!(input.pkt_out.len(), nv);
                assert_eq!(input.states.len(), ns);
                let mut path = Vec::with_capacity(nv + ns + nv);
                for i in 0..nv {
                    path.push(2 * input.pkt_in[i] as usize + input.pkt_start[i] as usize);
                }
                for &s in &input.states {
                    assert!(s < ns, "state value out of range");
                    path.push(s);
                }
                for i in 0..nv {
                    path.push(2 * input.pkt_end[i] as usize + input.pkt_out[i] as usize);
                }
                path
            })
            .collect();

        // Conflict check: the full path identifies the input, so a repeated
        // path with opposite labels is a genuine conflict.
        let mut seen: HashMap<&[usize], bool> = HashMap::new();
        for (path, &(_, label)) in decisions.iter().zip(examples) {
            match seen.insert(path, label) {
                Some(prev) if prev != label => return Err(ConflictError),
                _ => {}
            }
        }

        let mut builder = Builder::default();

        // Empty input: we just accept everything
        if examples.is_empty() {
            let mut node = builder.mk(CandNode::Root(store.top));
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

        // Bottom layer: each example maps to the one/zero SPP terminal.
        let mut spp_map: HashMap<usize, spp::SPP> = (0..examples.len())
            .map(|e| (e, spp::SPP::new(examples[e].1 as u32)))
            .collect();

        // Output levels: build the SPP bottom-up.
        for depth in (nv + ns..nv + ns + nv).rev() {
            spp_map = build_layer(4, depth, &decisions, &spp_map, |row| {
                store.mk(row[0], row[1], row[2], row[3])
            });
        }

        // Boundary: wrap each SPP in a `Root` node.
        let mut cand_map: HashMap<usize, CandIdx> = HashMap::new();
        for (&e, &spp) in &spp_map {
            let node = builder.mk(CandNode::Root(spp));
            cand_map.insert(e, node);
        }

        // State levels (width `num_states`).
        for depth in (nv..nv + ns).rev() {
            cand_map = build_layer(ns, depth, &decisions, &cand_map, |row| {
                builder.mk(CandNode::NextState(row.to_vec()))
            });
        }

        // Field levels (width 4).
        for depth in (0..nv).rev() {
            cand_map = build_layer(4, depth, &decisions, &cand_map, |row| {
                builder.mk(CandNode::NextField {
                    b00: row[0],
                    b01: row[1],
                    b10: row[2],
                    b11: row[3],
                })
            });
        }

        // At depth 0 every example shares the empty prefix, so they all map to
        // the single root node.
        let head = cand_map[&0];
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

/// Build one layer of the diagram, bottom-up.
///
/// `below` maps each example to the node it reaches *just below* this layer
/// (depth `depth + 1`).  Examples are grouped by their length-`depth` prefix —
/// everything sharing a prefix reaches the same node here — and each group's
/// `width` child branches are populated from the examples' decision at `depth`.
/// Branches no example exercises are filled with one of the group's defined
/// children, then `mk` hash-conses the node.  Returns the map from each example
/// to its node at this layer.
fn build_layer<Id: Copy + Eq + std::hash::Hash>(
    width: usize,
    depth: usize,
    decisions: &[Vec<usize>],
    below: &HashMap<usize, Id>,
    mut mk: impl FnMut(&[Id]) -> Id,
) -> HashMap<usize, Id> {
    // Group examples by their prefix up to (but excluding) this layer.
    let mut groups: HashMap<&[usize], Vec<usize>> = HashMap::new();
    for &e in below.keys() {
        groups.entry(&decisions[e][..depth]).or_default().push(e);
    }

    let mut result = HashMap::new();
    for (_prefix, members) in groups {
        let mut row: Vec<Option<Id>> = vec![None; width];
        for &e in &members {
            row[decisions[e][depth]] = Some(below[&e]);
        }
        // Every group has at least one member, hence at least one defined
        // branch to fill the don't-cares with.
        let fill = row
            .iter()
            .flatten()
            .next()
            .copied()
            .expect("non-empty group");
        let full: Vec<Id> = row.into_iter().map(|c| c.unwrap_or(fill)).collect();
        let node = mk(&full);
        for e in members {
            result.insert(e, node);
        }
    }
    result
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

    /// A DFA with `num_states` states
    fn example_dfa(store: &mut spp::SPPstore, num_states: usize) -> ExplicitDFA {
        ExplicitDFA {
            start: 0,
            transitions: vec![vec![]; num_states],
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

    /// Contrast / control: over the *single*-`dup` DFA (`0 -one-> 1`, state 1
    /// dead-accepting), the most-permissive `Cand` reaches `Middle(_, [0, 1])`
    /// after one step.  That `Middle` already has no outgoing transitions (the
    /// dead state 1 can't step), but `dup` only needs the length-2 trace
    /// `[p, p] -> p`, which the ENFA *does* reach via the `Start -> Middle ->
    /// output` path.  So a single dup is representable — the stall only bites
    /// when a second step is needed.
    #[test]
    fn top_cand_over_one_dup_accepts_its_length_two_trace() {
        let mut store = spp::SPPstore::new(1);
        let dfa = expr_to_dfa(&Expr::dup(), &mut store);
        let top = Cand::from_examples(&mut store, &dfa, &[]).unwrap();
        let p = vec![false];
        assert!(top.dfa_accepts(&mut store, &[p.clone(), p.clone()], &p));
    }

    /// BUG (localized to the `Cand` ENFA / its use of `Exponential`):
    /// `Cand::from_examples(.., &[])` is the most-permissive candidate
    /// (`Cand::top`, documented "accepts everything"), yet over the `dup; dup`
    /// DFA its ENFA accepts only length-2 traces.
    ///
    /// From `Start` it reaches `Middle(_, [0, 1, 2])` (the *identity* state
    /// vector).  `Exponential::transitions` requires every component to step
    /// simultaneously (it intersects the per-component SPPs), but component 2 —
    /// the dead accepting state — has no transition, so the whole vector is
    /// stuck and `Middle([0,1,2])` has no outgoing edges.  Hence the ENFA can't
    /// produce the length-3 trace `[p, p, p] -> p` that `dup; dup` accepts, so
    /// `top ⊉ dup; dup`.  Because CEGIS starts from `top` and only ever shrinks
    /// it, this makes `solve_holes_full` wrongly report `Hole == dup; dup`
    /// infeasible (see `holes::mod::full_wrongly_infeasible_chained_dup`).
    ///
    /// Asserts the current (buggy) behaviour as a tripwire; once the ENFA is
    /// fixed the length-3 trace should be accepted and these assertions flipped.
    #[test]
    fn top_cand_over_two_dups_stalls_at_length_two() {
        let mut store = spp::SPPstore::new(1);
        let dfa = two_dup_dfa(&mut store);
        let top = Cand::from_examples(&mut store, &dfa, &[]).unwrap();
        let p = vec![false];

        // One routing edge out of Start, into a Middle that is then stuck.
        let start = top.start(&mut store);
        let edges = top.transitions(&mut store, &start);
        assert_eq!(edges.len(), 1, "Start has a single routing edge");
        let middle = edges[0].1.clone();
        assert!(
            matches!(middle, State::Middle(_, ref v) if *v == vec![0, 1, 2]),
            "Start lands in Middle with the identity state vector"
        );
        assert!(
            top.transitions(&mut store, &middle).is_empty(),
            "BUG: Middle([0,1,2]) has no transitions — Exponential stalls on \
             the dead state 2, so the ENFA can never advance past length 2"
        );

        // Length 2 is accepted; the length-3 trace `dup; dup` accepts is not.
        assert!(top.dfa_accepts(&mut store, &[p.clone(), p.clone()], &p));
        assert!(
            !top.dfa_accepts(&mut store, &[p.clone(), p.clone(), p.clone()], &p),
            "BUG: the most-permissive Cand should accept every trace dup;dup \
             does, including the length-3 [p, p, p] -> p"
        );
    }
}
