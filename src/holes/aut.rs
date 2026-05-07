use crate::sp;
use crate::spp;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

/// Nondeterministic symbolic NetKAT automaton with epsilon transitions.
///
/// States are either *visible* (consume a packet from the trace when entered)
/// or *invisible* (epsilon states that do not advance the trace position).
///
/// All trait methods take an explicit `&mut spp::SPPstore`, which the
/// underlying automaton may use to compute SPPs on demand.  Implementations
/// that don't need the store may simply ignore it.
pub trait ENFA {
    type State: Clone + Eq + Hash;

    /// The start: a list of `(spp, state)` pairs.  Operationally, the run
    /// begins by nondeterministically picking some `(spp, q)` and starting
    /// at `q` with current packet `p` chosen so that `(input, p) ∈ spp`.
    fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, Self::State)>;

    /// True for visible states (entering them consumes one packet from the trace);
    /// false for invisible (epsilon) states.
    fn is_visible(&self, store: &mut spp::SPPstore, q: &Self::State) -> bool;

    /// Outgoing transitions from `q`: each entry is a (packet_program, next_state) pair.
    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)>;

    /// Output SPP for `q`: the set of (current_packet, output_packet) pairs
    /// the automaton accepts when it terminates in state `q`.
    fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP;
}

/// Nondeterministic symbolic NetKAT automaton without epsilon transitions.
///
/// Every state is visible; `is_visible` must always return `true`.
pub trait NFA: ENFA {
    /// Returns `true` iff the concrete trace `(input, trace, output)` is
    /// accepted by *some* path through the NFA.
    ///
    /// Tracks the set of states reachable so far, broadening it at each step
    /// by every transition whose SPP relates the current packet to the next.
    fn nfa_accepts(
        &self,
        store: &mut spp::SPPstore,
        input: &[bool],
        trace: &[Vec<bool>],
        output: &[bool],
    ) -> bool {
        // Track `(spp_pre, q)` pairs where `spp_pre` relates `current_pkt` to
        // the current packet at `q`.  Initially `current_pkt = input` and the
        // pre-SPPs are the start SPPs; after the first consumption they all
        // collapse to identity since `current_pkt` from then on is exactly
        // the previously-consumed trace element.
        let mut frontier: Vec<(spp::SPP, Self::State)> = self.start(store);
        let mut current_pkt: Vec<bool> = input.to_vec();

        for next_pkt in trace {
            let mut new_states: HashSet<Self::State> = HashSet::new();
            for (spp_pre, q) in &frontier {
                for (spp_t, q_next) in self.transitions(store, q) {
                    let composed = store.sequence(*spp_pre, spp_t);
                    if store.accepts(composed, &current_pkt, next_pkt) {
                        new_states.insert(q_next);
                    }
                }
            }
            if new_states.is_empty() {
                return false;
            }
            let one = store.one;
            frontier = new_states.into_iter().map(|q| (one, q)).collect();
            current_pkt = next_pkt.clone();
        }

        for (spp_pre, q) in &frontier {
            let out_spp = self.output(store, q);
            let composed = store.sequence(*spp_pre, out_spp);
            if store.accepts(composed, &current_pkt, output) {
                return true;
            }
        }
        false
    }
}

/// Deterministic symbolic NetKAT automaton.
///
/// For each state, the SPPs labeling outgoing transitions are pairwise disjoint.
pub trait DFA: NFA {
    /// Returns `true` iff the concrete trace `(input, trace, output)` is
    /// accepted: starting at `start()` with packet `input`, each `trace[i]`
    /// is reachable from the previous packet via some transition SPP, and
    /// the final state's output SPP relates the last packet to `output`.
    fn dfa_accepts(
        &self,
        store: &mut spp::SPPstore,
        input: &[bool],
        trace: &[Vec<bool>],
        output: &[bool],
    ) -> bool {
        let (spp_start, q_start) = self.dfa_start(store);

        if trace.is_empty() {
            let out_spp = self.output(store, &q_start);
            let composed = store.sequence(spp_start, out_spp);
            return store.accepts(composed, input, output);
        }

        let next_pkt = &trace[0];
        let mut found: Option<Self::State> = None;
        for (spp_t, q_next) in self.transitions(store, &q_start) {
            let composed = store.sequence(spp_start, spp_t);
            if store.accepts(composed, input, next_pkt) {
                found = Some(q_next);
                break;
            }
        }
        let mut current_state = match found {
            Some(q) => q,
            None => return false,
        };
        let mut current_pkt: Vec<bool> = next_pkt.clone();

        for next_pkt in &trace[1..] {
            let mut next_state = None;
            for (spp, q_next) in self.transitions(store, &current_state) {
                if store.accepts(spp, &current_pkt, next_pkt) {
                    next_state = Some(q_next);
                    break;
                }
            }
            match next_state {
                Some(q) => {
                    current_state = q;
                    current_pkt = next_pkt.clone();
                }
                None => return false,
            }
        }

        let out_spp = self.output(store, &current_state);
        store.accepts(out_spp, &current_pkt, output)
    }

    /// The unique `(start_spp, start_state)` of a DFA.  Panics if `start`
    /// returns anything other than exactly one entry: a DFA must have a
    /// single deterministic starting point or it is not really a DFA.
    fn dfa_start(&self, store: &mut spp::SPPstore) -> (spp::SPP, Self::State) {
        let mut starts = self.start(store);
        assert_eq!(
            starts.len(),
            1,
            "DFA must have exactly one start; found {}",
            starts.len()
        );
        starts.pop().unwrap()
    }
}

// ---- Memo<T> ----------------------------------------------------------------

struct MemoData<S: Clone + Eq + Hash> {
    state_to_id: HashMap<S, usize>,
    id_to_state: Vec<S>,
    transitions_cache: HashMap<usize, Vec<(spp::SPP, usize)>>,
    output_cache: HashMap<usize, spp::SPP>,
    visible_cache: HashMap<usize, bool>,
}

impl<S: Clone + Eq + Hash> MemoData<S> {
    fn new() -> Self {
        MemoData {
            state_to_id: HashMap::new(),
            id_to_state: Vec::new(),
            transitions_cache: HashMap::new(),
            output_cache: HashMap::new(),
            visible_cache: HashMap::new(),
        }
    }

    fn get_or_insert(&mut self, state: S) -> usize {
        if let Some(&id) = self.state_to_id.get(&state) {
            return id;
        }
        let id = self.id_to_state.len();
        self.id_to_state.push(state.clone());
        self.state_to_id.insert(state, id);
        id
    }

    fn get_state(&self, id: usize) -> &S {
        &self.id_to_state[id]
    }
}

/// Wraps an automaton `T`, replacing its `State` type with `usize`.
///
/// The first time a `T::State` is encountered it is assigned a fresh integer
/// ID; subsequent lookups return the same ID.  All trait method results are
/// cached on that ID, so expensive computations in `T` are performed at most
/// once per state.
pub struct Memo<T: ENFA> {
    inner: T,
    data: RefCell<MemoData<T::State>>,
}

impl<T: ENFA> Memo<T> {
    pub fn new(inner: T) -> Self {
        Memo {
            inner,
            data: RefCell::new(MemoData::new()),
        }
    }
}

impl<T: ENFA> ENFA for Memo<T> {
    type State = usize;

    fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, usize)> {
        let raw = self.inner.start(store);
        let mut data = self.data.borrow_mut();
        raw.into_iter()
            .map(|(spp, s)| (spp, data.get_or_insert(s)))
            .collect()
    }

    fn is_visible(&self, store: &mut spp::SPPstore, id: &usize) -> bool {
        if let Some(&v) = self.data.borrow().visible_cache.get(id) {
            return v;
        }
        let state = self.data.borrow().get_state(*id).clone();
        let v = self.inner.is_visible(store, &state);
        self.data.borrow_mut().visible_cache.insert(*id, v);
        v
    }

    fn transitions(&self, store: &mut spp::SPPstore, id: &usize) -> Vec<(spp::SPP, usize)> {
        if let Some(cached) = self.data.borrow().transitions_cache.get(id) {
            return cached.clone();
        }
        let state = self.data.borrow().get_state(*id).clone();
        let raw = self.inner.transitions(store, &state);
        let mut result = Vec::with_capacity(raw.len());
        let mut data = self.data.borrow_mut();
        for (spp, next) in raw {
            let next_id = data.get_or_insert(next);
            result.push((spp, next_id));
        }
        data.transitions_cache.insert(*id, result.clone());
        result
    }

    fn output(&self, store: &mut spp::SPPstore, id: &usize) -> spp::SPP {
        if let Some(&spp) = self.data.borrow().output_cache.get(id) {
            return spp;
        }
        let state = self.data.borrow().get_state(*id).clone();
        let spp = self.inner.output(store, &state);
        self.data.borrow_mut().output_cache.insert(*id, spp);
        spp
    }
}

impl<T: NFA> NFA for Memo<T> {}
impl<T: DFA> DFA for Memo<T> {}

// ---- EpsilonClosure<T> -----------------------------------------------------

/// Wraps an `ENFA` and presents it as an `NFA` by collapsing epsilon paths.
///
/// The `State` type is unchanged, but the wrapper is only intended to be
/// queried at *visible* states of the underlying ENFA -- invisible states are
/// the ones that `epsilon_closure` walks through to produce the NFA's
/// transitions and outputs.
pub struct EpsilonClosure<T: ENFA> {
    inner: T,
    memo: RefCell<HashMap<T::State, Vec<(spp::SPP, T::State)>>>,
}

/// Insert `(key, spp)` into `map`, unioning the SPP with any existing entry.
/// Used inside the closure construction to keep `(state, spp)` lists free of
/// duplicate target states -- two paths to the same `v` collapse into one
/// entry whose SPP is the union of theirs.
fn union_into<S: Eq + Hash>(
    map: &mut HashMap<S, spp::SPP>,
    store: &mut spp::SPPstore,
    key: S,
    spp: spp::SPP,
) {
    let zero = store.zero;
    let entry = map.entry(key).or_insert(zero);
    *entry = store.union(*entry, spp);
}

impl<T: ENFA> EpsilonClosure<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            memo: RefCell::new(HashMap::new()),
        }
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns the ε-closure of `q`: a list of `(spp, p)` pairs where `p` is
    /// any state reachable from `q` along a path whose intermediate states
    /// (i.e., all but the last) are invisible.  `p` itself can be visible
    /// (in which case the path's last step is a "consumption" into a visible
    /// state) or invisible (a pure ε-path).  `(one, q)` is always included
    /// for the length-zero path.
    ///
    /// This closure is an internal helper; the trait methods filter it to
    /// the appropriate visibility for their needs (visible-only for
    /// transitions and start, invisible-only for the extra term in output).
    ///
    /// Results are memoized per state.
    fn epsilon_closure(
        &self,
        store: &mut spp::SPPstore,
        q: &T::State,
    ) -> Vec<(spp::SPP, T::State)> {
        if let Some(cached) = self.memo.borrow().get(q) {
            return cached.clone();
        }

        // BFS to find every state whose closure we need to solve for as part
        // of the linear system: `q` itself plus every invisible state
        // reachable via inner transitions to invisibles, except those that
        // are already memoized (we splice their cached closures in instead).
        let mut nodes_set: HashSet<T::State> = HashSet::new();
        let mut frontier: Vec<T::State> = vec![q.clone()];
        nodes_set.insert(q.clone());
        while let Some(s) = frontier.pop() {
            for (_, q_next) in self.inner.transitions(store, &s) {
                if self.inner.is_visible(store, &q_next) {
                    continue;
                }
                if self.memo.borrow().contains_key(&q_next) {
                    continue;
                }
                if nodes_set.insert(q_next.clone()) {
                    frontier.push(q_next);
                }
            }
        }

        let nodes: Vec<T::State> = nodes_set.into_iter().collect();
        let n = nodes.len();
        let index: HashMap<T::State, usize> = nodes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();

        // For each node `s_i` in our subgraph:
        //  * `consts[i]` starts with the self-entry `(one, s_i)` (length-zero
        //    path), and also absorbs:
        //    - direct visible neighbours `(spp, q_next_visible)` — those are
        //      length-1 paths ending with one consumption;
        //    - the spliced closures of any already-memoized invisible
        //      neighbours.
        //  * `edges[i][j]` is the SPP coefficient on the ε-edge from `s_i`
        //    to the unknown invisible neighbour `s_j`, unioned across
        //    multiple inner transitions.
        let mut consts: Vec<HashMap<T::State, spp::SPP>> = vec![HashMap::new(); n];
        let mut edges: Vec<Vec<spp::SPP>> = vec![vec![store.zero; n]; n];
        for i in 0..n {
            let s_i = nodes[i].clone();
            union_into(&mut consts[i], store, s_i.clone(), store.one);
            for (spp, q_next) in self.inner.transitions(store, &s_i) {
                if self.inner.is_visible(store, &q_next) {
                    union_into(&mut consts[i], store, q_next, spp);
                } else if let Some(memoized) = self.memo.borrow().get(&q_next).cloned() {
                    for (spp_inner, v) in memoized {
                        let combined = store.sequence(spp, spp_inner);
                        union_into(&mut consts[i], store, v, combined);
                    }
                } else {
                    let j = index[&q_next];
                    edges[i][j] = store.union(edges[i][j], spp);
                }
            }
        }

        // Forward pass: upper-triangularize via Arden's lemma.
        for k in 0..n {
            let l_star = store.star(edges[k][k]);

            // consts[k] := l_star · consts[k]
            let old = std::mem::take(&mut consts[k]);
            for (v, spp) in old {
                let new_spp = store.sequence(l_star, spp);
                union_into(&mut consts[k], store, v, new_spp);
            }

            // edges[k][j] := l_star · edges[k][j] for j > k
            for j in (k + 1)..n {
                edges[k][j] = store.sequence(l_star, edges[k][j]);
            }

            let consts_k_snap: Vec<(T::State, spp::SPP)> =
                consts[k].iter().map(|(v, &s)| (v.clone(), s)).collect();
            let edges_k_snap: Vec<spp::SPP> = edges[k].clone();

            for i in (k + 1)..n {
                let m = edges[i][k];
                if m == store.zero {
                    continue;
                }
                for (v, spp) in &consts_k_snap {
                    let new_spp = store.sequence(m, *spp);
                    union_into(&mut consts[i], store, v.clone(), new_spp);
                }
                for j in (k + 1)..n {
                    let new_edge = store.sequence(m, edges_k_snap[j]);
                    edges[i][j] = store.union(edges[i][j], new_edge);
                }
            }
        }

        // Backward pass: back-substitute.
        let mut closures: Vec<HashMap<T::State, spp::SPP>> = vec![HashMap::new(); n];
        for k in (0..n).rev() {
            let mut cl = std::mem::take(&mut consts[k]);
            for j in (k + 1)..n {
                let m = edges[k][j];
                if m == store.zero {
                    continue;
                }
                let entries: Vec<(T::State, spp::SPP)> =
                    closures[j].iter().map(|(v, &s)| (v.clone(), s)).collect();
                for (v, spp) in entries {
                    let new_spp = store.sequence(m, spp);
                    union_into(&mut cl, store, v, new_spp);
                }
            }
            closures[k] = cl;
        }

        // Memoize every node's closure (in deduped Vec form) and return q's.
        let q_idx = index[q];
        let mut memo = self.memo.borrow_mut();
        let mut q_closure: Vec<(spp::SPP, T::State)> = Vec::new();
        for (i, cl_map) in closures.into_iter().enumerate() {
            let cl_vec: Vec<(spp::SPP, T::State)> =
                cl_map.into_iter().map(|(v, s)| (s, v)).collect();
            if i == q_idx {
                q_closure = cl_vec.clone();
            }
            memo.insert(nodes[i].clone(), cl_vec);
        }
        q_closure
    }
}

impl<T: ENFA> ENFA for EpsilonClosure<T> {
    type State = T::State;

    /// Initial states: the ε-closure of every underlying initial state,
    /// filtered to visible.  Invisible states are kept out of the interface.
    fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, T::State)> {
        let inner_starts = self.inner.start(store);
        let mut result: HashMap<T::State, spp::SPP> = HashMap::new();
        for (spp, q) in inner_starts {
            for (path, p) in self.epsilon_closure(store, &q) {
                if !self.inner.is_visible(store, &p) {
                    continue;
                }
                let combined = store.sequence(spp, path);
                union_into(&mut result, store, p, combined);
            }
        }
        result.into_iter().map(|(v, s)| (s, v)).collect()
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &T::State) -> bool {
        true
    }

    /// For a visible `q`: take one underlying transition, ε-close the target,
    /// and discard non-visible states.  The combined SPP is the relational
    /// composition of the transition and the closure path.
    fn transitions(&self, store: &mut spp::SPPstore, q: &T::State) -> Vec<(spp::SPP, T::State)> {
        debug_assert!(
            self.inner.is_visible(store, q),
            "EpsilonClosure::transitions only accepts visible states",
        );
        let mut result: HashMap<T::State, spp::SPP> = HashMap::new();
        for (spp_t, q_next) in self.inner.transitions(store, q) {
            for (path, v) in self.epsilon_closure(store, &q_next) {
                if !self.inner.is_visible(store, &v) {
                    continue;
                }
                let combined = store.sequence(spp_t, path);
                union_into(&mut result, store, v, combined);
            }
        }
        result.into_iter().map(|(v, s)| (s, v)).collect()
    }

    /// For a visible `q`: union of `inner.output(q)` and, for every
    /// `(spp, q_inv)` in the ε-closure with `q_inv` *invisible*,
    /// `spp ; inner.output(q_inv)`.  Captures both terminating directly at
    /// `q` and ε-walking to an invisible state and terminating there.
    fn output(&self, store: &mut spp::SPPstore, q: &T::State) -> spp::SPP {
        debug_assert!(
            self.inner.is_visible(store, q),
            "EpsilonClosure::output only accepts visible states",
        );
        let mut result = self.inner.output(store, q);
        for (spp, p) in self.epsilon_closure(store, q) {
            if self.inner.is_visible(store, &p) {
                continue;
            }
            let inner_out = self.inner.output(store, &p);
            let composed = store.sequence(spp, inner_out);
            result = store.union(result, composed);
        }
        result
    }
}

impl<T: ENFA> NFA for EpsilonClosure<T> {}

// ---- SubsetDfa<T> ----------------------------------------------------------

/// Symbolic subset construction: turns an NFA `T` into a DFA whose state is
/// a *set* of `T`'s states (paired with a per-element "pending" SPP that
/// carries the start-spp through the first transition; on every later
/// transition this collapses to identity).
///
/// Each generated DFA transition partitions the `(in, out)` packet space
/// into regions of constant reachable subset, using SPP intersect/difference,
/// so transitions are SPP-disjoint by construction.  The result has a single
/// start state, one outgoing transition per non-empty target subset, and
/// `output` defined as the union of `spp_pre ; inner.output(q)` over the
/// set's members.
///
/// Requires `T::State: Ord` so the subset can be canonicalized as a
/// `BTreeSet`, which is what makes the `Hash + Eq` `State` key well-defined.
pub struct SubsetDfa<T: NFA>
where
    T::State: Ord,
{
    inner: T,
}

impl<T: NFA> SubsetDfa<T>
where
    T::State: Ord,
{
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T: NFA> ENFA for SubsetDfa<T>
where
    T::State: Ord,
{
    type State = BTreeSet<(spp::SPP, T::State)>;

    fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, Self::State)> {
        // Dedupe the inner start by NFA state, unioning the SPPs of duplicate
        // entries -- the DFA's start is then a single (one, set) pair.
        let inner_starts = self.inner.start(store);
        let zero = store.zero;
        let mut by_state: HashMap<T::State, spp::SPP> = HashMap::new();
        for (spp, q) in inner_starts {
            let entry = by_state.entry(q).or_insert(zero);
            *entry = store.union(*entry, spp);
        }
        let s: BTreeSet<(spp::SPP, T::State)> =
            by_state.into_iter().map(|(q, spp)| (spp, q)).collect();
        vec![(store.one, s)]
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
        true
    }

    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)> {
        // Aggregate transitions from the subset, indexed by inner target:
        //   spp_to[q'] = ⋃ over (spp_pre, q) ∈ S, (spp_t, q') ∈ T.transitions(q)
        //                  of sequence(spp_pre, spp_t).
        let zero = store.zero;
        let mut spp_to: HashMap<T::State, spp::SPP> = HashMap::new();
        for (spp_pre, inner_q) in q {
            for (spp_t, q_next) in self.inner.transitions(store, inner_q) {
                let composed = store.sequence(*spp_pre, spp_t);
                if composed == zero {
                    continue;
                }
                let entry = spp_to.entry(q_next).or_insert(zero);
                *entry = store.union(*entry, composed);
            }
        }

        // Iteratively partition the (in, out) space into disjoint regions
        // with constant reachable set: for each `(q', spp_q')` we split each
        // existing region by whether it overlaps `spp_q'`.
        let mut parts: Vec<(spp::SPP, BTreeSet<T::State>)> = vec![(store.top, BTreeSet::new())];
        for (q_next, spp_q) in spp_to {
            let mut new_parts: Vec<(spp::SPP, BTreeSet<T::State>)> =
                Vec::with_capacity(parts.len() * 2);
            for (region, set) in parts {
                let in_region = store.intersect(region, spp_q);
                let out_region = store.difference(region, spp_q);
                if in_region != zero {
                    let mut new_set = set.clone();
                    new_set.insert(q_next.clone());
                    new_parts.push((in_region, new_set));
                }
                if out_region != zero {
                    new_parts.push((out_region, set));
                }
            }
            parts = new_parts;
        }

        let one = store.one;
        parts
            .into_iter()
            .filter(|(_, set)| !set.is_empty())
            .map(|(spp, set)| {
                let new_state: BTreeSet<(spp::SPP, T::State)> =
                    set.into_iter().map(|q| (one, q)).collect();
                (spp, new_state)
            })
            .collect()
    }

    fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
        let mut result = store.zero;
        for (spp_pre, inner_q) in q {
            let inner_out = self.inner.output(store, inner_q);
            let composed = store.sequence(*spp_pre, inner_out);
            result = store.union(result, composed);
        }
        result
    }
}

impl<T: NFA> NFA for SubsetDfa<T> where T::State: Ord {}
impl<T: NFA> DFA for SubsetDfa<T> where T::State: Ord {}

// ---- ExplicitDFA -----------------------------------------------------------

/// A DFA stored as flat tables: `transitions[i]` and `outputs[i]` describe
/// state `i`, indexed by dense `usize`.  As a true DFA there is exactly one
/// start state (with implicit identity start SPP).  Fields are public so
/// callers can avoid going through the trait when raw access is needed.
pub struct ExplicitDFA {
    pub start: usize,
    pub transitions: Vec<Vec<(spp::SPP, usize)>>,
    pub outputs: Vec<spp::SPP>,
}

impl ExplicitDFA {
    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }
}

impl ENFA for ExplicitDFA {
    type State = usize;

    fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, usize)> {
        vec![(store.one, self.start)]
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &usize) -> bool {
        true
    }

    fn transitions(&self, _store: &mut spp::SPPstore, q: &usize) -> Vec<(spp::SPP, usize)> {
        self.transitions[*q].clone()
    }

    fn output(&self, _store: &mut spp::SPPstore, q: &usize) -> spp::SPP {
        self.outputs[*q]
    }
}

impl NFA for ExplicitDFA {}
impl DFA for ExplicitDFA {}

// ---- Aut conversion --------------------------------------------------------

/// Materializes the DFA reachable from `start` in `aut` as an `ExplicitDFA`
/// with dense `usize` state IDs.
///
/// BFSes through `aut`'s state graph, mapping each Aut state ID to a fresh
/// dense ID and copying out its transitions and output SPP.
pub fn aut_to_dfa(aut: &mut crate::aut::Aut, start: usize) -> ExplicitDFA {
    let mut state_to_id: HashMap<usize, usize> = HashMap::new();
    let mut id_to_state: Vec<usize> = Vec::new();

    state_to_id.insert(start, 0);
    id_to_state.push(start);

    let mut transitions: Vec<Vec<(spp::SPP, usize)>> = Vec::new();
    let mut outputs: Vec<spp::SPP> = Vec::new();

    let mut i = 0;
    while i < id_to_state.len() {
        let state = id_to_state[i];
        let st = aut.delta(state);
        let mut trans = Vec::new();
        for (&s, &spp) in st.get_transitions().iter() {
            let id = match state_to_id.get(&s) {
                Some(&id) => id,
                None => {
                    let id = id_to_state.len();
                    id_to_state.push(s);
                    state_to_id.insert(s, id);
                    id
                }
            };
            trans.push((spp, id));
        }
        let out = aut.epsilon(state);
        transitions.push(trans);
        outputs.push(out);
        i += 1;
    }

    ExplicitDFA {
        start: 0,
        transitions,
        outputs,
    }
}

// ---- Algorithms -------------------------------------------------------------

/// Returns `true` if no (input, trace, output) triple is accepted by `aut`.
pub fn is_empty<A: ENFA>(aut: &A, store: &mut spp::SPPstore) -> bool {
    is_empty_with_reachable(aut, store, &mut HashMap::new())
}

/// Same as [`is_empty`], but also populates `reachable` with a map from each
/// visited state to the SP of packets that have been verified to reach it.
///
/// When the search runs to a fixed point (returns `true`), `reachable[q]` is
/// the full set of packets that can reach state `q`.  On early termination
/// (returns `false` because an accepting output was found), the map reflects
/// partial progress at the moment the search stopped.
pub fn is_empty_with_reachable<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
    reachable: &mut HashMap<A::State, sp::SP>,
) -> bool {
    let starts = aut.start(store);
    let mut todo: HashMap<A::State, sp::SP> = HashMap::new();
    let mut worklist: Vec<A::State> = Vec::new();

    for (spp, q) in starts {
        let init_sp = store.push(store.sp.one, spp);
        if init_sp == store.sp.zero {
            continue;
        }
        let prev = todo.get(&q).copied().unwrap_or(store.sp.zero);
        let new_val = store.sp.union(prev, init_sp);
        if new_val != prev {
            todo.insert(q.clone(), new_val);
            worklist.push(q);
        }
    }

    while let Some(q) = worklist.pop() {
        let Some(todo_q) = todo.remove(&q) else {
            continue;
        };
        let done_q = reachable.get(&q).copied().unwrap_or(store.sp.zero);
        let diff = store.sp.difference(todo_q, done_q);

        if diff == store.sp.zero {
            continue;
        }

        let output_spp = aut.output(store, &q);
        let output_sp = store.push(diff, output_spp);
        if output_sp != store.sp.zero {
            return false;
        }

        for (spp, q_next) in aut.transitions(store, &q) {
            let reach = store.push(diff, spp);
            if reach != store.sp.zero {
                let prev = todo.get(&q_next).copied().unwrap_or(store.sp.zero);
                let new_val = store.sp.union(prev, reach);
                if new_val != prev {
                    todo.insert(q_next.clone(), new_val);
                    worklist.push(q_next);
                }
            }
        }

        let new_done = store.sp.union(done_q, diff);
        reachable.insert(q, new_done);
    }

    true
}

struct Step<S> {
    state: S,
    diff: sp::SP,
    predecessors: Vec<usize>,
}

/// Returns one concrete `(input, trace, output)` triple accepted by `aut`,
/// or `None` if the automaton is empty.
///
/// `trace` is the sequence of current packets at each state visited after
/// the start state, ending with the current packet at the accepting state.
pub fn get_any_trace<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
) -> Option<(Vec<bool>, Vec<Vec<bool>>, Vec<bool>)> {
    get_any_trace_with_states(aut, store).map(|(input, path, out)| {
        let trace: Vec<Vec<bool>> = path.into_iter().skip(1).map(|(_, p)| p).collect();
        (input, trace, out)
    })
}

/// Like [`get_any_trace`] but the middle component is the full sequence of
/// `(state, current_packet)` pairs visited, starting with the start state.
/// `path[0]` is `(q_start, p_start)` (no consumption yet), and each later
/// `path[i]` is `(q_i, p_i)` reached after consuming the i-th trace packet.
/// The packet trace returned by `get_any_trace` is `path[1..].iter().map(|(_, p)| p)`.
pub fn get_any_trace_with_states<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
) -> Option<(Vec<bool>, Vec<(A::State, Vec<bool>)>, Vec<bool>)> {
    let starts = aut.start(store);
    let mut todo: HashMap<A::State, sp::SP> = HashMap::new();
    let mut done: HashMap<A::State, sp::SP> = HashMap::new();
    let mut worklist: Vec<A::State> = Vec::new();
    let mut additions: HashMap<A::State, Vec<usize>> = HashMap::new();
    let mut steps: Vec<Step<A::State>> = Vec::new();

    // For each start state, track the union of start SPPs leading there;
    // used during trace reconstruction to recover an input.
    let mut start_spps: HashMap<A::State, spp::SPP> = HashMap::new();
    for (spp, q) in &starts {
        let entry = start_spps.entry(q.clone()).or_insert(store.zero);
        *entry = store.union(*entry, *spp);
    }

    for (spp, q) in starts {
        let init_sp = store.push(store.sp.one, spp);
        if init_sp == store.sp.zero {
            continue;
        }
        let prev = todo.get(&q).copied().unwrap_or(store.sp.zero);
        let new_val = store.sp.union(prev, init_sp);
        if new_val != prev {
            todo.insert(q.clone(), new_val);
            worklist.push(q);
        }
    }

    while let Some(q) = worklist.pop() {
        let preds = additions.remove(&q).unwrap_or_default();

        let Some(todo_q) = todo.remove(&q) else {
            continue;
        };
        let done_q = done.get(&q).copied().unwrap_or(store.sp.zero);
        let diff = store.sp.difference(todo_q, done_q);

        if diff == store.sp.zero {
            continue;
        }

        let step_idx = steps.len();
        steps.push(Step {
            state: q.clone(),
            diff,
            predecessors: preds,
        });

        let output_spp = aut.output(store, &q);
        let output_sp = store.push(diff, output_spp);
        if output_sp != store.sp.zero {
            return Some(reconstruct_trace(
                &steps,
                step_idx,
                aut,
                store,
                output_spp,
                &start_spps,
            ));
        }

        for (spp, q_next) in aut.transitions(store, &q) {
            let reachable = store.push(diff, spp);
            if reachable != store.sp.zero {
                let prev = todo.get(&q_next).copied().unwrap_or(store.sp.zero);
                let new_val = store.sp.union(prev, reachable);
                if new_val != prev {
                    todo.insert(q_next.clone(), new_val);
                    additions.entry(q_next.clone()).or_default().push(step_idx);
                    worklist.push(q_next);
                }
            }
        }

        let prev_done = done.get(&q).copied().unwrap_or(store.sp.zero);
        let new_done = store.sp.union(prev_done, diff);
        done.insert(q, new_done);
    }

    None
}

fn reconstruct_trace<A: ENFA>(
    steps: &[Step<A::State>],
    accepting_idx: usize,
    aut: &A,
    store: &mut spp::SPPstore,
    output_spp: spp::SPP,
    start_spps: &HashMap<A::State, spp::SPP>,
) -> (Vec<bool>, Vec<(A::State, Vec<bool>)>, Vec<bool>) {
    // At the accepting step, pick a current packet that's in the accepting
    // step's diff AND has at least one output through `output_spp`, then
    // sample one matching output packet for it.
    let acc_diff = steps[accepting_idx].diff;
    let bwd_output = store.bwd(output_spp);
    let valid_inputs = store.sp.intersect(acc_diff, bwd_output);
    let acc_pkt = store
        .sp
        .random_packet(valid_inputs)
        .expect("non-zero output_sp implies a valid input exists");
    let out_pkt = store
        .random_output_packet_from_input(output_spp, acc_pkt.clone())
        .expect("acc_pkt has an output by construction");

    let mut packets_back: Vec<Vec<bool>> = vec![acc_pkt.clone()];
    let mut states_back: Vec<A::State> = vec![steps[accepting_idx].state.clone()];
    let mut current_idx = accepting_idx;
    let mut current_pkt = acc_pkt;

    // Walk back until we reach an init step (one with no predecessors --
    // those are exactly the steps popped from the initial worklist).
    while !steps[current_idx].predecessors.is_empty() {
        let preds = steps[current_idx].predecessors.clone();
        let target = steps[current_idx].state.clone();
        let current_pkt_sp = singleton_sp(store, &current_pkt);

        let mut found: Option<(usize, Vec<bool>)> = None;
        'outer: for j in preds {
            let q_j = steps[j].state.clone();
            let d_j = steps[j].diff;

            for (spp, q_next) in aut.transitions(store, &q_j) {
                if q_next != target {
                    continue;
                }
                let valid_inputs = store.pull(spp, current_pkt_sp);
                let candidates = store.sp.intersect(d_j, valid_inputs);
                if candidates != store.sp.zero {
                    let p_j = store
                        .sp
                        .random_packet(candidates)
                        .expect("non-zero SP must contain a packet");
                    found = Some((j, p_j));
                    break 'outer;
                }
            }
        }

        let (j, p_j) = match found {
            Some(x) => x,
            None => unreachable!("non-start step has no predecessor producing the current packet"),
        };
        packets_back.push(p_j.clone());
        states_back.push(steps[j].state.clone());
        current_idx = j;
        current_pkt = p_j;
    }

    // We're at an init step: `current_pkt` is the packet at the start state
    // after the start SPP was applied.  Recover an input that maps to this
    // current packet via the start SPP.
    let q_start = steps[current_idx].state.clone();
    let spp_start = *start_spps
        .get(&q_start)
        .expect("init step's state must be a start state");
    let current_pkt_sp = singleton_sp(store, &current_pkt);
    let inputs = store.pull(spp_start, current_pkt_sp);
    let input = store
        .sp
        .random_packet(inputs)
        .expect("packet at start state must have an input via spp_start");

    packets_back.reverse();
    states_back.reverse();
    let path: Vec<(A::State, Vec<bool>)> = states_back
        .into_iter()
        .zip(packets_back.into_iter())
        .collect();
    (input, path, out_pkt)
}

fn singleton_sp(store: &mut spp::SPPstore, packet: &[bool]) -> sp::SP {
    // Build the BDD bottom-up.  At each step `sp_val` is the singleton at the
    // current depth and `zero` is the all-rejecting SP at the same depth;
    // both must have matching depth or downstream operations break.
    let mut sp_val = sp::SP::new(1);
    let mut zero = sp::SP::new(0);
    for &bit in packet.iter().rev() {
        let new_sp = if bit {
            store.sp.mk(zero, sp_val)
        } else {
            store.sp.mk(sp_val, zero)
        };
        zero = store.sp.mk(zero, zero);
        sp_val = new_sp;
    }
    sp_val
}

/// SPP for "any input → output = `packet`": accepts `(p_in, p_out)` iff
/// `p_out == packet`, regardless of `p_in`.
fn force_output_spp(store: &mut spp::SPPstore, packet: &[bool]) -> spp::SPP {
    let mut allow = spp::SPP::new(1);
    let mut zero = spp::SPP::new(0);
    for &bit in packet.iter().rev() {
        let new_allow = if bit {
            store.mk(zero, allow, zero, allow)
        } else {
            store.mk(allow, zero, allow, zero)
        };
        zero = store.mk(zero, zero, zero, zero);
        allow = new_allow;
    }
    allow
}

/// SPP for "input = `packet`, output any": accepts `(p_in, p_out)` iff
/// `p_in == packet`, regardless of `p_out`.
fn filter_input_spp(store: &mut spp::SPPstore, packet: &[bool]) -> spp::SPP {
    let mut allow = spp::SPP::new(1);
    let mut zero = spp::SPP::new(0);
    for &bit in packet.iter().rev() {
        let new_allow = if bit {
            store.mk(zero, zero, allow, allow)
        } else {
            store.mk(allow, allow, zero, zero)
        };
        zero = store.mk(zero, zero, zero, zero);
        allow = new_allow;
    }
    allow
}

/// One-step ENFA helper used by [`elaborate_step`]: wraps an inner ENFA so
/// every state is *visible* (forcing each transition to consume a packet),
/// pins the start to a single `(spp, q_source)` pair, and gives a non-zero
/// output only at `q_dest` (filtered to `p_dest`).
struct ElaborateStepHelper<'a, T: ENFA> {
    inner: &'a T,
    start_list: Vec<(spp::SPP, T::State)>,
    end: T::State,
    end_output: spp::SPP,
    zero_spp: spp::SPP,
}

impl<'a, T: ENFA> ENFA for ElaborateStepHelper<'a, T> {
    type State = T::State;

    fn start(&self, _store: &mut spp::SPPstore) -> Vec<(spp::SPP, T::State)> {
        self.start_list.clone()
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &T::State) -> bool {
        true
    }

    fn transitions(&self, store: &mut spp::SPPstore, q: &T::State) -> Vec<(spp::SPP, T::State)> {
        self.inner.transitions(store, q)
    }

    fn output(&self, _store: &mut spp::SPPstore, q: &T::State) -> spp::SPP {
        if *q == self.end {
            self.end_output
        } else {
            self.zero_spp
        }
    }
}

/// Elaborate one step of an `EpsilonClosure<T>` trace into a path through
/// the underlying ENFA `T`, covering any invisible intermediates between
/// two visible boundary states.
///
/// Returns the sequence of `(state, packet)` pairs visited *after* the
/// source — so the source itself is **not** included, but the destination
/// (with packet `p_dest`) is the last entry.
///
/// The construction follows the strategy of building a tiny helper ENFA
/// (every state visible, pinned start, single accept) and running
/// [`get_any_trace`] on it, then forward-simulating the returned trace
/// through `inner` to recover the state sequence.
pub fn elaborate_step<T: ENFA>(
    inner: &T,
    store: &mut spp::SPPstore,
    q_source: &T::State,
    p_source: &[bool],
    q_dest: &T::State,
    p_dest: &[bool],
) -> Vec<(T::State, Vec<bool>)> {
    let start_spp = force_output_spp(store, p_source);
    let output_spp = filter_input_spp(store, p_dest);
    let zero_spp = store.zero;

    let helper = ElaborateStepHelper {
        inner,
        start_list: vec![(start_spp, q_source.clone())],
        end: q_dest.clone(),
        end_output: output_spp,
        zero_spp,
    };

    let trace_result = get_any_trace(&helper, store)
        .expect("elaborate_step: no inner ENFA path between source and dest");
    let (_input, pkt_trace, _output_pkt) = trace_result;

    let mut path: Vec<(T::State, Vec<bool>)> = Vec::new();
    let mut state = q_source.clone();
    let mut pkt: Vec<bool> = p_source.to_vec();
    for next_pkt in pkt_trace {
        let mut found: Option<T::State> = None;
        for (spp, q_next) in inner.transitions(store, &state) {
            if store.accepts(spp, &pkt, &next_pkt) {
                found = Some(q_next);
                break;
            }
        }
        let q_next = found
            .expect("forward simulation must find a transition matching the helper's trace step");
        state = q_next;
        pkt = next_pkt;
        path.push((state.clone(), pkt.clone()));
    }

    debug_assert!(state == *q_dest);
    debug_assert!(pkt == p_dest);

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only ENFA wrapper that overlays explicit visibility flags on an
    /// `ExplicitDFA` (whose states are normally all visible).
    struct RandomVisibility<'a> {
        inner: &'a ExplicitDFA,
        visible: Vec<bool>,
    }

    impl<'a> ENFA for RandomVisibility<'a> {
        type State = usize;
        fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, usize)> {
            vec![(store.one, self.inner.start)]
        }
        fn is_visible(&self, _store: &mut spp::SPPstore, q: &usize) -> bool {
            self.visible[*q]
        }
        fn transitions(&self, _store: &mut spp::SPPstore, q: &usize) -> Vec<(spp::SPP, usize)> {
            self.inner.transitions[*q].clone()
        }
        fn output(&self, _store: &mut spp::SPPstore, q: &usize) -> spp::SPP {
            self.inner.outputs[*q]
        }
    }

    impl<'a> NFA for RandomVisibility<'a> {}

    #[test]
    fn fuzz_epsilon_closure_emptiness() {
        // Emptiness is a property of the underlying transition graph and is
        // independent of which states are visible.  So wrapping an
        // ExplicitDFA with random visibility flags and pushing it through
        // EpsilonClosure should give the same emptiness verdict.
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 500;

        for trial in 0..max_trials {
            let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);

            let mut aut = crate::aut::Aut::new(num_fields);
            let state = aut.expr_to_state(&expr);

            let dfa = aut_to_dfa(&mut aut, state);

            let original_empty = is_empty(&dfa, aut.spp_store_mut());

            let n = dfa.num_states();
            let mut visible: Vec<bool> = (0..n).map(|_| rand::random::<bool>()).collect();
            // Keep the start state visible so the wrapper presents a valid NFA.
            visible[dfa.start] = true;

            let wrapper = RandomVisibility {
                inner: &dfa,
                visible,
            };
            let closure = EpsilonClosure::new(wrapper);

            let closure_empty = is_empty(&closure, aut.spp_store_mut());

            assert_eq!(
                original_empty, closure_empty,
                "emptiness mismatch on trial {} for expr {}: original={}, closure={}",
                trial, expr, original_empty, closure_empty
            );
        }
    }

    #[test]
    fn fuzz_elaborate_step() {
        // For each step of an EpsilonClosure trace, `elaborate_step` should
        // recover an inner-ENFA path from one visible state to the next,
        // possibly through invisible intermediates.  Concatenating these per-
        // step paths yields a full path through the inner ENFA, every step of
        // which must be a valid `inner.transitions` step.
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 200;

        for trial in 0..max_trials {
            let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);

            let mut aut = crate::aut::Aut::new(num_fields);
            let state = aut.expr_to_state(&expr);
            let dfa = aut_to_dfa(&mut aut, state);

            let n = dfa.num_states();
            if n == 0 {
                continue;
            }
            let mut visible: Vec<bool> = (0..n).map(|_| rand::random::<bool>()).collect();
            visible[dfa.start] = true;

            let wrapper = RandomVisibility {
                inner: &dfa,
                visible,
            };
            let closure = EpsilonClosure::new(wrapper);

            if is_empty(&closure, aut.spp_store_mut()) {
                continue;
            }

            let trace = get_any_trace_with_states(&closure, aut.spp_store_mut())
                .expect("non-empty closure should yield a trace");
            let (_input, visible_path, _output_pkt) = trace;

            // Elaborate each consecutive pair in the visible path.
            let mut full_path: Vec<(usize, Vec<bool>)> = vec![visible_path[0].clone()];
            for w in visible_path.windows(2) {
                let (q_curr, p_curr) = &w[0];
                let (q_next, p_next) = &w[1];
                let step_path = elaborate_step(
                    closure.inner(),
                    aut.spp_store_mut(),
                    q_curr,
                    p_curr,
                    q_next,
                    p_next,
                );
                full_path.extend(step_path);
            }

            // Every consecutive `(q, p) -> (q', p')` in the elaborated path
            // must be backed by a real inner transition matching the packets.
            for i in 0..full_path.len().saturating_sub(1) {
                let (q_a, p_a) = full_path[i].clone();
                let (q_b, p_b) = full_path[i + 1].clone();
                let trans = closure.inner().transitions(aut.spp_store_mut(), &q_a);
                let mut found = false;
                for (spp, q_n) in trans {
                    if q_n == q_b && aut.spp_store_mut().accepts(spp, &p_a, &p_b) {
                        found = true;
                        break;
                    }
                }
                assert!(
                    found,
                    "trial {} expr {}: elaborated step #{} ({}, {:?}) -> ({}, {:?}) is not a valid inner transition",
                    trial, expr, i, q_a, p_a, q_b, p_b,
                );
            }

            // The visible-path endpoints must coincide with the elaboration's.
            let (start_q, start_p) = &visible_path[0];
            let (end_q, end_p) = visible_path.last().unwrap();
            assert_eq!(*start_q, full_path.first().unwrap().0);
            assert_eq!(start_p, &full_path.first().unwrap().1);
            assert_eq!(*end_q, full_path.last().unwrap().0);
            assert_eq!(end_p, &full_path.last().unwrap().1);
        }
    }

    #[test]
    fn fuzz_subset_dfa() {
        // Wrap a random ExplicitDFA in random visibility + EpsilonClosure to
        // get a real NFA, then determinize it via SubsetDfa.  The DFA must:
        //   * be empty iff the NFA is empty;
        //   * have exactly one start (so dfa_start works);
        //   * accept its own random trace under both dfa_accepts and the
        //     underlying NFA's nfa_accepts.
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 200;

        for trial in 0..max_trials {
            let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);

            let mut aut = crate::aut::Aut::new(num_fields);
            let state = aut.expr_to_state(&expr);
            let dfa = aut_to_dfa(&mut aut, state);

            let n = dfa.num_states();
            if n == 0 {
                continue;
            }
            let mut visible: Vec<bool> = (0..n).map(|_| rand::random::<bool>()).collect();
            visible[dfa.start] = true;

            let wrapper = RandomVisibility {
                inner: &dfa,
                visible,
            };
            let nfa = EpsilonClosure::new(wrapper);
            let subset = SubsetDfa::new(nfa);

            let nfa_empty = is_empty(subset.inner(), aut.spp_store_mut());
            let subset_empty = is_empty(&subset, aut.spp_store_mut());
            assert_eq!(
                nfa_empty, subset_empty,
                "trial {} expr {}: emptiness mismatch (nfa={}, subset={})",
                trial, expr, nfa_empty, subset_empty
            );

            // dfa_start must succeed (single start invariant).
            let _ = subset.dfa_start(aut.spp_store_mut());

            if !subset_empty {
                let trace = get_any_trace(&subset, aut.spp_store_mut())
                    .expect("non-empty subset DFA should yield a trace");
                let (input, t, output) = trace;
                assert!(
                    subset.dfa_accepts(aut.spp_store_mut(), &input, &t, &output),
                    "trial {} expr {}: subset DFA rejects its own trace",
                    trial,
                    expr,
                );
                assert!(
                    subset
                        .inner()
                        .nfa_accepts(aut.spp_store_mut(), &input, &t, &output),
                    "trial {} expr {}: underlying NFA rejects the subset DFA's trace",
                    trial,
                    expr,
                );
            }
        }
    }

    #[test]
    fn fuzz_explicit_dfa() {
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 500;

        for trial in 0..max_trials {
            let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);

            let mut aut = crate::aut::Aut::new(num_fields);
            let state = aut.expr_to_state(&expr);

            let dfa = aut_to_dfa(&mut aut, state);

            let our_is_empty = is_empty(&dfa, aut.spp_store_mut());
            let aut_is_empty = aut.is_empty(state);

            assert_eq!(
                our_is_empty, aut_is_empty,
                "is_empty mismatch on trial {} for expr {}: ours={}, aut={}",
                trial, expr, our_is_empty, aut_is_empty
            );

            if !our_is_empty {
                let trace = get_any_trace(&dfa, aut.spp_store_mut());
                assert!(
                    trace.is_some(),
                    "expected a trace for non-empty automaton on trial {} for expr {}",
                    trial,
                    expr
                );
                let (input, trace_pkts, output) = trace.unwrap();
                let accepted_dfa =
                    dfa.dfa_accepts(aut.spp_store_mut(), &input, &trace_pkts, &output);
                assert!(
                    accepted_dfa,
                    "trace not accepted by dfa_accepts on trial {} for expr {}: input={:?}, trace={:?}, output={:?}",
                    trial, expr, input, trace_pkts, output
                );
                let accepted_nfa =
                    dfa.nfa_accepts(aut.spp_store_mut(), &input, &trace_pkts, &output);
                assert!(
                    accepted_nfa,
                    "trace not accepted by nfa_accepts on trial {} for expr {}: input={:?}, trace={:?}, output={:?}",
                    trial, expr, input, trace_pkts, output
                );
            }
        }
    }
}
