use crate::sp;
use crate::spp;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

/// Nondeterministic symbolic NetKAT automaton with epsilon transitions.
///
/// Every automaton has a single distinguished start state.  The first packet
/// of any accepted trace is the carry-on packet at the start state -- so
/// the start state is treated as visible at trace position 0 regardless of
/// `is_visible(start)`.  At later trace positions, if the run returns to
/// `start` and `is_visible(start)` is false, it is bypassed like any other
/// invisible state.  Non-start states obey `is_visible` as usual: visible
/// states consume one trace packet on entry, invisible (epsilon) states do
/// not advance the trace position.
///
/// All trait methods take an explicit `&mut spp::SPPstore`, which the
/// underlying automaton may use to compute SPPs on demand.  Implementations
/// that don't need the store may simply ignore it.
pub trait ENFA {
    type State: Clone + Eq + Hash + std::fmt::Debug;

    /// The unique start state.
    fn start(&self, store: &mut spp::SPPstore) -> Self::State;

    /// True for visible states (entering them consumes one packet from the trace);
    /// false for invisible (epsilon) states.  See the trait docs for how this
    /// interacts with the start state.
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
    /// Returns `true` iff the concrete `(trace, output)` pair is accepted by
    /// *some* path through the NFA.  `trace` must be non-empty: `trace[0]`
    /// is the carry-on packet at the start state, and each subsequent
    /// `trace[i]` is the packet at the next visited state.
    fn nfa_accepts(&self, store: &mut spp::SPPstore, trace: &[Vec<bool>], output: &[bool]) -> bool {
        if trace.is_empty() {
            return false;
        }

        let mut frontier: HashSet<Self::State> = HashSet::new();
        frontier.insert(self.start(store));
        let mut current_pkt: &[bool] = &trace[0];

        for next_pkt in &trace[1..] {
            let mut new_states: HashSet<Self::State> = HashSet::new();
            for q in &frontier {
                for (spp_t, q_next) in self.transitions(store, q) {
                    if store.accepts(spp_t, current_pkt, next_pkt) {
                        new_states.insert(q_next);
                    }
                }
            }
            if new_states.is_empty() {
                return false;
            }
            frontier = new_states;
            current_pkt = next_pkt;
        }

        for q in &frontier {
            let out_spp = self.output(store, q);
            if store.accepts(out_spp, current_pkt, output) {
                return true;
            }
        }
        false
    }

    /// Returns every `(state, packet)` pair reachable by walking some prefix
    /// of `trace` from the start.  `trace[0]` is the carry-on packet at the
    /// start state, and at trace position `i` the packet is `trace[i]`.
    /// Pairs are deduped by `(state, position)`.  An empty `trace` yields
    /// an empty result.
    fn reachable_from_trace<'a>(
        &self,
        store: &mut spp::SPPstore,
        trace: &'a [Vec<bool>],
    ) -> Vec<(Self::State, &'a [bool])> {
        if trace.is_empty() {
            return Vec::new();
        }

        let mut result: Vec<(Self::State, &'a [bool])> = Vec::new();
        let mut seen: HashSet<(Self::State, usize)> = HashSet::new();

        let q_start = self.start(store);
        let mut frontier: Vec<Self::State> = vec![q_start.clone()];
        if seen.insert((q_start.clone(), 0)) {
            result.push((q_start, &trace[0][..]));
        }
        let mut current_pkt: &[bool] = &trace[0];

        for (i, next_pkt) in trace[1..].iter().enumerate() {
            let mut step_seen: HashSet<Self::State> = HashSet::new();
            for q in &frontier {
                for (spp_t, q_next) in self.transitions(store, q) {
                    if store.accepts(spp_t, current_pkt, next_pkt) {
                        step_seen.insert(q_next);
                    }
                }
            }
            let new_frontier: Vec<Self::State> = step_seen.into_iter().collect();
            for q in &new_frontier {
                if seen.insert((q.clone(), i + 1)) {
                    result.push((q.clone(), &next_pkt[..]));
                }
            }
            frontier = new_frontier;
            current_pkt = &next_pkt[..];
        }

        result
    }
}

/// Deterministic symbolic NetKAT automaton.
///
/// For each state, the SPPs labeling outgoing transitions are pairwise disjoint.
pub trait DFA: NFA {
    /// Returns `true` iff the concrete `(trace, output)` pair is accepted.
    /// `trace` must be non-empty: `trace[0]` is the packet at the start
    /// state, and each transition consumes the next packet.
    fn dfa_accepts(&self, store: &mut spp::SPPstore, trace: &[Vec<bool>], output: &[bool]) -> bool {
        if trace.is_empty() {
            return false;
        }

        let mut current_state = self.start(store);
        let mut current_pkt: &[bool] = &trace[0];

        for next_pkt in &trace[1..] {
            let mut next_state = None;
            for (spp, q_next) in self.transitions(store, &current_state) {
                if store.accepts(spp, current_pkt, next_pkt) {
                    next_state = Some(q_next);
                    break;
                }
            }
            match next_state {
                Some(q) => {
                    current_state = q;
                    current_pkt = next_pkt;
                }
                None => return false,
            }
        }

        let out_spp = self.output(store, &current_state);
        store.accepts(out_spp, current_pkt, output)
    }
}

// ---- Reference impls -------------------------------------------------------

/// An SPP represents a dup-free NetKAT program, and dup-free NetKAT programs are NetKAT programs
/// too.
///
/// An SPP can be an automaton with just one state and no transitions :)
impl ENFA for spp::SPP {
    type State = ();

    #[expect(clippy::unused_unit)]
    fn start(&self, _store: &mut spp::SPPstore) -> Self::State {
        ()
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
        // This doesn't matter actually, since visibility only matters for states _after_ the start
        // state, and there are no transitions.
        //
        // However, the contract of the `NFA` trait is that `is_visible` must always return `true`,
        // so for that reason we have to return `true`.
        true
    }

    /// No transitions!
    fn transitions(
        &self,
        _store: &mut spp::SPPstore,
        _q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)> {
        vec![]
    }

    /// The output from the start state is the SPP itself
    fn output(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> spp::SPP {
        *self
    }
}

impl NFA for spp::SPP {}
impl DFA for spp::SPP {}

// Blanket impls so that `&T` is itself an ENFA / NFA / DFA whenever `T` is.
// Lets callers borrow automata into wrappers (`Complement<&T>`, `Union<&A, &B>`,
// etc.) instead of cloning them.

impl<T: ENFA> ENFA for &T {
    type State = T::State;

    fn start(&self, store: &mut spp::SPPstore) -> Self::State {
        <T as ENFA>::start(*self, store)
    }

    fn is_visible(&self, store: &mut spp::SPPstore, q: &Self::State) -> bool {
        <T as ENFA>::is_visible(*self, store, q)
    }

    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)> {
        <T as ENFA>::transitions(*self, store, q)
    }

    fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
        <T as ENFA>::output(*self, store, q)
    }
}

impl<T: NFA> NFA for &T {}
impl<T: DFA> DFA for &T {}

// ---- Memo<T> ----------------------------------------------------------------

#[derive(Clone)]
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
#[derive(Clone)]
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

    fn start(&self, store: &mut spp::SPPstore) -> usize {
        let raw = self.inner.start(store);
        self.data.borrow_mut().get_or_insert(raw)
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
#[derive(Clone)]
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

    /// The only states that should surface in the public API of `EpsilonClosure<T>` are:
    ///
    /// * The start state
    /// * The visible states
    pub fn assert_is_valid_state(&self, store: &mut spp::SPPstore, state: &T::State) {
        debug_assert!(self.inner.is_visible(store, state) || state == &self.start(store));
    }

    /// Given a trace through the closure NFA -- a sequence of
    /// `(visible_state, packet)` pairs plus the trace's final `output_pkt` --
    /// elaborate it into the full path through the underlying ENFA `T`,
    /// splicing in the invisible states the closure abstracted away.
    ///
    /// Invisible states can hide in two places:
    /// * *Between* two visible states.
    /// * *After* the last visible state: the closure's `output` at a visible
    ///   `q` sums in `inner.output(q_inv)` for every invisible `q_inv` in
    ///   `q`'s ε-closure, so the triple's output may actually be emitted at an
    ///   invisible state reachable from `q` via an ε-only path. By convention
    ///   the start state is always included, so there is no symmetric gap at
    ///   the *start* of the trace.
    pub fn elaborate_trace(
        &self,
        store: &mut spp::SPPstore,
        visible_path: &[(T::State, Vec<bool>)],
        output_pkt: &[bool],
    ) -> Vec<(T::State, Vec<bool>)> {
        assert!(!visible_path.is_empty());

        // `elaborate_step` fills the between-states gaps; `elaborate_tail`
        // recovers the trailing ε-path to where the output is emitted.
        let mut full: Vec<(T::State, Vec<bool>)> = Vec::new();
        full.push(visible_path[0].clone());
        for w in visible_path.windows(2) {
            let (q_curr, p_curr) = &w[0];
            let (q_next, p_next) = &w[1];
            let step = elaborate_step(self.inner(), store, q_curr, p_curr, q_next, p_next);
            full.extend(step);
        }

        let (q_last, p_last) = visible_path.last().unwrap();
        let tail = elaborate_tail(self.inner(), store, q_last, p_last, output_pkt);
        full.extend(tail);

        full
    }

    /// Returns the ε-closure of `q`: a list of `(spp, p)` pairs where `p` is
    /// `q` itself (with `spp = one`) or any *invisible* state reachable from
    /// `q` along an invisible-only path; `spp` is the composition of SPPs
    /// along that path.  Visible states other than `q` itself are
    /// deliberately not included -- reaching them would require a real
    /// "consumption" transition, which lies outside ε-closure.
    ///
    /// Used internally to drive `start`, `transitions`, and `output`:
    ///   * `start` filters this closure to visible (just `q` if visible);
    ///   * `transitions` "pre-closes" the source `q`, then takes one inner
    ///     transition into a visible target;
    ///   * `output` sums `inner.output(q)` with the closure's invisible-end
    ///     contributions.
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
        //    path) and absorbs the spliced closures of any already-memoized
        //    invisible neighbours.
        //  * `edges[i][j]` is the SPP coefficient on the ε-edge from `s_i`
        //    to the unknown invisible neighbour `s_j`, unioned across
        //    multiple inner transitions.
        // Visible neighbours are skipped: they aren't part of the ε-closure.
        let mut consts: Vec<HashMap<T::State, spp::SPP>> = vec![HashMap::new(); n];
        let mut edges: Vec<Vec<spp::SPP>> = vec![vec![store.zero; n]; n];
        for i in 0..n {
            let s_i = nodes[i].clone();
            union_into(&mut consts[i], store, s_i.clone(), store.one);
            for (spp, q_next) in self.inner.transitions(store, &s_i) {
                if self.inner.is_visible(store, &q_next) {
                    continue;
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
            for spp in &mut edges[k][(k + 1)..] {
                *spp = store.sequence(l_star, *spp);
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

    /// The wrapper's start is the inner's start, treated as visible at
    /// trace position 0 regardless of `inner.is_visible(start)` (per the
    /// new ENFA semantics).  Outgoing transitions and output absorb the
    /// ε-closure machinery for non-start invisibles.
    fn start(&self, store: &mut spp::SPPstore) -> T::State {
        self.inner.start(store)
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &T::State) -> bool {
        true
    }

    /// For a visible `q`: pre-close `q` to bring its ε-reachable invisibles
    /// into scope, then take one inner transition out of each, keeping only
    /// transitions that land on a visible target (the "consumption" step).
    /// The combined SPP is the closure path composed with the transition's.
    fn transitions(&self, store: &mut spp::SPPstore, q: &T::State) -> Vec<(spp::SPP, T::State)> {
        self.assert_is_valid_state(store, q);
        let mut result: HashMap<T::State, spp::SPP> = HashMap::new();
        for (path, p) in self.epsilon_closure(store, q) {
            for (spp_t, q_next) in self.inner.transitions(store, &p) {
                if !self.inner.is_visible(store, &q_next) {
                    continue;
                }
                let combined = store.sequence(path, spp_t);
                union_into(&mut result, store, q_next, combined);
            }
        }
        result.into_iter().map(|(v, s)| (s, v)).collect()
    }

    /// For a visible `q`: union of `inner.output(q)` and, for every
    /// `(spp, q_inv)` in the ε-closure with `q_inv` *invisible*,
    /// `spp ; inner.output(q_inv)`.  Captures both terminating directly at
    /// `q` and ε-walking to an invisible state and terminating there.
    fn output(&self, store: &mut spp::SPPstore, q: &T::State) -> spp::SPP {
        self.assert_is_valid_state(store, q);
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
#[derive(Clone)]
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

    fn start(&self, store: &mut spp::SPPstore) -> Self::State {
        let one = store.one;
        let q = self.inner.start(store);
        BTreeSet::from([(one, q)])
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

// ---- ops module ------------------------------------------------------------

/// Boolean operations on automata.
///
/// Each operation is a wrapper type that defers all real work to its
/// constituent automata; nothing is materialized eagerly.
pub mod ops {
    use super::*;

    // ---- Complement ------------------------------------------------------

    /// State of [`Complement`]: either an inner state, or a synthesized sink
    /// reached when the inner DFA's transitions don't cover the current
    /// `(in, out)` packet pair.  The sink loops to itself with `top` and has
    /// `top` as its output, so any packet pair in any extension of the trace
    /// is accepted by the complement once it's reached.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum CompState<S> {
        Inner(S),
        Sink,
    }

    /// Complement of a DFA: accepts exactly the triples the inner DFA rejects.
    /// Implements `DFA` (single start, disjoint transitions, total).
    #[derive(Clone)]
    pub struct Complement<T: DFA> {
        inner: T,
    }

    /// Build the complement of a DFA.
    pub fn complement<T: DFA>(inner: T) -> Complement<T> {
        Complement { inner }
    }

    impl<T: DFA> ENFA for Complement<T> {
        type State = CompState<T::State>;

        fn start(&self, store: &mut spp::SPPstore) -> Self::State {
            CompState::Inner(self.inner.start(store))
        }

        fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
            true
        }

        fn transitions(
            &self,
            store: &mut spp::SPPstore,
            q: &Self::State,
        ) -> Vec<(spp::SPP, Self::State)> {
            match q {
                CompState::Inner(s) => {
                    let inner_trans = self.inner.transitions(store, s);
                    let mut covered = store.zero;
                    let mut result: Vec<(spp::SPP, Self::State)> =
                        Vec::with_capacity(inner_trans.len() + 1);
                    for (spp, q_next) in inner_trans {
                        covered = store.union(covered, spp);
                        result.push((spp, CompState::Inner(q_next)));
                    }
                    let missing = store.difference(store.top, covered);
                    if missing != store.zero {
                        result.push((missing, CompState::Sink));
                    }
                    result
                }
                CompState::Sink => vec![(store.top, CompState::Sink)],
            }
        }

        fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
            match q {
                CompState::Inner(s) => {
                    let inner_out = self.inner.output(store, s);
                    store.complement(inner_out)
                }
                CompState::Sink => store.top,
            }
        }
    }

    impl<T: DFA> NFA for Complement<T> {}
    impl<T: DFA> DFA for Complement<T> {}

    // ---- Union -----------------------------------------------------------

    /// State of [`Union`]: a fresh `Start` that fans out into either side's
    /// start, plus tagged inner states.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum UnionState<L, R> {
        Start,
        Left(L),
        Right(R),
    }

    /// Union of two NFAs: accepts traces accepted by either side.  A fresh
    /// start state has the disjoint union of the two constituents' start
    /// states' transitions, plus the union of their outputs.
    #[derive(Clone)]
    pub struct Union<A: NFA, B: NFA> {
        left: A,
        right: B,
    }

    /// Build the union of two NFAs.
    pub fn union<A: NFA, B: NFA>(left: A, right: B) -> Union<A, B> {
        Union { left, right }
    }

    impl<A: NFA, B: NFA> ENFA for Union<A, B> {
        type State = UnionState<A::State, B::State>;

        fn start(&self, _store: &mut spp::SPPstore) -> Self::State {
            UnionState::Start
        }

        fn is_visible(&self, store: &mut spp::SPPstore, q: &Self::State) -> bool {
            match q {
                UnionState::Start => true,
                UnionState::Left(s) => self.left.is_visible(store, s),
                UnionState::Right(s) => self.right.is_visible(store, s),
            }
        }

        fn transitions(
            &self,
            store: &mut spp::SPPstore,
            q: &Self::State,
        ) -> Vec<(spp::SPP, Self::State)> {
            match q {
                UnionState::Start => {
                    let l_start = self.left.start(store);
                    let r_start = self.right.start(store);
                    let mut out: Vec<(spp::SPP, Self::State)> = self
                        .left
                        .transitions(store, &l_start)
                        .into_iter()
                        .map(|(spp, q)| (spp, UnionState::Left(q)))
                        .collect();
                    out.extend(
                        self.right
                            .transitions(store, &r_start)
                            .into_iter()
                            .map(|(spp, q)| (spp, UnionState::Right(q))),
                    );
                    out
                }
                UnionState::Left(s) => self
                    .left
                    .transitions(store, s)
                    .into_iter()
                    .map(|(spp, q)| (spp, UnionState::Left(q)))
                    .collect(),
                UnionState::Right(s) => self
                    .right
                    .transitions(store, s)
                    .into_iter()
                    .map(|(spp, q)| (spp, UnionState::Right(q)))
                    .collect(),
            }
        }

        fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
            match q {
                UnionState::Start => {
                    let l_start = self.left.start(store);
                    let r_start = self.right.start(store);
                    let l_out = self.left.output(store, &l_start);
                    let r_out = self.right.output(store, &r_start);
                    store.union(l_out, r_out)
                }
                UnionState::Left(s) => self.left.output(store, s),
                UnionState::Right(s) => self.right.output(store, s),
            }
        }
    }

    impl<A: NFA, B: NFA> NFA for Union<A, B> {}

    // ---- Intersection ----------------------------------------------------

    /// Intersection of two NFAs via the product construction: each state is
    /// a pair `(q_a, q_b)`, transitions are SPP-intersections of paired
    /// transitions, and output is SPP-intersection of paired outputs.
    /// Implements `NFA` always, and `DFA` when both inputs are `DFA`s
    /// (since the product preserves "single start" and "disjoint
    /// transitions").
    #[derive(Clone)]
    pub struct Intersection<A: NFA, B: NFA> {
        left: A,
        right: B,
    }

    /// Build the intersection of two NFAs.
    pub fn intersection<A: NFA, B: NFA>(left: A, right: B) -> Intersection<A, B> {
        Intersection { left, right }
    }

    impl<A: NFA, B: NFA> ENFA for Intersection<A, B> {
        type State = (A::State, B::State);

        fn start(&self, store: &mut spp::SPPstore) -> Self::State {
            (self.left.start(store), self.right.start(store))
        }

        fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
            true
        }

        fn transitions(
            &self,
            store: &mut spp::SPPstore,
            q: &Self::State,
        ) -> Vec<(spp::SPP, Self::State)> {
            let trans_a = self.left.transitions(store, &q.0);
            let trans_b = self.right.transitions(store, &q.1);
            let mut out: Vec<(spp::SPP, Self::State)> =
                Vec::with_capacity(trans_a.len() * trans_b.len());
            for (spp_a, q_a_next) in &trans_a {
                for (spp_b, q_b_next) in &trans_b {
                    let combined = store.intersect(*spp_a, *spp_b);
                    if combined != store.zero {
                        out.push((combined, (q_a_next.clone(), q_b_next.clone())));
                    }
                }
            }
            out
        }

        fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
            let out_a = self.left.output(store, &q.0);
            let out_b = self.right.output(store, &q.1);
            store.intersect(out_a, out_b)
        }
    }

    impl<A: NFA, B: NFA> NFA for Intersection<A, B> {}
    impl<A: DFA, B: DFA> DFA for Intersection<A, B> {}

    /// n-tuples of states from `T`
    pub struct Exponential<A> {
        pub inner: A,
    }

    impl<A: NFA> Exponential<A> {
        pub fn transitions(
            &self,
            store: &mut spp::SPPstore,
            q: &[A::State],
        ) -> Vec<(spp::SPP, Vec<A::State>)> {
            match q {
                [] => vec![(store.top, vec![])],
                [init @ .., last] => {
                    let trans_a = self.transitions(store, init);
                    let trans_b = self.inner.transitions(store, last);
                    let mut out = Vec::with_capacity(trans_a.len() * trans_b.len());
                    for &(spp_a, ref target_a) in &trans_a {
                        for &(spp_b, ref target_b) in &trans_b {
                            let mut target = target_a.clone();
                            target.push(target_b.clone());
                            out.push((store.intersect(spp_a, spp_b), target));
                        }
                    }
                    out
                }
            }
        }
    }

    // ---- NAryProduct -----------------------------------------------------

    /// Intersection-style product of an arbitrary number of DFAs.
    ///
    /// Like [`Intersection`], but n-ary and over a borrowed slice of (possibly
    /// distinct) DFAs rather than a fixed pair: a state is the vector of the
    /// inner states (`q[i]` belongs to `inner[i]`), transitions are the
    /// SPP-intersections of one transition drawn from each component, and the
    /// output is the SPP-intersection of all the component outputs.  The empty
    /// product accepts everything (its single state is `vec![]`, with `top`
    /// output and a single `top` self-loop).
    ///
    /// Always an `NFA`, and a `DFA` because each component is a `DFA` and the
    /// product preserves "single start" and "disjoint transitions".
    #[derive(Clone)]
    pub struct NAryProduct<'a, T: DFA> {
        pub inner: &'a [T],
    }

    impl<'a, T: DFA> ENFA for NAryProduct<'a, T> {
        type State = Vec<T::State>;

        fn start(&self, store: &mut spp::SPPstore) -> Self::State {
            self.inner.iter().map(|dfa| dfa.start(store)).collect()
        }

        fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
            true
        }

        fn transitions(
            &self,
            store: &mut spp::SPPstore,
            q: &Self::State,
        ) -> Vec<(spp::SPP, Self::State)> {
            // Fold the cartesian product across components, intersecting SPPs and
            // pruning branches that become `zero` as early as possible.
            let mut acc: Vec<(spp::SPP, Self::State)> = vec![(store.top, Vec::new())];
            for (dfa, q_i) in self.inner.iter().zip(q) {
                let trans_i = dfa.transitions(store, q_i);
                let mut next = Vec::with_capacity(acc.len() * trans_i.len());
                for (spp_acc, target_acc) in &acc {
                    for (spp_i, q_i_next) in &trans_i {
                        let combined = store.intersect(*spp_acc, *spp_i);
                        if combined != store.zero {
                            let mut target = target_acc.clone();
                            target.push(q_i_next.clone());
                            next.push((combined, target));
                        }
                    }
                }
                acc = next;
            }
            acc
        }

        fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
            let mut out = store.top;
            for (dfa, q_i) in self.inner.iter().zip(q) {
                let out_i = dfa.output(store, q_i);
                out = store.intersect(out, out_i);
            }
            out
        }
    }

    impl<'a, T: DFA> NFA for NAryProduct<'a, T> {}
    impl<'a, T: DFA> DFA for NAryProduct<'a, T> {}

    // ---- WithSinkState ---------------------------------------------------

    /// State of [`WithSinkState`]: an inner state, or the synthesized sink.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum SinkOr<S> {
        Inner(S),
        Sink,
    }

    /// Totalizes an arbitrary [`DFA`] by adding one global, absorbing **sink**
    /// state, so the transition function becomes total without changing the
    /// language.
    ///
    /// At each real state the inner edges already cover some set of `(in, out)`
    /// packet pairs; the leftover pairs (`top` minus their union) are routed to
    /// the sink, which loops to itself on `top` and is non-accepting (`zero`
    /// output).
    #[derive(Clone)]
    pub struct WithSinkState<T>(pub T);

    impl<T: DFA> ENFA for WithSinkState<T> {
        type State = SinkOr<T::State>;

        fn start(&self, store: &mut spp::SPPstore) -> Self::State {
            SinkOr::Inner(self.0.start(store))
        }

        fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
            true
        }

        fn transitions(
            &self,
            store: &mut spp::SPPstore,
            q: &Self::State,
        ) -> Vec<(spp::SPP, Self::State)> {
            match q {
                SinkOr::Inner(s) => {
                    let inner_trans = self.0.transitions(store, s);
                    let mut covered = store.zero;
                    let mut result: Vec<(spp::SPP, Self::State)> =
                        Vec::with_capacity(inner_trans.len() + 1);
                    for (spp, q_next) in inner_trans {
                        covered = store.union(covered, spp);
                        result.push((spp, SinkOr::Inner(q_next)));
                    }
                    let missing = store.difference(store.top, covered);
                    if missing != store.zero {
                        result.push((missing, SinkOr::Sink));
                    }
                    result
                }
                SinkOr::Sink => vec![(store.top, SinkOr::Sink)],
            }
        }

        fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
            match q {
                SinkOr::Inner(s) => self.0.output(store, s),
                SinkOr::Sink => store.zero,
            }
        }
    }

    impl<T: DFA> NFA for WithSinkState<T> {}
    impl<T: DFA> DFA for WithSinkState<T> {}
}

// ---- ExplicitDFA -----------------------------------------------------------

/// A DFA stored as flat tables: `transitions[i]` and `outputs[i]` describe
/// state `i`, indexed by dense `usize`.  As a true DFA there is exactly one
/// start state (with implicit identity start SPP).  Fields are public so
/// callers can avoid going through the trait when raw access is needed.
#[derive(Debug, Clone)]
pub struct ExplicitDFA {
    pub start: usize,
    pub transitions: Vec<Vec<(spp::SPP, usize)>>,
    pub outputs: Vec<spp::SPP>,
}

impl ExplicitDFA {
    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }

    /// Materialize any [`DFA`] as a dense `ExplicitDFA` reachable from its start.
    ///
    /// The DFA is wrapped in a [`Memo`] (which assigns dense `usize` IDs to its
    /// states and caches their transitions/outputs), then explored by the same
    /// BFS-over-dense-IDs used in [`aut_to_dfa`]: the start state is ID `0`, and
    /// each `transitions` call hands out fresh contiguous IDs for any newly seen
    /// states, so processing IDs in increasing order visits the whole reachable
    /// DFA exactly once.
    pub fn from_dfa<T: DFA>(store: &mut spp::SPPstore, dfa: T) -> ExplicitDFA {
        let memo = Memo::new(dfa);

        let start = memo.start(store);
        debug_assert_eq!(start, 0);

        let mut transitions: Vec<Vec<(spp::SPP, usize)>> = Vec::new();
        let mut outputs: Vec<spp::SPP> = Vec::new();
        let mut num_states = 1;

        let mut i = 0;
        while i < num_states {
            let trans = memo.transitions(store, &i);
            for &(_, next) in &trans {
                num_states = num_states.max(next + 1);
            }
            let out = memo.output(store, &i);
            transitions.push(trans);
            outputs.push(out);
            i += 1;
        }

        ExplicitDFA {
            start,
            transitions,
            outputs,
        }
    }

    /// Build the n-ary intersection [`product`](ops::NAryProduct) of `dfas` and
    /// materialize it as a dense `ExplicitDFA` via [`from_dfa`](Self::from_dfa).
    pub fn product<T: DFA>(store: &mut spp::SPPstore, dfas: &[T]) -> ExplicitDFA {
        ExplicitDFA::from_dfa(store, ops::NAryProduct { inner: dfas })
    }
}

impl ENFA for ExplicitDFA {
    type State = usize;

    fn start(&self, _store: &mut spp::SPPstore) -> usize {
        self.start
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

/// Convert a [`crate::expr::Expr`] into a DFA.
///
/// A helper function which wraps [`crate::aut`].
pub fn expr_to_dfa(expr: &crate::expr::Expr, store: &mut spp::SPPstore) -> ExplicitDFA {
    // Take the SPP store temporarily (we need full ownership)
    let fake_store = spp::SPPstore::new(0);
    let mut aut = crate::aut::Aut::from_spp_store(std::mem::replace(store, fake_store));

    let start_state = aut.expr_to_state(expr);
    let dfa = aut_to_dfa(&mut aut, start_state);

    // Put the SPP store back
    std::mem::swap(store, aut.spp_store_mut());

    dfa
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
    let q_start = aut.start(store);
    let mut todo: HashMap<A::State, sp::SP> = HashMap::new();
    let mut worklist: Vec<A::State> = Vec::new();

    // The start state is reachable with any input packet.
    todo.insert(q_start.clone(), store.sp.one);
    worklist.push(q_start);

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

/// Returns one concrete `(trace, output)` pair accepted by `aut`, or `None`
/// if the automaton is empty.  `trace[0]` is the carry-on packet at the
/// start state and the trace is non-empty when `Some`.
pub fn get_any_trace<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
) -> Option<(Vec<Vec<bool>>, Vec<bool>)> {
    get_any_trace_with_states(aut, store).map(|(path, out)| {
        let trace: Vec<Vec<bool>> = path.into_iter().map(|(_, p)| p).collect();
        (trace, out)
    })
}

/// Like [`get_any_trace`] but returns the full sequence of
/// `(state, current_packet)` pairs visited, starting with the start state.
/// `path[0]` is `(q_start, p_start)` and `path[i]` is `(q_i, p_i)` reached
/// after consuming the i-th trace packet.  The packet trace returned by
/// `get_any_trace` is `path.iter().map(|(_, p)| p)`.
pub fn get_any_trace_with_states<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
) -> Option<(Vec<(A::State, Vec<bool>)>, Vec<bool>)> {
    let q_start = aut.start(store);
    let mut todo: HashMap<A::State, sp::SP> = HashMap::new();
    let mut done: HashMap<A::State, sp::SP> = HashMap::new();
    let mut worklist: Vec<A::State> = Vec::new();
    let mut additions: HashMap<A::State, Vec<usize>> = HashMap::new();
    let mut steps: Vec<Step<A::State>> = Vec::new();

    todo.insert(q_start.clone(), store.sp.one);
    worklist.push(q_start);

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
            return Some(reconstruct_trace(&steps, step_idx, aut, store, output_spp));
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
) -> (Vec<(A::State, Vec<bool>)>, Vec<bool>) {
    // At the accepting step, pick a current packet that's in the accepting
    // step's diff AND has at least one output through `output_spp`, then
    // sample one matching output packet for it.
    let acc_diff = steps[accepting_idx].diff;
    let bwd_output = store.bwd(output_spp);
    let valid_inputs = store.sp.intersect(acc_diff, bwd_output);
    let acc_pkt = store.sp.any_packet(valid_inputs);
    let out_pkt = store.any_output_packet_from_input(output_spp, &acc_pkt);

    let mut packets_back: Vec<Vec<bool>> = vec![acc_pkt.clone()];
    let mut states_back: Vec<A::State> = vec![steps[accepting_idx].state.clone()];
    let mut current_idx = accepting_idx;
    let mut current_pkt = acc_pkt;

    // Walk back until we reach the init step (no predecessors).
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
                    let p_j = store.sp.any_packet(candidates);
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

    packets_back.reverse();
    states_back.reverse();
    let path: Vec<(A::State, Vec<bool>)> = states_back.into_iter().zip(packets_back).collect();
    (path, out_pkt)
}

pub(crate) fn singleton_sp(store: &mut spp::SPPstore, packet: &[bool]) -> sp::SP {
    store.sp.singleton(packet)
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

/// Backward reachability through `aut` for a fixed `(trace, output)` pair.
///
/// Returns a map `back[(q, i)] = sp`, where every entry is a non-empty SP
/// of packets `pkt` such that, if execution were to arrive at state `q`
/// carrying packet `pkt` at trace position `i`, the remainder of the trace
/// (`trace[i+1..]`) and the final output packet are still reachable from
/// there.
///
/// Concretely, `F(pkt, q, i)` (membership in `back[(q, i)]`) holds when:
///
/// * `q` is visible and `pkt = trace[i]` (a visible state pins the carry
///   packet), or `q` is invisible (no constraint on `pkt` from the trace), AND
/// * either
///   - `i == trace.len() - 1` and `q.output` accepts `(pkt, output)`, or
///   - there is a transition `q --s--> q'` with `s.accepts(pkt, pkt')` such
///     that `F(pkt', q', i')` holds, where `i' = i + 1` if `q'` is visible
///     (only valid when `i + 1 < trace.len()`) and `i' = i` otherwise.
///
/// The algorithm enumerates every state forward-reachable from `aut.start()`
/// and then runs a round-robin fixed-point over `(state, position)`,
/// computing each entry as the union of its termination contribution
/// (when `i == n - 1`) and the `pull`-back of every outgoing transition's
/// target entry.
///
/// `trace` must be non-empty.
///
/// `extra_roots` lets callers seed the enumeration with states that are not
/// forward-reachable from `aut.start()`.  This matters because the backward
/// region naturally lives *outside* the forward-reachable set (e.g. an
/// output-emitting state sitting behind an as-yet-empty hole): a plain
/// forward BFS from the start would never enumerate it, leaving its (genuinely
/// non-empty) backward entry uncomputed.  Pass `&[]` for the classic behaviour.
pub fn backward_reachable<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
    trace: &[Vec<bool>],
    output: &[bool],
    extra_roots: &[A::State],
) -> HashMap<(A::State, usize), sp::SP> {
    assert!(!trace.is_empty(), "trace must be non-empty");
    let n = trace.len();

    // 1. Forward BFS: enumerate reachable states and cache their structure.
    //    Seeded with the start *and* any `extra_roots`, so states off the
    //    forward-reachable path still get their backward entries computed.
    let start = aut.start(store);
    let mut reachable: HashSet<A::State> = HashSet::new();
    let mut order: Vec<A::State> = Vec::new();
    let mut adj: HashMap<A::State, Vec<(spp::SPP, A::State)>> = HashMap::new();
    let mut outputs: HashMap<A::State, spp::SPP> = HashMap::new();
    let mut visible: HashMap<A::State, bool> = HashMap::new();

    reachable.insert(start.clone());
    order.push(start.clone());
    let mut stack = vec![start];
    for root in extra_roots {
        if reachable.insert(root.clone()) {
            order.push(root.clone());
            stack.push(root.clone());
        }
    }
    while let Some(q) = stack.pop() {
        let trans = aut.transitions(store, &q);
        outputs.insert(q.clone(), aut.output(store, &q));
        visible.insert(q.clone(), aut.is_visible(store, &q));
        for (_, qp) in &trans {
            if reachable.insert(qp.clone()) {
                order.push(qp.clone());
                stack.push(qp.clone());
            }
        }
        adj.insert(q, trans);
    }

    // 2. Concrete-packet SPs for the trace positions and the final output.
    let trace_singletons: Vec<sp::SP> = trace.iter().map(|p| singleton_sp(store, p)).collect();
    let output_singleton: sp::SP = singleton_sp(store, output);

    // 3. Round-robin fixed-point iteration.
    let sp_zero = store.sp.zero;
    let mut back: HashMap<(A::State, usize), sp::SP> = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for q in &order {
            let q_visible = *visible.get(q).unwrap();
            let q_output = *outputs.get(q).unwrap();
            // Clone the transitions to release the borrow on `adj` while we
            // mutably touch `store` inside the inner loop.
            let trans = adj.get(q).unwrap().clone();
            for (i, &trace_singleton) in trace_singletons.iter().enumerate() {
                let mut new_sp = sp_zero;
                // Termination contribution at the final trace position.
                if i == n - 1 {
                    let pre = store.pull(q_output, output_singleton);
                    new_sp = store.sp.union(new_sp, pre);
                }
                // Transition contributions.
                for (s, qp) in &trans {
                    let qp_visible = *visible.get(qp).unwrap();
                    let i_target = if qp_visible {
                        if i + 1 >= n {
                            continue;
                        }
                        i + 1
                    } else {
                        i
                    };
                    let sp_target = back
                        .get(&(qp.clone(), i_target))
                        .copied()
                        .unwrap_or(sp_zero);
                    if store.sp.is_zero(sp_target) {
                        continue;
                    }
                    let pre = store.pull(*s, sp_target);
                    new_sp = store.sp.union(new_sp, pre);
                }
                // Visible q pins the carry packet to trace[i].
                if q_visible {
                    new_sp = store.sp.intersect(new_sp, trace_singleton);
                }
                if store.sp.is_zero(new_sp) {
                    continue;
                }
                let key = (q.clone(), i);
                let old_sp = back.get(&key).copied().unwrap_or(sp_zero);
                let combined = store.sp.union(old_sp, new_sp);
                if combined != old_sp {
                    back.insert(key, combined);
                    changed = true;
                }
            }
        }
    }

    back
}

/// Forward reachability through `aut` along a fixed `trace`.
///
/// Returns a map `forward[(q, i)] = sp`, where every entry is a non-empty
/// SP of carry packets `pkt` such that there is a run from the start that
/// reaches `(q, pkt)` having consumed the trace up to position `i`.
///
/// Mirror of [`backward_reachable`]: a forward fixed-point using
/// [`spp::SPPstore::push`] instead of `pull`.  The start state is treated
/// as visible at position 0 (matching the ENFA-trait convention), so
/// `forward[(start, 0)]` is seeded with the singleton `{trace[0]}`
/// regardless of `is_visible(start)`.
///
/// At every other `(q, i)`:
///
/// * visible `q` is restricted to the singleton `{trace[i]}` on entry, and
/// * a transition `q --s--> q'` advances to position `i + 1` if `q'` is
///   visible (skipped when `i + 1 >= trace.len()`), else stays at `i`.
///
/// `trace` must be non-empty.
pub fn forward_reachable<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
    trace: &[Vec<bool>],
) -> HashMap<(A::State, usize), sp::SP> {
    assert!(!trace.is_empty(), "trace must be non-empty");
    let n = trace.len();

    // 1. Forward BFS: enumerate reachable states and cache their structure.
    let start = aut.start(store);
    let mut reachable: HashSet<A::State> = HashSet::new();
    let mut order: Vec<A::State> = Vec::new();
    let mut adj: HashMap<A::State, Vec<(spp::SPP, A::State)>> = HashMap::new();
    let mut visible: HashMap<A::State, bool> = HashMap::new();

    reachable.insert(start.clone());
    order.push(start.clone());
    let mut stack = vec![start.clone()];
    while let Some(q) = stack.pop() {
        let trans = aut.transitions(store, &q);
        visible.insert(q.clone(), aut.is_visible(store, &q));
        for (_, qp) in &trans {
            if reachable.insert(qp.clone()) {
                order.push(qp.clone());
                stack.push(qp.clone());
            }
        }
        adj.insert(q, trans);
    }

    // 2. Concrete-packet SPs for each trace position.
    let trace_singletons: Vec<sp::SP> = trace.iter().map(|p| singleton_sp(store, p)).collect();

    // 3. Seed: start carries trace[0] at position 0 (start treated as visible).
    let sp_zero = store.sp.zero;
    let mut forward: HashMap<(A::State, usize), sp::SP> = HashMap::new();
    forward.insert((start, 0), trace_singletons[0]);

    // 4. Round-robin fixed-point iteration.
    let mut changed = true;
    while changed {
        changed = false;
        for q in &order {
            let trans = adj.get(q).unwrap().clone();
            for (i, _) in trace_singletons.iter().enumerate() {
                let sp_at_q_i = forward.get(&(q.clone(), i)).copied().unwrap_or(sp_zero);
                if store.sp.is_zero(sp_at_q_i) {
                    continue;
                }
                for (s, qp) in &trans {
                    let qp_visible = *visible.get(qp).unwrap();
                    let i_target = if qp_visible {
                        if i + 1 >= n {
                            continue;
                        }
                        i + 1
                    } else {
                        i
                    };
                    let pushed = store.push(sp_at_q_i, *s);
                    let restricted = if qp_visible {
                        store.sp.intersect(pushed, trace_singletons[i_target])
                    } else {
                        pushed
                    };
                    if store.sp.is_zero(restricted) {
                        continue;
                    }
                    let key = (qp.clone(), i_target);
                    let old = forward.get(&key).copied().unwrap_or(sp_zero);
                    let combined = store.sp.union(old, restricted);
                    if combined != old {
                        forward.insert(key, combined);
                        changed = true;
                    }
                }
            }
        }
    }

    forward
}

/// One-step ENFA helper used by [`elaborate_step`].  Wraps an inner ENFA so
/// every state is *visible* and adds a synthetic `PreStart` whose single
/// transition pins the inner source packet to `p_source` via
/// `force_output_spp(p_source)`.  Output is non-zero only at `q_dest`
/// (filtered to `p_dest`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum HelperState<S> {
    Start,
    Inner(S),
}

struct ElaborateStepHelper<'a, T: ENFA> {
    inner: &'a T,
    /// `force_output_spp(p_source)`: relates anything to `p_source`.
    start_to_source_spp: spp::SPP,
    source: T::State,
    end: T::State,
    end_output: spp::SPP,
    zero_spp: spp::SPP,
}

impl<'a, T: ENFA> ENFA for ElaborateStepHelper<'a, T> {
    type State = HelperState<T::State>;

    fn start(&self, _store: &mut spp::SPPstore) -> Self::State {
        HelperState::Start
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
        true
    }

    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)> {
        // Between two consecutive *visible* boundary states of an
        // `EpsilonClosure` trace there are only invisible intermediates plus a
        // single final hop onto the (visible) destination.  So from either the
        // pinned source (`Start`) or an intermediate (`Inner`) we may traverse
        // invisible inner states freely, but the only visible state we are
        // allowed to step onto is the pinned `end`.  Following any *other*
        // visible state would splice a spurious extra boundary (e.g. a dup
        // inside a `Cand` hole, whose `Middle` states are visible) into the
        // reconstruction and mis-attribute it as an additional hole traversal.
        match q {
            HelperState::Start => {
                let trans = self.inner.transitions(store, &self.source);
                let mut result = Vec::new();
                for (spp, q_next) in trans {
                    if self.inner.is_visible(store, &q_next) && q_next != self.end {
                        continue;
                    }
                    result.push((
                        store.sequence(self.start_to_source_spp, spp),
                        HelperState::Inner(q_next),
                    ));
                }
                result
            }
            HelperState::Inner(s) => {
                let trans = self.inner.transitions(store, s);
                let mut result = Vec::new();
                for (spp, q_next) in trans {
                    if self.inner.is_visible(store, &q_next) && q_next != self.end {
                        continue;
                    }
                    result.push((spp, HelperState::Inner(q_next)));
                }
                result
            }
        }
    }

    fn output(&self, _store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
        match q {
            HelperState::Start => self.zero_spp,
            HelperState::Inner(s) if *s == self.end => self.end_output,
            HelperState::Inner(_) => self.zero_spp,
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
pub fn elaborate_step<T: ENFA>(
    inner: &T,
    store: &mut spp::SPPstore,
    q_source: &T::State,
    p_source: &[bool],
    q_dest: &T::State,
    p_dest: &[bool],
) -> Vec<(T::State, Vec<bool>)> {
    // Build a tiny helper ENFA (every state visible, pinned start, single
    // accept), run `get_any_trace` on it, then forward-simulate the returned
    // trace through `inner` to recover the state sequence.
    let start_spp = force_output_spp(store, p_source);
    let output_spp = filter_input_spp(store, p_dest);
    let zero_spp = store.zero;

    let helper = ElaborateStepHelper {
        inner,
        start_to_source_spp: start_spp,
        source: q_source.clone(),
        end: q_dest.clone(),
        end_output: output_spp,
        zero_spp,
    };

    // Recover the helper's *actual* state path rather than re-simulating the
    // packet trace through `inner`: when `inner` is nondeterministic, several
    // transitions can accept the same packet pair, so a greedy forward
    // simulation may follow a different path than the helper found and land in
    // the wrong state.  `get_any_trace_with_states` hands back the exact states.
    let (path_full, _output_pkt) = get_any_trace_with_states(&helper, store)
        .expect("elaborate_step: no inner ENFA path between source and dest");
    // path_full[0] is `(Start, _)`; the rest are the inner intermediates ending
    // at `(Inner(q_dest), p_dest)`.
    let path: Vec<(T::State, Vec<bool>)> = path_full
        .into_iter()
        .skip(1)
        .map(|(q, p)| match q {
            HelperState::Inner(s) => (s, p),
            HelperState::Start => unreachable!("Start only occurs at the trace head"),
        })
        .collect();

    // The last visited state/packet must be the destination (or, when no inner
    // step was needed, the source itself — which must then equal the dest).
    let (final_state, final_pkt) = path
        .last()
        .map(|(s, p)| (s, p.as_slice()))
        .unwrap_or((q_source, p_source));
    assert!(final_state == q_dest);
    assert!(final_pkt == p_dest);

    path
}

/// ENFA helper used by [`elaborate_tail`].  Like [`ElaborateStepHelper`] but
/// with no fixed destination: from the pinned source it walks only into
/// *invisible* inner states (the ε-closure the wrapper hides), and any state
/// whose inner output can emit `output_pkt` is accepting (its output is
/// intersected with `force_output` to pin the emitted packet).
struct ElaborateTailHelper<'a, T: ENFA> {
    inner: &'a T,
    /// `force_output_spp(p_source)`: relates anything to `p_source`.
    start_to_source_spp: spp::SPP,
    source: T::State,
    /// `force_output_spp(output_pkt)`: intersected with each state's inner
    /// output so a trace is accepting only when it can emit `output_pkt`.
    force_output: spp::SPP,
    zero_spp: spp::SPP,
}

impl<'a, T: ENFA> ENFA for ElaborateTailHelper<'a, T> {
    type State = HelperState<T::State>;

    fn start(&self, _store: &mut spp::SPPstore) -> Self::State {
        HelperState::Start
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &Self::State) -> bool {
        true
    }

    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)> {
        match q {
            HelperState::Start => vec![(
                self.start_to_source_spp,
                HelperState::Inner(self.source.clone()),
            )],
            HelperState::Inner(s) => {
                let trans = self.inner.transitions(store, s);
                let mut result = Vec::new();
                for (spp, q_next) in trans {
                    if self.inner.is_visible(store, &q_next) {
                        continue;
                    }
                    result.push((spp, HelperState::Inner(q_next)));
                }
                result
            }
        }
    }

    fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
        match q {
            HelperState::Start => self.zero_spp,
            HelperState::Inner(s) => {
                let out = self.inner.output(store, s);
                store.intersect(out, self.force_output)
            }
        }
    }
}

/// Elaborate the invisible *tail* of an `EpsilonClosure<T>` trace: starting
/// from the last visible state `q_last` (at packet `p_last`), recover the
/// ε-only path through invisible inner states that ends where `output_pkt` is
/// actually emitted.
///
/// Returns the sequence of `(state, packet)` pairs visited *after* `q_last`
/// (so `q_last` is **not** included).  The result is empty when `q_last`'s own
/// inner output already emits `output_pkt`.
pub fn elaborate_tail<T: ENFA>(
    inner: &T,
    store: &mut spp::SPPstore,
    q_last: &T::State,
    p_last: &[bool],
    output_pkt: &[bool],
) -> Vec<(T::State, Vec<bool>)> {
    // Same strategy as `elaborate_step`: build a tiny all-visible helper ENFA,
    // run `get_any_trace` on it, then forward-simulate through `inner`.
    let start_spp = force_output_spp(store, p_last);
    let force_output = force_output_spp(store, output_pkt);
    let zero_spp = store.zero;

    let helper = ElaborateTailHelper {
        inner,
        start_to_source_spp: start_spp,
        source: q_last.clone(),
        force_output,
        zero_spp,
    };

    // As in `elaborate_step`, read the helper's actual state path instead of
    // re-simulating: greedy forward simulation can pick the wrong transition
    // when `inner` is nondeterministic.
    let (path_full, _output_pkt) = get_any_trace_with_states(&helper, store)
        .expect("elaborate_tail: no inner ε-path emits the trace's output packet");
    // path_full[0] is `(Start, _)`; `path_full[1]` is the real start state;
    // the rest are the spliced invisible inner states.
    path_full
        .into_iter()
        .skip(2)
        .map(|(q, p)| match q {
            HelperState::Inner(s) => (s, p),
            HelperState::Start => unreachable!("PreStart only occurs at the trace head"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only ENFA wrapper that overlays explicit visibility flags on an
    /// inner automaton whose `State` is `usize`.  Owns the inner so the
    /// wrapper can be moved/cloned freely.
    #[derive(Clone)]
    struct RandomVisibility<T> {
        inner: T,
        visible: Vec<bool>,
    }

    impl<T> ENFA for RandomVisibility<T>
    where
        T: ENFA<State = usize>,
    {
        type State = usize;
        fn start(&self, store: &mut spp::SPPstore) -> usize {
            self.inner.start(store)
        }
        fn is_visible(&self, _store: &mut spp::SPPstore, q: &usize) -> bool {
            self.visible[*q]
        }
        fn transitions(&self, store: &mut spp::SPPstore, q: &usize) -> Vec<(spp::SPP, usize)> {
            self.inner.transitions(store, q)
        }
        fn output(&self, store: &mut spp::SPPstore, q: &usize) -> spp::SPP {
            self.inner.output(store, q)
        }
    }

    /// Generate a random ExplicitDFA from a fuzzed expression.  The DFA's
    /// states are dense `usize` indices; SPPs live in `aut`'s shared store,
    /// so multiple DFAs from the same `aut` interoperate freely.
    fn random_dfa(aut: &mut crate::aut::Aut, expr_depth: usize, num_fields: u32) -> ExplicitDFA {
        let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);
        let state = aut.expr_to_state(&expr);
        aut_to_dfa(aut, state)
    }

    /// Generate a random NFA: a [`random_dfa`] with random visibility flags
    /// (start kept visible) wrapped in [`EpsilonClosure`].  The closure
    /// makes it a real NFA (every wrapper state visible) over the same
    /// `usize` state space.
    fn random_nfa(
        aut: &mut crate::aut::Aut,
        expr_depth: usize,
        num_fields: u32,
    ) -> EpsilonClosure<RandomVisibility<ExplicitDFA>> {
        let dfa = random_dfa(aut, expr_depth, num_fields);
        let n = dfa.num_states();
        let wrapper = RandomVisibility {
            inner: dfa,
            visible: (0..n).map(|_| rand::random::<bool>()).collect(),
        };
        EpsilonClosure::new(wrapper)
    }

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
            let mut aut = crate::aut::Aut::new(num_fields);
            let nfa = random_nfa(&mut aut, expr_depth, num_fields);
            // The underlying ExplicitDFA is two `inner()` hops down; reach in
            // through the field on `RandomVisibility`.
            let dfa_empty = is_empty(&nfa.inner().inner, aut.spp_store_mut());
            let closure_empty = is_empty(&nfa, aut.spp_store_mut());
            assert_eq!(
                dfa_empty, closure_empty,
                "emptiness mismatch on trial {}: dfa={}, closure={}",
                trial, dfa_empty, closure_empty
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
        let max_trials = 500;

        for trial in 0..max_trials {
            let mut aut = crate::aut::Aut::new(num_fields);
            let closure = random_nfa(&mut aut, expr_depth, num_fields);

            if is_empty(&closure, aut.spp_store_mut()) {
                continue;
            }

            let trace = get_any_trace_with_states(&closure, aut.spp_store_mut())
                .expect("non-empty closure should yield a trace");
            let (visible_path, output_pkt) = trace;

            let full_path =
                closure.elaborate_trace(aut.spp_store_mut(), &visible_path, &output_pkt);

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
                    "trial {}: elaborated step #{} ({}, {:?}) -> ({}, {:?}) is not a valid inner transition",
                    trial, i, q_a, p_a, q_b, p_b,
                );
            }

            // The elaboration starts at the visible start...
            let (start_q, start_p) = &visible_path[0];
            assert_eq!(*start_q, full_path.first().unwrap().0);
            assert_eq!(start_p, &full_path.first().unwrap().1);

            // ...and ends at a state whose inner output emits the trace's
            // output packet -- possibly an invisible state spliced in past the
            // last visible one (the `elaborate_tail` fix).
            let (end_q, end_p) = full_path.last().unwrap().clone();
            let out_spp = closure.inner().output(aut.spp_store_mut(), &end_q);
            assert!(
                aut.spp_store_mut().accepts(out_spp, &end_p, &output_pkt),
                "trial {}: elaborated path ends at ({}, {:?}) which does not emit output {:?}",
                trial,
                end_q,
                end_p,
                output_pkt,
            );
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
        let max_trials = 500;

        for trial in 0..max_trials {
            let mut aut = crate::aut::Aut::new(num_fields);
            let nfa = random_nfa(&mut aut, expr_depth, num_fields);
            let subset = SubsetDfa::new(nfa);

            let nfa_empty = is_empty(subset.inner(), aut.spp_store_mut());
            let subset_empty = is_empty(&subset, aut.spp_store_mut());
            assert_eq!(
                nfa_empty, subset_empty,
                "trial {}: emptiness mismatch (nfa={}, subset={})",
                trial, nfa_empty, subset_empty
            );

            // Single-start invariant.
            let _ = subset.start(aut.spp_store_mut());

            if !subset_empty {
                let trace = get_any_trace(&subset, aut.spp_store_mut())
                    .expect("non-empty subset DFA should yield a trace");
                let (t, output) = trace;
                assert!(
                    subset.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: subset DFA rejects its own trace",
                    trial,
                );
                assert!(
                    subset.inner().nfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: underlying NFA rejects the subset DFA's trace",
                    trial,
                );
            }
        }
    }

    #[test]
    fn fuzz_reachable_from_trace() {
        // A trace produced by `get_any_trace_with_states` records, for each
        // step after the start, the `(state, current_packet)` pair the trace
        // visits.  Feeding the bare packet trace plus the input back through
        // `reachable_from_trace` must reach at least all of those pairs --
        // it explores the full nondeterministic frontier, which is a
        // superset of any one accepting path.
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 500;

        for trial in 0..max_trials {
            let mut aut = crate::aut::Aut::new(num_fields);
            let nfa = random_nfa(&mut aut, expr_depth, num_fields);

            if is_empty(&nfa, aut.spp_store_mut()) {
                continue;
            }

            let (path, _output) = get_any_trace_with_states(&nfa, aut.spp_store_mut())
                .expect("non-empty NFA should yield a trace");
            let trace_pkts: Vec<Vec<bool>> = path.iter().map(|(_, p)| p.clone()).collect();

            let reachable = nfa.reachable_from_trace(aut.spp_store_mut(), &trace_pkts);
            let reachable_set: HashSet<(_, Vec<bool>)> = reachable
                .into_iter()
                .map(|(q, p)| (q, p.to_vec()))
                .collect();

            for (i, (q, p)) in path.iter().enumerate() {
                assert!(
                    reachable_set.contains(&(*q, p.clone())),
                    "trial {}: path step {} ({:?}, {:?}) missing from reachable_from_trace result",
                    trial,
                    i,
                    q,
                    p,
                );
            }
        }
    }

    #[test]
    fn fuzz_complement() {
        // For a DFA `D` and any concrete triple `(input, trace, output)`:
        //   D.dfa_accepts(triple) iff NOT complement(D).dfa_accepts(triple).
        // Verify with a triple from each side (when non-empty).
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 500;

        for trial in 0..max_trials {
            let mut aut = crate::aut::Aut::new(num_fields);
            let dfa = random_dfa(&mut aut, expr_depth, num_fields);
            let comp = ops::complement(&dfa);

            // A trace accepted by the original is rejected by the complement.
            if let Some((t, output)) = get_any_trace(&dfa, aut.spp_store_mut()) {
                assert!(
                    dfa.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: dfa rejects its own trace",
                    trial
                );
                assert!(
                    !comp.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: complement accepts a trace the DFA accepts",
                    trial
                );
            }

            // A trace accepted by the complement is rejected by the original.
            if let Some((t, output)) = get_any_trace(&comp, aut.spp_store_mut()) {
                assert!(
                    comp.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: complement rejects its own trace",
                    trial
                );
                assert!(
                    !dfa.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: dfa accepts a trace the complement accepts",
                    trial
                );
            }
        }
    }

    #[test]
    fn fuzz_union() {
        // Property: union(A, B) accepts a triple iff A or B accepts it.
        // Forward: if A or B accepts a triple, union should too.
        // Reverse: any triple union accepts must be accepted by A or B.
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 500;

        for trial in 0..max_trials {
            let mut aut = crate::aut::Aut::new(num_fields);
            let nfa1 = random_nfa(&mut aut, expr_depth, num_fields);
            let nfa2 = random_nfa(&mut aut, expr_depth, num_fields);

            // Build the union (an ENFA) and run it through EpsilonClosure to
            // get a queryable NFA -- the union has no real ε states under
            // our list-based start, so the closure is a thin pass-through.
            let union_enfa = ops::union(&nfa1, &nfa2);
            let union_nfa = EpsilonClosure::new(union_enfa);

            // Forward: trace from nfa1 is accepted by union.
            if let Some((t, output)) = get_any_trace(&nfa1, aut.spp_store_mut()) {
                assert!(
                    union_nfa.nfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: union rejects a trace nfa1 accepts",
                    trial
                );
            }
            // Forward: trace from nfa2 is accepted by union.
            if let Some((t, output)) = get_any_trace(&nfa2, aut.spp_store_mut()) {
                assert!(
                    union_nfa.nfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: union rejects a trace nfa2 accepts",
                    trial
                );
            }
            // Reverse: trace from union is accepted by nfa1 or nfa2.
            if let Some((t, output)) = get_any_trace(&union_nfa, aut.spp_store_mut()) {
                let in1 = nfa1.nfa_accepts(aut.spp_store_mut(), &t, &output);
                let in2 = nfa2.nfa_accepts(aut.spp_store_mut(), &t, &output);
                assert!(
                    in1 || in2,
                    "trial {}: union accepts a trace neither nfa1 nor nfa2 accepts",
                    trial,
                );
            }
        }
    }

    #[test]
    fn fuzz_intersection() {
        // Property: intersection(A, B) accepts a triple iff both A and B do.
        // Tests both the NFA case and the DFA-DFA case (where the result is
        // a real DFA via the conditional `impl<A: DFA, B: DFA> DFA for ...`).
        let expr_depth = 4;
        let num_fields = 3;
        let max_trials = 500;

        for trial in 0..max_trials {
            let mut aut = crate::aut::Aut::new(num_fields);

            // NFA-NFA case.
            let nfa1 = random_nfa(&mut aut, expr_depth, num_fields);
            let nfa2 = random_nfa(&mut aut, expr_depth, num_fields);
            let inter = ops::intersection(&nfa1, &nfa2);

            // Reverse: any trace from intersection is accepted by both.
            if let Some((t, output)) = get_any_trace(&inter, aut.spp_store_mut()) {
                assert!(
                    nfa1.nfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: nfa1 rejects a trace from the intersection",
                    trial
                );
                assert!(
                    nfa2.nfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: nfa2 rejects a trace from the intersection",
                    trial
                );
            }

            // Forward (best-effort): if a trace from nfa1 is also accepted by
            // nfa2, the intersection must accept it.
            if let Some((t, output)) = get_any_trace(&nfa1, aut.spp_store_mut())
                && nfa2.nfa_accepts(aut.spp_store_mut(), &t, &output)
            {
                assert!(
                    inter.nfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: intersection rejects a trace accepted by both nfa1 and nfa2",
                    trial
                );
            }

            // DFA-DFA case: result is a DFA.
            let dfa1 = random_dfa(&mut aut, expr_depth, num_fields);
            let dfa2 = random_dfa(&mut aut, expr_depth, num_fields);
            let inter_dfa = ops::intersection(&dfa1, &dfa2);

            // Single-start invariant (always trivially holds now).
            let _ = inter_dfa.start(aut.spp_store_mut());

            if let Some((t, output)) = get_any_trace(&inter_dfa, aut.spp_store_mut()) {
                assert!(
                    dfa1.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: dfa1 rejects a trace from the DFA intersection",
                    trial
                );
                assert!(
                    dfa2.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: dfa2 rejects a trace from the DFA intersection",
                    trial
                );
                assert!(
                    inter_dfa.dfa_accepts(aut.spp_store_mut(), &t, &output),
                    "trial {}: DFA intersection rejects its own trace",
                    trial
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
                let (trace_pkts, output) = trace.unwrap();
                let accepted_dfa = dfa.dfa_accepts(aut.spp_store_mut(), &trace_pkts, &output);
                assert!(
                    accepted_dfa,
                    "trace not accepted by dfa_accepts on trial {} for expr {}: trace={:?}, output={:?}",
                    trial, expr, trace_pkts, output
                );
                let accepted_nfa = dfa.nfa_accepts(aut.spp_store_mut(), &trace_pkts, &output);
                assert!(
                    accepted_nfa,
                    "trace not accepted by nfa_accepts on trial {} for expr {}: trace={:?}, output={:?}",
                    trial, expr, trace_pkts, output
                );
            }
        }
    }

    // ── backward_reachable ────────────────────────────────────────────────

    #[test]
    fn backward_single_visible_accepting() {
        // One visible state with output = top; trace of length 1.
        // back[(0,0)] must be exactly {trace[0]} — visible pins the carry,
        // and the top output allows any output packet.
        let mut store = spp::SPPstore::new(3);
        let top = store.top;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![top],
        };
        let trace = vec![vec![true, false, true]];
        let output = vec![false, true, false];
        let back = backward_reachable(&dfa, &mut store, &trace, &output, &[]);

        assert_eq!(back.len(), 1);
        let expected = singleton_sp(&mut store, &trace[0]);
        assert_eq!(back[&(0, 0)], expected);
    }

    #[test]
    fn backward_no_output_is_empty() {
        // One visible state with output = zero; nothing can terminate.
        let mut store = spp::SPPstore::new(3);
        let zero = store.zero;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![zero],
        };
        let trace = vec![vec![true, false, true]];
        let output = vec![false, true, false];
        let back = backward_reachable(&dfa, &mut store, &trace, &output, &[]);
        assert!(back.is_empty(), "got: {:?}", back);
    }

    #[test]
    fn backward_linear_two_visible_states() {
        // 0 --top--> 1, both visible.  output[0]=zero, output[1]=top.
        // back[(1,1)] = {trace[1]}; back[(0,0)] = {trace[0]}; no others.
        let mut store = spp::SPPstore::new(2);
        let top = store.top;
        let zero = store.zero;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(top, 1)], vec![]],
            outputs: vec![zero, top],
        };
        let trace = vec![vec![false, false], vec![true, true]];
        let output = vec![true, false];
        let back = backward_reachable(&dfa, &mut store, &trace, &output, &[]);

        let s0 = singleton_sp(&mut store, &trace[0]);
        let s1 = singleton_sp(&mut store, &trace[1]);
        assert_eq!(back[&(0, 0)], s0);
        assert_eq!(back[&(1, 1)], s1);
        assert_eq!(back.len(), 2);
    }

    #[test]
    fn backward_with_invisible_intermediate() {
        // 0 (visible) --top--> 1 (invisible) --top--> 2 (visible).
        // output[0]=output[1]=zero, output[2]=top.
        // Trace length 2.  Expected entries:
        //   (2, 1) → {trace[1]}   (visible, pinned)
        //   (1, 0) → all packets  (invisible, transitions to (2,1) via top)
        //   (0, 0) → {trace[0]}   (visible, transitions through 1 to 2)
        let mut store = spp::SPPstore::new(2);
        let top = store.top;
        let zero = store.zero;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(top, 1)], vec![(top, 2)], vec![]],
            outputs: vec![zero, zero, top],
        };
        let wrap = RandomVisibility {
            inner: dfa,
            visible: vec![true, false, true],
        };
        let trace = vec![vec![false, false], vec![true, true]];
        let output = vec![false, true];
        let back = backward_reachable(&wrap, &mut store, &trace, &output, &[]);

        let s0 = singleton_sp(&mut store, &trace[0]);
        let s1 = singleton_sp(&mut store, &trace[1]);
        let all = store.sp.one;
        assert_eq!(back[&(2, 1)], s1);
        assert_eq!(back[&(1, 0)], all);
        assert_eq!(back[&(0, 0)], s0);
        assert_eq!(back.len(), 3);
    }

    // ── forward_reachable ────────────────────────────────────────────────

    #[test]
    fn forward_single_state() {
        // One visible state; trace of length 1.  forward[(0, 0)] = {trace[0]}.
        let mut store = spp::SPPstore::new(3);
        let top = store.top;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![top],
        };
        let trace = vec![vec![true, false, true]];
        let forward = forward_reachable(&dfa, &mut store, &trace);
        assert_eq!(forward.len(), 1);
        let expected = singleton_sp(&mut store, &trace[0]);
        assert_eq!(forward[&(0, 0)], expected);
    }

    #[test]
    fn forward_linear_two_visible_states() {
        // 0 --top--> 1, both visible.  forward[(0,0)] = {trace[0]}; forward[(1,1)] = {trace[1]}.
        let mut store = spp::SPPstore::new(2);
        let top = store.top;
        let zero = store.zero;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(top, 1)], vec![]],
            outputs: vec![zero, zero],
        };
        let trace = vec![vec![false, false], vec![true, true]];
        let forward = forward_reachable(&dfa, &mut store, &trace);
        let s0 = singleton_sp(&mut store, &trace[0]);
        let s1 = singleton_sp(&mut store, &trace[1]);
        assert_eq!(forward[&(0, 0)], s0);
        assert_eq!(forward[&(1, 1)], s1);
        assert_eq!(forward.len(), 2);
    }

    #[test]
    fn forward_with_invisible_intermediate() {
        // 0 (visible) --top--> 1 (invisible) --top--> 2 (visible).
        // forward[(0,0)] = {trace[0]}; forward[(1,0)] = all packets (no
        // visibility restriction); forward[(2,1)] = {trace[1]}.
        let mut store = spp::SPPstore::new(2);
        let top = store.top;
        let zero = store.zero;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(top, 1)], vec![(top, 2)], vec![]],
            outputs: vec![zero, zero, zero],
        };
        let wrap = RandomVisibility {
            inner: dfa,
            visible: vec![true, false, true],
        };
        let trace = vec![vec![false, false], vec![true, true]];
        let forward = forward_reachable(&wrap, &mut store, &trace);
        let s0 = singleton_sp(&mut store, &trace[0]);
        let s1 = singleton_sp(&mut store, &trace[1]);
        let all = store.sp.one;
        assert_eq!(forward[&(0, 0)], s0);
        assert_eq!(forward[&(1, 0)], all);
        assert_eq!(forward[&(2, 1)], s1);
        assert_eq!(forward.len(), 3);
    }

    #[test]
    fn forward_dead_end_after_first_step() {
        // 0 (visible) --top--> 1 (visible) --zero--> ... trace requires step but
        // no transition can be taken.  forward only contains (0, 0).
        let mut store = spp::SPPstore::new(2);
        let top = store.top;
        let zero = store.zero;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(zero, 1)], vec![]],
            outputs: vec![zero, top],
        };
        let trace = vec![vec![false, false], vec![true, true]];
        let forward = forward_reachable(&dfa, &mut store, &trace);
        let s0 = singleton_sp(&mut store, &trace[0]);
        assert_eq!(forward[&(0, 0)], s0);
        assert_eq!(forward.len(), 1);
    }

    #[test]
    fn backward_unreachable_output_pkt_is_empty() {
        // Output SPP = `one` (identity: only (p, p) accepted).  If
        // output_pkt = [false, false] but trace[0] = [true, true], the only
        // way to terminate at state 0 is pkt = output_pkt = [false, false],
        // but visibility pins pkt = trace[0] = [true, true].  Contradiction:
        // back is empty.
        let mut store = spp::SPPstore::new(2);
        let one = store.one;
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![one],
        };
        let trace = vec![vec![true, true]];
        let output = vec![false, false];
        let back = backward_reachable(&dfa, &mut store, &trace, &output, &[]);
        assert!(back.is_empty(), "got: {:?}", back);
    }
}
