use crate::sp;
use crate::spp;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;

/// Nondeterministic symbolic NetKAT automaton with epsilon transitions.
///
/// States are either *visible* (consume a packet from the trace when entered)
/// or *invisible* (epsilon states that do not advance the trace position).
pub trait ENFA {
    type State: Clone + Eq + Hash;

    /// The start state.
    fn start(&self) -> Self::State;

    /// True for visible states (entering them consumes one packet from the trace);
    /// false for invisible (epsilon) states.
    fn is_visible(&self, q: &Self::State) -> bool;

    /// Outgoing transitions from `q`: each entry is a (packet_program, next_state) pair.
    fn transitions(&self, q: &Self::State) -> Vec<(spp::SPP, Self::State)>;

    /// Output SPP for `q`: the set of (current_packet, output_packet) pairs
    /// the automaton accepts when it terminates in state `q`.
    fn output(&self, q: &Self::State) -> spp::SPP;
}

/// Nondeterministic symbolic NetKAT automaton without epsilon transitions.
///
/// Every state is visible; `is_visible` must always return `true`.
pub trait NFA: ENFA {}

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
        let mut current_state = self.start();
        let mut current_pkt: Vec<bool> = input.to_vec();

        for next_pkt in trace {
            let mut next_state = None;
            for (spp, q_next) in self.transitions(&current_state) {
                if spp_accepts(store, spp, &current_pkt, next_pkt) {
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

        spp_accepts(store, self.output(&current_state), &current_pkt, output)
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

    fn start(&self) -> usize {
        let s = self.inner.start();
        self.data.borrow_mut().get_or_insert(s)
    }

    fn is_visible(&self, id: &usize) -> bool {
        if let Some(&v) = self.data.borrow().visible_cache.get(id) {
            return v;
        }
        let state = self.data.borrow().get_state(*id).clone();
        let v = self.inner.is_visible(&state);
        self.data.borrow_mut().visible_cache.insert(*id, v);
        v
    }

    fn transitions(&self, id: &usize) -> Vec<(spp::SPP, usize)> {
        if let Some(cached) = self.data.borrow().transitions_cache.get(id) {
            return cached.clone();
        }
        let state = self.data.borrow().get_state(*id).clone();
        let raw = self.inner.transitions(&state);
        let mut result = Vec::with_capacity(raw.len());
        let mut data = self.data.borrow_mut();
        for (spp, next) in raw {
            let next_id = data.get_or_insert(next);
            result.push((spp, next_id));
        }
        data.transitions_cache.insert(*id, result.clone());
        result
    }

    fn output(&self, id: &usize) -> spp::SPP {
        if let Some(&spp) = self.data.borrow().output_cache.get(id) {
            return spp;
        }
        let state = self.data.borrow().get_state(*id).clone();
        let spp = self.inner.output(&state);
        self.data.borrow_mut().output_cache.insert(*id, spp);
        spp
    }
}

impl<T: NFA> NFA for Memo<T> {}
impl<T: DFA> DFA for Memo<T> {}

impl<T: DFA> Memo<T> {
    /// Exhaustively explores all states reachable from the start, then moves
    /// the cached transitions and outputs into an `ExplicitDFA`.
    ///
    /// Bypasses the trait `transitions`/`output` methods (which would clone
    /// from the cache) and writes directly into `self.data`, so each cached
    /// `Vec` is moved out exactly once.
    pub fn into_explicit(self) -> ExplicitDFA {
        let start_state = self.inner.start();
        let start_id = self.data.borrow_mut().get_or_insert(start_state);

        let mut i = 0;
        loop {
            let len = self.data.borrow().id_to_state.len();
            if i >= len {
                break;
            }

            let already_cached = {
                let data = self.data.borrow();
                data.transitions_cache.contains_key(&i) && data.output_cache.contains_key(&i)
            };
            if !already_cached {
                let state = self.data.borrow().id_to_state[i].clone();
                let raw = self.inner.transitions(&state);
                let out = self.inner.output(&state);
                let mut converted = Vec::with_capacity(raw.len());
                let mut data = self.data.borrow_mut();
                for (spp, next) in raw {
                    let next_id = data.get_or_insert(next);
                    converted.push((spp, next_id));
                }
                data.transitions_cache.insert(i, converted);
                data.output_cache.insert(i, out);
            }
            i += 1;
        }

        let mut data = self.data.into_inner();
        let n = data.id_to_state.len();
        let mut transitions = Vec::with_capacity(n);
        let mut outputs = Vec::with_capacity(n);
        for j in 0..n {
            transitions.push(
                data.transitions_cache
                    .remove(&j)
                    .expect("populated by BFS above"),
            );
            outputs.push(
                data.output_cache
                    .remove(&j)
                    .expect("populated by BFS above"),
            );
        }

        ExplicitDFA {
            start: start_id,
            transitions,
            outputs,
        }
    }
}

// ---- ExplicitDFA -----------------------------------------------------------

/// A DFA stored as flat tables: `transitions[i]` and `outputs[i]` describe
/// state `i`, indexed by dense `usize`.  Fields are public so callers can
/// avoid going through the trait when raw access is needed.
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

    fn start(&self) -> usize {
        self.start
    }

    fn is_visible(&self, _q: &usize) -> bool {
        true
    }

    fn transitions(&self, q: &usize) -> Vec<(spp::SPP, usize)> {
        self.transitions[*q].clone()
    }

    fn output(&self, q: &usize) -> spp::SPP {
        self.outputs[*q]
    }
}

impl NFA for ExplicitDFA {}
impl DFA for ExplicitDFA {}

// ---- Wrapper for crate::aut::Aut -------------------------------------------

/// Borrowed wrapper around `crate::aut::Aut`, used internally by `aut_to_dfa`
/// to drive a DFA traversal without taking ownership.
///
/// `Aut`'s query methods need `&mut self` due to memoization, so the borrow
/// lives behind a `RefCell` to satisfy the `ENFA` trait's `&self` methods.
struct AutDfaRef<'a> {
    inner: RefCell<&'a mut crate::aut::Aut>,
    start: usize,
}

impl<'a> ENFA for AutDfaRef<'a> {
    type State = usize;

    fn start(&self) -> usize {
        self.start
    }

    fn is_visible(&self, _q: &usize) -> bool {
        true
    }

    fn transitions(&self, q: &usize) -> Vec<(spp::SPP, usize)> {
        let st = self.inner.borrow_mut().delta(*q);
        st.get_transitions()
            .iter()
            .map(|(&s, &spp)| (spp, s))
            .collect()
    }

    fn output(&self, q: &usize) -> spp::SPP {
        self.inner.borrow_mut().epsilon(*q)
    }
}

impl<'a> NFA for AutDfaRef<'a> {}
impl<'a> DFA for AutDfaRef<'a> {}

/// Materializes the DFA reachable from `start` in `aut` as an `ExplicitDFA`
/// with dense `usize` state IDs.
pub fn aut_to_dfa(aut: &mut crate::aut::Aut, start: usize) -> ExplicitDFA {
    Memo::new(AutDfaRef {
        inner: RefCell::new(aut),
        start,
    })
    .into_explicit()
}

// ---- Algorithms -------------------------------------------------------------

/// Returns `true` if no (input, trace, output) triple is accepted by `aut`.
pub fn is_empty<A: ENFA>(aut: &A, store: &mut spp::SPPstore) -> bool {
    let start = aut.start();
    let mut todo: HashMap<A::State, sp::SP> = HashMap::new();
    let mut done: HashMap<A::State, sp::SP> = HashMap::new();
    let mut worklist: Vec<A::State> = Vec::new();

    todo.insert(start.clone(), store.sp.one);
    worklist.push(start);

    while let Some(q) = worklist.pop() {
        let todo_q = todo.get(&q).copied().unwrap_or(store.sp.zero);
        let done_q = done.get(&q).copied().unwrap_or(store.sp.zero);
        let diff = store.sp.difference(todo_q, done_q);

        if diff == store.sp.zero {
            continue;
        }

        let output_spp = aut.output(&q);
        let output_sp = store.push(diff, output_spp);
        if output_sp != store.sp.zero {
            return false;
        }

        for (spp, q_next) in aut.transitions(&q) {
            let reachable = store.push(diff, spp);
            if reachable != store.sp.zero {
                let prev = todo.get(&q_next).copied().unwrap_or(store.sp.zero);
                let new_val = store.sp.union(prev, reachable);
                if new_val != prev {
                    todo.insert(q_next.clone(), new_val);
                    worklist.push(q_next);
                }
            }
        }

        let prev_done = done.get(&q).copied().unwrap_or(store.sp.zero);
        let new_done = store.sp.union(prev_done, diff);
        done.insert(q, new_done);
    }

    true
}

struct Step<S> {
    state: S,
    diff: sp::SP,
    predecessors: Vec<usize>,
}

/// Returns one concrete `(start_packet, trace, output_packet)` triple accepted
/// by `aut`, or `None` if the automaton is empty.
///
/// `trace` is the sequence of current packets at each state visited after the
/// start state, ending with the current packet at the accepting state.  All
/// states are included regardless of visibility; callers that want only
/// visible states can filter afterwards.
pub fn get_any_trace<A: ENFA>(
    aut: &A,
    store: &mut spp::SPPstore,
) -> Option<(Vec<bool>, Vec<Vec<bool>>, Vec<bool>)> {
    let start = aut.start();
    let mut todo: HashMap<A::State, sp::SP> = HashMap::new();
    let mut done: HashMap<A::State, sp::SP> = HashMap::new();
    let mut worklist: Vec<A::State> = Vec::new();
    let mut additions: HashMap<A::State, Vec<usize>> = HashMap::new();
    let mut steps: Vec<Step<A::State>> = Vec::new();

    todo.insert(start.clone(), store.sp.one);
    worklist.push(start.clone());

    while let Some(q) = worklist.pop() {
        let preds = additions.remove(&q).unwrap_or_default();

        let todo_q = todo.get(&q).copied().unwrap_or(store.sp.zero);
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

        let output_spp = aut.output(&q);
        let output_sp = store.push(diff, output_spp);
        if output_sp != store.sp.zero {
            return Some(reconstruct_trace(
                &steps, step_idx, aut, store, output_spp, &start,
            ));
        }

        for (spp, q_next) in aut.transitions(&q) {
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
    start: &A::State,
) -> (Vec<bool>, Vec<Vec<bool>>, Vec<bool>) {
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
    let mut current_idx = accepting_idx;
    let mut current_pkt = acc_pkt;

    while &steps[current_idx].state != start {
        let preds = steps[current_idx].predecessors.clone();
        let target = steps[current_idx].state.clone();
        let current_pkt_sp = singleton_sp(store, &current_pkt);

        let mut found: Option<(usize, Vec<bool>)> = None;
        'outer: for j in preds {
            let q_j = steps[j].state.clone();
            let d_j = steps[j].diff;

            for (spp, q_next) in aut.transitions(&q_j) {
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
        current_idx = j;
        current_pkt = p_j;
    }

    packets_back.reverse();
    let start_pkt = packets_back.remove(0);
    (start_pkt, packets_back, out_pkt)
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

/// True iff `(input, output)` is a packet pair related by `spp`.
fn spp_accepts(store: &mut spp::SPPstore, spp: spp::SPP, input: &[bool], output: &[bool]) -> bool {
    let sp_in = singleton_sp(store, input);
    let pushed = store.push(sp_in, spp);
    let sp_out = singleton_sp(store, output);
    let intersected = store.sp.intersect(pushed, sp_out);
    intersected != store.sp.zero
}

#[cfg(test)]
mod tests {
    use super::*;

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
                let accepted = dfa.dfa_accepts(aut.spp_store_mut(), &input, &trace_pkts, &output);
                assert!(
                    accepted,
                    "trace not accepted by DFA on trial {} for expr {}: input={:?}, trace={:?}, output={:?}",
                    trial, expr, input, trace_pkts, output
                );
            }
        }
    }
}
