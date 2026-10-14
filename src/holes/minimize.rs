//! Minimize -- heuristically minimize netkat automata.
//!
//! This has four parts:
//!
//!  1. Identify conflicts
//!  2. Graph coloring
//!  3. Merge the states!
//!  4. Simplify the edges, alternating between [`reduce_forward`] and [`reduce_backward`]. The
//!     edges may then overlap, so the result is an NFA.

use crate::expr::{Exp, Expr};
use crate::holes::aut::{self, DFA, ENFA, ExplicitDFA, ExplicitNFA, NFA, ops};
use crate::sp;
use crate::spp::{self, reduce};
use petgraph::algo::{coloring, dominators, tarjan_scc};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::unionfind::UnionFind;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Step 1 -- taking an automaton and identifying (un)mergable states
///
/// Two states are mergable if any packet that reaches both of them has the same behavior when
/// progressing from either of them.
///
/// Concretely, we take the intersection `R` of the reaching sets of both, and check that the
/// automata rooted at `q` and at `r` (with start packets restricted to `R`) have empty symmetric
/// difference.
fn are_mergable(
    dfa: &ExplicitDFA,
    reachable_sets: &HashMap<usize, sp::SP>,
    store: &mut spp::SPPstore,
    q: usize,
    r: usize,
) -> bool {
    let reach_q = reachable_sets.get(&q).copied().unwrap_or(store.sp.zero);
    let reach_r = reachable_sets.get(&r).copied().unwrap_or(store.sp.zero);
    let shared = store.sp.intersect(reach_q, reach_r);
    if shared == store.sp.zero {
        return true;
    }

    let filter = store.ibwd(shared);
    let at_q = Rooted {
        dfa,
        root: q,
        filter,
    };
    let at_r = Rooted {
        dfa,
        root: r,
        filter,
    };
    let sym_diff = ops::union(
        ops::intersection(&at_q, ops::complement(&at_r)),
        ops::intersection(ops::complement(&at_q), &at_r),
    );
    aut::is_empty(&sym_diff, store)
}

/// State of [`Rooted`]: the fresh root, or a state of the underlying DFA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RootedState {
    Root,
    Inner(usize),
}

/// `dfa`, but starting at state `root`, and only accepting start packets in the input set of
/// `filter` (an SPP of the form `ibwd(sp)`).
///
/// The root is a fresh copy of `root`, so that if the run later comes back to `root` it is no
/// longer restricted.
struct Rooted<'a> {
    dfa: &'a ExplicitDFA,
    root: usize,
    filter: spp::SPP,
}

impl ENFA for Rooted<'_> {
    type State = RootedState;

    fn start(&self, _store: &mut spp::SPPstore) -> RootedState {
        RootedState::Root
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, _q: &RootedState) -> bool {
        true
    }

    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &RootedState,
    ) -> Vec<(spp::SPP, RootedState)> {
        match *q {
            RootedState::Root => self.dfa.transitions[self.root]
                .iter()
                .filter_map(|&(spp, next)| {
                    let restricted = store.intersect(self.filter, spp);
                    (restricted != store.zero).then_some((restricted, RootedState::Inner(next)))
                })
                .collect(),
            RootedState::Inner(s) => self.dfa.transitions[s]
                .iter()
                .map(|&(spp, next)| (spp, RootedState::Inner(next)))
                .collect(),
        }
    }

    fn output(&self, store: &mut spp::SPPstore, q: &RootedState) -> spp::SPP {
        match *q {
            RootedState::Root => store.intersect(self.filter, self.dfa.outputs[self.root]),
            RootedState::Inner(s) => self.dfa.outputs[s],
        }
    }
}

impl NFA for Rooted<'_> {}
impl DFA for Rooted<'_> {}

/// Step 2 -- graph coloring.
///
/// Construct a graph with an edge indicating that two states *cannot* be merged, and compute a
/// coloring. This will be the new small set of states: one new state per color.
///
/// Dead states (that no packet reaches) are discarded: they get no color. The merged automaton never
/// has an edge to them anyway, since its edges only carry packets that reach their target.
///
/// Return (coloring, number of colors).
fn coloring(
    dfa: &ExplicitDFA,
    reachable_sets: &HashMap<usize, sp::SP>,
    store: &mut spp::SPPstore,
) -> (HashMap<usize, usize>, usize) {
    let live: Vec<usize> = (0..dfa.num_states())
        .filter(|q| reachable_sets.get(q).is_some_and(|&r| r != store.sp.zero))
        .collect();

    // Node `i` of the conflict graph is state `live[i]`. Add every node up front: `from_edges`
    // alone would drop states with no conflicts
    let mut conflicts: UnGraph<(), (), usize> = UnGraph::default();
    for _ in &live {
        conflicts.add_node(());
    }
    conflicts.extend_with_edges(
        (0..live.len())
            .flat_map(|i| (0..i).map(move |j| (i, j)))
            .filter(|&(i, j)| !are_mergable(dfa, reachable_sets, store, live[i], live[j])),
    );

    let (coloring, _) = coloring::dsatur_coloring(&conflicts);
    let coloring: HashMap<usize, usize> = coloring
        .into_iter()
        .map(|(id, c)| (live[id.index()], c))
        .collect();

    split_colors(&live, reachable_sets, store, &coloring)
}

/// Split each color into groups of states with disjoint reachability sets.
///
/// If the states of one color can be partitioned into `Q1` and `Q2` such that no packet reaches
/// both a state in `Q1` and a state in `Q2`, then there's no need to merge them, so we give `Q1`
/// and `Q2` separate colors. Concretely, the groups are the connected components of the "reaching
/// sets overlap" relation within each color.
///
/// Return (coloring, number of colors).
fn split_colors(
    live: &[usize],
    reachable_sets: &HashMap<usize, sp::SP>,
    store: &mut spp::SPPstore,
    coloring: &HashMap<usize, usize>,
) -> (HashMap<usize, usize>, usize) {
    // Union-find over indices into `live`
    let mut groups = UnionFind::<usize>::new(live.len());
    for (i, &q) in live.iter().enumerate() {
        for (j, &r) in live[..i].iter().enumerate() {
            if coloring[&q] == coloring[&r]
                && store.sp.intersect(reachable_sets[&q], reachable_sets[&r]) != store.sp.zero
            {
                groups.union(i, j);
            }
        }
    }

    // Number the groups densely, in order of their first state
    let mut group_color: HashMap<usize, usize> = HashMap::new();
    let new_coloring: HashMap<usize, usize> = live
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let next = group_color.len();
            (q, *group_color.entry(groups.find_mut(i)).or_insert(next))
        })
        .collect();
    (new_coloring, group_color.len())
}

/// Step 3 -- building the smaller automaton
///
/// Each color becomes one state. For each incoming packet, the new state behaves like one of the
/// old states of that color which that packet can reach (the first one, in state order). Since
/// states of the same color are pairwise mergable, it doesn't matter which one we pick, and picking
/// just one keeps the transitions disjoint, so the result is still a DFA.
fn merge_states(
    dfa: &ExplicitDFA,
    reachable_sets: &HashMap<usize, sp::SP>,
    store: &mut spp::SPPstore,
) -> ExplicitDFA {
    let (coloring, colors) = coloring(dfa, reachable_sets, store);

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); colors];
    for q in 0..dfa.num_states() {
        if let Some(&c) = coloring.get(&q) {
            members[c].push(q);
        }
    }

    let mut transitions = Vec::with_capacity(colors);
    let mut outputs = Vec::with_capacity(colors);
    for states in &members {
        // Packets already handled by an earlier member of this color
        let mut covered = store.sp.zero;
        // Transitions, merged by target color
        let mut trans: HashMap<usize, spp::SPP> = HashMap::new();
        let mut output = store.zero;

        for &q in states {
            let reach_q = reachable_sets.get(&q).copied().unwrap_or(store.sp.zero);
            let owned = store.sp.difference(reach_q, covered);
            if owned == store.sp.zero {
                continue;
            }
            covered = store.sp.union(covered, owned);
            let filter = store.ibwd(owned);

            for &(spp, next) in &dfa.transitions[q] {
                let restricted = store.intersect(filter, spp);
                if restricted != store.zero {
                    // Some packet reaching `q` takes this edge, so `next` is not dead
                    let target = coloring[&next];
                    let prev = trans.get(&target).copied().unwrap_or(store.zero);
                    let merged = store.union(prev, restricted);
                    trans.insert(target, merged);
                }
            }
            let out = store.intersect(filter, dfa.outputs[q]);
            output = store.union(output, out);
        }

        let mut trans: Vec<(spp::SPP, usize)> = trans
            .into_iter()
            .map(|(target, spp)| (spp, target))
            .collect();
        trans.sort_by_key(|&(_, target)| target);

        transitions.push(trans);
        outputs.push(output);
    }

    ExplicitDFA {
        start: coloring[&dfa.start],
        transitions,
        outputs,
    }
}

/// Step 4a -- simplify each edge and output, keeping only what matters for packets that reach it.
///
/// An edge out of `q` (and the output of `q`) only ever sees packets that reach `q`, so it can be
/// anything on other input packets. We [`reduce`](reduce::reduce) it with that don't-care set.
///
/// This preserves the language, and no packet reaches a state it didn't before: by induction along a
/// run, every packet reaching `q` is one the edges out of `q` are kept exact on.
pub fn reduce_forward(nfa: &ExplicitNFA, store: &mut spp::SPPstore) -> ExplicitNFA {
    let reachable = aut::compute_reachable(nfa, store);
    let care: Vec<spp::SPP> = (0..nfa.num_states())
        .map(|q| {
            let reach_q = reachable.get(&q).copied().unwrap_or(store.sp.zero);
            store.ibwd(reach_q)
        })
        .collect();

    let zero = store.zero;
    let transitions = (0..nfa.num_states())
        .map(|q| {
            nfa.transitions[q]
                .iter()
                .map(|&(spp, next)| (reduce::reduce(store, spp, care[q]), next))
                .filter(|&(spp, _)| spp != zero)
                .collect()
        })
        .collect();
    let outputs = (0..nfa.num_states())
        .map(|q| reduce::reduce(store, nfa.outputs[q], care[q]))
        .collect();

    ExplicitNFA {
        start: nfa.start,
        transitions,
        outputs,
    }
}

/// Step 4b -- the mirror image of [`reduce_forward`]: simplify each edge, keeping only what matters
/// for packets that go on to be accepted.
///
/// An edge into `r` only matters for the packets it produces that co-reach `r` (can go on from `r` to
/// be accepted), so it can be anything on other output packets. The outputs are left alone: they
/// are where acceptance happens, like the start state is where [`reduce_forward`]'s runs begin.
///
/// This preserves the language, and no packet co-reaches a state it didn't before: by induction
/// along a run (backwards from acceptance), every edge it takes is one kept exact on its packets.
pub fn reduce_backward(nfa: &ExplicitNFA, store: &mut spp::SPPstore) -> ExplicitNFA {
    let coreachable = aut::compute_coreachable(nfa, store);
    let care: Vec<spp::SPP> = (0..nfa.num_states())
        .map(|r| {
            let coreach_r = coreachable.get(&r).copied().unwrap_or(store.sp.zero);
            store.ifwd(coreach_r)
        })
        .collect();

    let zero = store.zero;
    let transitions = (0..nfa.num_states())
        .map(|q| {
            nfa.transitions[q]
                .iter()
                .map(|&(spp, next)| (reduce::reduce(store, spp, care[next]), next))
                .filter(|&(spp, _)| spp != zero)
                .collect()
        })
        .collect();

    ExplicitNFA {
        start: nfa.start,
        transitions,
        outputs: nfa.outputs.clone(),
    }
}

/// How many rounds of [`reduce_forward`] and [`reduce_backward`] [`minimize`] does at most. Each
/// round can shrink the reachability or co-reachability sets, which lets the next round simplify
/// further.
const REDUCE_ROUNDS: usize = 4;

/// Heuristically minimize a DFA: merge compatible states, then simplify the edges.
///
/// The result is an NFA, since simplifying the edges can make them overlap.
pub fn minimize(dfa: &ExplicitDFA, store: &mut spp::SPPstore) -> ExplicitNFA {
    let reachable = aut::compute_reachable(dfa, store);
    let mut nfa = ExplicitNFA::from(merge_states(dfa, &reachable, store));
    for _ in 0..REDUCE_ROUNDS {
        let forward = reduce_forward(&nfa, store);
        let next = reduce_backward(&forward, store);
        if next == nfa {
            break;
        }
        nfa = next;
    }
    nfa
}

/// Convert a DFA to an equivalent NetKAT expression: [`minimize`] it, then use [`to_expr_tarjan`].
///
/// In simple testing (the `minimize_shrinks_exprs` test), this was the best configuration:
/// minimizing first never gave a larger expression, and on bigger random inputs it made them several
/// times smaller. After minimizing, Tarjan's algorithm gave slightly smaller expressions than Kleene
/// state elimination.
pub fn to_expr(dfa: &ExplicitDFA, store: &mut spp::SPPstore) -> Exp {
    let small = minimize(dfa, store);
    to_expr_tarjan(&small, store)
}

/// Convert an NFA to an equivalent NetKAT expression, by Kleene state elimination.
///
/// We build the edge-labelled NFA described in [`ExpNfa::from_nfa`], then eliminate the automaton's
/// states one at a time, until only the edge from initial to final is left.
///
/// States are eliminated greedily, picking the one that increases the number of edges the least.
/// Eliminating `q` adds an edge for each (incoming, outgoing) pair, and removes the incoming,
/// outgoing, and self-loop edges of `q`.
pub fn to_expr_kleene(aut: &ExplicitNFA, store: &mut spp::SPPstore) -> Exp {
    let mut nfa = ExpNfa::from_nfa(aut, store);

    let mut remaining: BTreeSet<usize> = (0..aut.num_states()).collect();
    while let Some(q) = remaining
        .iter()
        .copied()
        .min_by_key(|&q| nfa.elimination_cost(q))
    {
        nfa.eliminate(q);
        remaining.remove(&q);
    }

    nfa.edges[nfa.initial]
        .remove(&nfa.fin)
        .unwrap_or_else(Expr::zero)
}

/// Convert an NFA to an equivalent NetKAT expression, by Tarjan's path expression algorithm.
///
/// We build the edge-labelled NFA described in [`ExpNfa::from_nfa`], and compute the path
/// expression from its initial state to its final state. Following Tarjan ("Fast algorithms for
/// solving path problems", 1981), we use the dominator tree to split the problem into one small
/// problem per node of the tree:
///
/// - `path(q)` is all paths from the initial state to `q`. Every such path passes through
///   `idom(q)`, and after its last visit to `idom(q)` it stays among the nodes dominated by
///   `idom(q)`. So `path(q) = path(idom(q)); dpath(q)`, where `dpath(q)` is all paths from `idom(q)`
///   to `q` that stay strictly below `idom(q)`. Hence `path(final)` is the product of `dpath`
///   down the dominator tree.
/// - `dpath` is computed for the children of each node `v` together, bottom-up (see
///   [`solve_siblings`]).
pub fn to_expr_tarjan(aut: &ExplicitNFA, store: &mut spp::SPPstore) -> Exp {
    let nfa = ExpNfa::from_nfa(aut, store);
    let num_nodes = nfa.edges.len();

    let mut graph: DiGraph<(), (), usize> = DiGraph::default();
    for _ in 0..num_nodes {
        graph.add_node(());
    }
    for (p, out) in nfa.edges.iter().enumerate() {
        for &r in out.keys() {
            graph.add_edge(NodeIndex::new(p), NodeIndex::new(r), ());
        }
    }
    let doms = dominators::simple_fast(&graph, NodeIndex::new(nfa.initial));
    // `None` for the initial state and for unreachable states
    let idom: Vec<Option<usize>> = (0..num_nodes)
        .map(|q| {
            doms.immediate_dominator(NodeIndex::new(q))
                .map(|d| d.index())
        })
        .collect();
    if idom[nfa.fin].is_none() {
        return Expr::zero();
    }

    let mut children = vec![Vec::new(); num_nodes];
    for (q, &d) in idom.iter().enumerate() {
        if let Some(d) = d {
            children[d].push(q);
        }
    }

    // Post-order, so that `dpath` is known for every node strictly below `v` when we get to `v`
    let mut dpath = vec![Expr::zero(); num_nodes];
    for v in post_order(&children, nfa.initial) {
        solve_siblings(&nfa, &idom, v, &children[v], &mut dpath);
    }

    let mut chain = vec![];
    let mut q = nfa.fin;
    while q != nfa.initial {
        chain.push(q);
        q = idom[q].unwrap();
    }
    chain.iter().rev().fold(Expr::one(), |acc, &q| {
        Expr::sequence_simp(acc, dpath[q].clone())
    })
}

/// Compute `dpath` for the dominator tree children `kids` of `v`, given `dpath` for every node
/// below them.
///
/// A path from `v` to a child `c` that stays below `v` starts with an edge from `v` to some child
/// `c'`, then moves between the subtrees of the children. It can only enter a child's subtree at
/// the child itself (since the child dominates its subtree), so it is a sequence of hops `c' ~> c''`,
/// each of which wanders around the subtree of `c'` and then takes an edge to `c''`. The hops are
/// the edges of the *derived graph* on `kids`, and the hops from `c'` that end with the edge
/// `u -> c''` are `dpath(D2); ...; dpath(Dk); dpath(u); label(u -> c'')`, where
/// `c', D2, ..., Dk, u` is the dominator tree path from `c'` down to `u`.
///
/// This gives the (left-linear) system `dpath(c) = label(v -> c) + Σ_c' dpath(c'); hops(c' -> c)`.
/// We solve it by Gauss-Jordan elimination, visiting the children in topological order of the
/// derived graph's strongly connected components. So we only need Kleene stars for the children
/// that are actually in a cycle of the derived graph; for the others, elimination just substitutes
/// forwards.
fn solve_siblings(
    nfa: &ExpNfa,
    idom: &[Option<usize>],
    v: usize,
    kids: &[usize],
    dpath: &mut [Exp],
) {
    let k = kids.len();
    let index: HashMap<usize, usize> = kids.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // `b[j]` is the edge from `v` to `kids[j]`, and `a[i][j]` is the hops from `kids[i]` to `kids[j]`
    let mut b: Vec<Exp> = kids
        .iter()
        .map(|c| nfa.edges[v].get(c).cloned().unwrap_or_else(Expr::zero))
        .collect();
    let mut a: Vec<Vec<Exp>> = vec![vec![Expr::zero(); k]; k];
    for (j, &c) in kids.iter().enumerate() {
        for &u in &nfa.incoming[c] {
            // Skip the edge from `v` (already in `b`), and edges from unreachable states
            if u == v || idom[u].is_none() {
                continue;
            }
            let mut hop = nfa.edges[u][&c].clone();
            let mut w = u;
            while idom[w] != Some(v) {
                hop = Expr::sequence_simp(dpath[w].clone(), hop);
                w = idom[w].unwrap();
            }
            let i = index[&w];
            a[i][j] = Expr::union_simp(std::mem::replace(&mut a[i][j], Expr::zero()), hop);
        }
    }

    // `tarjan_scc` lists the components in reverse topological order
    let mut derived: DiGraph<(), (), usize> = DiGraph::default();
    for _ in 0..k {
        derived.add_node(());
    }
    for (i, row) in a.iter().enumerate() {
        for (j, hop) in row.iter().enumerate() {
            if **hop != Expr::Zero {
                derived.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            }
        }
    }
    let order: Vec<usize> = tarjan_scc(&derived)
        .into_iter()
        .rev()
        .flatten()
        .map(|n| n.index())
        .collect();

    let is_zero = |e: &Exp| **e == Expr::Zero;
    for &i in &order {
        // Solve equation `i` for `x_i` by Arden's rule: `x_i = (b_i + Σ_{j≠i} x_j; a_ji); a_ii*`
        let loop_star = Expr::star_simp(std::mem::replace(&mut a[i][i], Expr::zero()));
        if *loop_star != Expr::One {
            b[i] = Expr::sequence_simp(b[i].clone(), loop_star.clone());
            for row in a.iter_mut() {
                if !is_zero(&row[i]) {
                    row[i] = Expr::sequence_simp(row[i].clone(), loop_star.clone());
                }
            }
        }
        // Substitute `x_i` into every other equation that mentions it
        for l in 0..k {
            if l == i || is_zero(&a[i][l]) {
                continue;
            }
            let a_il = std::mem::replace(&mut a[i][l], Expr::zero());
            let via_i = Expr::sequence_simp(b[i].clone(), a_il.clone());
            b[l] = Expr::union_simp(b[l].clone(), via_i);
            for (j, row) in a.iter_mut().enumerate() {
                if j != i && !is_zero(&row[i]) {
                    let via_i = Expr::sequence_simp(row[i].clone(), a_il.clone());
                    row[l] = Expr::union_simp(row[l].clone(), via_i);
                }
            }
        }
    }

    for (j, &c) in kids.iter().enumerate() {
        dpath[c] = std::mem::replace(&mut b[j], Expr::zero());
    }
}

/// Post-order traversal of the tree given by `children`, starting from `root`.
fn post_order(children: &[Vec<usize>], root: usize) -> Vec<usize> {
    let mut order = vec![];
    let mut stack = vec![(root, false)];
    while let Some((q, expanded)) = stack.pop() {
        if expanded {
            order.push(q);
        } else {
            stack.push((q, true));
            stack.extend(children[q].iter().map(|&c| (c, false)));
        }
    }
    order
}

/// An NFA with edges labelled by expressions, for [`to_expr_kleene`] and [`to_expr_tarjan`].
/// Parallel edges are merged into one by union, so there is at most one edge between each
/// (ordered) pair of states.
struct ExpNfa {
    /// `edges[p][r]` is the label of the edge from `p` to `r`.
    edges: Vec<BTreeMap<usize, Exp>>,
    /// `incoming[r]` is the set of `p` with an edge from `p` to `r`.
    incoming: Vec<BTreeSet<usize>>,
    /// A fresh state with no incoming edges.
    initial: usize,
    /// A fresh state with no outgoing edges.
    fin: usize,
}

impl ExpNfa {
    /// A transition `(spp, next)` means "apply `spp`, record the new packet in the trace, then
    /// continue from `next`", so each state `q` denotes
    ///
    /// ```text
    /// E_q = output_q + Σ spp; dup; E_next
    /// ```
    ///
    /// So the NFA has the automaton's states, plus a fresh initial state with a `1` edge to the start
    /// state, an edge `spp; dup` per transition, and an edge `output_q` from each state to a fresh
    /// final state. The automaton denotes the sum of all paths from initial to final.
    fn from_nfa(aut: &ExplicitNFA, store: &mut spp::SPPstore) -> ExpNfa {
        let n = aut.num_states();
        let mut nfa = ExpNfa {
            edges: vec![BTreeMap::new(); n + 2],
            incoming: vec![BTreeSet::new(); n + 2],
            initial: n,
            fin: n + 1,
        };

        nfa.add_edge(nfa.initial, aut.start, Expr::one());
        for q in 0..n {
            for &(spp, next) in &aut.transitions[q] {
                let label = Expr::sequence_simp(store.to_expr(spp), Expr::dup());
                nfa.add_edge(q, next, label);
            }
            let output = store.to_expr(aut.outputs[q]);
            nfa.add_edge(q, nfa.fin, output);
        }
        nfa
    }

    fn add_edge(&mut self, from: usize, to: usize, label: Exp) {
        if *label == Expr::Zero {
            return;
        }
        let label = match self.edges[from].remove(&to) {
            Some(prev) => Expr::union_simp(prev, label),
            None => label,
        };
        self.edges[from].insert(to, label);
        self.incoming[to].insert(from);
    }

    /// Change in the number of edges from eliminating `q`.
    fn elimination_cost(&self, q: usize) -> isize {
        let has_loop = self.edges[q].contains_key(&q);
        let num_in = self.incoming[q].len() - has_loop as usize;
        let num_out = self.edges[q].len() - has_loop as usize;
        (num_in * num_out) as isize - (num_in + num_out + has_loop as usize) as isize
    }

    /// Remove `q`, replacing each path `p -> q -> r` by an edge `p -> r`.
    fn eliminate(&mut self, q: usize) {
        let mut outgoing = std::mem::take(&mut self.edges[q]);
        let self_loop = outgoing
            .remove(&q)
            .map(Expr::star_simp)
            .unwrap_or_else(Expr::one);
        let incoming = std::mem::take(&mut self.incoming[q]);

        for &r in outgoing.keys() {
            self.incoming[r].remove(&q);
        }
        for &p in &incoming {
            if p == q {
                continue;
            }
            let into_q = self.edges[p].remove(&q).unwrap();
            for (&r, out_of_q) in &outgoing {
                let label = Expr::sequence_simp(
                    Expr::sequence_simp(into_q.clone(), self_loop.clone()),
                    out_of_q.clone(),
                );
                self.add_edge(p, r, label);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holes::aut::SubsetDfa;

    /// True iff `a` and `b` accept exactly the same traces.
    fn equivalent(store: &mut spp::SPPstore, a: &ExplicitNFA, b: &ExplicitNFA) -> bool {
        // Complementing needs a DFA, so determinize first
        let sym_diff = ops::union(
            ops::intersection(SubsetDfa::new(a), ops::complement(SubsetDfa::new(b))),
            ops::intersection(ops::complement(SubsetDfa::new(a)), SubsetDfa::new(b)),
        );
        aut::is_empty(&sym_diff, store)
    }

    fn to_nfa(dfa: &ExplicitDFA) -> ExplicitNFA {
        ExplicitNFA::from(dfa.clone())
    }

    /// Round trip `expr -> DFA -> expr -> DFA` using `to_expr`, and check the language is
    /// preserved.
    fn fuzz_to_expr(seed: u64, to_expr: impl Fn(&ExplicitDFA, &mut spp::SPPstore) -> Exp) {
        crate::fuzz::seed_fuzzer(seed);
        let expr_depth = 3;
        let num_fields = 2;

        for trial in 0..200 {
            let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);
            let mut store = spp::SPPstore::new(num_fields);
            let dfa = aut::expr_to_dfa(&expr, &mut store);

            let back = to_expr(&dfa, &mut store);
            let dfa_back = aut::expr_to_dfa(&back, &mut store);
            assert!(
                equivalent(&mut store, &to_nfa(&dfa), &to_nfa(&dfa_back)),
                "to_expr changed the language on trial {trial}\n  expr: {expr}\n  back: {back}"
            );
        }
    }

    #[test]
    fn fuzz_to_expr_roundtrip() {
        fuzz_to_expr(0x5EED_0015, to_expr);
    }

    #[test]
    fn fuzz_to_expr_kleene_roundtrip() {
        fuzz_to_expr(0x5EED_0011, |dfa, store| {
            to_expr_kleene(&to_nfa(dfa), store)
        });
    }

    #[test]
    fn fuzz_minimize_to_expr_kleene_roundtrip() {
        fuzz_to_expr(0x5EED_0012, |dfa, store| {
            let small = minimize(dfa, store);
            to_expr_kleene(&small, store)
        });
    }

    #[test]
    fn fuzz_to_expr_tarjan_roundtrip() {
        fuzz_to_expr(0x5EED_0013, |dfa, store| {
            to_expr_tarjan(&to_nfa(dfa), store)
        });
    }

    #[test]
    fn fuzz_minimize_to_expr_tarjan_roundtrip() {
        fuzz_to_expr(0x5EED_0014, |dfa, store| {
            let small = minimize(dfa, store);
            to_expr_tarjan(&small, store)
        });
    }

    #[test]
    fn fuzz_minimize_preserves_language() {
        crate::fuzz::seed_fuzzer(0x5EED_0010);
        let expr_depth = 4;
        let num_fields = 3;

        for trial in 0..300 {
            let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);
            let mut aut = crate::aut::Aut::new(num_fields);
            let state = aut.expr_to_state(&expr);
            let dfa = aut::aut_to_dfa(&mut aut, state);
            let store = aut.spp_store_mut();

            let small = minimize(&dfa, store);
            assert!(
                small.num_states() <= dfa.num_states(),
                "minimize grew the automaton on trial {trial} for expr {expr}"
            );
            assert!(
                equivalent(store, &to_nfa(&dfa), &small),
                "minimize changed the language on trial {trial} for expr {expr}"
            );
        }
    }

    /// Each reduce pass on its own preserves the language.
    #[test]
    fn fuzz_reduce_passes_preserve_language() {
        crate::fuzz::seed_fuzzer(0x5EED_0016);
        for trial in 0..300 {
            let (expr, _) = crate::fuzz::genax(0, 4, 3);
            let mut store = spp::SPPstore::new(3);
            let nfa = to_nfa(&aut::expr_to_dfa(&expr, &mut store));

            let forward = reduce_forward(&nfa, &mut store);
            assert!(
                equivalent(&mut store, &nfa, &forward),
                "reduce_forward changed the language on trial {trial} for expr {expr}"
            );
            let backward = reduce_backward(&nfa, &mut store);
            assert!(
                equivalent(&mut store, &nfa, &backward),
                "reduce_backward changed the language on trial {trial} for expr {expr}"
            );
        }
    }

    /// The solution `nksynth --full` finds for `examples/chained_dup.nksynth`: after the backward
    /// pass sees that only `x0 = 1` is accepted, the havoc becomes an assignment, and then the
    /// forward pass sees that the test always passes.
    #[test]
    fn minimize_simplifies_havoc_then_test() {
        let expr = crate::parser::Parser::new(crate::parser::Lexer::new(
            "(x0 := 0 + x0 := 1); dup; x0 == 1",
        ))
        .parse_single_expression()
        .unwrap();
        let mut store = spp::SPPstore::new(1);
        let dfa = aut::expr_to_dfa(&expr, &mut store);
        let simplified = to_expr(&dfa, &mut store);
        assert_eq!(crate::printer::pretty(&simplified), "x0 := 1; dup");
    }

    /// Number of nodes in the syntax tree of `e`.
    fn expr_size(e: &Expr) -> usize {
        match e {
            Expr::Union(a, b)
            | Expr::Intersect(a, b)
            | Expr::Xor(a, b)
            | Expr::Difference(a, b)
            | Expr::Sequence(a, b)
            | Expr::LtlUntil(a, b) => 1 + expr_size(a) + expr_size(b),
            Expr::Star(a) | Expr::Complement(a) | Expr::TestNegation(a) | Expr::LtlNext(a) => {
                1 + expr_size(a)
            }
            Expr::IfThenElse(a, b, c) => 1 + expr_size(a) + expr_size(b) + expr_size(c),
            _ => 1,
        }
    }

    type ToExpr = fn(&ExplicitNFA, &mut spp::SPPstore) -> Exp;

    /// Does converting to an expression give smaller results after minimizing?
    ///
    /// Not a real test, just prints statistics. Run with
    /// `cargo test --release --lib minimize_shrinks_exprs -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn minimize_shrinks_exprs() {
        let algorithms: [(&str, ToExpr); 2] =
            [("kleene", to_expr_kleene), ("tarjan", to_expr_tarjan)];
        for (expr_depth, num_fields) in [(3, 2), (4, 3), (5, 3)] {
            crate::fuzz::seed_fuzzer(0x5EED_0020);
            let trials = 300;
            let mut states = [0, 0];
            let mut totals = [[0usize; 2]; 2];
            let mut smaller = [0; 2];
            let mut larger = [0; 2];
            let mut log_ratio = [0.0f64; 2];

            for _ in 0..trials {
                let (expr, _) = crate::fuzz::genax(0, expr_depth, num_fields);
                let mut store = spp::SPPstore::new(num_fields);
                let dfa = aut::expr_to_dfa(&expr, &mut store);
                let small = minimize(&dfa, &mut store);
                let dfa = to_nfa(&dfa);
                states[0] += dfa.num_states();
                states[1] += small.num_states();

                for (k, (_, to_expr)) in algorithms.iter().enumerate() {
                    let before = expr_size(&to_expr(&dfa, &mut store));
                    let after = expr_size(&to_expr(&small, &mut store));
                    totals[k][0] += before;
                    totals[k][1] += after;
                    smaller[k] += (after < before) as usize;
                    larger[k] += (after > before) as usize;
                    log_ratio[k] += (after as f64 / before as f64).ln();
                }
            }

            println!(
                "depth {expr_depth}, {num_fields} fields, {trials} trials: states {} -> {}",
                states[0], states[1]
            );
            for (k, (name, _)) in algorithms.iter().enumerate() {
                println!(
                    "  {name}: total size {} -> {}, smaller in {}, larger in {}, \
                     geometric mean ratio {:.3}",
                    totals[k][0],
                    totals[k][1],
                    smaller[k],
                    larger[k],
                    (log_ratio[k] / trials as f64).exp()
                );
            }
        }
    }
}
