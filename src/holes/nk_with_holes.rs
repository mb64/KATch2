//! NetKAT expressions extended with *holes*: formal variables that stand for
//! an unknown program. This module compiles such an expression into a special
//! automaton whose edges are each either a *concrete* edge (labelled by a
//! known SPP) or an *abstract* edge (labelled by a hole).  Multi-piece
//! transitions like `p ; H ; p'` are broken up by inserting invisible
//! intermediate states so every edge keeps the same simple two-variant
//! shape; the Antimirov derivative below is what does the splitting.
//!
//! The public API closely mirrors the `ENFA` trait — `transitions`,
//! `output`, `is_visible` — except that edges carry an [`EdgeLabel`]
//! (`Concrete(SPP)` or `Abstract(Hole)`) rather than a single SPP.
//!
//! The SPP store is threaded as a `&mut spp::SPPstore` parameter rather
//! than owned by the automaton, so callers can hold mutable borrows of
//! both at once.
//!
//! To use it, you can fill in the holes using `inst::Instantiate`.

use crate::hash::{HashMap, HashMapExt as _};
use crate::spp;

/// A formal hole standing for an unknown SPP.  The canonical definition now
/// lives in [`crate::expr`]; re-exported here so existing `nk_with_holes::Hole`
/// references keep working.
pub use crate::expr::Hole;

/// User-facing NetKAT-with-holes expression.
///
/// Concrete predicates and assignments are expected to be compiled into
/// `Spp(_)` ahead of time; this enum only carries the higher-level
/// combinators plus `Hole` and `Dup`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Spp(spp::SPP),
    Hole(Hole),
    Dup,
    Union(Box<Expr>, Box<Expr>),
    Sequence(Box<Expr>, Box<Expr>),
    Star(Box<Expr>),
}

impl Expr {
    pub fn spp(s: spp::SPP) -> Self {
        Expr::Spp(s)
    }
    pub fn hole(h: Hole) -> Self {
        Expr::Hole(h)
    }
    pub fn dup() -> Self {
        Expr::Dup
    }
    pub fn union(e1: Expr, e2: Expr) -> Self {
        Expr::Union(Box::new(e1), Box::new(e2))
    }
    pub fn sequence(e1: Expr, e2: Expr) -> Self {
        Expr::Sequence(Box::new(e1), Box::new(e2))
    }
    pub fn star(e: Expr) -> Self {
        Expr::Star(Box::new(e))
    }

    /// Recursively collapse fully-concrete subterms into single SPPs:
    /// `Sequence(Spp, Spp)`, `Union(Spp, Spp)`, and `Star(Spp)` fold via
    /// the corresponding [`spp::SPPstore`] operations.
    pub fn simplify(&self, store: &mut spp::SPPstore) -> Expr {
        match self {
            Expr::Spp(_) | Expr::Hole(_) | Expr::Dup => self.clone(),
            Expr::Union(a, b) => match (a.simplify(store), b.simplify(store)) {
                (Expr::Spp(a), Expr::Spp(b)) => Expr::Spp(store.union(a, b)),
                (a, b) => Expr::union(a, b),
            },
            Expr::Sequence(a, b) => match (a.simplify(store), b.simplify(store)) {
                (Expr::Spp(a), Expr::Spp(b)) => Expr::Spp(store.sequence(a, b)),
                (a, b) => Expr::sequence(a, b),
            },
            Expr::Star(e) => match e.simplify(store) {
                Expr::Spp(s) => Expr::Spp(store.star(s)),
                e => Expr::star(e),
            },
        }
    }
}

/// One edge of the automaton: either a concrete SPP or a hole to be filled
/// with a candidate program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeLabel {
    Concrete(spp::SPP),
    Abstract(Hole),
}

/// Hash-consed internal representation of NetKAT-with-holes expressions.
/// `Union` and `Sequence` are n-ary so the smart constructors can flatten
/// nested compositions naturally (same pattern as `crate::aut`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AExpr {
    Spp(spp::SPP),
    Hole(Hole),
    Dup,
    Union(Vec<AExprIdx>),
    Sequence(Vec<AExprIdx>),
    Star(AExprIdx),
}

/// Index into the hash-consed expression table.
pub type AExprIdx = usize;

/// A state of the automaton: a hash-consed expression paired with its
/// visibility flag.  The same expression can appear with either visibility
/// depending on whether it was reached via a within-segment edge or a
/// dup-crossing edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct State {
    pub aexpr: AExprIdx,
    pub visible: bool,
}

/// Automaton for NetKAT-with-holes.
///
/// Owns hash-consed expressions and lazy Antimirov derivatives. Constraint
/// compilation calls [`Self::preprocess_epsilon`] once to materialize concrete
/// epsilon shortcuts shared by all subsequent hole assignments and restarts.
/// The SPP store is supplied externally.
#[derive(Debug, Clone)]
pub struct AutWithHoles {
    aexprs: Vec<AExpr>,
    aexpr_map: HashMap<AExpr, AExprIdx>,
    /// Caches keyed by `AExprIdx`: the derivative recipe doesn't depend on
    /// the source state's visibility (only on its `AExpr`), so we cache
    /// once per expression and reuse across visibilities.
    delta_cache: HashMap<AExprIdx, Vec<(EdgeLabel, State)>>,
    // Elimination depends on visibility and incoming edges, so these results
    // must be keyed by the full state rather than just its expression.
    preprocessed: HashMap<State, PreprocessedState>,
}

/// Materialized results after eliminating concrete epsilon paths.
#[derive(Debug, Clone)]
struct PreprocessedState {
    transitions: Vec<(EdgeLabel, State)>,
    output: spp::SPP,
}

impl AutWithHoles {
    pub fn new() -> Self {
        Self {
            aexprs: Vec::new(),
            aexpr_map: HashMap::new(),
            delta_cache: HashMap::new(),
            preprocessed: HashMap::new(),
        }
    }

    /// Compile a user `Expr` into a hash-consed `AExprIdx`.
    pub fn expr_to_aexpr(&mut self, store: &mut spp::SPPstore, expr: &Expr) -> AExprIdx {
        match expr {
            Expr::Spp(s) => self.mk_spp(*s),
            Expr::Hole(h) => self.mk_hole(*h),
            Expr::Dup => self.mk_dup(),
            Expr::Union(a, b) => {
                let a = self.expr_to_aexpr(store, a);
                let b = self.expr_to_aexpr(store, b);
                self.mk_union(store, a, b)
            }
            Expr::Sequence(a, b) => {
                let a = self.expr_to_aexpr(store, a);
                let b = self.expr_to_aexpr(store, b);
                self.mk_sequence(store, a, b)
            }
            Expr::Star(e) => {
                let e = self.expr_to_aexpr(store, e);
                self.mk_star(store, e)
            }
        }
    }

    /// Compile a user `Expr` into the visible initial `State`.
    pub fn expr_to_state(&mut self, store: &mut spp::SPPstore, expr: &Expr) -> State {
        let aexpr = self.expr_to_aexpr(store, expr);
        State {
            aexpr,
            visible: true,
        }
    }

    // ---- ENFA-parallel API ---------------------------------------------

    /// Returns `true` iff entering this state should consume a packet from
    /// the trace.  Visible states are trace boundaries (post-dup or the
    /// initial state); invisible states are within-segment intermediates
    /// introduced to keep edges atomic.
    pub fn is_visible(&self, q: State) -> bool {
        q.visible
    }

    /// Outgoing edges of `q`.  Each edge is either a concrete SPP edge or
    /// an abstract (hole) edge.  Targets carry their own visibility.
    pub fn transitions(&mut self, store: &mut spp::SPPstore, q: State) -> Vec<(EdgeLabel, State)> {
        if let Some(state) = self.preprocessed.get(&q) {
            return state.transitions.clone();
        }
        self.delta_for_aexpr(store, q.aexpr)
    }

    /// The concrete SPP accepted from `q` without consuming another trace
    /// packet. Acceptance involving a hole is represented by abstract edges.
    pub fn output(&mut self, store: &mut spp::SPPstore, q: State) -> spp::SPP {
        if let Some(state) = self.preprocessed.get(&q) {
            return state.output;
        }
        self.output_for_aexpr(store, q.aexpr)
    }

    /// Eliminate concrete epsilon paths before substituting hole candidates.
    ///
    /// Pins `start` and all visible states. Other states are eligible when
    /// their incoming or outgoing edges are all concrete. Concrete self-loops
    /// are folded with Kleene star, and concrete predecessor/successor pairs
    /// are bypassed. Hole edges remain atomic and are never composed with SPPs.
    ///
    /// A state with abstract exits retains its incoming concrete edges; one
    /// with abstract arrivals retains its outgoing concrete edges and output.
    /// A state with neither is removed completely. Abstract edges on both
    /// sides (including abstract self-loops) prevent elimination.
    ///
    /// This preserves the language from the pinned start for every hole
    /// assignment, including candidates emitting dup. Eliminated states are
    /// not valid alternative entry points into this compiled automaton.
    pub fn preprocess_epsilon(&mut self, store: &mut spp::SPPstore, start: State) {
        let mut graph = ConcreteEpsilonGraph::collect(self, store, start);
        graph.eliminate(store);
        graph.install(self, store);
    }

    // ---- hash-consing ---------------------------------------------------

    fn intern(&mut self, aexpr: AExpr) -> AExprIdx {
        if let Some(&id) = self.aexpr_map.get(&aexpr) {
            return id;
        }
        let id = self.aexprs.len();
        self.aexprs.push(aexpr.clone());
        self.aexpr_map.insert(aexpr, id);
        id
    }

    fn get_expr(&self, aexpr: AExprIdx) -> &AExpr {
        &self.aexprs[aexpr]
    }

    // ---- smart constructors --------------------------------------------

    fn mk_spp(&mut self, s: spp::SPP) -> AExprIdx {
        self.intern(AExpr::Spp(s))
    }

    fn mk_hole(&mut self, h: Hole) -> AExprIdx {
        self.intern(AExpr::Hole(h))
    }

    fn mk_dup(&mut self) -> AExprIdx {
        self.intern(AExpr::Dup)
    }

    fn mk_union(&mut self, store: &mut spp::SPPstore, e1: AExprIdx, e2: AExprIdx) -> AExprIdx {
        self.mk_union_n(store, vec![e1, e2])
    }

    /// Build a union, flattening any nested `Union`s, combining the
    /// `Spp(_)` children into a single SPP, and dropping duplicates.
    fn mk_union_n(&mut self, store: &mut spp::SPPstore, states: Vec<AExprIdx>) -> AExprIdx {
        let mut flat: Vec<AExprIdx> = Vec::with_capacity(states.len());
        for s in states {
            match self.get_expr(s) {
                AExpr::Union(children) => {
                    let children = children.clone();
                    flat.extend(children);
                }
                _ => flat.push(s),
            }
        }

        let mut combined_spp = store.zero;
        let mut others: Vec<AExprIdx> = Vec::with_capacity(flat.len());
        for s in flat {
            match self.get_expr(s) {
                AExpr::Spp(spp) => {
                    let spp = *spp;
                    combined_spp = store.union(combined_spp, spp);
                }
                _ => others.push(s),
            }
        }
        if combined_spp != store.zero {
            let spp_state = self.mk_spp(combined_spp);
            others.push(spp_state);
        }

        others.sort();
        others.dedup();

        match others.len() {
            0 => self.mk_spp(store.zero),
            1 => others[0],
            _ => self.intern(AExpr::Union(others)),
        }
    }

    fn mk_sequence(&mut self, store: &mut spp::SPPstore, e1: AExprIdx, e2: AExprIdx) -> AExprIdx {
        self.mk_sequence_n(store, vec![e1, e2])
    }

    /// Build a sequence, flattening any nested `Sequence`s, absorbing
    /// `Spp(zero)`, dropping `Spp(one)`, and collapsing adjacent `Spp(_)`
    /// children via SPP `sequence`.
    fn mk_sequence_n(&mut self, store: &mut spp::SPPstore, states: Vec<AExprIdx>) -> AExprIdx {
        let zero = store.zero;
        let one = store.one;

        let mut flat: Vec<AExprIdx> = Vec::with_capacity(states.len());
        for s in states {
            match self.get_expr(s) {
                AExpr::Sequence(children) => {
                    let children = children.clone();
                    flat.extend(children);
                }
                _ => flat.push(s),
            }
        }

        let mut out: Vec<AExprIdx> = Vec::with_capacity(flat.len());
        for s in flat {
            match self.get_expr(s) {
                AExpr::Spp(spp) => {
                    let spp = *spp;
                    if spp == zero {
                        return self.mk_spp(zero);
                    }
                    if spp == one {
                        continue;
                    }
                    if let Some(&prev) = out.last()
                        && let AExpr::Spp(prev_spp) = *self.get_expr(prev)
                    {
                        out.pop();
                        let combined = store.sequence(prev_spp, spp);
                        if combined == zero {
                            return self.mk_spp(zero);
                        }
                        if combined != one {
                            let st = self.mk_spp(combined);
                            out.push(st);
                        }
                        continue;
                    }
                    let st = self.mk_spp(spp);
                    out.push(st);
                }
                _ => out.push(s),
            }
        }

        match out.len() {
            0 => self.mk_spp(one),
            1 => out[0],
            _ => self.intern(AExpr::Sequence(out)),
        }
    }

    /// Build a Kleene star.  `Star(_)` is idempotent, and a starred `Spp(s)`
    /// collapses to `Spp(store.star(s))`.
    fn mk_star(&mut self, store: &mut spp::SPPstore, e: AExprIdx) -> AExprIdx {
        match *self.get_expr(e) {
            AExpr::Star(_) => e,
            AExpr::Spp(s) => {
                let s_star = store.star(s);
                self.mk_spp(s_star)
            }
            _ => self.intern(AExpr::Star(e)),
        }
    }

    // ---- Antimirov derivative ------------------------------------------

    /// Compute `transitions` for an `AExpr` (visibility-agnostic), caching
    /// the result.  See module docs for the recipe.
    fn delta_for_aexpr(
        &mut self,
        store: &mut spp::SPPstore,
        aexpr: AExprIdx,
    ) -> Vec<(EdgeLabel, State)> {
        if let Some(cached) = self.delta_cache.get(&aexpr) {
            return cached.clone();
        }
        let result = self.delta_compute(store, aexpr);
        self.delta_cache.insert(aexpr, result.clone());
        result
    }

    fn delta_compute(
        &mut self,
        store: &mut spp::SPPstore,
        aexpr: AExprIdx,
    ) -> Vec<(EdgeLabel, State)> {
        let node = self.get_expr(aexpr).clone();
        match node {
            AExpr::Spp(_) => Vec::new(),
            AExpr::Hole(hole) => {
                let target = State {
                    aexpr: self.mk_spp(store.one),
                    visible: false,
                };
                vec![(EdgeLabel::Abstract(hole), target)]
            }
            AExpr::Dup => {
                // Crossing dup: identity SPP, target is `1` and visible.
                let one_aexpr = self.mk_spp(store.one);
                let target = State {
                    aexpr: one_aexpr,
                    visible: true,
                };
                vec![(EdgeLabel::Concrete(store.one), target)]
            }
            AExpr::Union(children) => {
                let mut out = Vec::new();
                for child in children {
                    out.extend(self.delta_for_aexpr(store, child));
                }
                out
            }
            AExpr::Sequence(children) => self.delta_sequence(store, &children),
            AExpr::Star(inner) => {
                // δ(e*) = δ(e ; e*).
                self.delta_sequence(store, &[inner, aexpr])
            }
        }
    }

    fn delta_sequence(
        &mut self,
        store: &mut spp::SPPstore,
        children: &[AExprIdx],
    ) -> Vec<(EdgeLabel, State)> {
        // In a sequence, either the head steps (and the tail comes along for
        // the ride), or the head outputs and we yield to the tail.
        let head = children[0];
        let tail: &[AExprIdx] = &children[1..];

        // δ(head) ; tail: each of the head's edges keeps its label and
        // visibility, but its target is re-sequenced with the tail.
        let head_edges = self.delta_for_aexpr(store, head);
        let mut edges = Vec::with_capacity(head_edges.len() + 1);
        for (label, t) in head_edges {
            let mut seq = Vec::with_capacity(1 + tail.len());
            seq.push(t.aexpr);
            seq.extend_from_slice(tail);
            let aexpr = self.mk_sequence_n(store, seq);
            edges.push((
                label,
                State {
                    aexpr,
                    visible: t.visible,
                },
            ));
        }

        // If the head accepts the empty trace with SPP `out`, yield into the
        // tail through an invisible intermediate carrying `out`.
        let out = self.output_for_aexpr(store, head);
        if out != store.zero {
            let target = State {
                aexpr: self.mk_sequence_n(store, tail.to_vec()),
                visible: false,
            };
            edges.push((EdgeLabel::Concrete(out), target));
        }

        edges
    }

    /// Compute `output` for an `AExpr`.  See module
    /// docs for the recipe.
    fn output_for_aexpr(&mut self, store: &mut spp::SPPstore, aexpr: AExprIdx) -> spp::SPP {
        let node = self.get_expr(aexpr).clone();
        match node {
            AExpr::Spp(s) => s,
            AExpr::Hole(_) => store.zero,
            AExpr::Dup => store.zero,
            AExpr::Union(children) => {
                // Can't use iterators bc it would borrow store mutably too many times
                let mut out = store.zero;
                for child in children {
                    let val = self.output_for_aexpr(store, child);
                    out = store.union(out, val);
                }
                out
            }
            AExpr::Sequence(_) => store.zero,
            AExpr::Star(_) => store.one,
        }
    }
}

/// Temporary sparse graph used only during concrete epsilon elimination.
/// Concrete edges compose as SPPs; abstract edges retain their hole labels.
struct ConcreteEpsilonGraph {
    states: Vec<State>,
    concrete: Vec<HashMap<usize, spp::SPP>>,
    abstract_edges: Vec<Vec<(Hole, usize)>>,
    outputs: Vec<spp::SPP>,
    has_abstract_incoming: Vec<bool>,
}

impl ConcreteEpsilonGraph {
    fn collect(aut: &mut AutWithHoles, store: &mut spp::SPPstore, start: State) -> Self {
        let mut states = vec![start];
        let mut index = HashMap::new();
        index.insert(start, 0);
        let mut concrete: Vec<HashMap<usize, spp::SPP>> = Vec::new();
        let mut abstract_edges: Vec<Vec<(Hole, usize)>> = Vec::new();
        let mut outputs = Vec::new();
        let mut has_abstract_incoming = vec![false];
        let mut cursor = 0;
        while cursor < states.len() {
            let state = states[cursor];
            let mut row = HashMap::new();
            let mut holes = Vec::new();
            for (label, target) in aut.transitions(store, state) {
                if matches!(label, EdgeLabel::Concrete(s) if s == store.zero) {
                    continue;
                }
                let next = *index.entry(target).or_insert_with(|| {
                    let next = states.len();
                    states.push(target);
                    has_abstract_incoming.push(false);
                    next
                });
                match label {
                    EdgeLabel::Concrete(s) => {
                        let entry = row.entry(next).or_insert(store.zero);
                        *entry = store.union(*entry, s);
                    }
                    EdgeLabel::Abstract(hole) => {
                        holes.push((hole, next));
                        has_abstract_incoming[next] = true;
                    }
                }
            }
            concrete.push(row);
            abstract_edges.push(holes);
            outputs.push(aut.output(store, state));
            cursor += 1;
        }

        Self {
            states,
            concrete,
            abstract_edges,
            outputs,
            has_abstract_incoming,
        }
    }

    fn eliminate(&mut self, store: &mut spp::SPPstore) {
        // Abstract edges are preserved throughout elimination, so incoming
        // and outgoing abstract-edge flags remain valid for the entire pass.
        // Index 0 is the pinned start, even if it is invisible.
        for k in 1..self.states.len() {
            let abstract_in = self.has_abstract_incoming[k];
            let abstract_out = !self.abstract_edges[k].is_empty();
            if self.states[k].visible || (abstract_in && abstract_out) {
                continue;
            }
            let loop_edge = self.concrete[k].remove(&k).unwrap_or(store.zero);
            let loop_star = store.star(loop_edge);
            for edge in self.concrete[k].values_mut() {
                *edge = store.sequence(loop_star, *edge);
            }
            self.outputs[k] = store.sequence(loop_star, self.outputs[k]);
            let outgoing = self.concrete[k].clone();
            let output = self.outputs[k];

            for i in 0..self.states.len() {
                if i == k {
                    continue;
                }
                let Some(incoming) = self.concrete[i].remove(&k) else {
                    continue;
                };
                let contribution = store.sequence(incoming, output);
                self.outputs[i] = store.union(self.outputs[i], contribution);
                for (&j, &edge) in &outgoing {
                    let contribution = store.sequence(incoming, edge);
                    if contribution != store.zero {
                        let entry = self.concrete[i].entry(j).or_insert(store.zero);
                        *entry = store.union(*entry, contribution);
                    }
                }
                if abstract_out {
                    // Keep the path to k's hole exits, absorbing its loop.
                    let incoming = store.sequence(incoming, loop_star);
                    self.concrete[i].insert(k, incoming);
                }
            }
            if !abstract_in {
                // All concrete exits and acceptance have moved to predecessors.
                self.concrete[k].clear();
                self.outputs[k] = store.zero;
            }
        }
    }

    fn install(self, aut: &mut AutWithHoles, store: &spp::SPPstore) {
        for (i, &state) in self.states.iter().enumerate() {
            let mut transitions: Vec<_> = self.concrete[i]
                .iter()
                .filter(|(_, s)| **s != store.zero)
                .map(|(&j, &s)| (EdgeLabel::Concrete(s), self.states[j]))
                .collect();
            transitions.extend(
                self.abstract_edges[i]
                    .iter()
                    .map(|&(hole, j)| (EdgeLabel::Abstract(hole), self.states[j])),
            );
            aut.preprocessed.insert(
                state,
                PreprocessedState {
                    transitions,
                    output: self.outputs[i],
                },
            );
        }
    }
}

impl Default for AutWithHoles {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_store() -> spp::SPPstore {
        spp::SPPstore::new(3)
    }

    // ---- hash-consing tests ------

    #[test]
    fn hash_cons_dedup() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let h = Hole(0);
        let a = aut.expr_to_aexpr(&mut store, &Expr::hole(h));
        let b = aut.expr_to_aexpr(&mut store, &Expr::hole(h));
        assert_eq!(a, b);
    }

    #[test]
    fn union_zero_identity() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let zero = store.zero;
        let h = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        let z = aut.expr_to_aexpr(&mut store, &Expr::spp(zero));
        assert_eq!(aut.mk_union(&mut store, z, h), h);
        assert_eq!(aut.mk_union(&mut store, h, z), h);
    }

    #[test]
    fn union_idempotent() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let h = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        assert_eq!(aut.mk_union(&mut store, h, h), h);
    }

    #[test]
    fn union_combines_spps() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let one = store.one;
        let top = store.top;
        let s1 = aut.expr_to_aexpr(&mut store, &Expr::spp(one));
        let s2 = aut.expr_to_aexpr(&mut store, &Expr::spp(top));
        let expected = aut.mk_spp(store.union(one, top));
        assert_eq!(aut.mk_union(&mut store, s1, s2), expected);
    }

    #[test]
    fn sequence_one_identity() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let one = store.one;
        let h = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        let o = aut.expr_to_aexpr(&mut store, &Expr::spp(one));
        assert_eq!(aut.mk_sequence(&mut store, o, h), h);
        assert_eq!(aut.mk_sequence(&mut store, h, o), h);
    }

    #[test]
    fn sequence_zero_absorbs() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let zero = store.zero;
        let h = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        let z = aut.expr_to_aexpr(&mut store, &Expr::spp(zero));
        let z_state = aut.mk_spp(zero);
        assert_eq!(aut.mk_sequence(&mut store, z, h), z_state);
        assert_eq!(aut.mk_sequence(&mut store, h, z), z_state);
    }

    #[test]
    fn sequence_collapses_adjacent_spps() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let one = store.one;
        let top = store.top;
        let s1 = aut.expr_to_aexpr(&mut store, &Expr::spp(top));
        let s2 = aut.expr_to_aexpr(&mut store, &Expr::spp(one));
        let direct = aut.mk_spp(top);
        assert_eq!(aut.mk_sequence(&mut store, s1, s2), direct);
    }

    #[test]
    fn star_idempotent() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let h = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        let s = aut.mk_star(&mut store, h);
        assert_eq!(aut.mk_star(&mut store, s), s);
    }

    #[test]
    fn star_of_spp_uses_store_star() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let top = store.top;
        let s = aut.expr_to_aexpr(&mut store, &Expr::spp(top));
        let starred = aut.mk_star(&mut store, s);
        let expected = aut.mk_spp(store.star(top));
        assert_eq!(starred, expected);
    }

    #[test]
    fn nested_sequences_flatten() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let h1 = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        let h2 = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(1)));
        let h3 = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(2)));
        let s_l = aut.expr_to_aexpr(
            &mut store,
            &Expr::sequence(
                Expr::sequence(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
                Expr::hole(Hole(2)),
            ),
        );
        let s_r = aut.expr_to_aexpr(
            &mut store,
            &Expr::sequence(
                Expr::hole(Hole(0)),
                Expr::sequence(Expr::hole(Hole(1)), Expr::hole(Hole(2))),
            ),
        );
        assert_eq!(s_l, s_r);
        match aut.get_expr(s_l) {
            AExpr::Sequence(ch) => assert_eq!(ch, &vec![h1, h2, h3]),
            other => panic!("expected Sequence, got {:?}", other),
        }
    }

    #[test]
    fn nested_unions_flatten_and_sort() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let h1 = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(0)));
        let h2 = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(1)));
        let h3 = aut.expr_to_aexpr(&mut store, &Expr::hole(Hole(2)));
        let s_l = aut.expr_to_aexpr(
            &mut store,
            &Expr::union(
                Expr::union(Expr::hole(Hole(0)), Expr::hole(Hole(1))),
                Expr::hole(Hole(2)),
            ),
        );
        let s_r = aut.expr_to_aexpr(
            &mut store,
            &Expr::union(
                Expr::hole(Hole(0)),
                Expr::union(Expr::hole(Hole(1)), Expr::hole(Hole(2))),
            ),
        );
        assert_eq!(s_l, s_r);
        match aut.get_expr(s_l) {
            AExpr::Union(ch) => {
                let mut sorted = vec![h1, h2, h3];
                sorted.sort();
                assert_eq!(ch, &sorted);
            }
            other => panic!("expected Union, got {:?}", other),
        }
    }

    // ---- concrete epsilon preprocessing tests --------------------------

    fn assert_preprocessing_equivalent(
        store: &mut spp::SPPstore,
        original: &AutWithHoles,
        optimized: &AutWithHoles,
        start: State,
        candidates: HashMap<Hole, crate::holes::aut::ExplicitDFA>,
    ) {
        use crate::holes::aut::{EpsilonClosure, ExplicitDFA, SubsetDfa, is_empty, ops};
        use crate::holes::inst::Instantiate;

        let before = Instantiate::new(original.clone(), start, candidates.clone());
        let after = Instantiate::new(optimized.clone(), start, candidates);
        let before = ExplicitDFA::from_dfa(store, SubsetDfa::new(EpsilonClosure::new(before)));
        let after = ExplicitDFA::from_dfa(store, SubsetDfa::new(EpsilonClosure::new(after)));
        assert!(is_empty(
            &ops::intersection(&before, ops::complement(&after)),
            store
        ));
        assert!(is_empty(
            &ops::intersection(&after, ops::complement(&before)),
            store
        ));
    }

    #[test]
    fn preprocess_epsilon_preserves_hole_boundaries_loops_outputs_and_visibility() {
        use crate::holes::aut::ExplicitDFA;

        let mut store = spp::SPPstore::new(2);
        let set = store.assign(0, true);
        let clear = store.assign(0, false);
        let test = store.test(1, true);
        let mut states: Vec<_> = (0..6)
            .map(|aexpr| State {
                aexpr,
                visible: aexpr == 0 || aexpr == 5,
            })
            .collect();
        // The visible and invisible variants of an expression must have
        // separate preprocessed entries.
        states[5].aexpr = states[1].aexpr;
        let c = |s, i| (EdgeLabel::Concrete(s), states[i]);
        let h = |hole, i| (EdgeLabel::Abstract(Hole(hole)), states[i]);
        let edges = [
            vec![
                c(test, 1),
                h(1, 2),
                c(store.one, 3),
                h(0, 4),
                c(store.one, 5),
            ],
            vec![c(set, 1), h(0, 2), c(test, 3)],
            vec![c(set, 2), c(clear, 3), c(store.one, 5)],
            vec![c(clear, 3), c(test, 1), c(store.one, 5)],
            vec![h(0, 4), c(set, 4), h(1, 5), c(store.one, 3)],
            vec![c(test, 1)],
        ];
        let outputs = [store.zero, test, clear, store.one, set, store.one];
        let mut original = AutWithHoles::new();
        for ((&state, transitions), output) in states.iter().zip(edges).zip(outputs) {
            original.preprocessed.insert(
                state,
                PreprocessedState {
                    transitions,
                    output,
                },
            );
        }
        let mut optimized = original.clone();
        optimized.preprocess_epsilon(&mut store, states[0]);
        assert!(optimized.transitions(&mut store, states[3]).is_empty());
        assert_eq!(optimized.output(&mut store, states[3]), store.zero);
        assert!(
            optimized
                .transitions(&mut store, states[1])
                .iter()
                .all(|(edge, _)| matches!(edge, EdgeLabel::Abstract(_)))
        );
        assert!(
            optimized
                .transitions(&mut store, states[4])
                .contains(&h(0, 4))
        );
        assert!(!optimized.transitions(&mut store, states[5]).is_empty());

        let candidates = [
            ExplicitDFA {
                start: 0,
                transitions: vec![vec![]],
                outputs: vec![store.zero],
            },
            ExplicitDFA {
                start: 0,
                transitions: vec![vec![]],
                outputs: vec![store.one],
            },
            ExplicitDFA {
                start: 0,
                transitions: vec![vec![]],
                outputs: vec![clear],
            },
            // A candidate emitting dup, including multiple visible steps.
            ExplicitDFA {
                start: 0,
                transitions: vec![vec![(set, 0)]],
                outputs: vec![test],
            },
        ];
        for first in &candidates {
            for second in &candidates {
                assert_preprocessing_equivalent(
                    &mut store,
                    &original,
                    &optimized,
                    states[0],
                    HashMap::from_iter([(Hole(0), first.clone()), (Hole(1), second.clone())]),
                );
            }
        }
        // Running preprocessing again must preserve the same language.
        let mut twice = optimized.clone();
        twice.preprocess_epsilon(&mut store, states[0]);
        assert_preprocessing_equivalent(
            &mut store,
            &optimized,
            &twice,
            states[0],
            HashMap::from_iter([
                (Hole(0), candidates[3].clone()),
                (Hole(1), candidates[2].clone()),
            ]),
        );
    }

    #[test]
    fn fuzz_preprocess_epsilon_preserves_language() {
        use crate::holes::aut::ExplicitDFA;
        crate::fuzz::seed_fuzzer(0xEC10_2026);
        for _ in 0..100 {
            let mut store = spp::SPPstore::new(2);
            let mut original = AutWithHoles::new();
            let states: Vec<_> = (0..5)
                .map(|aexpr| State {
                    aexpr,
                    visible: aexpr == 0 || crate::fuzz::random::<bool>(),
                })
                .collect();
            for &state in &states {
                let mut transitions = Vec::new();
                for &target in &states {
                    if crate::fuzz::random::<u8>().is_multiple_of(4) {
                        let label = if crate::fuzz::random::<u8>().is_multiple_of(3) {
                            EdgeLabel::Abstract(Hole(0))
                        } else {
                            EdgeLabel::Concrete(store.rand())
                        };
                        transitions.push((label, target));
                    }
                }
                let output = store.rand();
                original.preprocessed.insert(
                    state,
                    PreprocessedState {
                        transitions,
                        output,
                    },
                );
            }
            let mut optimized = original.clone();
            optimized.preprocess_epsilon(&mut store, states[0]);
            let edge = store.rand();
            let output = store.rand();
            let candidate = ExplicitDFA {
                start: 0,
                transitions: vec![vec![(edge, 0)]],
                outputs: vec![output],
            };
            assert_preprocessing_equivalent(
                &mut store,
                &original,
                &optimized,
                states[0],
                HashMap::from_iter([(Hole(0), candidate)]),
            );
        }
    }

    // ---- derivative tests ----------------------------------------------

    #[test]
    fn initial_state_is_visible() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let q = aut.expr_to_state(&mut store, &Expr::hole(Hole(0)));
        assert!(aut.is_visible(q));
    }

    #[test]
    fn spp_state_output_and_transitions() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let top = store.top;
        let q = aut.expr_to_state(&mut store, &Expr::spp(top));
        assert_eq!(aut.output(&mut store, q), top);
        assert_eq!(aut.transitions(&mut store, q), vec![]);

        // Spp(zero) outputs the zero SPP.
        let zero = store.zero;
        let q0 = aut.expr_to_state(&mut store, &Expr::spp(zero));
        assert_eq!(aut.output(&mut store, q0), store.zero);
    }

    #[test]
    fn dup_state_crosses_to_visible_one() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let q = aut.expr_to_state(&mut store, &Expr::dup());
        assert_eq!(aut.output(&mut store, q), store.zero);

        let trans = aut.transitions(&mut store, q);
        assert_eq!(trans.len(), 1);
        let (label, target) = trans[0];
        assert_eq!(label, EdgeLabel::Concrete(store.one));
        assert!(target.visible);
        // Target's aexpr is `Spp(one)`.
        let one_aexpr = aut.mk_spp(store.one);
        assert_eq!(target.aexpr, one_aexpr);
    }

    #[test]
    fn sequence_spp_hole_chains_through_invisible() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let top = store.top;
        let q = aut.expr_to_state(
            &mut store,
            &Expr::sequence(Expr::spp(top), Expr::hole(Hole(0))),
        );
        // Sequence with mixed Spp/Hole head has zero local output...
        assert_eq!(aut.output(&mut store, q), store.zero);
        // ...and a single concrete-SPP edge into an invisible Hole state.
        let trans = aut.transitions(&mut store, q);
        assert_eq!(trans.len(), 1);
        let (label, target) = trans[0];
        assert_eq!(label, EdgeLabel::Concrete(top));
        assert!(!target.visible);
        let hole_aexpr = aut.mk_hole(Hole(0));
        assert_eq!(target.aexpr, hole_aexpr);
        // The hole state contributes no local output; the hole is a transition.
        assert_eq!(aut.output(&mut store, target), store.zero);
    }

    #[test]
    fn sequence_with_dup_target_is_visible() {
        // `p ; dup ; q`: one concrete-p edge into an invisible "dup ; q",
        // then a concrete-one edge into a visible "q".
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let top = store.top;
        let q = aut.expr_to_state(
            &mut store,
            &Expr::sequence(Expr::spp(top), Expr::sequence(Expr::dup(), Expr::spp(top))),
        );

        let trans = aut.transitions(&mut store, q);
        assert_eq!(trans.len(), 1);
        let (label_1, target_1) = trans[0];
        assert_eq!(label_1, EdgeLabel::Concrete(top));
        assert!(!target_1.visible, "post-Spp target should be invisible");

        let trans_2 = aut.transitions(&mut store, target_1);
        assert_eq!(trans_2.len(), 1);
        let (label_2, target_2) = trans_2[0];
        assert_eq!(label_2, EdgeLabel::Concrete(store.one));
        assert!(target_2.visible, "post-Dup target should be visible");

        // After the dup we're at `Spp(top)` with output `top`.
        assert_eq!(aut.output(&mut store, target_2), top);
    }

    #[test]
    fn star_of_hole_self_loops_via_invisible() {
        let mut store = mk_store();
        let mut aut = AutWithHoles::new();
        let q = aut.expr_to_state(&mut store, &Expr::star(Expr::hole(Hole(0))));

        // 0 iterations: identity in the output.
        assert_eq!(aut.output(&mut store, q), store.one);

        // One iteration loops back to the same expression but invisible.
        let trans = aut.transitions(&mut store, q);
        assert_eq!(trans.len(), 1);
        let (label, target) = trans[0];
        assert_eq!(label, EdgeLabel::Abstract(Hole(0)));
        assert!(!target.visible, "self-loop target must be invisible");
        assert_eq!(target.aexpr, q.aexpr);

        // From the invisible variant, output is still the identity and the
        // edge still loops to the invisible variant.
        assert_eq!(aut.output(&mut store, target), store.one);
        let trans_inv = aut.transitions(&mut store, target);
        assert_eq!(trans_inv.len(), 1);
        let (label_inv, target_inv) = trans_inv[0];
        assert_eq!(label_inv, EdgeLabel::Abstract(Hole(0)));
        assert!(!target_inv.visible);
        assert_eq!(target_inv, target);
    }
}
