//! NetKAT expressions extended with *holes*: formal variables that stand for
//! an unknown SPP.  This module compiles such an expression into a special
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

use crate::spp;
use std::collections::HashMap;

/// A formal hole standing for an unknown SPP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hole(pub u32);

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
}

/// One edge of the automaton: either a concrete SPP or a hole standing for
/// an SPP-to-be-filled-in.
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
/// Owns hash-consed expressions and lazy caches for the Antimirov
/// derivative.  The SPP store is supplied externally to each method that
/// needs it.
pub struct AutWithHoles {
    aexprs: Vec<AExpr>,
    aexpr_map: HashMap<AExpr, AExprIdx>,
    /// Caches keyed by `AExprIdx`: the derivative recipe doesn't depend on
    /// the source state's visibility (only on its `AExpr`), so we cache
    /// once per expression and reuse across visibilities.
    delta_cache: HashMap<AExprIdx, Vec<(EdgeLabel, State)>>,
}

impl AutWithHoles {
    pub fn new() -> Self {
        Self {
            aexprs: Vec::new(),
            aexpr_map: HashMap::new(),
            delta_cache: HashMap::new(),
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
        self.delta_for_aexpr(store, q.aexpr)
    }

    /// Atomic summands of `ε(q)`: the sum of edge labels under which `q`
    /// itself accepts the empty trace.  An empty vector means "the zero
    /// SPP" (no acceptance).  A single entry `Concrete(one)` means "the
    /// identity SPP" (always accept the empty trace).
    pub fn output(&mut self, store: &mut spp::SPPstore, q: State) -> spp::SPP {
        self.output_for_aexpr(store, q.aexpr)
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
            // self.output_sequence(store, &children),
            AExpr::Star(_) => store.one,
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
