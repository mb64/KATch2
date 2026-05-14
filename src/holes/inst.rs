//! Instantiation: an [`AutWithHoles`] together with a concrete SPP
//! assignment for each [`Hole`], wrapped to implement [`ENFA`].
//!
//! The inner automaton's lazy caches mean its query methods take
//! `&mut self`, but [`ENFA`]'s methods take `&self`; we bridge with a
//! `RefCell`, the same trick `EpsilonClosure` uses for its memo table.

use crate::holes::aut::ENFA;
use crate::holes::nk_with_holes::{AutWithHoles, EdgeLabel, Hole, State};
use crate::spp;
use std::cell::RefCell;
use std::collections::HashMap;

/// An [`AutWithHoles`] plus a concrete SPP assignment for each hole that
/// can appear on an edge or in an output.
pub struct Instantiate {
    aut: RefCell<AutWithHoles>,
    start: State,
    holes: HashMap<Hole, spp::SPP>,
}

impl Instantiate {
    /// Build an instantiation from an automaton, its initial state, and a
    /// hole-to-SPP map.  Every hole the inner automaton emits must be
    /// present in `holes` (otherwise the [`ENFA`] methods will panic when
    /// they encounter it).
    pub fn new(aut: AutWithHoles, start: State, holes: HashMap<Hole, spp::SPP>) -> Self {
        Self {
            aut: RefCell::new(aut),
            start,
            holes,
        }
    }

    fn resolve(&self, label: EdgeLabel) -> spp::SPP {
        match label {
            EdgeLabel::Concrete(s) => s,
            EdgeLabel::Abstract(h) => *self
                .holes
                .get(&h)
                .unwrap_or_else(|| panic!("Instantiate: missing SPP for hole {h:?}")),
        }
    }
}

impl ENFA for Instantiate {
    type State = State;

    fn start(&self, store: &mut spp::SPPstore) -> Vec<(spp::SPP, Self::State)> {
        vec![(store.one, self.start)]
    }

    fn is_visible(&self, _store: &mut spp::SPPstore, q: &Self::State) -> bool {
        self.aut.borrow().is_visible(*q)
    }

    fn transitions(
        &self,
        store: &mut spp::SPPstore,
        q: &Self::State,
    ) -> Vec<(spp::SPP, Self::State)> {
        let edges = self.aut.borrow_mut().transitions(store, *q);
        edges
            .into_iter()
            .map(|(label, target)| (self.resolve(label), target))
            .collect()
    }

    fn output(&self, store: &mut spp::SPPstore, q: &Self::State) -> spp::SPP {
        // ε is the union of the resolved atomic summands.
        let summands = self.aut.borrow_mut().output(store, *q);
        let mut result = store.zero;
        for label in summands {
            let s = self.resolve(label);
            result = store.union(result, s);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holes::nk_with_holes::Expr;

    fn mk_store() -> spp::SPPstore {
        spp::SPPstore::new(3)
    }

    fn build_inst(
        store: &mut spp::SPPstore,
        expr: &Expr,
        holes: HashMap<Hole, spp::SPP>,
    ) -> Instantiate {
        let mut aut = AutWithHoles::new();
        let start = aut.expr_to_state(store, expr);
        Instantiate::new(aut, start, holes)
    }

    #[test]
    fn resolves_concrete_output() {
        let mut store = mk_store();
        let top = store.top;
        let inst = build_inst(&mut store, &Expr::spp(top), HashMap::new());
        let starts = inst.start(&mut store);
        assert_eq!(starts.len(), 1);
        let (spp_start, q) = starts[0];
        assert_eq!(spp_start, store.one);
        assert!(inst.is_visible(&mut store, &q));
        assert_eq!(inst.transitions(&mut store, &q), vec![]);
        assert_eq!(inst.output(&mut store, &q), top);
    }

    #[test]
    fn resolves_abstract_output_from_map() {
        let mut store = mk_store();
        let top = store.top;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), top);
        let inst = build_inst(&mut store, &Expr::hole(Hole(0)), holes);
        let q = inst.start(&mut store)[0].1;
        // Hole(0) → top, and there are no other summands, so output == top.
        assert_eq!(inst.output(&mut store, &q), top);
        assert_eq!(inst.transitions(&mut store, &q), vec![]);
    }

    #[test]
    fn unions_atomic_summands() {
        // Union(Spp(top), Hole(0)) with Hole(0) → one.  output should be
        // union(top, one) == top.
        let mut store = mk_store();
        let top = store.top;
        let one = store.one;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), one);
        let inst = build_inst(
            &mut store,
            &Expr::union(Expr::spp(top), Expr::hole(Hole(0))),
            holes,
        );
        let q = inst.start(&mut store)[0].1;
        let want = store.union(top, one);
        assert_eq!(inst.output(&mut store, &q), want);
    }

    #[test]
    fn resolves_abstract_edges() {
        // Sequence(Hole(0), Dup, Spp(top)) with Hole(0) → top.
        // From start (visible), transitions = [(Abstract(0), invisible_state)].
        // After resolving Hole(0) → top, the concrete SPP is `top`.
        let mut store = mk_store();
        let top = store.top;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), top);
        let inst = build_inst(
            &mut store,
            &Expr::sequence(
                Expr::hole(Hole(0)),
                Expr::sequence(Expr::dup(), Expr::spp(top)),
            ),
            holes,
        );
        let q = inst.start(&mut store)[0].1;
        let trans = inst.transitions(&mut store, &q);
        assert_eq!(trans.len(), 1);
        let (spp, target) = trans[0];
        assert_eq!(spp, top);
        // Target is invisible (within-segment).
        assert!(!inst.is_visible(&mut store, &target));
    }

    #[test]
    fn is_empty_for_zero() {
        // Spp(zero) instantiated has no acceptance.
        let mut store = mk_store();
        let zero = store.zero;
        let inst = build_inst(&mut store, &Expr::spp(zero), HashMap::new());
        assert!(crate::holes::aut::is_empty(&inst, &mut store));
    }

    #[test]
    fn is_empty_for_hole_mapped_to_zero() {
        // Hole(0) → zero ⇒ acceptance is zero ⇒ empty.
        let mut store = mk_store();
        let zero = store.zero;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), zero);
        let inst = build_inst(&mut store, &Expr::hole(Hole(0)), holes);
        assert!(crate::holes::aut::is_empty(&inst, &mut store));
    }

    #[test]
    fn non_empty_for_hole_mapped_to_top() {
        // Hole(0) → top ⇒ non-empty.
        let mut store = mk_store();
        let top = store.top;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), top);
        let inst = build_inst(&mut store, &Expr::hole(Hole(0)), holes);
        assert!(!crate::holes::aut::is_empty(&inst, &mut store));
    }

    #[test]
    fn dup_crosses_into_visible_state() {
        // `dup` with empty holes: transitions out of the start consume a
        // trace element and land in a visible "1" state with output one.
        let mut store = mk_store();
        let inst = build_inst(&mut store, &Expr::dup(), HashMap::new());
        let q = inst.start(&mut store)[0].1;
        assert_eq!(inst.output(&mut store, &q), store.zero);
        let trans = inst.transitions(&mut store, &q);
        assert_eq!(trans.len(), 1);
        let (spp, target) = trans[0];
        assert_eq!(spp, store.one);
        assert!(inst.is_visible(&mut store, &target));
        assert_eq!(inst.output(&mut store, &target), store.one);
        // The full automaton is non-empty: traces like (input, [input], input).
        assert!(!crate::holes::aut::is_empty(&inst, &mut store));
    }

    #[test]
    #[should_panic(expected = "missing SPP for hole")]
    fn panics_on_missing_hole() {
        let mut store = mk_store();
        let inst = build_inst(&mut store, &Expr::hole(Hole(42)), HashMap::new());
        let q = inst.start(&mut store)[0].1;
        // Reaching the hole forces a resolution -> panic.
        let _ = inst.output(&mut store, &q);
    }
}
