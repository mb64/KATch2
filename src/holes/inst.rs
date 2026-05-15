//! Instantiation: an [`AutWithHoles`] together with a concrete SPP
//! assignment for each [`Hole`], wrapped to implement [`ENFA`].
//!
//! The inner automaton's lazy caches mean its query methods take
//! `&mut self`, but [`ENFA`]'s methods take `&self`; we bridge with a
//! `RefCell`, the same trick `EpsilonClosure` uses for its memo table.

use crate::holes::aut::{DFA, ENFA, EpsilonClosure, get_any_trace_with_states, ops};
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

    /// Check whether this instantiated automaton is contained in `upper_bound`.
    ///
    /// Returns `Ok(())` if every triple accepted by `self` is also accepted by
    /// `upper_bound`.  Otherwise returns `Err(holes)`, where `holes` lists the
    /// `(hole, (in_packet, out_packet))` witnesses recorded along a single
    /// counterexample trace: at every visited edge or output-summand that was
    /// abstract, we record which hole was relied on and the packet pair it
    /// transmitted.
    ///
    /// Implementation: build the product NFA `self ∩ complement(upper_bound)`,
    /// extract a visible trace from it, elaborate the trace through the
    /// underlying [`AutWithHoles`] to recover invisible intermediates, then
    /// walk every transition and the final output to collect abstract labels.
    pub fn check_less_than<U: DFA>(
        &self,
        store: &mut spp::SPPstore,
        upper_bound: &U,
    ) -> Result<(), Vec<(Hole, (Vec<bool>, Vec<bool>))>> {
        let self_nfa = EpsilonClosure::new(self);
        let complement_ub = ops::complement(upper_bound);
        let product = ops::intersection(&self_nfa, &complement_ub);

        let Some((visible_path, output_pkt)) = get_any_trace_with_states(&product, store) else {
            return Ok(());
        };

        // Project onto the self-side state component.
        let self_visible_path: Vec<(State, Vec<bool>)> = visible_path
            .iter()
            .map(|((q_self, _q_ub), p)| (*q_self, p.clone()))
            .collect();

        // Splice the invisible intermediates back in.
        let full_path: Vec<(State, Vec<bool>)> =
            self_nfa.elaborate_trace(store, &self_visible_path);

        Err(self.holes_used_in_trace(store, &full_path, &output_pkt))
    }

    /// Collect the holes used by a single trace through the inner
    /// [`AutWithHoles`].
    ///
    /// `path` is a sequence of `(state, current_packet)` pairs starting at
    /// the start state and including every visited state (visible *and*
    /// invisible); `output_pkt` is the final output packet.  At every
    /// transition and at the final output summand, we pick the first
    /// label whose resolved SPP accepts the relevant packet pair as the
    /// witness — and if that label is abstract, we record the hole along
    /// with the `(in_packet, out_packet)` pair it carried.
    pub fn holes_used_in_trace(
        &self,
        store: &mut spp::SPPstore,
        path: &[(State, Vec<bool>)],
        output_pkt: &[bool],
    ) -> Vec<(Hole, (Vec<bool>, Vec<bool>))> {
        let mut holes_used: Vec<(Hole, (Vec<bool>, Vec<bool>))> = Vec::new();

        for w in path.windows(2) {
            let (q_a, p_a) = &w[0];
            let (q_b, p_b) = &w[1];
            let edges = self.aut.borrow_mut().transitions(store, *q_a);
            for (label, q_next) in edges {
                if q_next != *q_b {
                    continue;
                }
                let resolved = self.resolve(label);
                if store.accepts(resolved, p_a, p_b) {
                    if let EdgeLabel::Abstract(h) = label {
                        holes_used.push((h, (p_a.clone(), p_b.clone())));
                    }
                    break;
                }
            }
        }

        let (q_end, p_end) = path.last().expect("trace must have at least the start");
        let summands = self.aut.borrow_mut().output(store, *q_end);
        for label in summands {
            let resolved = self.resolve(label);
            if store.accepts(resolved, p_end, output_pkt) {
                if let EdgeLabel::Abstract(h) = label {
                    holes_used.push((h, (p_end.clone(), output_pkt.to_vec())));
                }
                break;
            }
        }

        holes_used
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

    fn start(&self, _store: &mut spp::SPPstore) -> Self::State {
        self.start
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
        let q = inst.start(&mut store);
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
        let q = inst.start(&mut store);
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
        let q = inst.start(&mut store);
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
        let q = inst.start(&mut store);
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
        let q = inst.start(&mut store);
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

    fn top_dfa(store: &spp::SPPstore) -> crate::holes::aut::ExplicitDFA {
        crate::holes::aut::ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.top],
        }
    }

    fn zero_dfa(store: &spp::SPPstore) -> crate::holes::aut::ExplicitDFA {
        crate::holes::aut::ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.zero],
        }
    }

    #[test]
    fn check_less_than_empty_self_is_ok() {
        let mut store = mk_store();
        let zero = store.zero;
        let inst = build_inst(&mut store, &Expr::spp(zero), HashMap::new());
        let ub = zero_dfa(&store);
        assert!(inst.check_less_than(&mut store, &ub).is_ok());
    }

    #[test]
    fn check_less_than_top_upper_bound_is_ok() {
        let mut store = mk_store();
        let top = store.top;
        let inst = build_inst(&mut store, &Expr::spp(top), HashMap::new());
        let ub = top_dfa(&store);
        assert!(inst.check_less_than(&mut store, &ub).is_ok());
    }

    #[test]
    fn check_less_than_concrete_violation_returns_err() {
        // self = top (non-empty), upper bound = zero (empty) → violation,
        // no holes to record.
        let mut store = mk_store();
        let top = store.top;
        let inst = build_inst(&mut store, &Expr::spp(top), HashMap::new());
        let ub = zero_dfa(&store);
        let result = inst.check_less_than(&mut store, &ub);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), vec![]);
    }

    #[test]
    fn check_less_than_hole_in_output_is_recorded() {
        // self = Hole(0) with Hole(0) → top. upper bound = zero.  The
        // violation should record Hole(0) as used in the final output step.
        let mut store = mk_store();
        let top = store.top;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), top);
        let inst = build_inst(&mut store, &Expr::hole(Hole(0)), holes);
        let ub = zero_dfa(&store);
        let result = inst.check_less_than(&mut store, &ub);
        let used = result.expect_err("should be a violation");
        assert_eq!(used.len(), 1);
        assert_eq!(used[0].0, Hole(0));
    }

    #[test]
    fn check_less_than_hole_to_zero_is_ok() {
        // self = Hole(0) with Hole(0) → zero (no acceptance).  Even against
        // the zero DFA, containment holds vacuously.
        let mut store = mk_store();
        let zero = store.zero;
        let mut holes = HashMap::new();
        holes.insert(Hole(0), zero);
        let inst = build_inst(&mut store, &Expr::hole(Hole(0)), holes);
        let ub = zero_dfa(&store);
        assert!(inst.check_less_than(&mut store, &ub).is_ok());
    }

    #[test]
    fn check_less_than_hole_on_edge_is_recorded() {
        // self = Sequence(Hole(0), Dup, Spp(top)) with Hole(0) → top.
        // Non-empty, upper bound zero → violation.  The trace traverses an
        // abstract edge labeled Hole(0).
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
        let ub = zero_dfa(&store);
        let used = inst
            .check_less_than(&mut store, &ub)
            .expect_err("should be a violation");
        assert!(used.iter().any(|(h, _)| *h == Hole(0)));
    }

    #[test]
    #[should_panic(expected = "missing SPP for hole")]
    fn panics_on_missing_hole() {
        let mut store = mk_store();
        let inst = build_inst(&mut store, &Expr::hole(Hole(42)), HashMap::new());
        let q = inst.start(&mut store);
        // Reaching the hole forces a resolution -> panic.
        let _ = inst.output(&mut store, &q);
    }
}
