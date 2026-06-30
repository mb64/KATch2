//! The [`Candidate`] trait: the thing CEGIS synthesizes for each hole.
//!
//! A candidate is whatever fills a [`crate::holes::nk_with_holes::Hole`].  It
//! must be an [`NFA`] (so [`crate::holes::inst::Instantiate`] can embed it as a
//! sub-automaton) and it must know how to drive the SMT learner: allocate its
//! own slot, read itself back out of a [`Solution`], and turn CEGIS
//! counterexamples into learner [`Literal`]s.
//!
//! Three implementations:
//!
//! * [`SPP`] — a single dup-free step
//! * [`Cand`] — a richer, DFA-state-dependent sub-program that pulls its states
//!   from the **upper-bound** DFA.
//! * [`ops::Union`] -- union together an `SPP` and a `Cand` to actually solve full NetKAT synthesis
//!
//! The upper-bound DFA is passed to the context-needing methods as
//! `&'a ExplicitDFA`; the SPP implementation ignores it.

use std::hash::Hash;

use crate::holes::aut::{ENFA, ExplicitDFA, NFA, ops};
use crate::holes::cand::{Cand, State as CandState};
use crate::holes::problem::Constraint;
use crate::holes::smt::{AbstractBit, CandVar, Existential, Literal, SmtLearner, Solution, SppVar};
use crate::sp::SP;
use crate::spp::{SPP, SPPstore};

/// A thing CEGIS can synthesize for a hole.
///
/// `'a` is the lifetime of the upper-bound DFA that [`Cand`] candidates borrow.
pub trait Candidate<'a>: ENFA<State: Ord> + NFA + Clone {
    /// The learner slot handle for this candidate kind ([`SppVar`] / [`CandVar`]).
    type Var: Copy + Eq + Hash;

    /// Allocate a fresh learnable slot in `learner`.
    fn fresh_var(learner: &mut SmtLearner<'a>, upper_bound: &'a ExplicitDFA) -> Self::Var;

    /// Read the candidate learned for `var` out of a solved [`Solution`].
    fn from_solution(solution: &Solution<'a>, var: Self::Var) -> Self;

    /// The most-permissive candidate (accepts everything), used for the
    /// "plausibly connected" backward pass in lower-bound processing.
    fn top(store: &mut SPPstore, upper_bound: &'a ExplicitDFA) -> Self;

    /// Build the freestanding *reference DFA* that grounds candidates of this
    /// kind for `constraints`: the DFA whose states [`Cand`] candidates draw on
    /// and that the all-top fallback uses.  Caller-owned, independent of any
    /// single constraint's DFA.
    ///
    /// **Invariant:** the returned DFA is *complete* — its transition function
    /// is total (every state has an outgoing edge for every `(in, out)` packet
    /// pair, e.g. via a sink state) — which is what stepping [`Cand`] requires.
    fn make_reference_dfa(store: &mut SPPstore, constraints: &[Constraint]) -> ExplicitDFA;

    /// Build the upper-bound literal: "`var` must **not** accept this witness."
    ///
    /// `pkt_in` is the packet entering the hole, `pkt_out` the packet leaving
    /// it, and `inner` the candidate-internal trace recorded in between (states
    /// plus packets).
    fn reject_literal(
        var: Self::Var,
        pkt_in: &[bool],
        inner: &[(<Self as ENFA>::State, Vec<bool>)],
        pkt_out: &[bool],
    ) -> Literal;

    /// Build the lower-bound literal for one hole site and span: "`var` must
    /// accept a traversal that enters with `ap1 ∈ in_sp`, consumes `trace`
    /// internally, and leaves with `ap2 ∈ out_sp`."  May allocate existentials /
    /// membership constraints on `learner` as a side effect.
    ///
    /// `trace` is the subtrace the hole consumes between its in- and out-side
    /// (empty if it consumes nothing).  Returns `None` if this candidate kind
    /// cannot realize a traversal that consumes exactly `trace` (e.g. an
    /// [`SPP`] consumes nothing, so any non-empty `trace` is `None`).
    fn accept_literal(
        var: Self::Var,
        learner: &mut SmtLearner<'a>,
        store: &mut SPPstore,
        upper_bound: &'a ExplicitDFA,
        in_sp: SP,
        out_sp: SP,
        trace: &[Vec<bool>],
    ) -> Option<Literal>;
}

/// Concrete-packet abstract bits.
fn concrete(pkt: &[bool]) -> Vec<AbstractBit> {
    pkt.iter().map(|&b| AbstractBit::Concrete(b)).collect()
}

impl<'a> Candidate<'a> for SPP {
    type Var = SppVar;

    fn fresh_var(learner: &mut SmtLearner<'a>, _upper_bound: &'a ExplicitDFA) -> SppVar {
        learner.fresh_spp()
    }

    fn from_solution(solution: &Solution<'a>, var: SppVar) -> SPP {
        solution.spps[&var]
    }

    fn top(store: &mut SPPstore, _upper_bound: &'a ExplicitDFA) -> SPP {
        store.top
    }

    fn make_reference_dfa(store: &mut SPPstore, _constraints: &[Constraint]) -> ExplicitDFA {
        // SPP candidates ignore the reference DFA; a single complete (total,
        // self-looping) state is enough to satisfy the invariant.
        ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.zero],
        }
    }

    fn reject_literal(
        var: SppVar,
        pkt_in: &[bool],
        inner: &[((), Vec<bool>)],
        pkt_out: &[bool],
    ) -> Literal {
        assert!(inner.is_empty());
        Literal::Spp {
            spp: var,
            ap1: concrete(pkt_in),
            ap2: concrete(pkt_out),
            polarity: false,
        }
    }

    fn accept_literal(
        var: SppVar,
        _learner: &mut SmtLearner<'a>,
        _store: &mut SPPstore,
        _upper_bound: &'a ExplicitDFA,
        in_sp: SP,
        out_sp: SP,
        trace: &[Vec<bool>],
    ) -> Option<Literal> {
        // An SPP is a single dup-free step: it consumes no trace internally.
        // Existentialization of `(in_sp, out_sp)` is deferred to the learner
        // (see [`crate::holes::smt::SmtLearner::add_clause`]) so the SP sets
        // survive to be merged across disjuncts.
        if !trace.is_empty() {
            return None;
        }
        Some(Literal::SppMember {
            spp: var,
            in_sp,
            out_sp,
        })
    }
}

impl<'a> Candidate<'a> for Cand<'a> {
    type Var = CandVar;

    fn fresh_var(learner: &mut SmtLearner<'a>, upper_bound: &'a ExplicitDFA) -> CandVar {
        learner.fresh_cand(upper_bound)
    }

    fn from_solution(solution: &Solution<'a>, var: CandVar) -> Cand<'a> {
        solution.cands[&var].clone()
    }

    fn top(store: &mut SPPstore, upper_bound: &'a ExplicitDFA) -> Cand<'a> {
        // No examples ⇒ the empty-input default, which accepts everything.
        Cand::from_examples(store, upper_bound, &[]).expect("empty examples never conflict")
    }

    fn make_reference_dfa(store: &mut SPPstore, constraints: &[Constraint]) -> ExplicitDFA {
        // Ground on the product of every upper-bound DFA, each completed with a
        // sink so the product's transition function is total (the invariant the
        // `Cand` ENFA relies on when stepping its [`ops::Exponential`]).
        let completed: Vec<ops::WithSinkState<&ExplicitDFA>> = constraints
            .iter()
            .filter_map(|c| match c {
                Constraint::UpperBound { dfa, .. } | Constraint::Equality { dfa, .. } => {
                    Some(ops::WithSinkState(dfa))
                }
                Constraint::LowerBound { .. } => None,
            })
            .collect();
        ExplicitDFA::product(store, &completed)
    }

    fn reject_literal(
        var: CandVar,
        pkt_in: &[bool],
        inner: &[(CandState, Vec<bool>)],
        pkt_out: &[bool],
    ) -> Literal {
        // A Cand hole always takes at least one `Start -> Middle` step before it
        // can output, so `inner` is non-empty: its first packet is `pkt_start`,
        // its last `(Middle, packet)` carries the final state vector and
        // `pkt_end`.
        let (_, pkt_start) = inner.first().expect("cand hole traversal is non-empty");
        let (last_state, pkt_end) = inner.last().expect("cand hole traversal is non-empty");
        let CandState::Middle(_, states) = last_state else {
            unreachable!("a cand hole exits from a Middle state");
        };
        Literal::Cand {
            cand: var,
            pkt_in: concrete(pkt_in),
            pkt_start: concrete(pkt_start),
            states: states.clone(),
            pkt_end: concrete(pkt_end),
            pkt_out: concrete(pkt_out),
            polarity: false,
        }
    }

    fn accept_literal(
        var: CandVar,
        learner: &mut SmtLearner<'a>,
        store: &mut SPPstore,
        upper_bound: &'a ExplicitDFA,
        in_sp: SP,
        out_sp: SP,
        trace: &[Vec<bool>],
    ) -> Option<Literal> {
        // A Cand hole consumes at least one packet (the `Start -> Middle` step).
        let (pkt_start, rest) = trace.split_first()?;
        let pkt_end = rest.last().unwrap_or(pkt_start);

        // The state vector starts at the identity (one token per DFA state) and
        // evolves by the upper-bound DFA's exponential over the consumed
        // subtrace.  This steps `upper_bound` directly, so it relies on the DFA
        // being *complete* (total transition function, e.g. via a sink state) —
        // the same requirement `Cand`'s ENFA has: a component on a dead state
        // must flow into a sink rather than stall the product (which would
        // otherwise wrongly make the subtrace untraversable).
        let mut states: Vec<usize> = (0..upper_bound.num_states()).collect();
        let exp = ops::Exponential { inner: upper_bound };
        for pair in trace.windows(2) {
            let next = exp
                .transitions(store, &states)
                .into_iter()
                .find(|(spp, _)| store.accepts(*spp, &pair[0], &pair[1]))
                .map(|(_, target)| target)?;
            states = next;
        }

        // Existentials pin the carry-in / carry-out packets to (in_sp, out_sp).
        let nv = store.num_vars() as usize;
        let in_vars: Vec<Existential> = (0..nv).map(|_| learner.fresh_existential()).collect();
        let out_vars: Vec<Existential> = (0..nv).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(in_sp, &in_vars, &store.sp);
        learner.add_sp_membership(out_sp, &out_vars, &store.sp);

        Some(Literal::Cand {
            cand: var,
            pkt_in: in_vars.into_iter().map(AbstractBit::Exist).collect(),
            pkt_start: concrete(pkt_start),
            states,
            pkt_end: concrete(pkt_end),
            pkt_out: out_vars.into_iter().map(AbstractBit::Exist).collect(),
            polarity: true,
        })
    }
}

impl<'a> Candidate<'a> for ops::Union<SPP, Cand<'a>> {
    type Var = (
        <SPP as Candidate<'a>>::Var,
        <Cand<'a> as Candidate<'a>>::Var,
    );

    fn fresh_var(learner: &mut SmtLearner<'a>, upper_bound: &'a ExplicitDFA) -> Self::Var {
        (
            SPP::fresh_var(learner, upper_bound),
            Cand::fresh_var(learner, upper_bound),
        )
    }

    fn from_solution(solution: &Solution<'a>, var: Self::Var) -> Self {
        ops::union(
            SPP::from_solution(solution, var.0),
            Cand::from_solution(solution, var.1),
        )
    }

    fn top(store: &mut SPPstore, upper_bound: &'a ExplicitDFA) -> Self {
        ops::union(SPP::top(store, upper_bound), Cand::top(store, upper_bound))
    }

    fn make_reference_dfa(store: &mut SPPstore, constraints: &[Constraint]) -> ExplicitDFA {
        // The SPP side ignores the reference DFA, so the union's reference DFA
        // is just the `Cand` side's.
        Cand::make_reference_dfa(store, constraints)
    }

    fn reject_literal(
        var: Self::Var,
        pkt_in: &[bool],
        inner: &[(<Self as ENFA>::State, Vec<bool>)],
        pkt_out: &[bool],
    ) -> Literal {
        if inner.is_empty() {
            SPP::reject_literal(var.0, pkt_in, &[], pkt_out)
        } else {
            // A non-empty inner trace must have gone through the `Cand` (Right)
            // side: the `SPP` side has no transitions, so it can only output
            // directly (an empty inner trace).  Union transitions keep `Right`
            // states `Right`, so every recorded state is `Right(cand_state)`.
            let inner: Vec<(CandState, Vec<bool>)> = inner
                .iter()
                .map(|(q, pk)| match q {
                    ops::UnionState::Right(cs) => (cs.clone(), pk.clone()),
                    ops::UnionState::Start | ops::UnionState::Left(_) => {
                        unreachable!("a non-empty union hole traversal goes through the Cand side")
                    }
                })
                .collect();
            Cand::reject_literal(var.1, pkt_in, &inner, pkt_out)
        }
    }

    fn accept_literal(
        var: Self::Var,
        learner: &mut SmtLearner<'a>,
        store: &mut SPPstore,
        upper_bound: &'a ExplicitDFA,
        in_sp: SP,
        out_sp: SP,
        trace: &[Vec<bool>],
    ) -> Option<Literal> {
        if trace.is_empty() {
            SPP::accept_literal(var.0, learner, store, upper_bound, in_sp, out_sp, trace)
        } else {
            Cand::accept_literal(var.1, learner, store, upper_bound, in_sp, out_sp, trace)
        }
    }
}
