//! The [`Candidate`] trait: the thing CEGIS synthesizes for each hole.
//!
//! A candidate is whatever fills a [`crate::holes::nk_with_holes::Hole`].  It
//! must be an [`NFA`] (so [`crate::holes::inst::Instantiate`] can embed it as a
//! sub-automaton) and it must know how to drive the SMT learner: allocate its
//! own slot, read itself back out of a [`Solution`], and turn CEGIS
//! counterexamples into learner [`Literal`]s.
//!
//! Two implementations:
//!
//! * [`spp::SPP`] — a single dup-free step; the historical candidate.
//! * [`Cand`] — a richer, DFA-state-dependent sub-program that pulls its states
//!   from the **upper-bound** DFA.  Its lower-bound (`accept_literal`) hook is
//!   not yet implemented.
//!
//! The upper-bound DFA is passed to the context-needing methods as
//! `&'a ExplicitDFA`; the SPP implementation ignores it.

use std::hash::Hash;

use crate::holes::aut::{ENFA, ExplicitDFA, NFA};
use crate::holes::cand::{Cand, State as CandState};
use crate::holes::smt::{AbstractBit, CandVar, Existential, Literal, SmtLearner, Solution, SppVar};
use crate::sp::SP;
use crate::spp::{SPP, SPPstore};

/// A thing CEGIS can synthesize for a hole.
///
/// `'a` is the lifetime of the upper-bound DFA that [`Cand`] candidates borrow.
pub trait Candidate<'a>: NFA + Clone {
    /// The learner slot handle for this candidate kind ([`SppVar`] / [`CandVar`]).
    type Var: Copy + Eq + Hash;

    /// Allocate a fresh learnable slot in `learner`.
    fn fresh_var(learner: &mut SmtLearner<'a>, upper_bound: &'a ExplicitDFA) -> Self::Var;

    /// Read the candidate learned for `var` out of a solved [`Solution`].
    fn from_solution(solution: &Solution<'a>, var: Self::Var) -> Self;

    /// The most-permissive candidate (accepts everything), used for the
    /// "plausibly connected" backward pass in lower-bound processing.
    fn top(store: &mut SPPstore, upper_bound: &'a ExplicitDFA) -> Self;

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

    /// Build the lower-bound literal for one hole site: "`var` must accept some
    /// `(ap1 ∈ in_sp, ap2 ∈ out_sp)`."  May allocate existentials / membership
    /// constraints on `learner` as a side effect.
    fn accept_literal(
        var: Self::Var,
        learner: &mut SmtLearner<'a>,
        store: &mut SPPstore,
        in_sp: SP,
        out_sp: SP,
    ) -> Literal;
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

    fn reject_literal(
        var: SppVar,
        pkt_in: &[bool],
        _inner: &[((), Vec<bool>)],
        pkt_out: &[bool],
    ) -> Literal {
        Literal::Spp {
            spp: var,
            ap1: concrete(pkt_in),
            ap2: concrete(pkt_out),
            polarity: false,
        }
    }

    fn accept_literal(
        var: SppVar,
        learner: &mut SmtLearner<'a>,
        store: &mut SPPstore,
        in_sp: SP,
        out_sp: SP,
    ) -> Literal {
        let n = store.num_vars() as usize;
        let in_vars: Vec<Existential> = (0..n).map(|_| learner.fresh_existential()).collect();
        let out_vars: Vec<Existential> = (0..n).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(in_sp, &in_vars, &store.sp);
        learner.add_sp_membership(out_sp, &out_vars, &store.sp);
        Literal::Spp {
            spp: var,
            ap1: in_vars.into_iter().map(AbstractBit::Exist).collect(),
            ap2: out_vars.into_iter().map(AbstractBit::Exist).collect(),
            polarity: true,
        }
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
        _var: CandVar,
        _learner: &mut SmtLearner<'a>,
        _store: &mut SPPstore,
        _in_sp: SP,
        _out_sp: SP,
    ) -> Literal {
        unimplemented!("lower-bound clause generation for Cand is not yet implemented")
    }
}
