//! A synthesis *problem*: the constraints relating hole-bearing automata to
//! reference DFAs, and the [`ProblemInstance`] that bundles a store, the holes
//! to fill, and those constraints into something solvable.
//!
//! [`ProblemInstance`] is what [`crate::holes::parser::desugar`] produces from a
//! parsed `.nksynth` file; its [`ProblemInstance::solve`] /
//! [`ProblemInstance::solve_full`] methods run the CEGIS loop
//! ([`crate::holes::cegis`]) and report a per-hole solution.

use std::collections::HashMap;

use crate::holes::aut::{ExplicitDFA, SubsetDfa, ops};
use crate::holes::cand::Cand;
use crate::holes::candidate::Candidate;
use crate::holes::cegis::{CegisError, run};
use crate::holes::nk_with_holes::{AutWithHoles, Expr, State};
use crate::spp::{self, SPP};

/// The formal hole standing for an unknown sub-program. Re-exported from
/// [`crate::expr`] for convenience alongside the problem types.
pub use crate::expr::Hole;

/// A single constraint relating a hole-bearing automaton to a concrete DFA.
///
/// Each variant pairs an [`AutWithHoles`] (with its pinned start [`State`])
/// against an [`ExplicitDFA`].  Once the holes are filled with concrete
/// candidates, the constraint asserts a containment between the resulting
/// instantiated automaton and the DFA; see the variant docs for the
/// direction.
///
/// Build one with [`Constraint::upper_bound`], [`Constraint::lower_bound`], or
/// [`Constraint::equality`], which compile a [`nk_with_holes::Expr`](Expr) into
/// the underlying automaton for you.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// `dfa ⊆ aut[holes]`: the DFA is a lower bound on the instantiated
    /// automaton (every triple the DFA accepts must also be accepted).
    LowerBound {
        aut: AutWithHoles,
        start: State,
        dfa: ExplicitDFA,
    },
    /// `aut[holes] ⊆ dfa`: the DFA is an upper bound on the instantiated
    /// automaton (every triple the automaton accepts must also be accepted by
    /// the DFA).
    UpperBound {
        aut: AutWithHoles,
        start: State,
        dfa: ExplicitDFA,
    },
    /// `aut[holes] == dfa`: the instantiated automaton must accept exactly the
    /// DFA's language — both an upper and a lower bound at once.
    Equality {
        aut: AutWithHoles,
        start: State,
        dfa: ExplicitDFA,
    },
}

impl Constraint {
    /// Upper-bound constraint `expr[holes] ⊆ dfa`, compiling `expr` into the
    /// hole-bearing automaton.
    pub fn upper_bound(store: &mut spp::SPPstore, expr: &Expr, dfa: ExplicitDFA) -> Self {
        let (aut, start) = compile(store, expr);
        Constraint::UpperBound { aut, start, dfa }
    }

    /// Lower-bound constraint `dfa ⊆ expr[holes]`, compiling `expr` into the
    /// hole-bearing automaton.
    pub fn lower_bound(store: &mut spp::SPPstore, expr: &Expr, dfa: ExplicitDFA) -> Self {
        let (aut, start) = compile(store, expr);
        Constraint::LowerBound { aut, start, dfa }
    }

    /// Equality constraint `expr[holes] == dfa`, compiling `expr` into the
    /// hole-bearing automaton.
    pub fn equality(store: &mut spp::SPPstore, expr: &Expr, dfa: ExplicitDFA) -> Self {
        let (aut, start) = compile(store, expr);
        Constraint::Equality { aut, start, dfa }
    }

    /// The hole-bearing automaton this constraint is over.
    pub fn aut(&self) -> &AutWithHoles {
        match self {
            Constraint::LowerBound { aut, .. }
            | Constraint::UpperBound { aut, .. }
            | Constraint::Equality { aut, .. } => aut,
        }
    }

    /// The pinned start state of [`Constraint::aut`].
    pub fn start(&self) -> State {
        match self {
            Constraint::LowerBound { start, .. }
            | Constraint::UpperBound { start, .. }
            | Constraint::Equality { start, .. } => *start,
        }
    }

    /// The concrete DFA this constraint compares against.
    pub fn dfa(&self) -> &ExplicitDFA {
        match self {
            Constraint::LowerBound { dfa, .. }
            | Constraint::UpperBound { dfa, .. }
            | Constraint::Equality { dfa, .. } => dfa,
        }
    }
}

/// Compile a hole-bearing [`Expr`] into its automaton and visible start state.
fn compile(store: &mut spp::SPPstore, expr: &Expr) -> (AutWithHoles, State) {
    let mut aut = AutWithHoles::new();
    let start = aut.expr_to_state(store, expr);
    (aut, start)
}

/// A synthesis problem lowered into something solvable: an SPP store sized to
/// the program's fields, the holes to synthesize, and the constraints the holes
/// must jointly satisfy.
pub struct ProblemInstance {
    pub store: spp::SPPstore,
    pub holes: Vec<Hole>,
    pub constraints: Vec<Constraint>,
}

impl ProblemInstance {
    /// Solve the problem with dup-free [`SPP`] candidates, returning one SPP per
    /// hole (or a [`CegisError`] if no solution exists / the search is
    /// inconclusive).
    ///
    /// This is the dup-free counterpart to [`ProblemInstance::solve_full`].
    pub fn solve(&mut self) -> Result<HashMap<Hole, SPP>, CegisError> {
        // SPP candidates ignore the reference DFA, but the loop still needs one.
        let reference_dfa =
            <SPP as Candidate>::make_reference_dfa(&mut self.store, &self.constraints);
        run::<SPP>(
            &self.constraints,
            &self.holes,
            &reference_dfa,
            &mut self.store,
            None,
        )
    }

    /// Solve the problem with full candidates (a union of an [`SPP`] and a
    /// dup-ful [`Cand`]), returning each hole's synthesized sub-program as an
    /// [`ExplicitDFA`].
    ///
    /// This searches a strictly larger space than [`ProblemInstance::solve`], so
    /// it can solve problems the dup-free solver reports as
    /// [`CegisError::Infeasible`].
    pub fn solve_full(&mut self) -> Result<HashMap<Hole, ExplicitDFA>, CegisError> {
        let reference_dfa = <ops::Union<SPP, Cand<'_>> as Candidate>::make_reference_dfa(
            &mut self.store,
            &self.constraints,
        );
        let candidates: HashMap<Hole, ops::Union<SPP, Cand<'_>>> = run(
            &self.constraints,
            &self.holes,
            &reference_dfa,
            &mut self.store,
            None,
        )?;
        // Materialize each candidate (an NFA) into a concrete DFA via subset
        // construction while `reference_dfa` is still alive.
        let result = candidates
            .into_iter()
            .map(|(h, cand)| {
                let dfa = ExplicitDFA::from_dfa(&mut self.store, SubsetDfa::new(cand));
                (h, dfa)
            })
            .collect();
        Ok(result)
    }
}
