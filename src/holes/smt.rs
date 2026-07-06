//! Learner for SPPs and [`Cand`]s.
//!
//! [`SmtLearner`] is the interface CEGIS drives; [`Z3`] is the current
//! implementation, backed by Z3's quantifier-free UF solver.  Only that
//! quantifier-free fragment is used (congruence closure over the slot
//! membership functions), so the trait could equally be implemented by
//! congruence closure atop a plain SAT solver — hence the abstraction.
//!
//! The learner accumulates abstract clauses and extracts, for each allocated
//! slot, a concrete decision diagram consistent with all of them.  Slots come
//! in two flavours:
//!
//! * **SPP slots** ([`SmtLearner::fresh_spp`]) — learned via
//!   [`crate::spp::learner::learn_spp`].
//! * **Cand slots** ([`SmtLearner::fresh_cand`]) — learned via
//!   [`Cand::from_examples`]; each is tied to the [`ExplicitDFA`] it predicates
//!   over.
//!
//! Abstract packet bits may be:
//!
//! * [`AbstractBit::Concrete`] — a fixed `false`/`true`, or
//! * [`AbstractBit::Exist`] — a globally scoped existential boolean variable
//!   (same `id` shares the same Z3 const across all clauses).
//!
//! DFA state values in a [`Literal::Cand`] are always concrete (`usize`).
//!
//! Positive SPP membership disjuncts are stated over packet *sets*
//! ([`Literal::SppMember`], a triple `(spp, in_sp, out_sp)` meaning
//! "∃ i ∈ in_sp, o ∈ out_sp. (i,o) ∈ spp").  [`SmtLearner::add_clause`] first
//! collates these triples with [`merge_by`] (the [`flags::clause_merge`]
//! optimization) — unioning output sets that share an `(spp, in_sp)`, then
//! input sets that share an `(spp, out_sp)` — and only
//! then allocates the existentials and SP-membership constraints that encode
//! each surviving disjunct.  Deferring existentialization this way is what lets
//! the set-merge fire; the merge is equivalence-preserving because the shared
//! `∃` factors out of the union.
//!
//! There are **no universally quantified bits**.  Universal quantifiers cause
//! Z3 to choose interpretations where the uninterpreted function collapses to a
//! constant via the `else` value of its function interpretation; that hands the
//! inductive bias to Z3 rather than to the BDD-style learners, which is the
//! wrong shape.  Adding universals back is a future MBQI-style outer CEGIS loop.
//!
//! Internally:
//! 1. Declare one uninterpreted function per slot meaning "this concrete input
//!    is a member".  SPP slots have signature `Bool^{2n} -> Bool`; Cand slots
//!    have `Bool^{n} (pkt_in) x Bool^{n} (pkt_start) x Int^{N} (states) x
//!    Bool^{n} (pkt_end) x Bool^{n} (pkt_out) -> Bool`, where `N` is the DFA's
//!    state count.
//! 2. Each clause becomes a disjunction of literals, each an application of the
//!    slot's function (or its negation when polarity is false); each asserted
//!    literal is also recorded so the clause structure survives to extraction.
//! 3. After Z3 solves, a greedy set-cover pass (the [`flags::example_cover`]
//!    optimization; flag off → every constrained atom is kept) walks the
//!    clauses in assertion order and selects, for each clause not already
//!    covered, one literal that is true under the model.  Atoms are keyed by
//!    (slot, concrete evaluated arguments), so one selected atom covers every
//!    clause it appears in.
//! 4. Only the selected literals become training examples for the BDD-style
//!    learners: the argument bits evaluated under the model, labelled with the
//!    literal's polarity.  This is sound — every clause keeps a true literal,
//!    and the learners agree with every example fed to them — and dropping the
//!    unselected atoms frees the learners to generalize, producing smaller
//!    diagrams than replaying Z3's full function interpretations would.

use z3::{
    FuncDecl, Model, SatResult, Solver, Sort,
    ast::{Ast, Bool, Dynamic, Int},
};

use crate::flags;
use crate::holes::aut::ExplicitDFA;
use crate::holes::cand::{Cand, Input};
use crate::sp::{SP, SPstore};
use crate::spp::learner::{Example, learn_spp};
use crate::spp::{SPP, SPPstore, Var};

use std::collections::{HashMap, HashSet};

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Opaque handle to an existential boolean variable owned by the learner.
/// Obtain one via [`SmtLearner::fresh_existential`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Existential(u32);

/// Opaque handle to a learnable SPP slot owned by the learner.
/// Obtain one via [`SmtLearner::fresh_spp`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SppVar(u32);

/// Opaque handle to a learnable [`Cand`] slot owned by the learner.
/// Obtain one via [`SmtLearner::fresh_cand`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandVar(u32);

/// One bit of an abstract packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractBit {
    /// A fixed value (`false` or `true`).
    Concrete(bool),
    /// An existential boolean variable allocated by the learner.
    Exist(Existential),
}

/// An abstract packet: one [`AbstractBit`] per packet field.
pub type AbstractPacket = Vec<AbstractBit>;

/// A literal: an abstract membership assertion about one slot, plus a polarity.
///
/// `polarity = true` means "the input is a member"; `false` means "not a
/// member".
pub enum Literal {
    /// A membership literal over an [`SppVar`]: is the pair `(ap1, ap2)` in the
    /// SPP?
    Spp {
        spp: SppVar,
        ap1: AbstractPacket,
        ap2: AbstractPacket,
        polarity: bool,
    },
    /// A positive membership disjunct stated over *sets*: "∃ i ∈ in_sp, o ∈ out_sp. (i,o) ∈ spp".
    /// Existentials are allocated lazily by `add_clause`, after [`merge_by`] collates these
    /// triples by shared input/output set.
    SppMember { spp: SppVar, in_sp: SP, out_sp: SP },
    /// A membership literal over a [`CandVar`]: does the candidate accept this
    /// (abstract) [`Input`]?  `states` is concrete.
    Cand {
        cand: CandVar,
        pkt_in: AbstractPacket,
        pkt_start: AbstractPacket,
        states: Vec<usize>,
        pkt_end: AbstractPacket,
        pkt_out: AbstractPacket,
        polarity: bool,
    },
}

/// A disjunction of literals.  Literals may target a mix of SPP and Cand slots.
pub struct AbstractClause {
    pub literals: Vec<Literal>,
}

/// The result of a successful [`SmtLearner::extract`]: one learned diagram per
/// allocated slot.
pub struct Solution<'a> {
    pub spps: HashMap<SppVar, SPP>,
    pub cands: HashMap<CandVar, Cand<'a>>,
}

/// Returned by [`SmtLearner::extract`] when the clauses are mutually
/// inconsistent (Z3 returned UNSAT).
#[derive(Debug, Clone)]
pub struct InconsistentError;

impl std::fmt::Display for InconsistentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inconsistent clauses")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Learner
// ────────────────────────────────────────────────────────────────────────────

/// The interface CEGIS needs from a learner: allocate learnable slots (SPP /
/// [`Cand`] / existential), assert SP-membership and refining clauses, and
/// extract a concrete [`Solution`].
///
/// [`Z3`] is the current implementation.  The problem is quantifier-free UF
/// (`QF_UF`): congruence closure over the slot membership functions plus the
/// SP-membership BDDs.  Nothing here uses richer Z3 theories, so this same trait
/// could be backed by congruence closure atop a SAT solver such as CaDiCaL — the
/// reason for the abstraction.
///
/// `'a` is the lifetime of the [`ExplicitDFA`]s that [`Cand`] slots predicate
/// over.
pub trait SmtLearner<'a> {
    /// Create a learner for packets with `num_vars` fields.
    fn new(num_vars: Var) -> Self
    where
        Self: Sized;

    /// Allocate a fresh learnable SPP slot, referenced in [`Literal::Spp`] and
    /// [`Literal::SppMember`].
    fn fresh_spp(&mut self) -> SppVar;

    /// Allocate a fresh learnable [`Cand`] slot predicating over `dfa`,
    /// referenced in [`Literal::Cand`].
    fn fresh_cand(&mut self, dfa: &'a ExplicitDFA) -> CandVar;

    /// Allocate a fresh existential boolean variable (usable in
    /// [`AbstractBit::Exist`] and [`Self::add_sp_membership`]).
    fn fresh_existential(&mut self) -> Existential;

    /// Constrain the packet formed by `vars` (one per field) to lie in `sp`.
    fn add_sp_membership(&mut self, sp: SP, vars: &[Existential], sp_store: &SPstore);

    /// Assert one refining [`AbstractClause`] (a disjunction of literals).
    fn add_clause(&mut self, clause: AbstractClause, sp_store: &mut SPstore);

    /// Solve once and read back one concrete diagram per allocated slot, or
    /// [`InconsistentError`] if the accumulated clauses are unsatisfiable.
    fn extract(&mut self, spp_store: &mut SPPstore) -> Result<Solution<'a>, InconsistentError>;
}

/// Which learnable slot a [`TrackedLit`] applies, by slot index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum SlotRef {
    Spp(u32),
    Cand(u32),
}

/// One asserted literal, mirrored from its Z3 assertion for extraction: the
/// slot whose membership function is applied, the exact argument ASTs it was
/// applied to (these may contain existential Bool consts), and the polarity.
struct TrackedLit {
    slot: SlotRef,
    args: Vec<Dynamic>,
    polarity: bool,
}

/// The examples chosen by [`Z3::select_examples`], grouped by slot index.
/// Slots absent from a map had no literal selected for them.
struct SelectedExamples {
    spps: HashMap<u32, Vec<Example>>,
    cands: HashMap<u32, Vec<(Input, bool)>>,
}

/// [`SmtLearner`] backed by Z3, driving its `QF_UF` solver.
pub struct Z3<'a> {
    num_vars: Var,
    solver: Solver,
    /// One uninterpreted function `Bool^{2n} -> Bool` per learnable SPP,
    /// indexed by [`SppVar::0`].
    spps: Vec<FuncDecl>,
    /// One uninterpreted function per learnable [`Cand`], paired with the DFA
    /// it predicates over, indexed by [`CandVar::0`].
    cands: Vec<(FuncDecl, &'a ExplicitDFA)>,
    existentials: Vec<Bool>,
    /// One entry per asserted clause, in assertion order: the literals of the
    /// disjunction, mirroring the Z3 assertion literal-for-literal.  Consumed
    /// by the example selection in [`SmtLearner::extract`].
    clauses: Vec<Vec<TrackedLit>>,
}

impl<'a> SmtLearner<'a> for Z3<'a> {
    fn new(num_vars: Var) -> Self {
        Self {
            num_vars,
            solver: Solver::new(),
            spps: Vec::new(),
            cands: Vec::new(),
            existentials: Vec::new(),
            clauses: Vec::new(),
        }
    }

    /// Allocate a fresh learnable SPP slot.  The returned [`SppVar`] can be
    /// referenced in [`Literal::Spp`].
    fn fresh_spp(&mut self) -> SppVar {
        let id = self.spps.len() as u32;
        let bool_sort = Sort::bool();
        let domain_refs: Vec<&Sort> = (0..2 * self.num_vars).map(|_| &bool_sort).collect();
        let f = FuncDecl::new(format!("spp_mem_{}", id), &domain_refs, &Sort::bool());
        self.spps.push(f);
        SppVar(id)
    }

    /// Allocate a fresh learnable [`Cand`] slot predicating over `dfa`.  The
    /// returned [`CandVar`] can be referenced in [`Literal::Cand`].
    fn fresh_cand(&mut self, dfa: &'a ExplicitDFA) -> CandVar {
        let id = self.cands.len() as u32;
        let n = self.num_vars;
        let ns = dfa.num_states() as u32;
        let bool_sort = Sort::bool();
        let int_sort = Sort::int();
        // pkt_in, pkt_start (bools), states (ints), pkt_end, pkt_out (bools).
        let mut domain_refs: Vec<&Sort> = Vec::with_capacity((4 * n + ns) as usize);
        domain_refs.extend((0..2 * n).map(|_| &bool_sort));
        domain_refs.extend((0..ns).map(|_| &int_sort));
        domain_refs.extend((0..2 * n).map(|_| &bool_sort));
        let f = FuncDecl::new(format!("cand_mem_{}", id), &domain_refs, &Sort::bool());
        self.cands.push((f, dfa));
        CandVar(id)
    }

    fn fresh_existential(&mut self) -> Existential {
        let id = self.existentials.len() as u32;
        self.existentials.push(Bool::new_const(format!("e_{}", id)));
        Existential(id)
    }

    /// Constrain the bit-vector formed by `vars[0..n]` (where `n` matches the
    /// SP's depth) to represent a packet that lies in `sp`.
    ///
    /// Implemented by recursively walking the SP BDD, building one Z3 Bool
    /// AST per node (`ite(vars[depth], rec(x1), rec(x0))`), and asserting the
    /// result.  Hash-consing on the SP plus a memo table over the recursion
    /// keep the work proportional to the BDD size, not exponential.
    ///
    /// Asserting membership in an empty SP makes the solver immediately UNSAT;
    /// membership in the full SP is a no-op.
    fn add_sp_membership(&mut self, sp: SP, vars: &[Existential], sp_store: &SPstore) {
        let mut memo: HashMap<SP, Bool> = HashMap::new();
        let body = self.build_sp_bool(sp, vars, 0, sp_store, &mut memo);
        self.solver.assert(&body);
    }

    /// Add an abstract clause.  The corresponding Z3 assertion is added
    /// immediately.
    ///
    /// When [`flags::clause_merge`] is on, [`Literal::SppMember`] disjuncts
    /// are first collated by [`merge_by`] — merging output sets that share an
    /// `(spp, in_sp)`, then input sets that share an `(spp, out_sp)` — before
    /// each surviving triple is turned into existentials plus SP-membership
    /// constraints.  The merge is equivalence-preserving (the shared `∃`
    /// factors out of the union), so it shrinks the disjunction without
    /// changing its meaning.
    fn add_clause(&mut self, clause: AbstractClause, sp_store: &mut SPstore) {
        let n = self.num_vars as usize;

        // Split set-stated membership disjuncts from the rest so they can be
        // collated before existentialization.
        let mut members: Vec<(SppVar, SP, SP)> = Vec::new();
        let mut others: Vec<Literal> = Vec::new();
        for lit in clause.literals {
            match lit {
                Literal::SppMember { spp, in_sp, out_sp } => members.push((spp, in_sp, out_sp)),
                other => others.push(other),
            }
        }

        // Collate the membership disjuncts (the [`flags::clause_merge`]
        // optimization): rule 1 unions output sets sharing a (hole, input
        // set); rule 2 unions input sets sharing a (hole, output set).
        // Disabled → each triple is existentialized on its own.
        let members = if flags::clause_merge() {
            let members = merge_by(
                members,
                |&(h, s, _)| (h.0, s.0),
                |a, b| (a.0, a.1, sp_store.union(a.2, b.2)),
            );
            merge_by(
                members,
                |&(h, _, t)| (h.0, t.0),
                |a, b| (a.0, sp_store.union(a.1, b.1), a.2),
            )
        } else {
            members
        };

        let mut lit_asts: Vec<Bool> = Vec::with_capacity(members.len() + others.len());
        let mut tracked: Vec<TrackedLit> = Vec::with_capacity(members.len() + others.len());

        // Each merged membership triple becomes a fresh existential pair pinned
        // to its (in_sp, out_sp) by SP-membership, applied positively.
        for (spp, in_sp, out_sp) in members {
            let in_vars: Vec<Existential> = (0..n).map(|_| self.fresh_existential()).collect();
            let out_vars: Vec<Existential> = (0..n).map(|_| self.fresh_existential()).collect();
            self.add_sp_membership(in_sp, &in_vars, sp_store);
            self.add_sp_membership(out_sp, &out_vars, sp_store);
            let ap1: AbstractPacket = in_vars.into_iter().map(AbstractBit::Exist).collect();
            let ap2: AbstractPacket = out_vars.into_iter().map(AbstractBit::Exist).collect();
            let slot = SlotRef::Spp(spp.0);
            let args = self.spp_args(&ap1, &ap2);
            lit_asts.push(self.apply_slot(slot, &args, true));
            tracked.push(TrackedLit {
                slot,
                args,
                polarity: true,
            });
        }

        for lit in &others {
            let (slot, args, polarity) = match lit {
                Literal::Spp {
                    spp,
                    ap1,
                    ap2,
                    polarity,
                } => (SlotRef::Spp(spp.0), self.spp_args(ap1, ap2), *polarity),
                Literal::SppMember { .. } => unreachable!("SppMember split out above"),
                Literal::Cand {
                    cand,
                    pkt_in,
                    pkt_start,
                    states,
                    pkt_end,
                    pkt_out,
                    polarity,
                } => (
                    SlotRef::Cand(cand.0),
                    self.cand_args(*cand, pkt_in, pkt_start, states, pkt_end, pkt_out),
                    *polarity,
                ),
            };
            lit_asts.push(self.apply_slot(slot, &args, polarity));
            tracked.push(TrackedLit {
                slot,
                args,
                polarity,
            });
        }

        let body = match lit_asts.len() {
            0 => Bool::from_bool(false),
            1 => lit_asts.into_iter().next().unwrap(),
            _ => Bool::or(&lit_asts),
        };
        self.solver.assert(&body);
        self.clauses.push(tracked);
    }

    /// Solve once and extract a concrete diagram for each allocated slot.
    ///
    /// After Z3 finds a model, [`Z3::select_examples`] picks the training
    /// examples for [`crate::spp::learner::learn_spp`] (SPP slots) and
    /// [`Cand::from_examples`] (Cand slots).  Under [`flags::example_cover`]
    /// (the default) it greedily picks one literal true under the model per
    /// not-yet-covered clause; feeding fewer examples keeps every clause
    /// satisfied while leaving more room for the BDD-style inductive bias of
    /// those learners.
    fn extract(&mut self, spp_store: &mut SPPstore) -> Result<Solution<'a>, InconsistentError> {
        match self.solver.check() {
            SatResult::Unsat => return Err(InconsistentError),
            SatResult::Unknown => panic!("Z3 returned unknown"),
            SatResult::Sat => {}
        }
        let model = self.solver.get_model().expect("model after Sat");
        let mut examples = self.select_examples(&model, flags::example_cover());

        // SPP slots.  A slot with no selected literals gets no examples and
        // learns the empty SPP.
        let mut spps = HashMap::with_capacity(self.spps.len());
        for idx in 0..self.spps.len() as u32 {
            let exs = examples.spps.remove(&idx).unwrap_or_default();
            let spp = learn_spp(exs, spp_store).expect("Z3 model is internally consistent");
            spps.insert(SppVar(idx), spp);
        }

        // Cand slots.
        let mut cands = HashMap::with_capacity(self.cands.len());
        for (idx, (_, dfa)) in self.cands.iter().enumerate() {
            let exs = examples.cands.remove(&(idx as u32)).unwrap_or_default();
            let cand = Cand::from_examples(spp_store, dfa, &exs)
                .expect("Z3 model is internally consistent");
            cands.insert(CandVar(idx as u32), cand);
        }

        Ok(Solution { spps, cands })
    }
}

impl<'a> Z3<'a> {
    fn exist(&self, e: Existential) -> Bool {
        self.existentials[e.0 as usize].clone()
    }

    /// Z3 argument for one abstract bit.
    fn bit_arg(&self, b: &AbstractBit) -> Dynamic {
        match b {
            AbstractBit::Concrete(v) => Dynamic::from_ast(&Bool::from_bool(*v)),
            AbstractBit::Exist(e) => Dynamic::from_ast(&self.exist(*e)),
        }
    }

    /// Recursively walk the SP BDD into a Z3 Bool
    /// (`ite(vars[depth], rec(x1), rec(x0))`), memoized over the BDD so the work
    /// stays proportional to its size.
    fn build_sp_bool(
        &self,
        sp: SP,
        vars: &[Existential],
        depth: usize,
        sp_store: &SPstore,
        memo: &mut HashMap<SP, Bool>,
    ) -> Bool {
        if sp == SP::new(0) {
            return Bool::from_bool(false);
        }
        if sp == SP::new(1) {
            return Bool::from_bool(true);
        }
        if let Some(b) = memo.get(&sp) {
            return b.clone();
        }
        let node = sp_store.get(sp);
        let x0_b = self.build_sp_bool(node.x0, vars, depth + 1, sp_store, memo);
        let x1_b = self.build_sp_bool(node.x1, vars, depth + 1, sp_store, memo);
        let var_b = self.exist(vars[depth]);
        let result = var_b.ite(&x1_b, &x0_b);
        memo.insert(sp, result.clone());
        result
    }

    /// Z3 arguments for an SPP-membership application: the bits of `ap1`
    /// followed by the bits of `ap2`.
    fn spp_args(&self, ap1: &[AbstractBit], ap2: &[AbstractBit]) -> Vec<Dynamic> {
        let n = self.num_vars as usize;
        assert_eq!(ap1.len(), n, "ap1 length mismatch");
        assert_eq!(ap2.len(), n, "ap2 length mismatch");
        let mut args: Vec<Dynamic> = Vec::with_capacity(2 * n);
        for side in [ap1, ap2] {
            for b in side {
                args.push(self.bit_arg(b));
            }
        }
        args
    }

    /// Z3 arguments for a Cand-membership application:
    /// `pkt_in ++ pkt_start ++ states ++ pkt_end ++ pkt_out`.
    fn cand_args(
        &self,
        cand: CandVar,
        pkt_in: &[AbstractBit],
        pkt_start: &[AbstractBit],
        states: &[usize],
        pkt_end: &[AbstractBit],
        pkt_out: &[AbstractBit],
    ) -> Vec<Dynamic> {
        let n = self.num_vars as usize;
        let (_, dfa) = &self.cands[cand.0 as usize];
        let ns = dfa.num_states();
        assert_eq!(pkt_in.len(), n, "pkt_in length mismatch");
        assert_eq!(pkt_start.len(), n, "pkt_start length mismatch");
        assert_eq!(pkt_end.len(), n, "pkt_end length mismatch");
        assert_eq!(pkt_out.len(), n, "pkt_out length mismatch");
        assert_eq!(states.len(), ns, "states length mismatch");
        let mut args: Vec<Dynamic> = Vec::with_capacity(4 * n + ns);
        for b in pkt_in {
            args.push(self.bit_arg(b));
        }
        for b in pkt_start {
            args.push(self.bit_arg(b));
        }
        for &s in states {
            // The DFA is complete (any sink is a real state), so every
            // state value lies in `0..ns`.
            assert!(s < ns, "state value out of range");
            args.push(Dynamic::from_ast(&Int::from_u64(s as u64)));
        }
        for b in pkt_end {
            args.push(self.bit_arg(b));
        }
        for b in pkt_out {
            args.push(self.bit_arg(b));
        }
        args
    }

    /// The uninterpreted membership function declared for `slot`.
    fn slot_fn(&self, slot: SlotRef) -> &FuncDecl {
        match slot {
            SlotRef::Spp(i) => &self.spps[i as usize],
            SlotRef::Cand(i) => &self.cands[i as usize].0,
        }
    }

    /// Build the Z3 Bool for a membership literal: apply `slot`'s
    /// uninterpreted function to `args`, negating when `!polarity`.
    fn apply_slot(&self, slot: SlotRef, args: &[Dynamic], polarity: bool) -> Bool {
        let arg_refs: Vec<&dyn Ast> = args.iter().map(|a| a as &dyn Ast).collect();
        let a = self
            .slot_fn(slot)
            .apply(&arg_refs)
            .as_bool()
            .expect("slot fn has Bool range");
        if polarity { a } else { a.not() }
    }

    /// Choose the training examples for the passive learners from `model`.
    ///
    /// With `minimize` (the [`flags::example_cover`] optimization) this is a
    /// greedy set cover of the tracked clauses by literals true under the
    /// model.  Clauses are walked in assertion order.  A clause containing a
    /// true literal whose *atom* — its (slot, concrete evaluated arguments)
    /// pair — was already selected is covered for free; otherwise the first
    /// true literal is selected.  Each selected atom yields exactly one
    /// training example: its evaluated argument bits labelled with the
    /// selecting literal's polarity (the literal is true under the model, so
    /// the slot function's value at those arguments *is* the polarity).
    /// Linear in the total number of literals; no attempt at an optimal
    /// cover.  Panics if some clause has no true literal — impossible after
    /// `Sat`.
    ///
    /// Without `minimize`, every atom appearing in any clause becomes an
    /// example (deduplicated), labelled with the slot function's value under
    /// the model — the pre-cover behavior, kept as the flag-off baseline.
    fn select_examples(&self, model: &Model, minimize: bool) -> SelectedExamples {
        let mut selected: HashSet<(SlotRef, Vec<u64>)> = HashSet::new();
        let mut out = SelectedExamples {
            spps: HashMap::new(),
            cands: HashMap::new(),
        };

        for clause in &self.clauses {
            // First true literal of the clause, kept in case no already-selected
            // atom covers it.
            let mut fallback: Option<(Vec<u64>, Vec<Dynamic>, &TrackedLit)> = None;
            let mut covered = false;
            for lit in clause {
                // Evaluate the arguments (existentials included) to concrete
                // values, with model completion for anything left unpinned.
                let concrete: Vec<Dynamic> = lit
                    .args
                    .iter()
                    .map(|a| {
                        model
                            .eval(a, true)
                            .expect("model completion yields a value")
                    })
                    .collect();
                // The literal is true iff the slot function's value at those
                // arguments matches the polarity.
                let value = model
                    .eval(&self.apply_slot(lit.slot, &concrete, true), true)
                    .and_then(|b| b.as_bool())
                    .expect("model completion yields a concrete Bool");
                let key = encode_args(&concrete);
                if !minimize {
                    // Keep every atom, labelled with its value under the model.
                    if selected.insert((lit.slot, key)) {
                        self.push_example(&mut out, lit.slot, &concrete, value);
                    }
                    continue;
                }
                if value != lit.polarity {
                    continue;
                }
                if selected.contains(&(lit.slot, key.clone())) {
                    covered = true;
                    break;
                }
                if fallback.is_none() {
                    fallback = Some((key, concrete, lit));
                }
            }
            if !minimize || covered {
                continue;
            }
            let Some((key, concrete, lit)) = fallback else {
                unreachable!("Z3 said Sat but a clause has no true literal")
            };
            selected.insert((lit.slot, key));
            self.push_example(&mut out, lit.slot, &concrete, lit.polarity);
        }

        out
    }

    /// Append one training example to `out`: `slot`'s membership at the
    /// concrete (model-evaluated) arguments `concrete`, labelled `label`.
    fn push_example(
        &self,
        out: &mut SelectedExamples,
        slot: SlotRef,
        concrete: &[Dynamic],
        label: bool,
    ) {
        let n = self.num_vars as usize;
        match slot {
            SlotRef::Spp(i) => {
                let bits: Vec<bool> = concrete.iter().map(read_bit).collect();
                assert_eq!(bits.len(), 2 * n);
                out.spps.entry(i).or_default().push(Example {
                    input: bits[..n].to_vec(),
                    output: bits[n..].to_vec(),
                    in_spp: label,
                });
            }
            SlotRef::Cand(i) => {
                let ns = self.cands[i as usize].1.num_states();
                assert_eq!(concrete.len(), 4 * n + ns);
                let pkt_in = concrete[..n].iter().map(read_bit).collect();
                let pkt_start = concrete[n..2 * n].iter().map(read_bit).collect();
                let states: Vec<usize> =
                    concrete[2 * n..2 * n + ns].iter().map(read_state).collect();
                let pkt_end = concrete[2 * n + ns..3 * n + ns]
                    .iter()
                    .map(read_bit)
                    .collect();
                let pkt_out = concrete[3 * n + ns..4 * n + ns]
                    .iter()
                    .map(read_bit)
                    .collect();
                out.cands.entry(i).or_default().push((
                    Input {
                        pkt_in,
                        pkt_start,
                        states,
                        pkt_end,
                        pkt_out,
                    },
                    label,
                ));
            }
        }
    }
}

/// Collate `items` by `key`, folding every run of equal keys into one element
/// with `merge`.  Because equal keys become adjacent after the stable-ish sort,
/// a single linear pass (peeking the last kept element) suffices.
///
/// `merge(a, b)` is called left-to-right within each key group; it must keep
/// the key invariant (its result must share the key of both inputs).
fn merge_by<T, K: Ord>(
    mut items: Vec<T>,
    mut key: impl FnMut(&T) -> K,
    mut merge: impl FnMut(T, T) -> T,
) -> Vec<T> {
    items.sort_by_key(&mut key);
    let mut out: Vec<T> = Vec::with_capacity(items.len());
    for item in items {
        match out.pop() {
            Some(prev) if key(&prev) == key(&item) => out.push(merge(prev, item)),
            Some(prev) => {
                out.push(prev);
                out.push(item);
            }
            None => out.push(item),
        }
    }
    out
}

/// Read a concrete `bool` out of a model-evaluated argument.
fn read_bit(d: &Dynamic) -> bool {
    d.as_bool()
        .and_then(|b| b.as_bool())
        .expect("evaluated arg is a concrete Bool")
}

/// Read a concrete state index out of a model-evaluated argument.
fn read_state(d: &Dynamic) -> usize {
    d.as_int()
        .and_then(|i| i.as_u64())
        .expect("evaluated arg is a concrete Int") as usize
}

/// Canonical hashable encoding of concrete (model-evaluated) arguments: Bools
/// become 0/1, Ints their value.  Argument sorts are fixed per slot, so the
/// encoding is injective within a slot — good enough for the atom dedup key.
fn encode_args(args: &[Dynamic]) -> Vec<u64> {
    args.iter()
        .map(|a| match a.as_bool().and_then(|b| b.as_bool()) {
            Some(b) => b as u64,
            None => read_state(a) as u64,
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: Var = 3;

    fn spp_accepts(store: &SPPstore, spp: SPP, p1: &[bool], p2: &[bool]) -> bool {
        let mut cur = spp;
        for (&b1, &b2) in p1.iter().zip(p2.iter()) {
            if cur == SPP::new(0) {
                return false;
            }
            if cur == SPP::new(1) {
                return true;
            }
            let node = store.get(cur);
            cur = match (b1, b2) {
                (false, false) => node.x00,
                (false, true) => node.x01,
                (true, false) => node.x10,
                (true, true) => node.x11,
            };
        }
        cur == SPP::new(1)
    }

    fn c(b: bool) -> AbstractBit {
        AbstractBit::Concrete(b)
    }

    fn ap_concrete(bits: &[bool]) -> AbstractPacket {
        bits.iter().map(|&b| c(b)).collect()
    }

    fn spp_lit(spp: SppVar, ap1: AbstractPacket, ap2: AbstractPacket, polarity: bool) -> Literal {
        Literal::Spp {
            spp,
            ap1,
            ap2,
            polarity,
        }
    }

    #[test]
    fn test_empty() {
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        let result = learner.extract(&mut store).unwrap();
        assert_eq!(result.spps[&s], store.zero);
    }

    #[test]
    fn test_empty_no_spps() {
        // No SPP slots allocated: extract returns an empty map.
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let result = learner.extract(&mut store).unwrap();
        assert!(result.spps.is_empty());
    }

    #[test]
    fn test_single_positive_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(
                    s,
                    ap_concrete(&[false, false, false]),
                    ap_concrete(&[true, true, true]),
                    true,
                )],
            },
            &mut store.sp,
        );
        let spp = learner.extract(&mut store).unwrap().spps[&s];
        assert!(spp_accepts(
            &store,
            spp,
            &[false, false, false],
            &[true, true, true],
        ));
    }

    #[test]
    fn test_single_negative_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(
                    s,
                    ap_concrete(&[true, true, true]),
                    ap_concrete(&[false, false, false]),
                    false,
                )],
            },
            &mut store.sp,
        );
        let spp = learner.extract(&mut store).unwrap().spps[&s];
        assert!(!spp_accepts(
            &store,
            spp,
            &[true, true, true],
            &[false, false, false],
        ));
    }

    #[test]
    fn test_conflicting_unit_clauses() {
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        let ap1 = ap_concrete(&[false, false, false]);
        let ap2 = ap_concrete(&[true, false, false]);
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(s, ap1.clone(), ap2.clone(), true)],
            },
            &mut store.sp,
        );
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(s, ap1, ap2, false)],
            },
            &mut store.sp,
        );
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn test_disjunctive_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        // (0,0,0)->(0,0,0) is in SPP  OR  (1,1,1)->(1,1,1) is not in SPP
        learner.add_clause(
            AbstractClause {
                literals: vec![
                    spp_lit(
                        s,
                        ap_concrete(&[false, false, false]),
                        ap_concrete(&[false, false, false]),
                        true,
                    ),
                    spp_lit(
                        s,
                        ap_concrete(&[true, true, true]),
                        ap_concrete(&[true, true, true]),
                        false,
                    ),
                ],
            },
            &mut store.sp,
        );
        let spp = learner.extract(&mut store).unwrap().spps[&s];
        let a = spp_accepts(&store, spp, &[false, false, false], &[false, false, false]);
        let b = spp_accepts(&store, spp, &[true, true, true], &[true, true, true]);
        assert!(a || !b, "clause disjunction not satisfied");
    }

    #[test]
    fn test_two_spps_independent() {
        // Two SPP slots constrained independently: s1 must accept (0,0,0)→(1,1,1),
        // s2 must NOT accept (0,0,0)→(0,0,0).  Both constraints satisfiable
        // in isolation; extract returns one SPP per slot.
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s1 = learner.fresh_spp();
        let s2 = learner.fresh_spp();
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(
                    s1,
                    ap_concrete(&[false, false, false]),
                    ap_concrete(&[true, true, true]),
                    true,
                )],
            },
            &mut store.sp,
        );
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(
                    s2,
                    ap_concrete(&[false, false, false]),
                    ap_concrete(&[false, false, false]),
                    false,
                )],
            },
            &mut store.sp,
        );
        let result = learner.extract(&mut store).unwrap();
        assert!(spp_accepts(
            &store,
            result.spps[&s1],
            &[false, false, false],
            &[true, true, true],
        ));
        assert!(!spp_accepts(
            &store,
            result.spps[&s2],
            &[false, false, false],
            &[false, false, false],
        ));
    }

    /// Singleton SP for a concrete packet (test helper, mirrors aut::singleton_sp).
    fn singleton_sp_for_test(store: &mut SPPstore, pkt: &[bool]) -> SP {
        let mut sp_val = SP::new(1);
        let mut zero = SP::new(0);
        for &b in pkt.iter().rev() {
            let next = if b {
                store.sp.mk(zero, sp_val)
            } else {
                store.sp.mk(sp_val, zero)
            };
            zero = store.sp.mk(zero, zero);
            sp_val = next;
        }
        sp_val
    }

    #[test]
    fn test_sp_membership_full_set_is_noop() {
        // sp = full set; adding membership shouldn't change satisfiability.
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let _s = learner.fresh_spp();
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(store.sp.one, &e, &store.sp);
        assert!(learner.extract(&mut store).is_ok());
    }

    #[test]
    fn test_sp_membership_empty_set_is_unsat() {
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let _s = learner.fresh_spp();
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(store.sp.zero, &e, &store.sp);
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn test_sp_membership_singleton_pins_existentials() {
        // Constrain existentials to be a specific packet, then assert a
        // literal that uses them; the resulting SPP must accept that pair.
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        let target = vec![true, false, true];
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        let pkt_sp = singleton_sp_for_test(&mut store, &target);
        learner.add_sp_membership(pkt_sp, &e, &store.sp);

        // Literal: s accepts (existentials, [false, true, false]).  With
        // existentials pinned to `target`, this forces (target, [false,true,false]) ∈ s.
        let out = vec![false, true, false];
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(
                    s,
                    e.iter().map(|&ev| AbstractBit::Exist(ev)).collect(),
                    out.iter().map(|&b| AbstractBit::Concrete(b)).collect(),
                    true,
                )],
            },
            &mut store.sp,
        );

        let result = learner.extract(&mut store).unwrap();
        assert!(spp_accepts(&store, result.spps[&s], &target, &out));
    }

    #[test]
    fn test_clause_spanning_two_spps() {
        // s1 accepts (0,0,0)→(0,0,0)  OR  s2 does NOT accept (1,1,1)→(1,1,1).
        // A disjunction across distinct SPPs.  The learner must satisfy *one*.
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s1 = learner.fresh_spp();
        let s2 = learner.fresh_spp();
        learner.add_clause(
            AbstractClause {
                literals: vec![
                    spp_lit(
                        s1,
                        ap_concrete(&[false, false, false]),
                        ap_concrete(&[false, false, false]),
                        true,
                    ),
                    spp_lit(
                        s2,
                        ap_concrete(&[true, true, true]),
                        ap_concrete(&[true, true, true]),
                        false,
                    ),
                ],
            },
            &mut store.sp,
        );
        let result = learner.extract(&mut store).unwrap();
        let a = spp_accepts(
            &store,
            result.spps[&s1],
            &[false, false, false],
            &[false, false, false],
        );
        let b = spp_accepts(
            &store,
            result.spps[&s2],
            &[true, true, true],
            &[true, true, true],
        );
        assert!(a || !b, "no disjunct satisfied across two SPPs");
    }

    #[test]
    fn test_cand_single_positive() {
        // A single positive Cand literal must be reproduced by the learned Cand.
        let mut store = SPPstore::new(N);
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.zero],
        };
        let mut learner = Z3::new(N);
        let cv = learner.fresh_cand(&dfa);
        learner.add_clause(
            AbstractClause {
                literals: vec![Literal::Cand {
                    cand: cv,
                    pkt_in: ap_concrete(&[false, false, false]),
                    pkt_start: ap_concrete(&[true, false, true]),
                    states: vec![0],
                    pkt_end: ap_concrete(&[true, true, true]),
                    pkt_out: ap_concrete(&[false, false, false]),
                    polarity: true,
                }],
            },
            &mut store.sp,
        );
        let solution = learner.extract(&mut store).unwrap();
        let cand = &solution.cands[&cv];
        assert!(cand.accepts_input(
            &mut store,
            &Input {
                pkt_in: vec![false, false, false],
                pkt_start: vec![true, false, true],
                states: vec![0],
                pkt_end: vec![true, true, true],
                pkt_out: vec![false, false, false],
            },
        ));
    }

    #[test]
    fn test_merge_by_collates_adjacent_keys() {
        // Key on `.0`, sum the `.1` of each group: (0,1),(0,2),(1,3) → (0,3),(1,3).
        let merged = merge_by(
            vec![(0, 1), (0, 2), (1, 3)],
            |&(k, _)| k,
            |a, b| (a.0, a.1 + b.1),
        );
        assert_eq!(merged, vec![(0, 3), (1, 3)]);
    }

    #[test]
    fn test_merge_by_unsorted_input() {
        // Groups need not be pre-sorted; `merge_by` sorts first.
        let merged = merge_by(
            vec![(1, 10), (0, 1), (1, 20), (0, 2)],
            |&(k, _)| k,
            |a, b| (a.0, a.1 + b.1),
        );
        assert_eq!(merged, vec![(0, 3), (1, 30)]);
    }

    #[test]
    fn test_spp_member_shared_input_merges() {
        // Two positive membership disjuncts sharing an input set but with
        // different singleton output sets.  After merge they collapse into one
        // disjunct over the unioned output set, so the learned SPP must accept
        // at least one of the two in→out pairs.
        let mut store = SPPstore::new(N);
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();

        let in_pkt = vec![true, false, true];
        let out_a = vec![false, false, false];
        let out_b = vec![true, true, true];
        let in_sp = singleton_sp_for_test(&mut store, &in_pkt);
        let out_sp_a = singleton_sp_for_test(&mut store, &out_a);
        let out_sp_b = singleton_sp_for_test(&mut store, &out_b);

        learner.add_clause(
            AbstractClause {
                literals: vec![
                    Literal::SppMember {
                        spp: s,
                        in_sp,
                        out_sp: out_sp_a,
                    },
                    Literal::SppMember {
                        spp: s,
                        in_sp,
                        out_sp: out_sp_b,
                    },
                ],
            },
            &mut store.sp,
        );

        let spp = learner.extract(&mut store).unwrap().spps[&s];
        let a = spp_accepts(&store, spp, &in_pkt, &out_a);
        let b = spp_accepts(&store, spp, &in_pkt, &out_b);
        assert!(a || b, "merged membership disjunct not realized");
    }

    /// Two clauses sharing the atom `(p,q) ∈ s`: a unit clause and a
    /// disjunction with a second, unforced atom.  The greedy cover selects the
    /// shared atom once, covers the second clause with it, and feeds exactly
    /// one example.
    fn shared_atom_learner(store: &mut SPPstore, positive: bool) -> (Z3<'static>, SppVar) {
        let p = [false, false, false];
        let q = [true, true, true];
        let r = [true, false, true];
        let t = [false, true, false];
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(s, ap_concrete(&p), ap_concrete(&q), positive)],
            },
            &mut store.sp,
        );
        learner.add_clause(
            AbstractClause {
                literals: vec![
                    spp_lit(s, ap_concrete(&p), ap_concrete(&q), positive),
                    spp_lit(s, ap_concrete(&r), ap_concrete(&t), true),
                ],
            },
            &mut store.sp,
        );
        (learner, s)
    }

    #[test]
    fn test_greedy_cover_selects_shared_atom_once() {
        let mut store = SPPstore::new(N);
        let (learner, s) = shared_atom_learner(&mut store, true);
        assert_eq!(learner.solver.check(), SatResult::Sat);
        let model = learner.solver.get_model().unwrap();
        let selected = learner.select_examples(&model, true);
        let examples = &selected.spps[&s.0];
        assert_eq!(examples.len(), 1, "shared atom must be selected only once");
        assert_eq!(examples[0].input, vec![false, false, false]);
        assert_eq!(examples[0].output, vec![true, true, true]);
        assert!(examples[0].in_spp);
        assert!(selected.cands.is_empty());
    }

    #[test]
    fn test_greedy_cover_selects_shared_negative_atom_once() {
        // Same shape, with the shared atom appearing negatively: the unit
        // clause forces (p,q) ∉ s, so the second clause is covered by the
        // negative atom and the single example is labelled `false`.
        let mut store = SPPstore::new(N);
        let (learner, s) = shared_atom_learner(&mut store, false);
        assert_eq!(learner.solver.check(), SatResult::Sat);
        let model = learner.solver.get_model().unwrap();
        let selected = learner.select_examples(&model, true);
        let examples = &selected.spps[&s.0];
        assert_eq!(examples.len(), 1, "shared atom must be selected only once");
        assert_eq!(examples[0].input, vec![false, false, false]);
        assert_eq!(examples[0].output, vec![true, true, true]);
        assert!(!examples[0].in_spp);
    }

    #[test]
    fn test_select_all_examples_keeps_unshared_atom() {
        // Same clauses, but with minimization off: both atoms — the shared
        // (p,q) and the unforced (r,t) — become examples, deduplicated, each
        // labelled with its value under the model.
        let mut store = SPPstore::new(N);
        let (learner, s) = shared_atom_learner(&mut store, true);
        assert_eq!(learner.solver.check(), SatResult::Sat);
        let model = learner.solver.get_model().unwrap();
        let selected = learner.select_examples(&model, false);
        let examples = &selected.spps[&s.0];
        assert_eq!(examples.len(), 2, "both atoms kept, each exactly once");
        assert_eq!(examples[0].input, vec![false, false, false]);
        assert_eq!(examples[0].output, vec![true, true, true]);
        assert!(examples[0].in_spp, "the forced atom keeps its model value");
    }

    #[test]
    fn test_extract_with_shared_atom() {
        // End-to-end: extraction over the shared-atom clauses learns an SPP
        // from the single selected example, which must accept (p,q).
        let mut store = SPPstore::new(N);
        let (mut learner, s) = shared_atom_learner(&mut store, true);
        let spp = learner.extract(&mut store).unwrap().spps[&s];
        assert!(spp_accepts(
            &store,
            spp,
            &[false, false, false],
            &[true, true, true],
        ));
    }

    #[test]
    fn test_mixed_spp_cand_clause() {
        // One clause spanning an SPP slot and a Cand slot: the learner must
        // satisfy at least one disjunct.
        let mut store = SPPstore::new(N);
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![]],
            outputs: vec![store.zero],
        };
        let mut learner = Z3::new(N);
        let s = learner.fresh_spp();
        let cv = learner.fresh_cand(&dfa);
        let input = Input {
            pkt_in: vec![false, false, false],
            pkt_start: vec![true, false, true],
            states: vec![0],
            pkt_end: vec![true, true, true],
            pkt_out: vec![false, false, false],
        };
        learner.add_clause(
            AbstractClause {
                literals: vec![
                    spp_lit(
                        s,
                        ap_concrete(&[false, false, false]),
                        ap_concrete(&[true, true, true]),
                        true,
                    ),
                    Literal::Cand {
                        cand: cv,
                        pkt_in: ap_concrete(&input.pkt_in),
                        pkt_start: ap_concrete(&input.pkt_start),
                        states: input.states.clone(),
                        pkt_end: ap_concrete(&input.pkt_end),
                        pkt_out: ap_concrete(&input.pkt_out),
                        polarity: true,
                    },
                ],
            },
            &mut store.sp,
        );
        let solution = learner.extract(&mut store).unwrap();
        let a = spp_accepts(
            &store,
            solution.spps[&s],
            &[false, false, false],
            &[true, true, true],
        );
        let b = solution.cands[&cv].accepts_input(&mut store, &input);
        assert!(a || b, "no disjunct satisfied across SPP and Cand");
    }
}
