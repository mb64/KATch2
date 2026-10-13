//! Learner for SPPs and [`Cand`]s, using native bitblasting and lazy
//! Ackermannization over an incremental RustSAT/CaDiCaL solver.
//!
//! Packet bitvectors are represented directly by SAT literals. SP membership
//! BDDs become shared Tseitin ITE gates. Each distinct symbolic predicate
//! application gets a SAT variable. After solving, applications are grouped by
//! their slot and evaluated arguments; conflicting results trigger congruence
//! lemmas `(args_a = args_b) => (result_a = result_b)` and another solve.
//! No quadratic eager expansion is needed, and the lemmas survive CEGIS rounds.
//!
//! A consistent SAT assignment is copied into a native array once per solve.
//! The existing greedy example cover then selects training examples for the
//! BDD learners, preserving the clause merge and example cover optimizations.
//! There are no universal quantifiers, and DFA state arguments are concrete.

mod sat;

use rustsat::types::Lit;
use sat::{FALSE, Model, Sat, TRUE};

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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Debug)]
pub struct AbstractClause {
    pub literals: Vec<Literal>,
}

impl AbstractClause {
    /// Compare only a pair of clauses, modulo disjunct order, repetition, and
    /// the same membership-set merging used when asserting them.
    pub(crate) fn same_literals(&self, other: &Self, sp_store: &mut SPstore) -> bool {
        let key = |clause: Self, sp_store: &mut SPstore| {
            let (members, others) = clause.collate(sp_store);
            members
                .into_iter()
                .map(|(spp, in_sp, out_sp)| Literal::SppMember { spp, in_sp, out_sp })
                .chain(others)
                .collect::<HashSet<_>>()
        };
        key(self.clone(), sp_store) == key(other.clone(), sp_store)
    }

    fn collate(self, sp_store: &mut SPstore) -> (Vec<(SppVar, SP, SP)>, Vec<Literal>) {
        // Split set-stated membership disjuncts from the rest so they can be
        // collated before existentialization.
        let mut members: Vec<(SppVar, SP, SP)> = Vec::new();
        let mut others: Vec<Literal> = Vec::new();
        for lit in self.literals {
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

        (members, others)
    }
}

/// The result of a successful [`SmtLearner::extract`]: one learned diagram per
/// allocated slot.
pub struct Solution<'a> {
    pub spps: HashMap<SppVar, SPP>,
    pub cands: HashMap<CandVar, Cand<'a>>,
}

/// Returned by [`SmtLearner::extract`] when the clauses are mutually
/// inconsistent (the native solver returned UNSAT).
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
/// [`Native`] implements this with Boolean bitblasting and lazy predicate
/// congruence constraints. Packet fields are individual bits; DFA states are
/// concrete values, so no arithmetic theory is required.
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

/// Predicate identity. SPP and Cand slots occupy separate namespaces.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum SlotRef {
    Spp(u32),
    Cand(u32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Arg {
    Bit(Lit),
    State(usize),
}

impl Arg {
    fn value(self, model: &Model) -> u64 {
        match self {
            Self::Bit(lit) => model.value(lit) as u64,
            Self::State(state) => state as u64,
        }
    }
}

struct Atom {
    slot: SlotRef,
    args: Vec<Arg>,
    result: Lit,
}

struct TrackedLit {
    atom: usize,
    polarity: bool,
}

struct SelectedExamples {
    spps: HashMap<u32, Vec<Example>>,
    cands: HashMap<u32, Vec<(Input, bool)>>,
}

/// Native bitvector and uninterpreted predicate learner using RustSAT/CaDiCaL.
/// Clauses, gates, and learned congruence lemmas persist across extractions.
pub struct Native<'a> {
    num_vars: Var,
    solver: Sat,
    num_spps: u32,
    cands: Vec<&'a ExplicitDFA>,
    existentials: Vec<Lit>,
    atoms: Vec<Atom>,
    applications: HashMap<(SlotRef, Vec<Arg>), usize>,
    clauses: Vec<Vec<TrackedLit>>,
}

/// Compatibility name for callers of the former Z3 backend.
/// This alias uses the native solver and does not link to Z3.
pub type Z3<'a> = Native<'a>;

impl<'a> SmtLearner<'a> for Native<'a> {
    fn new(num_vars: Var) -> Self {
        Self {
            num_vars,
            solver: Sat::new(),
            atoms: Vec::new(),
            applications: HashMap::new(),
            num_spps: 0,
            cands: Vec::new(),
            existentials: Vec::new(),
            clauses: Vec::new(),
        }
    }

    /// Allocate a fresh learnable SPP slot.  The returned [`SppVar`] can be
    /// referenced in [`Literal::Spp`].
    fn fresh_spp(&mut self) -> SppVar {
        let id = self.num_spps;
        self.num_spps += 1;
        SppVar(id)
    }

    /// Allocate a fresh learnable [`Cand`] slot predicating over `dfa`.  The
    /// returned [`CandVar`] can be referenced in [`Literal::Cand`].
    fn fresh_cand(&mut self, dfa: &'a ExplicitDFA) -> CandVar {
        let id = self.cands.len() as u32;
        self.cands.push(dfa);
        CandVar(id)
    }

    fn fresh_existential(&mut self) -> Existential {
        let id = self.existentials.len() as u32;
        let bit = self.solver.fresh();
        self.existentials.push(bit);
        Existential(id)
    }

    /// Constrain the bit-vector formed by `vars[0..n]` (where `n` matches the
    /// SP's depth) to represent a packet that lies in `sp`.
    ///
    /// Implemented by recursively walking the SP BDD, building one SAT literal
    /// per node (`ite(vars[depth], rec(x1), rec(x0))`), and asserting the
    /// result.  Hash-consing on the SP plus a memo table over the recursion
    /// keep the work proportional to the BDD size, not exponential.
    ///
    /// Asserting membership in an empty SP makes the solver immediately UNSAT;
    /// membership in the full SP is a no-op.
    fn add_sp_membership(&mut self, sp: SP, vars: &[Existential], sp_store: &SPstore) {
        let mut memo: HashMap<(SP, usize), Lit> = HashMap::new();
        let body = self.build_sp_bool(sp, vars, 0, sp_store, &mut memo);
        self.solver.clause([body]);
    }

    /// Add an abstract clause. The corresponding CNF clause is added
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

        let (members, others) = clause.collate(sp_store);

        let mut sat_lits: Vec<Lit> = Vec::with_capacity(members.len() + others.len());
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
            let atom = self.apply_slot(slot, args);
            sat_lits.push(self.atoms[atom].result);
            tracked.push(TrackedLit {
                atom,
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
            let atom = self.apply_slot(slot, args);
            let result = self.atoms[atom].result;
            sat_lits.push(if polarity { result } else { !result });
            tracked.push(TrackedLit { atom, polarity });
        }

        self.solver.clause(sat_lits);
        self.clauses.push(tracked);
    }

    /// Solve once and extract a concrete diagram for each allocated slot.
    ///
    /// After a congruence-consistent model is found, `select_examples` picks
    /// the training examples for [`crate::spp::learner::learn_spp`] (SPP slots) and
    /// [`Cand::from_examples`] (Cand slots).  Under [`flags::example_cover`]
    /// (the default) it greedily picks one literal true under the model per
    /// not-yet-covered clause; feeding fewer examples keeps every clause
    /// satisfied while leaving more room for the BDD-style inductive bias of
    /// those learners.
    fn extract(&mut self, spp_store: &mut SPPstore) -> Result<Solution<'a>, InconsistentError> {
        let model = self.solve()?;
        let mut examples = self.select_examples(&model, flags::example_cover());

        // SPP slots.  A slot with no selected literals gets no examples and
        // learns the empty SPP.
        let mut spps = HashMap::with_capacity(self.num_spps as usize);
        for idx in 0..self.num_spps {
            let exs = examples.spps.remove(&idx).unwrap_or_default();
            let spp = learn_spp(exs, spp_store).expect("native model is internally consistent");
            spps.insert(SppVar(idx), spp);
        }

        // Cand slots.
        let mut cands = HashMap::with_capacity(self.cands.len());
        for (idx, dfa) in self.cands.iter().enumerate() {
            let exs = examples.cands.remove(&(idx as u32)).unwrap_or_default();
            let cand = Cand::from_examples(spp_store, dfa, &exs)
                .expect("native model is internally consistent");
            cands.insert(CandVar(idx as u32), cand);
        }

        Ok(Solution { spps, cands })
    }
}

impl<'a> Native<'a> {
    fn bit_arg(&self, b: &AbstractBit) -> Arg {
        Arg::Bit(match b {
            AbstractBit::Concrete(true) => TRUE,
            AbstractBit::Concrete(false) => FALSE,
            AbstractBit::Exist(e) => self.existentials[e.0 as usize],
        })
    }

    /// Shared ITE encoding of an SP BDD, with depth in the memo key because
    /// the variable tested at a node depends on its position in the packet.
    fn build_sp_bool(
        &mut self,
        sp: SP,
        vars: &[Existential],
        depth: usize,
        sp_store: &SPstore,
        memo: &mut HashMap<(SP, usize), Lit>,
    ) -> Lit {
        if sp == SP::new(0) {
            return FALSE;
        }
        if sp == SP::new(1) {
            return TRUE;
        }
        if let Some(&bit) = memo.get(&(sp, depth)) {
            return bit;
        }
        let node = sp_store.get(sp);
        let x0 = self.build_sp_bool(node.x0, vars, depth + 1, sp_store, memo);
        let x1 = self.build_sp_bool(node.x1, vars, depth + 1, sp_store, memo);
        let cond = self.existentials[vars[depth].0 as usize];
        let result = self.solver.ite(cond, x1, x0);
        memo.insert((sp, depth), result);
        result
    }

    fn spp_args(&self, ap1: &[AbstractBit], ap2: &[AbstractBit]) -> Vec<Arg> {
        let n = self.num_vars as usize;
        assert_eq!(ap1.len(), n, "ap1 length mismatch");
        assert_eq!(ap2.len(), n, "ap2 length mismatch");
        ap1.iter().chain(ap2).map(|b| self.bit_arg(b)).collect()
    }

    fn cand_args(
        &self,
        cand: CandVar,
        pkt_in: &[AbstractBit],
        pkt_start: &[AbstractBit],
        states: &[usize],
        pkt_end: &[AbstractBit],
        pkt_out: &[AbstractBit],
    ) -> Vec<Arg> {
        let n = self.num_vars as usize;
        let ns = self.cands[cand.0 as usize].num_states();
        assert_eq!(pkt_in.len(), n, "pkt_in length mismatch");
        assert_eq!(pkt_start.len(), n, "pkt_start length mismatch");
        assert_eq!(pkt_end.len(), n, "pkt_end length mismatch");
        assert_eq!(pkt_out.len(), n, "pkt_out length mismatch");
        assert_eq!(states.len(), ns, "states length mismatch");
        let mut args = Vec::with_capacity(4 * n + ns);
        args.extend(pkt_in.iter().chain(pkt_start).map(|b| self.bit_arg(b)));
        args.extend(states.iter().map(|&s| {
            assert!(s < ns, "state value out of range");
            Arg::State(s)
        }));
        args.extend(pkt_end.iter().chain(pkt_out).map(|b| self.bit_arg(b)));
        args
    }

    /// Identical symbolic applications share a result variable immediately.
    /// Applications whose arguments only coincide in a model are handled lazily.
    fn apply_slot(&mut self, slot: SlotRef, args: Vec<Arg>) -> usize {
        match slot {
            SlotRef::Spp(i) => {
                assert!(i < self.num_spps, "SPP slot out of range");
            }
            SlotRef::Cand(i) => {
                let _ = self.cands[i as usize];
            }
        }
        let key = (slot, args);
        if let Some(&atom) = self.applications.get(&key) {
            return atom;
        }
        let atom = self.atoms.len();
        self.atoms.push(Atom {
            slot,
            args: key.1.clone(),
            result: self.solver.fresh(),
        });
        self.applications.insert(key, atom);
        atom
    }

    /// Solve the Boolean abstraction, then check predicate congruence with a
    /// hash table. Each violating pair adds a globally valid Ackermann lemma.
    /// A pair cannot violate again, so refinement terminates after at most
    /// quadratically many lemmas, without eagerly allocating them all.
    fn solve(&mut self) -> Result<Model, InconsistentError> {
        loop {
            let model = self.solver.solve().ok_or(InconsistentError)?;
            let mut groups: HashMap<(SlotRef, Vec<u64>), usize> = HashMap::new();
            let mut conflicts = Vec::new();
            for (idx, atom) in self.atoms.iter().enumerate() {
                let key = (
                    atom.slot,
                    atom.args.iter().map(|a| a.value(&model)).collect(),
                );
                if let Some(&other) = groups.get(&key) {
                    if model.value(atom.result) != model.value(self.atoms[other].result) {
                        conflicts.push((other, idx));
                    }
                } else {
                    groups.insert(key, idx);
                }
            }
            if conflicts.is_empty() {
                return Ok(model);
            }
            for (a, b) in conflicts {
                self.ackermannize(a, b);
            }
        }
    }

    fn ackermannize(&mut self, a: usize, b: usize) {
        let a = &self.atoms[a];
        let b = &self.atoms[b];
        assert_eq!(a.slot, b.slot);
        assert_eq!(a.args.len(), b.args.len());
        let mut guard = Vec::new();
        for (&x, &y) in a.args.iter().zip(&b.args) {
            match (x, y) {
                (Arg::Bit(x), Arg::Bit(y)) => {
                    let eq = self.solver.equal(x, y);
                    if eq == FALSE {
                        return;
                    }
                    if eq != TRUE {
                        guard.push(!eq);
                    }
                }
                (Arg::State(x), Arg::State(y)) => {
                    if x != y {
                        return;
                    }
                }
                _ => unreachable!("predicate argument sorts are fixed"),
            }
        }
        // If all arguments are equal, the result variables must agree.
        let mut forward = guard.clone();
        forward.extend([!a.result, b.result]);
        self.solver.clause(forward);
        guard.extend([a.result, !b.result]);
        self.solver.clause(guard);
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
    /// a consistent SAT assignment.
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
            let mut fallback = None;
            let mut covered = false;
            for lit in clause {
                let atom = &self.atoms[lit.atom];
                let value = model.value(atom.result);
                if minimize && value != lit.polarity {
                    continue;
                }
                let concrete: Vec<u64> = atom.args.iter().map(|a| a.value(model)).collect();
                if !minimize {
                    if selected.insert((atom.slot, concrete.clone())) {
                        self.push_example(&mut out, atom.slot, &concrete, value);
                    }
                    continue;
                }
                if selected.contains(&(atom.slot, concrete.clone())) {
                    covered = true;
                    break;
                }
                if fallback.is_none() {
                    fallback = Some((concrete, atom.slot, lit.polarity));
                }
            }
            if !minimize || covered {
                continue;
            }
            let (concrete, slot, polarity) = fallback.expect("SAT model must satisfy every clause");
            selected.insert((slot, concrete.clone()));
            self.push_example(&mut out, slot, &concrete, polarity);
        }

        out
    }

    /// Append one training example to `out`: `slot`'s membership at the
    /// concrete (model-evaluated) arguments `concrete`, labelled `label`.
    fn push_example(
        &self,
        out: &mut SelectedExamples,
        slot: SlotRef,
        concrete: &[u64],
        label: bool,
    ) {
        let n = self.num_vars as usize;
        match slot {
            SlotRef::Spp(i) => {
                let bits: Vec<bool> = concrete.iter().map(|&b| b != 0).collect();
                assert_eq!(bits.len(), 2 * n);
                out.spps.entry(i).or_default().push(Example {
                    input: bits[..n].to_vec(),
                    output: bits[n..].to_vec(),
                    in_spp: label,
                });
            }
            SlotRef::Cand(i) => {
                let ns = self.cands[i as usize].num_states();
                assert_eq!(concrete.len(), 4 * n + ns);
                let pkt_in = concrete[..n].iter().map(|&b| b != 0).collect();
                let pkt_start = concrete[n..2 * n].iter().map(|&b| b != 0).collect();
                let states: Vec<usize> = concrete[2 * n..2 * n + ns]
                    .iter()
                    .map(|&s| s as usize)
                    .collect();
                let pkt_end = concrete[2 * n + ns..3 * n + ns]
                    .iter()
                    .map(|&b| b != 0)
                    .collect();
                let pkt_out = concrete[3 * n + ns..4 * n + ns]
                    .iter()
                    .map(|&b| b != 0)
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
        let mut learner = Native::new(N);
        let s = learner.fresh_spp();
        let result = learner.extract(&mut store).unwrap();
        assert_eq!(result.spps[&s], store.zero);
    }

    #[test]
    fn test_empty_no_spps() {
        // No SPP slots allocated: extract returns an empty map.
        let mut store = SPPstore::new(N);
        let mut learner = Native::new(N);
        let result = learner.extract(&mut store).unwrap();
        assert!(result.spps.is_empty());
    }

    #[test]
    fn test_single_positive_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
        let _s = learner.fresh_spp();
        let e: Vec<Existential> = (0..N).map(|_| learner.fresh_existential()).collect();
        learner.add_sp_membership(store.sp.one, &e, &store.sp);
        assert!(learner.extract(&mut store).is_ok());
    }

    #[test]
    fn test_sp_membership_empty_set_is_unsat() {
        let mut store = SPPstore::new(N);
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
        let mut learner = Native::new(N);
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
    fn shared_atom_learner(store: &mut SPPstore, positive: bool) -> (Native<'static>, SppVar) {
        let p = [false, false, false];
        let q = [true, true, true];
        let r = [true, false, true];
        let t = [false, true, false];
        let mut learner = Native::new(N);
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
        let (mut learner, s) = shared_atom_learner(&mut store, true);
        let model = learner.solve().unwrap();
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
        let (mut learner, s) = shared_atom_learner(&mut store, false);
        let model = learner.solve().unwrap();
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
        let (mut learner, s) = shared_atom_learner(&mut store, true);
        let model = learner.solve().unwrap();
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
        let mut learner = Native::new(N);
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
    #[test]
    fn lazy_congruence_rejects_equal_existential_arguments() {
        let mut store = SPPstore::new(1);
        let mut learner = Native::new(1);
        let s = learner.fresh_spp();
        let x = learner.fresh_existential();
        let y = learner.fresh_existential();
        let singleton = store.sp.singleton(&[true]);
        learner.add_sp_membership(singleton, &[x], &store.sp);
        learner.add_sp_membership(singleton, &[y], &store.sp);
        for (e, polarity) in [(x, true), (y, false)] {
            learner.add_clause(
                AbstractClause {
                    literals: vec![spp_lit(
                        s,
                        vec![AbstractBit::Exist(e)],
                        ap_concrete(&[false]),
                        polarity,
                    )],
                },
                &mut store.sp,
            );
        }
        // The Boolean abstraction is satisfiable until predicate congruence
        // notices that distinct symbolic applications have equal arguments.
        assert!(learner.solver.solve().is_some());
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn congruence_can_separate_arguments_and_survives_refinement() {
        let mut store = SPPstore::new(1);
        let mut learner = Native::new(1);
        let s = learner.fresh_spp();
        let x = learner.fresh_existential();
        let y = learner.fresh_existential();
        for (e, polarity) in [(x, true), (y, false)] {
            learner.add_clause(
                AbstractClause {
                    literals: vec![spp_lit(
                        s,
                        vec![AbstractBit::Exist(e)],
                        ap_concrete(&[false]),
                        polarity,
                    )],
                },
                &mut store.sp,
            );
        }
        let model = learner.solve().unwrap();
        assert_ne!(
            model.value(learner.existentials[x.0 as usize]),
            model.value(learner.existentials[y.0 as usize])
        );
        assert!(learner.extract(&mut store).is_ok());
        // Allocate and constrain a new application after earlier solves.
        let z = learner.fresh_existential();
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(
                    s,
                    vec![AbstractBit::Exist(z)],
                    ap_concrete(&[false]),
                    true,
                )],
            },
            &mut store.sp,
        );
        assert!(learner.extract(&mut store).is_ok());
        let singleton = store.sp.singleton(&[true]);
        for e in [x, y, z] {
            learner.add_sp_membership(singleton, &[e], &store.sp);
        }
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn zero_width_packets_and_empty_clause() {
        let mut store = SPPstore::new(0);
        let mut learner = Native::new(0);
        let s = learner.fresh_spp();
        learner.add_clause(
            AbstractClause {
                literals: vec![spp_lit(s, vec![], vec![], true)],
            },
            &mut store.sp,
        );
        assert_eq!(learner.extract(&mut store).unwrap().spps[&s], store.one);
        learner.add_clause(AbstractClause { literals: vec![] }, &mut store.sp);
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn cand_congruence_includes_all_concrete_states() {
        let mut store = SPPstore::new(1);
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![], vec![], vec![]],
            outputs: vec![store.zero; 3],
        };
        let mut learner = Native::new(1);
        let cv = learner.fresh_cand(&dfa);
        let x = learner.fresh_existential();
        let y = learner.fresh_existential();
        let singleton = store.sp.singleton(&[true]);
        for e in [x, y] {
            learner.add_sp_membership(singleton, &[e], &store.sp);
        }
        let make_lit = |e, states, polarity| Literal::Cand {
            cand: cv,
            pkt_in: vec![AbstractBit::Exist(e)],
            pkt_start: ap_concrete(&[false]),
            states,
            pkt_end: ap_concrete(&[false]),
            pkt_out: ap_concrete(&[false]),
            polarity,
        };
        learner.add_clause(
            AbstractClause {
                literals: vec![make_lit(x, vec![2, 0, 1], true)],
            },
            &mut store.sp,
        );
        learner.add_clause(
            AbstractClause {
                literals: vec![make_lit(y, vec![2, 1, 1], false)],
            },
            &mut store.sp,
        );
        let solution = learner.extract(&mut store).unwrap();
        let mut input = Input {
            pkt_in: vec![true],
            pkt_start: vec![false],
            states: vec![2, 0, 1],
            pkt_end: vec![false],
            pkt_out: vec![false],
        };
        assert!(solution.cands[&cv].accepts_input(&mut store, &input));
        input.states[1] = 1;
        assert!(!solution.cands[&cv].accepts_input(&mut store, &input));
        learner.add_clause(
            AbstractClause {
                literals: vec![make_lit(y, vec![2, 0, 1], false)],
            },
            &mut store.sp,
        );
        assert!(learner.solver.solve().is_some());
        assert!(learner.extract(&mut store).is_err());
    }

    #[test]
    fn sp_bitblasting_matches_every_two_bit_set() {
        for mask in 0..16 {
            for packet in 0..4 {
                let mut store = SPPstore::new(2);
                let mut sp = store.sp.zero;
                for p in 0..4 {
                    if mask & (1 << p) != 0 {
                        let singleton = store.sp.singleton(&[p & 1 != 0, p & 2 != 0]);
                        sp = store.sp.union(sp, singleton);
                    }
                }
                let mut learner = Native::new(2);
                let vars = [learner.fresh_existential(), learner.fresh_existential()];
                learner.add_sp_membership(sp, &vars, &store.sp);
                let pin = store.sp.singleton(&[packet & 1 != 0, packet & 2 != 0]);
                learner.add_sp_membership(pin, &vars, &store.sp);
                assert_eq!(learner.solve().is_ok(), mask & (1 << packet) != 0);
            }
        }
    }

    #[test]
    fn randomized_incremental_uf_matches_exhaustive_oracle() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xACCE_1234);
        // Two one-bit packet predicates have four truth-table entries each.
        // Enumerate all existential assignments and predicate interpretations
        // independently of both the SAT encoding and its congruence checker.
        for trial in 0..400 {
            let mut store = SPPstore::new(1);
            let mut learner = Native::new(1);
            let slots = [learner.fresh_spp(), learner.fresh_spp()];
            let exists = [learner.fresh_existential(), learner.fresh_existential()];
            let masks: [u32; 2] = [rng.random_range(1..4), rng.random_range(1..4)];
            for (e, mask) in exists.into_iter().zip(masks) {
                let sp = match mask {
                    1 => store.sp.singleton(&[false]),
                    2 => store.sp.singleton(&[true]),
                    3 => store.sp.one,
                    _ => unreachable!(),
                };
                learner.add_sp_membership(sp, &[e], &store.sp);
            }
            let mut clauses: Vec<Vec<(usize, usize, usize, bool)>> = Vec::new();
            let eval = |arg: usize, assignment: usize| match arg {
                0 => false,
                1 => true,
                2 => assignment & 1 != 0,
                3 => assignment & 2 != 0,
                _ => unreachable!(),
            };
            for round in 0..10 {
                let clause: Vec<_> = (0..rng.random_range(1..4))
                    .map(|_| {
                        (
                            rng.random_range(0..2),
                            rng.random_range(0..4),
                            rng.random_range(0..4),
                            rng.random::<bool>(),
                        )
                    })
                    .collect();
                let arg = |a| match a {
                    0 => c(false),
                    1 => c(true),
                    2 => AbstractBit::Exist(exists[0]),
                    3 => AbstractBit::Exist(exists[1]),
                    _ => unreachable!(),
                };
                learner.add_clause(
                    AbstractClause {
                        literals: clause
                            .iter()
                            .map(|&(slot, p, q, polarity)| {
                                spp_lit(slots[slot], vec![arg(p)], vec![arg(q)], polarity)
                            })
                            .collect(),
                    },
                    &mut store.sp,
                );
                clauses.push(clause);
                let expected = (0..4).any(|assignment| {
                    if masks
                        .iter()
                        .enumerate()
                        .any(|(idx, mask)| mask & (1 << ((assignment >> idx) & 1)) == 0)
                    {
                        return false;
                    }
                    (0..256).any(|table| {
                        clauses.iter().all(|clause| {
                            clause.iter().any(|&(slot, p, q, polarity)| {
                                let bit = 4 * slot
                                    + 2 * eval(p, assignment) as usize
                                    + eval(q, assignment) as usize;
                                (table & (1 << bit) != 0) == polarity
                            })
                        })
                    })
                });
                let result = learner.solve();
                assert_eq!(
                    result.is_ok(),
                    expected,
                    "trial {trial}, round {round}, clauses {clauses:?}"
                );
                let Ok(model) = result else {
                    break;
                };
                let assignment = exists.iter().enumerate().fold(0, |acc, (idx, e)| {
                    acc | ((model.value(learner.existentials[e.0 as usize]) as usize) << idx)
                });
                assert!(
                    masks
                        .iter()
                        .enumerate()
                        .all(|(idx, mask)| mask & (1 << ((assignment >> idx) & 1)) != 0)
                );
                // Both extraction modes must train diagrams satisfying the
                // original clauses under the returned existential assignment.
                for minimize in [true, false] {
                    let mut examples = learner.select_examples(&model, minimize);
                    let spps: Vec<_> = slots
                        .iter()
                        .map(|s| {
                            learn_spp(examples.spps.remove(&s.0).unwrap_or_default(), &mut store)
                                .unwrap()
                        })
                        .collect();
                    assert!(
                        clauses.iter().all(|clause| clause.iter().any(
                            |&(slot, p, q, polarity)| {
                                spp_accepts(
                                    &store,
                                    spps[slot],
                                    &[eval(p, assignment)],
                                    &[eval(q, assignment)],
                                ) == polarity
                            }
                        )),
                        "extraction trial {trial}, round {round}, minimize {minimize}"
                    );
                }
            }
        }
    }
    #[test]
    fn clause_pair_comparison_ignores_order_and_repeated_literals() {
        let mut store = SPPstore::new(1);
        let mut learner = Native::new(1);
        let s = learner.fresh_spp();
        let t = learner.fresh_spp();
        let member = Literal::SppMember {
            spp: s,
            in_sp: store.sp.one,
            out_sp: store.sp.one,
        };
        let negative = spp_lit(t, ap_concrete(&[true]), ap_concrete(&[false]), false);
        let first = AbstractClause {
            literals: vec![member.clone(), negative.clone()],
        };
        let reordered = AbstractClause {
            literals: vec![negative, member.clone(), member.clone()],
        };
        assert!(first.same_literals(&reordered, &mut store.sp));
        let different = AbstractClause {
            literals: vec![member],
        };
        assert!(!first.same_literals(&different, &mut store.sp));
        let x = learner.fresh_existential();
        let y = learner.fresh_existential();
        let existential = |e, polarity| AbstractClause {
            literals: vec![spp_lit(
                s,
                vec![AbstractBit::Exist(e)],
                ap_concrete(&[false]),
                polarity,
            )],
        };
        assert!(!existential(x, true).same_literals(&existential(x, false), &mut store.sp));
        assert!(!existential(x, true).same_literals(&existential(y, true), &mut store.sp));
        assert!(
            learner.clauses.is_empty(),
            "comparison does not assert clauses"
        );
    }

    #[test]
    fn clause_pair_comparison_uses_membership_merging_before_allocation() {
        let mut store = SPPstore::new(1);
        let mut learner = Native::new(1);
        let s = learner.fresh_spp();
        let input = store.sp.singleton(&[false]);
        let out0 = store.sp.singleton(&[false]);
        let out1 = store.sp.singleton(&[true]);
        let split = AbstractClause {
            literals: vec![
                Literal::SppMember {
                    spp: s,
                    in_sp: input,
                    out_sp: out0,
                },
                Literal::SppMember {
                    spp: s,
                    in_sp: input,
                    out_sp: out1,
                },
            ],
        };
        let merged = AbstractClause {
            literals: vec![Literal::SppMember {
                spp: s,
                in_sp: input,
                out_sp: store.sp.one,
            }],
        };
        assert_eq!(
            split.same_literals(&merged, &mut store.sp),
            flags::clause_merge()
        );
        assert!(learner.existentials.is_empty());
        assert!(learner.atoms.is_empty());
    }
}
