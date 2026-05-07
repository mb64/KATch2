//! Clause-based SPP learner.
//!
//! # Overview
//!
//! An [`AbstractClause`] is a disjunction of [`Literal`]s.  Each literal
//! asserts that an abstract packet pair is (or is not) in the SPP.  Like
//! [`super::learner::AbstractPacket`]s, abstract pairs are parameterized by a
//! *reference packet* `p`: every `None` field at position `i` is filled with
//! `p[i]`.  `(None, None)` at position `i` is a correlated wildcard — both
//! input and output bits equal `p[i]`.
//!
//! [`ClauseLearner`] accumulates abstract clauses and extracts a consistent
//! SPP:
//!
//! ```ignore
//! let mut learner = ClauseLearner::new(num_vars);
//! learner.add_clause(AbstractClause { literals: vec![
//!     Literal { ap1, ap2, polarity: true },
//! ]});
//! let spp = learner.extract(&mut spp_store)?;  // Err(InconsistentError) if UNSAT
//! ```
//!
//! # Guarantees
//!
//! * The extracted SPP satisfies every abstract clause for every reference
//!   packet.
//! * [`extract`](ClauseLearner::extract) returns [`InconsistentError`] if the
//!   clauses are mutually contradictory (the underlying SAT instance is UNSAT).
//!
//! # Algorithm
//!
//! [`ClauseLearner::extract`] runs a CEGAR loop:
//! 1. Grow a set of concrete packet pairs by instantiating clauses with a few
//!    seed reference packets, then saturating via trie lookup.
//! 2. Build and solve a SAT instance (one Boolean variable per concrete pair).
//! 3. Generalize the SAT assignment back to abstract examples and feed them to
//!    the [`super::learner::Learner`].
//! 4. Validate the resulting SPP against every abstract clause. If a clause
//!    fails, extract a counterexample reference packet, instantiate the clause,
//!    add the new concrete pairs, and repeat from step 1.

use std::collections::{HashMap, HashSet, VecDeque};

use rustsat::solvers::{Solve, SolverResult};
use rustsat::types::{Clause, TernaryVal, Var as SatVar};
use rustsat_cadical::CaDiCaL;

use super::learner::{AbstractPacket, Learner};
use super::{SPP, SPPstore, Var};

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// One literal in a clause: an abstract pair plus a polarity.
///
/// `polarity = true` means "the pair is in the SPP"; `false` means "the pair
/// is not in the SPP".
pub struct Literal {
    pub ap1: AbstractPacket,
    pub ap2: AbstractPacket,
    pub polarity: bool,
}

/// An abstract clause: a disjunction of [`Literal`]s.
///
/// The clause is satisfied by a reference packet `p` when at least one literal
/// holds after instantiating the abstract packets with `p`.
pub struct AbstractClause {
    pub literals: Vec<Literal>,
}

/// Error returned by [`ClauseLearner::extract`] when the clauses are
/// mutually inconsistent (the SAT instance is UNSAT).
#[derive(Debug, Clone)]
pub struct InconsistentError;

impl std::fmt::Display for InconsistentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inconsistent clauses")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Abstract packet pair trie
// ────────────────────────────────────────────────────────────────────────────

/// A node in the abstract packet-pair trie.
///
/// Each level has nine branches, one per abstract bit-pair pattern:
/// - `b00 / b01 / b10 / b11`: both bits are concrete.
/// - `b0w / b1w`: first bit is concrete (false/true), second is a wildcard.
/// - `wb0 / wb1`: first bit is a wildcard, second is concrete (false/true).
/// - `nn`: correlated wildcard — both bits come from the reference packet (must be equal).
///
/// Insertion never expands half-wildcards; each abstract pair goes into exactly one branch.
/// Lookup follows all branches whose abstract pattern matches the given concrete bits.
struct TrieNode {
    b00: Option<Box<TrieNode>>,
    b01: Option<Box<TrieNode>>,
    b10: Option<Box<TrieNode>>,
    b11: Option<Box<TrieNode>>,
    b0w: Option<Box<TrieNode>>, // (Some(false), None)
    b1w: Option<Box<TrieNode>>, // (Some(true),  None)
    wb0: Option<Box<TrieNode>>, // (None, Some(false))
    wb1: Option<Box<TrieNode>>, // (None, Some(true))
    /// Correlated wildcard: `(None, None)` — both bits come from the reference packet.
    nn: Option<Box<TrieNode>>,
    /// Clause/literal indices stored at this leaf (populated only at depth == num_vars).
    data: Vec<(usize, usize)>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            b00: None,
            b01: None,
            b10: None,
            b11: None,
            b0w: None,
            b1w: None,
            wb0: None,
            wb1: None,
            nn: None,
            data: Vec::new(),
        }
    }
}

struct ApPairTrie {
    num_vars: Var,
    root: TrieNode,
}

impl ApPairTrie {
    fn new(num_vars: Var) -> Self {
        Self {
            num_vars,
            root: TrieNode::new(),
        }
    }

    /// Insert abstract pair `(ap1, ap2)` into the trie, associating it with
    /// `(clause_idx, lit_idx)` at the leaf.
    fn insert(
        &mut self,
        ap1: &AbstractPacket,
        ap2: &AbstractPacket,
        clause_idx: usize,
        lit_idx: usize,
    ) {
        assert_eq!(
            ap1.len(),
            self.num_vars as usize,
            "ap1 length {} != num_vars {}",
            ap1.len(),
            self.num_vars
        );
        assert_eq!(
            ap2.len(),
            self.num_vars as usize,
            "ap2 length {} != num_vars {}",
            ap2.len(),
            self.num_vars
        );
        Self::insert_rec(
            &mut self.root,
            ap1,
            ap2,
            0,
            self.num_vars,
            clause_idx,
            lit_idx,
        );
    }

    fn insert_rec(
        node: &mut TrieNode,
        ap1: &AbstractPacket,
        ap2: &AbstractPacket,
        depth: Var,
        num_vars: Var,
        clause_idx: usize,
        lit_idx: usize,
    ) {
        if depth == num_vars {
            node.data.push((clause_idx, lit_idx));
            return;
        }
        let b1 = ap1[depth as usize];
        let b2 = ap2[depth as usize];
        let child = Self::get_child_mut(node, b1, b2);
        Self::insert_rec(child, ap1, ap2, depth + 1, num_vars, clause_idx, lit_idx);
    }

    fn get_child_mut(node: &mut TrieNode, b1: Option<bool>, b2: Option<bool>) -> &mut TrieNode {
        let slot = match (b1, b2) {
            (Some(false), Some(false)) => &mut node.b00,
            (Some(false), Some(true)) => &mut node.b01,
            (Some(true), Some(false)) => &mut node.b10,
            (Some(true), Some(true)) => &mut node.b11,
            (Some(false), None) => &mut node.b0w,
            (Some(true), None) => &mut node.b1w,
            (None, Some(false)) => &mut node.wb0,
            (None, Some(true)) => &mut node.wb1,
            (None, None) => &mut node.nn,
        };
        slot.get_or_insert_with(|| Box::new(TrieNode::new()))
    }

    /// Return all `(clause_idx, lit_idx)` pairs for abstract pairs that match
    /// the concrete pair `(p1, p2)`.
    fn lookup(&self, p1: &[bool], p2: &[bool]) -> Vec<(usize, usize)> {
        assert_eq!(
            p1.len(),
            self.num_vars as usize,
            "p1 length {} != num_vars {}",
            p1.len(),
            self.num_vars
        );
        assert_eq!(
            p2.len(),
            self.num_vars as usize,
            "p2 length {} != num_vars {}",
            p2.len(),
            self.num_vars
        );
        let mut result = Vec::new();
        Self::lookup_rec(&self.root, p1, p2, 0, self.num_vars, &mut result);
        result
    }

    fn lookup_rec(
        node: &TrieNode,
        p1: &[bool],
        p2: &[bool],
        depth: Var,
        num_vars: Var,
        result: &mut Vec<(usize, usize)>,
    ) {
        if depth == num_vars {
            result.extend_from_slice(&node.data);
            return;
        }
        let b1 = p1[depth as usize];
        let b2 = p2[depth as usize];

        // Concrete branch: pattern (Some(b1), Some(b2)).
        let concrete = match (b1, b2) {
            (false, false) => node.b00.as_deref(),
            (false, true) => node.b01.as_deref(),
            (true, false) => node.b10.as_deref(),
            (true, true) => node.b11.as_deref(),
        };
        if let Some(child) = concrete {
            Self::lookup_rec(child, p1, p2, depth + 1, num_vars, result);
        }

        // Input-wildcard branch: pattern (Some(b1), None) — matches any b2.
        let bw = if b1 {
            node.b1w.as_deref()
        } else {
            node.b0w.as_deref()
        };
        if let Some(child) = bw {
            Self::lookup_rec(child, p1, p2, depth + 1, num_vars, result);
        }

        // Output-wildcard branch: pattern (None, Some(b2)) — matches any b1.
        let wb = if b2 {
            node.wb1.as_deref()
        } else {
            node.wb0.as_deref()
        };
        if let Some(child) = wb {
            Self::lookup_rec(child, p1, p2, depth + 1, num_vars, result);
        }

        // Correlated-wildcard branch: pattern (None, None) — matches when b1 == b2.
        if b1 == b2
            && let Some(child) = node.nn.as_deref()
        {
            Self::lookup_rec(child, p1, p2, depth + 1, num_vars, result);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// run_abstract: evaluate SPP on abstract pair → SP
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate an SPP on an abstract packet pair, returning an SP.
///
/// The returned SP represents the set of *reference packets* `p` such that
/// `(fill(ap1, p), fill(ap2, p))` is in `spp`.
///
/// The crucial property: `spp` accepts the pair for **every** instantiation
/// of `(ap1, ap2)` iff the returned SP equals the full set (`sp.one`).
///
/// Memoizes on `(spp, depth)` to avoid exponential blowup on shared SPP subtrees.
fn run_abstract(
    store: &mut SPPstore,
    spp: SPP,
    ap1: &AbstractPacket,
    ap2: &AbstractPacket,
) -> super::super::sp::SP {
    let mut memo = HashMap::new();
    run_abstract_rec(store, spp, ap1, ap2, 0, &mut memo)
}

fn run_abstract_rec(
    store: &mut SPPstore,
    spp: SPP,
    ap1: &AbstractPacket,
    ap2: &AbstractPacket,
    depth: Var,
    memo: &mut HashMap<(SPP, Var), super::super::sp::SP>,
) -> super::super::sp::SP {
    use super::super::sp::SP;

    let num_vars = ap1.len() as Var;

    // Leaf level: return the SP terminal (not the full-depth BDDs sp.zero/sp.one).
    if depth == num_vars {
        return if spp == SPP::new(1) {
            SP::new(1)
        } else {
            SP::new(0)
        };
    }

    // Check memo. Key is (spp, depth); ap1/ap2 are fixed for the whole run_abstract call.
    let key = (spp, depth);
    if let Some(&cached) = memo.get(&key) {
        return cached;
    }

    // Unpack SPP children (handle terminals defensively with virtual all-same children).
    let (x00, x01, x10, x11) = if spp == SPP::new(0) {
        (SPP::new(0), SPP::new(0), SPP::new(0), SPP::new(0))
    } else if spp == SPP::new(1) {
        (SPP::new(1), SPP::new(1), SPP::new(1), SPP::new(1))
    } else {
        let n = store.get(spp);
        (n.x00, n.x01, n.x10, n.x11)
    };
    let b1 = ap1[depth as usize];
    let b2 = ap2[depth as usize];

    // Determine which SPP children to recurse into for reference-packet bit = 0 and = 1.
    let (child_0, child_1) = match (b1, b2) {
        // Both concrete: p[depth] doesn't matter; both branches use the same child.
        (Some(false), Some(false)) => (x00, x00),
        (Some(false), Some(true)) => (x01, x01),
        (Some(true), Some(false)) => (x10, x10),
        (Some(true), Some(true)) => (x11, x11),
        // Input concrete, output from reference packet.
        (Some(false), None) => (x00, x01), // p=0→(false,false)=x00; p=1→(false,true)=x01
        (Some(true), None) => (x10, x11),  // p=0→(true,false)=x10;  p=1→(true,true)=x11
        // Output concrete, input from reference packet.
        (None, Some(false)) => (x00, x10), // p=0→(false,false)=x00; p=1→(true,false)=x10
        (None, Some(true)) => (x01, x11),  // p=0→(false,true)=x01;  p=1→(true,true)=x11
        // Correlated wildcard: p[depth] determines both bits equally.
        (None, None) => (x00, x11), // p=0→(false,false)=x00; p=1→(true,true)=x11
    };

    let sp0 = run_abstract_rec(store, child_0, ap1, ap2, depth + 1, memo);
    let sp1 = run_abstract_rec(store, child_1, ap1, ap2, depth + 1, memo);
    let res = store.sp.mk(sp0, sp1);
    memo.insert(key, res);
    res
}

// ────────────────────────────────────────────────────────────────────────────
// Helper: fill (instantiate) an abstract packet with a reference packet
// ────────────────────────────────────────────────────────────────────────────

fn fill(ap: &AbstractPacket, p: &[bool]) -> Vec<bool> {
    ap.iter()
        .enumerate()
        .map(|(i, &bit)| bit.unwrap_or(p[i]))
        .collect()
}

/// Compute the reference packet that, when used to instantiate `(ap1, ap2)`,
/// produces the concrete pair `(p1, p2)`.
///
/// If `ap1[i]` is `None`, the reference packet bit `i` is `p1[i]`.
/// If `ap2[i]` is `None` (and `ap1[i]` is not), it is `p2[i]`.
/// If both are `Some`, the reference bit is unused (set to `false`).
fn reference_packet(
    ap1: &AbstractPacket,
    ap2: &AbstractPacket,
    p1: &[bool],
    p2: &[bool],
) -> Vec<bool> {
    (0..ap1.len())
        .map(|i| {
            if ap1[i].is_none() {
                p1[i]
            } else if ap2[i].is_none() {
                p2[i]
            } else {
                false // unused
            }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// min_packet: deterministic counterexample extraction
// ────────────────────────────────────────────────────────────────────────────

/// Extract the lexicographically smallest (all-false-preferring) packet from an SP.
///
/// Always follows the `x0` (false) branch before `x1` (true), so the result
/// is deterministic and minimal. Returns `None` if the SP is empty.
fn min_packet(
    sp_store: &mut super::super::sp::SPstore,
    sp: super::super::sp::SP,
) -> Option<Vec<bool>> {
    use super::super::sp::SP;
    if sp_store.is_zero(sp) {
        return None;
    }
    if sp == SP::new(1) {
        return Some(vec![]);
    }
    let node = sp_store.get(sp);
    for (bit, child) in [(false, node.x0), (true, node.x1)] {
        if let Some(mut packet) = min_packet(sp_store, child) {
            packet.insert(0, bit);
            return Some(packet);
        }
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// ClauseLearner
// ────────────────────────────────────────────────────────────────────────────

/// Clause-based SPP learner.
///
/// Stores a set of abstract clauses, then uses a CEGAR loop to extract a
/// consistent SPP via SAT solving and the passive [`Learner`].
pub struct ClauseLearner {
    num_vars: Var,
    clauses: Vec<AbstractClause>,
    trie: ApPairTrie,

    // Incremental SAT solver.
    solver: CaDiCaL<'static, 'static>,

    // Bijection between concrete pairs and SAT variables.
    pair_to_sat_var: HashMap<(Vec<bool>, Vec<bool>), SatVar>,
    sat_var_to_pair: Vec<(Vec<bool>, Vec<bool>)>,

    // Tracks (clause_idx, ref_packet) instantiations already added to the solver,
    // to avoid duplicate SAT clauses.
    seen_instantiations: HashSet<(usize, Vec<bool>)>,
}

impl ClauseLearner {
    /// Create a new clause learner for packets with `num_vars` fields.
    pub fn new(num_vars: Var) -> Self {
        Self {
            num_vars,
            clauses: Vec::new(),
            trie: ApPairTrie::new(num_vars),
            solver: CaDiCaL::default(),
            pair_to_sat_var: HashMap::new(),
            sat_var_to_pair: Vec::new(),
            seen_instantiations: HashSet::new(),
        }
    }

    /// Add an abstract clause.
    pub fn add_clause(&mut self, clause: AbstractClause) {
        let clause_idx = self.clauses.len();
        for (lit_idx, lit) in clause.literals.iter().enumerate() {
            self.trie.insert(&lit.ap1, &lit.ap2, clause_idx, lit_idx);
        }
        self.clauses.push(clause);
    }

    /// Extract an SPP consistent with all stored clauses.
    ///
    /// Returns `Err(InconsistentError)` if the clauses are mutually contradictory.
    pub fn extract(&mut self, spp_store: &mut SPPstore) -> Result<SPP, InconsistentError> {
        // Seed the worklist with one reference-packet instantiation of each clause.
        let zero_ref: Vec<bool> = vec![false; self.num_vars as usize];
        let mut new_pairs: Vec<(Vec<bool>, Vec<bool>)> = Vec::new();
        for clause in &self.clauses {
            for lit in &clause.literals {
                let p1 = fill(&lit.ap1, &zero_ref);
                let p2 = fill(&lit.ap2, &zero_ref);
                new_pairs.push((p1, p2));
            }
        }

        loop {
            // Step 1: Saturate and build SAT clauses.
            self.saturate(new_pairs);

            // Step 2: Solve.
            let result = self.solver.solve().expect("SAT solver error");
            if result == SolverResult::Unsat {
                return Err(InconsistentError);
            }

            // Step 3: Generalize the assignment and feed it to the passive learner.
            let mut learner = Learner::new(self.num_vars);
            for (pair, &sat_var) in &self.pair_to_sat_var {
                let (p1, p2) = pair;
                let polarity = match self
                    .solver
                    .lit_val(sat_var.pos_lit())
                    .expect("SAT solver error")
                {
                    TernaryVal::True => true,
                    TernaryVal::False => false,
                    TernaryVal::DontCare => continue,
                };
                let (gen_ap1, gen_ap2) = self.generalize(p1, p2);
                if learner.add_example(gen_ap1, gen_ap2, polarity).is_err() {
                    assert!(false, "SAT model should be consistent by construction");
                }
            }

            // Step 4: Extract SPP from learner.
            let spp = learner.extract(spp_store);

            // Step 5: Validate the SPP against all abstract clauses.
            new_pairs = self.validate(spp, spp_store);
            if new_pairs.is_empty() {
                return Ok(spp);
            }
            // Counterexample pairs drive the next CEGAR iteration.
        }
    }

    // ── internal helpers ──────────────────────────────────────────────────

    /// Get or create the SAT variable for a concrete pair.
    fn get_or_create_sat_var(&mut self, p1: &[bool], p2: &[bool]) -> SatVar {
        let key = (p1.to_vec(), p2.to_vec());
        if let Some(&v) = self.pair_to_sat_var.get(&key) {
            return v;
        }
        let idx = self.sat_var_to_pair.len() as u32;
        let v = SatVar::new(idx);
        self.sat_var_to_pair.push(key.clone());
        self.pair_to_sat_var.insert(key, v);
        v
    }

    /// Saturation: process `initial` pairs, discover all reachable pairs via
    /// the trie, and add corresponding SAT clauses to the solver.
    fn saturate(&mut self, initial: Vec<(Vec<bool>, Vec<bool>)>) {
        let mut queue: VecDeque<(Vec<bool>, Vec<bool>)> = initial.into_iter().collect();
        let mut seen_pairs: HashSet<(Vec<bool>, Vec<bool>)> = HashSet::new();

        while let Some((p1, p2)) = queue.pop_front() {
            if !seen_pairs.insert((p1.clone(), p2.clone())) {
                continue;
            }
            self.get_or_create_sat_var(&p1, &p2);

            // Find all abstract literals that match (p1, p2).
            let matches = self.trie.lookup(&p1, &p2);

            for (clause_idx, lit_idx) in matches {
                // Pre-collect clause data to avoid conflicting borrows on `self`.
                let p_ref = {
                    let lit = &self.clauses[clause_idx].literals[lit_idx];
                    reference_packet(&lit.ap1, &lit.ap2, &p1, &p2)
                };
                let inst_key = (clause_idx, p_ref.clone());
                if self.seen_instantiations.contains(&inst_key) {
                    continue;
                }
                self.seen_instantiations.insert(inst_key);

                // Collect (concrete_pair, polarity) for each literal in this clause.
                let pairs_and_polarities: Vec<((Vec<bool>, Vec<bool>), bool)> = self.clauses
                    [clause_idx]
                    .literals
                    .iter()
                    .map(|lit| {
                        (
                            (fill(&lit.ap1, &p_ref), fill(&lit.ap2, &p_ref)),
                            lit.polarity,
                        )
                    })
                    .collect();

                // Enqueue newly discovered pairs and build the SAT clause.
                let mut sat_clause = Clause::new();
                for ((q1, q2), polarity) in pairs_and_polarities {
                    if !seen_pairs.contains(&(q1.clone(), q2.clone())) {
                        queue.push_back((q1.clone(), q2.clone()));
                    }
                    let sat_var = self.get_or_create_sat_var(&q1, &q2);
                    let sat_lit = if polarity {
                        sat_var.pos_lit()
                    } else {
                        sat_var.neg_lit()
                    };
                    sat_clause.add(sat_lit);
                }
                self.solver
                    .add_clause_ref(&sat_clause)
                    .expect("SAT solver error");
            }
        }
    }

    /// Compute the generalized abstract pair for concrete `(p1, p2)` by
    /// intersecting all matching abstract literals from the trie.
    ///
    /// A position is constrained (`Some(v)`) only if at least one matching
    /// abstract literal has a concrete value there; otherwise it is `None`.
    fn generalize(&self, p1: &[bool], p2: &[bool]) -> (AbstractPacket, AbstractPacket) {
        let n = self.num_vars as usize;
        let mut constrained_ap1 = vec![false; n]; // whether ap1[i] is constrained
        let mut constrained_ap2 = vec![false; n];

        let matches = self.trie.lookup(p1, p2);
        for (clause_idx, lit_idx) in matches {
            let lit = &self.clauses[clause_idx].literals[lit_idx];
            for i in 0..n {
                if lit.ap1[i].is_some() {
                    constrained_ap1[i] = true;
                }
                if lit.ap2[i].is_some() {
                    constrained_ap2[i] = true;
                }
            }
        }

        let gen_ap1: AbstractPacket = (0..n)
            .map(|i| {
                if constrained_ap1[i] {
                    Some(p1[i])
                } else {
                    None
                }
            })
            .collect();
        let gen_ap2: AbstractPacket = (0..n)
            .map(|i| {
                if constrained_ap2[i] {
                    Some(p2[i])
                } else {
                    None
                }
            })
            .collect();
        (gen_ap1, gen_ap2)
    }

    /// Validate the SPP against all abstract clauses.
    ///
    /// Returns a list of concrete pairs that are counterexamples (pairs from
    /// clause instantiations that witness a failing clause). An empty result
    /// means the SPP satisfies all clauses.
    fn validate(&self, spp: SPP, spp_store: &mut SPPstore) -> Vec<(Vec<bool>, Vec<bool>)> {
        let mut bad_pairs: Vec<(Vec<bool>, Vec<bool>)> = Vec::new();

        for clause in &self.clauses {
            // Build the SP of reference packets that satisfy this clause.
            let mut clause_sp = spp_store.sp.zero;
            for lit in &clause.literals {
                let lit_sp = run_abstract(spp_store, spp, &lit.ap1, &lit.ap2);
                let lit_sp = if lit.polarity {
                    lit_sp
                } else {
                    spp_store.sp.complement(lit_sp)
                };
                clause_sp = spp_store.sp.union(clause_sp, lit_sp);
            }

            // Check if clause_sp is the full set (all reference packets satisfy the clause).
            let bad_sp = spp_store.sp.complement(clause_sp);
            if !spp_store.sp.is_zero(bad_sp) {
                // Extract the smallest counterexample reference packet for determinism.
                if let Some(p_ref) = min_packet(&mut spp_store.sp, bad_sp) {
                    for lit in &clause.literals {
                        let q1 = fill(&lit.ap1, &p_ref);
                        let q2 = fill(&lit.ap2, &p_ref);
                        bad_pairs.push((q1, q2));
                    }
                }
            }
        }

        bad_pairs
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::ApPairTrie;
    use super::*;

    const N: Var = 3;

    // ── trie property tests ───────────────────────────────────────────────
    //
    // Core property: for any abstract pair (ap1, ap2) inserted at (c, l),
    // and any concrete pair (p1, p2),
    //     trie.lookup(p1, p2).contains(&(c, l))
    //         iff
    //     abstract_matches(ap1, ap2, p1, p2)
    //
    // where abstract_matches holds iff at every position i:
    //   (Some(v1), Some(v2)) => p1[i]==v1 && p2[i]==v2
    //   (Some(v1), None)     => p1[i]==v1
    //   (None, Some(v2))     => p2[i]==v2
    //   (None, None)         => p1[i]==p2[i]   (correlated)

    /// Ground-truth matching predicate.
    fn abstract_matches(
        ap1: &[Option<bool>],
        ap2: &[Option<bool>],
        p1: &[bool],
        p2: &[bool],
    ) -> bool {
        ap1.iter()
            .zip(ap2)
            .zip(p1)
            .zip(p2)
            .all(|(((b1, b2), c1), c2)| match (b1, b2) {
                (Some(v1), Some(v2)) => c1 == v1 && c2 == v2,
                (Some(v1), None) => c1 == v1,
                (None, Some(v2)) => c2 == v2,
                (None, None) => c1 == c2,
            })
    }

    /// Generate a random abstract bit: Some(false), Some(true), or None, each with prob 1/3.
    fn rand_abstract_bit() -> Option<bool> {
        match rand::random::<u8>() % 3 {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn rand_abstract_packet(n: usize) -> Vec<Option<bool>> {
        (0..n).map(|_| rand_abstract_bit()).collect()
    }

    fn rand_concrete_packet(n: usize) -> Vec<bool> {
        (0..n).map(|_| rand::random::<bool>()).collect()
    }

    /// Enumerate all 2^n concrete packets of length n.
    fn all_concrete_packets(n: usize) -> Vec<Vec<bool>> {
        (0u32..(1u32 << n))
            .map(|mask| (0..n).map(|i| (mask >> i) & 1 == 1).collect())
            .collect()
    }

    /// Property: single insert — lookup returns the item iff the abstract pair matches.
    /// Uses exhaustive enumeration of concrete pairs for small num_vars.
    #[test]
    fn test_trie_single_insert_soundness_completeness() {
        const ITERS: usize = 400;
        for _ in 0..ITERS {
            // Use num_vars 1–3 so exhaustive enumeration stays cheap (≤64 pairs).
            let num_vars = (rand::random::<u32>() % 3 + 1) as Var;
            let n = num_vars as usize;
            let ap1 = rand_abstract_packet(n);
            let ap2 = rand_abstract_packet(n);

            let mut trie = ApPairTrie::new(num_vars);
            trie.insert(&ap1, &ap2, 0, 0);

            for p1 in all_concrete_packets(n) {
                for p2 in all_concrete_packets(n) {
                    let found = trie.lookup(&p1, &p2).contains(&(0, 0));
                    let expected = abstract_matches(&ap1, &ap2, &p1, &p2);
                    assert_eq!(
                        found, expected,
                        "ap1={:?} ap2={:?} p1={:?} p2={:?}: expected {} got {}",
                        ap1, ap2, p1, p2, expected, found
                    );
                }
            }
        }
    }

    /// Property: multiple independent inserts — each abstract pair is found iff it matches,
    /// regardless of what other pairs were inserted.
    #[test]
    fn test_trie_multiple_inserts_independence() {
        const ITERS: usize = 200;
        const MAX_PAIRS: usize = 8;
        for _ in 0..ITERS {
            let num_vars = (rand::random::<u32>() % 3 + 1) as Var;
            let n = num_vars as usize;
            let num_pairs = rand::random::<u32>() as usize % MAX_PAIRS + 2;

            // Insert a batch of random abstract pairs.
            let mut trie = ApPairTrie::new(num_vars);
            let pairs: Vec<(Vec<Option<bool>>, Vec<Option<bool>>)> = (0..num_pairs)
                .map(|_| (rand_abstract_packet(n), rand_abstract_packet(n)))
                .collect();
            for (i, (ap1, ap2)) in pairs.iter().enumerate() {
                trie.insert(ap1, ap2, i, 0);
            }

            // For each random concrete pair, each abstract pair must appear in the result
            // iff it matches — independently of all others.
            for _ in 0..(1 << n).min(32) {
                let p1 = rand_concrete_packet(n);
                let p2 = rand_concrete_packet(n);
                let results = trie.lookup(&p1, &p2);
                for (i, (ap1, ap2)) in pairs.iter().enumerate() {
                    let found = results.contains(&(i, 0));
                    let expected = abstract_matches(ap1, ap2, &p1, &p2);
                    assert_eq!(
                        found, expected,
                        "pair {}: ap1={:?} ap2={:?} p1={:?} p2={:?}",
                        i, ap1, ap2, p1, p2
                    );
                }
            }
        }
    }

    /// Property: no spurious entries — lookup never returns a (clause, lit) that was not inserted.
    #[test]
    fn test_trie_no_spurious_results() {
        const ITERS: usize = 200;
        for _ in 0..ITERS {
            let num_vars = (rand::random::<u32>() % 3 + 1) as Var;
            let n = num_vars as usize;
            let num_pairs = rand::random::<u32>() as usize % 6 + 1;

            let mut trie = ApPairTrie::new(num_vars);
            let pairs: Vec<(Vec<Option<bool>>, Vec<Option<bool>>)> = (0..num_pairs)
                .map(|_| (rand_abstract_packet(n), rand_abstract_packet(n)))
                .collect();
            for (i, (ap1, ap2)) in pairs.iter().enumerate() {
                trie.insert(ap1, ap2, i, 0);
            }

            for p1 in all_concrete_packets(n) {
                for p2 in all_concrete_packets(n) {
                    let results = trie.lookup(&p1, &p2);
                    // Every returned entry must correspond to an actual match.
                    for &(c, l) in &results {
                        assert_eq!(l, 0, "unexpected lit_idx");
                        assert!(c < num_pairs, "out-of-range clause_idx {}", c);
                        let (ap1, ap2) = &pairs[c];
                        assert!(
                            abstract_matches(ap1, ap2, &p1, &p2),
                            "spurious result ({}): ap1={:?} ap2={:?} p1={:?} p2={:?}",
                            c,
                            ap1,
                            ap2,
                            p1,
                            p2
                        );
                    }
                }
            }
        }
    }

    /// Property: same abstract pair inserted multiple times produces duplicate results.
    /// This confirms the trie accumulates data rather than deduplicating.
    #[test]
    fn test_trie_duplicate_inserts_preserved() {
        const ITERS: usize = 100;
        for _ in 0..ITERS {
            let num_vars = (rand::random::<u32>() % 3 + 1) as Var;
            let n = num_vars as usize;
            let ap1 = rand_abstract_packet(n);
            let ap2 = rand_abstract_packet(n);

            // Insert the same abstract pair twice with different clause indices.
            let mut trie = ApPairTrie::new(num_vars);
            trie.insert(&ap1, &ap2, 0, 0);
            trie.insert(&ap1, &ap2, 1, 0);

            // On any matching concrete pair, both entries must appear.
            for p1 in all_concrete_packets(n) {
                for p2 in all_concrete_packets(n) {
                    if abstract_matches(&ap1, &ap2, &p1, &p2) {
                        let results = trie.lookup(&p1, &p2);
                        assert!(results.contains(&(0, 0)), "missing first insert");
                        assert!(results.contains(&(1, 0)), "missing second insert");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_trie_insert_wrong_length_ap1() {
        let mut trie = ApPairTrie::new(3);
        trie.insert(&vec![None, None], &vec![None::<bool>, None, None], 0, 0); // ap1 too short
    }

    #[test]
    #[should_panic]
    fn test_trie_insert_wrong_length_ap2() {
        let mut trie = ApPairTrie::new(3);
        trie.insert(&vec![None::<bool>, None, None], &vec![None, None], 0, 0); // ap2 too short
    }

    #[test]
    #[should_panic]
    fn test_trie_lookup_wrong_length_p1() {
        let trie = ApPairTrie::new(3);
        let _ = trie.lookup(&[false, false], &[false, false, false]); // p1 too short
    }

    #[test]
    #[should_panic]
    fn test_trie_lookup_wrong_length_p2() {
        let trie = ApPairTrie::new(3);
        let _ = trie.lookup(&[false, false, false], &[false, false]); // p2 too short
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Check whether a concrete packet pair is accepted by an SPP.
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

    /// Check that the extracted SPP satisfies a clause for a given reference packet.
    fn clause_satisfied(
        store: &mut SPPstore,
        spp: SPP,
        clause: &AbstractClause,
        p_ref: &[bool],
    ) -> bool {
        clause.literals.iter().any(|lit| {
            let p1 = fill(&lit.ap1, p_ref);
            let p2 = fill(&lit.ap2, p_ref);
            let in_spp = spp_accepts(store, spp, &p1, &p2);
            (lit.polarity && in_spp) || (!lit.polarity && !in_spp)
        })
    }

    /// All reference packets that can instantiate a clause that uses wildcards.
    /// We check all 2^N reference packets for thoroughness in tests.
    fn all_reference_packets(num_vars: Var) -> Vec<Vec<bool>> {
        let n = num_vars as usize;
        (0u32..(1u32 << n))
            .map(|mask| (0..n).map(|i| (mask >> i) & 1 == 1).collect())
            .collect()
    }

    // ── tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_clause_learner() {
        let mut store = SPPstore::new(N);
        let mut learner = ClauseLearner::new(N);
        // No clauses → should return the zero SPP without error.
        let spp = learner.extract(&mut store).unwrap();
        assert_eq!(spp, store.zero);
    }

    #[test]
    fn test_single_positive_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ClauseLearner::new(N);

        // Clause: (0,0,0) → (1,1,1) is in the SPP.
        let ap1 = vec![Some(false), Some(false), Some(false)];
        let ap2 = vec![Some(true), Some(true), Some(true)];
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: ap1.clone(),
                ap2: ap2.clone(),
                polarity: true,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();

        // The SPP must accept (0,0,0) → (1,1,1).
        assert!(spp_accepts(
            &store,
            spp,
            &[false, false, false],
            &[true, true, true]
        ));
    }

    #[test]
    fn test_single_negative_unit_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ClauseLearner::new(N);

        // Clause: (1,1,1) → (0,0,0) is NOT in the SPP.
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: vec![Some(true), Some(true), Some(true)],
                ap2: vec![Some(false), Some(false), Some(false)],
                polarity: false,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();

        assert!(!spp_accepts(
            &store,
            spp,
            &[true, true, true],
            &[false, false, false]
        ));
    }

    #[test]
    fn test_conflicting_unit_clauses() {
        let mut learner = ClauseLearner::new(N);

        // Clause 1: (0,0,0) → (1,0,0) is in the SPP.
        // Clause 2: (0,0,0) → (1,0,0) is NOT in the SPP.
        let ap1 = vec![Some(false), Some(false), Some(false)];
        let ap2 = vec![Some(true), Some(false), Some(false)];
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: ap1.clone(),
                ap2: ap2.clone(),
                polarity: true,
            }],
        });
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1,
                ap2,
                polarity: false,
            }],
        });

        let mut store = SPPstore::new(N);
        let result = learner.extract(&mut store);
        assert!(
            result.is_err(),
            "conflicting clauses should yield InconsistentError"
        );
    }

    #[test]
    fn test_disjunctive_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ClauseLearner::new(N);

        // Clause: either (0,0,0)→(0,0,0) is in the SPP  OR  (1,1,1)→(1,1,1) is not in the SPP.
        // A positive clause with a negative literal that is easy to satisfy.
        let ap1_pos = vec![Some(false), Some(false), Some(false)];
        let ap2_pos = vec![Some(false), Some(false), Some(false)];
        let ap1_neg = vec![Some(true), Some(true), Some(true)];
        let ap2_neg = vec![Some(true), Some(true), Some(true)];
        learner.add_clause(AbstractClause {
            literals: vec![
                Literal {
                    ap1: ap1_pos,
                    ap2: ap2_pos,
                    polarity: true,
                },
                Literal {
                    ap1: ap1_neg,
                    ap2: ap2_neg,
                    polarity: false,
                },
            ],
        });
        let spp = learner.extract(&mut store).unwrap();

        // The zero ref packet should satisfy the clause: (0,0,0)→(0,0,0) must be accepted,
        // OR (1,1,1)→(1,1,1) must not be accepted (either is fine).
        let zero_ref = vec![false, false, false];
        assert!(clause_satisfied(
            &mut store,
            spp,
            &learner.clauses[0],
            &zero_ref
        ));
        let one_ref = vec![true, true, true];
        assert!(clause_satisfied(
            &mut store,
            spp,
            &learner.clauses[0],
            &one_ref
        ));
    }

    #[test]
    fn test_wildcard_correlated_clause() {
        let mut store = SPPstore::new(N);
        let mut learner = ClauseLearner::new(N);

        // Clause: (p, p) is in the SPP for all p (wildcard, correlated).
        // Expressed as one literal with all (None, None) fields.
        learner.add_clause(AbstractClause {
            literals: vec![Literal {
                ap1: vec![None, None, None],
                ap2: vec![None, None, None],
                polarity: true,
            }],
        });
        let spp = learner.extract(&mut store).unwrap();

        // All correlated pairs (p, p) must be accepted.
        for p in all_reference_packets(N) {
            assert!(
                spp_accepts(&store, spp, &p, &p),
                "SPP should accept ({:?}, {:?})",
                p,
                p
            );
        }
    }

    #[test]
    fn test_randomized() {
        const N_RAND: Var = 4;
        const ITERS: usize = 100;
        const CLAUSES_PER_ITER: usize = 5;

        for _ in 0..ITERS {
            let mut store = SPPstore::new(N_RAND);
            let mut learner = ClauseLearner::new(N_RAND);
            let n = N_RAND as usize;

            // Generate consistent concrete unit clauses (skip any that would conflict).
            // The `test_conflicting_unit_clauses` test covers the inconsistency path.
            let mut forced_in: HashSet<(Vec<bool>, Vec<bool>)> = HashSet::new();
            let mut forced_out: HashSet<(Vec<bool>, Vec<bool>)> = HashSet::new();

            for _ in 0..CLAUSES_PER_ITER {
                let p1: Vec<bool> = (0..n).map(|_| rand::random::<bool>()).collect();
                let p2: Vec<bool> = (0..n).map(|_| rand::random::<bool>()).collect();
                let polarity = rand::random::<bool>();

                let key = (p1.clone(), p2.clone());
                // Skip if this would create a conflict.
                if polarity && forced_out.contains(&key) {
                    continue;
                }
                if !polarity && forced_in.contains(&key) {
                    continue;
                }
                if polarity {
                    forced_in.insert(key);
                } else {
                    forced_out.insert(key);
                }

                let ap1: AbstractPacket = p1.iter().map(|&b| Some(b)).collect();
                let ap2: AbstractPacket = p2.iter().map(|&b| Some(b)).collect();
                learner.add_clause(AbstractClause {
                    literals: vec![Literal { ap1, ap2, polarity }],
                });
            }

            let spp = learner
                .extract(&mut store)
                .expect("consistent clauses should extract");

            // All abstract pairs are fully concrete (all Some), so any reference packet
            // instantiates them to the same concrete pair. Use all-zeros for simplicity.
            let zero_ref = vec![false; n];
            for clause in &learner.clauses {
                assert!(
                    clause_satisfied(&mut store, spp, clause, &zero_ref),
                    "clause not satisfied"
                );
            }
        }
    }

    /// Build a target SPP from random concrete clauses, then derive abstract unit clauses
    /// (with `None` wildcard fields) that the target unambiguously satisfies or rejects for
    /// every reference packet.  Verify the extracted SPP satisfies each clause for all
    /// reference packets.
    ///
    /// "Unambiguous" means `run_abstract(target, ap1, ap2)` is either `sp.one` (positive unit
    /// clause) or `sp.zero` (negative unit clause); mixed cases are skipped so the clauses are
    /// guaranteed to be consistent.
    #[test]
    fn test_randomized_wildcard_clauses() {
        const ITERS: usize = 100;
        const N_RAND: Var = 3; // small enough to check all 2^3 reference packets exhaustively

        for _ in 0..ITERS {
            let mut store = SPPstore::new(N_RAND);
            let n = N_RAND as usize;

            // Build a random target SPP from conflict-free concrete unit clauses.
            let mut seed = ClauseLearner::new(N_RAND);
            let mut forced_in: HashSet<(Vec<bool>, Vec<bool>)> = HashSet::new();
            let mut forced_out: HashSet<(Vec<bool>, Vec<bool>)> = HashSet::new();
            for _ in 0..12 {
                let p1 = rand_concrete_packet(n);
                let p2 = rand_concrete_packet(n);
                let pol = rand::random::<bool>();
                let key = (p1.clone(), p2.clone());
                if pol && forced_out.contains(&key) {
                    continue;
                }
                if !pol && forced_in.contains(&key) {
                    continue;
                }
                if pol {
                    forced_in.insert(key);
                } else {
                    forced_out.insert(key);
                }
                let ap1: AbstractPacket = p1.iter().map(|&b| Some(b)).collect();
                let ap2: AbstractPacket = p2.iter().map(|&b| Some(b)).collect();
                seed.add_clause(AbstractClause {
                    literals: vec![Literal {
                        ap1,
                        ap2,
                        polarity: pol,
                    }],
                });
            }
            let target = seed.extract(&mut store).expect("consistent seed");

            // Try to collect abstract unit clauses with wildcards consistent with target.
            let sp_one = store.sp.one;
            let mut learner = ClauseLearner::new(N_RAND);
            for _ in 0..20 {
                let ap1 = rand_abstract_packet(n);
                let ap2 = rand_abstract_packet(n);

                // run_abstract returns the SP of reference packets p where
                // (fill(ap1,p), fill(ap2,p)) ∈ target.
                let sp = run_abstract(&mut store, target, &ap1, &ap2);
                let all_in = sp == sp_one;
                let all_out = store.sp.is_zero(sp);

                if all_in {
                    learner.add_clause(AbstractClause {
                        literals: vec![Literal {
                            ap1,
                            ap2,
                            polarity: true,
                        }],
                    });
                } else if all_out {
                    learner.add_clause(AbstractClause {
                        literals: vec![Literal {
                            ap1,
                            ap2,
                            polarity: false,
                        }],
                    });
                }
                // Mixed polarity across reference packets: skip (would need disjunction).
            }

            let spp = learner
                .extract(&mut store)
                .expect("abstract clauses should be consistent");

            // Verify every clause is satisfied for every reference packet.
            let all_refs = all_reference_packets(N_RAND);
            for clause in &learner.clauses {
                for p_ref in &all_refs {
                    assert!(
                        clause_satisfied(&mut store, spp, clause, p_ref),
                        "clause not satisfied for p_ref={:?}",
                        p_ref
                    );
                }
            }
        }
    }
}
