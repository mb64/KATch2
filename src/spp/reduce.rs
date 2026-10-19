//! Reduce an SPP given a set of assumptions
//!
//! Input: a target SPP, and an SPP encoding the set of outputs we care about
//!
//! (Equivalently, it is the opposite of the don't-care set.)
//!
//! Output: a new SPP, hopefully smaller than input, with the goal that
//!
//! ```text
//! intersect(output, care) == intersect(target, care)
//! ```
//!
//! This is the BDD restrict algorithm (Coudert & Madre), adapted to SPPs.
//!
//! In a BDD, restrict simplifies a node by dropping it: when the care set only lives on one branch,
//! the node is replaced by that branch. SPPs never skip levels, so instead we pick the simplest
//! *shape* of node that is consistent with the care set. From simplest to most general:
//!
//!  1. Identity: `x00 == x11`, and `x01`, `x10` are zero (the bit is left alone, like `skip`).
//!  2. Assign 1: `x01 == x11`, and `x00`, `x10` are zero (like `x := 1`).
//!  3. Assign 0: `x00 == x10`, and `x01`, `x11` are zero (like `x := 0`).
//!  4. Havoc: all four children are equal (the bit is ignored and set to anything).
//!  5. Test: `x01`, `x10` are zero, but `x00` and `x11` may differ.
//!  6. General: each child is restricted on its own, and don't-care children become zero.
//!
//! Assignments come before havoc: both have a single child, but an assignment relates fewer packets
//! and is a smaller NetKAT term (`x := 1` rather than `x := 0 + x := 1`).
//!
//! Shapes 1 to 4 merge several children into one. The merged child must agree with each original
//! child on that child's care set, so it is the restriction of the union of the cared-about parts,
//! over the union of the care sets. This is feasible exactly when the children agree wherever their
//! care sets overlap. This merging is the SPP analogue of BDD restrict's "replace the node by one of
//! its children" step.

use crate::expr::{Exp, Expr};
use crate::spp::{self, SPP};
use ahash::{HashMap, HashMapExt as _, HashSetExt as _};

pub fn reduce(store: &mut spp::SPPstore, target: SPP, care: SPP) -> SPP {
    let mut reducer = Reducer {
        store,
        memo: HashMap::new(),
    };
    let depth = reducer.store.num_vars() as usize;
    reducer.reduce(target, care, depth)
}

struct Reducer<'a> {
    store: &'a mut spp::SPPstore,
    memo: HashMap<(SPP, SPP), SPP>,
}

impl Reducer<'_> {
    /// `target` and `care` are SPPs on `depth` fields.
    fn reduce(&mut self, target: SPP, care: SPP, depth: usize) -> SPP {
        if care == self.store.zero_at_depth(depth as spp::Var) {
            // Nothing is cared about, so the simplest answer will do.
            return self.store.zero_at_depth(depth as spp::Var);
        }
        if depth == 0 {
            // `care` is one, so we must keep `target` exactly.
            return target;
        }
        if let Some(&result) = self.memo.get(&(target, care)) {
            return result;
        }

        let t = self.store.get(target);
        let c = self.store.get(care);
        let ts = [t.x00, t.x01, t.x10, t.x11];
        let cs = [c.x00, c.x01, c.x10, c.x11];
        let zero = self.store.zero_at_depth(depth as spp::Var - 1);

        // Whether each child may be replaced by zero, i.e. it is zero everywhere we care about.
        let zero_children: [bool; 4] =
            [0, 1, 2, 3].map(|i| self.store.intersect(ts[i], cs[i]) == zero);
        let can_zero = |idxs: &[usize]| idxs.iter().all(|&i| zero_children[i]);

        let result = if let Some(r) = can_zero(&[1, 2])
            .then(|| self.merge(&ts, &cs, &[0, 3], depth - 1))
            .flatten()
        {
            // 1. Identity
            self.store.mk(r, zero, zero, r)
        } else if let Some(r) = can_zero(&[0, 2])
            .then(|| self.merge(&ts, &cs, &[1, 3], depth - 1))
            .flatten()
        {
            // 2. Assign 1
            self.store.mk(zero, r, zero, r)
        } else if let Some(r) = can_zero(&[1, 3])
            .then(|| self.merge(&ts, &cs, &[0, 2], depth - 1))
            .flatten()
        {
            // 3. Assign 0
            self.store.mk(r, zero, r, zero)
        } else if let Some(r) = self.merge(&ts, &cs, &[0, 1, 2, 3], depth - 1) {
            // 4. Havoc
            self.store.mk(r, r, r, r)
        } else if can_zero(&[1, 2]) {
            // 5. Test
            let r00 = self.reduce(ts[0], cs[0], depth - 1);
            let r11 = self.reduce(ts[3], cs[3], depth - 1);
            self.store.mk(r00, zero, zero, r11)
        } else {
            // 6. General
            let [r00, r01, r10, r11] = [0, 1, 2, 3].map(|i| {
                if zero_children[i] {
                    zero
                } else {
                    self.reduce(ts[i], cs[i], depth - 1)
                }
            });
            self.store.mk(r00, r01, r10, r11)
        };

        self.memo.insert((target, care), result);
        result
    }

    /// Try to find a single SPP that agrees with `ts[i]` on `cs[i]` for every `i` in `idxs`.
    ///
    /// Returns `None` if two of the children disagree somewhere both are cared about.
    fn merge(&mut self, ts: &[SPP; 4], cs: &[SPP; 4], idxs: &[usize], depth: usize) -> Option<SPP> {
        let mut merged_target = self.store.zero_at_depth(depth as spp::Var);
        let mut merged_care = self.store.zero_at_depth(depth as spp::Var);
        for &i in idxs {
            let cared = self.store.intersect(ts[i], cs[i]);
            merged_target = self.store.union(merged_target, cared);
            merged_care = self.store.union(merged_care, cs[i]);
        }
        // `merged_target` contains each child's cared-about part; check it adds nothing extra.
        for &i in idxs {
            let restricted = self.store.intersect(merged_target, cs[i]);
            let cared = self.store.intersect(ts[i], cs[i]);
            if restricted != cared {
                return None;
            }
        }
        Some(self.reduce(merged_target, merged_care, depth))
    }
}

// ---- Decision trees ----------------------------------------------------------------------------
//
// To turn an SPP into an expression, we case-split on one field at a time. The obvious order
// (field 0 first, then field 1, and so on) can be a bad one: an SPP that only depends on field 5 in
// a simple way can be complicated to describe field-by-field from the top. So at each step we
// choose which field to split on with a heuristic, as in decision tree learning.

/// How [`to_expr_tree`] chooses which field to split on next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitHeuristic {
    /// The lowest-numbered field: case-split on field 0, then field 1, and so on.
    FirstField,
    /// The field whose split leaves the fewest SPP nodes: the total node count of the distinct,
    /// nonzero sub-SPPs. Splits whose sub-SPPs coincide (an assignment, havoc, ...) are rewarded,
    /// since each distinct sub-SPP becomes a separate sub-expression.
    Size,
    /// The field with the most information gain: the sub-SPPs with the least total entropy, where
    /// the entropy of a sub-SPP is the binary entropy of the fraction of (input, output) pairs it
    /// relates. A sub-SPP that relates nothing, or everything, has zero entropy.
    Entropy,
    /// Try every field, build the whole expression for each, and keep the smallest. This finds the
    /// smallest expression [`to_expr_tree`] can produce with any split order, but is much slower
    /// than the other heuristics.
    Exhaustive,
    /// Look `k` splits ahead: try every field for the next `k` splits, finishing each branch with
    /// [`Entropy`](SplitHeuristic::Entropy), and keep the smallest resulting expression.
    ///
    /// `Lookahead(0)` is the same as [`Entropy`](SplitHeuristic::Entropy), and `Lookahead(k)` for
    /// `k` at least the number of fields is the same as [`Exhaustive`](SplitHeuristic::Exhaustive).
    Lookahead(u32),
    /// Look `depth` splits ahead, trying only the `width` fields with the
    /// lowest entropy at each search node. Finish with entropy when the depth
    /// budget runs out, and keep the smallest completed expression.
    ///
    /// `width` must be positive. Width 1 behaves like [`Entropy`](Self::Entropy);
    /// width at least the number of fields explores the same split candidates
    /// as [`Lookahead`](Self::Lookahead), with entropy-ranked tie breaking.
    LookaheadTop { depth: u32, width: usize },
}

/// Instantiate `field` of `spp` to `input -> output`: the sub-SPP of the (input, output) pairs where
/// `field` goes from `input` to `output`.
///
/// Every node for `field` is replaced by a do-nothing node, where `x00` and `x11` both point to the
/// node's `x_{input, output}` child, and `x01` and `x10` are zero. So the result leaves `field`
/// alone, and relates the other fields just like `spp` does when `field` goes from `input` to
/// `output`. This means `spp` is the union over `input`, `output` of
/// `field == input; field := output; instantiate(spp, field, input, output)`.
pub fn instantiate(
    store: &mut spp::SPPstore,
    spp: SPP,
    field: spp::Var,
    input: bool,
    output: bool,
) -> SPP {
    Instantiator::new().instantiate(store, spp, field, input, output)
}

/// [`instantiate`], with a cache shared across calls.
struct Instantiator {
    /// Keyed by (node, field, input, output). A node's level is determined by the node, since every
    /// path down an SPP has the same length.
    memo: HashMap<(SPP, spp::Var, bool, bool), SPP>,
}

impl Instantiator {
    fn new() -> Self {
        Instantiator {
            memo: HashMap::new(),
        }
    }

    fn instantiate(
        &mut self,
        store: &mut spp::SPPstore,
        spp: SPP,
        field: spp::Var,
        input: bool,
        output: bool,
    ) -> SPP {
        self.helper(store, spp, 0, field, input, output)
    }

    fn helper(
        &mut self,
        store: &mut spp::SPPstore,
        spp: SPP,
        level: spp::Var,
        field: spp::Var,
        input: bool,
        output: bool,
    ) -> SPP {
        let key = (spp, field, input, output);
        if let Some(&result) = self.memo.get(&key) {
            return result;
        }
        let node = store.get(spp);
        let result = if level == field {
            let child = match (input, output) {
                (false, false) => node.x00,
                (false, true) => node.x01,
                (true, false) => node.x10,
                (true, true) => node.x11,
            };
            // The zero SPP below `field`
            let zero = store.zero_at_depth(store.num_vars() - field - 1);
            store.mk(child, zero, zero, child)
        } else {
            let [x00, x01, x10, x11] = [node.x00, node.x01, node.x10, node.x11]
                .map(|child| self.helper(store, child, level + 1, field, input, output));
            store.mk(x00, x01, x10, x11)
        };
        self.memo.insert(key, result);
        result
    }
}

/// Convert `spp` to an equivalent expression, case-splitting on fields in the order chosen by
/// `heuristic`. May be exponentially larger than the SPP.
pub fn to_expr_tree(store: &mut spp::SPPstore, spp: SPP, heuristic: SplitHeuristic) -> Exp {
    if let SplitHeuristic::LookaheadTop { width, .. } = heuristic {
        assert!(width > 0, "lookahead width must be positive");
    }
    let n = store.num_vars();
    let mut builder = TreeBuilder {
        zero: store.zero,
        instantiator: Instantiator::new(),
        store,
        heuristic,
        memo: HashMap::new(),
        counts: HashMap::new(),
    };
    let budget = match heuristic {
        SplitHeuristic::Exhaustive => u32::MAX,
        SplitHeuristic::Lookahead(k) => k,
        SplitHeuristic::LookaheadTop { depth, .. } => depth,
        _ => 0,
    };
    builder.build(spp, (0..n).collect(), budget)
}

struct TreeBuilder<'a> {
    store: &'a mut spp::SPPstore,
    heuristic: SplitHeuristic,
    /// The zero SPP (on all fields)
    zero: SPP,
    instantiator: Instantiator,
    memo: HashMap<(SPP, Vec<spp::Var>, u32), Exp>,
    /// Memo for [`TreeBuilder::count`]
    counts: HashMap<SPP, f64>,
}

impl TreeBuilder<'_> {
    /// `spp` leaves every field not in `remaining` alone. For the lookahead heuristics, `budget` is
    /// how many more splits explore alternative fields; it is ignored otherwise.
    fn build(&mut self, spp: SPP, remaining: Vec<spp::Var>, budget: u32) -> Exp {
        if spp == self.zero {
            return Expr::zero();
        }
        // A budget of at least the number of fields left means "try everything" either way
        let budget = budget.min(remaining.len() as u32);
        let key = (spp, remaining, budget);
        if let Some(e) = self.memo.get(&key) {
            return e.clone();
        }
        let (spp, remaining, _) = key;

        // Split candidates, with their sub-SPPs. Fields that `spp` already leaves alone need no
        // split, so they are dropped.
        let mut candidates: Vec<(spp::Var, [SPP; 4])> = vec![];
        for &field in &remaining {
            let subs = [(false, false), (false, true), (true, false), (true, true)]
                .map(|(i, o)| self.instantiator.instantiate(self.store, spp, field, i, o));
            let [s00, s01, s10, s11] = subs;
            let leaves_alone = s00 == spp && s11 == spp && s01 == self.zero && s10 == self.zero;
            if !leaves_alone {
                candidates.push((field, subs));
            }
        }
        let remaining: Vec<spp::Var> = candidates.iter().map(|&(f, _)| f).collect();

        let rest_without = |field: spp::Var| -> Vec<spp::Var> {
            remaining.iter().copied().filter(|&f| f != field).collect()
        };
        let result = if candidates.is_empty() {
            // `spp` leaves every field alone, and isn't zero, so it's the identity
            Expr::one()
        } else if budget > 0 {
            // Rank all fields cheaply, then complete only the best few trees.
            // Partial selection is O(n); sorting the shortlist is O(width log
            // width). Field numbers break equal entropy scores deterministically.
            let trials = if let SplitHeuristic::LookaheadTop { width, .. } = self.heuristic {
                let mut ranked: Vec<_> = candidates
                    .iter()
                    .map(|&(field, subs)| (self.entropy_score(&subs), field, subs))
                    .collect();
                let compare = |a: &(f64, spp::Var, [SPP; 4]), b: &(f64, spp::Var, [SPP; 4])| {
                    a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
                };
                if ranked.len() > width {
                    ranked.select_nth_unstable_by(width - 1, compare);
                    ranked.truncate(width);
                }
                ranked.sort_unstable_by(compare);
                ranked
                    .into_iter()
                    .map(|(_, field, subs)| (field, subs))
                    .collect::<Vec<_>>()
            } else {
                candidates.clone()
            };
            trials
                .iter()
                .map(|&(field, subs)| {
                    self.split_expr(field, subs, &rest_without(field), budget - 1)
                })
                .min_by_key(|e| expr_size(e))
                .unwrap()
        } else {
            let (field, subs) = self.choose(&candidates);
            self.split_expr(field, subs, &rest_without(field), 0)
        };
        self.memo.insert((spp, remaining, budget), result.clone());
        result
    }

    /// Pick the candidate split the heuristic likes best (the first one, on ties). `candidates`
    /// must be nonempty.
    fn choose(&mut self, candidates: &[(spp::Var, [SPP; 4])]) -> (spp::Var, [SPP; 4]) {
        let mut best: Option<(f64, (spp::Var, [SPP; 4]))> = None;
        for &(field, subs) in candidates {
            let score = match self.heuristic {
                SplitHeuristic::FirstField => return (field, subs),
                SplitHeuristic::Size => self.size_score(&subs),
                // Out of lookahead: fall back to entropy
                SplitHeuristic::Entropy
                | SplitHeuristic::Exhaustive
                | SplitHeuristic::Lookahead(_)
                | SplitHeuristic::LookaheadTop { .. } => self.entropy_score(&subs),
            };
            if best.as_ref().is_none_or(|&(b, _)| score < b) {
                best = Some((score, (field, subs)));
            }
        }
        best.unwrap().1
    }

    fn size_score(&mut self, subs: &[SPP; 4]) -> f64 {
        let mut distinct: Vec<SPP> = subs.iter().copied().filter(|&s| s != self.zero).collect();
        distinct.sort();
        distinct.dedup();
        distinct.iter().map(|&s| self.num_nodes(s) as f64).sum()
    }

    fn entropy_score(&mut self, subs: &[SPP; 4]) -> f64 {
        // Every sub-SPP leaves the same fields alone, so they all live in the same universe; its
        // size cancels out of comparisons between candidates, but we still need a fraction
        let n = self.store.num_vars() as i32;
        let universe = 4f64.powi(n);
        subs.iter()
            .map(|&s| {
                let p = self.count(s) / universe;
                binary_entropy(p)
            })
            .sum()
    }

    /// Number of distinct nodes in `spp`.
    fn num_nodes(&self, spp: SPP) -> usize {
        let mut seen = ahash::HashSet::new();
        let mut stack = vec![spp];
        while let Some(x) = stack.pop() {
            if x.as_u32() <= 1 || !seen.insert(x) {
                continue;
            }
            let node = self.store.get(x);
            stack.extend([node.x00, node.x01, node.x10, node.x11]);
        }
        seen.len()
    }

    /// Number of (input, output) pairs `spp` relates.
    fn count(&mut self, spp: SPP) -> f64 {
        if spp.as_u32() <= 1 {
            return spp.as_u32() as f64;
        }
        if let Some(&c) = self.counts.get(&spp) {
            return c;
        }
        let node = self.store.get(spp);
        let c = [node.x00, node.x01, node.x10, node.x11]
            .iter()
            .map(|&x| self.count(x))
            .sum();
        self.counts.insert(spp, c);
        c
    }

    /// The expression for splitting on `field`, given its sub-SPPs `[s00, s01, s10, s11]`, recursing
    /// on the sub-SPPs with fields `rest` left to split.
    ///
    /// This picks out the special cases havoc, test, and assignment, before the general case.
    fn split_expr(
        &mut self,
        field: spp::Var,
        subs: [SPP; 4],
        rest: &[spp::Var],
        budget: u32,
    ) -> Exp {
        let [s00, s01, s10, s11] = subs;
        let [z00, z01, z10, z11] = subs.map(|s| s == self.zero);

        // `field` is set to anything
        if s00 == s01 && s00 == s10 && s00 == s11 {
            let havoc = Expr::union(Expr::assign(field, false), Expr::assign(field, true));
            let rest_expr = self.build(s00, rest.to_vec(), budget);
            return Expr::sequence_simp(havoc, rest_expr);
        }
        // `field` is tested
        if z01 && z10 {
            return self.branch(field, s00, s11, rest, budget);
        }
        // `field` is assigned: every output has the same value
        for (value, zero_other, if_false, if_true) in
            [(true, z00 && z10, s01, s11), (false, z01 && z11, s00, s10)]
        {
            if !zero_other {
                continue;
            }
            let assign = Expr::assign(field, value);
            return if if_false == if_true {
                let rest_expr = self.build(if_false, rest.to_vec(), budget);
                Expr::sequence_simp(assign, rest_expr)
            } else {
                // Which fields come next depends on the input value of `field`
                let branch = self.branch(field, if_false, if_true, rest, budget);
                Expr::sequence(branch, assign)
            };
        }

        // General case: a sum over the (input, output) values of `field`
        let mut result = Expr::zero();
        for (input, output, sub) in [
            (false, false, s00),
            (false, true, s01),
            (true, false, s10),
            (true, true, s11),
        ] {
            let rest_expr = self.build(sub, rest.to_vec(), budget);
            if *rest_expr == Expr::Zero {
                continue;
            }
            let mut prefix = Expr::test(field, input);
            if input != output {
                prefix = Expr::sequence(prefix, Expr::assign(field, output));
            }
            result = Expr::union_simp(result, Expr::sequence_simp(prefix, rest_expr));
        }
        result
    }

    /// `(field == 0; if_false) + (field == 1; if_true)`.
    fn branch(
        &mut self,
        field: spp::Var,
        if_false: SPP,
        if_true: SPP,
        rest: &[spp::Var],
        budget: u32,
    ) -> Exp {
        let mut result = Expr::zero();
        for (value, sub) in [(false, if_false), (true, if_true)] {
            let rest_expr = self.build(sub, rest.to_vec(), budget);
            if *rest_expr != Expr::Zero {
                result = Expr::union_simp(
                    result,
                    Expr::sequence_simp(Expr::test(field, value), rest_expr),
                );
            }
        }
        result
    }
}

/// Number of nodes in the syntax tree of `e`.
fn expr_size(e: &Expr) -> usize {
    match e {
        Expr::Union(a, b) | Expr::Sequence(a, b) => 1 + expr_size(a) + expr_size(b),
        Expr::Star(a) => 1 + expr_size(a),
        _ => 1,
    }
}

/// `-p log p - (1 - p) log (1 - p)`, which is zero at `p = 0` and `p = 1`.
fn binary_entropy(p: f64) -> f64 {
    let term = |x: f64| if x <= 0.0 { 0.0 } else { -x * x.log2() };
    term(p) + term(1.0 - p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(store: &mut spp::SPPstore, target: SPP, care: SPP) -> SPP {
        let result = reduce(store, target, care);
        let lhs = store.intersect(result, care);
        let rhs = store.intersect(target, care);
        assert_eq!(
            lhs, rhs,
            "reduce({target}, {care}) = {result} disagrees on the care set"
        );
        result
    }

    const HEURISTICS: [SplitHeuristic; 9] = [
        SplitHeuristic::FirstField,
        SplitHeuristic::Size,
        SplitHeuristic::Entropy,
        SplitHeuristic::Lookahead(1),
        SplitHeuristic::Lookahead(2),
        SplitHeuristic::Lookahead(3),
        SplitHeuristic::LookaheadTop { depth: 1, width: 5 },
        SplitHeuristic::LookaheadTop { depth: 3, width: 1 },
        SplitHeuristic::Exhaustive,
    ];

    #[test]
    fn instantiate_splits_the_relation() {
        // `spp` is the union over (input, output) of `field == input; field := output; sub`
        let mut store = spp::SPPstore::new(3);
        for _ in 0..200 {
            let spp = store.rand();
            for field in 0..3 {
                let mut union = store.zero;
                for (input, output) in [(false, false), (false, true), (true, false), (true, true)]
                {
                    let sub = instantiate(&mut store, spp, field, input, output);
                    let test = store.test(field, input);
                    let assign = store.assign(field, output);
                    let prefix = store.sequence(test, assign);
                    let part = store.sequence(prefix, sub);
                    union = store.union(union, part);
                }
                assert_eq!(union, spp);
            }
        }
    }

    #[test]
    fn to_expr_tree_roundtrip() {
        for n in [2, 4] {
            let mut aut = crate::aut::Aut::new(n);
            for _ in 0..200 {
                let spp = aut.spp_store_mut().rand();
                for heuristic in HEURISTICS {
                    let expr = to_expr_tree(aut.spp_store_mut(), spp, heuristic);
                    let state = aut.expr_to_state(&expr);
                    assert_eq!(aut.epsilon(state), spp, "{heuristic:?}: {expr}");
                }
            }
        }
    }

    #[test]
    fn lookahead_endpoints() {
        let mut store = spp::SPPstore::new(4);
        for _ in 0..100 {
            let spp = store.rand();
            let entropy = to_expr_tree(&mut store, spp, SplitHeuristic::Entropy);
            let zero = to_expr_tree(&mut store, spp, SplitHeuristic::Lookahead(0));
            assert_eq!(entropy, zero);
            for depth in [0, 1, 3] {
                let narrow = to_expr_tree(
                    &mut store,
                    spp,
                    SplitHeuristic::LookaheadTop { depth, width: 1 },
                );
                assert_eq!(entropy, narrow);
            }
            let exhaustive = to_expr_tree(&mut store, spp, SplitHeuristic::Exhaustive);
            for k in [4, 5, 100] {
                let lookahead = to_expr_tree(&mut store, spp, SplitHeuristic::Lookahead(k));
                assert_eq!(exhaustive, lookahead, "Lookahead({k})");
            }
            let unpruned = to_expr_tree(
                &mut store,
                spp,
                SplitHeuristic::LookaheadTop { depth: 4, width: 5 },
            );
            assert_eq!(expr_size(&exhaustive), expr_size(&unpruned));
            // More lookahead never hurts
            let sizes: Vec<usize> = (0..=4)
                .map(|k| expr_size(&to_expr_tree(&mut store, spp, SplitHeuristic::Lookahead(k))))
                .collect();
            assert!(sizes.windows(2).all(|w| w[1] <= w[0]), "{sizes:?}");
        }
    }

    #[test]
    fn entropy_shortlist_preserves_relation_when_fields_are_pruned() {
        let mut aut = crate::aut::Aut::new(6);
        for _ in 0..30 {
            let spp = aut.spp_store_mut().rand();
            for width in [1, 2, 5] {
                let expr = to_expr_tree(
                    aut.spp_store_mut(),
                    spp,
                    SplitHeuristic::LookaheadTop { depth: 1, width },
                );
                let state = aut.expr_to_state(&expr);
                assert_eq!(aut.epsilon(state), spp, "width={width}: {expr}");
            }
        }
    }

    #[test]
    fn size_heuristic_finds_the_relevant_field() {
        // Only field 3 does anything interesting; the others are left alone
        let mut store = spp::SPPstore::new(4);
        let test = store.test(3, true);
        let assign = store.assign(3, false);
        let spp = store.sequence(test, assign);
        let expr = to_expr_tree(&mut store, spp, SplitHeuristic::Size);
        assert_eq!(crate::printer::pretty(&expr), "x3 == 1; x3 := 0");
    }

    /// Compare expression sizes from each [`SplitHeuristic`] against [`SplitHeuristic::FirstField`].
    ///
    /// Not a real test, just prints statistics. Run with
    /// `cargo test --release --lib split_heuristics_compared -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn split_heuristics_compared() {
        use crate::holes::{aut as haut, minimize};

        // Corpus 1: random SPPs
        let mut random_store = spp::SPPstore::new(5);
        let random: Vec<SPP> = (0..300).map(|_| random_store.rand()).collect();

        // Corpus 2: SPPs of random expressions
        crate::fuzz::seed_fuzzer(0x5EED_0040);
        let mut aut = crate::aut::Aut::new(6);
        let mut compiled = vec![];
        while compiled.len() < 300 {
            let (expr, _) = crate::fuzz::genax(0, 5, 6);
            let state = aut.expr_to_state(&expr);
            let spp = aut.epsilon(state);
            if !aut.spp_store_mut().is_zero(spp) {
                compiled.push(spp);
            }
        }

        // Corpus 3: edges and outputs of minimized automata
        crate::fuzz::seed_fuzzer(0x5EED_0041);
        let mut edge_store = spp::SPPstore::new(5);
        let mut edges = vec![];
        for _ in 0..200 {
            let (expr, _) = crate::fuzz::genax(0, 4, 5);
            let dfa = haut::expr_to_dfa(&expr, &mut edge_store);
            let nfa = minimize::minimize(&dfa, &mut edge_store);
            for q in 0..nfa.num_states() {
                edges.extend(nfa.transitions[q].iter().map(|&(spp, _)| spp));
                edges.push(nfa.outputs[q]);
            }
        }
        edges.retain(|&spp| !edge_store.is_zero(spp));
        edges.sort();
        edges.dedup();

        let corpora: [(&str, &mut spp::SPPstore, Vec<SPP>); 3] = [
            ("random SPPs, 5 fields", &mut random_store, random),
            (
                "compiled expressions, 6 fields",
                aut.spp_store_mut(),
                compiled,
            ),
            (
                "minimized automaton edges, 5 fields",
                &mut edge_store,
                edges,
            ),
        ];
        for (name, store, spps) in corpora {
            println!("{name} ({} SPPs):", spps.len());
            let baseline: Vec<usize> = spps
                .iter()
                .map(|&spp| expr_size(&to_expr_tree(store, spp, SplitHeuristic::FirstField)))
                .collect();
            let heuristics = [
                SplitHeuristic::FirstField,
                SplitHeuristic::Size,
                SplitHeuristic::Entropy,
                SplitHeuristic::Lookahead(1),
                SplitHeuristic::LookaheadTop { depth: 1, width: 2 },
                SplitHeuristic::LookaheadTop { depth: 1, width: 3 },
                SplitHeuristic::LookaheadTop { depth: 1, width: 5 },
                SplitHeuristic::Lookahead(2),
                SplitHeuristic::Lookahead(3),
                SplitHeuristic::Lookahead(4),
                SplitHeuristic::Lookahead(5),
                SplitHeuristic::Exhaustive,
            ];
            for heuristic in heuristics {
                let start = std::time::Instant::now();
                let sizes: Vec<usize> = spps
                    .iter()
                    .map(|&spp| expr_size(&to_expr_tree(store, spp, heuristic)))
                    .collect();
                let elapsed = start.elapsed();
                let total: usize = sizes.iter().sum();
                let better = sizes.iter().zip(&baseline).filter(|(s, b)| s < b).count();
                let worse = sizes.iter().zip(&baseline).filter(|(s, b)| s > b).count();
                let log_ratio: f64 = sizes
                    .iter()
                    .zip(&baseline)
                    .map(|(&s, &b)| (s as f64 / b as f64).ln())
                    .sum();
                println!(
                    "  {:<13} total size {total:>8}, better in {better:>3}, worse in {worse:>3}, \
                     geometric mean ratio {:.3}, {:.2?}",
                    format!("{heuristic:?}"),
                    (log_ratio / spps.len() as f64).exp(),
                    elapsed
                );
            }
        }
    }

    #[test]
    fn exhaustive_one_field() {
        let mut store = spp::SPPstore::new(1);
        let all = store.all();
        for &target in &all {
            for &care in &all {
                check(&mut store, target, care);
            }
        }
    }

    #[test]
    fn random_agrees_on_care_set() {
        for n in 2..=4 {
            let mut store = spp::SPPstore::new(n);
            for _ in 0..500 {
                let target = store.rand();
                let care = store.rand();
                check(&mut store, target, care);
            }
        }
    }

    #[test]
    fn simple_cases() {
        let mut store = spp::SPPstore::new(3);
        let (zero, one, top) = (store.zero, store.one, store.top);
        let t = store.test(1, true);

        // Caring about everything changes nothing
        assert_eq!(check(&mut store, t, top), t);
        // Caring about nothing gives zero
        assert_eq!(check(&mut store, t, zero), zero);
        // Caring only about the identity part of top gives identity
        assert_eq!(check(&mut store, top, one), one);
        assert_eq!(check(&mut store, one, one), one);
        // A test, if we only care where it passes, is the identity
        assert_eq!(check(&mut store, t, t), one);
        // An assignment stays an assignment, even if we only care about some inputs
        let assign = store.assign(1, true);
        assert_eq!(check(&mut store, assign, top), assign);
        let t0 = store.test(0, false);
        let care = store.sequence(t0, top);
        assert_eq!(check(&mut store, assign, care), assign);
    }
}
