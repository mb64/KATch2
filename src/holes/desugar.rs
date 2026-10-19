//! Compile packet-only subexpressions immediately, keeping shared definitions
//! compact instead of expanding them into boxed per-bit expression trees.
use ahash::{HashMap, HashMapExt as _};
use std::rc::Rc;

use super::{
    aut::{ExplicitDFA, aut_to_dfa},
    nk_with_holes::Expr as HExpr,
};
use crate::{
    aut::Aut,
    desugar::{DesugarEnv, DesugarError, desugar_with_env},
    expr::{Expr, Hole, Pattern},
    spp::{SPP, SPPstore},
};

#[derive(Clone)]
pub(super) struct Term {
    node: Rc<Node>,
    pub has_holes: bool,
    is_test: bool,
}

#[derive(Clone, Copy)]
enum Binary {
    Union,
    Intersect,
    Xor,
    Difference,
    Sequence,
    Until,
}
#[derive(Clone, Copy)]
enum Unary {
    Complement,
    Star,
    Next,
}
enum Node {
    Packet(SPP),
    Hole(Hole),
    Dup,
    Top,
    Binary(Binary, Term, Term),
    Unary(Unary, Term),
}

impl Term {
    fn packet(s: SPP, is_test: bool) -> Self {
        Self {
            node: Rc::new(Node::Packet(s)),
            has_holes: false,
            is_test,
        }
    }
    pub fn hole(h: Hole) -> Self {
        Self {
            node: Rc::new(Node::Hole(h)),
            has_holes: true,
            is_test: false,
        }
    }
    fn atom(node: Node) -> Self {
        Self {
            node: Rc::new(node),
            has_holes: false,
            is_test: false,
        }
    }
    fn spp(&self) -> Option<SPP> {
        match *self.node {
            Node::Packet(s) => Some(s),
            _ => None,
        }
    }
    fn binary(op: Binary, a: Self, b: Self, store: &mut SPPstore) -> Self {
        let is_test = a.is_test && b.is_test && !matches!(op, Binary::Until);
        if let (Some(a), Some(b)) = (a.spp(), b.spp()) {
            let s = match op {
                Binary::Union => Some(store.union(a, b)),
                Binary::Intersect => Some(store.intersect(a, b)),
                Binary::Xor => Some(store.xor(a, b)),
                Binary::Difference => Some(store.difference(a, b)),
                Binary::Sequence => Some(store.sequence(a, b)),
                Binary::Until => None,
            };
            if let Some(s) = s {
                return Self::packet(s, is_test);
            }
        }
        Self {
            has_holes: a.has_holes || b.has_holes,
            is_test,
            node: Rc::new(Node::Binary(op, a, b)),
        }
    }
    fn unary(op: Unary, e: Self, store: &mut SPPstore) -> Self {
        // Language complement also accepts longer traces: it is not SPP complement.
        if matches!(op, Unary::Star) {
            if let Some(s) = e.spp() {
                return Self::packet(store.star(s), false);
            }
        }
        Self {
            has_holes: e.has_holes,
            is_test: false,
            node: Rc::new(Node::Unary(op, e)),
        }
    }
    pub fn to_hole_expr(&self) -> HExpr {
        match &*self.node {
            Node::Packet(s) => HExpr::spp(*s),
            Node::Hole(h) => HExpr::hole(*h),
            Node::Dup => HExpr::dup(),
            Node::Binary(Binary::Union, a, b) => HExpr::union(a.to_hole_expr(), b.to_hole_expr()),
            Node::Binary(Binary::Sequence, a, b) => {
                HExpr::sequence(a.to_hole_expr(), b.to_hole_expr())
            }
            Node::Unary(Unary::Star, e) => HExpr::star(e.to_hole_expr()),
            _ => panic!("construct not supported by nk_with_holes::Expr"),
        }
    }
    pub fn to_dfa(&self, store: &mut SPPstore) -> ExplicitDFA {
        assert!(!self.has_holes);
        let mut aut = Aut::from_spp_store(std::mem::replace(store, SPPstore::new(0)));
        let start = self.to_state(&mut aut, &mut HashMap::new());
        let dfa = aut_to_dfa(&mut aut, start);
        std::mem::swap(store, aut.spp_store_mut());
        dfa
    }
    fn to_state(&self, aut: &mut Aut, memo: &mut HashMap<*const Node, usize>) -> usize {
        let key = Rc::as_ptr(&self.node);
        if let Some(&state) = memo.get(&key) {
            return state;
        }
        let state = match &*self.node {
            Node::Packet(s) => aut.mk_spp(*s),
            Node::Dup => aut.mk_dup(),
            Node::Top => aut.mk_top(),
            Node::Hole(_) => unreachable!("hole in concrete expression"),
            Node::Binary(op, a, b) => {
                let a = a.to_state(aut, memo);
                let b = b.to_state(aut, memo);
                match op {
                    Binary::Union => aut.mk_union(a, b),
                    Binary::Intersect => aut.mk_intersect(a, b),
                    Binary::Xor => aut.mk_xor(a, b),
                    Binary::Difference => aut.mk_difference(a, b),
                    Binary::Sequence => aut.mk_sequence(a, b),
                    Binary::Until => aut.mk_until(a, b),
                }
            }
            Node::Unary(op, e) => {
                let e = e.to_state(aut, memo);
                match op {
                    Unary::Complement => aut.mk_complement(e),
                    Unary::Star => aut.mk_star(e),
                    Unary::Next => aut.mk_next(e),
                }
            }
        };
        memo.insert(key, state);
        state
    }
}

#[derive(Clone, Default)]
pub(super) struct Env {
    bindings: HashMap<String, Term>,
    aliases: DesugarEnv,
}
impl Env {
    pub fn bind(&mut self, name: String, term: Term) {
        self.bindings.insert(name, term);
    }

    pub fn compile(&self, expr: &Expr, store: &mut SPPstore) -> Result<Term, DesugarError> {
        let binary = |op, a: &Expr, b: &Expr, store: &mut SPPstore| {
            Ok(Term::binary(
                op,
                self.compile(a, store)?,
                self.compile(b, store)?,
                store,
            ))
        };
        Ok(match expr {
            Expr::Zero => Term::packet(store.zero, true),
            Expr::One => Term::packet(store.one, true),
            Expr::Test(f, v) => Term::packet(store.test(*f, *v), true),
            Expr::Assign(f, v) => Term::packet(store.assign(*f, *v), false),
            Expr::Hole(h) => Term::hole(*h),
            Expr::Dup => Term::atom(Node::Dup),
            Expr::Top => Term::atom(Node::Top),
            Expr::End => Term::packet(store.top, false),
            Expr::BitRangeAssign(start, end, bits) => range(*start, *end, bits, false, store)?,
            Expr::BitRangeTest(start, end, bits) => range(*start, *end, bits, true, store)?,
            Expr::BitRangeMatch(start, end, Pattern::Exact(bits)) => {
                let width = (end - start) as usize;
                if bits.len() > width {
                    return Err(error(format!("Pattern has {} bits but field has only {} bits", bits.len(), width)));
                }
                // Preserve the existing pattern padding convention.
                let mut padded = vec![false; width - bits.len()];
                padded.extend_from_slice(bits);
                range(*start, *end, &padded, true, store)?
            }
            Expr::VarAssign(name, bits) | Expr::VarTest(name, bits) => {
                let Some((start, end)) = self.aliases.lookup_alias(name) else {
                    let operation = if matches!(expr, Expr::VarTest(..)) { "test" } else { "assignment" };
                    return Err(error(format!("Unknown alias '{}' in {}", name, operation)));
                };
                let width = (end - start) as usize;
                if bits.len() > width {
                    return Err(error(format!("Bit width mismatch: value has {} bits but alias '{}' expects {} bits", bits.len(), name, width)));
                }
                let mut padded = bits.clone(); padded.resize(width, false);
                range(start, end, &padded, matches!(expr, Expr::VarTest(..)), store)?
            }
            Expr::VarMatch(name, pattern) => {
                let Some((start, end)) = self.aliases.lookup_alias(name) else {
                    return Err(error(format!("Unknown alias '{}' in pattern match", name)));
                };
                self.compile(&Expr::BitRangeMatch(start, end, pattern.clone()), store)?
            }
            Expr::BitRangeMatch(..) => {
                // Complex patterns expand only a single bounded-width predicate;
                // compile it immediately rather than retaining its bit-level AST.
                let expanded = desugar_with_env(expr, &self.aliases)?;
                self.compile(&expanded, store)?
            }
            Expr::Union(..) => {
                // Generated policies contain hundreds of thousands of alternatives.
                // Flatten with an explicit worklist rather than retaining a native
                // stack frame for every left-associated union.
                let mut pending = vec![expr];
                // Merge equal-sized batches, avoiding an SPP for every growing
                // prefix of a huge union. Only O(log n) partial terms stay live.
                let mut batches: Vec<Option<Term>> = Vec::new();
                while let Some(e) = pending.pop() {
                    if let Expr::Union(a, b) = e {
                        pending.push(b);
                        pending.push(a);
                    } else {
                        let mut term = self.compile(e, store)?;
                        let mut level = 0;
                        loop {
                            if level == batches.len() { batches.push(None); }
                            if let Some(previous) = batches[level].take() {
                                term = Term::binary(Binary::Union, previous, term, store);
                                level += 1;
                            } else {
                                batches[level] = Some(term);
                                break;
                            }
                        }
                    }
                }
                batches.into_iter().rev().flatten().reduce(|a, b| {
                    Term::binary(Binary::Union, a, b, store)
                }).unwrap()
            },
            Expr::Intersect(a,b) => binary(Binary::Intersect, a,b,store)?,
            Expr::Xor(a,b) => binary(Binary::Xor, a,b,store)?,
            Expr::Difference(a,b) => binary(Binary::Difference, a,b,store)?,
            Expr::Sequence(a,b) => binary(Binary::Sequence, a,b,store)?,
            Expr::LtlUntil(a,b) => binary(Binary::Until, a,b,store)?,
            Expr::Complement(e) => Term::unary(Unary::Complement, self.compile(e, store)?, store),
            Expr::Star(e) => Term::unary(Unary::Star, self.compile(e, store)?, store),
            Expr::LtlNext(e) => Term::unary(Unary::Next, self.compile(e, store)?, store),
            Expr::TestNegation(e) => {
                let t = self.compile(e, store)?;
                if !t.is_test { return Err(error(format!("Test negation (!) can only be applied to test fragment expressions, but was applied to: {}", e))); }
                Term::packet(store.difference(store.one, t.spp().expect("test compiles to SPP")), true)
            }
            Expr::IfThenElse(c,t,e) => {
                let condition = c;
                let c = self.compile(c, store)?;
                let t = self.compile(t, store)?;
                let e = self.compile(e, store)?;
                if !c.is_test { return Err(error(format!("If-then-else condition must be a test fragment expression, but got: {}", condition))); }
                let nc = Term::packet(store.difference(store.one, c.spp().unwrap()), true);
                let yes = Term::binary(Binary::Sequence, c, t, store);
                let no = Term::binary(Binary::Sequence, nc, e, store);
                Term::binary(Binary::Union, yes, no, store)
            }
            Expr::Let(name, definition, body) => {
                let value = self.compile(definition, store)?;
                let mut scoped = self.clone(); scoped.bind(name.clone(), value);
                scoped.compile(body, store)?
            }
            Expr::LetBitRange(name, start, end, body) => {
                let mut scoped = self.clone(); scoped.aliases.add_alias(name.clone(), *start, *end);
                scoped.compile(body, store)?
            }
            Expr::Var(name) => self.bindings.get(name).cloned().ok_or_else(|| {
                if self.aliases.lookup_alias(name).is_some() {
                    error(format!("Bit range alias '{}' must be used with assignment (:=) or test (==) operations", name))
                } else { error(format!("Unbound variable: {}", name)) }
            })?,
        })
    }
}

fn error(message: impl Into<String>) -> DesugarError {
    DesugarError {
        message: message.into(),
    }
}

fn range(
    start: u32,
    end: u32,
    bits: &[bool],
    test: bool,
    store: &mut SPPstore,
) -> Result<Term, DesugarError> {
    if bits.len() != (end - start) as usize {
        return Err(error(format!(
            "Bit range [{}, {}) expects {} bits, but got {}",
            start,
            end,
            end - start,
            bits.len()
        )));
    }
    let mut result = store.one;
    for (i, &bit) in bits.iter().enumerate().rev() {
        let field = start + i as u32;
        let s = if test {
            store.test(field, bit)
        } else {
            store.assign(field, bit)
        };
        result = if test {
            store.intersect(s, result)
        } else {
            store.sequence(s, result)
        };
    }
    Ok(Term::packet(result, test || bits.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holes::{
        aut::{expr_to_dfa, is_empty, ops},
        parser::{Stmt, parse_program},
    };

    fn expression(source: &str) -> Box<Expr> {
        let program = parse_program(&format!("def value = {source}")).unwrap();
        let Stmt::Def(_, e) = program.statements.into_iter().next().unwrap() else {
            unreachable!()
        };
        e
    }

    #[test]
    fn eager_compilation_matches_legacy_language() {
        for source in [
            "x[0..3] := 5",
            "x[0..3] ~ 5",
            "x[0..3] ~ 1",
            "!(x0 == 1; x1 == 0)",
            "!(x0 == 1 ^ x1 == 0)",
            "!(x0 == 1 - x1 == 0)",
            "if x0 == 1 then x1 := 0 else x1 := 1",
            "(x0 := 1 + x1 := 0); (x0 := 0)*",
            "(x0 := 1; dup) + (dup; x1 := 0)",
            "~(x0 == 1)",
            "(dup + 1) & (dup + x0 == 1)",
            "(dup + 1) - (dup; dup)",
            "(dup + 1) ^ dup",
            "X (x0 == 1)",
            "(x0 == 1) U (x1 == 0)",
            "end",
            "T",
            "let p = x[0..3] := 5 in p; p",
            "let p = &x[0..3] in p := 2",
            "let p = &x[0..3] in p ~ 2",
        ] {
            let e = expression(source);
            let mut store = SPPstore::new(e.num_fields());
            let legacy = desugar_with_env(&e, &DesugarEnv::new()).unwrap();
            let expected = expr_to_dfa(&legacy, &mut store);
            let compiled = Env::default().compile(&e, &mut store).unwrap();
            let actual = compiled.to_dfa(&mut store);
            assert!(
                is_empty(
                    &ops::intersection(&actual, ops::complement(&expected)),
                    &mut store
                ),
                "extra behavior: {source}"
            );
            assert!(
                is_empty(
                    &ops::intersection(&expected, ops::complement(&actual)),
                    &mut store
                ),
                "missing behavior: {source}"
            );
        }
    }

    #[test]
    fn negated_tests_preserve_packets_and_compact_definitions_are_shared() {
        let mut store = SPPstore::new(3);
        let mut env = Env::default();
        let negated = env
            .compile(&expression("!(x0 == 1)"), &mut store)
            .unwrap()
            .spp()
            .unwrap();
        assert_eq!(negated, store.test(0, false));
        let policy = env
            .compile(&expression("(x[0..3] := 5 + x[0..3] := 2)*"), &mut store)
            .unwrap();
        assert!(policy.spp().is_some());
        env.bind("policy".to_string(), policy.clone());
        let reference = env
            .compile(&Expr::Var("policy".to_string()), &mut store)
            .unwrap();
        assert!(Rc::ptr_eq(&policy.node, &reference.node));
        env.bind("h".to_string(), Term::hole(Hole(0)));
        let mixed = env
            .compile(&expression("policy; h; policy"), &mut store)
            .unwrap();
        assert!(mixed.has_holes);
        let HExpr::Sequence(_, _) = mixed.to_hole_expr() else {
            panic!("hole must remain symbolic")
        };
    }

    #[test]
    fn aliases_patterns_and_validation_match_legacy() {
        use crate::expr::Pattern;
        let patterns = [
            Pattern::Exact(vec![true]),
            Pattern::Cidr {
                address: vec![false, true, false, true],
                prefix_len: 2,
            },
            Pattern::Wildcard {
                address: vec![false, true, false, true],
                mask: vec![true, false, false, true],
            },
            Pattern::IpRange {
                start: vec![false, true, false, false],
                end: vec![true, false, true, false],
            },
        ];
        for pattern in patterns {
            let e = Expr::BitRangeMatch(0, 4, pattern);
            let mut store = SPPstore::new(4);
            let expanded = desugar_with_env(&e, &DesugarEnv::new()).unwrap();
            let expected = Env::default()
                .compile(&expanded, &mut store)
                .unwrap()
                .spp()
                .unwrap();
            assert_eq!(
                Env::default().compile(&e, &mut store).unwrap().spp(),
                Some(expected)
            );
        }
        for source in [
            "!(x0 := 1)",
            "if x0 := 1 then 1 else 0",
            "let p = &x[0..2] in p := 7",
            "missing",
        ] {
            let e = expression(source);
            let old = desugar_with_env(&e, &DesugarEnv::new());
            let new = Env::default().compile(&e, &mut SPPstore::new(e.num_fields()));
            assert!(old.is_err() && new.is_err(), "{source}");
        }
    }
}
