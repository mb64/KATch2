//! Parser for `.nksynth` synthesis files.
//!
//! A `.nksynth` file is a sequence of statements, each one of:
//!
//! ```text
//! hole X                  // declare a synthesis hole named `X`
//! def A = (some expr)     // name a sub-expression
//! assert (expr) <= (expr) // language-containment goal
//! assert (expr) == (expr) // language-equivalence goal
//! ```
//!
//! Expressions are parsed by the ordinary NetKAT expression parser
//! ([`crate::parser::Parser::parse_single_expression`]); this module only
//! adds the statement-level grammar around it, reusing the same lexer.
//!
//! Holes and definitions are referenced inside expressions by name, where they
//! appear as `Expr::Var(name)`.  Resolving those names — turning declared holes
//! into [`crate::expr::Expr::Hole`] and inlining definitions — is a separate
//! pass and not the parser's job.

use crate::desugar::{DesugarEnv, DesugarError, desugar_with_env};
use crate::expr::{Exp, Expr, Hole};
use crate::holes::aut::expr_to_dfa;
use crate::holes::cegis::Constraint;
use crate::holes::nk_with_holes::Expr as HExpr;
use crate::parser::{Lexer, ParseError, Parser, Span, TokenKind};
use crate::spp;

/// A fully-parsed `.nksynth` file: an ordered list of statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

/// One top-level statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `hole NAME` — declares a synthesis hole.
    Hole(String),
    /// `def NAME = expr` — names a sub-expression.
    Def(String, Exp),
    /// `assert lhs <= rhs` or `assert lhs == rhs`.
    Assert { lhs: Exp, op: AssertOp, rhs: Exp },
}

/// The only two comparison operators allowed in an assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertOp {
    /// `<=` (language containment / refinement)
    Leq,
    /// `==` (language equivalence)
    Eq,
}

/// Parse a whole `.nksynth` source string into a [`Program`].
pub fn parse_program(input: &str) -> Result<Program, ParseError> {
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let mut statements = Vec::new();

    loop {
        let tok = parser.peek_cloned()?;
        match tok.kind {
            TokenKind::Eof => break,
            TokenKind::Hole => {
                parser.advance()?; // consume `hole`
                let name = expect_ident(&mut parser, "hole")?;
                statements.push(Stmt::Hole(name));
            }
            TokenKind::Def => {
                parser.advance()?; // consume `def`
                let name = expect_ident(&mut parser, "def")?;
                expect_eq(&mut parser)?; // consume `=`
                let expr = parser.parse_single_expression()?;
                statements.push(Stmt::Def(name, expr));
            }
            // `assert` is not a lexer keyword, so it arrives as a plain ident.
            TokenKind::Ident(ref s) if s == "assert" => {
                parser.advance()?; // consume `assert`
                let lhs = parser.parse_single_expression()?;
                let op = expect_assert_op(&mut parser)?;
                let rhs = parser.parse_single_expression()?;
                statements.push(Stmt::Assert { lhs, op, rhs });
            }
            _ => {
                return Err(err(
                    format!(
                        "Expected `hole`, `def`, or `assert` at start of statement, found {:?}",
                        tok.kind
                    ),
                    tok.span,
                ));
            }
        }
    }

    Ok(Program { statements })
}

/// Consume an identifier token, returning its name. Hole/def names must be
/// plain identifiers (reserved single letters like `X` and `xN` field tokens
/// are rejected).
fn expect_ident(parser: &mut Parser, context: &str) -> Result<String, ParseError> {
    let tok = parser.advance()?;
    match tok.kind {
        TokenKind::Ident(name) => Ok(name),
        other => Err(err(
            format!("Expected an identifier name after `{context}`, found {other:?}"),
            tok.span,
        )),
    }
}

/// Consume the `=` token of a `def`. The lexer maps both `=` and `==` to
/// `TokenKind::Eq`.
fn expect_eq(parser: &mut Parser) -> Result<(), ParseError> {
    let tok = parser.advance()?;
    match tok.kind {
        TokenKind::Eq => Ok(()),
        other => Err(err(
            format!("Expected `=` after definition name, found {other:?}"),
            tok.span,
        )),
    }
}

/// Consume the comparison operator of an `assert` (`<=` or `==`).
fn expect_assert_op(parser: &mut Parser) -> Result<AssertOp, ParseError> {
    let tok = parser.advance()?;
    match tok.kind {
        TokenKind::Lte => Ok(AssertOp::Leq),
        TokenKind::Eq => Ok(AssertOp::Eq),
        other => Err(err(
            format!("Expected `<=` or `==` in assertion, found {other:?}"),
            tok.span,
        )),
    }
}

fn err(message: String, span: Span) -> ParseError {
    ParseError { message, span }
}

// --- Desugaring a Program into a synthesis problem ----------------------------

/// A `.nksynth` program lowered into a concrete synthesis problem: an SPP store
/// sized to the program's fields, the holes to synthesize, and the constraints
/// the holes must jointly satisfy.
pub struct ProblemInstance {
    pub store: spp::SPPstore,
    pub holes: Vec<Hole>,
    pub constraints: Vec<Constraint>,
}

/// The number of packet fields the program references: the max over the
/// `num_fields` of every expression it mentions (definitions and both sides of
/// every assertion).  This sizes the SPP store.
pub fn num_fields(program: &Program) -> u32 {
    let mut max = 0;
    for stmt in &program.statements {
        match stmt {
            Stmt::Hole(_) => {}
            Stmt::Def(_, expr) => max = max.max(expr.num_fields()),
            Stmt::Assert { lhs, rhs, .. } => {
                max = max.max(lhs.num_fields()).max(rhs.num_fields());
            }
        }
    }
    max
}

/// Whether a desugared expression mentions any hole.
pub fn has_holes(expr: &Expr) -> bool {
    match expr {
        Expr::Hole(_) => true,
        Expr::Zero
        | Expr::One
        | Expr::Top
        | Expr::Dup
        | Expr::End
        | Expr::Assign(_, _)
        | Expr::Test(_, _)
        | Expr::VarAssign(_, _)
        | Expr::VarTest(_, _)
        | Expr::BitRangeAssign(_, _, _)
        | Expr::BitRangeTest(_, _, _)
        | Expr::BitRangeMatch(_, _, _)
        | Expr::VarMatch(_, _)
        | Expr::Var(_) => false,
        Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::Xor(a, b)
        | Expr::Difference(a, b)
        | Expr::Sequence(a, b)
        | Expr::LtlUntil(a, b) => has_holes(a) || has_holes(b),
        Expr::Complement(e) | Expr::TestNegation(e) | Expr::Star(e) | Expr::LtlNext(e) => {
            has_holes(e)
        }
        Expr::IfThenElse(c, t, e) => has_holes(c) || has_holes(t) || has_holes(e),
        Expr::Let(_, def, body) => has_holes(def) || has_holes(body),
        Expr::LetBitRange(_, _, _, body) => has_holes(body),
    }
}

/// Convert a desugared [`Expr`] into a [`nk_with_holes::Expr`](HExpr),
/// compiling predicate/assignment leaves into SPPs.
///
/// Panics if the expression uses a construct `nk_with_holes::Expr` cannot
/// represent: intersection, xor, difference, complement, the LTL operators, or
/// any sugar that should have been desugared away.
pub fn to_hole_expr(expr: &Expr, store: &mut spp::SPPstore) -> HExpr {
    match expr {
        Expr::Zero => HExpr::spp(store.zero),
        Expr::One => HExpr::spp(store.one),
        Expr::Top => HExpr::spp(store.top),
        Expr::Assign(f, v) => {
            let s = store.assign(*f, *v);
            HExpr::spp(s)
        }
        Expr::Test(f, v) => {
            let s = store.test(*f, *v);
            HExpr::spp(s)
        }
        Expr::Dup => HExpr::dup(),
        Expr::Hole(h) => HExpr::hole(*h),
        Expr::Union(a, b) => HExpr::union(to_hole_expr(a, store), to_hole_expr(b, store)),
        Expr::Sequence(a, b) => HExpr::sequence(to_hole_expr(a, store), to_hole_expr(b, store)),
        Expr::Star(e) => HExpr::star(to_hole_expr(e, store)),
        other => panic!("to_hole_expr: construct not supported by nk_with_holes::Expr: {other:?}"),
    }
}

/// Lower a parsed [`Program`] into a [`ProblemInstance`].
///
/// Maintains a [`DesugarEnv`]: each `hole NAME` binds `NAME` to a fresh
/// [`Expr::Hole`] (so expressions referencing it pick up the hole during
/// desugaring), and each `def NAME = expr` binds `NAME` to its desugared body.
/// For every assertion, [`has_holes`] decides the constraint shape:
///
/// * `lhs <= rhs` with holes on the left  → [`Constraint::upper_bound`] (`lhs[holes] ⊆ rhs`)
/// * `lhs <= rhs` with holes on the right → [`Constraint::lower_bound`] (`lhs ⊆ rhs[holes]`)
/// * `lhs == rhs`                          → [`Constraint::equality`]
///
/// The hole-free side is compiled into the reference DFA; the hole-bearing side
/// into the hole automaton. If neither side has holes, the left side plays the
/// hole-automaton role (a pure verification constraint).
pub fn desugar(program: &Program) -> Result<ProblemInstance, DesugarError> {
    let n = num_fields(program);
    let mut store = spp::SPPstore::new(n);
    let mut env = DesugarEnv::new();
    let mut holes: Vec<Hole> = Vec::new();
    let mut constraints: Vec<Constraint> = Vec::new();

    for stmt in &program.statements {
        match stmt {
            Stmt::Hole(name) => {
                let h = Hole(holes.len() as u32);
                holes.push(h);
                env.add_variable(name.clone(), Expr::hole(h));
            }
            Stmt::Def(name, expr) => {
                let desugared = desugar_with_env(expr, &env)?;
                env.add_variable(name.clone(), desugared);
            }
            Stmt::Assert { lhs, op, rhs } => {
                let dl = desugar_with_env(lhs, &env)?;
                let dr = desugar_with_env(rhs, &env)?;
                let constraint = build_constraint(&mut store, &dl, *op, &dr)?;
                constraints.push(constraint);
            }
        }
    }

    Ok(ProblemInstance {
        store,
        holes,
        constraints,
    })
}

/// Build the constraint for a single (desugared) assertion. The hole-bearing
/// side becomes the hole automaton; the hole-free side becomes the reference
/// DFA. Errors if both sides contain holes.
fn build_constraint(
    store: &mut spp::SPPstore,
    lhs: &Expr,
    op: AssertOp,
    rhs: &Expr,
) -> Result<Constraint, DesugarError> {
    let lhs_holes = has_holes(lhs);
    let rhs_holes = has_holes(rhs);
    if lhs_holes && rhs_holes {
        return Err(DesugarError {
            message:
                "both sides of an assertion contain holes; at least one side must be hole-free"
                    .to_string(),
        });
    }

    let constraint = match op {
        AssertOp::Eq => {
            // Equality is symmetric: put the hole-bearing side (or `lhs` when
            // neither has holes) on the automaton side.
            let (hole_side, dfa_side) = if rhs_holes { (rhs, lhs) } else { (lhs, rhs) };
            let dfa = expr_to_dfa(dfa_side, store);
            let hole_expr = to_hole_expr(hole_side, store);
            Constraint::equality(store, &hole_expr, dfa)
        }
        AssertOp::Leq if rhs_holes => {
            // lhs ⊆ rhs[holes]: the DFA is a lower bound on the automaton.
            let dfa = expr_to_dfa(lhs, store);
            let hole_expr = to_hole_expr(rhs, store);
            Constraint::lower_bound(store, &hole_expr, dfa)
        }
        AssertOp::Leq => {
            // lhs[holes] ⊆ rhs: the DFA is an upper bound on the automaton.
            let dfa = expr_to_dfa(rhs, store);
            let hole_expr = to_hole_expr(lhs, store);
            Constraint::upper_bound(store, &hole_expr, dfa)
        }
    };

    Ok(constraint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    #[test]
    fn parses_holes_defs_and_asserts() {
        let src = "\
            hole foo\n\
            hole bar\n\
            def a = x0 := 1\n\
            def b = x0 == 1\n\
            assert a <= b\n\
            assert a == b\n";
        let prog = parse_program(src).expect("should parse");
        assert_eq!(prog.statements.len(), 6);

        assert_eq!(prog.statements[0], Stmt::Hole("foo".to_string()));
        assert_eq!(prog.statements[1], Stmt::Hole("bar".to_string()));

        match &prog.statements[2] {
            Stmt::Def(name, expr) => {
                assert_eq!(name, "a");
                assert_eq!(**expr, Expr::Assign(0, true));
            }
            other => panic!("expected Def, got {other:?}"),
        }

        match &prog.statements[4] {
            Stmt::Assert { lhs, op, rhs } => {
                assert_eq!(*op, AssertOp::Leq);
                assert_eq!(**lhs, Expr::Var("a".to_string()));
                assert_eq!(**rhs, Expr::Var("b".to_string()));
            }
            other => panic!("expected Assert, got {other:?}"),
        }

        match &prog.statements[5] {
            Stmt::Assert { op, .. } => assert_eq!(*op, AssertOp::Eq),
            other => panic!("expected Assert, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_is_empty_program() {
        let prog = parse_program("   \n  // just a comment\n").expect("should parse");
        assert!(prog.statements.is_empty());
    }

    #[test]
    fn assert_with_compound_expressions() {
        let src = "assert (x0 := 1) ; dup <= dup ; (x0 := 1)\n";
        let prog = parse_program(src).expect("should parse");
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0] {
            Stmt::Assert { lhs, op, rhs } => {
                assert_eq!(*op, AssertOp::Leq);
                // lhs = (x0 := 1) ; dup
                assert!(matches!(**lhs, Expr::Sequence(_, _)));
                // rhs = dup ; (x0 := 1)
                assert!(matches!(**rhs, Expr::Sequence(_, _)));
            }
            other => panic!("expected Assert, got {other:?}"),
        }
    }

    #[test]
    fn rejects_reserved_letter_as_hole_name() {
        // `X` lexes as the LTL-next operator, not an identifier.
        let result = parse_program("hole X\n");
        assert!(result.is_err());
    }

    // ---- desugaring tests ----------------------------------------------

    #[test]
    fn num_fields_takes_max_over_all_expressions() {
        let prog = parse_program(
            "def a = x0 := 1\n\
             def b = x3 := 1\n\
             assert (x1 == 1) <= (x2 == 1)\n",
        )
        .unwrap();
        // x3 := 1 mentions field 3, so 4 fields total.
        assert_eq!(num_fields(&prog), 4);
    }

    #[test]
    fn has_holes_detects_nested_holes() {
        assert!(has_holes(&Expr::Hole(Hole(0))));
        assert!(has_holes(
            &Expr::Sequence(Expr::dup(), Expr::hole(Hole(1)),)
        ));
        assert!(!has_holes(&Expr::Sequence(Expr::dup(), Expr::one())));
    }

    #[test]
    fn desugar_equality_constraint() {
        let prog = parse_program(
            "hole h\n\
             def target = x0 := 1\n\
             assert h == target\n",
        )
        .unwrap();
        let pi = desugar(&prog).unwrap();
        assert_eq!(pi.store.num_vars(), 1);
        assert_eq!(pi.holes, vec![Hole(0)]);
        assert_eq!(pi.constraints.len(), 1);
        assert!(matches!(pi.constraints[0], Constraint::Equality { .. }));
    }

    #[test]
    fn desugar_leq_picks_bound_direction() {
        // Holes on the left → upper bound (lhs[holes] ⊆ rhs).
        let upper = desugar(&parse_program("hole h\nassert h <= (x0 == 1)\n").unwrap()).unwrap();
        assert!(matches!(
            upper.constraints[0],
            Constraint::UpperBound { .. }
        ));

        // Holes on the right → lower bound (lhs ⊆ rhs[holes]).
        let lower = desugar(&parse_program("hole h\nassert (x0 == 1) <= h\n").unwrap()).unwrap();
        assert!(matches!(
            lower.constraints[0],
            Constraint::LowerBound { .. }
        ));
    }

    #[test]
    fn desugar_errors_when_both_sides_have_holes() {
        let prog = parse_program("hole h\nhole g\nassert h <= g\n").unwrap();
        assert!(desugar(&prog).is_err());
    }

    #[test]
    #[should_panic(expected = "not supported by nk_with_holes")]
    fn to_hole_expr_panics_on_intersection() {
        let mut store = spp::SPPstore::new(1);
        let expr = Expr::Intersect(Expr::test(0, true), Expr::test(0, false));
        to_hole_expr(&expr, &mut store);
    }
}
