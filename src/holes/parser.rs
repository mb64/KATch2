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

use super::desugar::{Env, Term};
use crate::desugar::DesugarError;
use crate::expr::{Exp, Expr, Hole};
use crate::hash::HashMapExt as _;
use crate::holes::aut::{is_empty, ops};
use crate::holes::problem::{Constraint, ProblemInstance};
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

/// Lower a parsed [`Program`] into a [`ProblemInstance`].
///
/// Maintains shared compiled definitions: each `hole NAME` binds `NAME` to a fresh
/// [`Expr::Hole`] (so expressions referencing it pick up the hole during
/// desugaring), and each `def NAME = expr` binds `NAME` to its desugared body.
/// Cached hole information decides the constraint shape for every assertion:
///
/// * `lhs <= rhs` with holes on the left  → [`Constraint::upper_bound`] (`lhs[holes] ⊆ rhs`)
/// * `lhs <= rhs` with holes on the right → [`Constraint::lower_bound`] (`lhs ⊆ rhs[holes]`)
/// * `lhs == rhs` with holes              → [`Constraint::equality`]
///
/// The hole-free side is compiled into the reference DFA; the hole-bearing side
/// into the hole automaton. Hole-free assertions are checked here: passing
/// assertions are omitted, and failing assertions become [`Constraint::False`].
pub fn desugar(program: &Program) -> Result<ProblemInstance, DesugarError> {
    let n = num_fields(program);
    let mut store = spp::SPPstore::new(n);
    let mut env = Env::default();
    let mut holes: Vec<Hole> = Vec::new();
    let mut constraints: Vec<Constraint> = Vec::new();

    for stmt in &program.statements {
        match stmt {
            Stmt::Hole(name) => {
                let h = Hole(holes.len() as u32);
                holes.push(h);
                env.bind(name.clone(), Term::hole(h));
            }
            Stmt::Def(name, expr) => {
                let desugared = env.compile(expr, &mut store)?;
                env.bind(name.clone(), desugared);
            }
            Stmt::Assert { lhs, op, rhs } => {
                let dl = env.compile(lhs, &mut store)?;
                let dr = env.compile(rhs, &mut store)?;
                if let Some(constraint) = build_constraint(&mut store, &dl, *op, &dr)? {
                    constraints.push(constraint);
                }
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
/// DFA. Passing hole-free assertions are verified immediately and return `None`.
/// A false assertion returns [`Constraint::False`]; having holes on both
/// sides is an error.
fn build_constraint(
    store: &mut spp::SPPstore,
    lhs: &Term,
    op: AssertOp,
    rhs: &Term,
) -> Result<Option<Constraint>, DesugarError> {
    let lhs_holes = lhs.has_holes;
    let rhs_holes = rhs.has_holes;
    if lhs_holes && rhs_holes {
        return Err(DesugarError {
            message:
                "both sides of an assertion contain holes; at least one side must be hole-free"
                    .to_string(),
        });
    }

    if !lhs_holes && !rhs_holes {
        let lhs_dfa = lhs.to_dfa(store);
        let rhs_dfa = rhs.to_dfa(store);
        let contained = |store: &mut spp::SPPstore, lhs, rhs| {
            is_empty(&ops::intersection(lhs, ops::complement(rhs)), store)
        };
        if !contained(store, &lhs_dfa, &rhs_dfa)
            || (op == AssertOp::Eq && !contained(store, &rhs_dfa, &lhs_dfa))
        {
            return Ok(Some(Constraint::False));
        }
        return Ok(None);
    }

    let constraint = match op {
        AssertOp::Eq => {
            // Equality is symmetric: put the hole-bearing side on the automaton side.
            let (hole_side, dfa_side) = if rhs_holes { (rhs, lhs) } else { (lhs, rhs) };
            let dfa = dfa_side.to_dfa(store);
            let hole_expr = hole_side.to_hole_expr();
            Constraint::equality(store, &hole_expr, dfa)
        }
        AssertOp::Leq if rhs_holes => {
            // lhs ⊆ rhs[holes]: the DFA is a lower bound on the automaton.
            let dfa = lhs.to_dfa(store);
            let hole_expr = rhs.to_hole_expr();
            Constraint::lower_bound(store, &hole_expr, dfa)
        }
        AssertOp::Leq => {
            // lhs[holes] ⊆ rhs: the DFA is an upper bound on the automaton.
            let dfa = rhs.to_dfa(store);
            let hole_expr = lhs.to_hole_expr();
            Constraint::upper_bound(store, &hole_expr, dfa)
        }
    };

    Ok(Some(constraint))
}

// --- Plugging a solution into a Program ---------------------------------------

/// Fill in the holes of `program` with the definitions in `solution`, giving a hole-free program.
///
/// `solution` must consist only of `def NAME = expr` statements, one per hole of `program` (this is
/// the format `nksynth --output` writes). Each `hole NAME` statement is replaced, in place, by the
/// solution's `def NAME = expr`, so the rest of the program sees the solution wherever it used the
/// hole.
pub fn fill_holes(program: &Program, solution: &Program) -> Result<Program, String> {
    let mut defs: crate::hash::HashMap<&str, &Exp> = crate::hash::HashMap::new();
    for stmt in &solution.statements {
        match stmt {
            Stmt::Def(name, expr) => {
                if defs.insert(name, expr).is_some() {
                    return Err(format!("solution defines `{name}` more than once"));
                }
            }
            _ => return Err("solution may only contain `def` statements".to_string()),
        }
    }

    let mut statements = Vec::with_capacity(program.statements.len());
    for stmt in &program.statements {
        statements.push(match stmt {
            Stmt::Hole(name) => {
                let expr = defs
                    .remove(name.as_str())
                    .ok_or_else(|| format!("solution has no definition for hole `{name}`"))?;
                Stmt::Def(name.clone(), expr.clone())
            }
            other => other.clone(),
        });
    }

    if let Some(name) = defs.keys().next() {
        return Err(format!("solution defines `{name}`, which is not a hole"));
    }
    Ok(Program { statements })
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
    fn hole_free_assertions_are_checked_during_construction() {
        for source in [
            "assert (x0 == 1) <= 1",
            "assert (x0 == 1) + (x0 == 0) == 1",
            "assert dup <= dup + 1",
            "assert (dup + 1) - dup == 1",
            "assert (dup + 1) & dup == dup",
            "assert 0 == 0",
        ] {
            let instance = desugar(&parse_program(source).unwrap()).unwrap();
            assert!(instance.constraints.is_empty(), "{source}");
        }

        for source in [
            "assert 1 <= (x0 == 1)",
            "assert (x0 == 1) == 1",
            "assert 1 == (x0 == 1)",
            "assert dup + 1 == dup",
            "assert dup == dup + 1",
            "assert dup <= 1",
            "hole h\nassert h == 1\nassert 1 == 0",
        ] {
            let mut instance = desugar(&parse_program(source).unwrap()).unwrap();
            assert!(
                instance
                    .constraints
                    .iter()
                    .any(|c| matches!(c, Constraint::False)),
                "{source}"
            );
            assert_eq!(
                instance.solve(Some(0)),
                Err(crate::holes::cegis::CegisError::Infeasible),
                "{source}"
            );
            assert!(
                matches!(
                    instance.solve_full(Some(0)),
                    Err(crate::holes::cegis::CegisError::Infeasible)
                ),
                "{source}"
            );
        }

        let source = "hole h\ndef identity = 1\nassert identity == 1\nassert h == (x0 := 1)\nassert dup <= dup + 1";
        let mut instance = desugar(&parse_program(source).unwrap()).unwrap();
        assert_eq!(instance.constraints.len(), 1);
        assert!(matches!(
            instance.constraints[0],
            Constraint::Equality { .. }
        ));
        assert!(instance.solve(Some(100)).unwrap().contains_key(&Hole(0)));
        assert!(
            instance
                .solve_full(Some(100))
                .unwrap()
                .contains_key(&Hole(0))
        );
    }

    #[test]
    fn fill_holes_checks_solutions() {
        let prog = parse_program("hole h\nhole g\nassert h; g == (x0 := 1; x1 := 0)\n").unwrap();
        let check = |solution: &str| {
            let filled = fill_holes(&prog, &parse_program(solution).unwrap())?;
            let mut instance = desugar(&filled).map_err(|e| e.to_string())?;
            assert!(instance.holes.is_empty());
            Ok::<bool, String>(instance.solve(None).is_ok())
        };

        assert_eq!(check("def h = x0 := 1\ndef g = x1 := 0\n"), Ok(true));
        assert_eq!(check("def g = x1 := 0\ndef h = x0 := 1\n"), Ok(true));
        assert_eq!(check("def h = x0 := 0\ndef g = x1 := 0\n"), Ok(false));
        assert!(check("def h = x0 := 1\n").is_err());
        assert!(check("def h = x0 := 1\ndef g = 1\ndef k = 1\n").is_err());
        assert!(check("def h = 1\ndef h = 1\ndef g = 1\n").is_err());
        assert!(check("hole h\ndef g = 1\n").is_err());
    }
}
