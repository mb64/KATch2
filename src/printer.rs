//! Pretty-printer for NetKAT expressions, the inverse of [`crate::parser`].
//!
//! Unlike the [`Display`](std::fmt::Display) impl on [`Expr`], which parenthesizes every compound
//! expression, this only adds the parentheses needed for [`crate::parser`] to read the expression
//! back as the same tree. It follows the parser's precedence table (from loosest to tightest):
//!
//! | Level | Operators                  | Associativity |
//! |-------|----------------------------|---------------|
//! | 0     | `U`                        | right         |
//! | 1     | `+`, `^`, `-`              | left          |
//! | 2     | `;`                        | left          |
//! | 3     | `&`                        | left          |
//! | 4     | prefix `~`, `!`, `X`       | right         |
//! | 5     | postfix `*`                | left          |
//! | 6     | atoms: `0`, `x3 := 1`, ... |               |
//!
//! `if`/`let` are always parenthesized unless they are the whole expression.

use crate::expr::Expr;
use std::fmt::Write;

/// Pretty-print `expr` with as few parentheses as the parser allows.
pub fn pretty(expr: &Expr) -> String {
    let mut out = String::new();
    write_expr(&mut out, expr, LOWEST);
    out
}

/// Below every operator, so that nothing needs parentheses at the top level.
const LOWEST: i8 = -1;
const UNTIL: i8 = 0;
const ADDITIVE: i8 = 1;
const SEQUENCE: i8 = 2;
const INTERSECT: i8 = 3;
const PREFIX: i8 = 4;
const POSTFIX: i8 = 5;
const ATOM: i8 = 6;

fn precedence(expr: &Expr) -> i8 {
    match expr {
        Expr::IfThenElse(..) | Expr::Let(..) | Expr::LetBitRange(..) => LOWEST,
        Expr::LtlUntil(..) => UNTIL,
        Expr::Union(..) | Expr::Xor(..) | Expr::Difference(..) => ADDITIVE,
        Expr::Sequence(..) => SEQUENCE,
        Expr::Intersect(..) => INTERSECT,
        Expr::Complement(_) | Expr::TestNegation(_) | Expr::LtlNext(_) => PREFIX,
        Expr::Star(_) => POSTFIX,
        _ => ATOM,
    }
}

/// Write `expr`, in a position where the parser reads an expression of precedence at least
/// `min_prec`, parenthesizing it if its own precedence is lower.
fn write_expr(out: &mut String, expr: &Expr, min_prec: i8) {
    let prec = precedence(expr);
    let parens = prec < min_prec;
    if parens {
        out.push('(');
    }

    match expr {
        Expr::LtlUntil(a, b) => write_binary(out, a, " U ", b, UNTIL + 1, UNTIL),
        Expr::Union(a, b) => write_binary(out, a, " + ", b, ADDITIVE, ADDITIVE + 1),
        Expr::Xor(a, b) => write_binary(out, a, " ^ ", b, ADDITIVE, ADDITIVE + 1),
        Expr::Difference(a, b) => write_binary(out, a, " - ", b, ADDITIVE, ADDITIVE + 1),
        Expr::Sequence(a, b) => write_binary(out, a, "; ", b, SEQUENCE, SEQUENCE + 1),
        Expr::Intersect(a, b) => write_binary(out, a, " & ", b, INTERSECT, INTERSECT + 1),
        Expr::Complement(e) => write_prefix(out, "~", e),
        Expr::TestNegation(e) => write_prefix(out, "!", e),
        Expr::LtlNext(e) => write_prefix(out, "X ", e),
        Expr::Star(e) => {
            write_expr(out, e, POSTFIX);
            out.push('*');
        }
        Expr::IfThenElse(c, t, e) => {
            out.push_str("if ");
            write_expr(out, c, LOWEST);
            out.push_str(" then ");
            write_expr(out, t, LOWEST);
            out.push_str(" else ");
            write_expr(out, e, LOWEST);
        }
        Expr::Let(name, def, body) => {
            write!(out, "let {name} = ").unwrap();
            write_expr(out, def, LOWEST);
            out.push_str(" in ");
            write_expr(out, body, LOWEST);
        }
        Expr::LetBitRange(name, start, end, body) => {
            write!(out, "let {name} = x[{start}..{end}] in ").unwrap();
            write_expr(out, body, LOWEST);
        }
        Expr::Top => out.push('T'),
        // The remaining atoms already print in the parser's syntax
        atom => write!(out, "{atom}").unwrap(),
    }

    if parens {
        out.push(')');
    }
}

fn write_binary(out: &mut String, a: &Expr, op: &str, b: &Expr, a_prec: i8, b_prec: i8) {
    write_expr(out, a, a_prec);
    out.push_str(op);
    write_expr(out, b, b_prec);
}

fn write_prefix(out: &mut String, op: &str, e: &Expr) {
    out.push_str(op);
    write_expr(out, e, PREFIX);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Lexer, Parser};

    fn parse(s: &str) -> Box<Expr> {
        Parser::new(Lexer::new(s))
            .parse_single_expression()
            .unwrap_or_else(|e| panic!("failed to parse {s:?}: {}", e.message))
    }

    #[test]
    fn minimal_parentheses() {
        let cases = [
            "x0 == 1; x1 := 0 + x2 == 1",
            "(x0 == 1 + x1 == 0); dup; x1 := 1",
            "(x0 := 1; dup)*; x1 == 0",
            "a + (b + c)",
            "a; (b; c)",
            "~(a + b) & c*",
            "(!a)*",
            "!a*",
            "a U b U c",
            "(a U b) U c",
            "0 + 1 + T",
        ];
        for s in cases {
            assert_eq!(pretty(&parse(s)), s);
        }
    }

    #[test]
    fn roundtrip_random() {
        crate::fuzz::seed_fuzzer(0x5EED_0030);
        for _ in 0..1000 {
            let (expr, _) = crate::fuzz::genax(0, 5, 3);
            let printed = pretty(&expr);
            assert_eq!(parse(&printed), expr, "printed as {printed}");
        }
    }
}
