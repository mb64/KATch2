use crate::pre::{Field, Value};
use bumpalo::Bump;

/// A formal hole standing for an unknown sub-expression (an unknown SPP once
/// compiled).  Holes are introduced by the `.nksynth` synthesis front-end and
/// are *not* valid in expressions handed to `crate::aut`: they must be filled
/// in before automaton construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hole(pub u32);

/// Represents different pattern types for matching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern<'a> {
    /// CIDR notation: 192.168.1.0/24
    Cidr {
        address: &'a [bool], // Base IP address as bits
        prefix_len: usize,   // Number of bits in prefix
    },
    /// IP range: 192.168.1.10-192.168.1.20
    IpRange {
        start: &'a [bool], // Start IP as bits
        end: &'a [bool],   // End IP as bits
    },
    /// Wildcard mask: 192.168.1.0 with mask 0.0.0.255
    Wildcard {
        address: &'a [bool], // Base IP address as bits
        mask: &'a [bool],    // Wildcard mask as bits (1 = don't care)
    },
    /// Exact match (for consistency)
    Exact(&'a [bool]),
}

/// NetKAT syntax with LTL extensions, borrowing all storage for one lifetime.
///
/// Child nodes and bit slices live in the parser's [`Bump`]; names are slices of
/// the original input. Neither this enum nor [`Pattern`] owns heap allocations,
/// so releasing the arena needs no recursive traversal or node destructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expr<'a> {
    Zero,                                        // 0
    One,                                         // 1
    Top,                                         // T
    Assign(Field, Value),                        // field := value
    Test(Field, Value),                          // field == value
    VarAssign(&'a str, &'a [bool]),              // var := value (for aliases)
    VarTest(&'a str, &'a [bool]),                // var == value (for aliases)
    BitRangeAssign(Field, Field, &'a [bool]),    // x[start..end] := value (as bits)
    BitRangeTest(Field, Field, &'a [bool]),      // x[start..end] == value (as bits)
    BitRangeMatch(Field, Field, Pattern<'a>),    // x[start..end] ~ pattern
    VarMatch(&'a str, Pattern<'a>),              // var ~ pattern (for aliases)
    Union(Exp<'a>, Exp<'a>),                     // e1 + e2
    Intersect(Exp<'a>, Exp<'a>),                 // e1 & e2
    Xor(Exp<'a>, Exp<'a>),                       // e1 ^ e2
    Difference(Exp<'a>, Exp<'a>),                // e1 - e2
    Complement(Exp<'a>),                         // ~e1
    TestNegation(Exp<'a>),                       // !e (test fragment only)
    IfThenElse(Exp<'a>, Exp<'a>, Exp<'a>),       // if e1 then e2 else e3 (e1 must be test fragment)
    Var(&'a str),                                // Variable reference
    Let(&'a str, Exp<'a>, Exp<'a>),              // let x = e1 in e2
    LetBitRange(&'a str, Field, Field, Exp<'a>), // let alias = x[start..end] in e
    Sequence(Exp<'a>, Exp<'a>),                  // e1; e2
    Star(Exp<'a>),                               // e*
    Dup,                                         // dup
    LtlNext(Exp<'a>),                            // X e
    LtlUntil(Exp<'a>, Exp<'a>),                  // e1 U e2
    End,                                         // end
    Hole(Hole), // a synthesis hole (filled in before automaton construction)
}

/// An immutable expression allocated in a caller-owned arena.
pub type Exp<'a> = &'a Expr<'a>;

impl<'a> Expr<'a> {
    pub fn zero(arena: &'a Bump) -> Exp<'a> {
        arena.alloc(Expr::Zero)
    }
    pub fn one(arena: &'a Bump) -> Exp<'a> {
        arena.alloc(Expr::One)
    }
    pub fn top(arena: &'a Bump) -> Exp<'a> {
        arena.alloc(Expr::Top)
    }
    pub fn assign(arena: &'a Bump, field: Field, value: Value) -> Exp<'a> {
        arena.alloc(Expr::Assign(field, value))
    }
    pub fn test(arena: &'a Bump, field: Field, value: Value) -> Exp<'a> {
        arena.alloc(Expr::Test(field, value))
    }
    pub fn var_assign(arena: &'a Bump, var: &'a str, bits: impl AsRef<[bool]>) -> Exp<'a> {
        arena.alloc(Expr::VarAssign(var, arena.alloc_slice_copy(bits.as_ref())))
    }
    pub fn var_test(arena: &'a Bump, var: &'a str, bits: impl AsRef<[bool]>) -> Exp<'a> {
        arena.alloc(Expr::VarTest(var, arena.alloc_slice_copy(bits.as_ref())))
    }
    pub fn bit_range_assign(
        arena: &'a Bump,
        start: Field,
        end: Field,
        bits: impl AsRef<[bool]>,
    ) -> Exp<'a> {
        arena.alloc(Expr::BitRangeAssign(
            start,
            end,
            arena.alloc_slice_copy(bits.as_ref()),
        ))
    }
    pub fn bit_range_test(
        arena: &'a Bump,
        start: Field,
        end: Field,
        bits: impl AsRef<[bool]>,
    ) -> Exp<'a> {
        arena.alloc(Expr::BitRangeTest(
            start,
            end,
            arena.alloc_slice_copy(bits.as_ref()),
        ))
    }
    pub fn bit_range_match(
        arena: &'a Bump,
        start: Field,
        end: Field,
        pattern: Pattern<'a>,
    ) -> Exp<'a> {
        arena.alloc(Expr::BitRangeMatch(start, end, pattern))
    }
    pub fn var_match(arena: &'a Bump, var: &'a str, pattern: Pattern<'a>) -> Exp<'a> {
        arena.alloc(Expr::VarMatch(var, pattern))
    }
    pub fn union(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Union(e1, e2))
    }
    pub fn intersect(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Intersect(e1, e2))
    }
    pub fn xor(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Xor(e1, e2))
    }
    pub fn difference(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Difference(e1, e2))
    }
    pub fn complement(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Complement(e))
    }
    pub fn test_negation(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::TestNegation(e))
    }
    pub fn if_then_else(
        arena: &'a Bump,
        cond: Exp<'a>,
        then_expr: Exp<'a>,
        else_expr: Exp<'a>,
    ) -> Exp<'a> {
        arena.alloc(Expr::IfThenElse(cond, then_expr, else_expr))
    }
    pub fn var(arena: &'a Bump, name: &'a str) -> Exp<'a> {
        arena.alloc(Expr::Var(name))
    }
    pub fn let_in(arena: &'a Bump, var_name: &'a str, def: Exp<'a>, body: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Let(var_name, def, body))
    }
    pub fn let_bit_range(
        arena: &'a Bump,
        alias_name: &'a str,
        start: Field,
        end: Field,
        body: Exp<'a>,
    ) -> Exp<'a> {
        arena.alloc(Expr::LetBitRange(alias_name, start, end, body))
    }
    pub fn sequence(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Sequence(e1, e2))
    }
    pub fn star(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Star(e))
    }
    /// `e1 + e2`, but simplifying away `0`.
    pub fn union_simp(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        match (e1, e2) {
            (Expr::Zero, _) => e2,
            (_, Expr::Zero) => e1,
            _ => Expr::union(arena, e1, e2),
        }
    }
    /// `e1; e2`, but simplifying away `0` and `1`.
    pub fn sequence_simp(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        match (e1, e2) {
            (Expr::Zero, _) | (_, Expr::Zero) => Expr::zero(arena),
            (Expr::One, _) => e2,
            (_, Expr::One) => e1,
            _ => Expr::sequence(arena, e1, e2),
        }
    }
    /// `e*`, but simplifying `0*` and `1*` to `1`.
    pub fn star_simp(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        match e {
            Expr::Zero | Expr::One => Expr::one(arena),
            _ => Expr::star(arena, e),
        }
    }
    pub fn dup(arena: &'a Bump) -> Exp<'a> {
        arena.alloc(Expr::Dup)
    }
    pub fn ltl_next(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::LtlNext(e))
    }
    /// Weak next operator `(X' e = X e \/ end)`
    pub fn ltl_weak_next(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Union(Expr::ltl_next(arena, e), Expr::end(arena)))
    }
    pub fn ltl_until(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::LtlUntil(e1, e2))
    }
    /// Weak until operator `(e1 W e2 = e1 U e2 \/ G e1)`
    pub fn ltl_weak_until(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Union(
            Expr::ltl_until(arena, e1, e2),
            Expr::ltl_globally(arena, e1),
        ))
    }
    pub fn end(arena: &'a Bump) -> Exp<'a> {
        arena.alloc(Expr::End)
    }
    pub fn hole(arena: &'a Bump, hole: Hole) -> Exp<'a> {
        arena.alloc(Expr::Hole(hole))
    }

    /// Check if an expression is in the test fragment
    /// Test fragment consists of: 0, 1, logical operators (+, &, ^, -),
    /// field tests (==), test negation (!), and sequence (;)
    pub fn is_test_fragment(&self) -> bool {
        match self {
            Expr::Zero | Expr::One => true,
            Expr::Test(_, _)
            | Expr::BitRangeTest(_, _, _)
            | Expr::VarTest(_, _)
            | Expr::BitRangeMatch(_, _, _)
            | Expr::VarMatch(_, _) => true,
            Expr::Union(e1, e2)
            | Expr::Intersect(e1, e2)
            | Expr::Xor(e1, e2)
            | Expr::Difference(e1, e2)
            | Expr::Sequence(e1, e2) => e1.is_test_fragment() && e2.is_test_fragment(),
            Expr::TestNegation(e) => e.is_test_fragment(),
            // Everything else is not in test fragment
            Expr::Top
            | Expr::Assign(_, _)
            | Expr::BitRangeAssign(_, _, _)
            | Expr::VarAssign(_, _)
            | Expr::Complement(_)
            | Expr::Star(_)
            | Expr::Dup
            | Expr::LtlNext(_)
            | Expr::LtlUntil(_, _)
            | Expr::End
            | Expr::IfThenElse(_, _, _)
            | Expr::Var(_)
            | Expr::Let(_, _, _)
            | Expr::Hole(_)
            | Expr::LetBitRange(_, _, _, _) => false,
        }
    }

    pub fn num_fields(&self) -> u32 {
        match self {
            Expr::Zero | Expr::One | Expr::Top | Expr::Dup | Expr::End => 0,
            Expr::Assign(field, _) | Expr::Test(field, _) => field + 1,
            Expr::VarAssign(_, _) | Expr::VarTest(_, _) | Expr::VarMatch(_, _) => 0, // Variables don't directly reference fields
            Expr::BitRangeAssign(_start, end, _)
            | Expr::BitRangeTest(_start, end, _)
            | Expr::BitRangeMatch(_start, end, _) => end + 1,
            Expr::Union(e1, e2)
            | Expr::Intersect(e1, e2)
            | Expr::Xor(e1, e2)
            | Expr::Difference(e1, e2)
            | Expr::Sequence(e1, e2)
            | Expr::LtlUntil(e1, e2) => e1.num_fields().max(e2.num_fields()),
            Expr::IfThenElse(cond, then_expr, else_expr) => cond
                .num_fields()
                .max(then_expr.num_fields())
                .max(else_expr.num_fields()),
            Expr::Let(_, def, body) => def.num_fields().max(body.num_fields()),
            Expr::LetBitRange(_, _start, end, body) => (end + 1).max(body.num_fields()),
            Expr::Complement(e) | Expr::TestNegation(e) | Expr::Star(e) | Expr::LtlNext(e) => {
                e.num_fields()
            }
            Expr::Var(_) => 0,  // Variables don't directly reference fields
            Expr::Hole(_) => 0, // Holes don't directly reference fields
        }
    }

    /// Helper function for constructing `F e` using the equivalence `F e ≡ true U e`
    pub fn ltl_finally(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::LtlUntil(Expr::top(arena), e))
    }

    // Helper function for constructing `G e` using the equivalence `G e ≡ ¬(true U ¬e)`
    pub fn ltl_globally(arena: &'a Bump, e: Exp<'a>) -> Exp<'a> {
        let until_expr = Expr::ltl_until(arena, Expr::top(arena), Expr::complement(arena, e));
        arena.alloc(Expr::Complement(until_expr))
    }

    // Helper function for constructing `e1 R e2` (weak release) using the equivalence `e1 R e2 ≡ ¬(¬e1 U ¬e2)`
    pub fn ltl_release(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        let not_e1 = Expr::complement(arena, e1);
        let not_e2 = Expr::complement(arena, e2);
        let until_expr = Expr::ltl_until(arena, not_e1, not_e2);
        arena.alloc(Expr::Complement(until_expr))
    }

    /// Strong release operator `(e1 M e2 = e1 R e2 /\ F e1)`
    pub fn ltl_strong_release(arena: &'a Bump, e1: Exp<'a>, e2: Exp<'a>) -> Exp<'a> {
        arena.alloc(Expr::Intersect(
            Expr::ltl_release(arena, e1, e2),
            Expr::ltl_finally(arena, e1),
        ))
    }

    /// Substitute all occurrences of variable `var` with expression `replacement` in `self`
    pub fn substitute(&self, arena: &'a Bump, var: &str, replacement: Exp<'a>) -> Exp<'a> {
        match self {
            // Base cases
            Expr::Zero => Expr::zero(arena),
            Expr::One => Expr::one(arena),
            Expr::Top => Expr::top(arena),
            Expr::Dup => Expr::dup(arena),
            Expr::End => Expr::end(arena),
            Expr::Hole(h) => Expr::hole(arena, *h),
            Expr::Assign(f, v) => Expr::assign(arena, *f, *v),
            Expr::Test(f, v) => Expr::test(arena, *f, *v),
            Expr::VarAssign(var, bits) => Expr::var_assign(arena, var, bits),
            Expr::VarTest(var, bits) => Expr::var_test(arena, var, bits),

            // Variable case - this is where substitution happens
            Expr::Var(name) => {
                if *name == var {
                    replacement
                } else {
                    Expr::var(arena, name)
                }
            }

            // Recursive cases
            Expr::Union(e1, e2) => Expr::union(
                arena,
                e1.substitute(arena, var, replacement),
                e2.substitute(arena, var, replacement),
            ),
            Expr::Intersect(e1, e2) => Expr::intersect(
                arena,
                e1.substitute(arena, var, replacement),
                e2.substitute(arena, var, replacement),
            ),
            Expr::Xor(e1, e2) => Expr::xor(
                arena,
                e1.substitute(arena, var, replacement),
                e2.substitute(arena, var, replacement),
            ),
            Expr::Difference(e1, e2) => Expr::difference(
                arena,
                e1.substitute(arena, var, replacement),
                e2.substitute(arena, var, replacement),
            ),
            Expr::Sequence(e1, e2) => Expr::sequence(
                arena,
                e1.substitute(arena, var, replacement),
                e2.substitute(arena, var, replacement),
            ),
            Expr::Complement(e) => Expr::complement(arena, e.substitute(arena, var, replacement)),
            Expr::TestNegation(e) => {
                Expr::test_negation(arena, e.substitute(arena, var, replacement))
            }
            Expr::Star(e) => Expr::star(arena, e.substitute(arena, var, replacement)),
            Expr::LtlNext(e) => Expr::ltl_next(arena, e.substitute(arena, var, replacement)),
            Expr::LtlUntil(e1, e2) => Expr::ltl_until(
                arena,
                e1.substitute(arena, var, replacement),
                e2.substitute(arena, var, replacement),
            ),
            Expr::IfThenElse(c, t, e) => Expr::if_then_else(
                arena,
                c.substitute(arena, var, replacement),
                t.substitute(arena, var, replacement),
                e.substitute(arena, var, replacement),
            ),

            // Let binding case - careful with variable shadowing
            Expr::Let(bound_var, def, body) => {
                if *bound_var == var {
                    // Variable is shadowed, don't substitute in body
                    Expr::let_in(
                        arena,
                        bound_var,
                        def.substitute(arena, var, replacement),
                        body,
                    )
                } else {
                    // Substitute in both definition and body
                    Expr::let_in(
                        arena,
                        bound_var,
                        def.substitute(arena, var, replacement),
                        body.substitute(arena, var, replacement),
                    )
                }
            }

            // Let bit range case - careful with variable shadowing
            Expr::LetBitRange(alias_name, start, end, body) => {
                if *alias_name == var {
                    // Alias is shadowed, don't substitute in body
                    Expr::let_bit_range(arena, alias_name, *start, *end, body)
                } else {
                    // Substitute in body
                    Expr::let_bit_range(
                        arena,
                        alias_name,
                        *start,
                        *end,
                        body.substitute(arena, var, replacement),
                    )
                }
            }

            // Bit range operations don't contain variables
            Expr::BitRangeAssign(start, end, value) => {
                Expr::bit_range_assign(arena, *start, *end, value)
            }
            Expr::BitRangeTest(start, end, value) => {
                Expr::bit_range_test(arena, *start, *end, value)
            }
            Expr::BitRangeMatch(start, end, pattern) => {
                Expr::bit_range_match(arena, *start, *end, *pattern)
            }
            Expr::VarMatch(var_name, pattern) => Expr::var_match(arena, var_name, *pattern),
        }
    }
}

impl std::fmt::Display for Expr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Zero => write!(f, "0"),
            Expr::One => write!(f, "1"),
            Expr::Top => write!(f, "⊤"),
            Expr::Assign(field, value) => {
                write!(f, "x{} := {}", field, if *value { "1" } else { "0" })
            }
            Expr::Test(field, value) => {
                write!(f, "x{} == {}", field, if *value { "1" } else { "0" })
            }
            Expr::VarAssign(var, bits) => {
                let num = bits_to_number(bits);
                write!(f, "{} := {}", var, num)
            }
            Expr::VarTest(var, bits) => {
                let num = bits_to_number(bits);
                write!(f, "{} == {}", var, num)
            }
            Expr::Union(e1, e2) => write!(f, "({} + {})", e1, e2),
            Expr::Intersect(e1, e2) => write!(f, "({} & {})", e1, e2),
            Expr::Xor(e1, e2) => write!(f, "({} ^ {})", e1, e2),
            Expr::Difference(e1, e2) => write!(f, "({} - {})", e1, e2),
            Expr::Complement(e) => write!(f, "~{}", e),
            Expr::TestNegation(e) => write!(f, "!{}", e),
            Expr::IfThenElse(cond, then_expr, else_expr) => {
                write!(f, "(if {} then {} else {})", cond, then_expr, else_expr)
            }
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Let(var, def, body) => write!(f, "(let {} = {} in {})", var, def, body),
            Expr::LetBitRange(alias, start, end, body) => {
                write!(f, "(let {} = x[{}..{}] in {})", alias, start, end, body)
            }
            Expr::Sequence(e1, e2) => write!(f, "({} ; {})", e1, e2),
            Expr::Star(e) => write!(f, "({})*", e),
            Expr::Dup => write!(f, "dup"),
            Expr::Hole(Hole(h)) => write!(f, "?{}", h),
            Expr::LtlNext(e) => write!(f, "X({})", e),
            Expr::LtlUntil(e1, e2) => write!(f, "({} U {})", e1, e2),
            Expr::End => write!(f, "end"),
            Expr::BitRangeAssign(start, end, value) => {
                let num = bits_to_number(value);
                write!(f, "x[{}..{}] := {}", start, end, num)
            }
            Expr::BitRangeTest(start, end, value) => {
                let num = bits_to_number(value);
                write!(f, "x[{}..{}] == {}", start, end, num)
            }
            Expr::BitRangeMatch(start, end, pattern) => {
                write!(f, "x[{}..{}] ~ {}", start, end, pattern)
            }
            Expr::VarMatch(var, pattern) => {
                write!(f, "{} ~ {}", var, pattern)
            }
        }
    }
}

// Helper function to convert bit vector to number for display
fn bits_to_number(bits: &[bool]) -> u128 {
    bits.iter().enumerate().fold(
        0u128,
        |acc, (i, &bit)| {
            if bit { acc | (1 << i) } else { acc }
        },
    )
}

impl std::fmt::Display for Pattern<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pattern::Exact(bits) => {
                let num = bits_to_number(bits);
                write!(f, "{}", num)
            }
            Pattern::Cidr {
                address,
                prefix_len,
            } => {
                // Convert to IP address format if 32 bits
                if address.len() == 32 {
                    let num = bits_to_number(address);
                    let a = (num >> 24) & 0xff;
                    let b = (num >> 16) & 0xff;
                    let c = (num >> 8) & 0xff;
                    let d = num & 0xff;
                    write!(f, "{}.{}.{}.{}/{}", a, b, c, d, prefix_len)
                } else {
                    let num = bits_to_number(address);
                    write!(f, "{}/{}", num, prefix_len)
                }
            }
            Pattern::Wildcard { address, mask } => {
                let addr_num = bits_to_number(address);
                let mask_num = bits_to_number(mask);
                write!(f, "{} mask {}", addr_num, mask_num)
            }
            Pattern::IpRange { start, end } => {
                // Convert to IP address format if 32 bits
                if start.len() == 32 && end.len() == 32 {
                    let start_num = bits_to_number(start);
                    let end_num = bits_to_number(end);
                    let s_a = (start_num >> 24) & 0xff;
                    let s_b = (start_num >> 16) & 0xff;
                    let s_c = (start_num >> 8) & 0xff;
                    let s_d = start_num & 0xff;
                    let e_a = (end_num >> 24) & 0xff;
                    let e_b = (end_num >> 16) & 0xff;
                    let e_c = (end_num >> 8) & 0xff;
                    let e_d = end_num & 0xff;
                    write!(
                        f,
                        "{}.{}.{}.{}-{}.{}.{}.{}",
                        s_a, s_b, s_c, s_d, e_a, e_b, e_c, e_d
                    )
                } else {
                    let start_num = bits_to_number(start);
                    let end_num = bits_to_number(end);
                    write!(f, "{}-{}", start_num, end_num)
                }
            }
        }
    }
}
