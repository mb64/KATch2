# KATch2-synth

A solver for Existential NetKAT, based on [KATch2](https://github.com/julesjacobs/KATch2).

## Quick start

1. Install Rust (using `rustup`), a C++ compiler, and libclang (for the bundled CaDiCaL build)
2. `cargo build --release`
3. `./target/release/nksynth example.nksynth`

Command-line options:

```
Solve NetKAT synthesis (`.nksynth`) problems, reporting SAT or UNSAT

Usage: nksynth [OPTIONS] <FILE>...

Arguments:
  <FILE>...  `.nksynth` file(s) to solve

Options:
      --full                 Use the full solver (holes may emit `dup`)
      --no-full              Use the dup-free solver (the default)
      --iteration-limit <N>  Give up after this many CEGIS refinement rounds, reporting `UNKNOWN` instead of looping until solved or proven infeasible
  -h, --help                 Print help
  -V, --version              Print version
```

For what can appear in a `.nksynth` file, see `example.nksynth`.


## SPs and SPPs

- SP: represents symbolic packets (sets of concrete packets, represented as a BDD)
- SPP: represents symbolic packet programs (relations on concrete packets, represented as a BDD)

Our BDDs always store all intermediate levels. This is particularly relevant for SPP, where it is not clear what a missing level would indicate (zero, one, or top for the missing variables). In the future, we can investigate whether it is profitable do introduce a more complex scheme that can skip intermediate levels.

## Upper-bound inclusion checking

Upper-bound checks search the ENFA directly, without materializing an epsilon
closure NFA. An outer worklist propagates packet sets at visible product states
with the upper-bound DFA. For each new set, an inner worklist propagates
relations from the preceding visible packet to the current packet through
invisible transitions. At a visible boundary, it intersects with the DFA
transition before projecting to a packet set; at termination, it intersects
with the complement's output relation. Propagation records reconstruct the
full path, including invisible states, without a separate elaboration pass.
Lower-bound checks still use epsilon closure and determinization.

Build a binary with debug symbols for Samply using
`cargo build --profile profiling --bin nksynth`.

## Symbolic upper bounds

When an upper-bound counterexample traverses exactly one SPP hole, the solver
learns an entire forbidden relation for that hole. It keeps the concrete state
path and the other trace packets fixed, while generalizing the two endpoints
of the segment containing the hole. Prefix and suffix reachability are computed
in the product of this path with the upper-bound complement: transition
relations are intersected before projecting to packet sets. Complementing the
DFA preserves its transitions and complements its final output relations,
including a sink for missing transitions.

For a concrete context `A; P; B`, if `bad` relates segment endpoints that violate
the upper bound, the forbidden hole relation is `flip(A); bad; flip(B)`. Its
complement is the condition `phi` in `P(x,y) => phi(x,y)`.

The SAT learner checks each model against these regions alongside lazy
Ackermannization. A violating symbolic application gets an instance
`P(args) => !forbidden(args)`, then the solver retries. Candidate learning also
uses the whole forbidden diagram as negative evidence. It merges shared
positive/negative suffix diagrams level by level, without enumerating their
packet pairs. Multi-hole counterexamples and `Cand` traversals retain concrete
negative clauses; SPP traversals in the full solver benefit as well.

With `KATCH2_METRICS=1`, native counters include `forbidden_regions`,
`forbidden_refinement_rounds`, and `forbidden_instances` when applicable.


## Aut

Automata are unlabeled nodes connected via SPPs. Since each SPP represents packet pairs (pk1, pk2), the language of an Aut is a string of such packet pairs. However, since this represents a packet transformation from pk1 to pk2, the n-th out packet must be the same as the (n+1)-th in packet. That is, in a string ... (in_i, out_i) (in_{i+1}, out_{i+1}) ... we must have out_i = in_{i+1}. Strings that violate this principle are not considered to be part of the language accepted by the Aut.

## Syntax

The language supports the following expressions:

```
e ::= 
    | 0           -- zero, drop packet
    | 1           -- one, forward packet
    | T           -- top, turns any packet into any other
    | field := value  -- field assignment
    | field == value  -- field test
    | e1 + e2     -- union, nondeterminism
    | e1 & e2     -- intersection
    | e1 ^ e2     -- xor
    | e1 - e2     -- difference
    | ~e1         -- complement, negation
    | !e1         -- test negation (only for test fragment)
    | if e1 then e2 else e3  -- conditional (e1 must be test fragment)
    | let x = e1 in e2       -- let binding
    | x[start..end] := value -- bit range assignment
    | x[start..end] == value -- bit range test
    | e1; e2      -- sequence
    | e*          -- star, iteration
    | dup         -- log current packet to trace
    | X e         -- LTL next
    | e1 U e2     -- LTL until (maybe change this into LDL)

field ::= x0 | x1 | x2 | ... | xk  -- packet forms a bitfield
value ::= 0 | 1 | number | 0b... | 0x... | ip -- for bit ranges, supports multiple formats
```

Notes:
- The parser takes `k` as an argument to determine the number of available fields.
- Test negation `!e` is only valid for expressions in the test fragment, which consists of:
  - Constants: 0, 1
  - Field tests: x == value
  - Logical operators: +, &, ^, -, ;
  - Test negation itself: !
  - Expressions built from the above
- The `!` operator is eliminated during desugaring using De Morgan's laws.
- The `if-then-else` expression is desugared to `(cond ; then) + (!cond ; else)`.
- The `let x = e1 in e2` expression is desugared by substituting all occurrences of `x` in `e2` with `e1`.
- Variables can be any identifier except reserved keywords. Let bindings can be nested and support shadowing.
- Bit range operations `x[start..end] := value` and `x[start..end] == value` operate on multiple bits at once:
  - `x[0..8] := 255` assigns bits 0-7 to the binary representation of 255
  - `x[0..4] == 5` tests if bits 0-3 equal the binary representation of 5
  - `x[0..8] := 0xFF` uses hexadecimal notation (equivalent to 255)
  - `x[0..4] := 0b1010` uses binary notation (equivalent to 10)
  - `x[0..32] := 192.168.1.1` uses IP address notation (converted to 32-bit integer)
  - These are desugared into sequences of individual bit operations
  - The range `[start..end)` is half-open (excludes end)
  - All literal formats are converted to little-endian bit vectors

## License

This project is licensed under the MIT License. See `LICENSE`.
