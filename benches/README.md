# Synthesis benchmarks

`benches/synthesis.rs` times the end-to-end CEGIS loop
(`ProblemInstance::solve` / `solve_full`). It measures the clause builders and
solver-shrinking optimizations:

- the **forward frontier**, additional **backward frontier**, and additional
  **min-cut clause** builders (`src/holes/cegis.rs`),
- the **SMT clause simplification / merge** (`src/holes/smt.rs`), and
- the **greedy example cover** before passive learning (`src/holes/smt.rs`).

## Workload

- **`corpus/spp_dup_free`**, **`corpus/full_dupful`** — the headline metric:
  total time to solve a *seeded, deterministic corpus* of `CORPUS_SIZE` random
  satisfiable-by-construction instances (the same generator as the crate's
  round-trip fuzzers, ported with a threaded `StdRng`). The fixed seed +
  `rand`'s reproducible `StdRng` make the corpus byte-identical run-to-run and
  branch-to-branch, so it is a broad sample that is still directly comparable.
  `spp_dup_free` uses the dup-free `solve`; `full_dupful` uses `solve_full`.
- **`pigeon/pigeon2`**, **`pigeon/pigeon3`**, **`pigeon/pigeon4`** — deterministic
  pigeonhole workloads with 4, 8, and 16 pigeons respectively. These instances
  are unsatisfiable, but a measured run can hit the refinement cap before
  proving infeasibility. Timings include both proven UNSAT and cap-hit UNKNOWN
  outcomes. This most directly stresses the lower-bound clause machinery.

## Running

```sh
cargo bench --bench synthesis            # all groups
cargo bench --bench synthesis -- corpus  # filter by name substring
```

## Did the optimizations help? (A/B via runtime flags)

Each optimization is a runtime flag (`katch2::flags`, all on by default),
toggled with an environment variable, so every on/off combination runs from
one compiled tree — no rebuilds or branch-switching needed:

| env var | optimization |
| --- | --- |
| `KATCH2_LB_MINCUT` | add a min-cut clause alongside the enabled frontiers |
| `KATCH2_LB_BACKWARD` | add a backward-frontier clause alongside the forward frontier |
| `KATCH2_CLAUSE_MERGE` | collate SMT membership disjuncts before solving |
| `KATCH2_EXAMPLE_COVER` | greedy set cover of examples before passive learning |

The `Makefile` runs the synthesis bench under every combo, saving a Criterion
baseline per combo so they auto-compare:

```sh
make bench-all          # baselines: default, none, mincut, merge, cover
# then compare any two saved baselines (negative % = first is faster):
cargo bench --bench synthesis -- --load-baseline mincut --baseline none
```

Or drive a single combo directly:

```sh
cargo bench --bench synthesis                      # all on (default)
KATCH2_LB_MINCUT=0 KATCH2_LB_BACKWARD=0 \
  KATCH2_CLAUSE_MERGE=0 KATCH2_EXAMPLE_COVER=0 \
  cargo bench --bench synthesis                    # all off (baseline)
KATCH2_CLAUSE_MERGE=0 KATCH2_EXAMPLE_COVER=0 \
  cargo bench --bench synthesis                    # all three clauses, no merge or cover
```

(The corpus is seeded, so it is identical across combos and the medians are
directly comparable.)

## Frontier comparison

Release CLI wall-time medians from seven runs per configuration, alternating
configuration order, with the same binary and a 1,000-round refinement cap.
Clause merging, example cover, and per-counterexample clause deduplication are
enabled throughout. Pigeonhole 3 proves UNSAT; pigeonhole 4 hits the cap.

| Case | One frontier | One frontier + min cut | Both frontiers | Both frontiers + min cut |
| --- | ---: | ---: | ---: | ---: |
| Pigeonhole 3 (UNSAT) | 0.974 s | 0.959 s | 0.765 s | 0.837 s |
| Pigeonhole 4 (UNKNOWN) | 1.070 s | 1.057 s | 0.620 s | 0.657 s |

The forward frontier is always enabled. Select the other builders independently:

| Configuration | `KATCH2_LB_BACKWARD` | `KATCH2_LB_MINCUT` | CLI flags |
| --- | ---: | ---: | --- |
| One frontier | 0 | 0 | `--no-lb-backward --no-lb-mincut` |
| One frontier + min cut | 0 | 1 | `--no-lb-backward` |
| Both frontiers | 1 | 0 | `--no-lb-mincut` |
| Both frontiers + min cut (default) | 1 | 1 | none |

The raw measurements are saved locally in
`out/profiles/pigeonhole-frontier-benchmarks.json`.

## Solver measurements

Enable `KATCH2_METRICS=1` to emit counters and nanosecond timings to stderr.
Measurements are opt-in; normal runs do not read clocks or query CaDiCaL's
statistics. Reproduce the four-configuration comparison with:

```sh
CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --release --bin nksynth
python3 benches/measure_frontiers.py
```

The script runs seven alternating repetitions per configuration with a 1,000
round cap. The results below were measured separately from builds and tests.
Every non-time counter was identical across the seven repetitions. Wall times
include instrumentation and are a separate series from the earlier timing table.
`out/profiles/pigeonhole-frontier-metrics.json` contains the raw results.

### Pigeonhole 3 (UNSAT)

| Measurement | One frontier | One frontier + min cut | Both frontiers | Both frontiers + min cut |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 1.185 s | 1.054 s | 0.915 s | 0.905 s |
| CEGIS rounds | 780 | 763 | 531 | 529 |
| SAT calls | 2,213 | 2,192 | 1,337 | 1,363 |
| Ackermann refinement rounds | 1,432 | 1,428 | 805 | 833 |
| Ackermann lemmas | 2,837 | 2,905 | 2,198 | 2,402 |
| Abstract clauses | 780 | 791 | 683 | 709 |
| CNF clauses submitted | 20,762 | 20,997 | 14,819 | 16,457 |
| SAT variables allocated | 9,495 | 9,605 | 7,013 | 7,734 |
| CaDiCaL conflicts | 25,349 | 25,086 | 44,671 | 44,786 |
| CaDiCaL decisions | 616,979 | 645,967 | 317,335 | 376,164 |
| CaDiCaL propagations | 2,873,056 | 3,059,595 | 2,265,400 | 2,554,123 |
| SAT search time | 799.5 ms | 684.8 ms | 729.6 ms | 724.7 ms |
| SAT model extraction time | 200.8 ms | 190.5 ms | 71.6 ms | 79.5 ms |
| Congruence checking time | 65.4 ms | 66.8 ms | 31.6 ms | 34.1 ms |
| Ackermann encoding time | 7.3 ms | 6.2 ms | 3.5 ms | 3.6 ms |
| Candidate construction time | 43.4 ms | 40.7 ms | 24.0 ms | 23.5 ms |
| Candidate checking + clause learning | 45.4 ms | 44.9 ms | 27.5 ms | 26.1 ms |

### Pigeonhole 4 (UNKNOWN)

| Measurement | One frontier | One frontier + min cut | Both frontiers | Both frontiers + min cut |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 1.177 s | 1.158 s | 0.683 s | 0.702 s |
| CEGIS rounds | 1,000 | 1,000 | 1,000 | 1,000 |
| SAT calls | 2,344 | 2,269 | 1,935 | 2,169 |
| Ackermann refinement rounds | 1,343 | 1,268 | 934 | 1,168 |
| Ackermann lemmas | 3,236 | 3,175 | 2,353 | 3,286 |
| Abstract clauses | 1,000 | 1,026 | 1,089 | 1,131 |
| CNF clauses submitted | 29,446 | 28,114 | 16,242 | 22,078 |
| SAT variables allocated | 10,466 | 10,109 | 6,459 | 8,814 |
| CaDiCaL conflicts | 4,435 | 4,164 | 2,996 | 4,338 |
| CaDiCaL decisions | 3,538,235 | 3,411,617 | 1,656,312 | 1,266,768 |
| CaDiCaL propagations | 6,316,638 | 6,067,405 | 2,997,053 | 3,021,254 |
| SAT search time | 709.8 ms | 708.7 ms | 350.6 ms | 321.7 ms |
| SAT model extraction time | 178.9 ms | 165.9 ms | 83.3 ms | 107.1 ms |
| Congruence checking time | 105.2 ms | 101.9 ms | 80.8 ms | 96.6 ms |
| Ackermann encoding time | 12.5 ms | 11.5 ms | 5.7 ms | 6.9 ms |
| Candidate construction time | 86.4 ms | 87.7 ms | 88.6 ms | 87.6 ms |
| Candidate checking + clause learning | 60.7 ms | 67.2 ms | 55.8 ms | 57.3 ms |

SAT model extraction includes CaDiCaL model reconstruction (`External::extend`)
and copying all allocated variable values into Rust. Candidate construction
includes example selection and learning the candidate BDDs. Checking + learning
includes automaton checks, counterexample reachability, cut construction,
deduplication, and asserting abstract clauses. Medians for individual phases
need not sum to the median wall time. The initial empty extraction is included
in native/SAT counters; SAT calls can exceed candidate extractions because of
lazy Ackermann refinement. CNF counts include clauses submitted by our wrapper,
excluding the initial constant unit and CaDiCaL's internally learned clauses.

For pigeonhole 4, adding the backward frontier without min-cut reduces SAT calls
by 17%, decisions by 53%, and CNF clauses by 45%, while retaining 1,000 CEGIS
rounds. Mean search time per SAT call also falls, from about 303 to 181 µs.
Model extraction more than halves, while candidate construction stays near
87–89 ms. The extra abstract clauses change the candidate sequence and reduce
refinement/encoding work; these measurements do not isolate propagation effects
on an identical SAT instance.

For pigeonhole 3, both frontiers reduce CEGIS rounds from 780 to 531 and SAT
calls from 2,213 to 1,337. Conflicts increase from 25,349 to 44,671, even though
decisions and total runtime fall: fewer calls and less model/extraction work
matter as well as SAT search. Min-cut with both frontiers lowers pigeonhole 4's
SAT search time further, but increases model and congruence overhead.

## Symbolic upper-bound comparison

A single SPP hole equal to the 64-field all-zero packet test (`x0 == 0; ...;
x63 == 0`) illustrates the benefit of symbolic negative evidence. Seven
alternating release CLI runs compared the previous binary with symbolic
upper-bound learning, with instrumentation enabled and a 1,000-round cap.
Every run reported SAT, and the new solution passed the separate checker.

| Measurement | Previous | Symbolic upper bound |
| --- | ---: | ---: |
| CEGIS rounds / SAT calls | 66 / 66 | 3 / 3 |
| Concrete abstract clauses | 65 | 1 |
| Symbolic forbidden regions | 0 | 1 |
| Checking, learning, extraction, and candidate installation | 25.4 ms | 1.4 ms |
| CLI wall time | 1.156 s | 1.134 s |

Timings are medians. The loop timing excludes its initial empty extraction;
work outside the loop dominates the wall time for this case. A subsequent
Samply profile attributes 99% of samples to `spp::reduce::to_expr_tree`, which
converts the solved SPP into a printable expression. The CLI currently does
this even when no output file is requested. This cost occurs after synthesis,
not during setup. The profile is saved locally as
`out/profiles/symbolic-upper64.samply.json.gz` with its `.syms.json` companion.
The old binary
needs a separate negative packet example for each field. The new binary learns
the entire complement of the test as negative evidence in one upper-bound
refinement. Its positive SAT example already lies outside the forbidden region,
so this case does not require lazy universal SAT instantiation; the exhaustive
solver tests cover that separately.

Local artifacts: `out/profiles/symbolic-upper64.nksynth`,
`out/profiles/symbolic-upper64.sol`, and
`out/profiles/symbolic-upper64-benchmark.json`.

## Entropy shortlist for expression conversion

The default SPP expression converter now uses
`SplitHeuristic::LookaheadTop { depth: 1, width: 5 }`: score every candidate
field by entropy, complete trees for the best five, and retain the smallest.
Partial selection keeps shortlist construction linear in the number of fields;
equal entropy scores are ordered by field number. The original unrestricted
`Lookahead(1)` remains available through `to_expr_tree` for comparisons.

Seven alternating release CLI runs on the same 64-field example above, without
instrumentation, measured:

| Expression search | Median CLI wall time |
| --- | ---: |
| Every first field | 1.097 s |
| Five lowest-entropy first fields | 0.048 s |

The resulting expressions were identical (761 bytes), and the new solution
passed the separate checker. This is about a 23x speedup for this case. Pruning
can produce larger expressions on other inputs; it preserves their semantics.
Round-trip tests exercise shortlist widths 1, 2, and 5 on six-field relations,
and verify that width 1 agrees with entropy and an unpruned shortlist recovers
exhaustive expression size. Raw times are saved locally in
`out/profiles/top-five-lookahead-benchmark.json`.

## Direct ENFA upper-bound checking

Compared profiling builds on `ex3_repair.nkpl` with
`--iteration-limit 5000 --output <solution-file>`. Both implementations report
UNKNOWN after 5,000 rounds. Output paths under `out/profiles/` preserve any
existing user solution file. The baseline binary was saved before changing
upper-bound checking; both use the same profiling settings.

Three alternating wall-time runs per implementation, with `KATCH2_METRICS=1`,
gave medians of **19.21 s before** and **9.36 s after** (about 2.05x faster).
Ranges were 18.95–19.40 s and 9.30–9.65 s, respectively. Every run recorded
exactly 5,000 CEGIS rounds and reported UNKNOWN. Raw timings and counters are
in `out/profiles/ex3-direct-comparison-timings.json`.

Separate Samply runs measured:

| Measurement | Epsilon-closure check | Direct ENFA check |
| --- | ---: | ---: |
| Profiled duration | 18.40 s | 9.65 s |
| Upper-bound check, inclusive samples | 69.75% (~12.8 s) | 32.82% (~3.2 s) |
| Epsilon closure, inclusive samples | 58.94% | 0.75% |
| Trace elaboration, inclusive samples | 10.36% | 0% |
| Candidate extraction, inclusive samples | 29.97% | 65.75% |

Inclusive percentages overlap and are taken from the active solver thread,
excluding the waiting CLI thread. The remaining epsilon-closure samples in
the new profile all belong to lower-bound checking. Approximate phase seconds
come from 1 ms sample counts. Different witness selection changes the candidate
sequence, so this is an end-to-end comparison, not identical candidate checks.
The known `sol3` from the input also passes the new separate checker (VALID).

The new checker is tested against the original closure-based product on 500
seeded random ENFAs with random visibility, including cycles and incomplete
upper-bound DFAs. Tests validate every returned concrete edge, its output,
and acceptance by the ENFA and rejection by the upper bound.

Profile artifacts are saved locally as
`out/profiles/ex3-epsilon-baseline.samply.json.gz` and
`out/profiles/ex3-direct-enfa.samply.json.gz`, with their `.syms.json`
companions. Sample summaries are in
`out/profiles/ex3-direct-comparison-profile.json`.

Reproduce the updated profile:

```sh
cargo build --profile profiling --bin nksynth
samply record --save-only --unstable-presymbolicate \
  --output out/profiles/ex3-direct-enfa.samply.json.gz \
  target/profiling/nksynth --iteration-limit 5000 \
  --output out/profiles/ex3-direct-enfa.sol ex3_repair.nkpl
samply load out/profiles/ex3-direct-enfa.samply.json.gz
```

## Notes

- The native learner uses incremental CaDiCaL through RustSAT. Criterion's
  confidence intervals capture within-run variance; for cross-run drift, take a second run — these benchmarks
  are low-variance (CIs typically <2%) and reproduce closely.
- `corpus` uses 20 samples; `pigeon` uses 10.
- All benchmark solves run under a 1000-round refinement cap (`CAP`), so a divergent
  instance reports `IterationLimit` instead of hanging.
- Tunables at the top of `synthesis.rs`: `FIELDS`, `CORPUS_SEED`, `CORPUS_SIZE`,
  `CAP`.
