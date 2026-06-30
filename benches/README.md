# Synthesis benchmarks

`benches/synthesis.rs` times the end-to-end CEGIS loop
(`ProblemInstance::solve` / `solve_full`). It exists to measure the two
clause-shrinking optimizations:

- the lower-bound **min-cut clause** builder (`src/holes/cegis.rs`), and
- the **SMT clause simplification / merge** (`src/holes/smt.rs`).

## Workload

- **`corpus/spp_dup_free`**, **`corpus/full_dupful`** — the headline metric:
  total time to solve a *seeded, deterministic corpus* of `CORPUS_SIZE` random
  satisfiable-by-construction instances (the same generator as the crate's
  round-trip fuzzers, ported with a threaded `StdRng`). The fixed seed +
  `rand`'s reproducible `StdRng` make the corpus byte-identical run-to-run and
  branch-to-branch, so it is a broad sample that is still directly comparable.
  `spp_dup_free` uses the dup-free `solve`; `full_dupful` uses `solve_full`.
- **`pigeon/pigeon2_unsat`**, **`pigeon/pigeon3_unsat`** — a deterministic,
  refinement-heavy UNSAT workload (pigeonhole). This most directly stresses the
  lower-bound clause machinery.

## Running

```sh
cargo bench --bench synthesis            # all groups
cargo bench --bench synthesis -- corpus  # filter by name substring
```

## Did the optimizations help? (A/B via feature flags)

Each optimization is a Cargo feature (both on by default), so all four on/off
combinations build from one tree — no branch-switching needed:

| feature | optimization |
| --- | --- |
| `lb_mincut` | lower-bound clause via min cut (vs. the frontier clause) |
| `clause_merge` | collate SMT membership disjuncts before solving |

The `Makefile` runs the synthesis bench under every combo, saving a Criterion
baseline per combo so they auto-compare:

```sh
make bench-all          # baselines: default, none, mincut, merge
# then compare any two saved baselines (negative % = first is faster):
cargo bench --bench synthesis -- --load-baseline mincut --baseline none
```

Or drive a single combo directly:

```sh
cargo bench --bench synthesis                         # both on (default)
cargo bench --bench synthesis --no-default-features   # both off (baseline)
cargo bench --bench synthesis --no-default-features --features lb_mincut
```

(The corpus is seeded, so it is identical across combos and the medians are
directly comparable.)

## Notes

- Solve time is dominated by Z3. Criterion's confidence intervals capture
  within-run variance; for cross-run drift, take a second run — these benchmarks
  are low-variance (CIs typically <2%) and reproduce closely.
- `corpus` uses 20 samples; `pigeon` uses 10 (each solve is ~ms).
- All solves run under a 64-round refinement cap (`CAP`), so a divergent
  instance reports `IterationLimit` instead of hanging.
- Tunables at the top of `synthesis.rs`: `FIELDS`, `CORPUS_SEED`, `CORPUS_SIZE`,
  `CAP`.
