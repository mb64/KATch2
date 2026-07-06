# Synthesis benchmarks

`benches/synthesis.rs` times the end-to-end CEGIS loop
(`ProblemInstance::solve` / `solve_full`). It exists to measure the three
solver-shrinking optimizations:

- the lower-bound **min-cut clause** builder (`src/holes/cegis.rs`),
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
- **`pigeon/pigeon2_unsat`**, **`pigeon/pigeon3_unsat`** — a deterministic,
  refinement-heavy UNSAT workload (pigeonhole). This most directly stresses the
  lower-bound clause machinery.

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
| `KATCH2_LB_MINCUT` | lower-bound clause via min cut (vs. the frontier clause) |
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
KATCH2_LB_MINCUT=0 KATCH2_CLAUSE_MERGE=0 KATCH2_EXAMPLE_COVER=0 \
  cargo bench --bench synthesis                    # all off (baseline)
KATCH2_CLAUSE_MERGE=0 KATCH2_EXAMPLE_COVER=0 \
  cargo bench --bench synthesis                    # min cut only
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
