//! End-to-end synthesis benchmarks.
//!
//! These time the full CEGIS loop ([`ProblemInstance::solve`] /
//! [`ProblemInstance::solve_full`]), so they measure the cost of everything the
//! optimizations touch: the lower-bound min-cut clause builder (`holes::cegis`)
//! and the SMT clause simplification (`holes::smt`). Both shrink the formulas
//! handed to Z3, so the signal shows up as reduced wall-clock solve time.
//!
//! The headline workload is a **synthetic corpus**: a seeded, deterministic set
//! of randomly-generated *satisfiable-by-construction* instances (same
//! generator as the crate's round-trip fuzzers — random per-hole programs
//! substituted into a random hole-bearing template). Because the RNG is seeded
//! with a fixed value and `rand`'s `StdRng` is reproducible within a major
//! version, the corpus is identical run-to-run and branch-to-branch, so it is
//! directly comparable while still being a broad, representative sample rather
//! than two hand-picked cases.
//!
//! Two targeted pigeonhole instances (`pigeon*`) round out the suite with a
//! deterministic, refinement-heavy UNSAT workload — the case that most stresses
//! the lower-bound clause machinery.
//!
//! Each optimization is a Cargo feature (`lb_mincut`, `clause_merge`, both on
//! by default), so all four on/off combinations build from one tree — see
//! `benches/README.md` and the `Makefile` (`make bench-all`) for the A/B.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use katch2::expr::{Exp, Expr};
use katch2::holes::aut::expr_to_dfa;
use katch2::holes::nk_with_holes::{Expr as HExpr, Hole};
use katch2::holes::problem::{Constraint, ProblemInstance};
use katch2::spp::SPPstore;

/// Refinement-round cap, so a diverging instance can't hang the benchmark.
const CAP: Option<usize> = Some(64);
/// Packet width for the corpus. Small keeps the SPPs and DFAs tiny.
const FIELDS: u32 = 2;
/// Fixed seed → identical corpus every run and on every branch.
const CORPUS_SEED: u64 = 0xCA7_C0FFEE;
/// Number of random instances per corpus.
const CORPUS_SIZE: usize = 40;

// ────────────────────────────────────────────────────────────────────────────
// Seeded random instance generation
//
// Ported verbatim (modulo a threaded `StdRng`) from the round-trip fuzzers in
// `src/holes/mod.rs`, so the corpus matches the distribution the solver is
// already tested against. Keeping the generator *here* (not in the library)
// means the baseline branch gets the exact same code when these bench files are
// copied over, so the corpus stays identical across branches.
// ────────────────────────────────────────────────────────────────────────────

/// A reusable recipe for one instance; cheap to clone, rebuilt into a fresh
/// (store-owning) [`ProblemInstance`] per measured solve.
#[derive(Clone)]
struct Recipe {
    hole_expr: HExpr,
    holes: Vec<Hole>,
    target: Exp,
}

/// A random dup-free expression (a single-SPP candidate for `solve`).
fn random_dup_free(rng: &mut StdRng, depth: u32) -> Exp {
    if depth == 0 || rng.random::<f64>() < 0.4 {
        return match rng.random_range(0..4u32) {
            0 => Expr::one(),
            1 => Expr::zero(),
            2 => Expr::test(rng.random_range(0..FIELDS), rng.random()),
            _ => Expr::assign(rng.random_range(0..FIELDS), rng.random()),
        };
    }
    match rng.random_range(0..3u32) {
        0 => Expr::union(
            random_dup_free(rng, depth - 1),
            random_dup_free(rng, depth - 1),
        ),
        1 => Expr::sequence(
            random_dup_free(rng, depth - 1),
            random_dup_free(rng, depth - 1),
        ),
        _ => Expr::star(random_dup_free(rng, depth - 1)),
    }
}

/// A random expression that may contain `dup` (a dup-ful candidate for
/// `solve_full`).
fn random_maybe_dupful(rng: &mut StdRng, depth: u32) -> Exp {
    if depth == 0 || rng.random::<f64>() < 0.4 {
        return match rng.random_range(0..5u32) {
            0 => Expr::one(),
            1 => Expr::zero(),
            2 => Expr::test(rng.random_range(0..FIELDS), rng.random()),
            3 => Expr::assign(rng.random_range(0..FIELDS), rng.random()),
            _ => Expr::dup(),
        };
    }
    match rng.random_range(0..3u32) {
        0 => Expr::union(
            random_maybe_dupful(rng, depth - 1),
            random_maybe_dupful(rng, depth - 1),
        ),
        1 => Expr::sequence(
            random_maybe_dupful(rng, depth - 1),
            random_maybe_dupful(rng, depth - 1),
        ),
        _ => Expr::star(random_maybe_dupful(rng, depth - 1)),
    }
}

/// Build a random hole-bearing expression and, in lock-step, the concrete
/// target it becomes once each `Hole(i)` is replaced by `insts[i]`.
fn random_template(
    rng: &mut StdRng,
    depth: u32,
    insts: &[Exp],
    used: &mut BTreeSet<Hole>,
) -> (HExpr, Exp) {
    if depth == 0 || rng.random::<f64>() < 0.4 {
        if rng.random::<f64>() < 0.85 {
            let i = rng.random_range(0..insts.len());
            used.insert(Hole(i as u32));
            return (HExpr::hole(Hole(i as u32)), insts[i].clone());
        }
        return (HExpr::dup(), Expr::dup());
    }
    match rng.random_range(0..3u32) {
        0 => {
            let (a1, a2) = random_template(rng, depth - 1, insts, used);
            let (b1, b2) = random_template(rng, depth - 1, insts, used);
            (HExpr::union(a1, b1), Expr::union(a2, b2))
        }
        1 => {
            let (a1, a2) = random_template(rng, depth - 1, insts, used);
            let (b1, b2) = random_template(rng, depth - 1, insts, used);
            (HExpr::sequence(a1, b1), Expr::sequence(a2, b2))
        }
        _ => {
            let (a1, a2) = random_template(rng, depth - 1, insts, used);
            (HExpr::star(a1), Expr::star(a2))
        }
    }
}

/// One satisfiable-by-construction instance.
fn random_instance(
    rng: &mut StdRng,
    template_depth: u32,
    inst_depth: u32,
    inst_gen: fn(&mut StdRng, u32) -> Exp,
) -> Recipe {
    let num_holes = 1 + rng.random_range(0..3usize); // 1..=3 holes
    let insts: Vec<Exp> = (0..num_holes).map(|_| inst_gen(rng, inst_depth)).collect();
    let mut used = BTreeSet::new();
    let (hole_expr, target) = random_template(rng, template_depth, &insts, &mut used);
    Recipe {
        hole_expr,
        holes: used.into_iter().collect(),
        target,
    }
}

/// A deterministic corpus of `CORPUS_SIZE` instances.
fn corpus(inst_gen: fn(&mut StdRng, u32) -> Exp) -> Vec<Recipe> {
    let mut rng = StdRng::seed_from_u64(CORPUS_SEED);
    (0..CORPUS_SIZE)
        .map(|_| random_instance(&mut rng, 3, 2, inst_gen))
        .collect()
}

/// Materialize a recipe into a fresh, unsolved problem (`hole_expr == target`).
fn build(r: &Recipe) -> ProblemInstance {
    let mut store = SPPstore::new(FIELDS);
    let target_dfa = expr_to_dfa(&r.target, &mut store);
    let constraints = vec![Constraint::equality(&mut store, &r.hole_expr, target_dfa)];
    ProblemInstance {
        store,
        holes: r.holes.clone(),
        constraints,
    }
}

/// Pigeonhole UNSAT: `Hole(0) ; (⋁ x_i==1) ; Hole(1) == 1` over `n`-bit packets.
/// `Hole(0)` would have to inject 2^`n` pigeons into 2^`n`−1 holes — impossible,
/// but the learner needs many lower-bound refinement rounds to prove it.
fn pigeon(n: u32) -> ProblemInstance {
    let mut store = SPPstore::new(n);
    let target_dfa = expr_to_dfa(&Expr::one(), &mut store);
    let mut spp = store.zero;
    for i in 0..n {
        let spp_i = store.test(i, true);
        spp = store.union(spp, spp_i);
    }
    let hole_expr = HExpr::sequence(
        HExpr::hole(Hole(0)),
        HExpr::sequence(HExpr::spp(spp), HExpr::hole(Hole(1))),
    );
    let constraints = vec![Constraint::equality(&mut store, &hole_expr, target_dfa)];
    ProblemInstance {
        store,
        holes: vec![Hole(0), Hole(1)],
        constraints,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Benchmarks
// ────────────────────────────────────────────────────────────────────────────

fn bench_corpus(c: &mut Criterion) {
    let spp_corpus = corpus(random_dup_free);
    let full_corpus = corpus(random_maybe_dupful);

    let mut g = c.benchmark_group("corpus");
    g.sample_size(20).measurement_time(Duration::from_secs(20));

    // Aggregate: total time to solve the whole corpus once (dup-free / `solve`).
    g.bench_function("spp_dup_free", |b| {
        b.iter_batched(
            || spp_corpus.iter().map(build).collect::<Vec<_>>(),
            |insts| {
                for mut pi in insts {
                    black_box(pi.solve(CAP).ok());
                }
            },
            BatchSize::SmallInput,
        )
    });

    // Aggregate: whole corpus once with the full (dup-ful) solver.
    g.bench_function("full_dupful", |b| {
        b.iter_batched(
            || full_corpus.iter().map(build).collect::<Vec<_>>(),
            |insts| {
                for mut pi in insts {
                    black_box(pi.solve_full(CAP).ok());
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

fn bench_pigeon(c: &mut Criterion) {
    let mut g = c.benchmark_group("pigeon");
    g.sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(15));

    g.bench_function("pigeon2_unsat", |b| {
        b.iter_batched(
            || pigeon(2),
            |mut pi| black_box(pi.solve(CAP).err()),
            BatchSize::SmallInput,
        )
    });

    g.bench_function("pigeon3_unsat", |b| {
        b.iter_batched(
            || pigeon(3),
            |mut pi| black_box(pi.solve(CAP).err()),
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

criterion_group!(benches, bench_corpus, bench_pigeon);
criterion_main!(benches);
