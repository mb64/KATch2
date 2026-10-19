//! `nksynth`: solve NetKAT synthesis (`.nksynth`) problems.
//!
//! Each file is parsed, lowered into a synthesis problem, and solved; the
//! verdict (`SAT` if a hole assignment exists, `UNSAT` if proven infeasible,
//! `UNKNOWN` if the iteration limit was hit first) is printed one line per
//! file, in argument order.
//!
//! With `--output FILE`, the solution (if any) is also written to `FILE`, one `def hole = expr` line
//! per hole.
//!
//! With `--check FILE`, nothing is synthesized: instead the holes are filled in with the definitions
//! in `FILE` (in the format `--output` writes), and the result is checked against the assertions,
//! printing `VALID` or `INVALID`.

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;

use clap::Parser;
use katch2::expr::Exp;
use katch2::holes::cegis::CegisError;
use katch2::holes::metrics::Metrics;
use katch2::holes::minimize;
use katch2::holes::parser::{Program, Stmt, desugar, fill_holes, parse_program};
use katch2::printer::pretty;
use rand::RngExt;

/// Large topologies can produce deeply recursive NetKAT expressions during
/// automaton construction, which can overflow the default ~8MB main-thread
/// stack. Run the solve on a thread with a much bigger stack instead.
const STACK_SIZE: usize = 1 << 30; // 1 GiB

/// Solve NetKAT synthesis (`.nksynth`) problems, reporting SAT or UNSAT.
#[derive(Parser)]
#[command(name = "nksynth", version = env!("NKSYNTH_VERSION"), about)]
struct Cli {
    /// `.nksynth` file(s) to solve.
    #[arg(required = true, value_name = "FILE")]
    files: Vec<PathBuf>,

    /// Seed for the shared RNG used by all synthesis random choices.
    /// Omit to choose a seed from system entropy; the chosen seed is logged.
    #[arg(long, value_name = "SEED")]
    seed: Option<u64>,

    /// Learn only concrete upper-bound clauses; skip symbolic restrictions.
    #[arg(long)]
    no_symbolic_upper: bool,

    /// Preserve proven hole bounds across randomized restarts.
    #[arg(long)]
    save_restart_bounds: bool,

    /// Initial CEGIS iteration cutoff, multiplied by the Luby restart sequence.
    #[arg(long, default_value = "150", value_name = "N")]
    restart_baseline: NonZeroUsize,

    /// Write one CSV row per candidate guess and SPP slot (single input only).
    #[arg(long, value_name = "FILE")]
    candidate_stats: Option<PathBuf>,

    /// Emit cumulative KATCH2_METRICS snapshots every N rounds (0 disables).
    /// Requires KATCH2_METRICS=1.
    #[arg(long, default_value_t = 256, value_name = "N")]
    stats_interval: usize,

    /// Use the full solver (holes may emit `dup`).
    #[arg(long)]
    full: bool,

    /// Use the dup-free solver (the default).
    #[arg(long = "no-full", conflicts_with = "full")]
    no_full: bool,

    /// Give up after this many total CEGIS rounds across restarts, reporting `UNKNOWN`
    /// instead of looping until solved or proven infeasible.
    #[arg(long, value_name = "N")]
    iteration_limit: Option<usize>,

    /// Disable the additional lower-bound min-cut clause.
    #[arg(long)]
    no_lb_mincut: bool,

    /// Disable the additional backward-frontier clause (keep the forward frontier).
    #[arg(long)]
    no_lb_backward: bool,

    /// Disable collating SMT membership disjuncts by shared input/output set.
    #[arg(long)]
    no_clause_merge: bool,

    /// Disable shrinking the SMT model to a greedy set cover of examples
    /// before passive learning.
    #[arg(long)]
    no_example_cover: bool,

    /// Write the solution to this file, one `def hole = expr` line per hole.
    /// Only allowed with a single input file. Nothing is written unless the
    /// verdict is `SAT`.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Instead of synthesizing, check the solution in this file (in the format
    /// `--output` writes), printing `VALID` or `INVALID`. Only allowed with a
    /// single input file.
    #[arg(long, value_name = "FILE", conflicts_with = "output")]
    check: Option<PathBuf>,
}

fn main() -> ExitCode {
    thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .expect("failed to spawn solver thread")
        .join()
        .expect("solver thread panicked")
}

fn run() -> ExitCode {
    let cli = Cli::parse();
    let full = cli.full && !cli.no_full;

    if cli.no_lb_mincut {
        katch2::flags::set_lb_mincut(false);
    }
    if cli.no_lb_backward {
        katch2::flags::set_lb_backward(false);
    }
    if cli.no_clause_merge {
        katch2::flags::set_clause_merge(false);
    }
    if cli.no_example_cover {
        katch2::flags::set_example_cover(false);
    }

    if cli.files.len() > 1 && cli.candidate_stats.is_some() {
        eprintln!("error: --candidate-stats requires a single input file");
        return ExitCode::FAILURE;
    }
    if cli.files.len() > 1 {
        if cli.output.is_some() {
            eprintln!("error: --output can only be used with a single input file");
            return ExitCode::FAILURE;
        }
        if cli.check.is_some() {
            eprintln!("error: --check can only be used with a single input file");
            return ExitCode::FAILURE;
        }
    }

    if let Some(solution_path) = &cli.check {
        return match check_file(&cli.files[0], solution_path, cli.iteration_limit) {
            Ok(verdict) => {
                println!("{verdict}");
                if verdict == "VALID" {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let mut exit = ExitCode::SUCCESS;
    for path in &cli.files {
        let seed = cli.seed.unwrap_or_else(|| rand::rng().random());
        let mut config = katch2::experiment::Config::new(seed);
        config.symbolic_upper = !cli.no_symbolic_upper;
        config.save_restart_bounds = cli.save_restart_bounds;
        config.restart_baseline = cli.restart_baseline;
        config.candidate_stats = cli.candidate_stats.clone();
        config.stats_interval = cli.stats_interval;
        katch2::experiment::configure(config);
        eprintln!(
            "KATCH2_RANDOM seed={} symbolic_upper={} restart_baseline={}",
            seed, !cli.no_symbolic_upper, cli.restart_baseline
        );
        let mut timings = Metrics::new("cli");
        match solve_file(path, full, cli.iteration_limit, &mut timings) {
            Ok((verdict, solution)) => {
                println!("{verdict}");
                if let (Some(out_path), Some(solution)) = (&cli.output, solution) {
                    let started = timings.start();
                    let written = fs::write(out_path, solution);
                    timings.elapsed("write_ns", started);
                    if let Err(e) = written {
                        eprintln!("error: {}: could not write file: {e}", out_path.display());
                        exit = ExitCode::FAILURE;
                    }
                }
            }
            Err(e) => {
                eprintln!("error: {}: {}", path.display(), e);
                exit = ExitCode::FAILURE;
            }
        }
    }
    exit
}

/// Solve a single `.nksynth` file, returning the verdict string (`SAT`,
/// `UNSAT`, or `UNKNOWN`) and, if `SAT`, the solution as `hole = expr` lines.
/// Returns `Err` with a human-readable message for I/O, parse, or desugar
/// failures.
fn solve_file(
    path: &Path,
    full: bool,
    max_iters: Option<usize>,
    timings: &mut Metrics,
) -> Result<(&'static str, Option<String>), String> {
    let started = timings.start();
    let program = read_program(path)?;
    timings.elapsed("read_parse_ns", started);
    let started = timings.start();
    let mut instance = desugar(&program).map_err(|e| format!("desugar error: {e}"))?;
    timings.elapsed("desugar_ns", started);
    // The compiled problem owns everything needed for synthesis. Keep only the
    // small list of names needed to print solutions, not the original AST.
    let started = timings.start();
    let names: Vec<_> = program
        .statements
        .into_iter()
        .filter_map(|stmt| match stmt {
            Stmt::Hole(name) => Some(name),
            _ => None,
        })
        .collect();
    timings.elapsed("drop_syntax_ns", started);

    // Each hole's solution, as an expression, in `instance.holes` order
    let outcome: Result<Vec<Exp>, CegisError> = if full {
        let started = timings.start();
        let result = instance.solve_full(max_iters);
        timings.elapsed("solve_ns", started);
        timings.checkpoint();
        let started = timings.start();
        let converted = result.map(|dfas| {
            let holes = instance.holes.clone();
            holes
                .iter()
                .map(|h| minimize::to_expr(&dfas[h], &mut instance.store))
                .collect()
        });
        timings.elapsed("to_expr_ns", started);
        converted
    } else {
        let started = timings.start();
        let result = instance.solve(max_iters);
        timings.elapsed("solve_ns", started);
        timings.checkpoint();
        let started = timings.start();
        let converted = result.map(|spps| {
            let holes = instance.holes.clone();
            holes
                .iter()
                .map(|h| instance.store.to_expr(spps[h]))
                .collect()
        });
        timings.elapsed("to_expr_ns", started);
        converted
    };
    timings.checkpoint();
    let solution = match outcome {
        Ok(exprs) => exprs,
        Err(CegisError::Infeasible) => return Ok(("UNSAT", None)),
        Err(CegisError::IterationLimit) => return Ok(("UNKNOWN", None)),
    };

    // Names and compiled holes retain declaration order.
    let started = timings.start();
    let text = names
        .into_iter()
        .zip(&solution)
        .map(|(name, expr)| format!("def {name} = {}\n", pretty(expr)))
        .collect();
    timings.elapsed("format_ns", started);
    Ok(("SAT", Some(text)))
}

/// Check the solution in `solution_path` against the problem in `path`, returning `VALID` or
/// `INVALID` (or `UNKNOWN`, if the iteration limit is somehow hit), or `Err` with a human-readable
/// message for I/O, parse, desugar, or solution-format failures.
///
/// Filling in the holes gives a hole-free problem, which the synthesizer then just verifies: it is
/// solvable exactly when every assertion holds.
fn check_file(
    path: &Path,
    solution_path: &Path,
    max_iters: Option<usize>,
) -> Result<&'static str, String> {
    let mut instance = {
        let program = read_program(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let solution =
            read_program(solution_path).map_err(|e| format!("{}: {e}", solution_path.display()))?;
        let filled = fill_holes(&program, &solution)
            .map_err(|e| format!("{}: {e}", solution_path.display()))?;
        desugar(&filled).map_err(|e| format!("{}: desugar error: {e}", path.display()))?
    };
    debug_assert!(instance.holes.is_empty());

    Ok(match instance.solve(max_iters) {
        Ok(_) => "VALID",
        Err(CegisError::Infeasible) => "INVALID",
        Err(CegisError::IterationLimit) => "UNKNOWN",
    })
}

/// Read and parse a `.nksynth` file.
fn read_program(path: &Path) -> Result<Program, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("could not read file: {e}"))?;
    parse_program(&src).map_err(|e| format!("parse error: {}", e.message))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_experiment_controls() {
        let cli = Cli::try_parse_from([
            "nksynth",
            "--seed",
            "18446744073709551615",
            "--no-symbolic-upper",
            "--candidate-stats",
            "guesses.csv",
            "--stats-interval",
            "32",
            "--restart-baseline",
            "140",
            "input.nksynth",
        ])
        .unwrap();
        assert_eq!(cli.seed, Some(u64::MAX));
        assert!(cli.no_symbolic_upper);
        assert_eq!(cli.stats_interval, 32);
        assert_eq!(cli.restart_baseline.get(), 140);
        assert_eq!(
            Cli::try_parse_from(["nksynth", "input.nksynth"])
                .unwrap()
                .restart_baseline
                .get(),
            150
        );
        assert!(
            Cli::try_parse_from(["nksynth", "--restart-baseline", "0", "input.nksynth"]).is_err()
        );
        assert!(Cli::try_parse_from(["nksynth", "--seed", "-1", "input.nksynth"]).is_err());
    }
}
