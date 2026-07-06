//! `nksynth`: solve NetKAT synthesis (`.nksynth`) problems.
//!
//! Each file is parsed, lowered into a synthesis problem, and solved; the
//! verdict (`SAT` if a hole assignment exists, `UNSAT` if proven infeasible,
//! `UNKNOWN` if the iteration limit was hit first) is printed one line per
//! file, in argument order.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;

use clap::Parser;
use katch2::holes::cegis::CegisError;
use katch2::holes::parser::{desugar, parse_program};

/// Large topologies can produce deeply recursive NetKAT expressions during
/// automaton construction, which can overflow the default ~8MB main-thread
/// stack. Run the solve on a thread with a much bigger stack instead.
const STACK_SIZE: usize = 1 << 30; // 1 GiB

/// Solve NetKAT synthesis (`.nksynth`) problems, reporting SAT or UNSAT.
#[derive(Parser)]
#[command(name = "nksynth", version, about)]
struct Cli {
    /// `.nksynth` file(s) to solve.
    #[arg(required = true, value_name = "FILE")]
    files: Vec<PathBuf>,

    /// Use the full solver (holes may emit `dup`).
    #[arg(long)]
    full: bool,

    /// Use the dup-free solver (the default).
    #[arg(long = "no-full", conflicts_with = "full")]
    no_full: bool,

    /// Give up after this many CEGIS refinement rounds, reporting `UNKNOWN`
    /// instead of looping until solved or proven infeasible.
    #[arg(long, value_name = "N")]
    iteration_limit: Option<usize>,

    /// Disable the lower-bound min-cut optimization (emit the frontier clause
    /// instead of a min cut).
    #[arg(long)]
    no_lb_mincut: bool,

    /// Disable collating SMT membership disjuncts by shared input/output set.
    #[arg(long)]
    no_clause_merge: bool,

    /// Disable shrinking the SMT model to a greedy set cover of examples
    /// before passive learning.
    #[arg(long)]
    no_example_cover: bool,
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
    if cli.no_clause_merge {
        katch2::flags::set_clause_merge(false);
    }
    if cli.no_example_cover {
        katch2::flags::set_example_cover(false);
    }

    let mut exit = ExitCode::SUCCESS;
    for path in &cli.files {
        match solve_file(path, full, cli.iteration_limit) {
            Ok(verdict) => println!("{verdict}"),
            Err(e) => {
                eprintln!("error: {}: {}", path.display(), e);
                exit = ExitCode::FAILURE;
            }
        }
    }
    exit
}

/// Solve a single `.nksynth` file, returning the verdict string (`SAT`,
/// `UNSAT`, or `UNKNOWN`), or `Err` with a human-readable message for I/O,
/// parse, or desugar failures.
fn solve_file(path: &Path, full: bool, max_iters: Option<usize>) -> Result<&'static str, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("could not read file: {e}"))?;
    let program = parse_program(&src).map_err(|e| format!("parse error: {}", e.message))?;
    let mut instance = desugar(&program).map_err(|e| format!("desugar error: {e}"))?;

    let outcome = if full {
        instance.solve_full(max_iters).map(|_| ())
    } else {
        instance.solve(max_iters).map(|_| ())
    };
    Ok(match outcome {
        Ok(()) => "SAT",
        Err(CegisError::Infeasible) => "UNSAT",
        Err(CegisError::IterationLimit) => "UNKNOWN",
    })
}
