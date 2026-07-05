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
