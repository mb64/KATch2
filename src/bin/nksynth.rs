//! `nksynth`: solve NetKAT synthesis (`.nksynth`) problems.
//!
//! Each file is parsed, lowered into a synthesis problem, and solved; the
//! verdict (`SAT` if a hole assignment exists, `UNSAT` otherwise) is printed
//! one line per file, in argument order.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use katch2::holes::parser::{desugar, parse_program};

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
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let full = cli.full && !cli.no_full;

    let mut exit = ExitCode::SUCCESS;
    for path in &cli.files {
        match solve_file(path, full) {
            Ok(true) => println!("SAT"),
            Ok(false) => println!("UNSAT"),
            Err(e) => {
                eprintln!("error: {}: {}", path.display(), e);
                exit = ExitCode::FAILURE;
            }
        }
    }
    exit
}

/// Solve a single `.nksynth` file. Returns `Ok(true)` for SAT, `Ok(false)` for
/// UNSAT, or `Err` with a human-readable message for I/O, parse, or desugar
/// failures.
fn solve_file(path: &Path, full: bool) -> Result<bool, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("could not read file: {e}"))?;
    let program = parse_program(&src).map_err(|e| format!("parse error: {}", e.message))?;
    let mut instance = desugar(&program).map_err(|e| format!("desugar error: {e}"))?;

    // A solver `Ok` means the holes were filled (SAT); the only error with an
    // unbounded iteration budget is infeasibility (UNSAT).
    let sat = if full {
        instance.solve_full().is_ok()
    } else {
        instance.solve().is_ok()
    };
    Ok(sat)
}
