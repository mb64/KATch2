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
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use katch2::expr::Exp;
use katch2::holes::cegis::CegisError;
use katch2::holes::minimize;
use katch2::holes::parser::{Program, Stmt, desugar, fill_holes, parse_program};
use katch2::printer::pretty;

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
        match solve_file(path, full, cli.iteration_limit) {
            Ok((verdict, solution)) => {
                println!("{verdict}");
                if let (Some(out_path), Some(solution)) = (&cli.output, solution)
                    && let Err(e) = fs::write(out_path, solution)
                {
                    eprintln!("error: {}: could not write file: {e}", out_path.display());
                    exit = ExitCode::FAILURE;
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
) -> Result<(&'static str, Option<String>), String> {
    let program = read_program(path)?;
    let mut instance = desugar(&program).map_err(|e| format!("desugar error: {e}"))?;

    // Each hole's solution, as an expression, in `instance.holes` order
    let outcome: Result<Vec<Exp>, CegisError> = if full {
        instance.solve_full(max_iters).map(|dfas| {
            let holes = instance.holes.clone();
            holes
                .iter()
                .map(|h| minimize::to_expr(&dfas[h], &mut instance.store))
                .collect()
        })
    } else {
        instance.solve(max_iters).map(|spps| {
            let holes = instance.holes.clone();
            holes
                .iter()
                .map(|h| instance.store.to_expr(spps[h]))
                .collect()
        })
    };
    let solution = match outcome {
        Ok(exprs) => exprs,
        Err(CegisError::Infeasible) => return Ok(("UNSAT", None)),
        Err(CegisError::IterationLimit) => return Ok(("UNKNOWN", None)),
    };

    // `desugar` numbers the holes in declaration order
    let names = program.statements.iter().filter_map(|stmt| match stmt {
        Stmt::Hole(name) => Some(name),
        _ => None,
    });
    let text = names
        .zip(&solution)
        .map(|(name, expr)| format!("def {name} = {}\n", pretty(expr)))
        .collect();
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
    let program = read_program(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let solution =
        read_program(solution_path).map_err(|e| format!("{}: {e}", solution_path.display()))?;
    let filled =
        fill_holes(&program, &solution).map_err(|e| format!("{}: {e}", solution_path.display()))?;
    let mut instance =
        desugar(&filled).map_err(|e| format!("{}: desugar error: {e}", path.display()))?;
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
