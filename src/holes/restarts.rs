//! Randomized CEGIS restarts, retaining the compiled problem between attempts.
//!
//! Each attempt owns a clone of the initial SPP store. Failed attempts are
//! dropped in full; on success the caller receives the winning store, so the
//! returned candidates' handles remain valid. The shared RNG keeps advancing.

use crate::hash::HashMap;
use std::num::NonZeroUsize;

use super::{
    aut::ExplicitDFA,
    candidate::Candidate,
    cegis::{self, CegisError},
    problem::{Constraint, Hole},
    smt::SmtLearner,
};
use crate::spp::SPPstore;

/// The reluctant-doubling (Luby) sequence: `1, 1, 2, 1, 1, 2, 4, ...`.
pub fn reluctant_doubling() -> impl Iterator<Item = usize> {
    let (mut index, mut value) = (1usize, 1usize);
    std::iter::from_fn(move || {
        let result = value;
        if index.isolate_lowest_one() == value {
            index = index.saturating_add(1);
            value = 1;
        } else {
            value = value.saturating_mul(2);
        }
        Some(result)
    })
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Multiply each sequence element by this many CEGIS iterations.
    pub initial_cutoff: NonZeroUsize,
    /// Carry proven SPP lower/upper bounds across attempts, compacting their DAGs.
    pub save_bounds: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            initial_cutoff: NonZeroUsize::new(150).unwrap(),
            save_bounds: false,
        }
    }
}

/// Run CEGIS with fresh learners and stores until solved or proven infeasible.
///
/// `max_iters` is a total iteration budget across all attempts. The last
/// attempt's cutoff is shortened to fit the remaining budget. `None` permits
/// unlimited restarts. Only [`CegisError::IterationLimit`] triggers a restart.
/// Neither constraints nor the reference DFA are recompiled between attempts.
/// The template's operation caches are reset before cloning. On error its arenas
/// are unchanged; on success it becomes the winning store.
pub fn run<'a, C: Candidate<'a>, L: SmtLearner<'a>>(
    constraints: &[Constraint],
    holes: &[Hole],
    reference_dfa: &'a ExplicitDFA,
    store: &mut SPPstore,
    max_iters: Option<usize>,
    config: Config,
) -> Result<HashMap<Hole, C>, CegisError> {
    // Desugaring may leave large caches of intermediate computations. Do not
    // retain or copy those caches into every restart's working store.
    store.memo_reset();
    let mut remaining = max_iters;
    let mut completed = 0usize;
    let mut bounds = Vec::new();
    let mut carried_store = None;
    for (attempt, multiple) in reluctant_doubling().enumerate() {
        let cutoff = config.initial_cutoff.get().saturating_mul(multiple);
        let cutoff = remaining.map_or(cutoff, |remaining| cutoff.min(remaining));
        eprintln!(
            "KATCH2_RESTART attempt={} cutoff={} completed_iterations={}",
            attempt + 1,
            cutoff,
            completed
        );
        let mut attempt_store = carried_store.take().unwrap_or_else(|| store.clone());
        match cegis::run_with_bounds::<C, L>(
            constraints,
            holes,
            reference_dfa,
            &mut attempt_store,
            Some(cutoff),
            config.save_bounds.then_some(&mut bounds),
        ) {
            Ok(candidates) => {
                *store = attempt_store;
                return Ok(candidates);
            }
            Err(CegisError::Infeasible) => return Err(CegisError::Infeasible),
            Err(CegisError::IterationLimit) => {
                completed = completed.saturating_add(cutoff);
                if let Some(remaining) = &mut remaining {
                    *remaining -= cutoff;
                    if *remaining == 0 {
                        return Err(CegisError::IterationLimit);
                    }
                }
                if config.save_bounds {
                    // Start from the immutable compiled template every time,
                    // importing only current bound roots, not past garbage.
                    let mut compact = store.clone();
                    let roots: Vec<_> = bounds.iter().flat_map(|&(l, u)| [l, u]).collect();
                    let roots = compact.import(&attempt_store, &roots);
                    bounds = roots
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&[l, u]| (l, u))
                        .collect();
                    carried_store = Some(compact);
                }
                drop(attempt_store);
            }
        }
    }
    unreachable!("the restart sequence is infinite")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        holes::{
            parser::{desugar, parse_program},
            smt::Native,
        },
        spp::SPP,
    };

    #[test]
    fn sequence() {
        assert_eq!(
            reluctant_doubling().take(31).collect::<Vec<_>>(),
            [
                1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2,
                4, 8, 16
            ]
        );
    }

    #[test]
    fn restarts_preserve_the_template_and_return_valid_handles() {
        crate::experiment::with_seed(17, || {
            let csv_path = std::env::temp_dir().join(format!(
                "nksynth-restarts-{}-{}.csv",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut experiment = crate::experiment::Config::new(17);
            experiment.candidate_stats = Some(csv_path.clone());
            crate::experiment::configure(experiment);
            let mut problem =
                desugar(&parse_program("hole h\nassert h == (x0 := 1)").unwrap()).unwrap();
            let reference = SPP::make_reference_dfa(&mut problem.store, &problem.constraints);
            let config = Config {
                initial_cutoff: NonZeroUsize::new(1).unwrap(),
                ..Config::default()
            };
            // Operation caches may be reset, but template handles must remain valid.
            let roots = [
                problem.store.zero,
                problem.store.one,
                problem.store.top,
                problem.store.assign(0, true),
            ];
            let nodes = roots.map(|root| problem.store.get(root));
            // Two one-round attempts cannot refine and then verify a solution.
            assert_eq!(
                run::<SPP, Native>(
                    &problem.constraints,
                    &problem.holes,
                    &reference,
                    &mut problem.store,
                    Some(2),
                    config
                ),
                Err(CegisError::IterationLimit)
            );
            assert_eq!(roots.map(|root| problem.store.get(root)), nodes);

            let solution = run::<SPP, Native>(
                &problem.constraints,
                &problem.holes,
                &reference,
                &mut problem.store,
                Some(100),
                config,
            )
            .unwrap();
            let candidate = solution[&Hole(0)];
            for input in [false, true] {
                for output in [false, true] {
                    assert_eq!(
                        problem.store.accepts(candidate, &[input], &[output]),
                        output
                    );
                }
            }
            let csv = std::fs::read_to_string(&csv_path).unwrap();
            std::fs::remove_file(csv_path).unwrap();
            assert_eq!(
                csv.lines()
                    .filter(|line| line.starts_with("guess,"))
                    .count(),
                1
            );
            let runs: crate::hash::HashSet<_> = csv
                .lines()
                .skip(1)
                .map(|line| line.rsplit(',').next().unwrap().parse::<usize>().unwrap())
                .collect();
            assert!(runs.contains(&0) && runs.contains(&1) && runs.contains(&2));
        });
    }

    #[test]
    fn zero_budget_still_reports_proven_infeasibility() {
        let mut problem = desugar(&parse_program("assert 1 == 0").unwrap()).unwrap();
        assert_eq!(problem.solve(Some(0)), Err(CegisError::Infeasible));
        assert!(matches!(
            problem.solve_full(Some(0)),
            Err(CegisError::Infeasible)
        ));
        let mut problem = desugar(&parse_program("hole h\nassert h == 1").unwrap()).unwrap();
        assert_eq!(problem.solve(Some(0)), Err(CegisError::IterationLimit));
        assert!(matches!(
            problem.solve_full(Some(0)),
            Err(CegisError::IterationLimit)
        ));
    }
    #[test]
    fn retained_bounds_make_progress_even_with_one_round_attempts() {
        crate::experiment::with_seed(17, || {
            let mut problem =
                desugar(&parse_program("hole h\nassert h == (x0 := 1)").unwrap()).unwrap();
            let reference = SPP::make_reference_dfa(&mut problem.store, &problem.constraints);
            let mut bounds = Vec::new();
            let first = cegis::run_with_bounds::<SPP, Native>(
                &problem.constraints,
                &problem.holes,
                &reference,
                &mut problem.store,
                Some(1),
                Some(&mut bounds),
            );
            assert_eq!(first, Err(CegisError::IterationLimit));
            assert_eq!(bounds.len(), 1);
            assert!(bounds[0].0 != problem.store.zero || bounds[0].1 != problem.store.top);
            let solution = run::<SPP, Native>(
                &problem.constraints,
                &problem.holes,
                &reference,
                &mut problem.store,
                Some(100),
                Config {
                    save_bounds: true,
                    initial_cutoff: NonZeroUsize::new(1).unwrap(),
                },
            )
            .unwrap();
            for input in [false, true] {
                for output in [false, true] {
                    assert_eq!(
                        problem
                            .store
                            .accepts(solution[&Hole(0)], &[input], &[output]),
                        output
                    );
                }
            }
        });
    }
}
