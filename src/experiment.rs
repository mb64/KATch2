//! Configuration and independent random streams for synthesis experiments.
//!
//! `nksynth` initializes this on its solver thread. Library callers retain
//! deterministic behavior until they explicitly call [`configure`].
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{cell::RefCell, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub symbolic_upper: bool,
    pub candidate_stats: Option<PathBuf>,
    /// Emit cumulative measurements every N refinement rounds; zero disables.
    pub stats_interval: usize,
}
impl Config {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            symbolic_upper: true,
            candidate_stats: None,
            stats_interval: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seeds {
    pub master: u64,
    pub sat: i32,
    pub phases: u64,
    pub merges: u64,
    pub checks: u64,
}
struct State {
    config: Config,
    seeds: Seeds,
    merges: StdRng,
    checks: StdRng,
}
impl State {
    fn new(config: Config) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let seeds = Seeds {
            master: config.seed,
            sat: rng.random_range(0..=2_000_000_000),
            phases: rng.random(),
            merges: rng.random(),
            checks: rng.random(),
        };
        Self {
            config,
            seeds,
            merges: StdRng::seed_from_u64(seeds.merges),
            checks: StdRng::seed_from_u64(seeds.checks),
        }
    }
}
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }

/// Reset all experimental random streams on the current solver thread.
/// Log the returned seeds with the build and input when collecting results.
pub fn configure(config: Config) -> Seeds {
    let state = State::new(config);
    let seeds = state.seeds;
    STATE.with(|s| *s.borrow_mut() = Some(state));
    seeds
}
pub(crate) fn sat_parameters() -> Option<(i32, StdRng)> {
    STATE.with(|s| {
        s.borrow()
            .as_ref()
            .map(|s| (s.seeds.sat, StdRng::seed_from_u64(s.seeds.phases)))
    })
}
pub(crate) fn shuffle<T>(items: &mut [T]) {
    STATE.with(|s| {
        if let Some(s) = &mut *s.borrow_mut() {
            items.shuffle(&mut s.merges);
        }
    });
}
pub(crate) fn positive_first() -> bool {
    STATE.with(|s| s.borrow_mut().as_mut().is_some_and(|s| s.checks.random()))
}
pub(crate) fn symbolic_upper() -> bool {
    STATE.with(|s| s.borrow().as_ref().is_none_or(|s| s.config.symbolic_upper))
}
pub(crate) fn stats_interval() -> usize {
    STATE.with(|s| s.borrow().as_ref().map_or(0, |s| s.config.stats_interval))
}
pub(crate) fn candidate_stats_path() -> Option<PathBuf> {
    STATE.with(|s| {
        s.borrow()
            .as_ref()
            .and_then(|s| s.config.candidate_stats.clone())
    })
}
pub(crate) fn candidate_stats_enabled() -> bool {
    STATE.with(|s| {
        s.borrow()
            .as_ref()
            .is_some_and(|s| s.config.candidate_stats.is_some())
    })
}

#[cfg(test)]
pub(crate) fn with_seed<T>(seed: u64, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<State>);
    impl Drop for Restore {
        fn drop(&mut self) {
            STATE.with(|s| *s.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(STATE.with(|s| s.borrow_mut().replace(State::new(Config::new(seed)))));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streams_replay_and_do_not_consume_each_other() {
        let sample = || {
            let mut v: Vec<_> = (0..24).collect();
            shuffle(&mut v);
            (v, (0..32).map(|_| positive_first()).collect::<Vec<_>>())
        };
        assert_eq!(with_seed(17, sample), with_seed(17, sample));
        assert_ne!(with_seed(17, sample), with_seed(18, sample));
        let plain = with_seed(17, || (0..32).map(|_| positive_first()).collect::<Vec<_>>());
        let shuffled = with_seed(17, || {
            let mut v: Vec<_> = (0..24).collect();
            for _ in 0..50 {
                shuffle(&mut v)
            }
            (0..32).map(|_| positive_first()).collect::<Vec<_>>()
        });
        assert_eq!(plain, shuffled);
        let mut state = State::new(Config::new(17));
        assert!((0..=2_000_000_000).contains(&state.seeds.sat));
        assert_ne!(state.merges.random::<u64>(), state.checks.random::<u64>());
    }
}
