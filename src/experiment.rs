//! Synthesis configuration and one shared RNG per solver thread.
//!
//! The RNG starts from system entropy unless explicitly seeded with [`configure`].
//! All random choices, including seeds for new CaDiCaL instances, advance this
//! same stream. Reconfigure only to start or replay a seeded run.
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{cell::RefCell, num::NonZeroUsize, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub symbolic_upper: bool,
    pub save_restart_bounds: bool,
    pub restart_baseline: NonZeroUsize,
    pub candidate_stats: Option<PathBuf>,
    /// Emit cumulative measurements every N refinement rounds; zero disables.
    pub stats_interval: usize,
}
impl Config {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            symbolic_upper: true,
            save_restart_bounds: false,
            restart_baseline: NonZeroUsize::new(150).unwrap(),
            candidate_stats: None,
            stats_interval: 256,
        }
    }
}

struct State {
    config: Config,
    rng: StdRng,
    candidate_stats_runs: usize,
}
impl State {
    fn new(config: Config) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            rng,
            candidate_stats_runs: 0,
        }
    }
}
thread_local! {
    static STATE: RefCell<State> = RefCell::new({
        let mut config = Config::new(rand::rng().random());
        // Library callers leave checkpoints disabled unless configured.
        config.stats_interval = 0;
        State::new(config)
    });
}

/// Set configuration and seed the shared RNG on the current solver thread.
/// Keep advancing this stream across solver instances and randomized restarts.
pub fn configure(config: Config) {
    STATE.with(|s| *s.borrow_mut() = State::new(config));
}

pub(crate) fn with_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    STATE.with(|s| f(&mut s.borrow_mut().rng))
}

pub(crate) fn shuffle<T>(items: &mut [T]) {
    with_rng(|rng| items.shuffle(rng));
}
pub(crate) fn positive_first() -> bool {
    with_rng(|rng| rng.random())
}
pub(crate) fn save_restart_bounds() -> bool {
    STATE.with(|s| s.borrow().config.save_restart_bounds)
}
pub(crate) fn restart_baseline() -> NonZeroUsize {
    STATE.with(|s| s.borrow().config.restart_baseline)
}
pub(crate) fn symbolic_upper() -> bool {
    STATE.with(|s| s.borrow().config.symbolic_upper)
}
pub(crate) fn stats_interval() -> usize {
    STATE.with(|s| s.borrow().config.stats_interval)
}
/// Allocate a candidate-log run number without resetting the random stream.
pub(crate) fn candidate_stats_log() -> Option<(PathBuf, usize)> {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let path = state.config.candidate_stats.clone()?;
        let run = state.candidate_stats_runs;
        state.candidate_stats_runs += 1;
        Some((path, run))
    })
}
pub(crate) fn candidate_stats_enabled() -> bool {
    STATE.with(|s| s.borrow().config.candidate_stats.is_some())
}

#[cfg(test)]
pub(crate) fn with_seed<T>(seed: u64, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<State>);
    impl Drop for Restore {
        fn drop(&mut self) {
            STATE.with(|s| *s.borrow_mut() = self.0.take().unwrap());
        }
    }
    let _restore = Restore(Some(STATE.with(|s| {
        std::mem::replace(&mut *s.borrow_mut(), State::new(Config::new(seed)))
    })));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_stream_replays_and_advances_across_components() {
        let sample = || {
            let before = with_rng(|rng| rng.random::<u64>());
            let mut items: Vec<_> = (0..24).collect();
            shuffle(&mut items);
            let checks: Vec<_> = (0..32).map(|_| positive_first()).collect();
            let after = with_rng(|rng| rng.random::<u64>());
            (before, items, checks, after)
        };
        assert_eq!(with_seed(17, sample), with_seed(17, sample));
        assert_ne!(with_seed(17, sample), with_seed(18, sample));

        let plain = with_seed(17, || with_rng(|rng| rng.random::<u64>()));
        let after_shuffle = with_seed(17, || {
            shuffle(&mut [0, 1, 2, 3, 4, 5]);
            with_rng(|rng| rng.random::<u64>())
        });
        assert_ne!(plain, after_shuffle);
    }
}
