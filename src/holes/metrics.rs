//! Opt-in synthesis measurements, emitted to stderr with `KATCH2_METRICS=1`.
//! Timers in different scopes overlap; only timings within a scope are additive.

use std::{collections::BTreeMap, time::Instant};

pub(super) struct Metrics {
    scope: &'static str,
    enabled: bool,
    values: BTreeMap<&'static str, u64>,
}

impl Metrics {
    pub(super) fn new(scope: &'static str) -> Self {
        Self {
            scope,
            enabled: std::env::var("KATCH2_METRICS").is_ok_and(|v| v == "1"),
            values: BTreeMap::new(),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    pub(super) fn add(&mut self, key: &'static str, amount: u64) {
        if self.enabled {
            *self.values.entry(key).or_default() += amount;
        }
    }

    pub(super) fn elapsed(&mut self, key: &'static str, start: Option<Instant>) {
        if let Some(start) = start {
            self.add(key, start.elapsed().as_nanos().try_into().unwrap());
        }
    }
}

impl Drop for Metrics {
    fn drop(&mut self) {
        if self.enabled {
            let values = self
                .values
                .iter()
                .map(|(k, v)| format!(" {k}={v}"))
                .collect::<String>();
            eprintln!("KATCH2_METRICS scope={}{}", self.scope, values);
        }
    }
}
