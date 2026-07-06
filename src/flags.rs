//! Process-global optimization flags.
//!
//! Each optimization used to be a compile-time Cargo feature; they are now
//! runtime [`AtomicBool`]s so one binary can toggle them (deliberately global
//! and a little hacky — it saves threading a config object through every call
//! site).  Query with the getters wherever the optimization branches; nothing
//! caches a flag, so a set is visible from the next query onward.  Toggle at
//! startup, before solving begins: the optimizations are semantics-preserving
//! either way, but a mid-solve flip makes runs irreproducible.
//!
//! Every flag defaults to **on**.  Two ways to turn one off:
//!
//! * [`set_lb_mincut`] & co. — used by `nksynth`'s `--no-*` CLI flags.
//! * Environment variables `KATCH2_LB_MINCUT`, `KATCH2_CLAUSE_MERGE`,
//!   `KATCH2_EXAMPLE_COVER` (values `0`/`1`, `false`/`true`, `off`/`on`),
//!   read once at first access — how the `Makefile` runs the test and bench
//!   suites under each on/off combination.  Setters override the environment.

use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

static LB_MINCUT: AtomicBool = AtomicBool::new(true);
static CLAUSE_MERGE: AtomicBool = AtomicBool::new(true);
static EXAMPLE_COVER: AtomicBool = AtomicBool::new(true);

static ENV_INIT: Once = Once::new();

/// Fold the `KATCH2_*` environment overrides into the flag statics, once,
/// before the first get or set touches any flag.
fn env_init() {
    ENV_INIT.call_once(|| {
        for (name, flag) in [
            ("KATCH2_LB_MINCUT", &LB_MINCUT),
            ("KATCH2_CLAUSE_MERGE", &CLAUSE_MERGE),
            ("KATCH2_EXAMPLE_COVER", &EXAMPLE_COVER),
        ] {
            if let Ok(raw) = std::env::var(name) {
                flag.store(parse_flag(name, &raw), Ordering::Relaxed);
            }
        }
    });
}

/// Parse one environment override; panics on anything but a recognizable
/// boolean so a typo'd variable fails loudly instead of silently running the
/// wrong configuration.
fn parse_flag(name: &str, raw: &str) -> bool {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => true,
        "0" | "false" | "off" | "no" => false,
        _ => panic!("{name}: expected a boolean (0/1, false/true, off/on), got {raw:?}"),
    }
}

/// Lower-bound refinement clauses shrink via a min cut over the abstract
/// reachability graph instead of emitting every crossing hole edge
/// (`holes::cegis`).
pub fn lb_mincut() -> bool {
    env_init();
    LB_MINCUT.load(Ordering::Relaxed)
}

/// Override [`lb_mincut`] (wins over the default and the environment).
pub fn set_lb_mincut(value: bool) {
    env_init();
    LB_MINCUT.store(value, Ordering::Relaxed);
}

/// Collate SMT membership disjuncts by shared input/output set before
/// existentializing them (`holes::smt`).
pub fn clause_merge() -> bool {
    env_init();
    CLAUSE_MERGE.load(Ordering::Relaxed)
}

/// Override [`clause_merge`] (wins over the default and the environment).
pub fn set_clause_merge(value: bool) {
    env_init();
    CLAUSE_MERGE.store(value, Ordering::Relaxed);
}

/// After the SMT solver finds a model, keep only a greedy set cover of the
/// clauses by true literals as training examples for the passive learners,
/// instead of every constrained atom (`holes::smt`).
pub fn example_cover() -> bool {
    env_init();
    EXAMPLE_COVER.load(Ordering::Relaxed)
}

/// Override [`example_cover`] (wins over the default and the environment).
pub fn set_example_cover(value: bool) {
    env_init();
    EXAMPLE_COVER.store(value, Ordering::Relaxed);
}
