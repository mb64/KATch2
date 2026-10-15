//! Boolean bitblasting with an incremental RustSAT/CaDiCaL backend.

use rand::{RngExt, rngs::StdRng};
use std::collections::HashMap;

use rustsat::{
    solvers::{GetInternalStats, PhaseLit, Solve, SolverResult},
    types::{Clause, Lit, TernaryVal},
};
use rustsat_cadical::CaDiCaL;

/// Variable zero is permanently true; its negation represents false.
pub(super) const TRUE: Lit = Lit::positive(0);
pub(super) const FALSE: Lit = Lit::negative(0);

pub(super) struct Sat {
    solver: CaDiCaL<'static, 'static>,
    phase_rng: Option<StdRng>,
    next_var: u32,
    gates: HashMap<(Lit, Lit, Lit), Lit>,
    metrics: crate::holes::metrics::Metrics,
}

/// A completed SAT assignment, read once per solve. No FFI during extraction.
pub(super) struct Model(Vec<bool>);

impl Model {
    pub(super) fn value(&self, lit: Lit) -> bool {
        self.0[lit.vidx()] ^ lit.is_neg()
    }
}

impl Sat {
    pub(super) fn new() -> Self {
        let mut solver = CaDiCaL::default();
        let phase_rng = crate::experiment::sat_parameters().map(|(seed, rng)| {
            solver.set_option("seed", seed).expect("set CaDiCaL seed");
            rng
        });
        solver.add_unit(TRUE).expect("assert Boolean constant");
        Self {
            solver,
            phase_rng,
            next_var: 1,
            gates: HashMap::new(),
            metrics: crate::holes::metrics::Metrics::new("sat"),
        }
    }

    pub(super) fn checkpoint(&self) {
        self.metrics.checkpoint();
        if self.metrics.enabled() {
            eprintln!(
                "KATCH2_CHECKPOINT_SAT_STATS variables={} conflicts={} decisions={} propagations={}",
                self.next_var,
                self.solver.conflicts(),
                self.solver.decisions(),
                self.solver.propagations()
            );
        }
    }

    pub(super) fn fresh(&mut self) -> Lit {
        let lit = Lit::positive(self.next_var);
        self.next_var += 1;
        if let Some(rng) = &mut self.phase_rng {
            let preferred = if rng.random::<bool>() { lit } else { !lit };
            self.solver
                .phase_lit(preferred)
                .expect("set CaDiCaL phase preference");
        }
        lit
    }

    pub(super) fn clause(&mut self, lits: impl IntoIterator<Item = Lit>) {
        let mut clause = Vec::new();
        for lit in lits {
            if lit == TRUE {
                return;
            }
            if lit != FALSE {
                clause.push(lit);
            }
        }
        clause.sort_unstable();
        clause.dedup();
        if clause.windows(2).any(|pair| pair[0] == !pair[1]) {
            return;
        }
        self.metrics.add("clauses", 1);
        self.solver
            .add_clause(clause.into_iter().collect::<Clause>())
            .expect("add SAT clause");
    }

    /// Tseitin encoding of `result <=> if cond then yes else no`.
    pub(super) fn ite(&mut self, cond: Lit, yes: Lit, no: Lit) -> Lit {
        if cond == TRUE || yes == no {
            return yes;
        }
        if cond == FALSE {
            return no;
        }
        if yes == TRUE && no == FALSE {
            return cond;
        }
        if yes == FALSE && no == TRUE {
            return !cond;
        }
        if let Some(&lit) = self.gates.get(&(cond, yes, no)) {
            return lit;
        }
        let result = self.fresh();
        self.clause([!cond, !yes, result]);
        self.clause([!cond, yes, !result]);
        self.clause([cond, !no, result]);
        self.clause([cond, no, !result]);
        self.gates.insert((cond, yes, no), result);
        result
    }

    pub(super) fn equal(&mut self, a: Lit, b: Lit) -> Lit {
        if a == b {
            return TRUE;
        }
        if a == !b {
            return FALSE;
        }
        // Canonicalize to share equality gates across Ackermann lemmas.
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        self.ite(a, b, !b)
    }

    pub(super) fn solve(&mut self) -> Option<Model> {
        self.solver
            .reserve(Lit::positive(self.next_var - 1).var())
            .expect("reserve SAT variables");
        self.metrics.add("calls", 1);
        let started = self.metrics.start();
        let result = self.solver.solve().expect("solve SAT instance");
        self.metrics.elapsed("solve_ns", started);
        match result {
            SolverResult::Unsat => None,
            SolverResult::Interrupted => panic!("SAT solver unexpectedly interrupted"),
            SolverResult::Sat => {
                self.metrics.add("sat", 1);
                let started = self.metrics.start();
                let model = Model(
                    (0..self.next_var)
                        .map(|idx| {
                            // Unused variables may be unassigned. A consistent false
                            // completion is sufficient for unconstrained packet bits.
                            self.solver
                                .lit_val(Lit::positive(idx))
                                .expect("read SAT model")
                                == TernaryVal::True
                        })
                        .collect(),
                );
                self.metrics.elapsed("model_ns", started);
                Some(model)
            }
        }
    }
}

impl Drop for Sat {
    fn drop(&mut self) {
        if self.metrics.enabled() {
            self.metrics.add("variables", self.next_var as u64);
            self.metrics
                .add("conflicts", self.solver.conflicts() as u64);
            self.metrics
                .add("decisions", self.solver.decisions() as u64);
            self.metrics
                .add("propagations", self.solver.propagations() as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ite_truth_table_with_constants_negations_and_aliases() {
        // Every gate input can be a constant, a variable, or its negation;
        // repeated inputs exercise simplification as well as the CNF encoding.
        for c in 0..8 {
            for t in 0..8 {
                for e in 0..8 {
                    for assignment in 0..8 {
                        let mut sat = Sat::new();
                        let vars = [sat.fresh(), sat.fresh(), sat.fresh()];
                        let inputs = [
                            TRUE, FALSE, vars[0], !vars[0], vars[1], !vars[1], vars[2], !vars[2],
                        ];
                        let result = sat.ite(inputs[c], inputs[t], inputs[e]);
                        assert_eq!(sat.ite(inputs[c], inputs[t], inputs[e]), result);
                        for (idx, var) in vars.into_iter().enumerate() {
                            sat.clause([if assignment & (1 << idx) != 0 {
                                var
                            } else {
                                !var
                            }]);
                        }
                        let model = sat.solve().unwrap();
                        let expected = if model.value(inputs[c]) {
                            model.value(inputs[t])
                        } else {
                            model.value(inputs[e])
                        };
                        assert_eq!(model.value(result), expected);
                        // The encoding must imply the result, not just admit it.
                        sat.clause([if expected { !result } else { result }]);
                        assert!(sat.solve().is_none());
                    }
                }
            }
        }
    }

    #[test]
    fn incremental_variables_and_model_completion() {
        let mut sat = Sat::new();
        let unused = sat.fresh();
        let first = sat.solve().unwrap();
        assert!(first.value(TRUE));
        assert!(!first.value(FALSE));
        assert_ne!(first.value(unused), first.value(!unused));
        let new = sat.fresh();
        let equality = sat.equal(unused, new);
        sat.clause([new]);
        sat.clause([equality]);
        let model = sat.solve().unwrap();
        assert!(model.value(unused));
        assert!(model.value(new));
        sat.clause([!unused]);
        assert!(sat.solve().is_none());
        assert!(sat.solve().is_none());
    }
}

#[cfg(test)]
mod randomized_tests {
    use super::*;
    #[test]
    fn preferred_phases_preserve_incremental_constraints() {
        for seed in 0..32 {
            crate::experiment::with_seed(seed, || {
                let mut sat = Sat::new();
                let x = sat.fresh();
                let y = sat.fresh();
                sat.clause([x, y]);
                sat.clause([!x, !y]);
                let model = sat.solve().unwrap();
                assert_ne!(model.value(x), model.value(y));
                sat.clause([x]);
                let model = sat.solve().unwrap();
                assert!(model.value(x));
                assert!(!model.value(y));
                sat.clause([!x]);
                assert!(sat.solve().is_none());
            });
        }
    }
}
