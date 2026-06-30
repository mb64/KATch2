# Build/test/bench helpers for the two optional optimization features:
#   lb_mincut     — lower-bound clause via min cut          (holes::cegis)
#   clause_merge  — collate SMT membership disjuncts        (holes::smt)
#
# Both are on by default. The four combos below cover every on/off pairing so
# each optimization can be evaluated in isolation and together.

# combo name -> cargo feature flags
COMBOS := default none mincut merge
feat-default :=
feat-none    := --no-default-features
feat-mincut  := --no-default-features --features lb_mincut
feat-merge   := --no-default-features --features clause_merge

.PHONY: help test bench clippy test-all bench-all clippy-all

help:
	@echo "Targets:"
	@echo "  test            cargo test (default features: both optimizations on)"
	@echo "  bench           cargo bench (default features)"
	@echo "  test-all        run the test suite under all 4 feature combos"
	@echo "  bench-all       run the synthesis bench under all 4 combos,"
	@echo "                  saving a criterion baseline per combo"
	@echo "  clippy-all      clippy -D warnings under all 4 combos"
	@echo "  test-<combo>    one combo; <combo> in: $(COMBOS)"
	@echo "  bench-<combo>   one combo"
	@echo "  clippy-<combo>  one combo"
	@echo ""
	@echo "Compare two bench baselines (after bench-all):"
	@echo "  cargo bench --bench synthesis -- --load-baseline mincut --baseline none"

test:
	cargo test

bench:
	cargo bench --bench synthesis

clippy:
	cargo clippy --lib --tests -- -D warnings

# Generate test-/bench-/clippy-<combo> targets for each combo. `bench-<combo>`
# saves its results as a criterion baseline named after the combo, so the runs
# can be compared afterwards.
define COMBO_template
.PHONY: test-$(1) bench-$(1) clippy-$(1)
test-$(1):
	@echo "==> test [$(1)] $(feat-$(1))"
	cargo test $(feat-$(1))
bench-$(1):
	@echo "==> bench [$(1)] $(feat-$(1))"
	cargo bench --bench synthesis $(feat-$(1)) -- --save-baseline $(1)
clippy-$(1):
	@echo "==> clippy [$(1)] $(feat-$(1))"
	cargo clippy --lib --tests $(feat-$(1)) -- -D warnings
endef

$(foreach c,$(COMBOS),$(eval $(call COMBO_template,$(c))))

test-all:   $(addprefix test-,$(COMBOS))
bench-all:  $(addprefix bench-,$(COMBOS))
clippy-all: $(addprefix clippy-,$(COMBOS))
