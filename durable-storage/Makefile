# SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

# Where the space harness writes its samples for `bench-metrics` to read.
SPACE_SAMPLES ?= space-samples.jsonl

# Scale for the reference-machine runs. Large enough to be representative of a rollup
# node's storage, small enough that prepopulating it does not dominate the run.
REFERENCE_DATABASES ?= 4
REFERENCE_KEYS ?= 1000000
REFERENCE_MODIFIED ?= 25000

# Tag recorded with the reference metrics, so a change of scale is visible as a new series
# rather than a step change in an existing one.
REFERENCE_SHAPE ?= 4x1m

# Passed as MARKDOWN=--markdown to print the metrics as a table on stdout, for a CI comment;
# empty means one line per metric. The harness report goes to stderr either way, so redirecting
# stdout captures the table on its own.
MARKDOWN ?=

all: check test

check:
	@cargo clippy --all-targets --no-default-features -- --deny warnings
	@cargo clippy -p xtask --all-targets -- --deny warnings
	@cargo clippy --features unstable-test-utils --bin long_test -- --deny warnings
	@cargo clippy --features unstable-test-utils --bin gc_space -- --deny warnings

# distinct from the workspace-level test in that it actually turns off
# rocksdb: using the in-memory implementation instead
test:
	@cargo nextest run --no-default-features

# Regenerate the checked-in inputs and refresh the matching goldenfiles for
# Database regression tests
reset-database-regressions:
	@cargo run -p xtask -- gen-database-regression-inputs
	@UPDATE_GOLDENFILES=1 cargo nextest run test_database_regression

# Regenerate the checked-in inputs and refresh the matching goldenfiles for
# Registry regression tests
reset-registry-regressions:
	@cargo run -p xtask -- gen-registry-regression-inputs
	@UPDATE_GOLDENFILES=1 cargo nextest run test_registry_proof_regression

reset-regressions: reset-database-regressions reset-registry-regressions

build-long-test:
	@cargo build --release --features rocksdb,unstable-test-utils --bin long_test

run-database-long-test:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin long_test -- database test --max-minutes 10 --keep-epochs 1 \
		--fail-on-warning

run-registry-long-test:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin long_test -- registry test --max-minutes 10 --keep-epochs 1 \
		--fail-on-warning

build-gc-space:
	@cargo build --release --features rocksdb,unstable-test-utils --bin gc_space

# Space accounting at a scale small enough for a pull request. Reports how much of the
# storage is garbage, what retaining a commit costs, and how much of a commit's bytes are
# shared with the previous one — the assumption retention depends on.
space-metrics: build-gc-space
	@cargo run --release --features rocksdb,unstable-test-utils --bin gc_space -- \
		--databases 1 --keys 100000 --commits 10 --modified-keys 1000 \
		--sample-every 1 --json-out $(SPACE_SAMPLES) --simulate-dir-gc
	@cargo run -p bench-metrics -- --shape pr $(MARKDOWN) space $(SPACE_SAMPLES)

# Same, at the scale worth tracking over time. Intended to be driven by
# scripts/ci-durable-bench.sh on the reference machine: latencies measured anywhere else
# are noise.
space-metrics-reference: build-gc-space
	@cargo run --release --features rocksdb,unstable-test-utils --bin gc_space -- \
		--databases $(REFERENCE_DATABASES) --keys $(REFERENCE_KEYS) --commits 20 \
		--modified-keys $(REFERENCE_MODIFIED) --sample-every 2 \
		--json-out $(SPACE_SAMPLES) --simulate-dir-gc
	@cargo run -p bench-metrics -- --shape $(REFERENCE_SHAPE) $(MARKDOWN) \
		space $(SPACE_SAMPLES)

# The same measurement over more commits, for the merge queue and the default branch. Ten commits
# mostly show the base state; by thirty the dead-node share and the cost of retention have moved far
# enough to be worth comparing between merges. Around ninety seconds, and two gigabytes of scratch.
#
# --sample-every 1 is deliberate. Sharing compares consecutive samples, not consecutive commits, so
# measuring every other commit would report the sharing of two commits at once and make the figure
# incomparable with the pull request run.
space-metrics-long: build-gc-space
	@cargo run --release --features rocksdb,unstable-test-utils --bin gc_space -- \
		--databases 1 --keys 250000 --commits 30 --modified-keys 2500 \
		--sample-every 1 --json-out $(SPACE_SAMPLES) --simulate-dir-gc
	@cargo run -p bench-metrics -- --shape long $(MARKDOWN) space $(SPACE_SAMPLES)

# Registry lifecycle latencies, at the same reference scale.
lifecycle-metrics-reference:
	@REGISTRY_DATABASE_COUNT=$(REFERENCE_DATABASES) \
		PREPOPULATED_NODE_KEYS_COUNT=$(REFERENCE_KEYS) \
		COMMIT_MODIFIED_KEYS_COUNT=$(REFERENCE_MODIFIED) \
		cargo bench --features rocksdb --bench database_lifecycle
	@cargo run -p bench-metrics -- --shape $(REFERENCE_SHAPE) \
		latency ../target/criterion --group 'Registry lifecycle'

run-registry-long-test-stable:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin long_test -- registry test --max-minutes 10 --keep-epochs 1 \
		--keep-stable-size --fail-on-warning

.PHONY: all check test run-database-long-test reset-database-regressions reset-registry-regressions reset-regressions build-long-test run-registry-long-test run-registry-long-test-stable build-gc-space space-metrics space-metrics-reference space-metrics-long
