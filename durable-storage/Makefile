# SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

all: check test

check:
	@cargo clippy --all-targets --no-default-features -- --deny warnings
	@cargo clippy -p xtask --all-targets -- --deny warnings
	@cargo clippy --features unstable-test-utils --bin database_long_test -- --deny warnings

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

build-database-long-test:
	@cargo build --release --features rocksdb,unstable-test-utils --bin database_long_test

run-database-long-test:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin database_long_test -- database test --max-minutes 10 --keep-epochs 1

run-registry-long-test:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin database_long_test -- registry test --max-minutes 10 --keep-epochs 1

run-registry-long-test-stable:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin database_long_test -- registry test --max-minutes 10  --keep-epochs 1 \
		--keep-stable-size

.PHONY: all check test gen-regression-inputs run-database-long-test build-database-long-test reset-database-regressions reset-registry-regressions run-registry-long-test run-registry-long-test-stable
