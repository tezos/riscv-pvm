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

# Regenerate the checked-in inputs and refresh the matching goldenfiles
reset-regressions:
	@cargo run -p xtask -- gen-database-regression-inputs
	@UPDATE_GOLDENFILES=1 cargo nextest run test_database_regression

build-database-long-test:
	@cargo build --release --features rocksdb,unstable-test-utils --bin database_long_test

run-database-long-test:
	@cargo run --release --features rocksdb,unstable-test-utils \
		--bin database_long_test -- test --max-minutes 10 --keep-epochs 1

.PHONY: all check test gen-regression-inputs run-database-long-test build-database-long-test
