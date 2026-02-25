# SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

.PHONY: all
all: build check

.PHONY: build
build:
	@cargo build --release --workspace

.PHONY: test-deps
test-deps:
	@make -C ../.. test-deps

.PHONY: coverage-report
coverage-report: test-deps
	@cargo llvm-cov --package octez-riscv --html --open nextest

.PHONY: coverage
coverage:
	@../../scripts/isa-suite-coverage.sh

.PHONY: check
check:
	@exec ../../scripts/format.sh --all --check

	@cargo update --workspace --locked
	@cargo check --workspace --all-targets
	@cargo clippy --workspace --all-targets -- --deny warnings
	@cargo doc --document-private-items --no-deps

.PHONY: clean
clean:
	@cargo clean
