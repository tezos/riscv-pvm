# SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

.PHONY: all
all: build check

.PHONY: build
build:
	@cargo build --release -p octez-riscv

.PHONY: test-deps
test-deps:
	@make -C .. test-deps

.PHONY: test
test: test-deps
	@cargo nextest run -p octez-riscv

.PHONY: check
check:
	@cargo update --locked -p octez-riscv
	@cargo check --all-targets -p octez-riscv
	@cargo clippy --all-targets  -p octez-riscv -- --deny warnings
	@cargo doc --document-private-items --no-deps

.PHONY: coverage-report
coverage-report: test-deps
	@cargo llvm-cov -p octez-riscv --html --open nextest

.PHONY: coverage
coverage:
	@../scripts/isa-suite-coverage.sh

.PHONY: clean
clean:
	@cargo clean
