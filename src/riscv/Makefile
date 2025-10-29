# SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

.PHONY: all
all: build test check

# The version of nightly that we use for anything that requires it
NIGHTLY_VERSION = nightly-2025-01-30

.PHONY: build
build:
	@cargo build --release --workspace

.PHONY: build-deps-slim
build-deps-slim:
	# Ensure the toolchain is installed.
	# The second command triggers installation for Rustup 1.28+.
	@rustup show active-toolchain || rustup toolchain install
	@rustup component add rustfmt clippy

	# Install Nightly for formatting with its Rustfmt
	@rustup toolchain install $(NIGHTLY_VERSION) -c rustfmt -c rust-src

	# Iterate through all the toolchains. 'rustup show' (before Rustup 1.28) and
	# 'rustup toolchain install' (Rustup 1.28+) will install the toolchain.
	@find . -iname 'rust-toolchain*' -execdir sh -c "rustup show active-toolchain || rustup toolchain install" \; 2>/dev/null

	# Enable 'llvm-cov' subcommand for Cargo
	@cargo llvm-cov --version || cargo install --locked cargo-llvm-cov

.PHONY: build-deps
build-deps: build-deps-slim
	# Coverage deps
	@../../scripts/isa-suite-coverage.sh -d

.PHONY: test
test: test-deps
	@cargo nextest run --workspace

.PHONY: test-deps
test-deps:
	@make -C ../.. dummy/build
	@make -C ../.. page-cache-tester/build

.PHONY: codecov.json
codecov.json: test-deps
	@cargo llvm-cov --package octez-riscv --codecov --output-path $@ nextest

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
