# SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

all: build test check

build:
	@cargo build --release --workspace

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

build-deps: build-deps-slim
	# Coverage deps
	@../scripts/isa-suite-coverage.sh -d

test: test-deps
	@cargo nextest run --workspace
	@cargo test --workspace --doc

test-deps:
	@make -C .. dummy/build
	@make -C .. page-cache-tester/build

coverage-report: test-deps
	@cargo llvm-cov --package octez-riscv --html --open nextest

coverage:
	@../scripts/isa-suite-coverage.sh

check:
	@exec ../scripts/format.sh --all --check

	@cargo update --workspace --locked
	@cargo check --workspace --all-targets
	@cargo clippy --workspace --all-targets -- --deny warnings
	@cargo doc --document-private-items --no-deps

clean:
	@cargo clean

.PHONY: all build build-deps-slim build-deps test test-deps coverage-report coverage check clean
