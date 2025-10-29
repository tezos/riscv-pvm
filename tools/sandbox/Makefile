# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

# Mechanism to allow enabling additional features
SANDBOX_ENABLE_FEATURES ?=

### Generic top-level targets

all: check build

build: riscv-sandbox

riscv-sandbox: cargo-build-sandbox-release
	@ln -f ../../target/release/riscv-sandbox $@

riscv-sandbox.prof: cargo-build-sandbox-profiling
	@ln -f ../../target/profiling/riscv-sandbox $@

check: cargo-format cargo-check cargo-clippy

clean:
	@cargo clean
	@rm -f riscv-sandbox riscv-sandbox.prof

### We declare all the Cargo commands as targets to make sure they run with maximum parallelism.

cargo-build-sandbox-release:
	@cargo build --release $(SANDBOX_ENABLE_FEATURES:%=-F%)

cargo-build-sandbox-profiling:
	@cargo build --profile profiling $(SANDBOX_ENABLE_FEATURES:%=-F%)

cargo-format:
	@exec ../../scripts/format.sh --all

cargo-check:
	@cargo check --all-targets

cargo-clippy:
	@cargo clippy --all-targets -- --deny warnings

# Mark all targets as phony to make sure they're always executed
.PHONY: all $(MAKECMDGOALS)
