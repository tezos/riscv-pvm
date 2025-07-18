# SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

all: check build

build: cargo-build

check: cargo-format cargo-check cargo-clippy cargo-check-docs

clean:
	@cargo clean

### We declare all the Cargo commands as targets to make sure they run with maximum parallelism.

cargo-build:
	@cargo build --release

cargo-format:
	@exec ../../scripts/format.sh --check

cargo-check:
	@cargo check --workspace --locked

cargo-clippy:
	@cargo clippy -- --deny warnings

cargo-check-docs:
	@cargo doc --all-features --document-private-items --no-deps

# Mark all targets as phony to make sure they're always executed
.PHONY: all $(MAKECMDGOALS)
