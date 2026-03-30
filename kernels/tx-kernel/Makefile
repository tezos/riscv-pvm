# SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

all: check build

build: cargo-build

check: cargo-format cargo-check cargo-clippy cargo-check-docs

clean:
	@cargo clean

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

.PHONY: all $(MAKECMDGOALS)

