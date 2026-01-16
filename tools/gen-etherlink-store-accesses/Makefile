# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

all: check build

build:
	@cargo build --release

check: cargo-format cargo-check cargo-clippy

clean:
	@cargo clean

cargo-format:
	@exec ../../scripts/format.sh --all

cargo-check:
	@cargo check --all-targets

cargo-clippy:
	@cargo clippy --all-targets -- --deny warnings

# Mark all targets as phony to make sure they're always executed
.PHONY: all $(MAKECMDGOALS)
