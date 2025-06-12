# SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

.PHONY: all
all: build check

.PHONY: build
build:
	@cargo build --release

.PHONY: check
check:
	@exec ../scripts/format.sh --check
	@cargo check --workspace --locked
	@cargo clippy -- --deny warnings
	@cargo doc --all-features --document-private-items --no-deps

.PHONY: clean
clean:
	@cargo clean
