# SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
# SPDX-License-Identifier: MIT

all: check test

check:
	@cargo clippy --all-targets --no-default-features -- --deny warnings

# distinct from the workspace-level test in that it actually turns off
# rocksdb: using the in-memory implementation instead
test:
	@cargo nextest run --no-default-features

.PHONY: all check test
