# SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

NATIVE_TARGET ?= $(shell rustc -vV | grep 'host:' | awk '{print $$2}')
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

.PHONY: all
all: build check test inbox-bench

.PHONY: build-deps
build-deps:
ifneq ($(NATIVE_TARGET),)
	@rustup target add $(NATIVE_TARGET)
endif

.PHONY: build
build: build-kernel inbox-bench

.PHONY: build-kernel
build-kernel:
	@cargo build -p etherlink --release

.PHONY: test
test:
	@../../../scripts/etherlink-bench.sh

.PHONY: check
check:
	@exec ../../../scripts/format.sh --check
	@cargo check --workspace --locked
	@cargo clippy -- --deny warnings
	@cargo doc --all-features --document-private-items --no-deps

.PHONY: inbox-bench
inbox-bench:
	@cargo build $(NATIVE_OPT) --bin inbox-bench --release
	@cp target/$(NATIVE_TARGET)/release/inbox-bench .

.PHONY: clean
clean:
	@cargo clean
