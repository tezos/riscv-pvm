# SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

NATIVE_TARGET ?= $(shell rustc -vV | grep 'host:' | awk '{print $$2}')
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

INBOX_FILE ?= "../../../../src/riscv/assets/etherlink-erc20-inbox.json"

all: check build test inbox-bench

build-deps:
ifneq ($(NATIVE_TARGET),)
	@rustup target add $(NATIVE_TARGET)
endif

build: build-kernel inbox-bench

build-kernel: cargo-build-kernel

build-kernel-static: cargo-build-kernel-static

build-kernel-native: cargo-build-kernel-native

test: build-kernel build-kernel-native
	@../../scripts/etherlink-bench.sh
	@../../scripts/etherlink-bench.sh -sn

check: cargo-format cargo-check cargo-clippy cargo-check-docs

inbox-bench: cargo-build-inbox-bench
	@ln -f target/$(NATIVE_TARGET)/release/inbox-bench $@

clean:
	@cargo clean

### We declare all the Cargo commands as targets to make sure they run with maximum parallelism.

cargo-format:
	@exec ../../scripts/format.sh --check

cargo-check:
	@INBOX_FILE=$(INBOX_FILE) cargo check --all-features --workspace --locked

cargo-clippy:
	@INBOX_FILE=$(INBOX_FILE) cargo clippy --all-features -- --deny warnings

cargo-check-docs:
	@INBOX_FILE=$(INBOX_FILE) cargo doc --all-features --document-private-items --no-deps

cargo-build-kernel:
	@cargo build --package etherlink --release

cargo-build-kernel-static:
	@INBOX_FILE=$(INBOX_FILE) cargo build -p etherlink --release --features static-inbox

cargo-build-kernel-native:
	@INBOX_FILE=$(INBOX_FILE) cargo build -p etherlink --release --features static-inbox $(NATIVE_OPT)

cargo-build-inbox-bench:
	@cargo build $(NATIVE_OPT) --bin inbox-bench --release

# Mark all targets as phony to make sure they're always executed
.PHONY: all $(MAKECMDGOALS)
