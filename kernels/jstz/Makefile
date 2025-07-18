# SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

NATIVE_TARGET ?= $(shell rustc -vV | grep 'host:' | awk '{print $$2}')
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

INBOX_FILE ?= $(shell mktemp)

all: build test check inbox-bench

print-native-target:
	$(info $(NATIVE_TARGET))

build-deps:
ifneq ($(NATIVE_TARGET),)
	@rustup target add $(NATIVE_TARGET)
endif

build: build-kernel inbox-bench

build-kernel: cargo-build-kernel

build-kernel-static: cargo-build-kernel-static

build-kernel-native: cargo-build-kernel-native

test: cargo-test test-bench-script

test-bench-script: build
	@../../scripts/jstz-bench.sh -t 1
	@../../scripts/jstz-bench.sh -t 1 -s
	@../../scripts/jstz-bench.sh -t 1 -sn

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
	@cargo build --package jstz --release

cargo-build-inbox-bench:
	@cargo build $(NATIVE_OPT) --bin inbox-bench --release

cargo-test:
	@cargo test --no-default-features $(NATIVE_OPT)

cargo-build-kernel-static:
	@INBOX_FILE=$(INBOX_FILE) cargo build -p jstz --release --features static-inbox

cargo-build-kernel-native:
	@INBOX_FILE=$(INBOX_FILE) cargo build -p jstz --release --features static-inbox $(NATIVE_OPT)

# Mark all targets as phony to make sure they're always executed
.PHONY: all $(MAKECMDGOALS)
