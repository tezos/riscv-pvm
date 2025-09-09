# SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

NATIVE_TARGET ?= $(shell rustc -vV | grep 'host:' | awk '{print $$2}')
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

ifneq ($(FEATURES),)
FEATURES_OPT := --features "$(FEATURES)"
endif

# Build the kernel in release mode, unless a profile is passed
ifneq ($(PROFILE),)
BUILD_OPT := --profile $(PROFILE)
else
BUILD_OPT := --release
endif

# TODO RV-769: Remove when xtask alias works in this directory
XTASK := cargo run $(NATIVE_OPT) -p xtask

INBOX_FILE ?= "../../../../assets/etherlink-erc20-inbox.json"

all: check build test inbox-bench

build-deps:
ifneq ($(NATIVE_TARGET),)
	@rustup target add $(NATIVE_TARGET)
endif

build: build-kernel inbox-bench

build-kernel: cargo-build-kernel

build-kernel-native: cargo-build-kernel-native

test: bench-kernel bench-kernel-native trace-kernel profile-kernel

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
	@INBOX_FILE=$(INBOX_FILE) cargo build --package etherlink $(BUILD_OPT) $(FEATURES_OPT)

cargo-build-kernel-native:
	@INBOX_FILE=$(INBOX_FILE) cargo build --package etherlink $(BUILD_OPT) $(FEATURES_OPT) $(NATIVE_OPT)

cargo-build-inbox-bench:
	@cargo build $(NATIVE_OPT) --bin inbox-bench --release

# TODO RV-769: Make xtask usable directly via alias
bench-kernel: build-kernel
	@$(XTASK) bench

bench-kernel-native: build-kernel-native
	@$(XTASK) bench -sn

trace-kernel:
	@$(XTASK) bench -t

profile-kernel:
	@$(XTASK) profile -s 500

# Mark all targets as phony to make sure they're always executed
.PHONY: all $(MAKECMDGOALS)
