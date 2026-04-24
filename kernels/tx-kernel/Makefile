# SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

NATIVE_TARGET ?= $(shell rustc -vV | grep 'host:' | awk '{print $$2}')
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

TRANSACTIONS ?= 1000
BLOCK_FREQUENCY ?= 100
NUMBER_OF_ACCOUNTS ?= 1024
PREPOPULATED_ACCOUNTS ?= $(NUMBER_OF_ACCOUNTS)
DURABLE_STORAGE_DIR ?= /tmp/tx-kernel-bench-db
BENCHMARK_SCENARIO ?= eth-transfer
INBOX_FILE ?=

all: check build

build: cargo-build cargo-build-bench

check: cargo-format cargo-check cargo-clippy cargo-check-docs cargo-check-bench

clean:
	@cargo clean

cargo-build:
	@cargo build --release

cargo-build-bench:
	@cargo build $(NATIVE_OPT) --release --manifest-path inbox-generator/Cargo.toml

cargo-build-sandbox:
	@cd ../.. && cargo build --release -p riscv-sandbox --offline

cargo-format:
	@exec ../../scripts/format.sh --check

cargo-check:
	@cargo check --workspace --locked

cargo-clippy:
	@cargo clippy -- --deny warnings

cargo-check-docs:
	@cargo doc --all-features --document-private-items --no-deps

cargo-check-bench:
	@cargo check $(NATIVE_OPT) --manifest-path inbox-generator/Cargo.toml

prepare-context: cargo-build-bench
	@cargo run $(NATIVE_OPT) --release --manifest-path inbox-generator/Cargo.toml -- \
		prepare-context \
		--durable-storage-dir "$(DURABLE_STORAGE_DIR)" \
		--accounts $(PREPOPULATED_ACCOUNTS) \
		--scenario $(BENCHMARK_SCENARIO) \
		--rebuild

benchmark: cargo-build cargo-build-bench cargo-build-sandbox
	@cargo run $(NATIVE_OPT) --release --manifest-path inbox-generator/Cargo.toml -- \
		benchmark \
		--transactions $(TRANSACTIONS) \
		--block-frequency $(BLOCK_FREQUENCY) \
		--accounts $(NUMBER_OF_ACCOUNTS) \
		--scenario $(BENCHMARK_SCENARIO) \
		--durable-storage-dir "$(DURABLE_STORAGE_DIR)" \
		$(if $(strip $(INBOX_FILE)),--inbox-file "$(INBOX_FILE)")

benchmark-native: cargo-build-bench
	@cargo run $(NATIVE_OPT) --release --manifest-path inbox-generator/Cargo.toml -- \
		benchmark \
		--native \
		--transactions $(TRANSACTIONS) \
		--block-frequency $(BLOCK_FREQUENCY) \
		--accounts $(NUMBER_OF_ACCOUNTS) \
		--scenario $(BENCHMARK_SCENARIO) \
		--durable-storage-dir "$(DURABLE_STORAGE_DIR)" \
		$(if $(strip $(INBOX_FILE)),--inbox-file "$(INBOX_FILE)")

.PHONY: all $(MAKECMDGOALS)
