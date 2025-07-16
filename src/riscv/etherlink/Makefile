# SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

NATIVE_TARGET ?= $(shell rustc -vV | grep 'host:' | awk '{print $$2}')
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

INBOX_FILE ?= "../../../assets/etherlink-erc20-inbox.json"

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

.PHONY: build-kernel-static
build-kernel-static:
	@INBOX_FILE=$(INBOX_FILE) cargo build -p etherlink --release --features static-inbox

.PHONY: build-kernel-native
build-kernel-native:
	@INBOX_FILE=$(INBOX_FILE) cargo build -p etherlink --release --features static-inbox $(NATIVE_OPT)

.PHONY: test
test: build-kernel build-kernel-native
	@../../../scripts/etherlink-bench.sh
	@../../../scripts/etherlink-bench.sh -sn

.PHONY: check
check:
	@exec ../../../scripts/format.sh --check
	@INBOX_FILE=$(INBOX_FILE) cargo check --all-features --workspace --locked
	@INBOX_FILE=$(INBOX_FILE) cargo clippy --all-features -- --deny warnings
	@INBOX_FILE=$(INBOX_FILE) cargo doc --all-features --document-private-items --no-deps

.PHONY: inbox-bench
inbox-bench:
	@cargo build $(NATIVE_OPT) --bin inbox-bench --release
	@cp target/$(NATIVE_TARGET)/release/inbox-bench .

.PHONY: clean
clean:
	@cargo clean
