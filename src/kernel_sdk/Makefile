# SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

# The SDK contains some executables, that may be used by tezt.
# Since the tezt tests run on a different container to the one used
# for `make build-kernels`, we allow to compile executables using the
# musl toolchain, to make them fully statically-linked - allowing them
# to run in the tezt container, by setting the `NATIVE_TARGET`
# environment variable.

CC?=clang

.PHONY: all
all: build test doc

NATIVE_TARGET ?=
ifneq ($(NATIVE_TARGET),)
NATIVE_OPT := --target "$(NATIVE_TARGET)"
endif

RISCV_OPTS = --target riscv64gc-unknown-linux-musl --no-default-features
RISCV_OPTS_ALPHA = $(RISCV_OPTS) --features proto-alpha
RISCV_RUSTC_VERSION = 1.86.0

SDK_VERSION := "0.2.2"

.PHONY: check
check: installer-template publish-check doc riscv-check
	@cargo clippy --all-targets --all-features -- --deny warnings
	@cargo check --no-default-features
	@cargo check --target wasm32-unknown-unknown

# Ensure that individual crates have setup features on
# their own dependencies correctly - when building as
# a workspace, features enabled by one crate are enabled
# on all.
.PHONY: publish-check
publish-check: installer-template
	@cargo check -p tezos-smart-rollup-constants
	@cargo check -p tezos-smart-rollup-core
	@cargo check -p tezos-smart-rollup-host
	@cargo check -p tezos-smart-rollup-debug
	@cargo check -p tezos-smart-rollup-panic-hook
	@cargo check -p tezos-smart-rollup-entrypoint
	@cargo check -p tezos-smart-rollup-encoding
	@cargo check -p tezos-smart-rollup-storage
	@cargo check -p tezos-smart-rollup-mock
	@cargo check -p tezos-smart-rollup-installer-config
	@cargo check -p tezos-smart-rollup-installer
	@cargo check -p tezos-smart-rollup

.PHONY: riscv-check
riscv-check:
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-constants $(RISCV_OPTS)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-core $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-host $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-debug $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-panic-hook $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-entrypoint $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-encoding $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-mock $(RISCV_OPTS_ALPHA)
	@cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup $(RISCV_OPTS_ALPHA)

	# Does not work yet:
	# @cargo +${RISCV_RUSTC_VERSION} check -p tezos-smart-rollup-storage $(RISCV_OPTS_ALPHA)

.PHONY: build
build: installer-template
	@cargo build -p tezos-smart-rollup-installer \
                --release \
                $(NATIVE_OPT)

.PHONY: installer-template
installer-template:
	@cargo build -p installer-kernel \
	       --target wasm32-unknown-unknown \
	       --release \
	       --no-default-features \
	       --features entrypoint
	@cp target/wasm32-unknown-unknown/release/installer_kernel.wasm \
	    installer-client/installer.wasm
	@wasm-strip installer-client/installer.wasm

.PHONY: test
test:
	@cargo test --all-features

.PHONY: doc
doc:
	@cargo doc --no-deps --all-features

.PHONY: build-deps
build-deps:
	@rustup target add wasm32-unknown-unknown $(NATIVE_TARGET)
	@rustup toolchain install ${RISCV_RUSTC_VERSION} --target riscv64gc-unknown-linux-musl

.PHONY: build-dev-deps
build-dev-deps: build-deps
	@rustup component add rustfmt clippy

.PHONY: clean
clean:
	@cargo clean
	@rm -rf installer-client/installer.wasm

.PHONY: publish-deps
publish-deps: build-deps
	@cargo install cargo-publish-workspace-v2 --version "0.2.1"

# --allow-dirty required to include the built installer.wasm
# and README copied to sdk/, which are not checked in
.PHONY: publish
publish: installer-template
	@cargo publish-workspace -p 'tezos-smart-rollup-' \
	       --exclude installer-kernel \
	       --target-version $(SDK_VERSION) \
	       --aligned-versions-only \
	       -- --allow-dirty
