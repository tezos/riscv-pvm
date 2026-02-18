# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

### Generic top-level targets

all: riscv/all sandbox/all jstz/all dummy/all page-cache-tester/all etherlink/all docs/all

check: riscv/check jstz/check dummy/check page-cache-tester/check etherlink/check assets/check docs/check check-format

build: sandbox/build jstz/build dummy/build page-cache-tester/build etherlink/build

test: riscv/test jstz/test etherlink/test

clean: riscv/clean sandbox/clean jstz/clean dummy/clean page-cache-tester/clean etherlink/clean

### Specific top-level targets

audit:
	@cargo audit

build-deps-slim: jstz/build-deps etherlink/build-deps
	# Ensure the toolchain is installed.
	# The second command triggers installation for Rustup 1.28+.
	@rustup show active-toolchain || rustup toolchain install
	@rustup component add rustfmt clippy

	# Install Nightly for formatting with its Rustfmt
	@rustup toolchain install $(NIGHTLY_VERSION) -c rustfmt -c rust-src

	# Iterate through all the toolchains. 'rustup show' (before Rustup 1.28) and
	# 'rustup toolchain install' (Rustup 1.28+) will install the toolchain.
	@find . -iname 'rust-toolchain*' -execdir sh -c "rustup show active-toolchain || rustup toolchain install" \; 2>/dev/null

	# Enable 'llvm-cov' subcommand for Cargo
	@cargo llvm-cov --version || cargo install --locked cargo-llvm-cov

build-deps: build-deps-slim
	# Coverage deps
	@scripts/isa-suite-coverage.sh -d

check-nix:
	@nix fmt

check-format: taplo-check-format

taplo-check-format:
	@taplo format --check

codecov.json: riscv/test-deps
	@cargo llvm-cov \
		--package octez-riscv \
		--package octez-riscv-data \
		--codecov \
		--output-path $@ \
		nextest

### Target proxies

riscv/%:
	@make -C src/riscv ${@:riscv/%=%}

sandbox/%:
	@make -C tools/sandbox ${@:sandbox/%=%}

jstz/%:
	@make -C kernels/jstz ${@:jstz/%=%}

dummy/%:
	@make -C kernels/dummy ${@:dummy/%=%}

page-cache-tester/%:
	@make -C kernels/page-cache-tester ${@:page-cache-tester/%=%}

etherlink/%:
	@make -C kernels/etherlink ${@:etherlink/%=%}

assets/%:
	@make -C assets ${@:assets/%=%}

docs/%:
	@make -C docs ${@:docs/%=%}

# Mark all non-pattern targets as phony to make sure they're always executed
.PHONY: all build-deps build-deps-slim check check-nix check-format taplo-check-format codecov.json audit build test clean
