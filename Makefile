# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

### Generic top-level targets

all: riscv/all jstz/all dummy/all block-cache-tester/all signal-tester/all etherlink/all

build-deps: riscv/build-deps jstz/build-deps etherlink/build-deps

build-deps-slim: riscv/build-deps-slim

check: riscv/check jstz/check dummy/check block-cache-tester/all signal-tester/check etherlink/check

build: riscv/build jstz/build dummy/build block-cache-tester/build etherlink/build

test: riscv/test jstz/test etherlink/test 

test-miri: riscv/test-miri

clean: riscv/clean jstz/clean dummy/clean block-cache-tester/clean signal-tester/clean etherlink/clean

### Specific top-level targets

audit:
	@cargo audit

### Target proxies

riscv/%:
	@make -C src/riscv ${@:riscv/%=%}

jstz/%:
	@make -C kernels/jstz ${@:jstz/%=%}

dummy/%:
	@make -C kernels/dummy ${@:dummy/%=%}

block-cache-tester/%:
	@make -C kernels/block-cache-tester ${@:block-cache-tester/%=%}

signal-tester/%:
	@make -C kernels/signal-tester ${@:signal-tester/%=%}

etherlink/%:
	@make -C kernels/etherlink ${@:etherlink/%=%}

# Mark all non-pattern targets as phony to make sure they're always executed
.PHONY: all build-deps build-deps-slim check audit build test test-miri clean 
