# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

### Generic top-level targets

all: riscv/all sandbox/all jstz/all dummy/all block-cache-tester/all etherlink/all

build-deps: riscv/build-deps jstz/build-deps etherlink/build-deps

build-deps-slim: riscv/build-deps-slim

check: riscv/check sandbox/check jstz/check dummy/check block-cache-tester/all etherlink/check assets/check

build: sandbox/build jstz/build dummy/build block-cache-tester/build etherlink/build

test: riscv/test jstz/test etherlink/test 

test-miri: riscv/test-miri

clean: riscv/clean sandbox/clean jstz/clean dummy/clean block-cache-tester/clean etherlink/clean

### Specific top-level targets

audit:
	@cargo audit

### Target proxies

riscv/%: 
	@make -C src/riscv ${@:riscv/%=%}

sandbox/%: 
	@make -C tools/sandbox ${@:sandbox/%=%}

jstz/%: 
	@make -C kernels/jstz ${@:jstz/%=%}

dummy/%: 
	@make -C kernels/dummy ${@:dummy/%=%}

block-cache-tester/%: 
	@make -C kernels/block-cache-tester ${@:block-cache-tester/%=%}

etherlink/%: 
	@make -C kernels/etherlink ${@:etherlink/%=%}

assets/%:
	@make -C assets ${@:assets/%=%}

# Mark all non-pattern targets as phony to make sure they're always executed
.PHONY: all build-deps build-deps-slim check audit build test test-miri clean 
