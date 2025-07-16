# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

### Generic top-level targets

all: riscv/all jstz/all dummy/all etherlink/all

build-deps: riscv/build-deps jstz/build-deps etherlink/build-deps

build-deps-slim: riscv/build-deps-slim

check: riscv/check jstz/check dummy/check etherlink/check

audit: riscv/audit

build: riscv/build jstz/build dummy/build etherlink/build

test: riscv/test jstz/test etherlink/test 

test-long: riscv/test-long

test-miri: riscv/test-miri

clean: riscv/clean jstz/clean dummy/clean etherlink/clean

### Target proxies

riscv/%: 
	@make -C src/riscv ${@:riscv/%=%}

jstz/%: 
	@make -C kernels/jstz ${@:jstz/%=%}

dummy/%: 
	@make -C kernels/dummy ${@:dummy/%=%}

etherlink/%: 
	@make -C kernels/etherlink ${@:etherlink/%=%}

# Mark all non-pattern targets as phony to make sure they're always executed
.PHONY: all build-deps build-deps-slim check audit build test test-long test-miri clean 
