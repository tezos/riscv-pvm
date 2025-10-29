# SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

### Generic top-level targets

all: riscv/all sandbox/all jstz/all dummy/all page-cache-tester/all etherlink/all

build-deps: riscv/build-deps jstz/build-deps etherlink/build-deps

build-deps-slim: riscv/build-deps-slim

check: riscv/check jstz/check dummy/check page-cache-tester/check etherlink/check assets/check check-format

build: sandbox/build jstz/build dummy/build page-cache-tester/build etherlink/build

test: riscv/test jstz/test etherlink/test

clean: riscv/clean sandbox/clean jstz/clean dummy/clean page-cache-tester/clean etherlink/clean 

### Specific top-level targets

audit:
	@cargo audit

check-nix:
	@nix fmt

check-format: taplo-check-format
	
taplo-check-format:
	@taplo format --check

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

# Mark all non-pattern targets as phony to make sure they're always executed
.PHONY: all build-deps build-deps-slim check check-nix audit build test clean 
