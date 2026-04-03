# SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

RISCV_LIBS = \
	octez-riscv \
	octez-riscv-durable-storage \
	octez-riscv-data

RISCV_PACKAGE_ARGS = $(addprefix -p ,$(RISCV_LIBS))

.PHONY: all
all: build check

.PHONY: build
build:
	@cargo build --release $(RISCV_PACKAGE_ARGS)

.PHONY: test-deps
test-deps:
	@make test-deps

.PHONY: test
test: test-deps
	@cargo nextest run $(RISCV_PACKAGE_ARGS)

.PHONY: check
check:
	@cargo update $(RISCV_PACKAGE_ARGS) --locked
	@cargo check $(RISCV_PACKAGE_ARGS) --all-targets
	@cargo clippy $(RISCV_PACKAGE_ARGS) --all-targets -- --deny warnings
	@cargo doc --document-private-items --no-deps

codecov.json: test-deps
	@cargo llvm-cov $(RISCV_PACKAGE_ARGS) --codecov --output-path $@ nextest

.PHONY: clean
clean:
	@cargo clean
