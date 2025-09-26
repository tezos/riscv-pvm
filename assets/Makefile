# SPDX-FileCopyrightText: 2023-2025 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

update-assets: riscv-dummy.elf jstz

riscv-dummy.elf riscv-dummy.elf.checksum:
	@make -C .. dummy/build
	@cp ../kernels/dummy/target/riscv64gc-unknown-linux-musl/release/riscv-dummy $@
	@sha256sum $@ > $@.checksum

jstz jstz.checksum:
	@make -C .. jstz/build-kernel
	@cp ../kernels/jstz/target/riscv64gc-unknown-linux-musl/release/jstz $@
	@sha256sum $@ > $@.checksum

.PHONY: update-assets riscv-dummy.elf riscv-dummy.elf.checksum jstz jstz.checksum
