# SPDX-FileCopyrightText: 2023-2025 Nomadic Labs <contact@nomadic-labs.com>
# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

update-assets: riscv-dummy.elf jstz etherlink

riscv-dummy.elf riscv-dummy.elf.checksum:
	@make -C .. dummy/build
	@cp ../kernels/dummy/target/riscv64gc-unknown-linux-musl/release/riscv-dummy $@
	@sha256sum $@ > $@.checksum

jstz jstz.checksum:
	@make -C .. jstz/build-kernel
	@cp ../kernels/jstz/target/riscv64gc-unknown-linux-musl/release/jstz $@
	@sha256sum $@ > $@.checksum

etherlink etherlink.checksum:
	@make -C .. etherlink/build-kernel
	@cp ../kernels/etherlink/target/riscv64gc-unknown-linux-musl/release/etherlink $@
	@sha256sum $@ > $@.checksum

check:
	@echo "Verifying asset checksums..."
	@sha256sum -c riscv-dummy.elf.checksum jstz.checksum etherlink.checksum

.PHONY: update-assets riscv-dummy.elf riscv-dummy.elf.checksum jstz jstz.checksum etherlink etherlink.checksum check
