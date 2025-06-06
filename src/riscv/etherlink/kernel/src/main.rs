// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use tezos_smart_rollup::entrypoint;
use tezos_smart_rollup::prelude::*;

#[entrypoint::main]
pub fn entry(host: &mut impl Runtime) {
    evm_kernel::kernel_loop(host);
}
