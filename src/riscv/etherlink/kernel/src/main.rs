// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::sync::Once;

use tezos_smart_rollup::entrypoint;
use tezos_smart_rollup::prelude::*;
use tezos_smart_rollup::storage::path::RefPath;

/// Values which need to be set in the Etherlink kernel's durable storage.
const VALUES: [(&str, &str); 6] = [
    (
        "/evm/world_state/eth_accounts/d9e5c94a12f78a96640757ac97ba0c257e8aa262/balance",
        "00000000000000000000dc0a0713000c1e020000000000000000000000000000",
    ),
    (
        "/evm/ticketer",
        "4b5431445841445357587563414a3650756a5a6555534b39627270546f46744339667a36",
    ),
    ("/evm/feature_flags/enable_fa_bridge", "01"),
    (
        "/evm/world_state/eth_accounts/0000000000000000000000000000000000000000/ticket_table/8cddc35907a9062880f22f5c815bb5f808ddfc4839363eb2762a4f0300917cbe/f0affc80a5f69f4a9a3ee01a640873b6ba53e539",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7F",
    ),
    (
        "/evm/world_state/eth_accounts/0000000000000000000000000000000000000000/ticket_table/fc2a0f0107d0e2d1b32bd9d99dd3da5bbda4edf016e64d5eec8d64310d3eeb88/f0affc80a5f69f4a9a3ee01a640873b6ba53e539",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7F",
    ),
    (
        "/tezlink/context/contracts/index/000002298c03ed7d454a101eb7022bc95f7e5f41ac78/balance",
        "00",
    ),
];

#[entrypoint::main]
#[cfg_attr(
    feature = "static-inbox",
    entrypoint::runtime(static_inbox = "$INBOX_FILE")
)]
pub fn entry(host: &mut impl Runtime) {
    static ONCE: Once = Once::new();

    ONCE.call_once(|| {
        for (path, value) in VALUES {
            let data = hex::decode(value).unwrap();
            host.store_write_all(&RefPath::assert_from(path.as_bytes()), &data)
                .unwrap();
        }
    });

    evm_kernel::kernel_loop(host);
}
