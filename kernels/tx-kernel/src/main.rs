// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod keyspace;
mod sbi_crypto;
mod sbi_durable;

use std::str::FromStr;

use keyspace::DurableKeySpace;
use keyspace::DurableKeySpaceLoader;
use keyspace::Key;
use keyspace::KeySpace;
use keyspace::KeySpaceLoader;
use keyspace::Name;
use riscv_tx_kernel::AsyncKeccak;
use riscv_tx_kernel::ChainKernel;
use riscv_tx_kernel::ContextLoader;
use riscv_tx_kernel::ContextStore;
use riscv_tx_kernel::Crypto;
use riscv_tx_kernel::Logger;
use tezos_smart_rollup::entrypoint;
use tezos_smart_rollup::inbox::InboxMessage;
use tezos_smart_rollup::inbox::InternalInboxMessage;
use tezos_smart_rollup::michelson::MichelsonUnit;
use tezos_smart_rollup::prelude::*;

impl ContextStore for DurableKeySpace {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let key = Key::from_bytes(key).expect("kernel key must be valid");
        KeySpace::get(self, key).map_err(|error| error.to_string())
    }

    fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), String> {
        let key = Key::from_bytes(key).expect("kernel key must be valid");
        KeySpace::set(self, key, value).map_err(|error| error.to_string())
    }

    fn contains(&self, key: &[u8]) -> Result<bool, String> {
        let key = Key::from_bytes(key).expect("kernel key must be valid");
        KeySpace::contains(self, key).map_err(|error| error.to_string())
    }

    fn hash(&self) -> Result<Vec<u8>, String> {
        KeySpace::hash(self).map_err(|error| error.to_string())
    }
}

impl ContextLoader for DurableKeySpaceLoader {
    type Context = DurableKeySpace;

    fn load_or_create(&mut self, name: &str) -> Result<Self::Context, String> {
        KeySpaceLoader::load_or_create(self, Name::from_str(name).unwrap())
            .map(|keyspace| keyspace.clone())
            .map_err(|error| error.to_string())
    }
}

struct SbiCrypto;

impl Crypto for SbiCrypto {
    fn keccak256(&self, bytes: &[u8]) -> Result<[u8; 32], String> {
        unsafe { sbi_crypto::keccak256_hash(bytes) }.map_err(|_| "keccak host call failed".into())
    }

    fn verify_signature(
        &self,
        public_key: &[u8; 65],
        signature: &[u8; 64],
        message_hash: &[u8; 32],
    ) -> bool {
        unsafe { sbi_crypto::secp256k1_verify(public_key, signature, message_hash) }
    }
}

impl AsyncKeccak for SbiCrypto {
    fn enqueue(&self, bytes: &[u8]) -> Result<(), String> {
        unsafe { sbi_crypto::keccak256_enqueue(bytes) }
            .map_err(|_| "keccak enqueue host call failed".into())
    }

    fn dequeue(&self) -> Result<[u8; 32], String> {
        let mut out = [0u8; 32];
        unsafe { sbi_crypto::keccak256_dequeue(&mut out) }
            .map_err(|_| "keccak dequeue host call failed".to_string())?;
        Ok(out)
    }
}

struct HostLogger<'a, H>(&'a mut H);

impl<H: Runtime> Logger for HostLogger<'_, H> {
    fn log(&mut self, message: &str) {
        debug_msg!(self.0, "{message}");
    }
}

#[entrypoint::main]
pub fn entry(host: &mut impl Runtime) {
    let mut loader = DurableKeySpaceLoader::default();
    let crypto = SbiCrypto;
    let mut kernel =
        ChainKernel::new(&mut loader, &crypto).expect("Could not create tx-kernel durable context");

    while let Some(input) = host.read_input().expect("Could not read inbox message") {
        let (_, message) =
            InboxMessage::<MichelsonUnit>::parse(input.as_ref()).expect("Invalid inbox message");

        match message {
            InboxMessage::Internal(InternalInboxMessage::InfoPerLevel(_)) => {}
            InboxMessage::Internal(InternalInboxMessage::StartOfLevel) => {}
            InboxMessage::Internal(other) => {
                debug_msg!(host, "ignored internal inbox message: {other:?}\n");
            }
            InboxMessage::External(payload) => {
                let mut logger = HostLogger(host);
                kernel.handle_external_payload(&mut logger, &crypto, payload);
            }
        }
    }
}
