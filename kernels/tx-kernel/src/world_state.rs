// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use crate::ContextStore;

pub const EVM_META_PREFIX: &[u8] = b"/evm/meta/";
pub const EVM_ACCOUNTS_PREFIX: &[u8] = b"/evm/accounts/";
pub const EVM_STORAGE_SUFFIX: &[u8] = b"/storage/";
pub const EVM_NONCE_SUFFIX: &[u8] = b"/nonce";
pub const EVM_BALANCE_SUFFIX: &[u8] = b"/balance";
pub const EVM_CODE_SUFFIX: &[u8] = b"/code";
pub const EVM_CODE_HASH_SUFFIX: &[u8] = b"/code_hash";
pub const EVM_CODE_BY_HASH_PREFIX: &[u8] = b"/evm/code_by_hash/";

pub const EVM_META_BOOTSTRAPPED_KEY: &[u8] = b"/evm/meta/bootstrapped";
pub const EVM_META_HEAD_KEY: &[u8] = b"/evm/meta/head";
pub const EVM_META_CHAIN_ID_KEY: &[u8] = b"/evm/meta/chain_id";
pub const EVM_META_BASE_FEE_KEY: &[u8] = b"/evm/meta/base_fee";
pub const EVM_META_BLOCK_GAS_LIMIT_KEY: &[u8] = b"/evm/meta/block_gas_limit";
pub const EVM_META_TIMESTAMP_KEY: &[u8] = b"/evm/meta/timestamp";
pub const EVM_META_SPEC_ID_KEY: &[u8] = b"/evm/meta/spec_id";

pub const DEFAULT_EVM_CHAIN_ID: u64 = 1_337;
pub const DEFAULT_EVM_BASE_FEE: u64 = 1_000_000_000;
pub const DEFAULT_EVM_BLOCK_GAS_LIMIT: u64 = 30_000_000;
pub const DEFAULT_EVM_TIMESTAMP: u64 = 0;
pub const DEFAULT_EVM_SPEC_ID: &[u8] = b"PRAGUE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvmAccountState {
    pub nonce: u64,
    pub balance: [u8; 32],
}

impl Default for EvmAccountState {
    fn default() -> Self {
        Self {
            nonce: 0,
            balance: [0u8; 32],
        }
    }
}

impl EvmAccountState {
    pub fn with_balance_u64(balance: u64) -> Self {
        Self {
            nonce: 0,
            balance: u64_to_be_u256(balance),
        }
    }
}

pub struct EvmWorldState<'a, C> {
    context: &'a mut C,
}

impl<'a, C: ContextStore> EvmWorldState<'a, C> {
    pub fn new(context: &'a mut C) -> Self {
        Self { context }
    }

    pub fn contains_account(&self, address: &[u8; 20]) -> Result<bool, String> {
        self.context.contains(&account_nonce_key(address))
    }

    pub fn read_account(&self, address: &[u8; 20]) -> Result<EvmAccountState, String> {
        Ok(EvmAccountState {
            nonce: self.read_nonce(address)?,
            balance: self.read_balance(address)?,
        })
    }

    pub fn write_account(
        &mut self,
        address: &[u8; 20],
        account: &EvmAccountState,
    ) -> Result<(), String> {
        self.write_nonce(address, account.nonce)?;
        self.write_balance(address, &account.balance)?;
        Ok(())
    }

    pub fn read_nonce(&self, address: &[u8; 20]) -> Result<u64, String> {
        match self.context.get(&account_nonce_key(address))? {
            Some(bytes) => decode_u64(&bytes, "invalid stored EVM nonce"),
            None => Ok(0),
        }
    }

    pub fn write_nonce(&mut self, address: &[u8; 20], nonce: u64) -> Result<(), String> {
        self.context
            .set(&account_nonce_key(address), &nonce.to_le_bytes())
    }

    pub fn read_balance(&self, address: &[u8; 20]) -> Result<[u8; 32], String> {
        match self.context.get(&account_balance_key(address))? {
            Some(bytes) => decode_u256(&bytes, "invalid stored EVM balance"),
            None => Ok([0u8; 32]),
        }
    }

    pub fn write_balance(&mut self, address: &[u8; 20], balance: &[u8; 32]) -> Result<(), String> {
        self.context.set(&account_balance_key(address), balance)
    }

    pub fn read_code(&self, address: &[u8; 20]) -> Result<Vec<u8>, String> {
        Ok(self
            .context
            .get(&account_code_key(address))?
            .unwrap_or_default())
    }

    pub fn write_code(&mut self, address: &[u8; 20], code: &[u8]) -> Result<(), String> {
        self.context.set(&account_code_key(address), code)
    }

    pub fn read_code_hash(&self, address: &[u8; 20]) -> Result<[u8; 32], String> {
        match self.context.get(&account_code_hash_key(address))? {
            Some(bytes) => decode_u256(&bytes, "invalid stored EVM code hash"),
            None => Ok([0u8; 32]),
        }
    }

    pub fn write_code_hash(
        &mut self,
        address: &[u8; 20],
        code_hash: &[u8; 32],
    ) -> Result<(), String> {
        self.context.set(&account_code_hash_key(address), code_hash)
    }

    pub fn read_code_by_hash(&self, code_hash: &[u8; 32]) -> Result<Vec<u8>, String> {
        Ok(self
            .context
            .get(&code_by_hash_key(code_hash))?
            .unwrap_or_default())
    }

    pub fn write_code_by_hash(&mut self, code_hash: &[u8; 32], code: &[u8]) -> Result<(), String> {
        self.context.set(&code_by_hash_key(code_hash), code)
    }

    pub fn read_storage(&self, address: &[u8; 20], slot: &[u8; 32]) -> Result<[u8; 32], String> {
        match self.context.get(&account_storage_key(address, slot))? {
            Some(bytes) => decode_u256(&bytes, "invalid stored EVM storage slot"),
            None => Ok([0u8; 32]),
        }
    }

    pub fn write_storage(
        &mut self,
        address: &[u8; 20],
        slot: &[u8; 32],
        value: &[u8; 32],
    ) -> Result<(), String> {
        self.context.set(&account_storage_key(address, slot), value)
    }

    pub fn read_meta_u64(&self, key: &[u8]) -> Result<u64, String> {
        match self.context.get(key)? {
            Some(bytes) => decode_u64(&bytes, "invalid EVM metadata value"),
            None => Ok(0),
        }
    }

    pub fn write_meta_u64(&mut self, key: &[u8], value: u64) -> Result<(), String> {
        self.context.set(key, &value.to_le_bytes())
    }

    pub fn read_meta_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        self.context.get(key)
    }

    pub fn write_meta_bytes(&mut self, key: &[u8], value: &[u8]) -> Result<(), String> {
        self.context.set(key, value)
    }
}

pub fn u64_to_be_u256(value: u64) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[24..].copy_from_slice(&value.to_be_bytes());
    bytes
}

pub fn account_nonce_key(address: &[u8; 20]) -> Vec<u8> {
    account_field_key(address, EVM_NONCE_SUFFIX)
}

pub fn account_balance_key(address: &[u8; 20]) -> Vec<u8> {
    account_field_key(address, EVM_BALANCE_SUFFIX)
}

pub fn account_code_key(address: &[u8; 20]) -> Vec<u8> {
    account_field_key(address, EVM_CODE_SUFFIX)
}

pub fn account_code_hash_key(address: &[u8; 20]) -> Vec<u8> {
    account_field_key(address, EVM_CODE_HASH_SUFFIX)
}

pub fn account_storage_key(address: &[u8; 20], slot: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(
        EVM_ACCOUNTS_PREFIX.len() + address.len() + EVM_STORAGE_SUFFIX.len() + slot.len(),
    );
    key.extend_from_slice(EVM_ACCOUNTS_PREFIX);
    key.extend_from_slice(address);
    key.extend_from_slice(EVM_STORAGE_SUFFIX);
    key.extend_from_slice(slot);
    key
}

pub fn code_by_hash_key(code_hash: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(EVM_CODE_BY_HASH_PREFIX.len() + code_hash.len());
    key.extend_from_slice(EVM_CODE_BY_HASH_PREFIX);
    key.extend_from_slice(code_hash);
    key
}

fn account_field_key(address: &[u8; 20], suffix: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(EVM_ACCOUNTS_PREFIX.len() + address.len() + suffix.len());
    key.extend_from_slice(EVM_ACCOUNTS_PREFIX);
    key.extend_from_slice(address);
    key.extend_from_slice(suffix);
    key
}

fn decode_u64(bytes: &[u8], message: &str) -> Result<u64, String> {
    let raw: [u8; 8] = bytes.try_into().map_err(|_| message.to_string())?;
    Ok(u64::from_le_bytes(raw))
}

fn decode_u256(bytes: &[u8], message: &str) -> Result<[u8; 32], String> {
    bytes.try_into().map_err(|_| message.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::ContextStore;

    #[derive(Clone, Default)]
    struct MemoryStore {
        values: BTreeMap<Vec<u8>, Vec<u8>>,
    }

    impl ContextStore for MemoryStore {
        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
            Ok(self.values.get(key).cloned())
        }

        fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), String> {
            self.values.insert(key.to_vec(), value.to_vec());
            Ok(())
        }

        fn contains(&self, key: &[u8]) -> Result<bool, String> {
            Ok(self.values.contains_key(key))
        }

        fn hash(&self) -> Result<Vec<u8>, String> {
            Ok(vec![])
        }
    }

    #[test]
    fn u64_balance_is_encoded_as_big_endian_u256() {
        let encoded = u64_to_be_u256(0x0102_0304_0506_0708);
        assert_eq!(&encoded[..24], &[0u8; 24]);
        assert_eq!(&encoded[24..], &0x0102_0304_0506_0708u64.to_be_bytes());
    }

    #[test]
    fn world_state_roundtrips_account_and_storage() {
        let mut store = MemoryStore::default();
        let mut world = EvmWorldState::new(&mut store);
        let address = [0x11; 20];
        let slot = [0x22; 32];
        let value = [0x33; 32];

        world
            .write_account(
                &address,
                &EvmAccountState {
                    nonce: 7,
                    balance: u64_to_be_u256(99),
                },
            )
            .unwrap();
        world.write_code(&address, &[0xaa, 0xbb]).unwrap();
        world.write_code_hash(&address, &[0x44; 32]).unwrap();
        world.write_storage(&address, &slot, &value).unwrap();

        assert_eq!(world.read_account(&address).unwrap().nonce, 7);
        assert_eq!(
            world.read_account(&address).unwrap().balance,
            u64_to_be_u256(99)
        );
        assert_eq!(world.read_code(&address).unwrap(), vec![0xaa, 0xbb]);
        assert_eq!(world.read_code_hash(&address).unwrap(), [0x44; 32]);
        assert_eq!(world.read_storage(&address, &slot).unwrap(), value);
    }
}
