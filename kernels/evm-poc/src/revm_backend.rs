// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use revm::context::BlockEnv;
use revm::context::TxEnv;
use revm::database_interface::DBErrorMarker;
use revm::database_interface::Database;
use revm::database_interface::DatabaseCommit;
use revm::primitives::Address;
use revm::primitives::B256;
use revm::primitives::Bytes;
use revm::primitives::HashMap;
use revm::primitives::TxKind;
use revm::primitives::U256;
use revm::primitives::hardfork::SpecId;
use revm::state::Account;
use revm::state::AccountInfo;
use revm::state::Bytecode;

use crate::ContextStore;
use crate::DEFAULT_EVM_BASE_FEE;
use crate::DEFAULT_EVM_BLOCK_GAS_LIMIT;
use crate::DEFAULT_EVM_CHAIN_ID;
use crate::DEFAULT_EVM_TIMESTAMP;
use crate::EVM_META_BASE_FEE_KEY;
use crate::EVM_META_BLOCK_GAS_LIMIT_KEY;
use crate::EVM_META_CHAIN_ID_KEY;
use crate::EVM_META_HEAD_KEY;
use crate::EVM_META_SPEC_ID_KEY;
use crate::EVM_META_TIMESTAMP_KEY;
use crate::EthereumTransaction;
use crate::EvmWorldState;

#[derive(Debug, Clone)]
pub struct RevmContextDb<C> {
    context: C,
}

#[derive(Debug, Clone)]
pub struct RevmDbError(pub String);

impl core::fmt::Display for RevmDbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RevmDbError {}

impl DBErrorMarker for RevmDbError {}

impl From<String> for RevmDbError {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl<C: ContextStore> RevmContextDb<C> {
    pub fn new(context: C) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &C {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut C {
        &mut self.context
    }
}

impl<C: ContextStore> Database for RevmContextDb<C> {
    type Error = RevmDbError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let world = EvmWorldState::new(&mut self.context);
        let address_bytes: [u8; 20] = address.into();
        let account = world
            .read_account(&address_bytes)
            .map_err(RevmDbError::from)?;
        let code_hash_bytes = world
            .read_code_hash(&address_bytes)
            .map_err(RevmDbError::from)?;
        let code = world.read_code(&address_bytes).map_err(RevmDbError::from)?;

        if account.nonce == 0
            && account.balance == [0u8; 32]
            && code_hash_bytes == [0u8; 32]
            && code.is_empty()
        {
            return Ok(None);
        }

        let bytecode = if code.is_empty() {
            Bytecode::default()
        } else {
            Bytecode::new_raw(Bytes::from(code))
        };

        Ok(Some(AccountInfo::new(
            u256_from_be_bytes(&account.balance),
            account.nonce,
            B256::from(code_hash_bytes),
            bytecode,
        )))
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        let world = EvmWorldState::new(&mut self.context);
        let code_hash_bytes: [u8; 32] = code_hash.into();
        let code = world
            .read_code_by_hash(&code_hash_bytes)
            .map_err(RevmDbError::from)?;
        if code.is_empty() {
            Ok(Bytecode::default())
        } else {
            Ok(Bytecode::new_raw(Bytes::from(code)))
        }
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let world = EvmWorldState::new(&mut self.context);
        let address_bytes: [u8; 20] = address.into();
        let slot = b256_to_bytes32(index_to_b256(index));
        let value = world
            .read_storage(&address_bytes, &slot)
            .map_err(RevmDbError::from)?;
        Ok(u256_from_be_bytes(&value))
    }

    fn block_hash(&mut self, _number: u64) -> Result<B256, Self::Error> {
        Ok(B256::ZERO)
    }
}

impl<C: ContextStore> DatabaseCommit for RevmContextDb<C> {
    fn commit(&mut self, changes: HashMap<Address, Account>) {
        let _ = commit_changes(&mut self.context, changes);
    }
}

pub fn commit_changes<C: ContextStore>(
    context: &mut C,
    changes: HashMap<Address, Account>,
) -> Result<(), String> {
    let mut world = EvmWorldState::new(context);

    for (address, account) in changes {
        let address_bytes: [u8; 20] = address.into();

        world.write_nonce(&address_bytes, account.info.nonce)?;
        let balance_bytes = u256_to_be_bytes(account.info.balance);
        world.write_balance(&address_bytes, &balance_bytes)?;

        let code_hash_bytes: [u8; 32] = account.info.code_hash.into();
        world.write_code_hash(&address_bytes, &code_hash_bytes)?;

        let code_bytes = account
            .info
            .code
            .as_ref()
            .map(|code| code.original_byte_slice().to_vec())
            .unwrap_or_default();
        world.write_code(&address_bytes, &code_bytes)?;
        if !code_bytes.is_empty() {
            world.write_code_by_hash(&code_hash_bytes, &code_bytes)?;
        }

        for (slot, value) in account.changed_storage_slots() {
            let slot_bytes = slot.to_be_bytes();
            let value_bytes = u256_to_be_bytes(value.present_value());
            world.write_storage(&address_bytes, &slot_bytes, &value_bytes)?;
        }
    }

    Ok(())
}

pub fn build_revm_block_env<C: ContextStore>(context: &mut C) -> Result<BlockEnv, String> {
    let world = EvmWorldState::new(context);
    Ok(BlockEnv {
        number: U256::from(world.read_meta_u64(EVM_META_HEAD_KEY)?),
        beneficiary: Address::ZERO,
        timestamp: U256::from(
            world
                .read_meta_u64(EVM_META_TIMESTAMP_KEY)?
                .max(DEFAULT_EVM_TIMESTAMP),
        ),
        gas_limit: {
            let gas_limit = world.read_meta_u64(EVM_META_BLOCK_GAS_LIMIT_KEY)?;
            if gas_limit == 0 {
                DEFAULT_EVM_BLOCK_GAS_LIMIT
            } else {
                gas_limit
            }
        },
        basefee: {
            let base_fee = world.read_meta_u64(EVM_META_BASE_FEE_KEY)?;
            if base_fee == 0 {
                DEFAULT_EVM_BASE_FEE
            } else {
                base_fee
            }
        },
        difficulty: U256::ZERO,
        prevrandao: Some(B256::ZERO),
        blob_excess_gas_and_price: BlockEnv::default().blob_excess_gas_and_price,
    })
}

pub fn revm_spec_id<C: ContextStore>(context: &mut C) -> Result<SpecId, String> {
    let world = EvmWorldState::new(context);
    let spec = world.read_meta_bytes(EVM_META_SPEC_ID_KEY)?;
    Ok(match spec.as_deref() {
        Some(b"FRONTIER") => SpecId::FRONTIER,
        Some(b"BERLIN") => SpecId::BERLIN,
        Some(b"LONDON") => SpecId::LONDON,
        Some(b"SHANGHAI") => SpecId::SHANGHAI,
        Some(b"CANCUN") => SpecId::CANCUN,
        Some(b"PRAGUE") | None => SpecId::PRAGUE,
        _ => SpecId::PRAGUE,
    })
}

pub fn build_revm_tx_env(
    caller: [u8; 20],
    transaction: &EthereumTransaction,
) -> Result<TxEnv, String> {
    match transaction {
        EthereumTransaction::Eip1559(tx) => {
            let mut env = TxEnv::default();
            env.tx_type = 2;
            env.caller = Address::from(caller);
            env.gas_limit = tx.gas_limit;
            env.gas_price = tx.max_fee_per_gas as u128;
            env.gas_priority_fee = Some(tx.max_priority_fee_per_gas as u128);
            env.kind = match tx.to {
                Some(address) => TxKind::Call(Address::from(address)),
                None => TxKind::Create,
            };
            env.value = u256_from_be_bytes(&tx.value);
            env.data = Bytes::from(tx.data.clone());
            env.nonce = tx.nonce;
            env.chain_id = Some(tx.chain_id);
            env.access_list = revm::context_interface::transaction::AccessList(
                tx.access_list
                    .iter()
                    .map(
                        |item| revm::context_interface::transaction::AccessListItem {
                            address: Address::from(item.address),
                            storage_keys: item
                                .storage_keys
                                .iter()
                                .copied()
                                .map(B256::from)
                                .collect(),
                        },
                    )
                    .collect(),
            );
            Ok(env)
        }
    }
}

pub fn evm_chain_id<C: ContextStore>(context: &mut C) -> Result<u64, String> {
    let world = EvmWorldState::new(context);
    let chain_id = world.read_meta_u64(EVM_META_CHAIN_ID_KEY)?;
    Ok(if chain_id == 0 {
        DEFAULT_EVM_CHAIN_ID
    } else {
        chain_id
    })
}

fn u256_from_be_bytes(bytes: &[u8; 32]) -> U256 {
    U256::from_be_slice(bytes)
}

fn u256_to_be_bytes(value: U256) -> [u8; 32] {
    value.to_be_bytes()
}

fn index_to_b256(index: U256) -> B256 {
    B256::from(index.to_be_bytes())
}

fn b256_to_bytes32(value: B256) -> [u8; 32] {
    value.into()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use revm::database_interface::Database;
    use revm::primitives::Address;
    use revm::primitives::U256;

    use super::*;
    use crate::ContextStore;
    use crate::Eip1559Transaction;
    use crate::u64_to_be_u256;

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
    fn database_basic_reads_account_from_world_state() {
        let mut store = MemoryStore::default();
        let address = [0x11; 20];
        {
            let mut world = EvmWorldState::new(&mut store);
            world.write_nonce(&address, 9).unwrap();
            world.write_balance(&address, &u64_to_be_u256(42)).unwrap();
            world.write_code_hash(&address, &[0x22; 32]).unwrap();
            world.write_code(&address, &[0x60, 0x00]).unwrap();
            world
                .write_code_by_hash(&[0x22; 32], &[0x60, 0x00])
                .unwrap();
        }

        let mut db = RevmContextDb::new(store);
        let info = db.basic(Address::from(address)).unwrap().unwrap();
        assert_eq!(info.nonce, 9);
        assert_eq!(info.balance, U256::from(42u64));
        assert_eq!(info.code_hash, B256::from([0x22; 32]));
    }

    #[test]
    fn tx_env_builder_maps_eip1559_fields() {
        let transaction = EthereumTransaction::Eip1559(Eip1559Transaction {
            chain_id: 1337,
            nonce: 3,
            max_priority_fee_per_gas: 10,
            max_fee_per_gas: 20,
            gas_limit: 21_000,
            to: Some([0x33; 20]),
            value: u64_to_be_u256(7),
            data: vec![0xaa],
            access_list: vec![],
            signature_y_parity: 0,
            signature_r: [0u8; 32],
            signature_s: [0u8; 32],
        });

        let tx_env = build_revm_tx_env([0x44; 20], &transaction).unwrap();
        assert_eq!(tx_env.caller, Address::from([0x44; 20]));
        assert_eq!(tx_env.gas_limit, 21_000);
        assert_eq!(tx_env.gas_price, 20);
        assert_eq!(tx_env.nonce, 3);
        assert_eq!(tx_env.value, U256::from(7u64));
    }
}
