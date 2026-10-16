// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! The recorded ERC-20 store-access trace and the helpers that replay it against a database.

use bytes::Bytes;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::database::Database;
use octez_riscv_durable_storage::errors::Error;
use octez_riscv_durable_storage::errors::InvalidArgumentError;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use serde::Deserialize;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

#[derive(Clone)]
pub(crate) enum Operation {
    Clone,
    Delete { key: Key },
    Exists { key: Key },
    Hash,
    Read { key: Key, size: usize },
    ValueLength { key: Key },
    Write { key: Key, size: usize },
}

impl From<SerialisedOperation> for Operation {
    fn from(serialised_operation: SerialisedOperation) -> Self {
        match serialised_operation {
            SerialisedOperation::Copy { .. } => Operation::Clone,
            SerialisedOperation::Delete { path } => Operation::Delete {
                key: Key::new(path.as_bytes()).expect("The path should be a valid key"),
            },
            SerialisedOperation::Has { path } => Operation::Exists {
                key: Key::new(path.as_bytes()).expect("The path should be a valid key"),
            },
            SerialisedOperation::Hash { path: _ } => Operation::Hash,
            SerialisedOperation::Read { path, size }
            | SerialisedOperation::ReadAll { path, size }
            | SerialisedOperation::ReadSlice { path, size } => Operation::Read {
                key: Key::new(path.as_bytes()).expect("The path should be a valid key"),
                size,
            },
            SerialisedOperation::ValueSize { path } => Operation::ValueLength {
                key: Key::new(path.as_bytes()).expect("The path should be a valid key"),
            },
            SerialisedOperation::Write { path, size }
            | SerialisedOperation::WriteAll { path, size } => Operation::Write {
                key: Key::new(path.as_bytes()).expect("The path should be a valid key"),
                size,
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub(crate) enum SerialisedOperation {
    #[serde(rename = "store_copy")]
    #[expect(dead_code, reason = "The result of the copy is ignored")]
    Copy { from_path: String, to_path: String },
    #[serde(rename = "store_delete")]
    Delete { path: String },
    #[serde(rename = "store_has")]
    Has { path: String },
    #[serde(rename = "__internal_store_get_hash")]
    #[expect(dead_code, reason = "The path of the hash is ignored")]
    Hash { path: String },
    #[serde(rename = "store_read")]
    Read { path: String, size: usize },
    #[serde(rename = "store_read_all")]
    ReadAll { path: String, size: usize },
    #[serde(rename = "store_read_slice")]
    ReadSlice { path: String, size: usize },
    #[serde(rename = "store_value_size")]
    ValueSize { path: String },
    #[serde(rename = "store_write")]
    Write { path: String, size: usize },
    #[serde(rename = "store_write_all")]
    WriteAll { path: String, size: usize },
}

#[derive(Deserialize)]
pub(crate) struct StoreAccesses {
    pub(crate) setup: Vec<SerialisedOperation>,
    pub(crate) transaction: Vec<SerialisedOperation>,
    pub(crate) block_creation: Vec<SerialisedOperation>,
}

/// Load the recorded ERC-20 store-access trace.
pub(crate) fn load_store_accesses() -> StoreAccesses {
    let store_accesses_data = include_str!("../erc20_core/store_accesses.json");
    serde_json::from_str(store_accesses_data).expect(
        "The benchmark data should be a valid serialisation of the store accesses representation",
    )
}

/// Apply a single replayed operation to `database`. Reads go to `read_buffer` and writes take
/// their data from `random_data`.
///
/// [`Operation::Clone`] and [`Operation::Hash`] need more than the database and must be handled
/// by the caller.
pub(crate) fn apply_operation(
    database: &mut Database<PersistenceLayer, Normal>,
    operation: Operation,
    read_buffer: &mut [u8],
    random_data: &[u8],
) {
    // `KeyNotFound` errors are allowed. Arbitrary operations may delete keys used later.
    match operation {
        Operation::Clone | Operation::Hash => {
            unreachable!("Clones and hashes are handled by the caller")
        }
        Operation::Delete { key } => match database.delete(key) {
            Ok(_) => {}
            Err(e) => panic!("The deletion should succeed: {e:?}"),
        },
        Operation::Exists { key } => {
            database
                .exists(&key)
                .expect("The existence check should succeed");
        }
        Operation::Read { key, size } => {
            match database_read(database, key, size, &mut read_buffer[..size]) {
                Ok(read) => {
                    assert!(read <= size, "Read more than {size} bytes from storage");
                }
                Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
                Err(e) => panic!("The read should succeed: {e:?}"),
            }
        }
        Operation::ValueLength { key } => match database.value_length(&key) {
            Ok(_) | Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
            Err(e) => panic!("The value length calculation should succeed: {e:?}"),
        },
        Operation::Write { key, size } => {
            match database_write_all(database, key, &random_data[..size]) {
                Ok(()) | Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
                Err(e) => panic!("The write should succeed: {e:?}"),
            }
        }
    }
}

/// Read an entire value (possibly larger than [`MAX_FILE_CHUNK_SIZE`]) from the given database.
///
/// The database API only supports reading chunks less than `MAX_FILE_CHUNK_SIZE` in length in one
/// go, for proof safety. This helper exists to read larger values from storage, similarly to how
/// a kernel might.
fn database_read(
    database: &mut Database<PersistenceLayer, Normal>,
    key: Key,
    size: usize,
    buffer: &mut [u8],
) -> Result<usize, Error> {
    let mut offset = 0;
    let mut remaining = size;

    assert!(remaining <= buffer.len(), "{remaining} <= {}", buffer.len());

    while remaining > 0 {
        let chunk = remaining.min(MAX_FILE_CHUNK_SIZE);
        database.read(&key, offset, &mut buffer[offset..(offset + chunk)])?;

        offset += chunk;
        remaining -= chunk;
    }

    Ok(offset)
}

/// Write an entire value (possibly larger than [`MAX_FILE_CHUNK_SIZE`]) to the given database.
///
/// The database API only supports writing chunks less than `MAX_FILE_CHUNK_SIZE` in length in one
/// go, for proof safety. This helper exists to write larger values to storage, similarly to how
/// a kernel might.
pub(crate) fn database_write_all(
    database: &mut Database<PersistenceLayer, Normal>,
    key: Key,
    data: &[u8],
) -> Result<(), Error> {
    if data.is_empty() {
        return database.set(key, Bytes::new());
    }

    let set_len = data.len().min(MAX_FILE_CHUNK_SIZE);
    database.set(key.clone(), Bytes::copy_from_slice(&data[..set_len]))?;

    let mut offset = set_len;
    while offset < data.len() {
        let end = (offset + MAX_FILE_CHUNK_SIZE).min(data.len());
        database.write(
            key.clone(),
            offset,
            Bytes::copy_from_slice(&data[offset..end]),
        )?;
        offset = end;
    }
    Ok(())
}
