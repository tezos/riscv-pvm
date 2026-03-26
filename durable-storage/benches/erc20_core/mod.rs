// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::hint::black_box;

use bytes::Bytes;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::database::Database;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::errors::Error;
use octez_riscv_durable_storage::errors::InvalidArgumentError;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use rand::rng;
use serde::Deserialize;
use tokio::runtime::Handle;

use crate::random::generate_keys;
use crate::random::generate_random_bytes;
use crate::random::generate_random_bytes_in_range;

const BLOCK_FREQUENCY: usize = 5_000;
const ERC_20_TRANSACTIONS: usize = 10_000;
const PREPOPULATED_NODE_KEYS_COUNT: usize = 10_000_000;

pub struct BenchmarkTemplate {
    pub(crate) operations: Vec<Operation>,
    pub(crate) random_data: Vec<u8>,
    pub(crate) read_buffer_len: usize,
}

pub struct BenchmarkState<'a> {
    pub database: Database<PersistenceLayer, Normal>,
    pub(crate) operations: Vec<Operation>,
    pub(crate) random_data: Vec<u8>,
    pub(crate) read_buffer: Vec<u8>,
    pub(crate) handle: &'a Handle,
    pub(crate) repo: &'a DirectoryManager,
}

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
enum SerialisedOperation {
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
struct StoreAccesses {
    setup: Vec<SerialisedOperation>,
    transaction: Vec<SerialisedOperation>,
    block_creation: Vec<SerialisedOperation>,
}

/// Builds the initial durable-storage database and the precomputed ERC-20
/// operation sequence used by the benchmarks.
///
/// The returned database contains the large prepopulated key set and any values
/// needed for read operations, while the returned template holds the
/// deterministic operation list and shared write buffer sizing information.
pub fn build_template(
    handle: &Handle,
    repo: &DirectoryManager,
) -> (BenchmarkTemplate, Database<PersistenceLayer, Normal>) {
    let mut database: Database<PersistenceLayer, _> =
        Database::try_new(handle, repo).expect("Creating a database should succeed");
    let mut rng = rng();

    // The performance of the replayed operations depends on the number of nodes
    // already present, so prepopulate the database up front before timing the
    // ERC-20-shaped workload.
    let keys = generate_keys(&mut rng, PREPOPULATED_NODE_KEYS_COUNT);
    for key in keys {
        let value = Bytes::from(generate_random_bytes_in_range(&mut rng, 1..32));
        database.set(key.clone(), value.clone()).ok();
    }

    // Load the recorded store-access trace and convert it into the normalised
    // operation representation used by the benchmark runner.
    let store_accesses_data = include_str!("store_accesses.json");
    let store_accesses: StoreAccesses = serde_json::from_str(store_accesses_data).expect(
        "The benchmark data should be a valid serialisation of the store accesses representation",
    );

    let erc_20_setup: Vec<Operation> = store_accesses
        .setup
        .into_iter()
        .map(Operation::from)
        .collect();
    let erc_20_transaction: Vec<Operation> = store_accesses
        .transaction
        .into_iter()
        .map(Operation::from)
        .collect();
    let erc_20_block_creation: Vec<Operation> = store_accesses
        .block_creation
        .into_iter()
        .map(Operation::from)
        .collect();

    // Read operations expect existing values, and the timed loop should not pay
    // for random-data generation or buffer sizing. Prepare all of that eagerly
    // so the benchmark measures storage behavior rather than setup work.
    let mut read_max_size = 0;
    let mut write_max_size = 0;
    for operation in erc_20_setup.iter().chain(
        erc_20_transaction
            .iter()
            .chain(erc_20_block_creation.iter()),
    ) {
        match operation {
            Operation::Read { key, size } => {
                database
                    .set(
                        key.clone(),
                        Bytes::from(generate_random_bytes(&mut rng, *size)),
                    )
                    .expect("The write should succeed");
                read_max_size = std::cmp::max(read_max_size, *size);
            }
            Operation::Write { size, .. } => {
                write_max_size = std::cmp::max(write_max_size, *size);
            }
            _ => (),
        }
    }

    // Expand the setup, transaction, and periodic block-creation phases into a
    // single deterministic operation list ahead of time so each Criterion
    // sample only replays the workload.
    let mut operations = Vec::new();
    operations.extend_from_slice(&erc_20_setup);
    for i in 0..ERC_20_TRANSACTIONS {
        operations.extend_from_slice(&erc_20_transaction);
        if i % BLOCK_FREQUENCY == 0 {
            operations.extend_from_slice(&erc_20_block_creation);
        }
    }

    // Random number generation is comparatively expensive, so prepare one write
    // buffer large enough for the maximum write and slice from it during replay.
    let random_data = generate_random_bytes(&mut rng, write_max_size);

    (
        BenchmarkTemplate {
            operations,
            random_data,
            read_buffer_len: read_max_size,
        },
        database,
    )
}

#[inline(never)]
/// Replays the prepared benchmark operations against a benchmark state.
///
/// The function performs no random generation or operation planning at runtime;
/// it only executes the already-expanded workload so Criterion measures storage
/// behavior rather than setup overhead.
pub fn bench_run(mut state: BenchmarkState<'_>) {
    // `KeyNotFound` errors are allowed. Arbitrary operations may delete keys used later.
    for operation in state.operations {
        match operation {
            Operation::Clone => {
                black_box(
                    state
                        .database
                        .try_clone_with(state.handle, state.repo)
                        .expect("The clone should succeed"),
                );
            }
            Operation::Delete { key } => match state.database.delete(key) {
                Ok(_) => {}
                Err(e) => panic!("The deletion should succeed: {e:?}"),
            },
            Operation::Exists { key } => {
                state
                    .database
                    .exists(&key)
                    .expect("The existence check should succeed");
            }
            Operation::Hash => {
                black_box(state.database.hash().expect("Hash should be calculated"));
            }
            Operation::Read { key, size } => {
                match state
                    .database
                    .read(&key, 0, &mut state.read_buffer[0..size])
                {
                    Ok(_) | Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
                    Err(e) => panic!("The read should succeed: {e:?}"),
                }
            }
            Operation::ValueLength { key } => match state.database.value_length(&key) {
                Ok(_) | Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
                Err(e) => panic!("The value length calculation should succeed: {e:?}"),
            },
            Operation::Write { key, size } => {
                match state
                    .database
                    .set(key, Bytes::copy_from_slice(&state.random_data[0..size]))
                {
                    Ok(_) | Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
                    Err(e) => panic!("The write should succeed: {e:?}"),
                }
            }
        }
    }
}
