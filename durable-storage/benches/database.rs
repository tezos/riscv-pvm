// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

mod random;

use std::hint::black_box;

use bytes::Bytes;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::database::Database;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::errors::Error;
use octez_riscv_durable_storage::errors::InvalidArgumentError;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_test_utils::TestableTmpdir;
use rand::rng;
use random::generate_keys;
use random::generate_random_bytes;
use random::generate_random_bytes_in_range;
use serde::Deserialize;
use tokio::runtime::Handle;

struct BenchmarkState<'a> {
    database: Database<PersistenceLayer, Normal>,
    operations: Vec<Operation>,
    random_data: Vec<u8>,
    read_buffer: Vec<u8>,
    handle: &'a Handle,
    repo: &'a DirectoryManager,
}

impl<'a> BenchmarkState<'a> {
    fn clone_with(&self, handle: &'a Handle, repo: &'a DirectoryManager) -> Self {
        Self {
            database: self
                .database
                .try_clone_with(handle, repo)
                .expect("Cloning the database should work"),
            operations: self.operations.clone(),
            read_buffer: vec![0u8; self.read_buffer.len()],
            random_data: self.random_data.clone(),
            handle,
            repo,
        }
    }
}

#[derive(Clone)]
enum Operation {
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

const BLOCK_FREQUENCY: usize = 5_000;
const ERC_20_TRANSACTIONS: usize = 10_000;
const PREPOPULATED_NODE_KEYS_COUNT: usize = 10_000_000;

fn setup_benchmark_state<'a>(handle: &'a Handle, repo: &'a DirectoryManager) -> BenchmarkState<'a> {
    let mut database: Database<PersistenceLayer, _> =
        Database::try_new(handle, repo).expect("Creating a database should succeed");
    let mut rng = rng();

    // The performance of operations depends on the number of nodes already stored
    let keys = generate_keys(&mut rng, PREPOPULATED_NODE_KEYS_COUNT);
    for key in keys {
        let value = Bytes::from(generate_random_bytes_in_range(&mut rng, 1..32));
        database.set(key.clone(), value.clone()).ok();
    }

    // Deserialise a series of operations describing an ERC-20 transaction
    let store_accesses_data = include_str!("store_accesses.json");
    let store_accesses: StoreAccesses = serde_json::from_str(store_accesses_data).expect(
        "The benchmark data should be a valid serialisation of the store accesses representation",
    );

    // Convert the deserialised form into a sequence of normalised operations
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

    // Populate nodes for read operations where keys are expected to exist and find the maximum
    // sizes that will be read and written
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
            Operation::Write { key: _, size } => {
                write_max_size = std::cmp::max(write_max_size, *size);
            }
            _ => (),
        }
    }

    // Unroll the operations into a sequence including setup, ERC_20_TRANSACTIONS*transactions, and
    // block creation with hashing
    let mut operations = Vec::with_capacity(
        erc_20_setup.len()
            + ERC_20_TRANSACTIONS * erc_20_transaction.len()
            + (ERC_20_TRANSACTIONS / BLOCK_FREQUENCY) * (erc_20_block_creation.len() + 1),
    );
    operations.extend_from_slice(&erc_20_setup);
    for i in 0..ERC_20_TRANSACTIONS {
        operations.extend_from_slice(&erc_20_transaction);
        if i % BLOCK_FREQUENCY == 0 {
            operations.extend_from_slice(&erc_20_block_creation);
        }
    }

    // Prepare random data to be used for write operations
    //
    // Random number generation is too expensive to be performed in the `bench_run` loop
    let random_data = generate_random_bytes(&mut rng, write_max_size);

    BenchmarkState {
        database,
        operations,
        random_data,
        read_buffer: vec![0u8; read_max_size],
        handle,
        repo,
    }
}

#[inline(never)]
fn bench_run(mut state: BenchmarkState) {
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
                Ok(_) | Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => {}
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

fn database_benchmark(c: &mut Criterion) {
    // Two threads are needed so the thread for receiving the hashes isn't blocked
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("Creating a Tokio runtime should succeed");

    let tmpdir = TestableTmpdir::new();
    let repo = DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
    let initial_state = setup_benchmark_state(runtime.handle(), &repo);

    let setup = || initial_state.clone_with(runtime.handle(), &repo);

    let mut group = c.benchmark_group("ERC-20");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(120));
    group.bench_function("10_000-transactions", |b| {
        b.iter_batched(setup, bench_run, BatchSize::SmallInput)
    });

    group.finish();
}

criterion_group!(benches, database_benchmark);
criterion_main!(benches);
