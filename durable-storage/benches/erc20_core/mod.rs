// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::hint::black_box;

use bytes::Bytes;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::database::Database;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use rand::rng;
use tokio::runtime::Handle;

use crate::erc20_trace::Operation;
use crate::erc20_trace::apply_operation;
use crate::erc20_trace::database_write_all;
use crate::erc20_trace::load_store_accesses;
use crate::random::generate_keys;
use crate::random::generate_random_bytes;
use crate::random::generate_random_bytes_in_range;

// Default values that can be overridden by passing environment variables.
const BLOCK_FREQUENCY: usize = 5_000;
const ERC_20_TRANSACTION_COUNT: usize = 10_000;
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
    let block_frequency: usize = std::env::var("BLOCK_FREQUENCY")
        .ok()
        .map(|v| v.parse().expect("BLOCK_FREQUENCY must be a number"))
        .unwrap_or(BLOCK_FREQUENCY);

    let transaction_count: usize = std::env::var("ERC_20_TRANSACTION_COUNT")
        .ok()
        .map(|v| {
            v.parse()
                .expect("ERC_20_TRANSACTION_COUNT must be a number")
        })
        .unwrap_or(ERC_20_TRANSACTION_COUNT);

    let node_keys_count: usize = std::env::var("PREPOPULATED_NODE_KEYS_COUNT")
        .ok()
        .map(|v| {
            v.parse()
                .expect("PREPOPULATED_NODE_KEYS_COUNT must be a number")
        })
        .unwrap_or(PREPOPULATED_NODE_KEYS_COUNT);
    let mut database: Database<PersistenceLayer, _> =
        Database::try_new(handle, repo).expect("Creating a database should succeed");
    let mut rng = rng();

    // The performance of the replayed operations depends on the number of nodes
    // already present, so prepopulate the database up front before timing the
    // ERC-20-shaped workload.
    let keys = generate_keys(&mut rng, node_keys_count);
    for key in keys {
        let value = Bytes::from(generate_random_bytes_in_range(&mut rng, 1..32));
        database.set(key.clone(), value.clone()).ok();
    }

    // Load the recorded store-access trace and convert it into the normalised
    // operation representation used by the benchmark runner.
    let store_accesses = load_store_accesses();

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
                let value = generate_random_bytes(&mut rng, *size);
                database_write_all(&mut database, key.clone(), &value)
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
    for i in 0..transaction_count {
        operations.extend_from_slice(&erc_20_transaction);
        if block_frequency > 0 && i % block_frequency == 0 {
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
            Operation::Hash => {
                black_box(state.database.hash().expect("Hash should be calculated"));
            }
            operation => apply_operation(
                &mut state.database,
                operation,
                &mut state.read_buffer,
                &state.random_data,
            ),
        }
    }
}
