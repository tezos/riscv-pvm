// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::commit::CommitId;
use octez_riscv_durable_storage::database::Database;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_durable_storage::registry::Registry;

/// A Normal-mode registry backed by the persistent (RocksDB) key-value store.
pub type Reg = Registry<PersistenceLayer, Normal>;

/// Read a `usize` from the environment variable `name`, or fall back to `default`.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(default)
}

/// Build a registry of `database_count` databases, filling each with `populate`.
///
/// Returns the (checked-out, lazily-loaded) registry, whatever `populate` returned for each
/// database, and the commit id of the populated registry.
///
/// The registry is built one database at a time: after populating each database we commit and
/// check the registry back out, which flushes the just-built database to disk and reloads every
/// database lazily. This keeps only a single fully-materialised database tree in memory at a
/// time - populating all of them at once may exhaust memory.
pub fn populate_registry<T>(
    repo: &DirectoryManager,
    database_count: usize,
    mut populate: impl FnMut(usize, &mut Database<PersistenceLayer, Normal>) -> T,
) -> (Reg, Vec<T>, CommitId) {
    let mut outputs = Vec::with_capacity(database_count);

    let mut registry = Registry::new(repo.clone());
    let mut commit_id = None;

    for index in 0..database_count {
        // Reload the previously-built databases lazily so they do not sit fully materialised in
        // memory while we build the next one.
        if let Some(commit_id) = commit_id {
            registry = Reg::checkout(repo.clone(), commit_id)
                .expect("Checking out the partially-populated registry should succeed");
        }

        registry
            .resize_tick(index + 1)
            .expect("Growing the registry should succeed");

        let database = registry
            .database_mut(index)
            .expect("The database index should be valid");
        outputs.push(populate(index, database));

        commit_id = Some(
            registry
                .commit()
                .expect("Committing the partially-populated registry should succeed"),
        );
    }

    let commit_id = commit_id.expect("At least one database should have been populated");

    // Check out once more so the returned registry holds lazily-loaded (low-memory) trees.
    let registry = Reg::checkout(repo.clone(), commit_id)
        .expect("Checking out the populated registry should succeed");

    (registry, outputs, commit_id)
}
