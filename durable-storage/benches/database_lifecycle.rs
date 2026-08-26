// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Focused lifecycle benchmarks for the durable-storage [`Registry`].
//!
//! The new durable storage is a [`Registry`] owning a vector of
//! [`Database`]s, so these benchmarks operate on the registry — not a bare
//! database — to capture the registry-level costs the PVM actually pays, in
//! particular the registry hash taken on every commit and re-derived on every
//! checkout.
//!
//! Unlike the ERC-20 workload benchmarks, these replay no store-access trace.
//! Each isolates one whole-database/registry lifecycle operation on a large,
//! prepopulated registry (`REGISTRY_DATABASE_COUNT` databases of
//! `PREPOPULATED_NODE_KEYS_COUNT` keys each):
//!
//! - **Copy** — `copy_database`: clone one large database over another. Times
//!   the clone plus the drop of the overwritten database.
//! - **Move** — `move_database`: as copy, but the source is left a fresh empty
//!   database, so the gap from copy is the clone.
//! - **Clear** — `clear_database`: replace a large database with an empty one.
//! - **Commit** — `Registry::commit`: with nothing dirty (the floor cost of a
//!   commit) and with `COMMIT_MODIFIED_KEYS_COUNT` modifications, which *settle
//!   the writes* (see below) so that the hashing — but not the execution work —
//!   falls inside the measurement.
//! - **Hash** — `Database::hash`: on a clean database, whose root hash the
//!   checkout cached, and on one with settled modifications, where the dirty
//!   paths must be rehashed.
//! - **Checkout** — `Registry::checkout`: restore the whole registry from a
//!   committed snapshot.
//!
//! An operation that takes a source database is measured twice: over a clean
//! one, and over one carrying uncommitted modifications.
//!
//! All scenarios use [`BatchSize::PerIteration`], so only one large registry is
//! live at a time and the teardown drop of the value *returned* from the routine
//! is untimed — only drops *inside* the routine are measured.
//!
//! **Settling the writes.** `Database::set` enqueues asynchronous Merkle-worker
//! commands, and the only worker sync points are `hash`, `commit` and `clone`.
//! An operation timed straight after un-settled writes would absorb the
//! *application* of those writes — execution work, not the operation's own. A
//! clone drains the worker without hashing, so every setup that modifies clones
//! once afterwards, leaving the tree dirty and unhashed.
//!
//! The sample size and measurement time default to small, stable values;
//! `--sample-size` and `--measurement-time` override them.
//!
//! Populating a large registry is slow. Set `LIFECYCLE_REGISTRY_DIR` to a path
//! on a **large filesystem** to persist the prepopulated registry there and
//! reuse it across runs. The manifest saved alongside it records the config it
//! was built with, so a mismatched directory is rebuilt rather than misused.
//! Unset → ephemeral temp dir, populate every run.
//!
//! [`Database`]: octez_riscv_durable_storage::database::Database

mod random;

use std::hint::black_box;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use bytes::Bytes;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::commit::CommitId;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_durable_storage::registry::Registry;
use octez_riscv_test_utils::TestableTmpdir;
use rand::prelude::*;
use rand::rng;
use serde::Deserialize;
use serde::Serialize;

use crate::random::generate_keys;
use crate::random::generate_random_bytes_in_range;

/// A Normal-mode registry backed by the persistent (RocksDB) key-value store.
type Reg = Registry<PersistenceLayer, Normal>;

/// Number of keys to prepopulate *each* database with, controlling how "large"
/// the databases under test are. Override with the `PREPOPULATED_NODE_KEYS_COUNT`
/// environment variable (shared with the ERC-20 database benchmarks).
const PREPOPULATED_NODE_KEYS_COUNT: usize = 10_000_000;

/// Number of databases held in the registry. Override with the
/// `REGISTRY_DATABASE_COUNT` environment variable.
const REGISTRY_DATABASE_COUNT: usize = 4;

/// Number of randomly-chosen keys the commit benchmark modifies (across all
/// databases) before committing. Override with `COMMIT_MODIFIED_KEYS_COUNT`.
const COMMIT_MODIFIED_KEYS_COUNT: usize = 10_000;

/// Number of keys retained per database for the modification scenarios. Keeping
/// *every* key (up to `PREPOPULATED_NODE_KEYS_COUNT` per database, each up to
/// [`KEY_MAX_SIZE`] bytes) would hold gigabytes of keys in memory for the whole
/// run; a bounded random sample is enough to pick keys to modify.
///
/// [`KEY_MAX_SIZE`]: octez_riscv_durable_storage::key::KEY_MAX_SIZE
const KEY_SAMPLE_PER_DB: usize = 100_000;

/// File name (within `LIFECYCLE_REGISTRY_DIR`) of the manifest describing a
/// prepopulated registry that can be reused across runs.
const PREPOPULATED_MANIFEST_FILE: &str = "prepopulated_lifecycle.json";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(default)
}

/// Manifest persisted alongside a prepopulated registry so a later run can skip
/// the (slow) populate phase.
#[derive(Serialize, Deserialize)]
struct PrepopulatedManifest {
    commit_id: CommitId,
    database_count: usize,
    node_keys_count: usize,
    key_sample_per_db: usize,
    all_keys: Vec<Vec<Key>>,
}

/// Try to reuse a registry prepopulated by a previous run: read the manifest at
/// `manifest_path`, and if it exists and matches the requested config, check the
/// registry out of `repo`. Returns `None` (→ caller populates from scratch) if
/// the manifest is missing, unreadable, config-mismatched, or the checkout fails.
fn try_load_prepopulated(
    repo: &DirectoryManager,
    manifest_path: &Path,
    database_count: usize,
    node_keys_count: usize,
) -> Option<(Reg, Vec<Vec<Key>>, CommitId)> {
    let bytes = std::fs::read(manifest_path).ok()?;
    let manifest: PrepopulatedManifest = serde_json::from_slice(&bytes).ok()?;
    if manifest.database_count != database_count
        || manifest.node_keys_count != node_keys_count
        || manifest.key_sample_per_db != KEY_SAMPLE_PER_DB
    {
        eprintln!(
            "Prepopulated registry at {} has a different config; repopulating.",
            manifest_path.display()
        );
        return None;
    }
    let source = Reg::checkout(repo.clone(), manifest.commit_id).ok()?;
    eprintln!(
        "Reusing prepopulated registry at {} (skipping populate).",
        manifest_path.display()
    );
    Some((source, manifest.all_keys, manifest.commit_id))
}

/// Persist a prepopulated registry's manifest next to its committed data so the
/// next run can reuse it.
fn save_prepopulated(
    manifest_path: &Path,
    commit_id: CommitId,
    all_keys: &[Vec<Key>],
    database_count: usize,
    node_keys_count: usize,
) {
    let manifest = PrepopulatedManifest {
        commit_id,
        database_count,
        node_keys_count,
        key_sample_per_db: KEY_SAMPLE_PER_DB,
        all_keys: all_keys.to_vec(),
    };
    let bytes = serde_json::to_vec(&manifest)
        .expect("Serialising the prepopulated manifest should succeed");
    std::fs::write(manifest_path, bytes).expect("Writing the prepopulated manifest should succeed");
}

/// Build a large registry: `REGISTRY_DATABASE_COUNT` databases, each
/// prepopulated with `PREPOPULATED_NODE_KEYS_COUNT` random keys/values.
///
/// Returns the (checked-out, lazily-loaded) registry, the keys of each database
/// — so the commit benchmark can modify existing entries — and the commit id of
/// the populated registry, used by the checkout benchmark.
///
/// The registry is built one database at a time: after populating each database
/// we commit and check the registry back out, which flushes the just-built
/// database to disk and reloads every database lazily. This keeps only a single
/// fully-materialised database tree in memory at a time — populating all of them
/// at once may exhaust memory.
fn populate_registry(repo: &DirectoryManager) -> (Reg, Vec<Vec<Key>>, CommitId) {
    let database_count = env_usize("REGISTRY_DATABASE_COUNT", REGISTRY_DATABASE_COUNT);
    let node_keys_count = env_usize("PREPOPULATED_NODE_KEYS_COUNT", PREPOPULATED_NODE_KEYS_COUNT);

    let mut rng = rng();
    let mut all_keys = Vec::with_capacity(database_count);

    let mut registry = Registry::new(repo.clone());
    let mut commit_id = None;

    for index in 0..database_count {
        // Reload the previously-built databases lazily so they do not sit fully
        // materialised in memory while we build the next one. (The very first
        // database starts from the empty in-memory registry above.)
        if let Some(commit_id) = commit_id {
            registry = Reg::checkout(repo.clone(), commit_id)
                .expect("Checking out the partially-populated registry should succeed");
        }

        registry
            .resize_tick(index + 1)
            .expect("Growing the registry should succeed");

        let keys = generate_keys(&mut rng, node_keys_count);
        {
            let database = registry
                .database_mut(index)
                .expect("The database index should be valid");
            for key in &keys {
                let value = Bytes::from(generate_random_bytes_in_range(&mut rng, 1..32));
                database.set(key.clone(), value).ok();
            }
        }
        // Retain only a bounded random sample of keys for the modification
        // scenarios; the keys are already random, so the first `KEY_SAMPLE_PER_DB`
        // are a fine sample. Keeping all of them would cost gigabytes.
        all_keys.push(keys.into_iter().take(KEY_SAMPLE_PER_DB).collect());

        commit_id = Some(
            registry
                .commit()
                .expect("Committing the partially-populated registry should succeed"),
        );
    }

    let commit_id = commit_id.expect("At least one database should have been populated");

    // Check out once more so the returned source holds lazily-loaded (low-memory)
    // trees, matching the state the scenarios clone from.
    let source = Reg::checkout(repo.clone(), commit_id)
        .expect("Checking out the populated registry should succeed");

    (source, all_keys, commit_id)
}

/// Apply `count` random modifications to existing keys, spread across the
/// registry's databases.
fn modify_n(
    registry: &mut Reg,
    all_keys: &[Vec<Key>],
    database_count: usize,
    count: usize,
    rng: &mut impl rand::Rng,
) {
    for _ in 0..count {
        let db_index = rng.random_range(0..database_count);
        let keys = &all_keys[db_index];
        let key = keys[rng.random_range(0..keys.len())].clone();
        let value = Bytes::from(generate_random_bytes_in_range(rng, 1..32));
        registry
            .database_mut(db_index)
            .expect("The database index should be valid")
            .set(key, value)
            .expect("Modifying a key should succeed");
    }
}

/// Clone `source`, modify `count` keys across all of its databases, then settle
/// the writes (see the module docs), leaving them dirty and unhashed.
fn clone_modified_spread(
    source: &Reg,
    all_keys: &[Vec<Key>],
    database_count: usize,
    count: usize,
) -> Reg {
    let mut working = source
        .try_clone()
        .expect("Cloning the registry should succeed");
    modify_n(&mut working, all_keys, database_count, count, &mut rng());
    working
        .try_clone()
        .expect("Settling the modifications should succeed")
}

/// Clone `source`, modify `count` keys and commit, then modify another `count`
/// and settle. The returned registry is ready for a timed *second* commit, which
/// should only pay for the newly-dirtied paths.
fn clone_for_second_commit(
    source: &Reg,
    all_keys: &[Vec<Key>],
    database_count: usize,
    count: usize,
) -> Reg {
    let mut working = source
        .try_clone()
        .expect("Cloning the registry should succeed");
    let mut rng = rng();
    modify_n(&mut working, all_keys, database_count, count, &mut rng);
    working.commit().expect("The first commit should succeed");
    modify_n(&mut working, all_keys, database_count, count, &mut rng);
    working
        .try_clone()
        .expect("Settling the modifications should succeed")
}

/// Like [`clone_for_second_commit`], but checks the registry out afresh from the
/// first commit, so the timed second commit starts from a cold, fully-lazy tree.
/// Both commit twice against the same on-disk engine, so the difference between
/// their timings isolates what the warm in-memory tree is worth.
fn clone_for_second_commit_fresh_checkout(
    source: &Reg,
    repo: &DirectoryManager,
    all_keys: &[Vec<Key>],
    database_count: usize,
    count: usize,
) -> Reg {
    let mut working = source
        .try_clone()
        .expect("Cloning the registry should succeed");
    let mut rng = rng();
    modify_n(&mut working, all_keys, database_count, count, &mut rng);
    let commit_id = working.commit().expect("The first commit should succeed");

    // Drop the warm working registry so the second round starts from a cold
    // checkout (lazy nodes reloaded from disk), not the in-memory tree.
    drop(working);

    let mut fresh = Reg::checkout(repo.clone(), commit_id)
        .expect("Checking out the first commit should succeed");
    modify_n(&mut fresh, all_keys, database_count, count, &mut rng);
    fresh
        .try_clone()
        .expect("Settling the modifications should succeed")
}

/// Clone `source`, apply `count` random modifications to a single database
/// (`db_index`), then settle the writes (see the module docs). Used by the
/// modified copy/move/clear scenarios, where the database being operated on has
/// uncommitted changes.
fn clone_modified_db(source: &Reg, all_keys: &[Vec<Key>], db_index: usize, count: usize) -> Reg {
    let mut working = source
        .try_clone()
        .expect("Cloning the registry should succeed");
    {
        let keys = &all_keys[db_index];
        let mut rng = rng();
        let database = working
            .database_mut(db_index)
            .expect("The database index should be valid");
        for _ in 0..count {
            let key = keys[rng.random_range(0..keys.len())].clone();
            let value = Bytes::from(generate_random_bytes_in_range(&mut rng, 1..32));
            database
                .set(key, value)
                .expect("Modifying a key should succeed");
        }
    }
    working
        .try_clone()
        .expect("Settling the modifications should succeed")
}

fn database_lifecycle_benchmark(c: &mut Criterion) {
    let database_count = env_usize("REGISTRY_DATABASE_COUNT", REGISTRY_DATABASE_COUNT);
    let node_keys_count = env_usize("PREPOPULATED_NODE_KEYS_COUNT", PREPOPULATED_NODE_KEYS_COUNT);
    let modified_keys_count = env_usize("COMMIT_MODIFIED_KEYS_COUNT", COMMIT_MODIFIED_KEYS_COUNT);
    assert!(
        database_count >= 2,
        "The copy/move scenarios need at least two databases"
    );

    // `LIFECYCLE_REGISTRY_DIR` must live on a large filesystem — a
    // multi-database 10M-key registry is several GB. Unset → ephemeral temp dir.
    let registry_dir = std::env::var("LIFECYCLE_REGISTRY_DIR").ok();
    let tmpdir = registry_dir.is_none().then(TestableTmpdir::new);
    let repo_path: PathBuf = match (&registry_dir, &tmpdir) {
        (Some(dir), _) => {
            std::fs::create_dir_all(dir).expect("Creating the registry dir should succeed");
            PathBuf::from(dir)
        }
        (None, Some(tmp)) => tmp.path().to_path_buf(),
        (None, None) => unreachable!("tmpdir is Some whenever registry_dir is None"),
    };
    let repo = DirectoryManager::new(&repo_path).expect("Failed to create directory manager");

    // One large source registry, reused (by cloning) as the starting point for
    // every scenario. It is already committed, so clones start from clean
    // (cached-hash) trees and only the modified paths are dirty. A configured
    // persistent dir is reused if its manifest matches, and saved if not.
    let manifest_path = repo_path.join(PREPOPULATED_MANIFEST_FILE);
    let (source, all_keys, snapshot_commit) = match registry_dir
        .as_ref()
        .and_then(|_| try_load_prepopulated(&repo, &manifest_path, database_count, node_keys_count))
    {
        Some(loaded) => loaded,
        None => {
            let (source, all_keys, commit) = populate_registry(&repo);
            if registry_dir.is_some() {
                save_prepopulated(
                    &manifest_path,
                    commit,
                    &all_keys,
                    database_count,
                    node_keys_count,
                );
            }
            (source, all_keys, commit)
        }
    };

    let mut group = c.benchmark_group("Registry lifecycle");

    // Copy database 0 over database 1: the clone, plus the drop of the
    // overwritten database. The setup's whole-registry clone and the returned
    // registry's teardown drop are both untimed.
    group.bench_function("Copy database over large database", |b| {
        b.iter_batched(
            || {
                source
                    .try_clone()
                    .expect("Cloning the registry should succeed")
            },
            |mut registry| {
                registry
                    .copy_database(0, 1)
                    .expect("Copying the database should succeed");
                registry
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function("Copy modified database over large database", |b| {
        b.iter_batched(
            || clone_modified_db(&source, &all_keys, 0, modified_keys_count),
            |mut registry| {
                registry
                    .copy_database(0, 1)
                    .expect("Copying the database should succeed");
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // Move database 0 over database 1. Database 0 is left empty; database 1 is
    // dropped.
    group.bench_function("Move database over large database", |b| {
        b.iter_batched(
            || {
                source
                    .try_clone()
                    .expect("Cloning the registry should succeed")
            },
            |mut registry| {
                registry
                    .move_database(0, 1)
                    .expect("Moving the database should succeed");
                registry
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function("Move modified database over large database", |b| {
        b.iter_batched(
            || clone_modified_db(&source, &all_keys, 0, modified_keys_count),
            |mut registry| {
                registry
                    .move_database(0, 1)
                    .expect("Moving the database should succeed");
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // Clear database 0 — replace it with an empty one, dropping the cleared
    // database.
    group.bench_function("Clear large database", |b| {
        b.iter_batched(
            || {
                source
                    .try_clone()
                    .expect("Cloning the registry should succeed")
            },
            |mut registry| {
                registry
                    .clear_database(0)
                    .expect("Clearing the database should succeed");
                registry
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function("Clear modified large database", |b| {
        b.iter_batched(
            || clone_modified_db(&source, &all_keys, 0, modified_keys_count),
            |mut registry| {
                registry
                    .clear_database(0)
                    .expect("Clearing the database should succeed");
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // The floor cost of a commit — re-storing the loaded nodes, hashing the
    // registry and writing the manifest, with nothing dirty.
    group.bench_function("Commit registry (no modifications)", |b| {
        b.iter_batched(
            || {
                source
                    .try_clone()
                    .expect("Cloning the registry should succeed")
            },
            |registry| {
                let commit_id = registry.commit().expect("Committing should succeed");
                black_box(commit_id);
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // The realistic commit. The setup settles the writes, so the dirty-node
    // persistence and the hashing are inside the measurement.
    group.bench_function("Commit registry (with modifications)", |b| {
        b.iter_batched(
            || clone_modified_spread(&source, &all_keys, database_count, modified_keys_count),
            |registry| {
                let commit_id = registry.commit().expect("Committing should succeed");
                black_box(commit_id);
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // The *second* commit. The first commit's nodes are already persisted and
    // keep their cached hashes, so this measures the cached-hash and
    // skip-clean-subtree optimisations against the commit above.
    group.bench_function("Commit registry (second commit)", |b| {
        b.iter_batched(
            || clone_for_second_commit(&source, &all_keys, database_count, modified_keys_count),
            |registry| {
                let commit_id = registry.commit().expect("Committing should succeed");
                black_box(commit_id);
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // The second commit from a fresh checkout, so it runs on a cold, fully-lazy
    // tree. The gap from the scenario above is what the warm tree is worth.
    group.bench_function("Commit registry (second commit, fresh checkout)", |b| {
        b.iter_batched(
            || {
                clone_for_second_commit_fresh_checkout(
                    &source,
                    &repo,
                    &all_keys,
                    database_count,
                    modified_keys_count,
                )
            },
            |registry| {
                let commit_id = registry.commit().expect("Committing should succeed");
                black_box(commit_id);
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // Request a database's root hash — cached from checkout for a clean
    // database, rehashed along the dirty paths for a modified one.
    group.bench_function("Hash clean database", |b| {
        b.iter_batched(
            || {
                source
                    .try_clone()
                    .expect("Cloning the registry should succeed")
            },
            |registry| {
                let hash = registry
                    .database(0)
                    .expect("The database index should be valid")
                    .hash()
                    .expect("Hashing should succeed");
                black_box(hash);
                registry
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function("Hash modified database", |b| {
        b.iter_batched(
            || clone_modified_db(&source, &all_keys, 0, modified_keys_count),
            |registry| {
                let hash = registry
                    .database(0)
                    .expect("The database index should be valid")
                    .hash()
                    .expect("Hashing should succeed");
                black_box(hash);
                registry
            },
            BatchSize::PerIteration,
        )
    });

    // Check out the whole registry from the committed snapshot. It is returned
    // so its teardown drop is not timed.
    group.bench_function("Checkout registry", |b| {
        b.iter_batched(
            || (),
            |()| {
                Reg::checkout(repo.clone(), snapshot_commit)
                    .expect("Checking out the committed snapshot should succeed")
            },
            BatchSize::PerIteration,
        )
    });

    group.finish();
}

criterion_group! {
    name = benches;
    // Defaults on the `Criterion` instance rather than the group, so that the
    // `--sample-size` and `--measurement-time` flags can override them.
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(60));
    targets = database_lifecycle_benchmark
}
criterion_main!(benches);
