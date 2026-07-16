// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Direct AVL-tree benchmark that exercises the *storage-resolution* path.
//!
//! Unlike `avl_tree.rs`, which builds the tree entirely in memory (so every
//! node is already resident and no `blob_get` is ever issued), this benchmark persists the
//! tree and then operates on a freshly checked-out *lazy* tree. Every traversed level must be
//! read back from the key-value store, so this isolates the cost this crate's node/tree
//! storage layout controls.

mod random;

use std::sync::Arc;
use std::time::Duration;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_data::hash::Hash;
use octez_riscv_durable_storage::avl::resolver::LazyNodeId;
use octez_riscv_durable_storage::avl::resolver::LazyResolver;
use octez_riscv_durable_storage::avl::tree::Tree;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::storage::KeyValueStore;
use octez_riscv_durable_storage::storage::Loadable;
use octez_riscv_durable_storage::storage::Storable;
use octez_riscv_durable_storage::storage::StoreOptions;
use rand::prelude::*;
use random::generate_keys;
use random::generate_random_bytes_in_range;

/// Number of keys inserted into the persisted tree. Override with `AVL_KEY_COUNT`.
const KEY_COUNT: usize = 200_000;

/// Number of operations replayed per timed sample.
const OPERATIONS_PER_SAMPLE: usize = 2_000;

/// Seed for the PRNG generator.
const SEED: u64 = 123;

fn key_count() -> usize {
    std::env::var("AVL_KEY_COUNT")
        .ok()
        .map(|v| v.parse().expect("AVL_KEY_COUNT must be a number"))
        .unwrap_or(KEY_COUNT)
}

/// Build an AVL tree with `keys`, persist it (including value data) into `store`, and return
/// the tree's root hash.
fn build_and_persist<KV: KeyValueStore>(rng: &mut impl Rng, store: &Arc<KV>, keys: &[Key]) -> Hash {
    let mut resolver = LazyResolver::new(store.clone());
    let mut tree: Tree<LazyNodeId> = Tree::default();
    for key in keys {
        let value = generate_random_bytes_in_range(rng, 1..20);
        tree.set(key, &value, &mut resolver)
            .expect("setting a value during setup should succeed");
    }
    tree.store(store.as_ref(), &StoreOptions::default().with_node_data())
        .expect("persisting the tree should succeed");
    Hash::from_foldable(&tree)
}

cfg_if::cfg_if! {
    if #[cfg(rocksdb)] {
        use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
        use octez_riscv_test_utils::TestableTmpdir;

        type TimedStore = PersistenceLayer;

        fn setup_timed_store() -> (Option<TestableTmpdir>, Arc<TimedStore>) {
            use octez_riscv_durable_storage::repo::DirectoryManager;

            let tmpdir = TestableTmpdir::new();
            let repo = DirectoryManager::new(tmpdir.path())
                .expect("creating the directory manager should succeed");
            let store = TimedStore::new(&repo).expect("creating the store should succeed");
            (Some(tmpdir), Arc::new(store))
        }
    } else {
        type TimedStore = InMemoryKeyValueStore;

        fn setup_timed_store() -> (Option<()>, Arc<TimedStore>) {
            let store = TimedStore::new(&InMemoryRepo::default())
                .expect("creating the store should succeed");
            (None, Arc::new(store))
        }
    }
}

/// Benchmark set/delete operations against a freshly checked-out (cold) persisted lazy tree.
///
/// Each sample checks out a fresh lazy tree with an empty in-process cache, so the timed work
/// includes reading the traversed nodes back from storage.
fn bench_avl_cold_operations(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let keys = generate_keys(&mut rng, key_count());

    let (_keepalive, store) = setup_timed_store();
    let tree_hash = build_and_persist(&mut rng, &store, &keys);

    c.bench_function("AVL cold set/delete (persisted, lazy resolution)", |b| {
        b.iter_batched(
            || {
                let resolver = LazyResolver::new(store.clone());
                let tree: Tree<LazyNodeId> = Tree::load(tree_hash, store.as_ref())
                    .expect("checking out the tree should succeed");
                let ops: Vec<(Key, bool)> = (0..OPERATIONS_PER_SAMPLE)
                    .map(|_| {
                        let key = keys.choose(&mut rng).expect("keys is non-empty").clone();
                        (key, rng.random_bool(0.5))
                    })
                    .collect();
                (tree, resolver, ops)
            },
            |(mut tree, mut resolver, ops)| {
                for (key, is_set) in ops {
                    if is_set {
                        tree.set(&key, b"value", &mut resolver)
                            .expect("set should succeed");
                    } else {
                        tree.delete(&key, &mut resolver)
                            .expect("delete should succeed");
                    }
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(20))
        .warm_up_time(Duration::from_secs(2))
        .sample_size(30)
}

criterion_group!(
    name = benches;
    config = configure_criterion();
    targets = bench_avl_cold_operations
);
criterion_main!(benches);
