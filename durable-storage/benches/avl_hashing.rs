// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Hashing-focused benchmark, in two parts.
//!
//! 1. **Value-level hashing** (`value_hash/*`): hash standalone `Bytes` values of fixed
//!    sizes. For each size we measure the *current* page-tree scheme
//!    (`Hash::from_foldable`) alongside a raw `blake3::hash` of the same bytes — the raw
//!    figure is the target the BLAKE3-direct scheme (see
//!    `data/docs/blake3-bytes-hashing.md`) should approach. Once
//!    `octez_riscv_data::components::blake3_bytes::hash_value` is implemented, add a third
//!    line here to compare directly.
//!
//! 2. **Partial-tree rehashing** (`tree_rehash/*`): build and persist an AVL tree, then in
//!    each sample check out a *cold* lazy tree, dirty a **deterministic** set of key paths
//!    (untimed), and time only the root-hash recomputation. Because the dirty set is fixed
//!    (sorted keys, fixed stride, seeded values), the number of nodes loaded and rehashed
//!    is deterministic run-to-run — only the aggregate is what we measure. This isolates
//!    the cost of hashing over a partially loaded tree, which is exactly what changes when
//!    the value hash changes.

mod random;

use std::sync::Arc;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::avl::resolver::LazyNodeId;
use octez_riscv_durable_storage::avl::resolver::LazyResolver;
use octez_riscv_durable_storage::avl::tree::Tree;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::storage::KeyValueStore;
use octez_riscv_durable_storage::storage::Loadable;
use octez_riscv_durable_storage::storage::Storable;
use octez_riscv_durable_storage::storage::StoreOptions;
use rand::SeedableRng;
use rand::rngs::StdRng;
use random::generate_keys;

/// Fixed seed so tree contents, and hence how many nodes get loaded, are reproducible.
const SEED: u64 = 0xB1A5_E35E_ED42;

/// Value sizes hashed in the `value_hash` group (bytes). Spans sub-chunk (64), exactly
/// one BLAKE3 chunk (1024), one page / 4 chunks (4096), and progressively larger values up
/// to 16 MiB so the short- and long-value regimes are both visible.
const VALUE_SIZES: &[usize] = &[64, 1024, 4096, 65_536, 1_048_576, 16_777_216];

/// Number of keys inserted into the persisted tree. Override with `AVL_KEY_COUNT`.
const KEY_COUNT: usize = 50_000;

/// Number of key paths dirtied (and thus rehashed) per timed sample.
const DIRTY_PATHS: usize = 1_000;

/// One in every `LARGE_EVERY` values — in the resident tree and the dirtied writes — is
/// multi-kilobyte rather than a handful of bytes, so the rehash actually exercises
/// value-hashing (the part that changes under the BLAKE3-direct scheme). The rest stay
/// tiny, roughly matching the size mix of a real tree.
const LARGE_EVERY: usize = 8;

/// Multi-kilobyte sizes cycled through for the "large" values.
const LARGE_SIZES: &[usize] = &[2_048, 8_192, 16_384];

/// Deterministic value for the `i`-th write: 8 bytes most of the time, but every
/// `LARGE_EVERY`-th is a multi-kilobyte value (cycling `LARGE_SIZES`). Deterministic so the
/// per-run work — and the number of large values hashed — is stable.
fn value_for(i: usize) -> Vec<u8> {
    let len = if i % LARGE_EVERY == 0 {
        LARGE_SIZES[(i / LARGE_EVERY) % LARGE_SIZES.len()]
    } else {
        8
    };
    (0..len)
        .map(|j| i.wrapping_mul(2654435761).wrapping_add(j) as u8)
        .collect()
}

fn key_count() -> usize {
    std::env::var("AVL_KEY_COUNT")
        .ok()
        .map(|v| v.parse().expect("AVL_KEY_COUNT must be a number"))
        .unwrap_or(KEY_COUNT)
}

// ------------------------------- value-level hashing -------------------------------

/// Deterministic filler so runs are comparable without an RNG.
fn filled(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 2654435761usize) as u8).collect()
}

fn make_bytes(data: &[u8]) -> Bytes<Normal> {
    let mut bytes = Bytes::<Normal>::new(0);
    bytes.set(data);
    bytes
}

fn bench_value_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_hash");
    for &size in VALUE_SIZES {
        let data = filled(size);
        let bytes = make_bytes(&data);
        group.throughput(Throughput::Bytes(size as u64));

        // Current scheme: length leaf + arity-4 page tree of blake3 leaf hashes.
        group.bench_with_input(BenchmarkId::new("page_tree", size), &bytes, |b, bytes| {
            b.iter(|| std::hint::black_box(Hash::from_foldable(bytes)))
        });

        // Target: raw BLAKE3 of the same bytes (what the BLAKE3-direct scheme computes).
        group.bench_with_input(BenchmarkId::new("raw_blake3", size), &data, |b, data| {
            b.iter(|| std::hint::black_box(blake3::hash(data)))
        });
    }
    group.finish();
}

// ------------------------------- partial-tree rehashing -------------------------------

/// Build an AVL tree with `keys` (expected pre-sorted for reproducibility), persist it
/// (including value data) into `store`, and return the tree's root hash. Values follow
/// [`value_for`] so most are tiny but roughly one in `LARGE_EVERY` is multi-kilobyte.
fn build_and_persist<KV: KeyValueStore>(store: &Arc<KV>, keys: &[Key]) -> Hash {
    let mut resolver = LazyResolver::new(store.clone());
    let mut tree: Tree<LazyNodeId> = Tree::default();
    for (i, key) in keys.iter().enumerate() {
        let value = value_for(i);
        tree.set(key, &value, &mut resolver)
            .expect("setting a value during setup should succeed");
    }
    tree.store(store.as_ref(), &StoreOptions::default().with_node_data())
        .expect("persisting the tree should succeed");
    Hash::from_foldable(&tree)
}

/// The deterministic subset of keys whose paths we dirty each sample: a fixed stride over
/// the sorted key set. Sorting removes the `HashSet` iteration-order nondeterminism so the
/// loaded-node count is stable across runs.
fn dirty_keys(mut keys: Vec<Key>) -> Vec<Key> {
    keys.sort();
    let stride = (keys.len() / DIRTY_PATHS).max(1);
    keys.into_iter().step_by(stride).take(DIRTY_PATHS).collect()
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
        use octez_riscv_durable_storage::storage::in_memory::InMemoryKeyValueStore;
        use octez_riscv_durable_storage::storage::in_memory::InMemoryRepo;

        type TimedStore = InMemoryKeyValueStore;

        fn setup_timed_store() -> (Option<()>, Arc<TimedStore>) {
            let store = TimedStore::new(&InMemoryRepo::default())
                .expect("creating the store should succeed");
            (None, Arc::new(store))
        }
    }
}

/// Time the root-hash recomputation over a cold, partially loaded, partially dirtied tree.
///
/// The dirtying (which loads a deterministic set of node paths and invalidates their
/// memoised hashes) happens in the untimed `iter_batched` setup; only
/// `Hash::from_foldable` is timed. Untouched siblings stay blinded (hash-only) and
/// contribute their stored hash without being loaded, so we measure exactly the rehash of
/// the dirtied spine over an otherwise unloaded tree.
fn bench_tree_rehash(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut keys = generate_keys(&mut rng, key_count());
    keys.sort(); // deterministic build order => deterministic tree + node-load count

    let (_keepalive, store) = setup_timed_store();
    let tree_hash = build_and_persist(&store, &keys);
    let dirty = dirty_keys(keys);

    let mut group = c.benchmark_group("tree_rehash");
    group.throughput(Throughput::Elements(dirty.len() as u64));
    group.bench_function("partial dirty set (deterministic)", |b| {
        b.iter_batched(
            || {
                let mut resolver = LazyResolver::new(store.clone());
                let mut tree: Tree<LazyNodeId> = Tree::load(tree_hash, store.as_ref())
                    .expect("checking out the tree should succeed");
                // Untimed: load + dirty the deterministic path set. Written values follow
                // `value_for`, so ~1/LARGE_EVERY of the rehashed nodes carry multi-KB
                // values and their value-hash cost lands in the timed rehash.
                for (i, key) in dirty.iter().enumerate() {
                    let value = value_for(i);
                    tree.set(key, &value, &mut resolver)
                        .expect("set should succeed");
                }
                tree
            },
            |tree| std::hint::black_box(Hash::from_foldable(&tree)),
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
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
    targets = bench_value_hash, bench_tree_rehash
);
criterion_main!(benches);
