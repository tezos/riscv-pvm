// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

#![cfg(feature = "rocksdb")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use bincode::de::Decode;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use derive_more::Add;
use derive_more::Sum;
use humansize::format_size;
use octez_riscv::machine_state::memory::M1M;
use octez_riscv::machine_state::memory::M4K;
use octez_riscv::machine_state::memory::M8M;
use octez_riscv::machine_state::memory::Memory;
use octez_riscv::machine_state::memory::MemoryConfig;
use octez_riscv::machine_state::memory::Permissions;
use octez_riscv::machine_state::memory::listener::NoopMemoryGovernanceListener;
use octez_riscv::machine_state::page_cache::EmptyPageCache;
use octez_riscv::pvm::Pvm;
use octez_riscv::pvm::durable_storage::DurableStorageDummy;
use octez_riscv::storage::PersistentBlobStore;
use octez_riscv::storage::Repo;
use octez_riscv::storage::StorageError;
use octez_riscv::storage::Store;
use octez_riscv::storage::rocksdb_store::RocksDBStore;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::Unfoldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashedData;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::store::BlobStore;
use octez_riscv_data::store::BlobStoreError;
use octez_riscv_data::store::fold::BlobStoreFold;
use octez_riscv_test_utils::TestableTmpdir;
use rand::Rng;
use rand::RngExt;
use rand::rng;

/// The statistics for a single blob in a `CountingBlobStore`.
struct BlobStats {
    bytes: usize,

    /// How many 'copies' or references to the blob are in the store.
    ref_count: usize,
}

#[derive(Default, Add, Sum)]
/// Struct to store the statistics for an entire `CountingBlobStore`.
struct TotalStats {
    /// Total bytes actually stored.
    bytes: usize,

    /// Total blobs actually stored.
    blobs: usize,

    /// Bytes that were saved by not saving identical blobs twice.
    overlapping_bytes: usize,

    /// Blobs that were saved by not saving identical blobs twice.
    overlapping_blobs: usize,
}

impl std::fmt::Debug for TotalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes_efficiency =
            self.overlapping_bytes as f64 / (self.bytes + self.overlapping_bytes) as f64;
        let blobs_efficiency =
            self.overlapping_blobs as f64 / (self.blobs + self.overlapping_blobs) as f64;

        f.debug_struct("TotalStats")
            .field(
                "bytes",
                &format!(
                    "{} ({:.1}% saved)",
                    format_size(self.bytes, humansize::BINARY),
                    bytes_efficiency * 100.0
                ),
            )
            .field(
                "blobs",
                &format!("{} ({:.1}% saved)", self.blobs, blobs_efficiency * 100.0),
            )
            .finish()
    }
}

impl From<&BlobStats> for TotalStats {
    fn from(blob_stats: &BlobStats) -> Self {
        let overlapping_blobs = blob_stats.ref_count - 1;
        Self {
            bytes: blob_stats.bytes,
            blobs: 1,
            overlapping_bytes: blob_stats.bytes * overlapping_blobs,
            overlapping_blobs,
        }
    }
}

impl BlobStats {
    fn new(bytes: usize) -> Self {
        Self {
            bytes,
            ref_count: 1,
        }
    }
}

/// An implementation of `BlobStore` that doesn't store the blobs, instead it tracks statistics of
/// how many bytes would be stored, and how much space is saved by not duplicating identical blobs.
struct CountingBlobStore {
    blob_stats: Mutex<HashMap<Hash, BlobStats>>,
}

impl CountingBlobStore {
    /// Produces a summary of the statistics for the current state of the blob store.
    fn totals(&self) -> TotalStats {
        let store = self.blob_stats.lock().expect("Should not be poisoned");

        store.values().map(TotalStats::from).sum()
    }
}

impl Default for CountingBlobStore {
    fn default() -> Self {
        Self {
            blob_stats: Mutex::new(HashMap::new()),
        }
    }
}

impl BlobStore for CountingBlobStore {
    /// This is not implemented, this store doesn't actually store the blobs, just counts them.
    fn blob_get(&self, _key: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
        Ok(vec![])
    }

    fn blob_set<Data: AsRef<[u8]>>(&self, blob: &HashedData<Data>) -> Result<(), BlobStoreError> {
        let mut store = self.blob_stats.lock().expect("Should not be poisoned");

        store
            .entry(blob.hash())
            .and_modify(|stats| stats.ref_count += 1)
            .or_insert(BlobStats::new(blob.data().len()));

        Ok(())
    }

    fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError> {
        let mut store = self.blob_stats.lock().expect("Should not be poisoned");

        let mut delete = false;
        if let Some(stats) = store.get_mut(&key) {
            if stats.ref_count == 1 {
                delete = true;
            } else {
                stats.ref_count -= 1;
            }
        }

        if delete {
            store.remove(&key);
        };

        Ok(())
    }
}

/// This is a dummy implementation just to allow the `CountingBlobStore` to be used with commit
/// methods.
impl PersistentBlobStore for CountingBlobStore {
    fn init_from_path(_path: impl AsRef<Path>) -> Result<Self, StorageError> {
        Ok(CountingBlobStore::default())
    }

    fn persist(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

type BenchPvm<MC> = Pvm<MC, EmptyPageCache, DurableStorageDummy, Normal>;

struct BenchState<MC: MemoryConfig, BS> {
    _dir: TestableTmpdir,
    repo: Repo<BS>,
    pvm: BenchPvm<MC>,
}

fn bench_run_serialised<MC, BS>(state: BenchState<MC, BS>)
where
    MC: MemoryConfig,
    <MC as MemoryConfig>::State<Normal>: Decode<()>,
    BS: PersistentBlobStore,
{
    let hash = state
        .repo
        .commit_serialised(&state.pvm)
        .expect("Should be able to commit PVM");
    let _: BenchPvm<MC> = state
        .repo
        .checkout_serialised(&hash)
        .expect("Should be able to checkout PVM");
}

fn bench_run_folded<MC, BS>(state: BenchState<MC, BS>)
where
    MC: MemoryConfig,
    <MC as MemoryConfig>::State<Normal>: Foldable<BlobStoreFold<BS>>,
    <MC as MemoryConfig>::State<Normal>: Unfoldable,
    BS: PersistentBlobStore,
{
    let hash = state
        .repo
        .commit_folded(&state.pvm)
        .expect("Should be able to commit PVM");
    let _: BenchPvm<MC> = state
        .repo
        .checkout_folded(&hash)
        .expect("Should be able to checkout PVM");
}

fn setup_fn<BS, MC>(setup_pvm: &mut impl FnMut() -> BenchPvm<MC>) -> BenchState<MC, BS>
where
    MC: MemoryConfig,
    BS: PersistentBlobStore,
{
    let tmpdir = TestableTmpdir::new();
    let store = BS::init_from_path(tmpdir.path()).expect("Should init");
    BenchState {
        _dir: tmpdir,
        repo: Repo::new(store),
        pvm: setup_pvm(),
    }
}

fn stats_for_pvm<MC>(setup_pvm: &mut impl FnMut() -> BenchPvm<MC>, tag: &str)
where
    MC: MemoryConfig,
    <MC as MemoryConfig>::State<Normal>: Foldable<BlobStoreFold<CountingBlobStore>>,
{
    let pvm = setup_pvm();

    let store = CountingBlobStore::default();
    let repo = Repo::new(store);

    repo.commit_serialised(&pvm)
        .expect("Should be able to commit PVM");

    let statistics = repo.backend().totals();
    println!("{tag}/serialised: {statistics:?}");

    let store = CountingBlobStore::default();
    let repo = Repo::new(store);

    repo.commit_folded(&pvm)
        .expect("Should be able to commit PVM");

    let statistics = repo.backend().totals();
    println!("{tag}/folded: {statistics:?}");
}

fn bench_for_pvm<MC>(c: &mut Criterion, setup_pvm: &mut impl FnMut() -> BenchPvm<MC>, tag: &str)
where
    MC: MemoryConfig,
    <MC as MemoryConfig>::State<Normal>: Decode<()>,
    <MC as MemoryConfig>::State<Normal>: Foldable<BlobStoreFold<Store>>,
    <MC as MemoryConfig>::State<Normal>: Foldable<BlobStoreFold<CountingBlobStore>>,
    <MC as MemoryConfig>::State<Normal>: Foldable<BlobStoreFold<RocksDBStore>>,
    <MC as MemoryConfig>::State<Normal>: Unfoldable,
{
    stats_for_pvm(setup_pvm, tag);

    let mut group = c.benchmark_group(tag);

    group.measurement_time(Duration::from_secs(30));

    group.bench_function("serialised-no-rocksdb", |b| {
        b.iter_batched(
            || setup_fn::<Store, MC>(setup_pvm),
            bench_run_serialised,
            BatchSize::SmallInput,
        )
    });

    group.bench_function("folded-no-rocksdb", |b| {
        b.iter_batched(
            || setup_fn::<Store, MC>(setup_pvm),
            bench_run_folded,
            BatchSize::SmallInput,
        )
    });

    group.bench_function("serialised-rocksdb", |b| {
        b.iter_batched(
            || setup_fn::<RocksDBStore, MC>(setup_pvm),
            bench_run_serialised,
            BatchSize::SmallInput,
        )
    });

    group.bench_function("folded-rocksdb", |b| {
        b.iter_batched(
            || setup_fn::<RocksDBStore, MC>(setup_pvm),
            bench_run_folded,
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn setup_random_pvm<MC: MemoryConfig>(rng: &mut impl Rng) -> BenchPvm<MC> {
    let mut pvm = BenchPvm::<MC>::default();

    let length = MC::TOTAL_BYTES.get();
    let mut v: Vec<u8> = Vec::with_capacity(length);

    // UNSAFE: this is ok as we've just called `Vec::with_capacity` with the same `length`, so the
    // first line is only trying to write to allocated memory. `set_len` is fine too, as the
    // previous line has initialised every byte up to `length`.
    unsafe {
        rng.fill(std::slice::from_raw_parts_mut(v.as_mut_ptr(), length));
        v.set_len(length);
    }

    pvm.machine_state
        .core
        .main_memory
        .protect_pages(
            0,
            MC::TOTAL_BYTES,
            Permissions::READ_WRITE,
            NoopMemoryGovernanceListener,
        )
        .expect("Should be able to change memory permissions");

    pvm.machine_state
        .core
        .main_memory
        .write_all(0, &v)
        .expect("Should be able to write to memory");
    pvm
}

fn bench_commit_and_checkout(c: &mut Criterion) {
    let mut rng = rng();

    bench_for_pvm(c, &mut || BenchPvm::<M4K>::default(), "M4K-empty");
    bench_for_pvm(c, &mut || setup_random_pvm::<M4K>(&mut rng), "M4K-random");

    bench_for_pvm(c, &mut || BenchPvm::<M1M>::default(), "M1M-empty");
    bench_for_pvm(c, &mut || setup_random_pvm::<M1M>(&mut rng), "M1M-random");

    bench_for_pvm(c, &mut || BenchPvm::<M8M>::default(), "M8M-empty");
    bench_for_pvm(c, &mut || setup_random_pvm::<M8M>(&mut rng), "M8M-random");
}

criterion_group!(benches, bench_commit_and_checkout);
criterion_main!(benches);
