// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of the RISC-V PVM durable storage's persistence layer.
//!
//! A persistence layer is tied to a repository on disk, identified by a directory.
//! Within that directory, the persistence layer needs to be able to perform basic KV operations:
//! - get
//! - set
//! - delete
//!
//! As well as repository-level operations:
//! - new
//! - clone
//! - commit (returning the commit hash)
//! - checkout a specific commit
//!
//! The folder structure of the [`DirectoryManager`] is:
//! ```text
//! <repo_path>:
//!    temporary/
//!        db_<random>/checkpoint/
//!            <rocksdb internals>
//!    commits/
//!       <commit_id>/
//!          <rocksdb internals>
//! ```

#![cfg(rocksdb)]

use std::mem::ManuallyDrop;
use std::path::Path;

use bincode::BorrowDecode;
use bincode::Encode;
use rocksdb::ColumnFamilyDescriptor;
use rocksdb::MergeOperands;
use rocksdb::checkpoint::Checkpoint;
use tempfile::TempDir;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::merkle_worker::CommittedRoot;
use crate::merkle_worker::MerkleWorker;
use crate::repo::DirectoryManager;
use crate::storage::PersistentKeyValueStore;
use crate::storage::ReadOnlyKeyValueStore;
use crate::storage::ReadableKeyValueStore;
use crate::storage::StoreId;
use crate::storage::WriteableKeyValueStore;

/// The name of the column family used for storing blob-keyed data.
pub(crate) const BLOB_CF: &str = "blob";

/// The name of the column family used for storing 'plain key'-ed data.
pub(crate) const KV_CF: &str = "default";

#[derive(BorrowDecode, Encode)]
struct OffsetWriteMergePayload<'a> {
    offset: usize,
    value: &'a [u8],
}

/// Defines an atomic merge operator used when writing with an offset.
///
/// Called every time [`rocksdb::DBCommon::merge`] is called on a database where this has been set
/// as a merge operator, passing in the key and the associated value's existing serialised
/// collection of operands. Optionally returns a value representing one or more merged operands
/// later passed to the full merge operator instead of the full collection of operands. Both
/// operators must be able to work with the same structured data.
///
/// The advantage of this is in preventing unnecessary work. For example, when the same part of the
/// associated value is written to by multiple merges, only the last would need to be fully merged.
///
/// See: <https://github.com/facebook/rocksdb/wiki/merge-operator>
fn offset_write_partial_merge(
    _new_key: &[u8],
    _left_operand: Option<&[u8]>,
    _operands: &MergeOperands,
) -> Option<Vec<u8>> {
    // Currently, the API for [`Database`] requires a Get operation for any non-zero
    // offset passed to `persistence_layer::write`. This forces RocksDB to perform a full merge,
    // reducing the utility of merging operands with non-zero offsets.
    //
    // Any merge must also ensure that the work saved outweighs the work of managing operands.
    //
    // For these reasons, this function simply returns `None`, meaning no operand merging is
    // currently performed.
    None
}

/// Defines an atomic merge operator used when writing with an offset.
///
/// Called when the associated value is retrieved or when RocksDB performs compaction on a database
/// where this operator has been set. Returns a value representing a single fully merged value.
///
/// See: <https://github.com/facebook/rocksdb/wiki/merge-operator>
fn offset_write_full_merge(
    _new_key: &[u8],
    existing_value: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result = existing_value.map(|v| v.to_vec()).unwrap_or_default();

    for mut op in operands {
        while !op.is_empty() {
            let decode = octez_riscv_data::serialisation::deserialise_borrowed(op);
            let (payload, len): (OffsetWriteMergePayload, _) =
                decode.expect("Should be a valid encoding");

            // Advance the slice
            op = &op[len..];

            let offset = payload.offset;
            let data = payload.value;

            // This shouldn't happen: it's prevented by the `Database` API.
            assert!(
                offset <= result.len(),
                "Oversized offset: `{offset:?}` is longer than the length of the associated item: `{:?}`",
                result.len()
            );

            let Some(new_data_end) = offset.checked_add(data.len()) else {
                panic!(
                    "Offset + data.len() overflows (`{offset:?}` + `{:?}`)",
                    data.len()
                );
            };

            // TODO RV-888 Add support for setting a more sensible limit
            if new_data_end > isize::MAX as usize {
                panic!("Can't allocate {new_data_end:?} bytes");
            }

            let overwrite_end = std::cmp::min(result.len(), new_data_end);

            // SAFETY: Unchecked subtraction OK as result.len() >= offset && new_data_end > offset
            let data_copy_end = overwrite_end - offset;

            result[offset..overwrite_end].copy_from_slice(&data[0..data_copy_end]);
            if result.len() < new_data_end {
                // SAFETY: Panics if the memory can't be allocated.
                result.extend_from_slice(&data[data_copy_end..]);
            }
        }
    }
    Some(result)
}

/// Changes the default skip list memtable implementation to a hash linked list one.
/// The skip list is an ordered data structure with O(log n) bounds for the relevant
/// operations, while we need a hash map like behaviour which has O(1) bounds for the
/// same.
fn set_memtable_to_hash_link_list(options: &mut rocksdb::Options) {
    let factory = rocksdb::MemtableFactory::HashLinkList {
        // The 1_000_000 below is a reasonable default for getting a good load factor.
        bucket_count: 1_000_000,
    };
    // Only the skip list based memtable supports concurrent writes.
    options.set_allow_concurrent_memtable_write(false);
    options.set_memtable_factory(factory);
}

fn add_merge_operator(options: &mut rocksdb::Options) {
    options.set_merge_operator(
        "offset_write",
        offset_write_full_merge,
        offset_write_partial_merge,
    );
}

fn add_creation_options(options: &mut rocksdb::Options) {
    options.create_if_missing(true);
    // This is used to avoid accidentally overwriting an existing database.
    options.set_error_if_exists(true);
}

/// Compaction knobs read from the environment, for measuring what compaction costs a repository
/// that retains checkpoints.
///
/// Retaining checkpoints makes compaction expensive in a way ordinary RocksDB use does not: a
/// checkpoint hard-links the files live when it was taken, so a rewrite ends the sharing it had
/// with every checkpoint still holding the files that rewrite replaced. These exist so a sweep
/// can measure that trade-off without a rebuild, and are compiled out of ordinary builds.
///
/// Sitting in [`rocksdb_default_options`] puts these on every instance this module opens, not
/// the working database alone. A read-only checkout of a commit never compacts, so they change
/// nothing there; the working database, across both its column families, is what they are for.
///
/// The `OCTEZ_NDS_` prefix is deliberate where the rest of this repository uses `OCTEZ_RISCV_`:
/// the new durable storage is not the RISC-V PVM's alone, and is meant to serve the WASM PVM
/// too.
///
/// - `OCTEZ_NDS_ROCKSDB_L0_TRIGGER`: files at level zero before compaction runs, overriding the
///   default of twenty set below. At RocksDB's own default of four, combined with the flush every
///   commit forces, compaction runs every four commits whatever they contain.
/// - `OCTEZ_NDS_ROCKSDB_DISABLE_AUTO_COMPACTION`: stop compacting the working database at all.
///   Presence is what is read rather than the value, so anything enables it, `0` included. Level
///   zero is then free to grow for as long as the sweep runs: RocksDB conditions each of its
///   level-zero write stalls on automatic compaction being enabled, precisely because no
///   compaction would arrive to relieve one.
#[cfg(rocksdb_test_utils)]
fn add_measurement_tuning(options: &mut rocksdb::Options) {
    if let Ok(trigger) = std::env::var("OCTEZ_NDS_ROCKSDB_L0_TRIGGER") {
        let trigger = trigger
            .parse()
            .expect("OCTEZ_NDS_ROCKSDB_L0_TRIGGER should be a number");

        options.set_level_zero_file_num_compaction_trigger(trigger);
    }

    if std::env::var("OCTEZ_NDS_ROCKSDB_DISABLE_AUTO_COMPACTION").is_ok() {
        options.set_disable_auto_compactions(true);
    }
}

/// Number of level-zero files to accumulate before compacting them downwards.
///
/// Compaction is unusually expensive here because commits are retained: a checkpoint hard-links the
/// files live when it was taken, so rewriting a file ends sharing for every checkpoint taken before
/// it. Keys are spread by hash, so a level-zero file spans the whole key space and overlaps
/// everything below, which makes each compaction a rewrite of the whole base level.
///
/// The trigger counts *files*, and creating a checkpoint flushes the memtable, so one file appears
/// per commit however little it contains. Left at RocksDB's default of four, compaction therefore
/// runs every four commits regardless of how much was written. Raising it makes the cadence coarser,
/// which is paid for by the bloom filters below: more files at level zero means more of them to
/// search on a lookup that finds nothing.
const LEVEL_ZERO_COMPACTION_TRIGGER: i32 = 20;

/// Bits per key for the bloom filters, giving roughly a 1% false positive rate.
const BLOOM_FILTER_BITS_PER_KEY: f64 = 10.0;

/// Add bloom filters, so a lookup can skip files that cannot hold the key.
///
/// Both column families are looked up by whole key and never scanned by range - blobs by content
/// hash, values by their key - which is exactly the shape bloom filters serve. Without them a
/// lookup consults the index of every file whose key range covers the key, which for hash-spread
/// keys is most of them.
fn add_bloom_filters(options: &mut rocksdb::Options) {
    let mut table_options = rocksdb::BlockBasedOptions::default();
    table_options.set_bloom_filter(BLOOM_FILTER_BITS_PER_KEY, false);

    options.set_block_based_table_factory(&table_options);
}

fn rocksdb_default_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    set_memtable_to_hash_link_list(&mut options);
    add_bloom_filters(&mut options);
    options.set_level_zero_file_num_compaction_trigger(LEVEL_ZERO_COMPACTION_TRIGGER);

    #[cfg(rocksdb_test_utils)]
    add_measurement_tuning(&mut options);

    options
}

/// These options are used for opening and closing a newly created rocksdb instance.
///
/// Although different fields are used for opening vs. destroying a rocksdb instance, you need to
/// ensure that the options used for destroying are valid with respect to the options used when
/// opening the db. There is no concrete documentation for which options should be kept in sync for
/// open/close, may need to investigate rocksdb source code:
/// <https://github.com/facebook/rocksdb/blob/a1dad12c8c9a7a65fa19d3bc78a5f7687ce6c1bd/db/db_impl/db_impl.cc#L5185>
/// (look for the function destroying a rocksdb instance)
fn rocksdb_creation_options() -> rocksdb::Options {
    let mut options = rocksdb_default_options();
    add_creation_options(&mut options);
    add_merge_operator(&mut options);
    options
}

fn rocksdb_blob_cf_creation_options() -> rocksdb::Options {
    let mut options = rocksdb_default_options();
    add_creation_options(&mut options);
    options
}

/// These options are used for opening a rocksdb instance from a checkpoint.
pub(crate) fn rocksdb_checkpoint_options() -> rocksdb::Options {
    let mut options = rocksdb_default_options();
    add_merge_operator(&mut options);
    options
}

/// The column family holding blob-keyed data.
///
/// Panics if the handle does not have it, which would mean it was not opened by this module.
fn blob_cf_of(db: &rocksdb::DB) -> &rocksdb::ColumnFamily {
    db.cf_handle(BLOB_CF)
        .expect("the rocksdb instance should always contain the data cf")
}

/// Retrieve the data associated with a blob key from `db`.
fn blob_get_from<'db>(
    db: &'db rocksdb::DB,
    key: &[u8],
) -> Result<rocksdb::DBPinnableSlice<'db>, Error> {
    let entry =
        db.get_pinned_cf(blob_cf_of(db), key)
            .map_err(|error| OperationalError::GetFailed {
                column: BLOB_CF.to_string(),
                key: key.to_owned(),
                error,
            })?;

    match entry {
        Some(value) => Ok(value),
        None => Err(InvalidArgumentError::KeyNotFound)?,
    }
}

/// Retrieve the value associated with a key from `db`.
fn get_from<'db>(db: &'db rocksdb::DB, key: &[u8]) -> Result<rocksdb::DBPinnableSlice<'db>, Error> {
    let value = db
        .get_pinned(key)
        .map_err(|error| OperationalError::GetFailed {
            column: KV_CF.to_owned(),
            key: key.to_owned(),
            error,
        })?;

    match value {
        Some(value) => Ok(value),
        None => Err(InvalidArgumentError::KeyNotFound)?,
    }
}

/// Open the database committed at `commit_path` in read-only mode.
///
/// A read-only instance does not acquire the directory's lock, replay its write-ahead log, rewrite
/// its manifest or write an info log, so the committed state is left exactly as it is found on
/// disk - not one byte of the directory is touched, and it can be opened this way any number of
/// times over. `test_read_only_checkout_reads_the_commit_in_place` holds this to the letter.
///
/// As a consequence such an instance only observes state that has been flushed to the commit; in
/// practice that is all of it, as commits are created through [`Checkpoint::create_checkpoint`],
/// which flushes first.
fn open_committed_read_only(commit_path: &Path) -> Result<rocksdb::DB, OperationalError> {
    let options = rocksdb_checkpoint_options();

    rocksdb::DB::open_cf_descriptors_read_only(
        &options,
        commit_path,
        [
            ColumnFamilyDescriptor::new(BLOB_CF, options.clone()),
            ColumnFamilyDescriptor::new(KV_CF, options.clone()),
        ],
        false,
    )
    .map_err(|error| OperationalError::OpenRocksDbFailed { error })
}

/// Whether `commit_path` holds a published commit.
///
/// A commit is published by renaming a complete checkpoint into place, so the `CURRENT` file that
/// every RocksDB directory holds is enough to tell a published commit from a directory left behind
/// by an interrupted removal.
///
/// Once published, a commit directory is never mutated or unlinked: it is read in place, by any
/// number of concurrent read-only checkouts, and committing the same id again leaves it alone. A
/// future commit GC has to honour that - it may remove a commit nothing refers to any more, but
/// never rewrite one in place, and never remove one while it is being read.
fn is_published_commit(commit_path: &Path) -> bool {
    commit_path.join("CURRENT").exists()
}

/// The outcome of an attempt to publish a staged checkpoint under a commit path.
enum Publish {
    /// The commit is published - either by this attempt, or by a concurrent one, which published
    /// the same state.
    Done,
    /// Something that is not a published commit occupies the commit path, holding the error the
    /// attempt failed with.
    Occupied(std::io::Error),
}

/// Move the checkpoint staged at `staged` into place under `commit_path`.
///
/// The rename is the only authority on whether the commit is published: it is a single step, so a
/// concurrent publisher of the same commit either lost to it or won it outright. Nothing observed
/// beforehand can be relied on to still hold by the time it runs.
fn publish_staged_commit(staged: &Path, commit_path: &Path) -> Result<Publish, OperationalError> {
    match std::fs::rename(staged, commit_path) {
        Ok(()) => Ok(Publish::Done),

        // Lost a race to publish this commit. The winner published the state staged here - the
        // commit id is a hash of it - so the commit is made either way, and the staged copy is
        // discarded along with its staging directory.
        Err(_) if is_published_commit(commit_path) => Ok(Publish::Done),

        // A directory in the way is the one failure the caller can repair.
        Err(error) if commit_path.is_dir() => Ok(Publish::Occupied(error)),

        Err(error) => Err(OperationalError::CommitPublishFailed {
            staged: staged.to_owned(),
            commit: commit_path.to_owned(),
            error,
        }),
    }
}

/// Persistence layer for durable solution used by the RISC-V PVM.
///
/// Invariants:
/// - The path in `temp_initial_db_path` is unique for each instance of [`PersistenceLayer`] and is
///   assumed to not be modified / known outside of this instance.
#[derive(Debug)]
pub struct PersistenceLayer {
    /// The underlying handle to the RocksDB instance.
    ///
    /// [`ManuallyDrop`] is used to ensure safety when dropping [`PersistenceLayer`]. Calling
    /// [`rocksdb::DB::destroy`] requires all connections to that path to be closed, which is called
    /// in [`rocksdb::DB`]'s drop method.
    db_instance: ManuallyDrop<rocksdb::DB>,

    /// The [`TempDir`] holding the rocksdb instance.
    ///
    /// We need to own this in order to keep alive the temporary directory for the lifetime of the
    /// persistence layer.
    _tempdir: TempDir,

    /// Distinguishes this store from every other, including checkpoints taken of it.
    store_id: StoreId,
}

impl PersistenceLayer {
    fn clone_as_checkpoint(
        db: &rocksdb::DB,
        repo: &DirectoryManager,
    ) -> Result<Self, OperationalError> {
        let tempdir = repo.temp_database_dir()?;
        let checkpoint_path = tempdir.path().join("checkpoint");

        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        Checkpoint::new(db)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })?
            .create_checkpoint(&checkpoint_path)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })?;

        let temp_db = rocksdb::DB::open_cf_descriptors(
            &rocksdb_checkpoint_options(),
            &checkpoint_path,
            [
                ColumnFamilyDescriptor::new(KV_CF, rocksdb_checkpoint_options()),
                ColumnFamilyDescriptor::new(BLOB_CF, rocksdb_checkpoint_options()),
            ],
        )
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

        Ok(Self {
            db_instance: ManuallyDrop::new(temp_db),
            _tempdir: tempdir,
            store_id: StoreId::next(),
        })
    }

    fn blob_cf(&self) -> &rocksdb::ColumnFamily {
        blob_cf_of(&self.db_instance)
    }
}

impl ReadableKeyValueStore for PersistenceLayer {
    type Repo = DirectoryManager;

    fn store_id(&self) -> StoreId {
        self.store_id
    }

    type Merkle = MerkleWorker<Self>;

    fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        blob_get_from(&self.db_instance, key.as_ref())
    }

    fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        get_from(&self.db_instance, key.as_ref())
    }
}

/// A read-only view of a committed [`PersistenceLayer`], opened in place.
///
/// This is safe as no mutable operations can be performed using this type.
#[derive(Debug)]
pub struct ReadOnlyPersistenceLayer {
    /// The underlying handle to the RocksDB instance.
    ///
    /// Unlike [`PersistenceLayer`], no manual-drop is needed: dropping this closes the handle
    /// and there is nothing to destroy afterwards.
    db_instance: rocksdb::DB,

    /// Distinguishes this view from every other store.
    store_id: StoreId,
}

impl ReadableKeyValueStore for ReadOnlyPersistenceLayer {
    type Repo = DirectoryManager;

    type Merkle = CommittedRoot;
    fn store_id(&self) -> StoreId {
        self.store_id
    }

    fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        blob_get_from(&self.db_instance, key.as_ref())
    }

    fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        get_from(&self.db_instance, key.as_ref())
    }
}

impl ReadOnlyKeyValueStore for ReadOnlyPersistenceLayer {
    type Writeable = PersistenceLayer;

    fn checkout_read_only(repo: &Self::Repo, id: &CommitId) -> Result<Self, OperationalError> {
        Self::checkout_read_only_from_path(&repo.database_commit_dir(id))
    }

    fn checkout_read_only_from_path(commit_path: &Path) -> Result<Self, OperationalError> {
        // A directory that is not a published commit is no commit at all - an interrupted removal
        // by an older version of `commit` can leave one behind, and opening it would fail further
        // in, with an error about the database rather than about the commit.
        if !is_published_commit(commit_path) {
            return Err(OperationalError::CommitNotFound);
        };

        Ok(Self {
            db_instance: open_committed_read_only(commit_path)?,
            store_id: StoreId::next(),
        })
    }

    fn to_writeable(&self, repo: &Self::Repo) -> Result<Self::Writeable, OperationalError> {
        // Checkpointing works from a read-only instance - the working copy is made
        // exactly as it is for a clone of a mutable layer.
        PersistenceLayer::clone_as_checkpoint(&self.db_instance, repo)
    }
}

#[cfg(rocksdb_test_utils)]
pub mod measurement;

impl WriteableKeyValueStore for PersistenceLayer {
    fn new(repo: &Self::Repo) -> Result<Self, OperationalError> {
        let tempdir = repo.temp_database_dir()?;
        let new_db_path = tempdir.path().join("checkpoint");

        let mut db = rocksdb::DB::open(&rocksdb_creation_options(), &new_db_path)
            .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;
        db.create_cf(BLOB_CF, &rocksdb_blob_cf_creation_options())
            .map_err(|error| OperationalError::ColumnFamilyCreationFailed {
                name: BLOB_CF.to_owned(),
                error,
            })?;

        Ok(Self {
            db_instance: ManuallyDrop::new(db),
            _tempdir: tempdir,
            store_id: StoreId::next(),
        })
    }

    fn try_clone(&self, repo: &Self::Repo) -> Result<Self, OperationalError> {
        Self::clone_as_checkpoint(&self.db_instance, repo)
    }

    fn blob_set(
        &self,
        key: impl AsRef<[u8]>,
        data: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError> {
        let key = key.as_ref();
        self.db_instance
            .put_cf(self.blob_cf(), key, data.as_ref())
            .map_err(|error| OperationalError::PutFailed {
                column: BLOB_CF.to_string(),
                key: key.to_owned(),
                error,
            })
    }

    fn blob_delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        let key = key.as_ref();
        self.db_instance
            .delete_cf(self.blob_cf(), key)
            .map_err(|error| OperationalError::DeleteFailed {
                column: BLOB_CF.to_string(),
                key: key.to_owned(),
                error,
            })
    }

    fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        self.db_instance
            .put(key.as_ref(), value.as_ref())
            .map_err(|error| OperationalError::PutFailed {
                column: KV_CF.to_owned(),
                key: key.as_ref().to_owned(),
                error,
            })
    }

    fn write(
        &self,
        key: impl AsRef<[u8]>,
        offset: usize,
        value: impl AsRef<[u8]>,
    ) -> Result<(), Error> {
        // If the offset is greater than 0 and the key exists, we have to do an expensive 'get'
        // operation to check if the existing value length is shorter than the offset.
        if offset > 0 {
            // `may_exist` can be cheaper than `get`
            let may_exist = self.db_instance.key_may_exist(&key);
            if !may_exist {
                return Err(InvalidArgumentError::OffsetTooLarge)?;
            }

            // Checking the length of a value requires a full retrieval. Returns an error if the
            // key does not exist.
            let len = self
                .get(key.as_ref())
                .map_or(0, |stored| stored.as_ref().len());
            if offset > len || offset.checked_add(value.as_ref().len()).is_none() {
                return Err(InvalidArgumentError::OffsetTooLarge)?;
            }
        }

        // misuse.

        let payload_struct = OffsetWriteMergePayload {
            offset,
            value: value.as_ref(),
        };
        let payload = octez_riscv_data::serialisation::serialise(payload_struct)
            .expect("Merge operator serialisation should always succeed");

        self.db_instance
            .merge(key.as_ref(), payload)
            .map_err(|error| {
                Error::from(OperationalError::MergeFailed {
                    key: key.as_ref().to_owned(),
                    offset,
                    error,
                })
            })
    }

    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        self.db_instance
            .delete(key.as_ref())
            .map_err(|error| OperationalError::DeleteFailed {
                column: KV_CF.to_owned(),
                key: key.as_ref().to_owned(),
                error,
            })
    }
}

impl PersistentKeyValueStore for PersistenceLayer {
    fn commit_to_path(&self, path: &Path) -> Result<(), OperationalError> {
        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        let checkpoint = Checkpoint::new(&self.db_instance)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })?;
        checkpoint
            .create_checkpoint(path)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })
    }

    fn commit(&self, repo: &DirectoryManager, id: &CommitId) -> Result<(), OperationalError> {
        let commit_path = repo.database_commit_dir(id);

        // A commit id is a hash of the state committed under it, so a commit already published
        // under `id` holds the state being committed here and there is nothing left to do.
        // Republishing it would unlink the files a concurrent read-only checkout of the commit is
        // reading, only to replace them with the same state. This only skips the work below - it
        // is the publishing rename that settles whether this call has anything to do.
        if is_published_commit(&commit_path) {
            return Ok(());
        }

        // Stage the checkpoint in a directory of its own and move it into place, rather than
        // checkpointing straight into `commit_path`: RocksDB stages a checkpoint at
        // `<destination>.tmp`, so two commits of one id would otherwise share - and clobber - a
        // staging directory. Staging among the repository's temporary databases keeps the staged
        // copy on the same filesystem as the commit, which both the rename below and the hard links
        // a checkpoint is made of require.
        let staging = repo.temp_database_dir()?;
        let staged_commit = staging.path().join("commit");

        self.commit_to_path(&staged_commit)?;

        let Publish::Occupied(_) = publish_staged_commit(&staged_commit, &commit_path)? else {
            return Ok(());
        };

        // A directory in the way holding no `CURRENT` file is not a published commit. A checkpoint
        // is built under a temporary name and renamed into place, so an interrupted commit cannot
        // leave one behind - only an interrupted removal by an older version of this function can,
        // and replacing it is the repair.
        //
        // It is moved aside rather than removed where it lies. Testing a directory for `CURRENT`
        // and removing it cannot be one step, so a commit published between the two would be
        // removed as though it were incomplete, unlinking the files a concurrent read-only
        // checkout is reading. A rename is one step, and nothing else can reach what it moves out
        // of the way, so what was moved can be examined once it is out of the way.
        log::warn!("Replacing incomplete commit: {}", id.hex_encode());

        let displaced = staging.path().join("displaced");

        std::fs::rename(&commit_path, &displaced).map_err(|error| {
            OperationalError::IncompleteCommitDisplacementFailed {
                commit: commit_path.clone(),
                displaced: displaced.clone(),
                error,
            }
        })?;

        // A commit published in that window is put back rather than replaced by the copy staged
        // here. The two hold the same state - the commit id is a hash of it - but not the same
        // files: a checkpoint is made of hard links to the database it was taken from, so the copy
        // staged here has file names and inodes of its own, and a read-only checkout part-way
        // through opening the commit is reading the published copy's. Putting it back also keeps
        // it out of the staging directory, which is removed when this call returns.
        if is_published_commit(&displaced) {
            return match std::fs::rename(&displaced, &commit_path) {
                Ok(()) => Ok(()),

                // Yet another publisher reached the commit path first, with the same state again.
                Err(_) if is_published_commit(&commit_path) => Ok(()),

                Err(error) => Err(OperationalError::CommitPublishFailed {
                    staged: displaced,
                    commit: commit_path,
                    error,
                }),
            };
        }

        match publish_staged_commit(&staged_commit, &commit_path)? {
            Publish::Done => Ok(()),

            Publish::Occupied(error) => Err(OperationalError::CommitPublishFailed {
                staged: staged_commit,
                commit: commit_path,
                error,
            }),
        }
    }

    fn checkout_from_path(
        commit_path: &Path,
        working_path: TempDir,
    ) -> Result<Self, OperationalError> {
        // A directory that is not a published commit is no commit at all - an interrupted removal
        // by an older version of `commit` can leave one behind, and opening it would fail further
        // in, with an error about the database rather than about the commit.
        if !is_published_commit(commit_path) {
            return Err(OperationalError::CommitNotFound);
        };

        // Open the previous commitment from the given source path in read-only mode
        let read_only_database = open_committed_read_only(commit_path)?;

        // Make a copy to ensure we're not modifying the commitment path's contents
        let checkpoint = Checkpoint::new(&read_only_database)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })?;
        let checkpoint_path = working_path.path().join("checkpoint");
        checkpoint
            .create_checkpoint(&checkpoint_path)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })?;

        let database = rocksdb::DB::open_cf_descriptors(
            &rocksdb_checkpoint_options(),
            &checkpoint_path,
            [
                ColumnFamilyDescriptor::new(KV_CF, rocksdb_checkpoint_options()),
                ColumnFamilyDescriptor::new(BLOB_CF, rocksdb_checkpoint_options()),
            ],
        )
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

        Ok(Self {
            db_instance: ManuallyDrop::new(database),
            _tempdir: working_path,
            store_id: StoreId::next(),
        })
    }

    fn checkout(repo: &DirectoryManager, id: &CommitId) -> Result<Self, OperationalError> {
        let commit_path = repo.database_commit_dir(id);
        let working_path = repo.temp_database_dir()?;

        Self::checkout_from_path(&commit_path, working_path)
    }
}

impl Drop for PersistenceLayer {
    /// Databases created from a new or clone operation will have `temp_initial_db_path` set. These
    /// databases do not have to be saved on disk as they have not been committed to storage.
    fn drop(&mut self) {
        let db_path = self.db_instance.path().to_path_buf();

        // Safety: This manual drop is called in this object's drop method.
        unsafe {
            ManuallyDrop::drop(&mut self.db_instance);
        }

        let options = rocksdb_creation_options();
        if let Err(e) = rocksdb::DB::destroy(&options, &db_path) {
            log::error!("Failed to destroy temporary rocksdb at {db_path:?}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::HashedData;
    use octez_riscv_test_utils::TestableTmpdir;
    use proptest::prelude::Strategy;
    use proptest::prelude::any;
    use proptest::proptest;
    use rocksdb::properties::ESTIMATE_NUM_KEYS;

    use super::*;
    use crate::commit::CommitId;
    use crate::storage::setup_repo;

    fn checkpoint_db_path(db: &PersistenceLayer) -> PathBuf {
        db.db_instance.path().to_path_buf()
    }

    fn string_of_length(len: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(any::<char>(), len).prop_map(|v| v.into_iter().collect())
    }

    fn create_and_clear_dir(path: &Path) {
        if path.exists() {
            std::fs::remove_dir_all(path).expect("Should be able to remove dir");
        }
        std::fs::create_dir_all(path).expect("Should be able to create dir");
    }

    fn assert_blob_value<Data: AsRef<[u8]>>(
        db: &impl ReadableKeyValueStore,
        blob: &HashedData<Data>,
    ) {
        let retrieved = db
            .blob_get(blob.hash())
            .expect("Expected blob to exist in persistence layer");
        assert_eq!(retrieved.as_ref(), blob.data());
    }

    fn assert_blob_missing(db: &impl ReadableKeyValueStore, hash: Hash) {
        assert!(matches!(
            db.blob_get(hash),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
    }

    /// A file in a commit directory, identified precisely enough to tell one left untouched from
    /// one that was removed and written again with the same contents: the same inode, modified at
    /// the same time, holding the same bytes.
    #[derive(Debug, PartialEq, Eq)]
    struct CommitFile {
        name: std::ffi::OsString,
        inode: u64,
        modified: std::time::SystemTime,
        contents: Hash,
    }

    /// Every file in `path`, with nothing filtered out. Opening a commit read-only writes nothing
    /// into it - not even an info log, which only a writeable open produces - so the whole
    /// directory is expected to hold still.
    fn commit_dir_contents(path: &Path) -> Vec<CommitFile> {
        use std::os::unix::fs::MetadataExt;

        let mut contents: Vec<_> = std::fs::read_dir(path)
            .expect("Commit directory should be readable")
            .map(|entry| {
                let entry = entry.expect("Directory entry should be readable");
                let metadata = entry.metadata().expect("Metadata should be readable");

                assert!(
                    metadata.is_file(),
                    "A commit directory should hold only files"
                );

                CommitFile {
                    name: entry.file_name(),
                    inode: metadata.ino(),
                    modified: metadata
                        .modified()
                        .expect("Modification time should be readable"),
                    contents: Hash::hash_bytes(
                        &std::fs::read(entry.path()).expect("File should be readable"),
                    ),
                }
            })
            .collect();

        contents.sort_by(|left, right| left.name.cmp(&right.name));
        contents
    }

    fn assert_key_missing(db: &impl ReadableKeyValueStore, key: impl AsRef<[u8]>) {
        assert!(matches!(
            db.get(key),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
    }

    #[test]
    fn test_new_persistence_layer() {
        let (_keepalive, repo) = setup_repo();
        let db_a =
            PersistenceLayer::new(&repo).expect("Should be able to create new persistence layer");

        let db_b = PersistenceLayer::new(&repo)
            .expect("Should be able to create another persistence layer");

        let path_a = checkpoint_db_path(&db_a);
        let path_b = checkpoint_db_path(&db_b);

        // check that the directories are different
        assert!(path_a != path_b);

        drop(db_a);
        drop(db_b);

        // Check that after dropping the databases, the directories are removed - since they are not
        // a committed database.
        assert!(!path_a.exists());
        assert!(
            !path_a
                .parent()
                .expect("Should have a db_<random> parent")
                .exists()
        );
        assert!(!path_b.exists());
        assert!(
            !path_b
                .parent()
                .expect("Should have a db_<random> parent")
                .exists()
        );
    }

    #[test]
    fn test_basic_ops() {
        let test = |value_a: String, value_b: String| {
            let (_keepalive, repo) = setup_repo();
            let db = PersistenceLayer::new(&repo)
                .expect("Should be able to create new persistence layer");

            let blob = HashedData::from_data(value_a.as_bytes());
            let key = blob.hash();

            // Initially the key should not be found
            assert_blob_missing(&db, key);

            db.blob_set(blob.hash(), blob.data())
                .expect("Should be able to set a value");

            assert_blob_value(&db, &blob);

            let blob2 = HashedData::from_data(value_b.as_bytes());
            let key2 = blob2.hash();
            db.blob_set(blob2.hash(), blob2.data())
                .expect("Should be able to set another value");

            assert_blob_value(&db, &blob2);
            assert_blob_value(&db, &blob);

            assert_eq!(
                db.db_instance
                    .property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                Ok(Some("2".to_string()))
            );

            db.blob_delete(key)
                .expect("Should be able to delete the value");
            assert_blob_missing(&db, key);

            assert_eq!(
                db.db_instance
                    .property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                Ok(Some("1".to_string()))
            );

            // These operations shouldn't affect the data column family
            assert_key_missing(&db, blob.hash());
            assert_key_missing(&db, blob2.hash());

            db.blob_delete(key2)
                .expect("Should be able to delete the second value");
            assert_blob_missing(&db, key2);

            let nonexistent_blob = HashedData::from_data(b"non_existent");
            assert!(matches!(db.blob_delete(nonexistent_blob.hash()), Ok(())));

            assert_eq!(
                db.db_instance
                    .property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                Ok(Some("0".to_string()))
            );
        };

        proptest!(|(value_a in string_of_length(10), value_b in string_of_length(12))| {
            test(value_a, value_b);
        });
    }

    #[test]
    fn test_clone_semantics() {
        // Create database A.
        // Perform operations on A.
        // Clone A to B.

        // Perform operations on B.
        // Ensure A's state is unchanged.

        // Perform operations on A.
        // Ensure B's state is unchanged.

        // We delete and recreate the directory to flush the metadb
        let (_keepalive, repo) = setup_repo();

        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        let initial_blob = HashedData::from_data(b"initial_value");
        let another_blob = HashedData::from_data(b"another_value");
        let third_blob = HashedData::from_data(b"third_value");

        db_a.blob_set(initial_blob.hash(), initial_blob.data())
            .expect("Failed to set initial blob in A");
        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");

        db_b.blob_set(another_blob.hash(), another_blob.data())
            .expect("Failed to set another blob in B");

        // get methods borrow the db so we have to drop the borrow to mutate the db in the next scope
        assert_blob_value(&db_a, &initial_blob);

        // Wrap in a scope so we can drop the db's later
        {
            assert_blob_value(&db_b, &initial_blob);

            db_a.blob_set(third_blob.hash(), third_blob.data())
                .expect("Failed to set third blob in A");
            assert_blob_missing(&db_b, third_blob.hash());
        }

        let path_a = checkpoint_db_path(&db_a);
        let path_b = checkpoint_db_path(&db_b);

        // We are dropping a db connection which is tied to a checkpoint directory, but not a commit.
        // Hence, when we drop the db, we also no longer care about this ephemeral checkpoint
        drop(db_b);
        assert!(!path_b.exists());

        // We created db_a with `new`, so dropping it should also delete its directory - if there are no checkpoints depending on it.
        drop(db_a);
        assert!(!path_a.exists());
    }

    #[test]
    fn test_multiple_checkpoints() {
        let (_keepalive, repo) = setup_repo();

        let blob = HashedData::from_data(b"some_value");

        // A -> (B, C)
        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.blob_set(blob.hash(), blob.data())
            .expect("Failed to set blob in A");

        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        let db_c = db_a.try_clone(&repo).expect("Failed to clone DB A to C");

        let checkpoint_path = checkpoint_db_path(&db_a);
        drop(db_a);
        assert!(!checkpoint_path.exists());

        assert_blob_value(&db_b, &blob);
        assert_blob_value(&db_c, &blob);

        let checkpoint_path = checkpoint_db_path(&db_b);
        drop(db_b);
        assert!(!checkpoint_path.exists());

        let checkpoint_path = checkpoint_db_path(&db_c);
        drop(db_c);
        assert!(!checkpoint_path.exists());
    }

    #[test]
    fn test_commit_and_checkout() {
        let (_keepalive, repo) = setup_repo();

        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        let blob = HashedData::from_data(b"some_value");
        db_a.blob_set(blob.hash(), blob.data())
            .expect("Failed to set blob in A");

        let commit_id = CommitId::from(Hash::hash_bytes(b"commit_1"));
        db_a.commit(&repo, &commit_id)
            .expect("Failed to commit DB A");
        let path_a = checkpoint_db_path(&db_a);
        drop(db_a);
        eprintln!("Path A: {path_a:?}");
        assert!(!path_a.exists());

        let db_b = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB B");

        assert_blob_value(&db_b, &blob);
        let zero_digest: [u8; Hash::DIGEST_SIZE] = [0u8; 32];
        let hash_zero_digest = Hash::from(zero_digest);
        assert_blob_missing(&db_b, hash_zero_digest);
        assert_key_missing(&db_b, [1u8; 32]);

        let path_b = repo.database_commit_dir(&commit_id);
        drop(db_b);
        assert!(path_b.exists(), "Checked out DB should persist on disk");
    }

    #[test]
    fn test_nonexistent_checkout() {
        let (_keepalive, repo) = setup_repo();

        let commit_id = CommitId::from(Hash::hash_bytes(b"nonexistent_commit"));
        let db_result = PersistenceLayer::checkout(&repo, &commit_id);
        assert!(matches!(db_result, Err(OperationalError::CommitNotFound)));

        let read_only_result = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id);
        assert!(matches!(
            read_only_result,
            Err(OperationalError::CommitNotFound)
        ));
    }

    /// Commit a database holding `blob`, and `"value"` under the key `"key"`.
    fn commit_populated_db<Data: AsRef<[u8]>>(
        repo: &DirectoryManager,
        blob: &HashedData<Data>,
    ) -> CommitId {
        let db = PersistenceLayer::new(repo).expect("Failed to create the database");
        db.blob_set(blob.hash(), blob.data())
            .expect("Failed to set the blob");
        db.set(b"key", b"value").expect("Failed to set the value");

        let commit_id = CommitId::from(Hash::hash_bytes(b"read_only_commit"));
        db.commit(repo, &commit_id).expect("Failed to commit");

        commit_id
    }

    #[test]
    fn test_read_only_checkout_reads_the_commit_in_place() {
        let (_keepalive, repo) = setup_repo();

        let blob = HashedData::from_data(b"read_only_value");
        let commit_id = commit_populated_db(&repo, &blob);
        let commit_path = repo.database_commit_dir(&commit_id);
        let commit_contents = commit_dir_contents(&commit_path);

        let read_only = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id)
            .expect("Read-only checkout should succeed");

        // The committed database is read where it lies: no working copy was made.
        assert_eq!(read_only.db_instance.path(), commit_path);

        assert_blob_value(&read_only, &blob);
        assert_eq!(
            read_only
                .get(b"key")
                .expect("Reading the value should succeed")
                .as_ref(),
            b"value"
        );
        assert_blob_missing(&read_only, Hash::from([0u8; Hash::DIGEST_SIZE]));
        assert_key_missing(&read_only, b"absent");

        // The same commit can be viewed any number of times over.
        let also_read_only = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id)
            .expect("A second read-only checkout should succeed");
        assert_blob_value(&also_read_only, &blob);
        drop(also_read_only);
        drop(read_only);

        // Dropping a read-only view leaves the commit alone: it neither destroys nor modifies it,
        // and the commit remains available for further checkouts. Not one file has changed and no
        // file has appeared - opening a commit read-only, twice over, writes nothing into it at
        // all, which is what lets commit directories be shared and read concurrently.
        assert!(commit_path.exists(), "The commit should still exist");
        assert_eq!(
            commit_dir_contents(&commit_path),
            commit_contents,
            "The commit directory should be untouched, down to the last file"
        );

        let checked_out = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Checking the commit out again should succeed");
        assert_blob_value(&checked_out, &blob);
    }

    #[test]
    fn test_read_only_checkout_to_writeable() {
        let (_keepalive, repo) = setup_repo();

        let blob = HashedData::from_data(b"read_only_value");
        let commit_id = commit_populated_db(&repo, &blob);
        let commit_path = repo.database_commit_dir(&commit_id);
        let commit_contents = commit_dir_contents(&commit_path);

        let read_only = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id)
            .expect("Read-only checkout should succeed");

        // Switching to writeable switches to a new directory
        let writeable = read_only
            .to_writeable(&repo)
            .expect("Making the read-only view writeable should succeed");

        let working_path = checkpoint_db_path(&writeable);
        assert_ne!(working_path, commit_path);
        assert_blob_value(&writeable, &blob);

        let new_blob = HashedData::from_data(b"new_value");
        writeable
            .blob_set(new_blob.hash(), new_blob.data())
            .expect("Writing to the working copy should succeed");
        writeable
            .set(b"key", b"new_value")
            .expect("Writing to the working copy should succeed");

        // The read-only view - and the commit it reads - are unaffected.
        assert_blob_missing(&read_only, new_blob.hash());
        assert_eq!(
            read_only
                .get(b"key")
                .expect("Reading the value should succeed")
                .as_ref(),
            b"value"
        );
        assert_eq!(commit_dir_contents(&commit_path), commit_contents);

        drop(read_only);

        // Only the working copy is temporary.
        drop(writeable);
        assert!(!working_path.exists());
        assert!(commit_path.exists());
    }

    #[test]
    fn test_checkout_of_incomplete_commit_is_not_found() {
        let (_keepalive, repo) = setup_repo();

        // A directory holding no `CURRENT` file, of the kind an interrupted removal by an older
        // version of `commit` leaves behind, is not a commit - and is reported as one that cannot
        // be found, rather than as a failure to open the database it does not hold.
        let commit_id = CommitId::from(Hash::hash_bytes(b"incomplete_commit"));
        let commit_path = repo.database_commit_dir(&commit_id);

        std::fs::create_dir_all(&commit_path).expect("Failed to create the incomplete commit");
        std::fs::write(commit_path.join("000123.sst"), b"leftover")
            .expect("Failed to write into the incomplete commit");

        let db_result = PersistenceLayer::checkout(&repo, &commit_id);
        assert!(matches!(db_result, Err(OperationalError::CommitNotFound)));

        let read_only_result = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id);
        assert!(matches!(
            read_only_result,
            Err(OperationalError::CommitNotFound)
        ));
    }

    #[test]
    fn test_clone_commit_and_checkout() {
        // A -> (mutate A) -> B (clone) -> (mutate B) -> B (commit: "commit_1") -> (mutate B)
        // D (checkout: "commit_1")

        let temp_dir_str = "/tmp/persistence_layer_test_clone_commit_checkout";
        let temp_dir = Path::new(temp_dir_str);
        create_and_clear_dir(temp_dir);
        let repo = DirectoryManager::new(temp_dir).expect("Failed to create directory manager");

        let blob_a = HashedData::from_data(b"some_value");
        let blob_b = HashedData::from_data(b"another_value");
        let blob_c = HashedData::from_data(b"third_value");

        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.blob_set(blob_a.hash(), blob_a.data())
            .expect("Failed to set blob in A");

        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        db_b.blob_set(blob_b.hash(), blob_b.data())
            .expect("Failed to set blob in B");

        let commit_id = CommitId::from(Hash::hash_bytes(b"commit_1"));
        db_b.commit(&repo, &commit_id)
            .expect("Failed to commit DB B");

        db_b.blob_set(blob_c.hash(), blob_c.data())
            .expect("Failed to set blob in B");

        drop(db_a);
        drop(db_b);

        // We should observe blob a & b after checking out the commit, but not c.
        let db_c = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB C");
        assert_blob_value(&db_c, &blob_a);
        assert_blob_value(&db_c, &blob_b);
        assert_blob_missing(&db_c, blob_c.hash());

        let path_c = repo.database_commit_dir(&commit_id);
        drop(db_c);
        assert!(path_c.exists(), "Checked out DB should persist on disk");
    }

    #[test]
    fn test_implied_mutability() {
        // A -> (mutate A) -> commit A (commit: "commit_1")
        // C (load "commit_1") -> (mutate C) -> commit C (commit: "commit_2") -> (mutate C)
        // Check commit_1 && commit_2

        let (_keepalive, repo) = setup_repo();

        let blob_a = HashedData::from_data(b"some_value");
        let blob_b = HashedData::from_data(b"another_value");
        let commit_id_1 = CommitId::from(Hash::hash_bytes(b"commit_1"));
        let commit_id_2 = CommitId::from(Hash::hash_bytes(b"commit_2"));
        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.blob_set(blob_a.hash(), blob_a.data())
            .expect("Failed to set blob in A");
        db_a.commit(&repo, &commit_id_1)
            .expect("Failed to commit DB A");

        let db_c = PersistenceLayer::checkout(&repo, &commit_id_1)
            .expect("Failed to checkout commit into DB C");
        db_c.blob_set(blob_b.hash(), blob_b.data())
            .expect("Failed to set blob in C");
        db_c.commit(&repo, &commit_id_2)
            .expect("Failed to commit DB C");
        db_c.blob_delete(blob_a.hash())
            // db_c.blob_delete(blob_a.hash())
            .expect("Failed to delete blob a in C");
        drop(db_a);
        drop(db_c);

        // check commit 1
        let db_check_1 =
            PersistenceLayer::checkout(&repo, &commit_id_1).expect("Failed to checkout commit 1");
        assert_blob_value(&db_check_1, &blob_a);
        assert_blob_missing(&db_check_1, blob_b.hash());

        // check commit 2
        let db_check_2 =
            PersistenceLayer::checkout(&repo, &commit_id_2).expect("Failed to checkout commit 2");
        assert_blob_value(&db_check_2, &blob_a);
        assert_blob_value(&db_check_2, &blob_b);
    }

    /// Two databases publish one commit id at the same moment while a third reads it.
    ///
    /// Which of the two wins the publishing rename is not determined, and the loser may well take
    /// the shortcut at the top of `commit` rather than the lost-race branch below it - what the
    /// test pins down is what a reader sees: once the commit is there, it stays there, holding the
    /// state both writers published. Republishing it, as `commit` used to, would have unlinked it
    /// under the reader.
    #[test]
    fn test_concurrent_commits_of_one_id_leave_a_reader_undisturbed() {
        let tempdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let blob = HashedData::from_data(b"contended_value");
        let commit_id = CommitId::from(Hash::hash_bytes(b"contended_commit"));

        let start = std::sync::Barrier::new(3);

        std::thread::scope(|scope| {
            let (repo, blob, commit_id, start) = (&repo, &blob, &commit_id, &start);

            for writer in 0..2 {
                scope.spawn(move || {
                    let db = PersistenceLayer::new(repo).expect("Failed to create the database");
                    db.blob_set(blob.hash(), blob.data())
                        .expect("Failed to set the blob");

                    start.wait();
                    db.commit(repo, commit_id).unwrap_or_else(|error| {
                        panic!("Writer {writer} failed to commit: {error}")
                    });
                });
            }

            scope.spawn(move || {
                start.wait();

                // Read until the commit has been seen a good number of times, rather than for a
                // fixed number of attempts: a checkout of a commit that is not there yet returns
                // at once, so a fixed count would race past the writers and read nothing at all.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                let mut reads = 0;

                while reads < 16 && std::time::Instant::now() < deadline {
                    match PersistenceLayer::checkout(repo, commit_id) {
                        Ok(db) => {
                            assert_blob_value(&db, blob);
                            reads += 1;
                        }

                        // Only ever before the first writer has published.
                        Err(OperationalError::CommitNotFound) => assert_eq!(
                            reads, 0,
                            "A published commit should not go missing while it is being read"
                        ),

                        Err(error) => panic!("Checking the commit out failed: {error}"),
                    }
                }

                assert!(
                    reads > 0,
                    "The reader should have seen the commit published"
                );
            });
        });

        let db = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("The commit should be readable once both writers are done");
        assert_blob_value(&db, &blob);
    }

    #[test]
    fn test_recommitting_a_published_id_leaves_its_files_untouched() {
        let tempdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");
        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");

        let blob = HashedData::from_data(b"some_value");
        db_a.blob_set(blob.hash(), blob.data())
            .expect("Failed to set blob in A");

        let commit_id = CommitId::from(Hash::hash_bytes(b"commit_1"));
        db_a.commit(&repo, &commit_id)
            .expect("Failed to commit DB A");

        let commit_path = repo.database_commit_dir(&commit_id);
        let commit_contents = commit_dir_contents(&commit_path);

        let blob_2 = HashedData::from_data(b"another_value");
        db_a.blob_set(blob_2.hash(), blob_2.data())
            .expect("Failed to set blob 2 in A");

        // A commit id is a hash of the state committed under it, so committing the same id a second
        // time is a no-op: the published commit already holds that state, and is left exactly as it
        // was found rather than being removed and written again. Every file is still the same
        // inode, modified when it always was, holding the bytes it always held - so a read-only
        // checkout of this commit, which reads these very files, is undisturbed by the second
        // commit.
        db_a.commit(&repo, &commit_id)
            .expect("Committing the same id a second time should succeed");
        assert_eq!(
            commit_dir_contents(&commit_path),
            commit_contents,
            "The published commit should be untouched, down to the files on disk"
        );

        drop(db_a);

        // The commit holds the state it was published with. The writes made afterwards are not part
        // of it - they were never committed under an id of their own.
        let db_a = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB A");

        assert_blob_value(&db_a, &blob);
        assert_blob_missing(&db_a, blob_2.hash());
    }
    /// A published commit is never rewritten, so committing the same id again disturbs neither a
    /// [`ReadOnlyPersistenceLayer`] already reading it nor one opened afterwards.
    #[test]
    fn test_duplicate_commit_leaves_read_only_checkouts_readable() {
        let (_keepalive, repo) = setup_repo();

        let blob = HashedData::from_data(b"read_only_value");
        let commit_id = commit_populated_db(&repo, &blob);
        let commit_path = repo.database_commit_dir(&commit_id);
        let commit_contents = commit_dir_contents(&commit_path);

        let read_only = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id)
            .expect("Read-only checkout should succeed");

        // A working copy of that commit, committing the same state under the same id again.
        let writeable = read_only
            .to_writeable(&repo)
            .expect("Making the read-only view writeable should succeed");
        writeable
            .commit(&repo, &commit_id)
            .expect("Committing the same state again should succeed");

        assert_eq!(
            commit_dir_contents(&commit_path),
            commit_contents,
            "The published commit should be untouched"
        );

        // The view opened before the second commit still reads it.
        assert_blob_value(&read_only, &blob);
        assert_eq!(
            read_only
                .get(b"key")
                .expect("Reading the value should succeed")
                .as_ref(),
            b"value"
        );

        // So does one opened after it.
        let reopened = ReadOnlyPersistenceLayer::checkout_read_only(&repo, &commit_id)
            .expect("A read-only checkout made after the second commit should succeed");
        assert_blob_value(&reopened, &blob);
    }

    #[test]
    fn test_commit_replaces_incomplete_commit_dir() {
        let tempdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");
        let db = PersistenceLayer::new(&repo).expect("Failed to create DB");

        let blob = HashedData::from_data(b"some_value");
        db.blob_set(blob.hash(), blob.data())
            .expect("Failed to set blob");

        // A directory holding no `CURRENT` file, of the kind an interrupted removal by an older
        // version of `commit` leaves behind. It is in the way of the publishing rename, and
        // replacing it is the repair.
        let commit_id = CommitId::from(Hash::hash_bytes(b"incomplete"));
        let commit_path = repo.database_commit_dir(&commit_id);

        std::fs::create_dir_all(&commit_path).expect("Failed to create the incomplete commit");
        std::fs::write(commit_path.join("000123.sst"), b"leftover")
            .expect("Failed to write into the incomplete commit");

        db.commit(&repo, &commit_id)
            .expect("Committing over an incomplete commit should succeed");

        assert!(
            is_published_commit(&commit_path),
            "The incomplete commit should have been replaced by a published one"
        );
        assert!(
            !commit_path.join("000123.sst").exists(),
            "The incomplete commit should not have been published in part"
        );

        // An empty directory is renamed over rather than repaired, so it takes the same path
        // through `commit` that an unoccupied commit path does.
        let empty_id = CommitId::from(Hash::hash_bytes(b"empty"));
        let empty_path = repo.database_commit_dir(&empty_id);

        std::fs::create_dir_all(&empty_path).expect("Failed to create the empty commit dir");

        db.commit(&repo, &empty_id)
            .expect("Committing over an empty directory should succeed");

        assert!(is_published_commit(&empty_path));

        drop(db);

        for id in [commit_id, empty_id] {
            let db = PersistenceLayer::checkout(&repo, &id).expect("Failed to checkout commit");
            assert_blob_value(&db, &blob);
        }
    }

    #[test]
    fn test_data_basic_ops() {
        let test = |key: String, value: String, value2: String| {
            let repo = DirectoryManager::new(std::path::Path::new("/tmp/test_data_basic_ops"))
                .expect("Failed to create directory manager");
            let db = PersistenceLayer::new(&repo)
                .expect("Should be able to create new persistence layer");

            // Initially the key should not be found
            assert!(matches!(
                db.get(&key),
                Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
            ));

            db.set(&key, &value).expect("Should be able to set a value");

            {
                // Now the key should be found
                let retrieved = db.get(&key).expect("Should be able to get the value");
                assert_eq!(retrieved.as_ref(), value.as_bytes());
            }

            db.set(&key, &value2)
                .expect("Should be able to set another value for the same key");

            {
                // Now the key should return the new value
                let retrieved = db.get(&key).expect("Should be able to get the value");
                assert_eq!(retrieved.as_ref(), value2.as_bytes());
            }

            {
                // Another key should still be unset
                let other_key = format!("other_{key}");
                assert!(matches!(
                    db.get(&other_key),
                    Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
                ));
            }

            db.set(&key, &value).expect("Should be able to reset value");
            db.write(&key, value.len(), &value2)
                .expect("Should be able to extend the value for the same key");

            {
                // Now the key should return the new value
                let retrieved = db.get(&key).expect("Should be able to get the value");
                let mut new_value = value.clone();
                new_value.insert_str(value.len(), &value2);
                assert_eq!(retrieved.as_ref(), new_value.as_bytes());
            }

            db.set(&key, &value).expect("Should be able to reset value");
            let start_index = value.len() - value.len() / 2;
            db.write(&key, start_index, &value2)
                .expect("Should be able to overwrite and extend the value for the same key");

            {
                // Now the key should return the new value
                let retrieved = db.get(&key).expect("Should be able to get the value");
                let mut new_value = value.as_bytes().to_vec();
                let data = value2.as_bytes();

                let overwrite_len = std::cmp::min(new_value.len() - start_index, data.len());
                new_value[start_index..start_index + overwrite_len]
                    .copy_from_slice(&data[..overwrite_len]);
                if data.len() > overwrite_len {
                    new_value.extend_from_slice(&data[overwrite_len..]);
                }
                assert_eq!(retrieved.as_ref(), new_value.as_slice());
            }

            db.set(&key, &value).expect("Should be able to reset value");
            let data_to_write = &value2.as_bytes()[0..value2.len() / 2];
            db.write(&key, 1, data_to_write)
                .expect("Should be able to overwrite part of the value for the same key");

            {
                // Now the key should return the new value
                let retrieved = db.get(&key).expect("Should be able to get the value");
                let mut new_value = value.as_bytes().to_vec();

                let offset = 1;
                let overwrite_len =
                    std::cmp::min(new_value.len().saturating_sub(offset), data_to_write.len());
                if overwrite_len > 0 {
                    new_value[offset..offset + overwrite_len]
                        .copy_from_slice(&data_to_write[..overwrite_len]);
                }
                if data_to_write.len() > overwrite_len {
                    new_value.extend_from_slice(&data_to_write[overwrite_len..]);
                }
                assert_eq!(retrieved.as_ref(), new_value.as_slice());
            }

            {
                // Another key should still be unset
                let other_key = format!("other_{key}");
                assert!(matches!(
                    db.get(&other_key),
                    Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
                ));
            }

            {
                // These operations shouldn't affect the blob column family.
                let blob1 = HashedData::from_data(value.as_bytes());
                let blob2 = HashedData::from_data(value2.as_bytes());
                assert!(matches!(
                    db.blob_get(blob1.hash()),
                    Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
                ));
                assert!(matches!(
                    db.blob_get(blob2.hash()),
                    Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
                ));
            }

            db.delete(&key).expect("Should be able to delete the value");
            assert!(matches!(
                db.get(&key),
                Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
            ));
        };

        proptest!(|(key in string_of_length(8), value in string_of_length(10), value2 in string_of_length(12))| {
            test(key, value, value2);
        });
    }
}
