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
use crate::repo::DirectoryManager;
use crate::storage::KeyValueStore;
use crate::storage::PersistentKeyValueStore;

/// The name of the column family used for storing blob-keyed data.
const BLOB_CF: &str = "blob";

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

fn rocksdb_default_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    set_memtable_to_hash_link_list(&mut options);
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
                ColumnFamilyDescriptor::new("default", rocksdb_checkpoint_options()),
                ColumnFamilyDescriptor::new(BLOB_CF, rocksdb_checkpoint_options()),
            ],
        )
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

        Ok(Self {
            db_instance: ManuallyDrop::new(temp_db),
            _tempdir: tempdir,
        })
    }

    fn blob_cf(&self) -> &rocksdb::ColumnFamily {
        self.db_instance
            .cf_handle(BLOB_CF)
            .expect("the rocksdb instance should always contain the data cf")
    }
}

impl KeyValueStore for PersistenceLayer {
    type Repo = DirectoryManager;

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
        })
    }

    fn try_clone(&self, repo: &Self::Repo) -> Result<Self, OperationalError> {
        Self::clone_as_checkpoint(&self.db_instance, repo)
    }

    fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        let key = key.as_ref();
        let entry = self
            .db_instance
            .get_pinned_cf(self.blob_cf(), key)
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

    fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        let value = self.db_instance.get_pinned(key.as_ref()).map_err(|error| {
            OperationalError::GetFailed {
                column: "default".to_owned(),
                key: key.as_ref().to_owned(),
                error,
            }
        })?;

        match value {
            Some(value) => Ok(value),
            None => Err(InvalidArgumentError::KeyNotFound)?,
        }
    }

    fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        self.db_instance
            .put(key.as_ref(), value.as_ref())
            .map_err(|error| OperationalError::PutFailed {
                column: "default".to_owned(),
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
                column: "default".to_owned(),
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
        let checkpoint_path = repo.database_commit_dir(id);

        // If the path already exists, we overwrite the existing commit. This is highly unlikely to
        // happen anyway if the commits are a hash of the content.
        if Path::exists(&checkpoint_path) {
            std::fs::remove_dir_all(&checkpoint_path).map_err(|error| {
                OperationalError::DirRemovalFailed {
                    path: checkpoint_path.clone(),
                    error,
                }
            })?;

            log::warn!("Overwriting existing commit: {}", id.hex_encode());
        }

        self.commit_to_path(&checkpoint_path)
    }

    fn checkout_from_path(
        commit_path: &Path,
        working_path: TempDir,
    ) -> Result<Self, OperationalError> {
        if !Path::exists(commit_path) {
            return Err(OperationalError::CommitNotFound);
        };

        // Open the previous commitment from the given source path in read-only mode
        let read_only_database = rocksdb::DB::open_cf_descriptors_read_only(
            &rocksdb_checkpoint_options(),
            commit_path,
            [
                ColumnFamilyDescriptor::new(BLOB_CF, rocksdb_checkpoint_options()),
                ColumnFamilyDescriptor::new("default", rocksdb_checkpoint_options()),
            ],
            false,
        )
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

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
                ColumnFamilyDescriptor::new("default", rocksdb_checkpoint_options()),
                ColumnFamilyDescriptor::new(BLOB_CF, rocksdb_checkpoint_options()),
            ],
        )
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

        Ok(Self {
            db_instance: ManuallyDrop::new(database),
            _tempdir: working_path,
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

    fn assert_blob_value<Data: AsRef<[u8]>>(db: &PersistenceLayer, blob: &HashedData<Data>) {
        let retrieved = db
            .blob_get(blob.hash())
            .expect("Expected blob to exist in persistence layer");
        assert_eq!(retrieved.as_ref(), blob.data());
    }

    fn assert_blob_missing(db: &PersistenceLayer, hash: Hash) {
        assert!(matches!(
            db.blob_get(hash),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
    }

    fn assert_key_missing(db: &PersistenceLayer, key: impl AsRef<[u8]>) {
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

    #[test]
    fn test_duplicate_commit() {
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

        let blob_2 = HashedData::from_data(b"another_value");
        db_a.blob_set(blob_2.hash(), blob_2.data())
            .expect("Failed to set blob 2 in A");

        // Committing again with the same id should work
        let result = db_a.commit(&repo, &commit_id);
        assert!(result.is_ok());

        drop(db_a);

        // Loading after the second commit should contain both blobs
        let db_a = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB A");

        let retrieved = db_a
            .blob_get(blob.hash())
            .expect("Failed to get blob from A");
        assert_eq!(retrieved.as_ref(), blob.data());

        let retrieved_2 = db_a
            .blob_get(blob_2.hash())
            .expect("Failed to get blob 2 from A");
        assert_eq!(retrieved_2.as_ref(), blob_2.data());
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
