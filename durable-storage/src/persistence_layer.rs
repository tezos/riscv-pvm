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
//! ```
//! <repo_path>:
//!    temporary/
//!        db_<random>/checkpoint/
//!            <rocksdb internals>
//! ```

use std::mem::ManuallyDrop;

use rocksdb::checkpoint::Checkpoint;
use tempfile::TempDir;

use crate::repo::DirectoryManager;
use crate::repo::DirectoryManagerError;

/// Type alias for a 32-byte hash used for identifying key-value blobs & commits.
type Hash = [u8; 32];

/// Represents a key-value blob to be stored in the persistence layer.
///
/// Invariant: [`HashedData`] is content-addressable by the stored hash.
#[derive(Debug, Clone)]
pub struct HashedData<'d> {
    /// The BLAKE3 hash of the value.
    hash: Hash,

    /// The addressable content of the blob.
    value: &'d [u8],
}

impl<'d> HashedData<'d> {
    /// Create a new [`HashedData`] instance from the given byte-array.
    pub fn from_value(value: &'d [u8]) -> Self {
        let hash = blake3::hash(value).into();
        Self { hash, value }
    }
}

/// Errors encountered when interacting with the persistence layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Key not found")]
    KeyNotFound,

    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("Directory manager error: {0}")]
    DirectoryManager(#[from] DirectoryManagerError),
}

/// Mode in which the [`PersistenceLayer`] was instantiated.
enum Mode {
    /// Either a new database, or a clone of an existing database.
    Temporary {
        /// We keep it here to keep it alive for the lifetime of the rockdb instance
        _tempdir: TempDir,
    },

    /// A database checked out from a specific commit.
    #[expect(dead_code, reason = "TODO: RV-797 Will be used for checkouts")]
    FromCommit,
}

/// These options are used for opening and closing a rocksdb instance.
///
/// Although different fields are used for opening vs. destroying a rocksdb instance, you need to
/// ensure that the options used for destroying are valid with respect to the options used when
/// opening the db. There is no concrete documentation for which options should be kept in sync for
/// open/close, may need to investigate rocksdb source code:
/// <https://github.com/facebook/rocksdb/blob/a1dad12c8c9a7a65fa19d3bc78a5f7687ce6c1bd/db/db_impl/db_impl.cc#L5185>
/// (look for the function destroying a rocksdb instance)
fn rocksdb_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    options.set_error_if_exists(true);
    options
}

/// Persistence layer for durable solution used by the RISC-V PVM.
///
/// Invariants:
/// - The path in `temp_initial_db_path` is unique for each instance of [`PersistenceLayer`] and is
///   assumed to not be modified / known outside of this instance.
pub struct PersistenceLayer {
    /// The underlying handle to the RocksDB instance.
    ///
    /// [`ManuallyDrop`] is used to ensure safety when dropping [`PersistenceLayer`]. Calling
    /// [`rocksdb::DB::destroy`] requires all connections to that path to be closed, which is called
    /// in [`rocksdb::DB`]'s drop method.
    db_instance: ManuallyDrop<rocksdb::DB>,

    /// What mode was the [`PersistenceLayer`] opened in.
    mode: Mode,
}

impl PersistenceLayer {
    /// Creates a new `PersistenceLayer` instance within the given `repo`.
    pub fn new(repo: &DirectoryManager) -> Result<Self, Error> {
        let tempdir = repo.new_temporary_dir()?;
        let new_db_path = tempdir.path().join("checkpoint");

        // To avoid accidentally overwriting an existing database, `error_if_exists` is set.
        let options = rocksdb_options();
        let db = rocksdb::DB::open(&options, &new_db_path)?;

        Ok(Self {
            mode: Mode::Temporary { _tempdir: tempdir },
            db_instance: ManuallyDrop::new(db),
        })
    }

    /// Clones the current `PersistenceLayer` instance.
    ///
    /// Operations on the cloned instance will have no effect on the original instance.
    pub fn try_clone(&self, repo: &DirectoryManager) -> Result<Self, Error> {
        let tempdir = repo.new_temporary_dir()?;
        let checkpoint_path = tempdir.path().join("checkpoint");

        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        Checkpoint::new(&self.db_instance)?.create_checkpoint(&checkpoint_path)?;

        // We expect the db at this checkpoint to exist.
        let db = rocksdb::DB::open(&rocksdb::Options::default(), &checkpoint_path)?;

        Ok(Self {
            mode: Mode::Temporary { _tempdir: tempdir },
            db_instance: ManuallyDrop::new(db),
        })
    }

    /// Retrieves a value associated with the given key.
    pub fn get(&self, key: &Hash) -> Result<impl AsRef<[u8]>, Error> {
        self.db_instance.get_pinned(key)?.ok_or(Error::KeyNotFound)
    }

    /// Sets a value for the given key.
    pub fn set(&mut self, blob: &HashedData) -> Result<(), Error> {
        Ok(self.db_instance.put(blob.hash, blob.value)?)
    }

    /// Deletes a value associated with the given key.
    pub fn delete(&mut self, key: &Hash) -> Result<(), Error> {
        Ok(self.db_instance.delete(key)?)
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

        // SAFETY: Although marked as safe, destroy on a path requires all rocksdb connections to
        // this path to be closed. This is why we need to manual drop the db_instance first & the
        // invariants of `PersistenceLayer` to be upheld.
        if let Mode::Temporary { .. } = &self.mode {
            // Destroy the rocksdb at this path. The parent folder will be deleted by the drop
            // method of the tempdir in the mode field.

            let options = rocksdb_options();
            if let Err(e) = rocksdb::DB::destroy(&options, &db_path) {
                log::error!("Failed to destroy temporary rocksdb at {db_path:?}: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use proptest::prelude::Strategy;
    use proptest::prelude::any;
    use proptest::proptest;
    use rocksdb::properties::ESTIMATE_NUM_KEYS;

    use super::*;

    struct TestableTmpdir {
        tempdir: TempDir,
    }

    impl TestableTmpdir {
        fn new() -> Self {
            let tempdir = TempDir::new().expect("Should be able to create temp dir");

            Self { tempdir }
        }

        fn path(&self) -> &Path {
            self.tempdir.path()
        }
    }

    impl Drop for TestableTmpdir {
        fn drop(&mut self) {
            if std::thread::panicking() {
                eprintln!(
                    "Test failed, preserving temp dir at {:?} for inspection",
                    self.tempdir.path()
                );
                self.tempdir.disable_cleanup(true);
            }
        }
    }

    fn checkpoint_db_path(db: &PersistenceLayer) -> PathBuf {
        db.db_instance.path().to_path_buf()
    }

    fn string_of_length(len: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(any::<char>(), len).prop_map(|v| v.into_iter().collect())
    }

    #[test]
    fn test_new_persistence_layer() {
        let tmpdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
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
        let tmpdir = TestableTmpdir::new();

        let test = |value_a: String, value_b: String| {
            let repo =
                DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
            let mut db = PersistenceLayer::new(&repo)
                .expect("Should be able to create new persistence layer");

            let blob = HashedData::from_value(value_a.as_bytes());
            let key = blob.hash;

            // Initially the key should not be found
            assert!(matches!(db.get(&key), Err(Error::KeyNotFound)));

            db.set(&blob).expect("Should be able to set a value");

            {
                // Now the key should be found
                let retrieved = db.get(&key).expect("Should be able to get the value");
                assert_eq!(retrieved.as_ref(), value_a.as_bytes());
            }

            let blob2 = HashedData::from_value(value_b.as_bytes());
            let key2 = blob2.hash;
            db.set(&blob2).expect("Should be able to set another value");

            {
                // Now the second key should be found
                let retrieved = db.get(&key2).expect("Should be able to get the value");
                assert_eq!(retrieved.as_ref(), value_b.as_bytes());

                let retrieved = db.get(&key).expect("Should be able to get the first value");
                assert_eq!(retrieved.as_ref(), value_a.as_bytes());
            }

            assert_eq!(
                db.db_instance.property_value(ESTIMATE_NUM_KEYS),
                Ok(Some("2".to_string()))
            );

            db.delete(&key).expect("Should be able to delete the value");
            assert!(matches!(db.get(&key), Err(Error::KeyNotFound)));

            assert_eq!(
                db.db_instance.property_value(ESTIMATE_NUM_KEYS),
                Ok(Some("1".to_string()))
            );

            db.delete(&key2)
                .expect("Should be able to delete the second value");
            assert!(matches!(db.get(&key2), Err(Error::KeyNotFound)));

            let nonexistent_blob = HashedData::from_value(b"non_existent");
            assert!(matches!(db.delete(&nonexistent_blob.hash), Ok(())));

            assert_eq!(
                db.db_instance.property_value(ESTIMATE_NUM_KEYS),
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
        let tempdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let mut db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        let initial_blob = &HashedData::from_value(b"initial_value");
        let another_blob = &HashedData::from_value(b"another_value");
        let third_blob = &HashedData::from_value(b"third_value");

        db_a.set(initial_blob)
            .expect("Failed to set initial blob in A");
        let mut db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");

        db_b.set(another_blob)
            .expect("Failed to set another blob in B");

        // get methods borrow the db so we have to drop the borrow to mutate the db in the next scope
        {
            let retrieved_a = db_a
                .get(&initial_blob.hash)
                .expect("Failed to get initial blob from A");
            assert_eq!(retrieved_a.as_ref(), b"initial_value");
        }

        // Wrap in a scope so we can drop the db's later
        {
            let retrieved_b = db_b
                .get(&initial_blob.hash)
                .expect("Failed to get initial blob from B");
            assert_eq!(retrieved_b.as_ref(), b"initial_value");

            db_a.set(third_blob).expect("Failed to set third blob in A");
            let retrieved_third_from_b = db_b.get(&third_blob.hash);
            assert!(
                retrieved_third_from_b.is_err()
                    && matches!(retrieved_third_from_b.err(), Some(Error::KeyNotFound))
            );
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
        let tempdir = TestableTmpdir::new();
        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let blob = &HashedData::from_value(b"some_value");

        // A -> (B, C)
        let mut db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.set(blob).expect("Failed to set blob in A");

        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        let db_c = db_a.try_clone(&repo).expect("Failed to clone DB A to C");

        let checkpoint_path = checkpoint_db_path(&db_a);
        drop(db_a);
        assert!(!checkpoint_path.exists());

        {
            let retrieved_b = db_b.get(&blob.hash).expect("Failed to get blob from B");
            assert_eq!(retrieved_b.as_ref(), b"some_value");

            let retrieved_c = db_c.get(&blob.hash).expect("Failed to get blob from C");
            assert_eq!(retrieved_c.as_ref(), b"some_value");
        }

        let checkpoint_path = checkpoint_db_path(&db_b);
        drop(db_b);
        assert!(!checkpoint_path.exists());

        let checkpoint_path = checkpoint_db_path(&db_c);
        drop(db_c);
        assert!(!checkpoint_path.exists());
    }
}
