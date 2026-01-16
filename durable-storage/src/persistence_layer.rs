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

use std::mem::ManuallyDrop;
use std::path::Path;

use octez_riscv_data::hash::Hash;
use rocksdb::checkpoint::Checkpoint;
use tempfile::TempDir;

use crate::repo::DirectoryManager;
use crate::repo::DirectoryManagerError;

/// [`CommitId`]'s are used to generate [`PersistenceLayer`] commits & to checkout specific commits
/// from a [`DirectoryManager`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitId(Hash);

impl From<Hash> for CommitId {
    fn from(hash: Hash) -> Self {
        Self(hash)
    }
}

impl CommitId {
    /// Returns the hex encoded commit id.
    pub fn hex_encode(&self) -> String {
        hex::encode(self.0)
    }
}

/// The name of the column family used for storing [`HashedData`].
const BLOB_CF: &str = "blob";

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
        let hash = Hash::hash_bytes(value);
        Self { hash, value }
    }
}

/// Errors encountered when interacting with the persistence layer.
#[derive(Debug, thiserror::Error)]
pub enum PersistenceLayerError {
    #[error("Commit not found")]
    CommitNotFound,

    #[error("Key not found")]
    KeyNotFound,

    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("Directory manager error: {0}")]
    DirectoryManager(#[from] DirectoryManagerError),
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

/// These options are used for opening and closing a newly created rocksdb instance.
///
/// Although different fields are used for opening vs. destroying a rocksdb instance, you need to
/// ensure that the options used for destroying are valid with respect to the options used when
/// opening the db. There is no concrete documentation for which options should be kept in sync for
/// open/close, may need to investigate rocksdb source code:
/// <https://github.com/facebook/rocksdb/blob/a1dad12c8c9a7a65fa19d3bc78a5f7687ce6c1bd/db/db_impl/db_impl.cc#L5185>
/// (look for the function destroying a rocksdb instance)
fn rocksdb_creation_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    options.set_error_if_exists(true);
    set_memtable_to_hash_link_list(&mut options);
    options
}

/// RocksDB options for when we clone as a checkpoint
fn rocksdb_clone_as_checkpoint_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    set_memtable_to_hash_link_list(&mut options);
    options
}

/// RocksDB options for when we create a DB from
/// a checkoint
fn rocksdb_checkpoint_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    set_memtable_to_hash_link_list(&mut options);
    options
}

/// RocksDB options for the blob column family creation
fn rocksdb_blob_cf_creation_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    set_memtable_to_hash_link_list(&mut options);
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
    /// Creates a checkpoint of the current database at the given `path`.
    fn checkpoint_at(&self, path: &Path) -> Result<(), PersistenceLayerError> {
        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        Ok(Checkpoint::new(&self.db_instance)?.create_checkpoint(path)?)
    }

    /// Creates a new `PersistenceLayer` instance within the given `repo`.
    pub fn new(repo: &DirectoryManager) -> Result<Self, PersistenceLayerError> {
        let tempdir = repo.new_temporary_dir()?;
        let new_db_path = tempdir.path().join("checkpoint");

        // To avoid accidentally overwriting an existing database, `error_if_exists` is set.
        let options = rocksdb_creation_options();
        let mut db = rocksdb::DB::open(&options, &new_db_path)?;
        db.create_cf(BLOB_CF, &rocksdb_blob_cf_creation_options())?;

        Ok(Self {
            db_instance: ManuallyDrop::new(db),
            _tempdir: tempdir,
        })
    }

    fn clone_as_checkpoint(
        db: &rocksdb::DB,
        repo: &DirectoryManager,
    ) -> Result<Self, PersistenceLayerError> {
        let tempdir = repo.new_temporary_dir()?;
        let checkpoint_path = tempdir.path().join("checkpoint");

        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        Checkpoint::new(db)?.create_checkpoint(&checkpoint_path)?;

        let temp_db =
            rocksdb::DB::open_cf(&rocksdb_clone_as_checkpoint_options(), &checkpoint_path, [
                BLOB_CF,
            ])?;

        Ok(Self {
            db_instance: ManuallyDrop::new(temp_db),
            _tempdir: tempdir,
        })
    }

    /// Clones the current `PersistenceLayer` instance.
    ///
    /// Operations on the cloned instance will have no effect on the original instance.
    pub fn try_clone(&self, repo: &DirectoryManager) -> Result<Self, PersistenceLayerError> {
        Self::clone_as_checkpoint(&self.db_instance, repo)
    }

    /// Checks out a specific commit in the repository from the given `repo`
    pub fn checkout(repo: &DirectoryManager, id: &CommitId) -> Result<Self, PersistenceLayerError> {
        let db_path = repo.commit_dir(id);

        // We assume the commit is not found if the folder does not exist.
        if !Path::exists(&db_path) {
            return Err(PersistenceLayerError::CommitNotFound);
        };

        let db = rocksdb::DB::open_cf(&rocksdb_checkpoint_options(), &db_path, [BLOB_CF])?;

        Self::clone_as_checkpoint(&db, repo)
    }

    /// Commits the current state to the repository within the given `repo`
    pub fn commit(
        &self,
        repo: &DirectoryManager,
        id: &CommitId,
    ) -> Result<(), PersistenceLayerError> {
        let checkpoint_path = repo.commit_dir(id);

        // If the path already exists, we overwrite the existing commit. This is highly unlikely to
        // happen anyway if the commits are a hash of the content.
        if Path::exists(&checkpoint_path) {
            std::fs::remove_dir_all(&checkpoint_path)
                .expect("Should be able to remove existing commit directory");

            log::warn!("Overwriting existing commit: {}", id.hex_encode());
        }

        self.checkpoint_at(&checkpoint_path)
    }
}

// Interface used by the Merkle layer which operates over implicitly hashed values.
impl PersistenceLayer {
    fn blob_cf(&self) -> &rocksdb::ColumnFamily {
        self.db_instance
            .cf_handle(BLOB_CF)
            .expect("the rocksdb instance should always contain the data cf")
    }

    /// Retrieves the hashed data associated with its hash as the key.
    pub fn blob_get(&self, key: &Hash) -> Result<impl AsRef<[u8]>, PersistenceLayerError> {
        self.db_instance
            .get_pinned_cf(self.blob_cf(), key.as_ref())?
            .ok_or(PersistenceLayerError::KeyNotFound)
    }

    /// Sets a value for the given key.
    pub fn blob_set(&self, blob: &HashedData) -> Result<(), PersistenceLayerError> {
        Ok(self
            .db_instance
            .put_cf(self.blob_cf(), blob.hash, blob.value)?)
    }

    /// Deletes a value associated with the given key.
    pub fn blob_delete(&self, key: &Hash) -> Result<(), PersistenceLayerError> {
        Ok(self.db_instance.delete_cf(self.blob_cf(), key.as_ref())?)
    }
}

// Interface used by the Data layer which operates over raw key-value pairs.
impl PersistenceLayer {
    /// Retrieves a value associated with the given key.
    pub fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, PersistenceLayerError> {
        self.db_instance
            .get_pinned(key.as_ref())?
            .ok_or(PersistenceLayerError::KeyNotFound)
    }

    /// Sets a value for the given key.
    pub fn set(
        &self,
        key: impl AsRef<[u8]>,
        value: impl AsRef<[u8]>,
    ) -> Result<(), PersistenceLayerError> {
        Ok(self.db_instance.put(key.as_ref(), value.as_ref())?)
    }

    /// Deletes a value associated with the given key.
    pub fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), PersistenceLayerError> {
        Ok(self.db_instance.delete(key.as_ref())?)
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

/// Utilities for testing or benchmarking the persistence layer.
#[cfg(any(test, feature = "bench"))]
pub mod utils {
    use std::path::Path;

    use tempfile::TempDir;

    /// A temporary directory used for testing
    pub struct TestableTmpdir {
        tempdir: TempDir,
    }

    impl TestableTmpdir {
        /// Create a new temporary directory for testing
        pub fn new() -> Self {
            let tempdir = TempDir::new().expect("Should be able to create temp dir");

            Self { tempdir }
        }

        // The path of the temporary directory
        pub fn path(&self) -> &Path {
            self.tempdir.path()
        }
    }

    impl Default for TestableTmpdir {
        fn default() -> Self {
            Self::new()
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
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use octez_riscv_data::hash::Hash;
    use proptest::prelude::Strategy;
    use proptest::prelude::any;
    use proptest::proptest;
    use rocksdb::properties::ESTIMATE_NUM_KEYS;

    use super::utils::TestableTmpdir;
    use super::*;

    /// Helper to get the checkpoint path of a persistence layer.
    fn checkpoint_db_path(db: &PersistenceLayer) -> PathBuf {
        db.db_instance.path().to_path_buf()
    }

    /// Proptest strategy for generating strings of a specific length.
    fn string_of_length(len: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(any::<char>(), len).prop_map(|v| v.into_iter().collect())
    }

    /// Helper to create a test repository and persistence layer.
    fn setup_test_db(tmpdir: &TestableTmpdir) -> (DirectoryManager, PersistenceLayer) {
        let repo = DirectoryManager::new(tmpdir.path())
            .expect("Failed to create directory manager");
        let db = PersistenceLayer::new(&repo)
            .expect("Should be able to create new persistence layer");
        (repo, db)
    }

    #[test]
    fn test_new_persistence_layer() {
        let tmpdir = TestableTmpdir::new();
        let repo = DirectoryManager::new(tmpdir.path())
            .expect("Failed to create directory manager");

        let db_a = PersistenceLayer::new(&repo)
            .expect("Should be able to create new persistence layer");
        let db_b = PersistenceLayer::new(&repo)
            .expect("Should be able to create another persistence layer");

        let path_a = checkpoint_db_path(&db_a);
        let path_b = checkpoint_db_path(&db_b);

        assert_ne!(path_a, path_b, "Each persistence layer should have unique directory");

        drop(db_a);
        drop(db_b);

        // Temporary databases should be cleaned up when dropped
        assert!(!path_a.exists(), "Database A directory should be removed");
        assert!(!path_a.parent().expect("Should have parent").exists(),
                "Database A parent directory should be removed");
        assert!(!path_b.exists(), "Database B directory should be removed");
        assert!(!path_b.parent().expect("Should have parent").exists(),
                "Database B parent directory should be removed");
    }

    #[test]
    fn test_basic_ops() {
        let tmpdir = TestableTmpdir::new();

        let test = |value_a: String, value_b: String| {
            let (_repo, db) = setup_test_db(&tmpdir);

            // Test: set and get first blob
            let blob_a = HashedData::from_value(value_a.as_bytes());
            assert!(matches!(db.blob_get(&blob_a.hash), Err(PersistenceLayerError::KeyNotFound)),
                    "Key should not exist initially");

            db.blob_set(&blob_a).expect("Should be able to set a value");
            let retrieved = db.blob_get(&blob_a.hash).expect("Should retrieve value after set");
            assert_eq!(retrieved.as_ref(), value_a.as_bytes());

            // Test: set and get second blob
            let blob_b = HashedData::from_value(value_b.as_bytes());
            db.blob_set(&blob_b).expect("Should be able to set another value");

            let retrieved_b = db.blob_get(&blob_b.hash).expect("Should retrieve second value");
            assert_eq!(retrieved_b.as_ref(), value_b.as_bytes());
            let retrieved_a = db.blob_get(&blob_a.hash).expect("First value should still exist");
            assert_eq!(retrieved_a.as_ref(), value_a.as_bytes());

            assert_eq!(db.db_instance.property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                       Ok(Some("2".to_string())), "Should have 2 keys");

            // Test: delete first blob
            db.blob_delete(&blob_a.hash).expect("Should be able to delete");
            assert!(matches!(db.blob_get(&blob_a.hash), Err(PersistenceLayerError::KeyNotFound)),
                    "Deleted key should not be found");
            assert_eq!(db.db_instance.property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                       Ok(Some("1".to_string())), "Should have 1 key after deletion");

            // Test: blob CF shouldn't affect default CF
            assert!(matches!(db.get(&blob_a.hash), Err(PersistenceLayerError::KeyNotFound)),
                    "Blob operations shouldn't affect default CF");
            assert!(matches!(db.get(&blob_b.hash), Err(PersistenceLayerError::KeyNotFound)),
                    "Blob operations shouldn't affect default CF");

            // Test: delete second blob and nonexistent blob
            db.blob_delete(&blob_b.hash).expect("Should be able to delete second value");
            assert!(matches!(db.blob_get(&blob_b.hash), Err(PersistenceLayerError::KeyNotFound)));

            let nonexistent_blob = HashedData::from_value(b"non_existent");
            assert!(db.blob_delete(&nonexistent_blob.hash).is_ok(),
                    "Deleting nonexistent key should succeed");

            assert_eq!(db.db_instance.property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                       Ok(Some("0".to_string())), "Should have 0 keys after all deletions");
        };

        proptest!(|(value_a in string_of_length(10), value_b in string_of_length(12))| {
            test(value_a, value_b);
        });
    }

    #[test]
    fn test_clone_semantics() {
        let tempdir = TestableTmpdir::new();
        let (repo, db_a) = setup_test_db(&tempdir);

        let initial_blob = &HashedData::from_value(b"initial_value");
        let another_blob = &HashedData::from_value(b"another_value");
        let third_blob = &HashedData::from_value(b"third_value");

        // Set initial value in A, then clone to B
        db_a.blob_set(initial_blob).expect("Failed to set initial blob in A");
        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");

        // Mutate B - should not affect A
        db_b.blob_set(another_blob).expect("Failed to set another blob in B");

        let retrieved_a = db_a.blob_get(&initial_blob.hash)
            .expect("Failed to get initial blob from A");
        assert_eq!(retrieved_a.as_ref(), b"initial_value", "A should have initial value");

        let retrieved_b = db_b.blob_get(&initial_blob.hash)
            .expect("Failed to get initial blob from B");
        assert_eq!(retrieved_b.as_ref(), b"initial_value", "B should have initial value");

        // Mutate A - should not affect B
        db_a.blob_set(third_blob).expect("Failed to set third blob in A");
        assert!(matches!(db_b.blob_get(&third_blob.hash),
                        Err(PersistenceLayerError::KeyNotFound)),
                "B should not see mutations to A");

        let path_a = checkpoint_db_path(&db_a);
        let path_b = checkpoint_db_path(&db_b);

        // Temporary databases should be cleaned up when dropped
        drop(retrieved_b);
        drop(db_b);
        assert!(!path_b.exists(), "Cloned DB should be removed when dropped");

        drop(retrieved_a);
        drop(db_a);
        assert!(!path_a.exists(), "Original DB should be removed when dropped");
    }

    #[test]
    fn test_multiple_checkpoints() {
        let tempdir = TestableTmpdir::new();
        let (repo, db_a) = setup_test_db(&tempdir);

        let blob = &HashedData::from_value(b"some_value");
        db_a.blob_set(blob).expect("Failed to set blob in A");

        // Create two independent clones
        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        let db_c = db_a.try_clone(&repo).expect("Failed to clone DB A to C");

        let path_a = checkpoint_db_path(&db_a);
        drop(db_a);
        assert!(!path_a.exists(), "Original DB should be cleaned up");

        // Both clones should have the data
        let retrieved_b = db_b.blob_get(&blob.hash).expect("Failed to get blob from B");
        assert_eq!(retrieved_b.as_ref(), b"some_value");

        let retrieved_c = db_c.blob_get(&blob.hash).expect("Failed to get blob from C");
        assert_eq!(retrieved_c.as_ref(), b"some_value");

        // Clean up clones
        let path_b = checkpoint_db_path(&db_b);
        drop(retrieved_b);
        drop(db_b);
        assert!(!path_b.exists(), "Clone B should be cleaned up");

        let path_c = checkpoint_db_path(&db_c);
        drop(retrieved_c);
        drop(db_c);
        assert!(!path_c.exists(), "Clone C should be cleaned up");
    }

    #[test]
    fn test_commit_and_checkout() {
        let tempdir = TestableTmpdir::new();
        let (repo, db_a) = setup_test_db(&tempdir);

        let blob = &HashedData::from_value(b"some_value");
        db_a.blob_set(blob).expect("Failed to set blob in A");

        // Commit the database
        let commit_id: CommitId = Hash::hash_bytes(b"commit_1").into();
        db_a.commit(&repo, &commit_id).expect("Failed to commit DB A");

        let path_a = checkpoint_db_path(&db_a);
        drop(db_a);
        assert!(!path_a.exists(), "Temporary DB should be cleaned up after commit");

        // Checkout the committed state
        let db_b = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB B");

        let retrieved_b = db_b.blob_get(&blob.hash).expect("Failed to get blob from B");
        assert_eq!(retrieved_b.as_ref(), blob.value, "Checked out DB should have committed data");

        // Verify nonexistent keys return errors
        let zero_hash: Hash = [0u8; Hash::DIGEST_SIZE].into();
        assert!(matches!(db_b.blob_get(&zero_hash), Err(PersistenceLayerError::KeyNotFound)),
                "Nonexistent blob key should not be found");
        assert!(matches!(db_b.get(&[1u8; 32]), Err(PersistenceLayerError::KeyNotFound)),
                "Nonexistent data key should not be found");

        let commit_path = repo.commit_dir(&commit_id);
        drop(retrieved_b);
        drop(db_b);
        assert!(commit_path.exists(), "Committed DB should persist on disk after checkout dropped");
    }

    #[test]
    fn test_nonexistent_checkout() {
        let tempdir = TestableTmpdir::new();
        let repo = DirectoryManager::new(tempdir.path())
            .expect("Failed to create directory manager");

        let commit_id: CommitId = Hash::hash_bytes(b"nonexistent_commit").into();
        let result = PersistenceLayer::checkout(&repo, &commit_id);
        assert!(matches!(result, Err(PersistenceLayerError::CommitNotFound)),
                "Checking out nonexistent commit should fail");
    }

    #[test]
    fn test_clone_commit_and_checkout() {
        let tempdir = TestableTmpdir::new();
        let (repo, db_a) = setup_test_db(&tempdir);

        let blob_a = &HashedData::from_value(b"some_value");
        let blob_b = &HashedData::from_value(b"another_value");
        let blob_c = &HashedData::from_value(b"third_value");

        // Create initial data, clone, add more data, commit, then add post-commit data
        db_a.blob_set(blob_a).expect("Failed to set blob in A");

        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        db_b.blob_set(blob_b).expect("Failed to set blob in B");

        let commit_id: CommitId = Hash::hash_bytes(b"commit_1").into();
        db_b.commit(&repo, &commit_id).expect("Failed to commit DB B");

        db_b.blob_set(blob_c).expect("Failed to set blob in B after commit");

        drop(db_a);
        drop(db_b);

        // Checkout should have blob_a and blob_b but not blob_c (added after commit)
        let db_c = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB C");

        let retrieved_a = db_c.blob_get(&blob_a.hash).expect("Failed to get blob a from C");
        assert_eq!(retrieved_a.as_ref(), blob_a.value, "Should have blob from before clone");

        let retrieved_b = db_c.blob_get(&blob_b.hash).expect("Failed to get blob b from C");
        assert_eq!(retrieved_b.as_ref(), blob_b.value, "Should have blob from before commit");

        assert!(matches!(db_c.blob_get(&blob_c.hash), Err(PersistenceLayerError::KeyNotFound)),
                "Should not have blob added after commit");

        let commit_path = repo.commit_dir(&commit_id);
        drop(retrieved_a);
        drop(retrieved_b);
        drop(db_c);
        assert!(commit_path.exists(), "Committed DB should persist on disk");
    }

    #[test]
    fn test_implied_mutability() {
        let tempdir = TestableTmpdir::new();
        let (repo, db_a) = setup_test_db(&tempdir);

        let blob_a = &HashedData::from_value(b"some_value");
        let blob_b = &HashedData::from_value(b"another_value");
        let commit_id_1: CommitId = Hash::hash_bytes(b"commit_1").into();
        let commit_id_2: CommitId = Hash::hash_bytes(b"commit_2").into();

        // Create first commit
        db_a.blob_set(blob_a).expect("Failed to set blob in A");
        db_a.commit(&repo, &commit_id_1).expect("Failed to commit DB A");

        // Checkout first commit, mutate, and create second commit
        let db_c = PersistenceLayer::checkout(&repo, &commit_id_1)
            .expect("Failed to checkout commit into DB C");
        db_c.blob_set(blob_b).expect("Failed to set blob in C");
        db_c.commit(&repo, &commit_id_2).expect("Failed to commit DB C");
        db_c.blob_delete(&blob_a.hash).expect("Failed to delete blob a in C");

        drop(db_a);
        drop(db_c);

        // Verify commit 1 has only blob_a
        let db_check_1 = PersistenceLayer::checkout(&repo, &commit_id_1)
            .expect("Failed to checkout commit 1");
        let retrieved_a = db_check_1.blob_get(&blob_a.hash)
            .expect("Failed to get blob a from commit 1");
        assert_eq!(retrieved_a.as_ref(), blob_a.value, "Commit 1 should have blob_a");
        assert!(matches!(db_check_1.blob_get(&blob_b.hash), Err(PersistenceLayerError::KeyNotFound)),
                "Commit 1 should not have blob_b");

        // Verify commit 2 has both blobs
        let db_check_2 = PersistenceLayer::checkout(&repo, &commit_id_2)
            .expect("Failed to checkout commit 2");
        let retrieved_a = db_check_2.blob_get(&blob_a.hash)
            .expect("Failed to get blob a from commit 2");
        assert_eq!(retrieved_a.as_ref(), blob_a.value, "Commit 2 should have blob_a");
        let retrieved_b = db_check_2.blob_get(&blob_b.hash)
            .expect("Failed to get blob b from commit 2");
        assert_eq!(retrieved_b.as_ref(), blob_b.value, "Commit 2 should have blob_b");
    }

    #[test]
    fn test_duplicate_commit() {
        let tempdir = TestableTmpdir::new();
        let (repo, db_a) = setup_test_db(&tempdir);

        let blob_1 = &HashedData::from_value(b"some_value");
        let blob_2 = &HashedData::from_value(b"another_value");
        let commit_id: CommitId = Hash::hash_bytes(b"commit_1").into();

        // Create initial commit
        db_a.blob_set(blob_1).expect("Failed to set blob in A");
        db_a.commit(&repo, &commit_id).expect("Failed to commit DB A");

        // Add more data and commit again with same ID (should overwrite)
        db_a.blob_set(blob_2).expect("Failed to set blob 2 in A");
        assert!(db_a.commit(&repo, &commit_id).is_ok(),
                "Re-committing with same ID should succeed");

        drop(db_a);

        // Checkout should contain both blobs from the second commit
        let db_check = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB A");

        let retrieved_1 = db_check.blob_get(&blob_1.hash).expect("Failed to get blob 1");
        assert_eq!(retrieved_1.as_ref(), blob_1.value, "Should have first blob");

        let retrieved_2 = db_check.blob_get(&blob_2.hash).expect("Failed to get blob 2");
        assert_eq!(retrieved_2.as_ref(), blob_2.value, "Should have second blob");
    }

    #[test]
    fn test_data_basic_ops() {
        let tmpdir = TestableTmpdir::new();

        let test = |key: String, value: String, value2: String| {
            let (_repo, db) = setup_test_db(&tmpdir);

            // Test: initial key should not exist
            assert!(matches!(db.get(&key), Err(PersistenceLayerError::KeyNotFound)),
                    "Key should not exist initially");

            // Test: set and get value
            db.set(&key, &value).expect("Should be able to set a value");
            let retrieved = db.get(&key).expect("Should be able to get the value");
            assert_eq!(retrieved.as_ref(), value.as_bytes());

            // Test: update existing key
            db.set(&key, &value2).expect("Should be able to update value");
            let retrieved = db.get(&key).expect("Should be able to get updated value");
            assert_eq!(retrieved.as_ref(), value2.as_bytes(), "Should return updated value");

            // Test: different key should not exist
            let other_key = format!("other_{key}");
            assert!(matches!(db.get(&other_key), Err(PersistenceLayerError::KeyNotFound)),
                    "Different key should not exist");

            // Test: data CF shouldn't affect blob CF
            let blob1 = HashedData::from_value(value.as_bytes());
            let blob2 = HashedData::from_value(value2.as_bytes());
            assert!(matches!(db.blob_get(&blob1.hash), Err(PersistenceLayerError::KeyNotFound)),
                    "Data operations shouldn't affect blob CF");
            assert!(matches!(db.blob_get(&blob2.hash), Err(PersistenceLayerError::KeyNotFound)),
                    "Data operations shouldn't affect blob CF");

            // Test: delete key
            db.delete(&key).expect("Should be able to delete the value");
            assert!(matches!(db.get(&key), Err(PersistenceLayerError::KeyNotFound)),
                    "Deleted key should not be found");
        };

        proptest!(|(key in string_of_length(8), value in string_of_length(10), value2 in string_of_length(12))| {
            test(key, value, value2);
        });
    }
}
