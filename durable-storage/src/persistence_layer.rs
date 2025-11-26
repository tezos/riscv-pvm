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

use rocksdb::checkpoint::Checkpoint;
use tempfile::TempDir;

use crate::repo::DirectoryManager;
use crate::repo::DirectoryManagerError;

/// Type alias for a 32-byte hash used for identifying key-value blobs & commits.
type Hash = [u8; 32];

/// [`CommitId`]'s are used to generate [`PersistenceLayer`] commits & to checkout specific commits
/// from a [`DirectoryManager`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitId(Hash);

impl From<blake3::Hash> for CommitId {
    fn from(hash: blake3::Hash) -> Self {
        Self(hash.into())
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
        let hash = blake3::hash(value).into();
        Self { hash, value }
    }
}

/// Errors encountered when interacting with the persistence layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Commit not found")]
    CommitNotFound,

    #[error("Key not found")]
    KeyNotFound,

    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("Directory manager error: {0}")]
    DirectoryManager(#[from] DirectoryManagerError),
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
    fn checkpoint_at(&self, path: &Path) -> Result<(), Error> {
        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        Ok(Checkpoint::new(&self.db_instance)?.create_checkpoint(path)?)
    }

    /// Creates a new `PersistenceLayer` instance within the given `repo`.
    pub fn new(repo: &DirectoryManager) -> Result<Self, Error> {
        let tempdir = repo.new_temporary_dir()?;
        let new_db_path = tempdir.path().join("checkpoint");

        // To avoid accidentally overwriting an existing database, `error_if_exists` is set.
        let options = rocksdb_creation_options();
        let mut db = rocksdb::DB::open(&options, &new_db_path)?;
        db.create_cf(BLOB_CF, &rocksdb::Options::default())?;

        Ok(Self {
            db_instance: ManuallyDrop::new(db),
            _tempdir: tempdir,
        })
    }

    fn clone_as_checkpoint(db: &rocksdb::DB, repo: &DirectoryManager) -> Result<Self, Error> {
        let tempdir = repo.new_temporary_dir()?;
        let checkpoint_path = tempdir.path().join("checkpoint");

        // Note that we want the checkpoint object to be dropped before opening the DB in order to
        // call its destroy method to avoid UB. This happens in this unit expression.
        Checkpoint::new(db)?.create_checkpoint(&checkpoint_path)?;

        let temp_db =
            rocksdb::DB::open_cf(&rocksdb::Options::default(), &checkpoint_path, [BLOB_CF])?;

        Ok(Self {
            db_instance: ManuallyDrop::new(temp_db),
            _tempdir: tempdir,
        })
    }

    /// Clones the current `PersistenceLayer` instance.
    ///
    /// Operations on the cloned instance will have no effect on the original instance.
    pub fn try_clone(&self, repo: &DirectoryManager) -> Result<Self, Error> {
        Self::clone_as_checkpoint(&self.db_instance, repo)
    }

    /// Checks out a specific commit in the repository from the given `repo`
    pub fn checkout(repo: &DirectoryManager, id: &CommitId) -> Result<Self, Error> {
        let db_path = repo.commit_dir(id);

        // We assume the commit is not found if the folder does not exist.
        if !Path::exists(&db_path) {
            return Err(Error::CommitNotFound);
        };

        let db = rocksdb::DB::open_cf(&rocksdb::Options::default(), &db_path, [BLOB_CF])?;

        Self::clone_as_checkpoint(&db, repo)
    }

    /// Commits the current state to the repository within the given `repo`
    pub fn commit(&self, repo: &DirectoryManager, id: &CommitId) -> Result<(), Error> {
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
    pub fn blob_get(&self, key: &Hash) -> Result<impl AsRef<[u8]>, Error> {
        self.db_instance
            .get_pinned_cf(self.blob_cf(), key.as_ref())?
            .ok_or(Error::KeyNotFound)
    }

    /// Sets a value for the given key.
    pub fn blob_set(&self, blob: &HashedData) -> Result<(), Error> {
        Ok(self
            .db_instance
            .put_cf(self.blob_cf(), blob.hash, blob.value)?)
    }

    /// Deletes a value associated with the given key.
    pub fn blob_delete(&self, key: &Hash) -> Result<(), Error> {
        Ok(self.db_instance.delete_cf(self.blob_cf(), key.as_ref())?)
    }
}

// Interface used by the Data layer which operates over raw key-value pairs.
impl PersistenceLayer {
    /// Retrieves a value associated with the given key.
    pub fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        self.db_instance
            .get_pinned(key.as_ref())?
            .ok_or(Error::KeyNotFound)
    }

    /// Sets a value for the given key.
    pub fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), Error> {
        Ok(self.db_instance.put(key.as_ref(), value.as_ref())?)
    }

    /// Deletes a value associated with the given key.
    pub fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), Error> {
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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use proptest::prelude::Strategy;
    use proptest::prelude::any;
    use proptest::proptest;
    use rocksdb::properties::ESTIMATE_NUM_KEYS;
    use tempfile::TempDir;

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

    fn create_and_clear_dir(path: &Path) {
        if path.exists() {
            std::fs::remove_dir_all(path).expect("Should be able to remove dir");
        }
        std::fs::create_dir_all(path).expect("Should be able to create dir");
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
            let db = PersistenceLayer::new(&repo)
                .expect("Should be able to create new persistence layer");

            let blob = HashedData::from_value(value_a.as_bytes());
            let key = blob.hash;

            // Initially the key should not be found
            assert!(matches!(db.blob_get(&key), Err(Error::KeyNotFound)));

            db.blob_set(&blob).expect("Should be able to set a value");

            {
                // Now the key should be found
                let retrieved = db.blob_get(&key).expect("Should be able to get the value");
                assert_eq!(retrieved.as_ref(), value_a.as_bytes());
            }

            let blob2 = HashedData::from_value(value_b.as_bytes());
            let key2 = blob2.hash;
            db.blob_set(&blob2)
                .expect("Should be able to set another value");

            {
                // Now the second key should be found
                let retrieved = db.blob_get(&key2).expect("Should be able to get the value");
                assert_eq!(retrieved.as_ref(), value_b.as_bytes());

                let retrieved = db
                    .blob_get(&key)
                    .expect("Should be able to get the first value");
                assert_eq!(retrieved.as_ref(), value_a.as_bytes());
            }

            assert_eq!(
                db.db_instance
                    .property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                Ok(Some("2".to_string()))
            );

            db.blob_delete(&key)
                .expect("Should be able to delete the value");
            assert!(matches!(db.blob_get(&key), Err(Error::KeyNotFound)));

            assert_eq!(
                db.db_instance
                    .property_value_cf(db.blob_cf(), ESTIMATE_NUM_KEYS),
                Ok(Some("1".to_string()))
            );

            {
                // These operations shouldn't affect the data column family
                let data_a = db.get(&blob.hash);
                let data_b = db.get(&blob2.hash);
                assert!(matches!(data_a, Err(Error::KeyNotFound)));
                assert!(matches!(data_b, Err(Error::KeyNotFound)));
            }

            db.blob_delete(&key2)
                .expect("Should be able to delete the second value");
            assert!(matches!(db.blob_get(&key2), Err(Error::KeyNotFound)));

            let nonexistent_blob = HashedData::from_value(b"non_existent");
            assert!(matches!(db.blob_delete(&nonexistent_blob.hash), Ok(())));

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
        let tempdir = TestableTmpdir::new();
        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        let initial_blob = &HashedData::from_value(b"initial_value");
        let another_blob = &HashedData::from_value(b"another_value");
        let third_blob = &HashedData::from_value(b"third_value");

        db_a.blob_set(initial_blob)
            .expect("Failed to set initial blob in A");
        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");

        db_b.blob_set(another_blob)
            .expect("Failed to set another blob in B");

        // get methods borrow the db so we have to drop the borrow to mutate the db in the next scope
        {
            let retrieved_a = db_a
                .blob_get(&initial_blob.hash)
                .expect("Failed to get initial blob from A");
            assert_eq!(retrieved_a.as_ref(), b"initial_value");
        }

        // Wrap in a scope so we can drop the db's later
        {
            let retrieved_b = db_b
                .blob_get(&initial_blob.hash)
                .expect("Failed to get initial blob from B");
            assert_eq!(retrieved_b.as_ref(), b"initial_value");

            db_a.blob_set(third_blob)
                .expect("Failed to set third blob in A");
            let retrieved_third_from_b = db_b.blob_get(&third_blob.hash);
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
        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.blob_set(blob).expect("Failed to set blob in A");

        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        let db_c = db_a.try_clone(&repo).expect("Failed to clone DB A to C");

        let checkpoint_path = checkpoint_db_path(&db_a);
        drop(db_a);
        assert!(!checkpoint_path.exists());

        {
            let retrieved_b = db_b
                .blob_get(&blob.hash)
                .expect("Failed to get blob from B");
            assert_eq!(retrieved_b.as_ref(), b"some_value");

            let retrieved_c = db_c
                .blob_get(&blob.hash)
                .expect("Failed to get blob from C");
            assert_eq!(retrieved_c.as_ref(), b"some_value");
        }

        let checkpoint_path = checkpoint_db_path(&db_b);
        drop(db_b);
        assert!(!checkpoint_path.exists());

        let checkpoint_path = checkpoint_db_path(&db_c);
        drop(db_c);
        assert!(!checkpoint_path.exists());
    }

    #[test]
    fn test_commit_and_checkout() {
        let tempdir = TestableTmpdir::new();
        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        let blob = &HashedData::from_value(b"some_value");
        db_a.blob_set(blob).expect("Failed to set blob in A");

        let commit_id: CommitId = blake3::hash(b"commit_1").into();
        db_a.commit(&repo, &commit_id)
            .expect("Failed to commit DB A");
        let path_a = checkpoint_db_path(&db_a);
        drop(db_a);
        eprintln!("Path A: {path_a:?}");
        assert!(!path_a.exists());

        let db_b = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB B");

        {
            let retrieved_b = db_b
                .blob_get(&blob.hash)
                .expect("Failed to get blob from B");
            assert_eq!(retrieved_b.as_ref(), blob.value);
            let retrieved_nonexistent = db_b.blob_get(&[0u8; 32]);
            assert!(matches!(retrieved_nonexistent, Err(Error::KeyNotFound)));
            let retrieved_nonexistent = db_b.get(&[1u8; 32]);
            assert!(matches!(retrieved_nonexistent, Err(Error::KeyNotFound)));
        }

        let path_b = repo.commit_dir(&commit_id);
        drop(db_b);
        assert!(path_b.exists(), "Checked out DB should persist on disk");
    }

    #[test]
    fn test_nonexistent_checkout() {
        let tempdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let commit_id: CommitId = blake3::hash(b"nonexistent_commit").into();
        let db_result = PersistenceLayer::checkout(&repo, &commit_id);
        assert!(matches!(db_result, Err(Error::CommitNotFound)));
    }

    #[test]
    fn test_clone_commit_and_checkout() {
        // A -> (mutate A) -> B (clone) -> (mutate B) -> B (commit: "commit_1") -> (mutate B)
        // D (checkout: "commit_1")

        let temp_dir_str = "/tmp/persistence_layer_test_clone_commit_checkout";
        let temp_dir = Path::new(temp_dir_str);
        create_and_clear_dir(temp_dir);
        let repo = DirectoryManager::new(temp_dir).expect("Failed to create directory manager");

        let blob_a = &HashedData::from_value(b"some_value");
        let blob_b = &HashedData::from_value(b"another_value");
        let blob_c = &HashedData::from_value(b"third_value");

        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.blob_set(blob_a).expect("Failed to set blob in A");

        let db_b = db_a.try_clone(&repo).expect("Failed to clone DB A to B");
        db_b.blob_set(blob_b).expect("Failed to set blob in B");

        let commit_id: CommitId = blake3::hash(b"commit_1").into();
        db_b.commit(&repo, &commit_id)
            .expect("Failed to commit DB B");

        db_b.blob_set(blob_c).expect("Failed to set blob in B");

        drop(db_a);
        drop(db_b);

        // We should observe blob a & b after checking out the commit, but not c.
        let db_c = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB C");
        {
            let retrieved_a = db_c
                .blob_get(&blob_a.hash)
                .expect("Failed to get blob a from C");
            assert_eq!(retrieved_a.as_ref(), blob_a.value);
            let retrieved_b = db_c
                .blob_get(&blob_b.hash)
                .expect("Failed to get blob b from C");
            assert_eq!(retrieved_b.as_ref(), blob_b.value);
            let retrieved_c = db_c.blob_get(&blob_c.hash);
            assert!(matches!(retrieved_c, Err(Error::KeyNotFound)));
        }

        let path_c = repo.commit_dir(&commit_id);
        drop(db_c);
        assert!(path_c.exists(), "Checked out DB should persist on disk");
    }

    #[test]
    fn test_implied_mutability() {
        // A -> (mutate A) -> commit A (commit: "commit_1")
        // C (load "commit_1") -> (mutate C) -> commit C (commit: "commit_2") -> (mutate C)
        // Check commit_1 && commit_2

        let tempdir = TestableTmpdir::new();
        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");

        let blob_a = &HashedData::from_value(b"some_value");
        let blob_b = &HashedData::from_value(b"another_value");
        let commit_id_1: CommitId = blake3::hash(b"commit_1").into();
        let commit_id_2: CommitId = blake3::hash(b"commit_2").into();
        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");
        db_a.blob_set(blob_a).expect("Failed to set blob in A");
        db_a.commit(&repo, &commit_id_1)
            .expect("Failed to commit DB A");

        let db_c = PersistenceLayer::checkout(&repo, &commit_id_1)
            .expect("Failed to checkout commit into DB C");
        db_c.blob_set(blob_b).expect("Failed to set blob in C");
        db_c.commit(&repo, &commit_id_2)
            .expect("Failed to commit DB C");
        db_c.blob_delete(&blob_a.hash)
            // db_c.blob_delete(&blob_a.hash)
            .expect("Failed to delete blob a in C");
        drop(db_a);
        drop(db_c);

        // check commit 1
        let db_check_1 =
            PersistenceLayer::checkout(&repo, &commit_id_1).expect("Failed to checkout commit 1");
        {
            let retrieved_a = db_check_1
                .blob_get(&blob_a.hash)
                .expect("Failed to get blob a from check 1");
            assert_eq!(retrieved_a.as_ref(), blob_a.value);
            let retrieved_b = db_check_1.blob_get(&blob_b.hash);
            assert!(matches!(retrieved_b, Err(Error::KeyNotFound)));
        }

        // check commit 2
        let db_check_2 =
            PersistenceLayer::checkout(&repo, &commit_id_2).expect("Failed to checkout commit 2");
        {
            let retrieved_a = db_check_2
                .blob_get(&blob_a.hash)
                .expect("Failed to get blob a from check 2");
            assert_eq!(retrieved_a.as_ref(), blob_a.value);
            let retrieved_b = db_check_2
                .blob_get(&blob_b.hash)
                .expect("Failed to get blob b from check 2");
            assert_eq!(retrieved_b.as_ref(), blob_b.value);
        }
    }

    #[test]
    fn test_duplicate_commit() {
        let tempdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tempdir.path()).expect("Failed to create directory manager");
        let db_a = PersistenceLayer::new(&repo).expect("Failed to create DB A");

        let blob = &HashedData::from_value(b"some_value");
        db_a.blob_set(blob).expect("Failed to set blob in A");

        let commit_id: CommitId = blake3::hash(b"commit_1").into();
        db_a.commit(&repo, &commit_id)
            .expect("Failed to commit DB A");

        let blob_2 = &HashedData::from_value(b"another_value");
        db_a.blob_set(blob_2).expect("Failed to set blob 2 in A");

        // Committing again with the same id should work
        let result = db_a.commit(&repo, &commit_id);
        assert!(result.is_ok());

        drop(db_a);

        // Loading after the second commit should contain both blobs
        let db_a = PersistenceLayer::checkout(&repo, &commit_id)
            .expect("Failed to checkout commit into DB A");

        let retrieved = db_a
            .blob_get(&blob.hash)
            .expect("Failed to get blob from A");
        assert_eq!(retrieved.as_ref(), blob.value);

        let retrieved_2 = db_a
            .blob_get(&blob_2.hash)
            .expect("Failed to get blob 2 from A");
        assert_eq!(retrieved_2.as_ref(), blob_2.value);
    }

    #[test]
    fn test_data_basic_ops() {
        let test = |key: String, value: String, value2: String| {
            let repo = DirectoryManager::new(std::path::Path::new("/tmp/test_data_basic_ops"))
                .expect("Failed to create directory manager");
            let db = PersistenceLayer::new(&repo)
                .expect("Should be able to create new persistence layer");

            // Initially the key should not be found
            assert!(matches!(db.get(&key), Err(Error::KeyNotFound)));

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
                assert!(matches!(db.get(&other_key), Err(Error::KeyNotFound)));
            }

            {
                // these operations shouldn't affect the default column family for hashed data
                let blob1 = HashedData::from_value(value.as_bytes());
                let blob2 = HashedData::from_value(value2.as_bytes());
                assert!(matches!(db.blob_get(&blob1.hash), Err(Error::KeyNotFound)));
                assert!(matches!(db.blob_get(&blob2.hash), Err(Error::KeyNotFound)));
            }

            db.delete(&key).expect("Should be able to delete the value");
            assert!(matches!(db.get(&key), Err(Error::KeyNotFound)));
        };

        proptest!(|(key in string_of_length(8), value in string_of_length(10), value2 in string_of_length(12))| {
            test(key, value, value2);
        });
    }
}
