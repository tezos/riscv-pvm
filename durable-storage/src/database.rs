// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Combined Database interface
//!
//! This module provides a database type to unify operations between the Merkle layer and the
//! key-value store.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::Arc;

use bytes::BufMut;
use bytes::Bytes;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use tokio::runtime::Handle;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::merkle_worker::MerkleWorker;
pub use crate::repo::DirectoryManager;
use crate::storage::KeyValueStore;
use crate::storage::PersistentKeyValueStore;

/// An isolated key-space, independent from other [`Database`]s, on which database operations can
/// be performed, e.g. read, write, delete.
///
/// This structure unifies the key-value store and Merkle layer to allow for persistent storage
/// alongside a representation which can provide a root hash.
#[repr(transparent)]
pub struct Database<KV, M: Mode> {
    inner: M::Select<DatabaseTemplate<KV>>,
}

impl<KV> Database<KV, Normal> {
    /// Construct a new, empty database backed by `repo`.
    ///
    /// The returned database owns an isolated working state. Mutations are applied immediately to
    /// that working state and are not persisted as a named snapshot until [`Database::commit`] is
    /// called.
    pub fn try_new(handle: &Handle, repo: &KV::Repo) -> Result<Self, OperationalError>
    where
        KV: BackgroundKeyValueStore,
    {
        let persistent = KV::new(repo)?;
        let persistent = Arc::new(persistent);

        let merkle = MerkleWorker::new(handle, persistent.clone());

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// Restore a database from a previously committed snapshot.
    ///
    /// The checked-out database is isolated from the committed snapshot: subsequent mutations are
    /// applied to a working copy, not to the committed state on disk.
    pub fn checkout(handle: &Handle, repo: &KV::Repo, commit_id: CommitId) -> Result<Self, Error>
    where
        KV: BackgroundPersistentKeyValueStore<Repo = DirectoryManager>,
    {
        let persistent = KV::checkout(repo, &commit_id)?;
        let persistent = Arc::new(persistent);

        let merkle = MerkleWorker::checkout(handle, persistent.clone(), commit_id)?;

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// Create a cheap clone of the current working state.
    ///
    /// The clone shares existing state efficiently with the original database and diverges on
    /// subsequent mutation. Neither database persists its state to a repository commit until
    /// [`Database::commit`] is called.
    pub fn try_clone_with(&self, handle: &Handle, repo: &KV::Repo) -> Result<Self, OperationalError>
    where
        KV: BackgroundKeyValueStore,
    {
        let persistent = self.inner.persistent.try_clone(repo)?;
        let persistent = Arc::new(persistent);

        let merkle = self.inner.merkle.clone_with(handle, persistent.clone())?;

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// Commit the current database state to the repository and return its commit identifier.
    ///
    /// The returned [`CommitId`] is derived from the Merkle root hash of the current working
    /// state. The commit can later be restored with [`Database::checkout`].
    pub fn commit(
        &self,
        repo: &DirectoryManager,
    ) -> Result<crate::commit::CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        let commit_id = self.inner.merkle.commit()?;
        self.inner.persistent.commit(repo, &commit_id)?;

        Ok(commit_id)
    }
}

impl<KV: BackgroundKeyValueStore, M: DatabaseMode> Database<KV, M> {
    /// Remove a key from the database.
    ///
    /// Deleting a missing key succeeds and leaves the database unchanged.
    /// TODO RV-943: Fix behaviour to returning an operational error when deleting a missing key.
    pub fn delete(&mut self, key: Key) -> Result<(), Error> {
        M::delete(self, key)
    }

    /// Returns true if the provided key exists in the database, false if it does not.
    pub fn exists(&self, key: &Key) -> Result<bool, Error> {
        M::exists(self, key)
    }

    /// Obtain, and possibly calculate, the root hash of the database.
    pub fn hash(&self) -> Result<Hash, OperationalError> {
        M::hash(self)
    }

    /// Read a portion of the value associated with the provided key. The read data will be written
    /// into `output`. `offset` specifies from where in the associated value to start reading.
    ///
    /// Returns the number of bytes read.
    ///
    /// Fails if:
    ///  - The key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub fn read(&self, key: &Key, offset: usize, output: impl BufMut) -> Result<usize, Error> {
        M::read(self, key, offset, output)
    }

    /// Inserts the value associated with the provided key, replacing any data already associated
    /// with the key.
    pub fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error> {
        M::set(self, key, data)
    }

    /// Modify the value associated with the provided key. `offset` specifies from where to start
    /// writing within the associated value, appending if it is equal to the length. Non-existent
    /// keys have the implicit length 0, so they are writeable.
    ///
    /// Non-zero offsets require checking the existence and length of an existing value, making
    /// them more expensive.
    ///
    /// Fails if:
    ///  - The offset is non-zero and the key does not exist.
    ///  - The offset is larger than the length of the associated value.
    ///  - The offset plus the length of the data would overflow.
    pub fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error> {
        M::write(self, key, offset, data)
    }

    /// Retrieve the length of the value associated with the provided key.
    ///
    /// Fails if:
    ///  - The key does not exist in the database.
    pub fn value_length(&self, key: &Key) -> Result<usize, Error> {
        M::value_length(self, key)
    }
}

impl<KV: KeyValueStore> Foldable<HashFold> for Database<KV, Normal> {
    fn fold(&self, _builder: HashFold) -> Hash {
        self.inner.merkle.hash().expect("Hashing should not fail")
    }
}

/// Modal template for the [`Database`]
///
/// This is used to select the appropriate implementation for the mode.
struct DatabaseTemplate<KV>(PhantomData<KV>, Infallible);

impl<KV> Modal for DatabaseTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = Infallible;

    type Verify = Infallible;
}

/// Modes that support the operational API exposed by [`Database`].
pub trait DatabaseMode: Mode {
    /// See [`Database::exists`]
    fn exists<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<bool, Error>;

    /// See [`Database::value_length`]
    fn value_length<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<usize, Error>;

    /// See [`Database::read`]
    fn read<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
        offset: usize,
        buffer: impl BufMut,
    ) -> Result<usize, Error>;

    /// See [`Database::set`]
    fn set<KV: BackgroundKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        value: Bytes,
    ) -> Result<(), Error>;

    /// See [`Database::write`]
    fn write<KV: BackgroundKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        offset: usize,
        value: Bytes,
    ) -> Result<usize, Error>;

    /// See [`Database::delete`]
    fn delete<KV: BackgroundKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
    ) -> Result<(), Error>;

    /// See [`Database::hash`]
    fn hash<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
    ) -> Result<Hash, OperationalError>;
}

impl DatabaseMode for Normal {
    fn exists<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<bool, Error> {
        match this.inner.persistent.get(key.as_ref()) {
            Ok(_) => Ok(true),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => Ok(false),
            Err(other_error) => Err(other_error),
        }
    }

    fn value_length<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<usize, Error> {
        let value = this.inner.persistent.get(key.as_ref())?;
        Ok(value.as_ref().len())
    }

    fn read<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
        offset: usize,
        mut output: impl BufMut,
    ) -> Result<usize, Error> {
        let value = this.inner.persistent.get(key.as_ref())?;
        let value_ref = value.as_ref();

        if offset > value_ref.len() {
            return Err(InvalidArgumentError::OffsetTooLarge)?;
        }

        let end = offset
            .saturating_add(output.remaining_mut())
            .min(value_ref.len());

        let source_slice = &value_ref[offset..end];
        output.put_slice(source_slice);
        Ok(source_slice.len())
    }

    fn set<KV: BackgroundKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        data: Bytes,
    ) -> Result<(), Error> {
        this.inner.persistent.set(&key, &data)?;
        this.inner.merkle.set(key, data)?;
        Ok(())
    }

    fn write<KV: BackgroundKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        offset: usize,
        data: Bytes,
    ) -> Result<usize, Error> {
        let written = data.len();
        this.inner.persistent.write(&key, offset, &data)?;
        this.inner.merkle.write(key, offset, data)?;
        Ok(written)
    }

    fn delete<KV: BackgroundKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
    ) -> Result<(), Error> {
        this.inner.persistent.delete(key.as_ref())?;
        this.inner.merkle.delete(key)?;
        Ok(())
    }

    fn hash<KV: BackgroundKeyValueStore>(
        this: &Database<KV, Self>,
    ) -> Result<Hash, OperationalError> {
        this.inner.merkle.hash()
    }
}

/// Registry implementation for the [`Database`] mode
struct NormalImpl<KV> {
    persistent: Arc<KV>,
    merkle: MerkleWorker<KV>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use tokio::runtime::Handle;

    use super::Database;
    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;
    use crate::storage::TestKeyValueStore;
    use crate::storage::TestRepo;
    use crate::storage::setup_repo;

    fn new_database(handle: &Handle, repo: TestRepo) -> Database<TestKeyValueStore, Normal> {
        Database::try_new(handle, &repo).expect("Creating a test database should succeed")
    }

    #[cfg(feature = "rocksdb")]
    type PersistentDatabase = Database<crate::persistence_layer::PersistenceLayer, Normal>;

    #[cfg(feature = "rocksdb")]
    fn new_persistent_database() -> (
        tokio::runtime::Runtime,
        octez_riscv_test_utils::TestableTmpdir,
        TestRepo,
        PersistentDatabase,
    ) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (keepalive, repo) = setup_repo();
        let database =
            Database::try_new(handle, &repo).expect("Creating a test database should succeed");

        (runtime, keepalive, repo, database)
    }

    #[cfg(feature = "rocksdb")]
    fn insert_entries(
        database: &mut PersistentDatabase,
        entries: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> std::collections::HashMap<Key, Bytes> {
        let mut expected = std::collections::HashMap::new();
        for (key, value) in entries {
            let key = Key::new(&key).expect("Size less than KEY_MAX_SIZE");
            let value = Bytes::copy_from_slice(&value);
            database
                .set(key.clone(), value.clone())
                .expect("Writing should succeed");
            expected.insert(key, value);
        }

        expected
    }

    #[cfg(feature = "rocksdb")]
    fn assert_database_value(database: &PersistentDatabase, key: &Key, expected: &[u8]) {
        let mut stored = vec![0; expected.len()];
        let read = database
            .read(key, 0, stored.as_mut_slice())
            .expect("Persisted value should exist");
        assert_eq!(read, stored.len());
        assert_eq!(stored.as_slice(), expected);
    }

    #[cfg(feature = "rocksdb")]
    fn assert_database_missing(database: &PersistentDatabase, key: &Key) {
        use crate::errors::Error;
        use crate::errors::InvalidArgumentError;

        assert!(matches!(
            database.read(key, 0, Vec::new()),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
    }

    #[cfg(feature = "rocksdb")]
    proptest! {
        #[test]
        fn test_database_commit_and_checkout(
            entries in prop::collection::vec(
                (prop::collection::vec(any::<u8>(), 1..=KEY_MAX_SIZE),
                 prop::collection::vec(any::<u8>(), 0..200)),
                1..50,
            ),
        ) {
            use crate::persistence_layer::PersistenceLayer;

            let (runtime, _keepalive, repo, mut database) = new_persistent_database();
            let handle = runtime.handle();
            let expected = insert_entries(&mut database, entries);

            let expected_hash = database.hash().expect("Hash should be calculated");
            let commit_id = database.commit(&repo).expect("Commit should succeed");

            let checked_out = Database::<PersistenceLayer, _>::checkout(handle, &repo, commit_id)
                .expect("Checkout should succeed");

            prop_assert_eq!(checked_out.hash().expect("Hash should be calculated"), expected_hash);

            for (key, value) in expected {
                assert_database_value(&checked_out, &key, value.as_ref());
            }
        }

        #[test]
        fn test_database_delete(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
        ) {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();

            let (_keepalive, repo) = setup_repo();
            let mut database = new_database(handle, repo);

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                database
                    .set(key.clone(), Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");
                prop_assert!(database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));

                let before = database.hash().expect("Hash should be calculated");
                database.delete(key.clone()).expect("Deleting should succeed");
                let after = database.hash().expect("Hash should be calculated");
                assert_ne!(before, after);
                prop_assert!(!database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));
            }
        }
    }

    #[test]
    fn test_database_delete_nonexistent() {
        // Receiving the hash requires a separate worker thread
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();

        let (_keepalive, repo) = setup_repo();
        let mut database = new_database(handle, repo);

        database
            .set(
                Key::new(&[1]).expect("Size less than KEY_MAX_SIZE"),
                Bytes::copy_from_slice(&[2, 3]),
            )
            .expect("Writing should succeed");

        let before = database.hash().expect("Hash should be calculated");

        // Delete a nonexistent key
        let nonexistent_key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        assert!(
            !database
                .exists(&nonexistent_key)
                .expect("There should be no other `PersistenceLayerError`s")
        );
        assert!(database.delete(nonexistent_key).is_ok());

        // Ensure the root hash is unchanged
        let after = database.hash().expect("Hash should be calculated");
        assert_eq!(before, after);
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn test_database_checkout_commit_creates_new_snapshot() {
        use crate::persistence_layer::PersistenceLayer;

        let (runtime, _keepalive, repo, mut original) = new_persistent_database();
        let handle = runtime.handle();

        let persisted_key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let derived_key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        original
            .set(persisted_key.clone(), Bytes::from_static(b"before"))
            .expect("Writing should succeed");

        let original_commit = original.commit(&repo).expect("Commit should succeed");

        let mut checked_out =
            Database::<PersistenceLayer, _>::checkout(handle, &repo, original_commit)
                .expect("Checkout should succeed");
        checked_out
            .set(persisted_key.clone(), Bytes::from_static(b"after"))
            .expect("Writing should succeed");
        checked_out
            .set(derived_key.clone(), Bytes::from_static(b"new"))
            .expect("Writing should succeed");

        let derived_commit = checked_out.commit(&repo).expect("Commit should succeed");
        assert_ne!(derived_commit, original_commit);

        let original_reloaded =
            Database::<PersistenceLayer, _>::checkout(handle, &repo, original_commit)
                .expect("Checkout should succeed");
        assert_database_value(&original_reloaded, &persisted_key, b"before");
        assert_database_missing(&original_reloaded, &derived_key);

        let derived_reloaded =
            Database::<PersistenceLayer, _>::checkout(handle, &repo, derived_commit)
                .expect("Checkout should succeed");
        assert_database_value(&derived_reloaded, &persisted_key, b"after");
        assert_database_value(&derived_reloaded, &derived_key, b"new");
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn test_database_checkout_missing_root_blob_fails_operationally() {
        use rocksdb::ColumnFamilyDescriptor;

        use crate::errors::Error;
        use crate::errors::OperationalError;
        use crate::persistence_layer::PersistenceLayer;

        let (runtime, _keepalive, repo, mut database) = new_persistent_database();
        let handle = runtime.handle();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key, Bytes::from_static(b"value"))
            .expect("Writing should succeed");

        let commit_id = database.commit(&repo).expect("Commit should succeed");
        let commit_path = repo.database_commit_dir(&commit_id);

        let commit_db = rocksdb::DB::open_cf_descriptors(
            &rocksdb::Options::default(),
            &commit_path,
            [
                ColumnFamilyDescriptor::new("blob", rocksdb::Options::default()),
                ColumnFamilyDescriptor::new("default", rocksdb::Options::default()),
            ],
        )
        .expect("Opening committed RocksDB should succeed");

        let blob_cf = commit_db
            .cf_handle("blob")
            .expect("Committed RocksDB should contain the blob column family");
        commit_db
            .delete_cf(blob_cf, commit_id.as_hash().as_ref())
            .expect("Deleting the root blob should succeed");
        drop(commit_db);

        assert!(matches!(
            Database::<PersistenceLayer, _>::checkout(handle, &repo, commit_id),
            Err(Error::Operational(OperationalError::CommitDataMissing { root }))
                if root == *commit_id.as_hash()
        ));
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn test_database_checkout_unknown_commit_fails() {
        use octez_riscv_data::hash::Hash;

        use crate::commit::CommitId;
        use crate::errors::Error;
        use crate::errors::OperationalError;
        use crate::persistence_layer::PersistenceLayer;

        let (runtime, _keepalive, repo, _database) = new_persistent_database();
        let handle = runtime.handle();

        let missing_commit = CommitId::from(Hash::hash_bytes(b"missing-commit"));

        assert!(matches!(
            Database::<PersistenceLayer, _>::checkout(handle, &repo, missing_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));
    }

    proptest! {
        #[test]
        fn test_database_exists(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
                                data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100), ) {

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let (_keepalive, repo) = setup_repo();
            let mut database = new_database(handle, repo);

            let mut seen = HashSet::new();

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data: &[u8] = data;

                prop_assert_ne!(database.exists(&key)
                        .expect("There should be no other `PersistenceLayerError`s"),
                    seen.insert(key.clone()));

                database
                    .set(key.clone(), Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");
                prop_assert!(database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));
            }
        }

        #[test]
        fn test_database_hash(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
                              data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100), ) {

            // Needs a thread for sending and a thread for receiving
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let (_keepalive, repo) = setup_repo();
            let mut database = new_database(handle, repo);

            let mut seen = HashSet::new();

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data: &[u8] = data;

                let before = database.hash().expect("Hash should be calculated");

                database
                    .set(key.clone(), Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");

                let after = database.hash().expect("Hash should be calculated");

                let existing_pair = !seen.insert((key, data));
                // Avoid the edge case of an identical hash from a previously seen identical
                // key-value pair, where no other keys were written to in between.
                if !existing_pair {
                    prop_assert_ne!(before, after);
                }
            }
        }
    }

    #[test]
    fn test_database_hash_revert() {
        // Needs a thread for sending and a thread for receiving
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = setup_repo();
        let mut database = new_database(handle, repo);

        let key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        let original_data = [1, 2, 3];
        let mutated_data = [3, 2, 1];

        database
            .set(key.clone(), Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");

        let before = database.hash().expect("Hash should be calculated");

        // Mutate the same key
        database
            .set(key.clone(), Bytes::copy_from_slice(&mutated_data))
            .expect("Writing should succeed");

        let after = database.hash().expect("Hash should be calculated");
        assert_ne!(before, after);

        // Revert the value of the same key to the original value and check the hash reverts to the
        // same value.
        database
            .set(key.clone(), Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");
        let reverted = database.hash().expect("Hash should be calculated");
        assert_eq!(before, reverted);
    }

    proptest! {
        #[test]
        fn test_database_read(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
                              data in prop::collection::vec(prop::collection::vec(any::<u8>(), 3..100), 0..100), ) {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let (_keepalive, repo) = setup_repo();
            let mut database = new_database(handle, repo);

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let mut read_data: [u8; 100] = [42; 100];

                let read_data_before = read_data;

                // Set the data
                database
                    .set(key.clone(), Bytes::copy_from_slice(data))
                    .expect("Setting should succeed");

                // The offset is bigger than the value
                prop_assert!(database.read(&key, data.len() + 1, read_data.as_mut_slice()).is_err());
                prop_assert_eq!(read_data, read_data_before);

                // Partial value write, where the output parameter is smaller than the data.
                prop_assert_eq!(
                    database
                        .read(&key, 0, read_data[1..data.len()].as_mut())
                        .expect(
                            "Reading a value larger than the output parameter's size should succeed"
                        ),
                    data.len() - 1
                );
                prop_assert_eq!(read_data[0], read_data_before[0]);
                prop_assert_eq!(&read_data[1..data.len()], &data[..data.len() - 1]);
                prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);
                let read_data_before = read_data;

                let read = database
                    .read(&key, data.len(), read_data.as_mut_slice())
                    .expect("A zero-sized write should succeed");
                prop_assert_eq!(read, 0);
                prop_assert_eq!(read_data, read_data_before);

                // Whole value write
                let read = database
                    .read(&key, 0, read_data.as_mut_slice())
                    .expect("Writing the whole value should succeed");
                prop_assert_eq!(read, data.len());
                prop_assert_eq!(&read_data[..data.len()], data.as_slice());
                prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);

                // Partial value write
                prop_assert_eq!(&read_data[2..data.len()], &data[2..]);
                let read = database
                    .read(&key, data.len() - 1, read_data[1..2].as_mut())
                    .expect("A partial write should succeed");
                prop_assert_eq!(read, 1);
                prop_assert_eq!(&read_data[1..2], &data[data.len() - 1..]);
                prop_assert_eq!(&read_data[2..data.len()], &data[2..]);
                prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);

                // Write limited by buffer
                let mut small_buffer: [u8; 3] = [0, 0 ,0];
                let read = database
                    .read(&key, 0, small_buffer.as_mut_slice())
                    .expect("Writing into a smaller buffer should succeed");
                prop_assert_eq!(read, small_buffer.len());
                prop_assert_eq!(&small_buffer, &data[0..3]);
            }
        }
    }

    #[test]
    fn test_database_read_no_key() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = setup_repo();
        let database = new_database(handle, repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let mut read_data: [u8; 100] = [42; 100];
        let read_data_before = read_data;

        // The key doesn't exist
        assert!(database.read(&key, 0, read_data.as_mut_slice()).is_err());
        assert_eq!(read_data_before, read_data);
    }

    proptest! {
        #[test]
        fn test_database_value_length(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
                                      data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..100), 0..100), ) {

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let (_keepalive, repo) = setup_repo();
            let mut database = new_database(handle, repo);

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data = Bytes::copy_from_slice(data);

                database
                    .set(key.clone(), Bytes::copy_from_slice(&data))
                    .expect("Writing should succeed");
                prop_assert_eq!(
                    database
                        .value_length(&key)
                        .expect("Getting the value length should succeed"),
                    data.len()
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_database_write(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..10),
                               offsets in prop::collection::vec(0..10usize, 0..10),
                               initial_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..10),
                               patch in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..10), ) {

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let (_keepalive, repo) = setup_repo();
            let mut database = new_database(handle, repo);

            for (((key, offset), initial_data), patch) in keys.iter().zip(offsets.iter()).zip(initial_data.iter()).zip(patch.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");

                let initial_data = Bytes::copy_from_slice(initial_data);
                assert!(database.set(key.clone(), initial_data.clone()).is_ok());

                let patch = Bytes::copy_from_slice(patch);
                let expected_written = patch.len();
                let result = database.write(key.clone(), *offset, patch.clone());
                if *offset > initial_data.len() {
                    prop_assert!(result.is_err());
                } else {
                    prop_assert_eq!(result.unwrap(), expected_written);
                    let expected_length = std::cmp::max(initial_data.len(), offset + patch.len());
                    prop_assert_eq!(database.value_length(&key).unwrap(), expected_length);
                }
            }
        }
    }

    #[test]
    fn test_database_write_new_nonzero_offset() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = setup_repo();
        let mut database = new_database(handle, repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[]);

        assert!(database.write(key.clone(), 1, data).is_err());
    }

    #[test]
    fn test_database_write_no_truncation() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = setup_repo();
        let mut database = new_database(handle, repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("a long value");
        let data2 = Bytes::from("good");
        let data3 = Bytes::from("nother");

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(database.write(key.clone(), 2, data2).is_ok());
        let mut output = vec![0; data.len()];
        assert!(database.read(&key, 0, output.as_mut_slice()).is_ok());
        assert_eq!(output.as_slice(), "a good value".as_bytes());

        assert!(database.write(key.clone(), 0, data3).is_ok());
        let mut output = vec![0; data.len()];
        assert!(database.read(&key, 0, output.as_mut_slice()).is_ok());
        assert_eq!(output.as_slice(), "nother value".as_bytes());
    }

    #[test]
    fn test_database_write_offset_append() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = setup_repo();
        let mut database = new_database(handle, repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[1, 2, 3]);

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(
            database
                .write(key.clone(), data.len(), data.clone())
                .is_ok()
        );
        let mut output = vec![0; 2 * data.len()];
        assert!(database.read(&key, 0, output.as_mut_slice()).is_ok());
        assert_eq!(output.as_slice(), [1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_database_write_oversized_offset() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = setup_repo();
        let mut database = new_database(handle, repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[]);

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(database.write(key.clone(), data.len() + 1, data).is_err());
    }
}
