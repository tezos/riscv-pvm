// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Combined Database interface
//!
//! This module provides a database type to unify operations between the Merkle worker and the
//! persistence layer.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::BufMut;
use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use tokio::runtime::Handle;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::merkle_worker::MerkleWorker;
use crate::persistence_layer::PersistenceLayer;
pub use crate::repo::DirectoryManager;
use crate::storage::KeyValueStore;
use crate::storage::PersistentKeyValueStore;

/// An isolated key-space, independent from other [`Database`]s, on which database operations can
/// be performed, e.g. read, write, delete.
///
/// This structure unifies the [`PersistenceLayer`] and Merkle layer (via the [`MerkleWorker`]) to
/// allow for persistent storage alongside a representation which can provide a root hash.
#[repr(transparent)]
pub struct Database<M: Mode> {
    inner: M::Select<DatabaseTemplate>,
}

impl Database<Normal> {
    /// Try to construct a new Database
    pub fn try_new(handle: &Handle, repo: &DirectoryManager) -> Result<Self, OperationalError> {
        let persistent: Arc<PersistenceLayer> = PersistenceLayer::new(repo)?.into();
        let merkle = MerkleWorker::new(handle, persistent.clone());
        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// Try to create a cheap clone of the Database.
    pub fn try_clone_with(
        &self,
        handle: &Handle,
        repo: &DirectoryManager,
    ) -> Result<Self, OperationalError> {
        let persistent: Arc<PersistenceLayer> = self.inner.persistent.try_clone(repo)?.into();
        Ok(Database {
            inner: NormalImpl {
                persistent: persistent.clone(),
                merkle: self.inner.merkle.clone_with(handle, persistent)?,
            },
        })
    }

    /// Commit the current database state to the repository and return its root hash.
    pub(crate) fn commit(&self, repo: &DirectoryManager) -> Result<CommitId, OperationalError> {
        let commit_id = self.inner.merkle.commit()?;
        self.inner.persistent.commit(repo, &commit_id)?;
        Ok(commit_id)
    }
}

impl<M: DatabaseMode> Database<M> {
    /// Remove a key from the database.
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

/// Modal template for the [`Database`]
///
/// This is used to select the appropriate implementation for the mode.
enum DatabaseTemplate {}

impl Modal for DatabaseTemplate {
    type Normal = NormalImpl;

    type Prove<'normal> = Infallible;

    type Verify = Infallible;
}

/// Modes that implement this support operations on [`Database`]
pub trait DatabaseMode: Mode {
    /// See [`Database::exists`]
    fn exists(this: &Database<Self>, key: &Key) -> Result<bool, Error>;

    /// See [`Database::value_length`]
    fn value_length(this: &Database<Self>, key: &Key) -> Result<usize, Error>;

    /// See [`Database::read`]
    fn read(
        this: &Database<Self>,
        key: &Key,
        offset: usize,
        buffer: impl BufMut,
    ) -> Result<usize, Error>;

    /// See [`Database::set`]
    fn set(this: &mut Database<Self>, key: Key, value: Bytes) -> Result<(), Error>;

    /// See [`Database::write`]
    fn write(
        this: &mut Database<Self>,
        key: Key,
        offset: usize,
        value: Bytes,
    ) -> Result<usize, Error>;

    /// See [`Database::delete`]
    fn delete(this: &mut Database<Self>, key: Key) -> Result<(), Error>;

    /// See [`Database::hash`]
    fn hash(this: &Database<Self>) -> Result<Hash, OperationalError>;
}

impl DatabaseMode for Normal {
    fn exists(this: &Database<Self>, key: &Key) -> Result<bool, Error> {
        match this.inner.persistent.get(key.as_ref()) {
            Ok(_) => Ok(true),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => Ok(false),
            Err(other_error) => Err(other_error),
        }
    }

    fn value_length(this: &Database<Self>, key: &Key) -> Result<usize, Error> {
        let value = this.inner.persistent.get(key.as_ref())?;
        Ok(value.as_ref().len())
    }

    fn read(
        this: &Database<Self>,
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

    fn set(this: &mut Database<Self>, key: Key, data: Bytes) -> Result<(), Error> {
        this.inner.persistent.set(&key, &data)?;
        this.inner.merkle.set(key, data)?;
        Ok(())
    }

    fn write(
        this: &mut Database<Self>,
        key: Key,
        offset: usize,
        data: Bytes,
    ) -> Result<usize, Error> {
        // If the offset is greater than 0 and the key exists, we have to do an expensive 'get'
        // operation to check if the existing value length is shorter than the offset.
        if offset > 0 {
            // `may_exist` can be cheaper than `get`
            let may_exist = this.inner.persistent.may_exist(&key)?;
            if !may_exist {
                return Err(InvalidArgumentError::KeyNotFound)?;
            }

            // Checking the length of a value requires a full retrieval. Returns an error if the
            // key does not exist.
            let len = this.value_length(&key)?;
            if offset > len || offset.checked_add(data.len()).is_none() {
                return Err(InvalidArgumentError::OffsetTooLarge)?;
            }
        }

        let written = data.len();
        this.inner.persistent.write(&key, offset, &data)?;
        this.inner.merkle.write(key, offset, data)?;
        Ok(written)
    }

    fn delete(this: &mut Database<Self>, key: Key) -> Result<(), Error> {
        this.inner.persistent.delete(key.as_ref())?;
        this.inner.merkle.delete(key)?;
        Ok(())
    }

    fn hash(this: &Database<Self>) -> Result<Hash, OperationalError> {
        this.inner.merkle.hash()
    }
}

/// Registry implementation for the [`Database`] mode
struct NormalImpl {
    persistent: Arc<PersistenceLayer>,
    merkle: MerkleWorker,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use tokio::runtime::Handle;

    use super::Database;
    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;
    use crate::persistence_layer::PersistenceLayer;
    use crate::repo::DirectoryManager;
    use crate::storage::KeyValueStore;
    use crate::storage::PersistentKeyValueStore;

    fn new_database(handle: &Handle) -> Database<Normal> {
        let tmpdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");

        Database::try_new(handle, &repo).expect("Creating a test database should succeed")
    }

    proptest! {
        #[test]
        fn test_database_commit_persists_state(
            entries in prop::collection::vec(
                (prop::collection::vec(any::<u8>(), 1..=KEY_MAX_SIZE),
                 prop::collection::vec(any::<u8>(), 0..200)),
                1..50,
            ),
        ) {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let tmpdir = TestableTmpdir::new();
            let repo = DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
            let mut database = Database::try_new(handle, &repo).expect("Creating a test database should succeed");

            let mut expected = std::collections::HashMap::new();
            for (key, value) in entries {
                let key = Key::new(&key).expect("Size less than KEY_MAX_SIZE");
                let value = Bytes::copy_from_slice(&value);
                database
                    .write(key.clone(), 0, value.clone())
                    .expect("Writing should succeed");
                expected.insert(key, value);
            }

            let expected_hash = database.hash().expect("Hash should be calculated");
            let commit_id = database.commit(&repo).expect("Commit should succeed");

            prop_assert_eq!(&expected_hash, commit_id.as_hash());

            let committed =
                PersistenceLayer::checkout(&repo, &commit_id).expect("Checkout should succeed");
            for (key, value) in expected {
                let stored = committed
                    .get(key.as_ref())
                    .expect("Committed value should exist");
                prop_assert_eq!(stored.as_ref(), value.as_ref());
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
            let mut database = new_database(handle);

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
        let mut database = new_database(handle);

        // Populate a database and obtain a root hash
        let keys: Vec<Key> = (1..=5)
            .map(|k| Key::new(&[k]).expect("Size less than KEY_MAX_SIZE"))
            .collect();

        let data: Vec<[u8; 1]> = (1..=5).map(|i| [i * 42]).collect();

        for (key, data) in keys.iter().zip(data.iter()) {
            database
                .set(key.clone(), Bytes::copy_from_slice(data))
                .expect("Writing should succeed");
            assert!(
                database
                    .exists(key)
                    .expect("There should be no other `PersistenceLayerError`s")
            );
        }
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

    proptest! {
        #[test]
        fn test_database_exists(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
                                data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100), ) {

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let mut database = new_database(handle);

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
    }

    proptest! {
        #[test]
        fn test_database_hash(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
                              data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100), ) {

            // Needs a thread for sending and a thread for receiving
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let mut database = new_database(handle);

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
        let mut database = new_database(handle);

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
            let mut database = new_database(handle);

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
        let database = new_database(handle);

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
            let mut database = new_database(handle);

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
            let mut database = new_database(handle);

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
        let mut database = new_database(handle);

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
        let mut database = new_database(handle);

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
        let mut database = new_database(handle);

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
        let mut database = new_database(handle);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[]);

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(database.write(key.clone(), data.len() + 1, data).is_err());
    }
}
