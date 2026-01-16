// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Combined Database interface
//!
//! This module provides a database type to unify operations between the Merkle worker and the
//! persistence layer.

use std::sync::Arc;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use tokio::runtime::Handle;

use crate::merkle_layer::Key;
use crate::merkle_worker::MerkleWorker;
use crate::merkle_worker::MerkleWorkerError;
use crate::persistence_layer::PersistenceLayer;
use crate::persistence_layer::PersistenceLayerError;
use crate::repo::DirectoryManager;

/// An isolated key-space, independent from other [`Database`]s, on which database operations can
/// be performed, e.g. read, write, delete.
///
/// This structure unifies the [`PersistenceLayer`] and Merkle layer (via the [`MerkleWorker`]) to
/// allow for persistent storage alongside a representation which can provide a root hash.
pub struct Database {
    persistent: Arc<PersistenceLayer>,
    merkle: MerkleWorker,
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur during operations on a [`Database`].
pub enum DatabaseError {
    #[error("The offset is too large")]
    OffsetTooLarge,

    #[error("The provided key is expected to exist but does not")]
    KeyNotFound,

    #[error("Merkle worker error: {0}")]
    MerkleWorker(#[from] MerkleWorkerError),

    #[error("Persistence layer error: {0}")]
    PersistenceLayer(#[from] PersistenceLayerError),
}

impl Database {
    /// Remove a key from the database.
    pub fn delete(&mut self, key: Key) -> Result<(), DatabaseError> {
        self.persistent.delete(key.as_ref())?;
        self.merkle.delete(key);
        Ok(())
    }

    /// Returns true if the provided key exists in the database, false if it does not.
    pub fn exists(&self, key: &Key) -> Result<bool, DatabaseError> {
        match self.persistent.get(key.as_ref()) {
            Ok(_) => Ok(true),
            Err(PersistenceLayerError::KeyNotFound) => Ok(false),
            Err(other_error) => Err(other_error.into()),
        }
    }

    /// Obtain, and possibly calculate, the root hash of the database>
    pub fn hash(&self) -> Hash {
        self.merkle.hash()
    }

    /// Read a portion of the value associated with the provided key. The read data will be written
    /// into `data`. `offset` specifies from where in the associated value to start reading.
    ///
    /// Returns the number of bytes read.
    ///
    /// Fails if:
    ///  - The key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub fn read(&self, key: &Key, offset: usize, data: &mut [u8]) -> Result<usize, DatabaseError> {
        let value = self.persistent.get(key.as_ref())?;
        let value_ref = value.as_ref();

        if offset > value_ref.len() {
            return Err(DatabaseError::OffsetTooLarge);
        }

        let source_slice = &value_ref[offset..];
        let bytes_to_copy = std::cmp::min(data.len(), source_slice.len());

        data[..bytes_to_copy].copy_from_slice(&source_slice[..bytes_to_copy]);

        Ok(bytes_to_copy)
    }

    /// Try to construct a new Database
    pub fn try_new(handle: &Handle, repo: &DirectoryManager) -> Result<Self, DatabaseError> {
        let persistent: Arc<PersistenceLayer> = PersistenceLayer::new(repo)?.into();
        let merkle = MerkleWorker::new(handle, persistent.clone())?;
        Ok(Self { persistent, merkle })
    }

    /// Modify the value associated with the provided key. `offset` specifies from where to start
    /// writing within the associated value, appending if it is equal to the length. Non-existent
    /// keys have the implicit length 0, so they are writeable.
    ///
    /// Fails if:
    ///  - The offset is non-zero and the key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, DatabaseError> {
        if offset != 0 {
            let value = None;
            // TODO : Implement [`MerkleLayer::node::get_mut`] in RV-827
            value.ok_or(DatabaseError::KeyNotFound)?
        } else {
            let written = data.len();
            self.persistent.set(&key, &data)?;
            self.merkle.set(key, data);
            Ok(written)
        }
    }

    /// Try to create a cheap clone of the Database.
    pub fn try_clone_with(
        &self,
        handle: &Handle,
        repo: &DirectoryManager,
    ) -> Result<Self, DatabaseError> {
        let persistent: Arc<PersistenceLayer> = self.persistent.try_clone(repo)?.into();
        Ok(Self {
            persistent: persistent.clone(),
            merkle: self.merkle.clone_with(handle, persistent)?,
        })
    }

    /// Retrieve the length of the value associated with the provided key.
    ///
    /// Fails if:
    ///  - The key does not exist in the database.
    pub fn value_length(&self, key: &Key) -> Result<usize, DatabaseError> {
        let value = self.persistent.get(key.as_ref())?;
        Ok(value.as_ref().len())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Bytes;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use tokio::runtime::Handle;

    use super::Database;
    use crate::merkle_layer::KEY_MAX_SIZE;
    use crate::merkle_layer::Key;
    use crate::persistence_layer::utils::TestableTmpdir;
    use crate::repo::DirectoryManager;

    /// Helper to create a test database with a runtime handle.
    fn new_database(handle: &Handle) -> Database {
        let tmpdir = TestableTmpdir::new();
        let repo = DirectoryManager::new(tmpdir.path())
            .expect("Failed to create directory manager");
        Database::try_new(handle, &repo).expect("Creating a test database should succeed")
    }

    /// Helper to create a test runtime with specified worker threads.
    fn new_test_runtime(worker_threads: usize) -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .build()
            .expect("Creating a Tokio runtime should succeed")
    }

    proptest! {
        #[test]
        fn test_database_delete(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
        ) {
            let runtime = new_test_runtime(1);
            let mut database = new_database(runtime.handle());

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let bytes_written = database
                    .write(key.clone(), 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");

                prop_assert_eq!(bytes_written, data.len());
                prop_assert!(database.exists(&key).expect("Exists check should succeed"));

                let hash_before = database.hash();
                database.delete(key.clone()).expect("Deleting should succeed");
                let hash_after = database.hash();

                prop_assert_ne!(hash_before, hash_after, "Hash should change after deletion");
                prop_assert!(!database.exists(&key).expect("Exists check should succeed"));
            }
        }
    }

    #[test]
    fn test_database_delete_nonexistent() {
        let runtime = new_test_runtime(1);
        let mut database = new_database(runtime.handle());

        // Populate database with some data
        let keys: Vec<Key> = (1..=5)
            .map(|k| Key::new(&[k]).expect("Size less than KEY_MAX_SIZE"))
            .collect();
        let data: Vec<[u8; 1]> = (1..=5).map(|i| [i * 42]).collect();

        for (key, data) in keys.iter().zip(data.iter()) {
            let bytes_written = database
                .write(key.clone(), 0, Bytes::copy_from_slice(data))
                .expect("Writing should succeed");
            assert_eq!(bytes_written, data.len());
            assert!(database.exists(key).expect("Exists check should succeed"));
        }

        let hash_before = database.hash();

        // Delete a nonexistent key - should succeed but not change hash
        let nonexistent_key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        assert!(!database.exists(&nonexistent_key).expect("Exists check should succeed"),
                "Nonexistent key should not exist");
        assert!(database.delete(nonexistent_key).is_ok(),
                "Deleting nonexistent key should succeed");

        let hash_after = database.hash();
        assert_eq!(hash_before, hash_after, "Hash should be unchanged when deleting nonexistent key");
    }

    proptest! {
        #[test]
        fn test_database_exists(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
        ) {
            let runtime = new_test_runtime(1);
            let mut database = new_database(runtime.handle());
            let mut seen = HashSet::new();

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let is_duplicate = !seen.insert(key.clone());

                // Key existence should match whether we've seen it before
                prop_assert_eq!(
                    database.exists(&key).expect("Exists check should succeed"),
                    is_duplicate,
                    "Key existence should match whether it was already written"
                );

                let bytes_written = database
                    .write(key.clone(), 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");
                prop_assert_eq!(bytes_written, data.len());
                prop_assert!(database.exists(&key).expect("Exists check should succeed"));
            }
        }
    }

    proptest! {
        #[test]
        fn test_database_hash(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
        ) {
            let runtime = new_test_runtime(2);
            let mut database = new_database(runtime.handle());
            let mut seen = HashSet::new();

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let hash_before = database.hash();

                let bytes_written = database
                    .write(key.clone(), 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");
                prop_assert_eq!(bytes_written, data.len());

                let hash_after = database.hash();

                // Hash should change unless we're writing the same key-value pair again
                let is_duplicate = !seen.insert((key, data));
                if !is_duplicate {
                    prop_assert_ne!(hash_before, hash_after, "Hash should change for new data");
                }
            }
        }
    }

    #[test]
    fn test_database_hash_revert() {
        let runtime = new_test_runtime(2);
        let mut database = new_database(runtime.handle());

        let key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        let original_data = [1, 2, 3];
        let mutated_data = [3, 2, 1];

        // Write initial data
        database
            .write(key.clone(), 0, Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");
        let hash_original = database.hash();

        // Mutate the key - hash should change
        database
            .write(key.clone(), 0, Bytes::copy_from_slice(&mutated_data))
            .expect("Writing should succeed");
        let hash_mutated = database.hash();
        assert_ne!(hash_original, hash_mutated, "Hash should change when data changes");

        // Revert to original data - hash should match original
        database
            .write(key.clone(), 0, Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");
        let hash_reverted = database.hash();
        assert_eq!(hash_original, hash_reverted,
                   "Hash should revert when data is reverted to original value");
    }

    proptest! {
        #[test]
        fn test_database_read(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 3..100), 0..100),
        ) {
            let runtime = new_test_runtime(1);
            let mut database = new_database(runtime.handle());

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let mut buffer: [u8; 100] = [42; 100];
                let initial_buffer = buffer;

                // Write the data
                let bytes_written = database
                    .write(key.clone(), 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");
                prop_assert_eq!(bytes_written, data.len());

                // Test: reading past end should fail
                prop_assert!(database.read(&key, data.len() + 1, &mut buffer).is_err(),
                            "Reading past end should fail");
                prop_assert_eq!(buffer, initial_buffer, "Buffer should be unchanged on error");

                // Test: partial read when buffer is smaller than data
                let bytes_read = database
                    .read(&key, 0, &mut buffer[1..data.len()])
                    .expect("Partial read should succeed");
                prop_assert_eq!(bytes_read, data.len() - 1);

                // Test: zero-sized read at end
                let previous_buffer = buffer;
                database.read(&key, data.len(), &mut buffer)
                    .expect("Zero-sized read should succeed");
                prop_assert_eq!(buffer, previous_buffer, "Zero-sized read shouldn't modify buffer");

                // Test: full value read
                database.read(&key, 0, &mut buffer)
                    .expect("Full read should succeed");
                prop_assert_eq!(&buffer[..data.len()], data.as_slice());

                // Test: partial read with offset
                database.read(&key, data.len() - 1, &mut buffer[1..2])
                    .expect("Partial read with offset should succeed");
                prop_assert_eq!(&buffer[1..2], &data[data.len() - 1..]);
            }
        }
    }

    #[test]
    fn test_database_read_no_key() {
        let runtime = new_test_runtime(1);
        let database = new_database(runtime.handle());

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let mut buffer: [u8; 100] = [42; 100];
        let initial_buffer = buffer;

        assert!(database.read(&key, 0, &mut buffer).is_err(),
                "Reading nonexistent key should fail");
        assert_eq!(buffer, initial_buffer, "Buffer should be unchanged when read fails");
    }

    proptest! {
        #[test]
        fn test_database_value_length(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..100), 0..100),
        ) {
            let runtime = new_test_runtime(1);
            let mut database = new_database(runtime.handle());

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data = Bytes::copy_from_slice(data);

                let bytes_written = database
                    .write(key.clone(), 0, data.clone())
                    .expect("Writing should succeed");
                prop_assert_eq!(bytes_written, data.len());

                let length = database
                    .value_length(&key)
                    .expect("Getting value length should succeed");
                prop_assert_eq!(length, data.len(), "Value length should match written data");
            }
        }
    }

    proptest! {
        #[test]
        fn test_database_write_zero_offset(
            keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
            data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
        ) {
            let runtime = new_test_runtime(1);
            let mut database = new_database(runtime.handle());

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data = Bytes::copy_from_slice(data);

                let bytes_written = database
                    .write(key, 0, data.clone())
                    .expect("Writing should succeed");
                prop_assert_eq!(bytes_written, data.len(), "Should write all bytes");
            }
        }
    }
}
