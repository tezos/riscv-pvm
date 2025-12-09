// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Combined Database interface
//!
//! This module provides a database type to unify operations between the Merkle worker and the
//! persistence layer.

use std::sync::Arc;

use bytes::Bytes;
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
    #[expect(dead_code, reason = "Implemented in RV-827")]
    /// Remove a key from the database.
    pub(crate) fn delete(&mut self, _key: &Key) -> Result<(), DatabaseError> {
        // TODO: Implement database deletes in RV-827
        todo!()
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "Implemented in RV-827"))]
    /// Returns true if the provided key exists in the database, false if it does not.
    pub(crate) fn exists(&self, key: &Key) -> Result<bool, DatabaseError> {
        match self.persistent.get(key.as_ref()) {
            Ok(_) => Ok(true),
            Err(PersistenceLayerError::KeyNotFound) => Ok(false),
            Err(other_error) => Err(other_error.into()),
        }
    }

    /// Obtain, and possibly calculate, the root hash of the database>
    #[cfg_attr(not(test), expect(dead_code, reason = "Implemented in RV-827"))]
    pub(crate) fn hash(&self) -> blake3::Hash {
        self.merkle.hash()
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "Implemented in RV-827"))]
    /// Read a portion of the value associated with the provided key. The read data will be written
    /// into `data`. `offset` specifies from where in the associated value to start reading.
    ///
    /// Fails if:
    ///  - The key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub(crate) fn read(
        &self,
        key: &Key,
        offset: usize,
        data: &mut [u8],
    ) -> Result<usize, DatabaseError> {
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

    #[expect(dead_code, reason = "Used in RV-828 and integration of the registry")]
    /// Try to construct a new Database
    pub fn try_new(handle: &Handle, repo: &DirectoryManager) -> Result<Self, DatabaseError> {
        let persistent: Arc<PersistenceLayer> = PersistenceLayer::new(repo)?.into();
        let merkle = MerkleWorker::new(handle, persistent.clone())?;
        Ok(Self { persistent, merkle })
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "Implemented in RV-827"))]
    /// Modify the value associated with the provided key. `offset` specifies from where to start
    /// writing within the associated value, appending if it is equal to the length. Non-existent
    /// keys have the implicit length 0, so they are writeable.
    ///
    /// Fails if:
    ///  - The offset is non-zero and the key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub(crate) fn write(
        &mut self,
        key: Key,
        offset: usize,
        data: Bytes,
    ) -> Result<usize, DatabaseError> {
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

    #[cfg_attr(not(test), expect(dead_code, reason = "Implemented in RV-827"))]
    /// Retrieve the length of the value associated with the provided key.
    ///
    /// Fails if:
    ///  - The key does not exist in the database.
    pub(crate) fn value_length(&self, key: &Key) -> Result<usize, DatabaseError> {
        let value = self.persistent.get(key.as_ref())?;
        Ok(value.as_ref().len())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use bytes::Bytes;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use tokio::runtime::Handle;

    use super::Database;
    use crate::merkle_layer::KEY_MAX_SIZE;
    use crate::merkle_layer::Key;
    use crate::merkle_worker::MerkleWorker;
    use crate::persistence_layer::PersistenceLayer;
    use crate::persistence_layer::utils::TestableTmpdir;
    use crate::repo::DirectoryManager;

    fn new_database(handle: &Handle) -> Database {
        let tmpdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");

        let persistent: Arc<PersistenceLayer> = PersistenceLayer::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();

        let merkle = MerkleWorker::new(handle, persistent.clone())
            .expect("Creating a Merkle worker should succeed");

        Database { persistent, merkle }
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
                let expected_written = data.len();

                prop_assert_ne!(database.exists(&key)
                        .expect("There should be no other `PersistenceLayerError`s"),
                    seen.insert(key.clone()));

                let result = database
                    .write(key.clone(), 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");
                prop_assert!(database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));

                prop_assert_eq!(result, expected_written);
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
                let expected_written = data.len();

                let before = database.hash();

                let result = database
                    .write(key.clone(), 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");

                prop_assert_eq!(result, expected_written);

                let after = database.hash();

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
            .write(key.clone(), 0, Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");

        let before = database.hash();

        // Mutate the same key
        database
            .write(key.clone(), 0, Bytes::copy_from_slice(&mutated_data))
            .expect("Writing should succeed");

        let after = database.hash();
        assert_ne!(before, after);

        // Revert the value of the same key to the original value and check the hash reverts to the
        // same value.
        database
            .write(key.clone(), 0, Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");
        let reverted = database.hash();
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

                // Write the data
                prop_assert_eq!(
                    database
                        .write(key.clone(), 0, Bytes::copy_from_slice(data))
                        .expect("Writing should succeed"),
                    data.len()
                );

                // The offset is bigger than the value
                prop_assert!(database.read(&key, data.len() + 1, &mut read_data).is_err());
                prop_assert_eq!(read_data, read_data_before);

                // Partial value write, where the output parameter is smaller than the data.
                prop_assert_eq!(
                    database
                        .read(&key, 0, &mut read_data[1..data.len()])
                        .expect(
                            "Reading a value larger than the output parameter's size should succeed"
                        ),
                    data.len() - 1
                );
                prop_assert_ne!(read_data, read_data_before);
                prop_assert_ne!(read_data, data.as_slice());
                let read_data_before = read_data;

                database
                    .read(&key, data.len(), &mut read_data)
                    .expect("A zero-sized write should succeed");
                prop_assert_eq!(read_data, read_data_before);

                // Whole value write
                database
                    .read(&key, 0, &mut read_data)
                    .expect("Writing the whole value should succeed");
                prop_assert_eq!(&read_data[..data.len()], data.as_slice());
                prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);

                // Partial value write
                prop_assert_eq!(&read_data[2..data.len()], &data[2..]);
                database
                    .read(&key, data.len() - 1, &mut read_data[1..2])
                    .expect("A partial write should succeed");
                prop_assert_eq!(&read_data[1..2], &data[data.len() - 1..]);
                prop_assert_eq!(&read_data[2..data.len()], &data[2..]);
                prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);
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
        assert!(database.read(&key, 0, &mut read_data).is_err());
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

                prop_assert_eq!(
                    database
                        .write(key.clone(), 0, Bytes::copy_from_slice(&data))
                        .expect("Writing should succeed"),
                    data.len()
                );
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
        fn test_database_write_zero_offset(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
                                           data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100), ) {

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let mut database = new_database(handle);

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data = Bytes::copy_from_slice(data);
                let expected_written = data.len();
                let result = database
                    .write(key, 0, data)
                    .expect("Writing should succeed");

                prop_assert_eq!(result, expected_written);
            }
        }
    }
}
