// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Combined Database interface
//!
//! This module provides a database type to unify operations between the Merkle worker and the
//! persistence layer.

use std::sync::Arc;

use bytes::Bytes;

use crate::merkle_layer::Key;
use crate::merkle_worker::MerkleWorker;
use crate::merkle_worker::MerkleWorkerError;
use crate::persistence_layer::PersistenceLayer;
use crate::persistence_layer::PersistenceLayerError;

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
    #[expect(dead_code, reason = "Implemented in RV-827")]
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

    #[expect(dead_code, reason = "Implemented in RV-827")]
    /// Returns true if the provided key exists in the database, false if it does not.
    pub(crate) fn exists(&self, _key: &Key) -> bool {
        // TODO: Implement database reads in RV-827
        todo!()
    }

    #[expect(dead_code, reason = "Implemented in RV-827")]
    /// Obtain, and possibly calculate, the root hash of the database>
    pub(crate) fn hash(&self) -> blake3::Hash {
        // TODO: Implement database root-hashing in RV-827
        todo!()
    }

    #[expect(dead_code, reason = "Implemented in RV-827")]
    /// Read a portion of the value associated with the provided key. The read data will be written
    /// into `_data`.
    ///
    /// Fails if:
    ///  - The key does not exist.
    ///  - The offset is larger than or equal to the length of the associated value.
    pub(crate) fn read(
        &self,
        _key: &Key,
        _offset: usize,
        _data: &mut [u8],
    ) -> Result<usize, DatabaseError> {
        let value = None; // TODO: Implement database reads in RV-827
        //                       : Compare the offset and the size of the value in RV-827
        value.ok_or(DatabaseError::KeyNotFound)
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

    #[expect(dead_code, reason = "Implemented in RV-827")]
    /// Retrieve the length of the value associated with the provided key.
    ///
    /// Fails if:
    ///  - The key does not exist in the database.
    pub(crate) fn value_length(&self, _key: &Key) -> Result<usize, DatabaseError> {
        let value = None; // TODO: Implement database reads in RV-827
        //                       : Compare the offset and the size of the value in RV-827
        value.ok_or(DatabaseError::KeyNotFound)
    }
}

#[cfg(test)]
mod tests {
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
        fn test_database_write_zero_offset(keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
                                           data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100), ) {

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle();
            let mut database = new_database(handle);

            for (key, data) in keys.iter().zip(data.iter()) {
                let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
                let data: &[u8] = data;
                let expected_written = data.len();
                let result = database
                    .write(key, 0, Bytes::copy_from_slice(data))
                    .expect("Writing should succeed");

                prop_assert_eq!(result, expected_written);
            }
        }
    }
}
