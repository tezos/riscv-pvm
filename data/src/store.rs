// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Content addressable 'blob' store trait and in-memory implementation.

use std::collections::HashMap;
use std::sync::RwLock;

use bytes::Bytes;

use crate::hash::Hash;
use crate::hash::HashedData;

/// Content addressable store that can contain arbitrary blobs of data indexed by their hashes.
pub trait BlobStore {
    type Error: std::error::Error;

    /// Retrieve a blob from its hash. If a hash is not found that must be somehow represented with
    /// the `Error` return type.
    fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, Self::Error>;

    /// Store a blob under its hash; should be a no-op if it is already present.
    fn blob_set<Data: AsRef<[u8]>>(&self, blob: HashedData<Data>) -> Result<(), Self::Error>;

    /// Remove an item from the store; should be a no-op if it is already absent.
    fn blob_delete(&self, key: Hash) -> Result<(), Self::Error>;
}

/// Basic implementation of the `BlobStore` trait for use in tests and any other time we don't want
/// to persist data to disk.
pub struct InMemoryBlobStore(RwLock<HashMap<Hash, Bytes>>);

#[derive(Debug, thiserror::Error)]
pub enum InMemoryError {
    #[error("Blob not found in store: {0}")]
    NotFound(Hash),

    #[error("RwLock in blob store was poisoned.")]
    LockPoisoned,
}

#[expect(
    dead_code,
    reason = "Will be used in future PR, see TZX-105 and TZX-106"
)]
impl InMemoryBlobStore {
    fn new() -> Self {
        Self(RwLock::new(HashMap::new()))
    }
}

impl BlobStore for InMemoryBlobStore {
    type Error = InMemoryError;

    fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, Self::Error> {
        let store = self.0.read().map_err(|_| InMemoryError::LockPoisoned)?;
        match store.get(&key) {
            Some(blob) => Ok(blob.clone()),
            None => Err(InMemoryError::NotFound(key)),
        }
    }

    fn blob_set<Data: AsRef<[u8]>>(&self, blob: HashedData<Data>) -> Result<(), Self::Error> {
        let mut store = self.0.write().map_err(|_| InMemoryError::LockPoisoned)?;
        store.insert(blob.hash(), Bytes::copy_from_slice(blob.data()));
        Ok(())
    }

    fn blob_delete(&self, key: Hash) -> Result<(), Self::Error> {
        let mut store = self.0.write().map_err(|_| InMemoryError::LockPoisoned)?;
        store.remove(&key);
        Ok(())
    }
}
