// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Content addressable 'blob' store trait and in-memory implementation.

pub mod fold;
pub mod unfold;

use std::collections::HashMap;
use std::sync::RwLock;

use bytes::Bytes;

use crate::hash::Hash;
use crate::hash::HashedData;

/// Error type for errors that can occur in any `BlobStore` implementation, including a `Custom`
/// variant to allow different implementations to include their own specific errors.
#[derive(Debug, thiserror::Error)]
pub enum BlobStoreError {
    #[error("Blob not found in store: {0}")]
    NotFound(Hash),

    #[error("Store-specific error: {0}")]
    Custom(Box<dyn std::error::Error + Send + Sync>),
}

/// Content addressable store that can contain arbitrary blobs of data indexed by their hashes.
pub trait BlobStore {
    /// Retrieve a blob from its hash. Will return `BlobStoreError::NotFound` if the hash is not
    /// present in the store.
    fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError>;

    /// Store a blob under its hash; should be a no-op if it is already present.
    fn blob_set<Data: AsRef<[u8]>>(&self, blob: &HashedData<Data>) -> Result<(), BlobStoreError>;

    /// Remove an item from the store; should be a no-op if it is already absent.
    fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError>;
}

/// Basic implementation of the `BlobStore` trait for use in tests and any other time we don't want
/// to persist data to disk.
pub struct InMemoryBlobStore(RwLock<HashMap<Hash, Bytes>>);

/// Error type for errors that are specific to the `InMemoryBlobStore`.
#[derive(Debug, thiserror::Error)]
pub enum InMemoryError {
    #[error("RwLock in blob store was poisoned.")]
    LockPoisoned,
}

impl From<InMemoryError> for BlobStoreError {
    fn from(e: InMemoryError) -> Self {
        Self::Custom(Box::new(e))
    }
}

#[cfg(test)]
impl InMemoryBlobStore {
    fn new() -> Self {
        Self(RwLock::new(HashMap::new()))
    }
}

impl BlobStore for InMemoryBlobStore {
    fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
        let store = self.0.read().map_err(|_| InMemoryError::LockPoisoned)?;
        match store.get(&key) {
            Some(blob) => Ok(blob.clone()),
            None => Err(BlobStoreError::NotFound(key)),
        }
    }

    fn blob_set<Data: AsRef<[u8]>>(&self, blob: &HashedData<Data>) -> Result<(), BlobStoreError> {
        let mut store = self.0.write().map_err(|_| InMemoryError::LockPoisoned)?;
        store.insert(blob.hash(), Bytes::copy_from_slice(blob.data()));
        Ok(())
    }

    fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError> {
        let mut store = self.0.write().map_err(|_| InMemoryError::LockPoisoned)?;
        store.remove(&key);
        Ok(())
    }
}
