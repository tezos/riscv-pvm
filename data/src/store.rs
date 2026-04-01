// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Content addressable 'blob' store trait and in-memory implementation.

pub mod fold;
pub mod unfold;

use std::collections::HashMap;
use std::ops::DerefMut;
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

/// Subtrait that adds the ability to label blobs in the store with a parameter type.
pub trait LabelledStore<Label>: BlobStore {
    /// Return the blob associated with a given label. Returns `Ok(None)` if the label does not
    /// exist.
    fn blob_get_labelled(
        &self,
        label: &Label,
    ) -> Result<Option<HashedData<impl AsRef<[u8]>>>, BlobStoreError>;

    /// Add a blob to the store with a label. If the blob is already in the store the new label is
    /// still added---the blob may have multiple labels.
    fn blob_set_labelled<Data: AsRef<[u8]>>(
        &self,
        label: &Label,
        blob: &HashedData<Data>,
    ) -> Result<(), BlobStoreError>;

    /// Return all the labels currently pointing at a given blob.
    fn get_labels(&self, key: Hash) -> Result<Vec<Label>, BlobStoreError>;

    /// Delete the label. This does not touch the blob, even if it no longer has any labels. To
    /// remove the blob itself use `blob_delete`, which should remove all labels attached to the
    /// blob as well as the blob itself.
    fn delete_label(&self, label: &Label) -> Result<(), BlobStoreError>;
}

/// Basic implementation of the `BlobStore` trait for use in tests and any other time we don't want
/// to persist data to disk.
pub struct InMemoryBlobStore<Label>(RwLock<InMemoryBlobStoreInner<Label>>);

struct InMemoryBlobStoreInner<Label> {
    blobs: HashMap<Hash, (Bytes, Vec<Label>)>,
    labels: HashMap<Label, Hash>,
}

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
impl<Label> InMemoryBlobStore<Label> {
    fn new() -> Self {
        Self(RwLock::new(InMemoryBlobStoreInner {
            blobs: HashMap::new(),
            labels: HashMap::new(),
        }))
    }
}

impl<Label: Eq + std::hash::Hash> BlobStore for InMemoryBlobStore<Label> {
    fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
        let inner = self.0.read().map_err(|_| InMemoryError::LockPoisoned)?;
        match inner.blobs.get(&key) {
            Some((blob, _labels)) => Ok(blob.clone()),
            None => Err(BlobStoreError::NotFound(key)),
        }
    }

    fn blob_set<Data: AsRef<[u8]>>(&self, blob: &HashedData<Data>) -> Result<(), BlobStoreError> {
        let mut inner = self.0.write().map_err(|_| InMemoryError::LockPoisoned)?;
        inner
            .blobs
            .insert(blob.hash(), (Bytes::copy_from_slice(blob.data()), vec![]));
        Ok(())
    }

    fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError> {
        let mut inner = self.0.write().map_err(|_| InMemoryError::LockPoisoned)?;

        // We have to do this to get independent references to the two fields of `inner`.
        let inner_ref = inner.deref_mut();
        let blobs = &mut inner_ref.blobs;
        let labels = &mut inner_ref.labels;

        if let Some((_blob, v)) = blobs.get(&key) {
            for label in v {
                labels.remove(label);
            }
        }

        blobs.remove(&key);

        Ok(())
    }
}

impl<Label: Eq + std::hash::Hash> LabelledStore<Label> for InMemoryBlobStore<Label> {
    fn blob_get_labelled(
        &self,
        _label: &Label,
    ) -> Result<Option<HashedData<impl AsRef<[u8]>>>, BlobStoreError> {
        // TODO this is a placeholder impl
        Ok(Some(HashedData::from_data(vec![])))
    }

    fn blob_set_labelled<Data: AsRef<[u8]>>(
        &self,
        _label: &Label,
        _blob: &HashedData<Data>,
    ) -> Result<(), BlobStoreError> {
        todo!()
    }

    fn get_labels(&self, _key: Hash) -> Result<Vec<Label>, BlobStoreError> {
        todo!()
    }

    fn delete_label(&self, _label: &Label) -> Result<(), BlobStoreError> {
        todo!()
    }
}
