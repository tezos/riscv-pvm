// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

#![cfg(feature = "rocksdb")]

//! Implementation of `BlobStore` that can persist to disk using RocksDB.

use std::path::Path;
use std::path::PathBuf;

use octez_riscv_data::hash;
use octez_riscv_data::store::BlobStore;
use octez_riscv_data::store::BlobStoreError;
use octez_riscv_durable_storage::errors::Error;
use octez_riscv_durable_storage::errors::InvalidArgumentError;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_durable_storage::repo::DirectoryManager;
use octez_riscv_durable_storage::storage::PersistentKeyValueStore;
use octez_riscv_durable_storage::storage::ReadableKeyValueStore;
use octez_riscv_durable_storage::storage::WriteableKeyValueStore;

use super::PersistentBlobStore;
use super::StorageError;

/// An on-disk blob store implementation that uses RocksDB. Basically a fairly simple wrapper
/// around the persistence layer implemented in the durable storage crate.
///
/// Does not behave exactly like `Store` in terms of persistence; when it is dropped, the
/// persistence layer deletes its main 'temporary' directory, so in order to make blobs persistent
/// beyond dropping this struct you must call the `persist` method.
pub struct RocksDBStore {
    inner: PersistenceLayer,
    persist_path: PathBuf,
    tmp_commit_path: PathBuf,
}

fn from_durable_storage_error(key: hash::Hash, e: Error) -> BlobStoreError {
    match e {
        Error::InvalidArgument(InvalidArgumentError::KeyNotFound) => BlobStoreError::NotFound(key),
        _ => BlobStoreError::Custom(Box::new(e)),
    }
}

impl BlobStore for RocksDBStore {
    fn blob_get(&self, key: hash::Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
        self.inner
            .blob_get(key)
            .map_err(|e| from_durable_storage_error(key, e))
    }

    fn blob_set<Data: AsRef<[u8]>>(
        &self,
        blob: &hash::HashedData<Data>,
    ) -> Result<(), BlobStoreError> {
        self.inner
            .blob_set(blob.hash(), blob.data())
            .map_err(|e| BlobStoreError::Custom(Box::new(e)))
    }

    fn blob_delete(&self, key: hash::Hash) -> Result<(), BlobStoreError> {
        self.inner
            .blob_delete(key)
            .map_err(|e| BlobStoreError::Custom(Box::new(e)))
    }
}

impl PersistentBlobStore for RocksDBStore {
    fn init_from_path(path: impl AsRef<Path>) -> Result<Self, StorageError>
    where
        Self: Sized,
    {
        let repo = DirectoryManager::new(path.as_ref())?;
        let tempdir = repo.temp_database_dir()?;

        let persist_path = PathBuf::from(path.as_ref()).join("committed");
        let tmp_commit_path = PathBuf::from(path.as_ref()).join("tmp");

        let inner = if persist_path.exists() {
            PersistenceLayer::checkout_from_path(persist_path.as_path(), tempdir)
        } else {
            PersistenceLayer::new(&repo)
        }?;

        Ok(Self {
            inner,
            persist_path,
            tmp_commit_path,
        })
    }

    // TODO (TZX-130): fix the trait to be more compatible with the persistence layer, removing the
    // need for this slightly hacky approach.
    fn persist(&self) -> Result<(), StorageError> {
        self.inner.commit_to_path(self.tmp_commit_path.as_path())?;

        if self.persist_path.as_path().exists() {
            std::fs::remove_dir_all(self.persist_path.as_path())?;
        }
        std::fs::rename(&self.tmp_commit_path, &self.persist_path)?;

        Ok(())
    }
}
