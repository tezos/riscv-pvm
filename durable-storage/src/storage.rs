// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Storage backends

pub mod in_memory;

use std::path::Path;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashedData;
use tempfile::TempDir;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::repo::DirectoryManager;

/// Types that implement this trait can be used as the underlying key-value store
pub trait KeyValueStore: Sized {
    /// Create a new instance of the key-value store.
    ///
    /// The backend will use the directories provided by the directory manager, if needed.
    fn new(repo: &DirectoryManager) -> Result<Self, OperationalError>;

    /// Attempt to make a copy of the key-value store.
    fn try_clone(&self, repo: &DirectoryManager) -> Result<Self, OperationalError>;

    /// Retrieves the data associated with the given hash.
    fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, Error>;

    /// Register some already-hashed data.
    fn blob_set<Data: AsRef<[u8]>>(&self, blob: HashedData<Data>) -> Result<(), OperationalError>;

    /// Deletes a value associated with the given hash.
    fn blob_delete(&self, key: Hash) -> Result<(), OperationalError>;

    /// Retrieves a value associated with the given key.
    fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error>;

    /// Sets a value for the given key.
    fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), OperationalError>;

    /// Writes a value for the given key with a given offset.
    fn write(
        &self,
        key: impl AsRef<[u8]>,
        offset: usize,
        value: impl AsRef<[u8]>,
    ) -> Result<(), Error>;

    /// Deletes a value associated with the given key.
    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError>;

    /// Checks if a value associated with the given key may exist. Returns false if the key
    /// definitely does not exist.
    ///
    /// # Performance
    ///
    /// `may_exist` may be cheaper to call than `get`, for the purposes of checking whether a
    /// value does not exist.
    fn may_exist(&self, key: impl AsRef<[u8]>) -> Result<bool, OperationalError>;
}

/// Types that implement this trait can be used as a persistent key-value store
pub trait PersistentKeyValueStore: KeyValueStore + Sized {
    /// Commits the current state of the store to the given path.
    ///
    /// Implementations will treat the path as a directory.
    fn commit_to_path(&self, path: &Path) -> Result<(), OperationalError>;

    /// Write the current key-value state to disk.
    fn commit(&self, repo: &DirectoryManager, id: &CommitId) -> Result<(), OperationalError> {
        let checkpoint_path = repo.database_commit_dir(id);

        // If the path already exists, we overwrite the existing commit. This is highly unlikely to
        // happen anyway if the commits are a hash of the content.
        if Path::exists(&checkpoint_path) {
            std::fs::remove_dir_all(&checkpoint_path).map_err(|error| {
                OperationalError::DirRemovalFailed {
                    path: checkpoint_path.clone(),
                    error,
                }
            })?;

            log::warn!("Overwriting existing commit: {}", id.hex_encode());
        }

        self.commit_to_path(&checkpoint_path)
    }

    /// Checkout the state from `source_path` but leave it untouched. Modifications to the
    /// resulting state will be reflected in `working_path`.
    fn checkout_from_path(
        source_path: &Path,
        working_path: TempDir,
    ) -> Result<Self, OperationalError>;

    /// Retrieve a key-value state from disk.
    fn checkout(repo: &DirectoryManager, id: &CommitId) -> Result<Self, OperationalError> {
        let commit_path = repo.database_commit_dir(id);
        let working_path = repo.temp_database_dir()?;

        Self::checkout_from_path(&commit_path, working_path)
    }
}

#[cfg(test)]
cfg_if::cfg_if! {
    if #[cfg(feature = "rocksdb")] {
        pub(crate) type TestKeyValueStore = crate::persistence_layer::PersistenceLayer;
    } else {
        pub(crate) type TestKeyValueStore = crate::storage::in_memory::InMemoryKeyValueStore;
    }
}
