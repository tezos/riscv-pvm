// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Storage backends

pub mod in_memory;

use std::path::Path;
use std::sync::Arc;

use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use tempfile::TempDir;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::repo::DirectoryManager;

/// Types that implement this trait can be used as the underlying key-value store
pub trait KeyValueStore: Sized {
    /// Type of repository required to initialise a key value store.
    type Repo;

    /// Create a new instance of the key-value store.
    ///
    /// The backend may make use of the repo provided, for persistence - if required.
    fn new(repo: &Self::Repo) -> Result<Self, OperationalError>;

    /// Attempt to make a copy of the key-value store.
    fn try_clone(&self, repo: &Self::Repo) -> Result<Self, OperationalError>;

    /// Retrieves the data associated with the given blob key.
    fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error>;

    /// Register data under a blob key.
    fn blob_set(
        &self,
        key: impl AsRef<[u8]>,
        data: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError>;

    /// Deletes a value associated with the given blob key.
    fn blob_delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError>;

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
        /// Key-value store backend used when the `rocksdb` feature is enabled.
        pub(crate) type TestKeyValueStore = crate::persistence_layer::PersistenceLayer;

        /// Repository type required to initialise [`TestKeyValueStore`].
        pub(crate) type TestRepo = <TestKeyValueStore as KeyValueStore>::Repo;

        /// Create a test repository for [`TestKeyValueStore`].
        ///
        /// Returns `(keepalive, repo)`:
        /// - `keepalive` is a temporary directory handle that must stay in scope for the lifetime
        ///   of `repo`.
        /// - `repo` is the backend repository value to pass into
        ///   [`KeyValueStore::new`] / [`KeyValueStore::try_clone`].
        ///
        /// TODO RV-942: Refactor the function to avoid the need for `keepalive` return value.
        pub(crate) fn setup_repo() -> (octez_riscv_test_utils::TestableTmpdir, TestRepo) {
            use crate::repo::DirectoryManager;

            let tmpdir = octez_riscv_test_utils::TestableTmpdir::new();
            let dir_manager = DirectoryManager::new(tmpdir.path()).expect("creating manager should succeed.");

            (tmpdir, dir_manager)
        }
    } else {
        /// Test key-value store backend used when the `rocksdb` feature is disabled.
        pub(crate) type TestKeyValueStore = crate::storage::in_memory::InMemoryKeyValueStore;

        /// Repository type required to initialise [`TestKeyValueStore`].
        pub(crate) type TestRepo = <TestKeyValueStore as KeyValueStore>::Repo;

        /// Create a test repository for [`TestKeyValueStore`].
        ///
        /// Returns `((), repo)` for signature compatibility with the `rocksdb` branch.
        pub(crate) fn setup_repo() -> ((), TestRepo) {
            ((), in_memory::InMemoryRepo)
        }
    }
}

/// Options for storing MAVL values in a [`KeyValueStore`]
#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// Persist the key-value pairs from MAVL nodes
    node_data: bool,

    /// Persist nested items
    deep: bool,
}

impl StoreOptions {
    /// When this is set, recursive child values will not be persisted.
    pub fn with_shallow(self) -> Self {
        Self {
            node_data: self.node_data,
            deep: false,
        }
    }

    /// Also stores nested nodes and trees.
    pub fn with_deep(self) -> Self {
        Self {
            node_data: self.node_data,
            deep: true,
        }
    }

    /// Persists the key-value data of nodes.
    ///
    /// Turning this option on ensures the nodes are persisted completely. When using the
    /// [`crate::merkle_layer::MerkleLayer`] in isolation, this is necessary as there is no other
    /// component that will be writing the key-value data to the store.
    pub fn with_node_data(self) -> Self {
        Self {
            node_data: true,
            deep: self.deep,
        }
    }

    /// Do not persist key-value data of nodes.
    ///
    /// This lets you avoid writing key-value data to the store when another component does this
    /// already. This is, for example, the case in the [`crate::database::Database`] component which
    /// mutates the store directly ahead of commitments. At commitment time, you only need to
    /// persist the remaining tree and node structures.
    pub fn without_node_data(self) -> Self {
        Self {
            node_data: false,
            deep: self.deep,
        }
    }

    /// Returns whether node key-value data should be persisted.
    pub fn node_data(&self) -> bool {
        self.node_data
    }

    /// Returns whether nested values should be persisted recursively.
    pub fn deep(&self) -> bool {
        self.deep
    }
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            node_data: false,
            deep: true,
        }
    }
}

/// This trait marks values that can be persisted into a [`KeyValueStore`].
pub trait Storable: Foldable<HashFold> {
    /// Persist this value into `store` according to `options`.
    fn store(
        &self,
        store: &impl KeyValueStore,
        options: &StoreOptions,
    ) -> Result<(), OperationalError>;
}

impl<T: Storable> Storable for Arc<T> {
    fn store(
        &self,
        store: &impl KeyValueStore,
        options: &StoreOptions,
    ) -> Result<(), OperationalError> {
        T::store(self, store, options)
    }
}

/// This trait marks values that can be reconstructed from a [`KeyValueStore`] by content hash.
pub trait Loadable: Sized {
    /// Load a value identified by `id` from `store`.
    fn load(id: Hash, store: &impl KeyValueStore) -> Result<Self, OperationalError>;
}

impl<T: Loadable> Loadable for Arc<T> {
    fn load(id: Hash, store: &impl KeyValueStore) -> Result<Self, OperationalError> {
        T::load(id, store).map(Arc::new)
    }
}
