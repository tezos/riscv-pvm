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
    /// Type of repository required to initialise a key value store.
    type Repo;

    /// Create a new instance of the key-value store.
    ///
    /// The backend may make use of the repo provided, for persistence - if required.
    fn new(repo: &Self::Repo) -> Result<Self, OperationalError>;

    /// Attempt to make a copy of the key-value store.
    fn try_clone(&self, repo: &Self::Repo) -> Result<Self, OperationalError>;

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
pub(crate) trait TestRepoTrait {
    /// The actual repository type expected by the KeyValueStore
    type Repo;

    /// Get a reference to the underlying repository
    fn as_repo(&self) -> &Self::Repo;

    /// Convert this test repo into the underlying repository type
    fn into_repo(self) -> Self::Repo;
}

#[cfg(test)]
cfg_if::cfg_if! {
    if #[cfg(feature = "rocksdb")] {
        use crate::persistence_layer::PersistenceLayer;
        use octez_riscv_test_utils::TestableTmpdir;

        /// Key-value store backend used when the `rocksdb` feature is enabled.
        pub(crate) type TestKeyValueStore = PersistenceLayer;

        /// Self-contained test repository that owns its temporary directory.
        /// This eliminates the need for callers to manage the temporary directory lifetime.
        pub(crate) struct TestDirectoryManager {
            _tmpdir: TestableTmpdir,
            /// The actual directory manager that interacts with the filesystem.
            manager: DirectoryManager,
        }

        impl TestDirectoryManager {
            fn new() -> Result<Self, crate::errors::OperationalError> {
                let tmpdir = TestableTmpdir::new();
                let manager = DirectoryManager::new(tmpdir.path())?;
                Ok(Self { _tmpdir: tmpdir, manager })
            }
        }

        impl TestRepoTrait for TestDirectoryManager {
            type Repo = DirectoryManager;

            fn as_repo(&self) -> &Self::Repo {
                &self.manager
            }

            fn into_repo(self) -> Self::Repo {
                self.manager
            }
        }

        /// Repository type required to initialise [`TestKeyValueStore`].
        pub(crate) type TestRepo = TestDirectoryManager;

        /// Create a test repository for [`TestKeyValueStore`].
        pub(crate) fn setup_repo() -> TestRepo {
            TestDirectoryManager::new().expect("creating test repository should succeed")
        }
    } else {
        use crate::storage::in_memory::InMemoryRepo;

        /// Test key-value store backend used when the `rocksdb` feature is disabled.
        pub(crate) type TestKeyValueStore = crate::storage::in_memory::InMemoryKeyValueStore;

        impl TestRepoTrait for InMemoryRepo {
            type Repo = InMemoryRepo;

            fn as_repo(&self) -> &Self::Repo {
                self
            }

            fn into_repo(self) -> Self::Repo {
                self
            }
        }

        /// Repository type required to initialise [`TestKeyValueStore`].
        pub(crate) type TestRepo = InMemoryRepo;

        /// Create a test repository for [`TestKeyValueStore`].
        pub(crate) fn setup_repo() -> TestRepo {
            in_memory::InMemoryRepo
        }
    }
}
