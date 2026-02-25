// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Registry of databases for the RISC-V PVM durable storage.
//!
//! This module provides the Registry struct, which is responsible for managing multiple
//! databases within the durable storage system.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::Arc;

use bincode::Decode;
use bincode::Encode;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::serialisation::serialise;
use tokio::runtime::Runtime;

use crate::commit::CommitId;
use crate::database::Database;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::repo::DirectoryManager;

pub(super) const REGISTRY_ARITY: usize = 2;

#[derive(Debug, Encode, Decode)]
/// Structure to store the result of serialising a registry.
struct RegistryManifest {
    database_hashes: Vec<CommitId>,
}

/// Registry that owns a set of databases backed by a directory manager.
#[repr(transparent)]
pub struct Registry<KV, M: Mode> {
    inner: M::Select<RegistryTemplate<KV>>,
}

impl<KV> Registry<KV, Normal> {
    /// Creates a new, empty Registry.
    ///
    /// The registry owns a Tokio [`Runtime`] and a [`DirectoryManager`] rooted at
    /// `base_dir`.
    pub fn new(repo: DirectoryManager) -> Result<Self, OperationalError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .map_err(|error| OperationalError::WorkerRuntimeCreationFailed { error })?;
        let runtime = Arc::new(runtime);

        Ok(Registry {
            inner: NormalImpl {
                repo,
                databases: Vec::new(),
                runtime,
            },
        })
    }
}

impl<KV: BackgroundPersistentKeyValueStore> Registry<KV, Normal> {
    /// Commit the registry state and return its commit ID.
    ///
    /// The registry state commit ID is computed as the Merkle root of the commit IDs
    /// of all underlying databases, and the registry manifest is stored at the
    /// corresponding commit path.
    pub fn commit(&self) -> Result<CommitId, OperationalError> {
        let mut database_hashes = Vec::with_capacity(self.inner.databases.len());

        for database in &self.inner.databases {
            let hash = database.commit(&self.inner.repo)?;
            database_hashes.push(hash);
        }
        let registry_commit = CommitId::compute_root_hash(&database_hashes);

        let manifest = RegistryManifest { database_hashes };
        let encoded =
            serialise(&manifest).expect("Serialising the registry manifest should not fail");

        let commit_path = self.inner.repo.registry_commit_file(&registry_commit);
        std::fs::write(&commit_path, &encoded)
            .map_err(|error| OperationalError::FileWriteFailed { error })?;

        Ok(registry_commit)
    }
}

impl<KV: BackgroundKeyValueStore, M: RegistryMode> Registry<KV, M> {
    /// Are there no databases in the registry?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of databases held in the registry.
    pub fn len(&self) -> usize {
        M::len(self)
    }

    /// Resize the registry to the given `new_size`.
    ///
    /// Growing the registry creates new databases, while shrinking drops
    /// databases from the end.
    pub fn resize(&mut self, new_size: usize) -> Result<(), Error> {
        M::resize(self, new_size)
    }

    /// Get a reference to the database at the given `index`.
    pub fn database(&self, index: usize) -> Result<&Database<KV, M>, Error> {
        M::database(self, index)
    }

    /// Get a mutable reference to the database at the given `index`.
    pub fn database_mut(&mut self, index: usize) -> Result<&mut Database<KV, M>, Error> {
        M::database_mut(self, index)
    }

    /// Copy the contents of database at `src_index` to database at `dst_index`.
    pub fn copy_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error> {
        M::copy_database(self, src_index, dst_index)
    }

    /// Move the contents of database at `src_index` to database at `dst_index`. The source
    /// database is replaced with an empty database.
    pub fn move_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error> {
        M::move_database(self, src_index, dst_index)
    }

    /// Clear the database at the given `index`.
    pub fn clear_database(&mut self, index: usize) -> Result<(), Error> {
        M::clear_database(self, index)
    }
}

impl<KV: BackgroundKeyValueStore, M: CloneRegistryMode> Registry<KV, M> {
    /// Try to clone the registry.
    ///
    /// This can fail for mode-specific reasons.
    pub fn try_clone(&self) -> Result<Self, OperationalError> {
        M::try_clone(self)
    }
}

/// Modal template for the [`Registry`]
///
/// This is used to select the appropriate implementation for the mode.
struct RegistryTemplate<KV>(PhantomData<KV>, Infallible);

impl<KV> Modal for RegistryTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = Infallible;

    type Verify = Infallible;
}

/// Modes that implement this support operations on [`Registry`]
pub trait RegistryMode: Mode {
    /// See [`Registry::len`]
    fn len<KV>(this: &Registry<KV, Self>) -> usize;

    /// See [`Registry::resize`]
    fn resize<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        new_size: usize,
    ) -> Result<(), Error>;

    /// See [`Registry::database`]
    fn database<KV>(this: &Registry<KV, Self>, index: usize) -> Result<&Database<KV, Self>, Error>;

    /// See [`Registry::database_mut`]
    fn database_mut<KV>(
        this: &mut Registry<KV, Self>,
        index: usize,
    ) -> Result<&mut Database<KV, Self>, Error>;

    /// See [`Registry::copy_database`]
    fn copy_database<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        src: usize,
        dst: usize,
    ) -> Result<(), Error>;

    /// See [`Registry::move_database`]
    fn move_database<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        src: usize,
        dst: usize,
    ) -> Result<(), Error>;

    /// See [`Registry::clear_database`]
    fn clear_database<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        index: usize,
    ) -> Result<(), Error>;
}

impl RegistryMode for Normal {
    fn len<KV>(this: &Registry<KV, Self>) -> usize {
        this.inner.databases.len()
    }

    fn resize<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        new_size: usize,
    ) -> Result<(), Error> {
        while this.len() < new_size {
            let database = Database::try_new(this.inner.runtime.handle(), &this.inner.repo)?;
            this.inner.databases.push(database);
        }

        this.inner.databases.truncate(new_size);

        Ok(())
    }

    fn database<KV>(this: &Registry<KV, Self>, index: usize) -> Result<&Database<KV, Self>, Error> {
        let database = this
            .inner
            .databases
            .get(index)
            .ok_or(InvalidArgumentError::DatabaseIndexOutOfBounds)?;
        Ok(database)
    }

    fn database_mut<KV>(
        this: &mut Registry<KV, Self>,
        index: usize,
    ) -> Result<&mut Database<KV, Self>, Error> {
        let database = this
            .inner
            .databases
            .get_mut(index)
            .ok_or(InvalidArgumentError::DatabaseIndexOutOfBounds)?;
        Ok(database)
    }

    fn copy_database<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), Error> {
        this.inner.validate_index(src_index)?;
        this.inner.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if copying to the same index.
            return Ok(());
        }

        let db_copy = this.inner.databases[src_index]
            .try_clone_with(this.inner.runtime.handle(), &this.inner.repo)?;
        this.inner.databases[dst_index] = db_copy;

        Ok(())
    }

    fn move_database<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), Error> {
        this.inner.validate_index(src_index)?;
        this.inner.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if moving to the same index.
            return Ok(());
        }

        let empty = Database::try_new(this.inner.runtime.handle(), &this.inner.repo)?;
        let db_to_move = std::mem::replace(&mut this.inner.databases[src_index], empty);
        this.inner.databases[dst_index] = db_to_move;

        Ok(())
    }

    fn clear_database<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        index: usize,
    ) -> Result<(), Error> {
        this.inner.validate_index(index)?;
        this.inner.databases[index] =
            Database::try_new(this.inner.runtime.handle(), &this.inner.repo)?;
        Ok(())
    }
}

/// Modes that implement this marker support cloning of the [`Registry`] type
pub trait CloneRegistryMode: Mode {
    /// See [`Registry::try_clone`]
    fn try_clone<KV: BackgroundKeyValueStore>(
        this: &Registry<KV, Self>,
    ) -> Result<Registry<KV, Self>, OperationalError>;
}

impl CloneRegistryMode for Normal {
    fn try_clone<KV: BackgroundKeyValueStore>(
        this: &Registry<KV, Self>,
    ) -> Result<Registry<KV, Self>, OperationalError> {
        let runtime = this.inner.runtime.clone();
        let repo = this.inner.repo.clone();

        let databases = this
            .inner
            .databases
            .iter()
            .map(|db| db.try_clone_with(runtime.handle(), &repo))
            .collect::<Result<_, _>>()?;

        Ok(Registry {
            inner: NormalImpl {
                repo,
                databases,
                runtime,
            },
        })
    }
}

/// Registry implementation for the [`Normal`] mode
struct NormalImpl<KV> {
    repo: DirectoryManager,
    databases: Vec<Database<KV, Normal>>,
    runtime: Arc<Runtime>,
}

impl<KV> NormalImpl<KV> {
    /// Check the given `index` is valid for a database in the registry.
    fn validate_index(&self, index: usize) -> Result<(), InvalidArgumentError> {
        if index >= self.databases.len() {
            Err(InvalidArgumentError::DatabaseIndexOutOfBounds)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::Registry;
    use crate::commit::CommitId;
    use crate::errors::Error;
    use crate::errors::InvalidArgumentError;
    use crate::key::Key;
    use crate::persistence_layer::PersistenceLayer;
    use crate::registry::RegistryManifest;
    use crate::repo::DirectoryManager;

    fn setup_registry() -> (TestableTmpdir, Registry<PersistenceLayer, Normal>) {
        let tmpdir = TestableTmpdir::new();
        let base_dir = tmpdir.path().join("registry");
        let repo = DirectoryManager::new(&base_dir).expect("creating manager should succeed.");
        let registry = Registry::new(repo).expect("Registry should be created");

        (tmpdir, registry)
    }

    fn setup_size_2_registry() -> (TestableTmpdir, Registry<PersistenceLayer, Normal>) {
        let (tmpdir, mut registry) = setup_registry();
        registry
            .resize(2)
            .expect("Growing the registry should succeed.");
        (tmpdir, registry)
    }

    fn seed_copy_move(
        registry: &mut Registry<PersistenceLayer, Normal>,
        src_index: usize,
        dst_index: usize,
    ) -> ([(Key, &'static [u8]); 2], Key) {
        // Before the copy/move, populate the source with key A and B, and the dest with key A and C.
        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        let key_c = Key::new(&[3]).expect("Size less than KEY_MAX_SIZE");

        let src_pairs = [
            (key_a.clone(), b"new_a".as_slice()),
            (key_b.clone(), b"new_b".as_slice()),
        ];

        for (key, value) in src_pairs.iter() {
            registry.inner.databases[src_index]
                .write(key.clone(), 0, Bytes::copy_from_slice(value))
                .expect("Writing to source should succeed");
        }

        // Write values to destination that should be overwritten.
        registry.inner.databases[dst_index]
            .write(key_a, 0, Bytes::copy_from_slice(b"old_a"))
            .expect("Writing to destination should succeed");
        registry.inner.databases[dst_index]
            .write(key_c.clone(), 0, Bytes::copy_from_slice(b"old_c"))
            .expect("Writing to destination should succeed");

        (src_pairs, key_c)
    }

    fn assert_pairs_present(
        registry: &Registry<PersistenceLayer, Normal>,
        db_index: usize,
        pairs: &[(Key, &'static [u8])],
    ) {
        for (key, value) in pairs.iter() {
            assert!(
                registry.inner.databases[db_index]
                    .exists(key)
                    .expect("Checking destination should succeed")
            );
            let mut buf = vec![0u8; value.len()];
            registry.inner.databases[db_index]
                .read(key, 0, buf.as_mut_slice())
                .expect("Reading from destination should succeed");
            assert_eq!(&buf, value);
        }
    }

    fn assert_pairs_absent(
        registry: &Registry<PersistenceLayer, Normal>,
        db_index: usize,
        pairs: &[(Key, &'static [u8])],
    ) {
        for (key, _value) in pairs.iter() {
            assert!(
                !registry.inner.databases[db_index]
                    .exists(key)
                    .expect("Checking source should succeed"),
                "Key should not exist in source after move."
            );
        }
    }

    #[test]
    fn test_new() {
        let (_tmpdir, registry) = setup_registry();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_resize() {
        let (_tmpdir, mut registry) = setup_registry();

        registry
            .resize(4)
            .expect("Growing the registry should succeed.");

        assert_eq!(registry.len(), 4);

        registry
            .resize(1)
            .expect("Shrinking the registry should succeed.");

        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_get_database() {
        let (_tmpdir, mut registry) = setup_registry();

        registry
            .resize(3)
            .expect("Growing the registry should succeed.");

        for i in 0..3 {
            registry.database(i).expect("Database should exist.");
        }
    }

    #[test]
    fn test_copy_database() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 0;
        let dst_index = 1;

        let (src_pairs, key_c) = seed_copy_move(&mut registry, src_index, dst_index);

        registry
            .copy_database(src_index, dst_index)
            .expect("Copying should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);

        assert!(
            !registry.inner.databases[dst_index]
                .exists(&key_c)
                .expect("Checking destination should succeed"),
            "Key C should not exist in destination after copy."
        );
    }

    #[test]
    fn test_copy_same_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 0;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, src_index, 1);

        registry
            .copy_database(src_index, dst_index)
            .expect("Copying should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);
    }

    #[test]
    fn test_copy_invalid_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let result = registry.copy_database(0, 2);
        assert!(
            matches!(
                result,
                Err(Error::InvalidArgument(
                    InvalidArgumentError::DatabaseIndexOutOfBounds
                ))
            ),
            "Copying to invalid index should return InvalidDatabaseIndex error."
        );

        let result = registry.copy_database(2, 0);
        assert!(
            matches!(
                result,
                Err(Error::InvalidArgument(
                    InvalidArgumentError::DatabaseIndexOutOfBounds
                ))
            ),
            "Copying from invalid index should return InvalidDatabaseIndex error."
        );
    }

    #[test]
    fn test_move_database() {
        // Test that the source database is emptied and the destination database
        // has all the data, and any data previously in the destination is lost.

        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 1;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, src_index, dst_index);

        registry
            .move_database(src_index, dst_index)
            .expect("Moving should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);
        assert_pairs_absent(&registry, src_index, &src_pairs);
    }

    #[test]
    fn test_move_invalid_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let result = registry.move_database(0, 2);
        assert!(
            matches!(
                result,
                Err(Error::InvalidArgument(
                    InvalidArgumentError::DatabaseIndexOutOfBounds
                ))
            ),
            "Moving to invalid index should return InvalidDatabaseIndex error."
        );

        let result = registry.move_database(2, 0);
        assert!(
            matches!(
                result,
                Err(Error::InvalidArgument(
                    InvalidArgumentError::DatabaseIndexOutOfBounds
                ))
            ),
            "Moving from invalid index should return InvalidDatabaseIndex error."
        );
    }

    #[test]
    fn test_move_same_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 0;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, src_index, 1);

        registry
            .move_database(src_index, dst_index)
            .expect("Moving should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);
    }

    #[test]
    fn test_clear_database() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let db_index = 0;
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry.inner.databases[db_index]
            .write(key.clone(), 0, Bytes::copy_from_slice(b"some_value"))
            .expect("Writing to database should succeed");

        registry
            .clear_database(db_index)
            .expect("Clearing the database should succeed");

        assert!(
            !registry.inner.databases[db_index]
                .exists(&key)
                .expect("Checking database should succeed"),
            "Key should not exist after clearing the database."
        );
    }

    #[test]
    fn test_registry_commit_empty() {
        let (_tmpdir, registry) = setup_registry();

        let expected_db_hashes: Vec<super::CommitId> = Vec::new();
        let expected_root = CommitId::compute_root_hash(&expected_db_hashes);

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        let commit_path = registry.inner.repo.registry_commit_file(&root_commit);
        let commit_bytes = std::fs::read(&commit_path).expect("Manifest should be written");
        let commit: RegistryManifest =
            deserialise(&commit_bytes).expect("Manifest should be deserialisable");
        assert_eq!(commit.database_hashes, expected_db_hashes);
    }

    #[test]
    fn test_registry_commit_size_1() {
        let (_tmpdir, mut registry) = setup_registry();
        registry
            .resize(1)
            .expect("Growing the registry should succeed.");

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry.inner.databases[0]
            .write(key.clone(), 0, Bytes::copy_from_slice(b"singleton"))
            .expect("Writing to database should succeed");

        let expected_db_hashes: Vec<super::CommitId> = registry
            .inner
            .databases
            .iter()
            .map(|db| db.hash().unwrap().into())
            .collect();
        let expected_root = CommitId::compute_root_hash(&expected_db_hashes);

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        let commit_path = registry.inner.repo.registry_commit_file(&root_commit);
        let commit_bytes = std::fs::read(&commit_path).expect("Manifest should be written");
        let commit: RegistryManifest =
            deserialise(&commit_bytes).expect("Manifest should be deserialisable");
        assert_eq!(commit.database_hashes, expected_db_hashes);
    }

    #[test]
    fn test_committing_identical_registry_succeeds() {
        let (_tmpdir, mut registry) = setup_registry();
        registry
            .resize(1)
            .expect("Growing the registry should succeed.");

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry.inner.databases[0]
            .write(key.clone(), 0, Bytes::copy_from_slice(b"singleton"))
            .expect("Writing to database should succeed");

        let first_commit = registry.commit().expect("First commit should succeed");
        let second_commit = registry.commit().expect("Second commit should succeed");

        assert_eq!(
            first_commit, second_commit,
            "Committing identical registry states should yield the same commit ID."
        );
    }

    #[test]
    fn test_registry_commit_writes_manifest() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        registry.inner.databases[0]
            .write(key_a.clone(), 0, Bytes::copy_from_slice(b"alpha"))
            .expect("Writing to database 0 should succeed");
        registry.inner.databases[1]
            .write(key_b.clone(), 0, Bytes::copy_from_slice(b"beta"))
            .expect("Writing to database 1 should succeed");

        let expected_db_hashes: Vec<super::CommitId> = registry
            .inner
            .databases
            .iter()
            .map(|db| db.hash().unwrap().into())
            .collect();
        let expected_root = CommitId::compute_root_hash(&expected_db_hashes);

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        let commit_path = registry.inner.repo.registry_commit_file(&root_commit);
        let commit_bytes = std::fs::read(&commit_path).expect("Manifest should be written");
        let commit: RegistryManifest =
            deserialise(&commit_bytes).expect("Manifest should be deserialisable");
        assert_eq!(commit.database_hashes, expected_db_hashes);
    }
}
