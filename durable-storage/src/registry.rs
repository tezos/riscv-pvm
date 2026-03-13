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
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::serialisation::deserialise;
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
use crate::storage::KeyValueStore;

pub(super) const REGISTRY_ARITY: usize = 2;

#[derive(Debug, Encode, Decode)]
/// Structure to store the result of serialising a registry.
struct RegistryManifest {
    database_hashes: Vec<CommitId>,
}

/// Registry that owns a set of databases backed by a directory manager.
#[repr(transparent)]
pub struct Registry<KV: KeyValueStore, M: Mode> {
    inner: M::Select<RegistryTemplate<KV>>,
}

impl<KV: BackgroundKeyValueStore> Registry<KV, Normal> {
    /// Creates a new, empty Registry.
    ///
    /// The registry owns a Tokio [`Runtime`] and a [`DirectoryManager`] rooted at
    /// `base_dir`.
    pub fn new(repo: KV::Repo) -> Result<Self, OperationalError> {
        let runtime = Self::build_runtime()?;

        Ok(Registry {
            inner: NormalImpl {
                repo,
                databases: Vec::new(),
                runtime,
            },
        })
    }

    fn build_runtime() -> Result<Arc<Runtime>, OperationalError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .map_err(|error| OperationalError::WorkerRuntimeCreationFailed { error })?;
        Ok(Arc::new(runtime))
    }
}

impl<KV: BackgroundPersistentKeyValueStore<Repo = DirectoryManager>> Registry<KV, Normal> {
    /// Restore a registry from a previously committed manifest.
    ///
    /// The restored databases are checked out from the database commits referenced by the
    /// manifest, then the reconstructed registry root is verified against the requested
    /// `commit_id`.
    pub fn checkout(repo: DirectoryManager, commit_id: CommitId) -> Result<Self, Error> {
        let manifest = Self::read_checkout_manifest(&repo, &commit_id)?;
        let runtime = Self::build_runtime()?;
        let databases = Self::checkout_databases(&runtime, &repo, &manifest.database_hashes)?;

        let registry = Registry {
            inner: NormalImpl {
                repo,
                databases,
                runtime,
            },
        };

        let actual_commit = CommitId::from(Hash::from_foldable(&registry));
        if actual_commit != commit_id {
            return Err(Error::Operational(OperationalError::RegistryCommitMismatch));
        }

        Ok(registry)
    }

    fn read_checkout_manifest(
        repo: &DirectoryManager,
        commit_id: &CommitId,
    ) -> Result<RegistryManifest, OperationalError> {
        let commit_path = repo.registry_commit_file(commit_id);
        let commit_bytes = std::fs::read(&commit_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                OperationalError::CommitNotFound
            } else {
                OperationalError::FileReadFailed { error }
            }
        })?;

        deserialise(&commit_bytes).map_err(OperationalError::from)
    }

    fn checkout_databases(
        runtime: &Arc<Runtime>,
        repo: &DirectoryManager,
        database_hashes: &[CommitId],
    ) -> Result<Vec<Database<KV, Normal>>, Error> {
        // TODO RV-946: Investigate parallelising the checkouts of individual databases.
        database_hashes
            .iter()
            .map(|&db_hash| Database::checkout(runtime.handle(), repo, db_hash))
            .collect()
    }

    /// Commit the registry state and return its commit ID.
    ///
    /// The registry state commit ID is computed as the Merkle root of the commit IDs
    /// of all underlying databases through the [`Foldable<HashFold>`] implementation,
    /// and the registry manifest is stored at the corresponding commit path.
    pub fn commit(&self) -> Result<CommitId, OperationalError> {
        let mut database_hashes = Vec::with_capacity(self.inner.databases.len());

        for database in &self.inner.databases {
            let hash = database.commit(&self.inner.repo)?;
            database_hashes.push(hash);
        }

        let registry_commit = CommitId::from(Hash::from_foldable(&self));

        let manifest = RegistryManifest { database_hashes };
        let encoded =
            serialise(&manifest).expect("Serialising the registry manifest should not fail");

        let commit_path = self.inner.repo.registry_commit_file(&registry_commit);
        std::fs::write(&commit_path, &encoded)
            .map_err(|error| OperationalError::FileWriteFailed { error })?;

        Ok(registry_commit)
    }
}

impl<KV: KeyValueStore, M: RegistryMode> Registry<KV, M> {
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
    /// Returns an error if the new size differs from the current size by more than 1. This
    /// function can be called in a loop.
    ///
    /// Growing the registry creates new databases, while shrinking drops
    /// databases from the end.
    pub fn resize_tick(&mut self, new_size: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
    {
        match self.len().abs_diff(new_size) {
            1 => (),
            0 => return Ok(()),
            _ => return Err(Error::from(InvalidArgumentError::RegistryResizeTooLarge)),
        }

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
    pub fn copy_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
    {
        M::copy_database(self, src_index, dst_index)
    }

    /// Move the contents of database at `src_index` to database at `dst_index`. The source
    /// database is replaced with an empty database.
    pub fn move_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
    {
        M::move_database(self, src_index, dst_index)
    }

    /// Clear the database at the given `index`.
    pub fn clear_database(&mut self, index: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
    {
        M::clear_database(self, index)
    }
}

impl<KV: BackgroundKeyValueStore, M: CloneRegistryMode> Registry<KV, M>
where
    KV::Repo: Clone,
{
    /// Try to clone the registry.
    ///
    /// This can fail for mode-specific reasons.
    pub fn try_clone(&self) -> Result<Self, OperationalError> {
        M::try_clone(self)
    }
}

impl<KV: KeyValueStore, F: Fold> Foldable<F> for Registry<KV, Normal>
where
    Database<KV, Normal>: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        let get_hash = |idx: usize| {
            self.database(idx)
                .expect("Getting a database at a valid index should succeed")
        };

        IndexableSeqAsTree::new(self.len(), REGISTRY_ARITY, &get_hash).fold(builder)
    }
}

/// Modal template for the [`Registry`]
///
/// This is used to select the appropriate implementation for the mode.
struct RegistryTemplate<KV: KeyValueStore>(PhantomData<KV>, Infallible);

impl<KV: KeyValueStore> Modal for RegistryTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = Infallible;

    type Verify = Infallible;
}

/// Modes that implement this support operations on [`Registry`]
pub trait RegistryMode: Mode {
    /// See [`Registry::len`]
    fn len<KV: KeyValueStore>(this: &Registry<KV, Self>) -> usize;

    /// See [`Registry::resize`]
    fn resize<KV: BackgroundKeyValueStore>(
        this: &mut Registry<KV, Self>,
        new_size: usize,
    ) -> Result<(), Error>;

    /// See [`Registry::database`]
    fn database<KV: KeyValueStore>(
        this: &Registry<KV, Self>,
        index: usize,
    ) -> Result<&Database<KV, Self>, Error>;

    /// See [`Registry::database_mut`]
    fn database_mut<KV: KeyValueStore>(
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
    fn len<KV: KeyValueStore>(this: &Registry<KV, Self>) -> usize {
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

    fn database<KV: KeyValueStore>(
        this: &Registry<KV, Self>,
        index: usize,
    ) -> Result<&Database<KV, Self>, Error> {
        let database = this
            .inner
            .databases
            .get(index)
            .ok_or(InvalidArgumentError::DatabaseIndexOutOfBounds)?;
        Ok(database)
    }

    fn database_mut<KV: KeyValueStore>(
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
    ) -> Result<Registry<KV, Self>, OperationalError>
    where
        KV::Repo: Clone;
}

impl CloneRegistryMode for Normal {
    fn try_clone<KV: BackgroundKeyValueStore>(
        this: &Registry<KV, Self>,
    ) -> Result<Registry<KV, Self>, OperationalError>
    where
        KV::Repo: Clone,
    {
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
struct NormalImpl<KV: KeyValueStore> {
    repo: KV::Repo,
    databases: Vec<Database<KV, Normal>>,
    runtime: Arc<Runtime>,
}

impl<KV: KeyValueStore> NormalImpl<KV> {
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
pub(super) mod tests {
    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;

    use super::Registry;
    use crate::errors::Error;
    use crate::errors::InvalidArgumentError;
    use crate::key::Key;
    use crate::storage::TestKeyValueStore;
    use crate::storage::TestRepo;
    use crate::storage::setup_repo;

    pub(super) fn setup_registry(repo: TestRepo) -> Registry<TestKeyValueStore, Normal> {
        Registry::new(repo).expect("Registry should be created")
    }

    pub(super) fn setup_size_2_registry(repo: TestRepo) -> Registry<TestKeyValueStore, Normal> {
        let mut registry = setup_registry(repo);
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        registry
            .resize_tick(2)
            .expect("Growing the registry should succeed.");

        registry
    }

    fn seed_copy_move(
        registry: &mut Registry<TestKeyValueStore, Normal>,
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
        registry: &Registry<TestKeyValueStore, Normal>,
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
        registry: &Registry<TestKeyValueStore, Normal>,
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

    pub(super) fn populate_database_with_key_value(
        registry: &mut Registry<TestKeyValueStore, Normal>,
        db_index: usize,
        key_bytes: &[u8],
        value: &[u8],
    ) {
        let key = Key::new(key_bytes).expect("Key should be valid");
        registry.inner.databases[db_index]
            .set(key, Bytes::copy_from_slice(value))
            .expect("Writing to database should succeed");
    }

    #[test]
    fn test_new() {
        let (_keepalive, repo) = setup_repo();
        let registry = setup_registry(repo);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_resize() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry(repo);

        while registry.len() < 4 {
            registry
                .resize_tick(registry.len() + 1)
                .expect("Growing the registry should succeed.");
        }
        assert_eq!(registry.len(), 4);

        while registry.len() > 1 {
            registry
                .resize_tick(registry.len() - 1)
                .expect("Shrinking the registry should succeed.");
        }
        assert_eq!(registry.len(), 1);

        assert!(registry.resize_tick(5).is_err());
    }

    #[test]
    fn test_get_database() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry(repo);

        while registry.len() < 3 {
            registry
                .resize_tick(registry.len() + 1)
                .expect("Growing the registry should succeed.");
        }

        for i in 0..3 {
            registry.database(i).expect("Database should exist.");
        }
    }

    #[test]
    fn test_copy_database() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo);

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
    fn test_database_operations_invalid_index() {
        macro_rules! assert_invalid_index_error {
            ($result:expr, $operation:expr, $direction:expr) => {
                assert!(
                    matches!(
                        $result,
                        Err(Error::InvalidArgument(
                            InvalidArgumentError::DatabaseIndexOutOfBounds
                        ))
                    ),
                    "{} {} invalid index should return DatabaseIndexOutOfBounds error",
                    $operation,
                    $direction,
                );
            };
        }

        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo);

        // Test copy operations with invalid indices
        assert_invalid_index_error!(registry.copy_database(0, 2), "copy", "to");

        assert_invalid_index_error!(registry.copy_database(2, 0), "copy", "from");

        // Test move operations with invalid indices
        assert_invalid_index_error!(registry.move_database(0, 2), "move", "to");

        assert_invalid_index_error!(registry.move_database(2, 0), "move", "from");

        // Test clear operation with invalid index
        assert_invalid_index_error!(registry.clear_database(2), "clear", "");
    }

    #[test]
    fn test_move_database() {
        // Test that the source database is emptied and the destination database
        // has all the data, and any data previously in the destination is lost.

        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo);

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
    fn test_database_operations_same_index() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo);

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, 0, 1);

        // Test copy with same index
        registry
            .copy_database(0, 0)
            .expect("Copying to same index should succeed");
        assert_pairs_present(&registry, 0, &src_pairs);

        // Test move with same index
        registry
            .move_database(0, 0)
            .expect("Moving to same index should succeed");
        assert_pairs_present(&registry, 0, &src_pairs);
    }

    #[test]
    fn test_clear_database() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo);

        let db_index = 0;
        populate_database_with_key_value(&mut registry, db_index, &[1], b"some_value");
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

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
}

#[cfg(feature = "rocksdb")]
#[cfg(test)]
mod rocksdb_tests {
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_data::serialisation::serialise;

    use super::Registry;
    use super::RegistryManifest;
    use super::tests::populate_database_with_key_value;
    use super::tests::setup_registry;
    use super::tests::setup_size_2_registry;
    use crate::commit::CommitId;
    use crate::errors::Error;
    use crate::errors::OperationalError;
    use crate::key::Key;
    use crate::storage::TestKeyValueStore;
    use crate::storage::setup_repo;

    impl Registry<TestKeyValueStore, Normal> {
        /// Assert that the manifest written for `commit_id` contains the expected database commit IDs.
        fn verify_manifest(
            &self,
            commit_id: &super::CommitId,
            expected_db_hashes: &[super::CommitId],
        ) {
            let commit_path = self.inner.repo.registry_commit_file(commit_id);
            let commit_bytes = std::fs::read(&commit_path).expect("Manifest should be written");
            let commit: RegistryManifest =
                deserialise(&commit_bytes).expect("Manifest should be deserialisable");
            assert_eq!(commit.database_hashes, expected_db_hashes);
        }
    }

    #[test]
    fn test_registry_commit_empty() {
        let (_keepalive, repo) = setup_repo();
        let registry = setup_registry(repo);

        let expected_db_hashes: Vec<CommitId> = Vec::new();
        let expected_root = CommitId::from(Hash::hash_bytes(&[]));

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        registry.verify_manifest(&root_commit, &expected_db_hashes);
    }

    #[test]
    fn test_registry_commit_size_1() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry(repo);
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value(&mut registry, 0, &[1], b"singleton");

        let expected_db_hashes: Vec<super::CommitId> = registry
            .inner
            .databases
            .iter()
            .map(|db| db.hash().unwrap().into())
            .collect();
        let expected_root = expected_db_hashes[0];

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        registry.verify_manifest(&root_commit, &expected_db_hashes);
    }

    #[test]
    fn test_committing_identical_registry_succeeds() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry(repo);
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value(&mut registry, 0, &[1], b"singleton");

        let first_commit = registry.commit().expect("First commit should succeed");
        let second_commit = registry.commit().expect("Second commit should succeed");

        assert_eq!(
            first_commit, second_commit,
            "Committing identical registry states should yield the same commit ID."
        );
    }

    #[test]
    fn test_registry_commit_writes_manifest() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo);

        populate_database_with_key_value(&mut registry, 0, &[1], b"alpha");
        populate_database_with_key_value(&mut registry, 1, &[2], b"beta");

        let expected_db_hashes: Vec<super::CommitId> = registry
            .inner
            .databases
            .iter()
            .map(|db| db.hash().unwrap().into())
            .collect();
        let expected_root = CommitId::from(Hash::combine_hashes(
            expected_db_hashes.iter().map(CommitId::as_hash),
        ));

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        registry.verify_manifest(&root_commit, &expected_db_hashes);
    }

    #[test]
    fn test_registry_checkout_roundtrip_empty() {
        let (_keepalive, repo) = setup_repo();
        let registry = setup_registry(repo.clone());

        let root_commit = registry.commit().expect("Commit should succeed");
        let checked_out = Registry::<TestKeyValueStore, Normal>::checkout(repo, root_commit)
            .expect("Checkout should succeed");

        assert!(checked_out.is_empty());
        assert_eq!(
            CommitId::from(Hash::from_foldable(&checked_out)),
            root_commit
        );
    }

    #[test]
    fn test_registry_checkout_roundtrip_populated() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_size_2_registry(repo.clone());

        populate_database_with_key_value(&mut registry, 0, &[1], b"alpha");
        populate_database_with_key_value(&mut registry, 1, &[2], b"beta");

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");

        let root_commit = registry.commit().expect("Commit should succeed");
        let checked_out = Registry::<TestKeyValueStore, Normal>::checkout(repo, root_commit)
            .expect("Checkout should succeed");

        assert_eq!(checked_out.len(), 2);
        checked_out.inner.databases[0].assert_database_value(&key_a, b"alpha");
        checked_out.inner.databases[1].assert_database_value(&key_b, b"beta");
        assert_eq!(
            CommitId::from(Hash::from_foldable(&checked_out)),
            root_commit
        );
    }

    #[test]
    fn test_registry_checkout_missing_commit_fails() {
        let (_keepalive, repo) = setup_repo();
        let missing_commit = CommitId::from(Hash::hash_bytes(b"missing-registry-commit"));

        assert!(matches!(
            Registry::<TestKeyValueStore, Normal>::checkout(repo, missing_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));
    }

    #[test]
    fn test_registry_checkout_missing_database_commit_fails() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry(repo.clone());
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value(&mut registry, 0, &[1], b"singleton");

        let root_commit = registry.commit().expect("Commit should succeed");
        let manifest_path = repo.registry_commit_file(&root_commit);
        let manifest_bytes = std::fs::read(&manifest_path).expect("Manifest should be readable");
        let manifest: RegistryManifest =
            deserialise(&manifest_bytes).expect("Manifest should be deserialisable");
        std::fs::remove_dir_all(repo.database_commit_dir(&manifest.database_hashes[0]))
            .expect("Database commit should be removable");

        assert!(matches!(
            Registry::<TestKeyValueStore, Normal>::checkout(repo, root_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));
    }

    #[test]
    fn test_registry_checkout_detects_manifest_root_mismatch() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry(repo.clone());
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value(&mut registry, 0, &[1], b"value");

        let root_commit = registry.commit().expect("Commit should succeed");
        let manifest_path = repo.registry_commit_file(&root_commit);
        let manifest_bytes = std::fs::read(&manifest_path).expect("Manifest should be readable");
        let manifest: RegistryManifest =
            deserialise(&manifest_bytes).expect("Manifest should be deserialisable");

        let fake_commit = CommitId::from(Hash::hash_bytes(b"fake-registry-commit"));
        let fake_manifest_path = repo.registry_commit_file(&fake_commit);
        let fake_manifest_bytes = serialise(&manifest).expect("Manifest should be serialisable");
        std::fs::write(&fake_manifest_path, fake_manifest_bytes)
            .expect("Writing the fake manifest should succeed");

        assert!(matches!(
            Registry::<TestKeyValueStore, Normal>::checkout(repo, fake_commit),
            Err(Error::Operational(OperationalError::RegistryCommitMismatch))
        ));
    }
}
