// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
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
use octez_riscv_data::components::vector::Vector;
use octez_riscv_data::components::vector::VectorMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::ProvableExt;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
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
use crate::repo::RegistryRepo;
use crate::storage::KeyValueStore;

#[derive(Debug, Encode, Decode)]
/// Structure to store the result of serialising a registry.
struct RegistryManifest {
    database_hashes: Vec<CommitId>,
}

/// Registry that owns a set of databases and the repository used to manage
/// registry state.
pub struct Registry<KV: KeyValueStore, M: Mode> {
    inner: M::Select<RegistryTemplate<KV>>,
    databases: Vector<Database<KV, M>, M>,
}

impl<KV: BackgroundKeyValueStore> Registry<KV, Normal> {
    /// Creates a new, empty Registry.
    ///
    /// The registry owns a Tokio [`Runtime`] and a register state repository.
    pub fn new(repo: KV::Repo) -> Result<Self, OperationalError> {
        let runtime = Self::build_runtime()?;

        Ok(Registry {
            inner: NormalImpl { repo, runtime },
            databases: Vector::new(Vec::new()),
        })
    }

    fn build_runtime() -> Result<Arc<Runtime>, OperationalError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .map_err(|error| OperationalError::WorkerRuntimeCreationFailed { error })?;
        Ok(Arc::new(runtime))
    }

    /// Get a [`Handle`] to the registry's runtime.
    #[cfg(any(test, feature = "unstable-test-utils"))]
    pub(crate) fn handle(&self) -> &tokio::runtime::Handle {
        self.inner.runtime.handle()
    }
}

impl<'normal, KV> ProvableExt<'normal, 'static, OperationalError> for Registry<KV, Normal>
where
    KV: BackgroundKeyValueStore,
    KV::Repo: Clone,
{
    type Prover = Registry<KV, Prove<'static>>;

    fn try_start_proof(&'normal self) -> Result<Self::Prover, OperationalError> {
        let Self {
            databases,
            inner: NormalImpl { repo, .. },
        } = self;

        let databases = databases.try_start_proof()?;
        let repo = repo.clone();

        Ok(Registry {
            inner: ProveImpl { repo },
            databases,
        })
    }
}

impl<KV: BackgroundPersistentKeyValueStore> Registry<KV, Normal>
where
    KV::Repo: RegistryRepo,
{
    /// Restore a registry from a previously committed manifest.
    ///
    /// The restored databases are checked out from the database commits referenced by the
    /// manifest, then the reconstructed registry root is verified against the requested
    /// `commit_id`.
    pub fn checkout(repo: KV::Repo, commit_id: CommitId) -> Result<Self, Error> {
        let manifest = Self::read_checkout_manifest(&repo, &commit_id)?;
        let runtime = Self::build_runtime()?;
        let databases = Self::checkout_databases(&runtime, &repo, &manifest.database_hashes)?;

        let registry = Registry {
            inner: NormalImpl { repo, runtime },
            databases: Vector::new(databases),
        };

        let actual_commit = CommitId::from(Hash::from_foldable(&registry));
        if actual_commit != commit_id {
            return Err(Error::Operational(OperationalError::RegistryCommitMismatch));
        }

        Ok(registry)
    }

    fn read_checkout_manifest(
        repo: &KV::Repo,
        commit_id: &CommitId,
    ) -> Result<RegistryManifest, OperationalError> {
        let commit_bytes = repo.read_registry_commit(commit_id)?;
        deserialise(&commit_bytes).map_err(OperationalError::from)
    }

    fn checkout_databases(
        runtime: &Arc<Runtime>,
        repo: &KV::Repo,
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
        let mut database_hashes = Vec::with_capacity(self.databases.len());

        for database in self.databases.iter() {
            let hash = database.commit(&self.inner.repo)?;
            database_hashes.push(hash);
        }

        let registry_commit = CommitId::from(Hash::from_foldable(&self));

        let manifest = RegistryManifest { database_hashes };
        let encoded =
            serialise(&manifest).expect("Serialising the registry manifest should not fail");

        self.inner
            .repo
            .write_registry_commit(&registry_commit, &encoded)?;

        Ok(registry_commit)
    }
}

impl<KV: KeyValueStore, M: Mode> Registry<KV, M> {
    /// Check the given `index` is valid for a database in the registry.
    fn validate_index(&self, index: usize) -> Result<(), InvalidArgumentError>
    where
        M: VectorMode,
    {
        if index >= self.databases.len() {
            Err(InvalidArgumentError::DatabaseIndexOutOfBounds)
        } else {
            Ok(())
        }
    }

    /// Are there no databases in the registry?
    pub fn is_empty(&self) -> bool
    where
        M: VectorMode,
    {
        self.len() == 0
    }

    /// Get the number of databases held in the registry.
    pub fn len(&self) -> usize
    where
        M: VectorMode,
    {
        self.databases.len()
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
        M: RegistryMode + VectorMode,
    {
        match self.len().abs_diff(new_size) {
            1 => (),
            0 => return Ok(()),
            _ => return Err(Error::from(InvalidArgumentError::RegistryResizeTooLarge)),
        }

        self.databases
            .try_resize_with(new_size, || M::try_new_database(&self.inner))?;

        Ok(())
    }

    /// Get a reference to the database at the given `index`.
    pub fn database(&self, index: usize) -> Result<&Database<KV, M>, Error>
    where
        M: VectorMode,
    {
        self.validate_index(index)?;
        Ok(&self.databases[index])
    }

    /// Get a mutable reference to the database at the given `index`.
    pub fn database_mut(&mut self, index: usize) -> Result<&mut Database<KV, M>, Error>
    where
        M: VectorMode,
    {
        self.validate_index(index)?;
        Ok(&mut self.databases[index])
    }

    /// Copy the contents of database at `src_index` to database at `dst_index`.
    pub fn copy_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
        M: RegistryMode + VectorMode,
    {
        self.validate_index(src_index)?;
        self.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if copying to the same index.
            return Ok(());
        }

        let db_copy = M::try_clone_database(&self.inner, &self.databases[src_index])?;
        self.databases[dst_index] = db_copy;

        Ok(())
    }

    /// Move the contents of database at `src_index` to database at `dst_index`. The source
    /// database is replaced with an empty database.
    pub fn move_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
        M: RegistryMode + VectorMode,
    {
        self.validate_index(src_index)?;
        self.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if moving to the same index.
            return Ok(());
        }

        let empty = M::try_new_database(&self.inner)?;
        let db_to_move = std::mem::replace(&mut self.databases[src_index], empty);
        self.databases[dst_index] = db_to_move;

        Ok(())
    }

    /// Clear the database at the given `index`.
    pub fn clear_database(&mut self, index: usize) -> Result<(), Error>
    where
        KV: BackgroundKeyValueStore,
        M: RegistryMode + VectorMode,
    {
        self.validate_index(index)?;
        self.databases[index] = M::try_new_database(&self.inner)?;
        Ok(())
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

impl<KV: KeyValueStore, M: Mode, F: Fold> Foldable<F> for Registry<KV, M>
where
    Database<KV, M>: Foldable<F>,
    Vector<Database<KV, M>, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        self.databases.fold(builder)
    }
}

/// Modal template for the [`Registry`]
///
/// This is used to select the appropriate implementation for the mode.
struct RegistryTemplate<KV: KeyValueStore>(PhantomData<KV>, Infallible);

impl<KV: KeyValueStore> Modal for RegistryTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = ProveImpl<KV>;

    type Verify = VerifyImpl<KV>;
}

/// Modes that implement this support operations on [`Registry`]
#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
pub trait RegistryMode: Mode {
    /// Create a new database.
    fn try_new_database<KV: BackgroundKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError>;

    /// Clone a database
    fn try_clone_database<KV: BackgroundKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        database: &Database<KV, Self>,
    ) -> Result<Database<KV, Self>, OperationalError>;
}

#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
impl RegistryMode for Normal {
    fn try_new_database<KV: BackgroundKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        Database::try_new(inner.runtime.handle(), &inner.repo)
    }

    fn try_clone_database<KV: BackgroundKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        database: &Database<KV, Self>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        database.try_clone_with(inner.runtime.handle(), &inner.repo)
    }
}

#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
impl RegistryMode for Prove<'static> {
    fn try_new_database<KV: BackgroundKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        let persistence = Arc::new(KV::new(&inner.repo)?);
        Ok(<Database<KV, Prove<'static>>>::empty(persistence))
    }

    fn try_clone_database<KV: BackgroundKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        database: &Database<KV, Self>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        let persistence = Arc::new(KV::new(&inner.repo)?);
        Ok(database.clone_with(persistence))
    }
}

#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
impl RegistryMode for Verify {
    fn try_new_database<KV: BackgroundKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        Ok(<Database<KV, Verify>>::empty())
    }

    fn try_clone_database<KV: BackgroundKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
        database: &Database<KV, Self>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        Ok(database.clone())
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
            .databases
            .iter()
            .map(|db| db.try_clone_with(runtime.handle(), &repo))
            .collect::<Result<_, _>>()?;
        let databases = Vector::new(databases);

        Ok(Registry {
            inner: NormalImpl { repo, runtime },

            databases,
        })
    }
}

/// Registry implementation for the [`Normal`] mode
struct NormalImpl<KV: KeyValueStore> {
    repo: KV::Repo,
    runtime: Arc<Runtime>,
}

/// Registry implementation for the [`Prove`] mode.
struct ProveImpl<KV: KeyValueStore> {
    repo: KV::Repo,
}

/// Registry implementation for the [`Verify`] mode.
struct VerifyImpl<KV: KeyValueStore>(PhantomData<KV>);

#[cfg(test)]
pub(super) mod tests {
    use std::marker::PhantomData;

    use bytes::Bytes;
    use octez_riscv_data::components::vector::VectorMode;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::ProvableExt;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_data::serialisation::serialise;

    use super::ProveImpl;
    use super::Registry;
    use super::RegistryManifest;
    use super::VerifyImpl;
    use crate::commit::CommitId;
    use crate::database::tests::to_verify;
    use crate::errors::Error;
    use crate::errors::InvalidArgumentError;
    use crate::errors::OperationalError;
    use crate::key::Key;
    use crate::merkle_worker::BackgroundKeyValueStore;
    use crate::merkle_worker::BackgroundPersistentKeyValueStore;
    use crate::repo::RegistryRepo;
    use crate::storage::TestKeyValueStoreSetup;
    use crate::storage::kv_test;

    pub(super) fn setup_registry<KV: BackgroundKeyValueStore>(
        repo: KV::Repo,
    ) -> Registry<KV, Normal> {
        Registry::new(repo).expect("Registry should be created")
    }

    pub(super) fn setup_size_2_registry<KV: BackgroundKeyValueStore>(
        repo: KV::Repo,
    ) -> Registry<KV, Normal> {
        let mut registry = setup_registry::<KV>(repo);
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        registry
            .resize_tick(2)
            .expect("Growing the registry should succeed.");

        registry
    }

    fn setup_prove_registry<KV: BackgroundKeyValueStore>(
        repo: KV::Repo,
    ) -> Registry<KV, Prove<'static>> {
        Registry {
            inner: ProveImpl { repo },
            databases: <Prove<'static> as VectorMode>::new(Vec::new()),
        }
    }

    fn setup_prove_size_2_registry<KV: BackgroundKeyValueStore>(
        repo: KV::Repo,
    ) -> Registry<KV, Prove<'static>> {
        let mut registry = setup_prove_registry::<KV>(repo);
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");
        registry
            .resize_tick(2)
            .expect("Growing the registry should succeed.");
        registry
    }

    fn setup_verify_registry<KV: BackgroundKeyValueStore>() -> Registry<KV, Verify> {
        Registry {
            inner: VerifyImpl(PhantomData),
            databases: <Verify as VectorMode>::new(Vec::new()),
        }
    }

    fn setup_verify_size_2_registry<KV: BackgroundKeyValueStore>() -> Registry<KV, Verify> {
        let mut registry = setup_verify_registry::<KV>();
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");
        registry
            .resize_tick(2)
            .expect("Growing the registry should succeed.");
        registry
    }

    fn seed_copy_move<KV: BackgroundKeyValueStore>(
        registry: &mut Registry<KV, Normal>,
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
            registry.databases[src_index]
                .write(key.clone(), 0, Bytes::copy_from_slice(value))
                .expect("Writing to source should succeed");
        }

        // Write values to destination that should be overwritten.
        registry.databases[dst_index]
            .write(key_a, 0, Bytes::copy_from_slice(b"old_a"))
            .expect("Writing to destination should succeed");
        registry.databases[dst_index]
            .write(key_c.clone(), 0, Bytes::copy_from_slice(b"old_c"))
            .expect("Writing to destination should succeed");

        (src_pairs, key_c)
    }

    fn assert_pairs_present<KV: BackgroundKeyValueStore>(
        registry: &Registry<KV, Normal>,
        db_index: usize,
        pairs: &[(Key, &'static [u8])],
    ) {
        for (key, value) in pairs.iter() {
            assert!(
                registry.databases[db_index]
                    .exists(key)
                    .expect("Checking destination should succeed")
            );
            let mut buf = vec![0u8; value.len()];
            registry.databases[db_index]
                .read(key, 0, buf.as_mut_slice())
                .expect("Reading from destination should succeed");
            assert_eq!(&buf, value);
        }
    }

    fn assert_pairs_absent<KV: BackgroundKeyValueStore>(
        registry: &Registry<KV, Normal>,
        db_index: usize,
        pairs: &[(Key, &'static [u8])],
    ) {
        for (key, _value) in pairs.iter() {
            assert!(
                !registry.databases[db_index]
                    .exists(key)
                    .expect("Checking source should succeed"),
                "Key should not exist in source after move."
            );
        }
    }

    pub(super) fn populate_database_with_key_value<KV: BackgroundKeyValueStore>(
        registry: &mut Registry<KV, Normal>,
        db_index: usize,
        key_bytes: &[u8],
        value: &[u8],
    ) {
        let key = Key::new(key_bytes).expect("Key should be valid");
        registry.databases[db_index]
            .set(key, Bytes::copy_from_slice(value))
            .expect("Writing to database should succeed");
    }

    kv_test!(test_new, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let registry = setup_registry::<KV>(repo);
        assert!(registry.is_empty());
    });

    kv_test!(test_resize, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_registry::<KV>(repo);

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
    });

    kv_test!(test_get_database, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_registry::<KV>(repo);

        while registry.len() < 3 {
            registry
                .resize_tick(registry.len() + 1)
                .expect("Growing the registry should succeed.");
        }

        for i in 0..3 {
            registry.database(i).expect("Database should exist.");
        }
    });

    kv_test!(test_copy_database, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo);

        let src_index = 0;
        let dst_index = 1;

        let (src_pairs, key_c) = seed_copy_move::<KV>(&mut registry, src_index, dst_index);

        registry
            .copy_database(src_index, dst_index)
            .expect("Copying should succeed");

        assert_pairs_present::<KV>(&registry, dst_index, &src_pairs);

        assert!(
            !registry.databases[dst_index]
                .exists(&key_c)
                .expect("Checking destination should succeed"),
            "Key C should not exist in destination after copy."
        );
    });

    kv_test!(test_database_operations_invalid_index, KV: BackgroundKeyValueStore, {
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

        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo);

        // Test copy operations with invalid indices
        assert_invalid_index_error!(registry.copy_database(0, 2), "copy", "to");

        assert_invalid_index_error!(registry.copy_database(2, 0), "copy", "from");

        // Test move operations with invalid indices
        assert_invalid_index_error!(registry.move_database(0, 2), "move", "to");

        assert_invalid_index_error!(registry.move_database(2, 0), "move", "from");

        // Test clear operation with invalid index
        assert_invalid_index_error!(registry.clear_database(2), "clear", "");
    });

    kv_test!(test_move_database, KV: BackgroundKeyValueStore, {
        // Test that the source database is emptied and the destination database
        // has all the data, and any data previously in the destination is lost.

        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo);

        let src_index = 1;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move::<KV>(&mut registry, src_index, dst_index);

        registry
            .move_database(src_index, dst_index)
            .expect("Moving should succeed");

        assert_pairs_present::<KV>(&registry, dst_index, &src_pairs);
        assert_pairs_absent::<KV>(&registry, src_index, &src_pairs);
    });

    kv_test!(test_database_operations_same_index, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo);

        let (src_pairs, _key_c) = seed_copy_move::<KV>(&mut registry, 0, 1);

        // Test copy with same index
        registry
            .copy_database(0, 0)
            .expect("Copying to same index should succeed");
        assert_pairs_present::<KV>(&registry, 0, &src_pairs);

        // Test move with same index
        registry
            .move_database(0, 0)
            .expect("Moving to same index should succeed");
        assert_pairs_present::<KV>(&registry, 0, &src_pairs);
    });

    kv_test!(test_clear_database, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo);

        let db_index = 0;
        populate_database_with_key_value::<KV>(&mut registry, db_index, &[1], b"some_value");
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        registry
            .clear_database(db_index)
            .expect("Clearing the database should succeed");

        assert!(
            !registry.databases[db_index]
                .exists(&key)
                .expect("Checking database should succeed"),
            "Key should not exist after clearing the database."
        );
    });

    impl<KV> Registry<KV, Normal>
    where
        KV: BackgroundPersistentKeyValueStore,
        KV::Repo: RegistryRepo,
    {
        /// Read and deserialise the manifest for `commit_id`.
        fn read_manifest(&self, commit_id: &CommitId) -> RegistryManifest {
            let bytes = self
                .inner
                .repo
                .read_registry_commit(commit_id)
                .expect("Manifest should be readable");
            deserialise(&bytes).expect("Manifest should be deserialisable")
        }

        /// Assert that the manifest written for `commit_id` contains the expected database commit IDs.
        fn verify_manifest(&self, commit_id: &CommitId, expected_db_hashes: &[CommitId]) {
            let manifest = self.read_manifest(commit_id);
            assert_eq!(manifest.database_hashes, expected_db_hashes);
        }
    }

    kv_test!(test_registry_commit_empty, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let registry = setup_registry::<KV>(repo.clone());

        let expected_db_hashes: Vec<CommitId> = Vec::new();

        // the hash is of the empty vector (zero length and empty list):
        // (hash (concat (hash 0) (hash-as-seq '())))
        let zero = Hash::hash_encodable(0u64).expect("hashing u64 should succeed");
        let empty_seq = Hash::hash_bytes(&[]);
        let expected_root = CommitId::from(Hash::combine_hashes([zero, empty_seq]));

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        registry.verify_manifest(&root_commit, &expected_db_hashes);
    });

    kv_test!(test_registry_commit_size_1, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_registry::<KV>(repo.clone());
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"singleton");

        let expected_db_hashes: Vec<CommitId> = registry
            .databases
            .iter()
            .map(|db| db.hash().unwrap().into())
            .collect();

        // the hash is of the single-entry vector (one length and list size 1):
        // (hash (concat (hash 1) (hash-as-seq '(db))))
        let one = Hash::hash_encodable(1u64).expect("hashing u64 should succeed");
        let db_hash = expected_db_hashes[0];
        let expected_root = CommitId::from(Hash::combine_hashes([&one, db_hash.as_hash()]));

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        registry.verify_manifest(&root_commit, &expected_db_hashes);
    });

    kv_test!(test_committing_identical_registry_succeeds, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_registry::<KV>(repo);
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"singleton");

        let first_commit = registry.commit().expect("First commit should succeed");
        let second_commit = registry.commit().expect("Second commit should succeed");

        assert_eq!(
            first_commit, second_commit,
            "Committing identical registry states should yield the same commit ID."
        );
    });

    kv_test!(test_registry_commit_writes_manifest, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo.clone());

        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"alpha");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"beta");

        // the hash is of the dual-entry vector (two length and list size 2):
        // (hash (concat (hash 2) (hash-as-seq '(db_0 db_1))))
        let expected_db_hashes: Vec<CommitId> = registry
            .databases
            .iter()
            .map(|db| db.hash().unwrap().into())
            .collect();
        let two = Hash::hash_encodable(2u64).expect("hashing u64 should succeed");
        let dbs_root_hash = Hash::combine_hashes(expected_db_hashes.iter().map(CommitId::as_hash));

        let expected_root = CommitId::from(Hash::combine_hashes([two, dbs_root_hash]));

        let root_commit = registry.commit().expect("Commit should succeed");
        assert_eq!(root_commit, expected_root);

        registry.verify_manifest(&root_commit, &expected_db_hashes);
    });

    kv_test!(test_registry_checkout_roundtrip_empty, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let registry = setup_registry::<KV>(repo.clone());

        let root_commit = registry.commit().expect("Commit should succeed");
        let checked_out = Registry::<KV, Normal>::checkout(repo, root_commit)
            .expect("Checkout should succeed");

        assert!(checked_out.is_empty());
        assert_eq!(
            CommitId::from(Hash::from_foldable(&checked_out)),
            root_commit
        );
    });

    kv_test!(test_registry_checkout_roundtrip_populated, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo.clone());

        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"alpha");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"beta");

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");

        let root_commit = registry.commit().expect("Commit should succeed");
        let checked_out = Registry::<KV, Normal>::checkout(repo, root_commit)
            .expect("Checkout should succeed");

        assert_eq!(checked_out.len(), 2);
        checked_out.databases[0].assert_database_value(&key_a, b"alpha");
        checked_out.databases[1].assert_database_value(&key_b, b"beta");
        assert_eq!(
            CommitId::from(Hash::from_foldable(&checked_out)),
            root_commit
        );
    });

    kv_test!(test_registry_checkout_missing_commit_fails, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let missing_commit = CommitId::from(Hash::hash_bytes(b"missing-registry-commit"));

        assert!(matches!(
            Registry::<KV, Normal>::checkout(repo, missing_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));
    });

    kv_test!(test_registry_checkout_detects_manifest_root_mismatch, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_registry::<KV>(repo.clone());
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"value");

        let root_commit = registry.commit().expect("Commit should succeed");
        let manifest = registry.read_manifest(&root_commit);

        let fake_commit = CommitId::from(Hash::hash_bytes(b"fake-registry-commit"));
        let fake_manifest_bytes =
            serialise(&manifest).expect("Manifest should be serialisable");
        repo.write_registry_commit(&fake_commit, &fake_manifest_bytes)
            .expect("Writing the fake manifest should succeed");

        assert!(matches!(
            Registry::<KV, Normal>::checkout(repo, fake_commit),
            Err(Error::Operational(OperationalError::RegistryCommitMismatch))
        ));
    });

    kv_test!(test_hashing_prove_registry_does_not_record_reads, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");

        let mut registry = setup_size_2_registry::<KV>(repo);

        registry
            .database_mut(0)
            .expect("database at index 0 exists")
            .set(key_a, Bytes::copy_from_slice(b"foo"))
            .expect("Setting in Normal mode should succeed");

        registry
            .database_mut(1)
            .expect("database at index 1 exists")
            .set(key_b, Bytes::copy_from_slice(b"bar"))
            .expect("Setting in Normal mode should succeed");

        let prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");

        let proof_before_hash =
            octez_riscv_data::merkle_proof::proof_tree::MerkleProof::from_foldable(
                &prove_registry.databases,
            );

        let proof_before_hash_bytes =
            serialise(&proof_before_hash).expect("Serialising proof should succeed");

        let normal_root = Hash::from_foldable(&registry);
        let prove_root = Hash::from_foldable(&prove_registry);
        assert_eq!(prove_root, normal_root);

        let proof_after_hash =
            octez_riscv_data::merkle_proof::proof_tree::MerkleProof::from_foldable(
                &prove_registry.databases,
            );
        let proof_after_hash_bytes =
            serialise(&proof_after_hash).expect("Serialising proof should succeed");

        assert_eq!(
            proof_after_hash_bytes, proof_before_hash_bytes,
            "Proof should be unchanged by hashing"
        );
    });

    kv_test!(test_prove_clear_database, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_size_2_registry::<KV>(repo);

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"value"))
            .expect("Setting a value should succeed");

        registry
            .clear_database(0)
            .expect("Clearing should succeed");

        assert!(
            !registry
                .database(0)
                .expect("Database should exist.")
                .exists(&key)
                .expect("Existence check should succeed"),
            "Database should be empty after clear."
        );
    });

    kv_test!(test_prove_copy_database, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_size_2_registry::<KV>(repo);

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"src"))
            .expect("Setting a value should succeed");
        registry
            .database_mut(1)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"dst"))
            .expect("Setting a value should succeed");

        registry
            .copy_database(0, 1)
            .expect("Copying should succeed");

        registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key, b"src");
    });

    kv_test!(test_prove_database_clone_independence, KV: BackgroundKeyValueStore, {
        // Cloning via copy should produce an independent database — mutations to the source
        // after the copy must not propagate to the destination.
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_size_2_registry::<KV>(repo);

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"original"))
            .expect("Setting a value should succeed");

        registry
            .copy_database(0, 1)
            .expect("Copying should succeed");

        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"mutated"))
            .expect("Setting a value should succeed");

        registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key, b"mutated");
        registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key, b"original");
    });

    kv_test!(test_prove_database_ops, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_size_2_registry::<KV>(repo);

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let database = registry.database_mut(0).expect("Database should exist.");
        database
            .set(key.clone(), Bytes::from_static(b"alpha"))
            .expect("Setting a value should succeed");
        database.assert_database_value(&key, b"alpha");

        database
            .delete(key.clone())
            .expect("Deleting a value should succeed");
        assert!(
            !database
                .exists(&key)
                .expect("Existence check should succeed")
        );
    });

    kv_test!(test_prove_invalid_index, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_size_2_registry::<KV>(repo);

        assert!(matches!(
            registry.copy_database(0, 2),
            Err(Error::InvalidArgument(
                InvalidArgumentError::DatabaseIndexOutOfBounds
            ))
        ));
        assert!(matches!(
            registry.move_database(2, 0),
            Err(Error::InvalidArgument(
                InvalidArgumentError::DatabaseIndexOutOfBounds
            ))
        ));
        assert!(matches!(
            registry.clear_database(2),
            Err(Error::InvalidArgument(
                InvalidArgumentError::DatabaseIndexOutOfBounds
            ))
        ));
    });

    kv_test!(test_prove_move_database, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_size_2_registry::<KV>(repo);

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"value"))
            .expect("Setting a value should succeed");

        registry
            .move_database(0, 1)
            .expect("Moving should succeed");

        assert!(
            !registry
                .database(0)
                .expect("Database should exist.")
                .exists(&key)
                .expect("Existence check should succeed"),
            "Source should be empty after move."
        );
        registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key, b"value");
    });

    kv_test!(test_prove_new, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let registry = setup_prove_registry::<KV>(repo);
        assert!(registry.is_empty());
    });

    kv_test!(test_prove_resize, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_prove_registry::<KV>(repo);

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
    });

    // Reading through a `Registry<Prove>` populated from snapshots of Normal-mode source
    // data records a proof; replaying the same reads through the resulting
    // `Registry<Verify>` must yield the same values.
    kv_test!(test_verify_replays_prove_reads, KV: BackgroundKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");

        let mut registry = setup_size_2_registry(repo);

        let db_0 = registry.database_mut(0)
            .expect("database at index 0 exists");

        db_0.set(key_a.clone(), Bytes::copy_from_slice(b"foo"))
            .expect("Setting in Normal mode should succeed");
        let db_0_hash = db_0.hash().expect("Hashing should succeed");

        let db_1 = registry.database_mut(1)
            .expect("database at index 1 exists");

        db_1.set(key_b.clone(), Bytes::copy_from_slice(b"bar"))
            .expect("Setting in Normal mode should succeed");
        let db_1_hash = db_1.hash().expect("Hashing should succeed");

        let expected_hashes = [db_0_hash, db_1_hash];

        let root_hash = Hash::from_foldable(&registry);

        let prove_registry = registry.try_start_proof()
            .expect("Converting to prove mode should succeed");

        let prove_hashes_before = (0..prove_registry.len())
            .map(|i| {
                prove_registry
                    .database(i)
                    .expect("Database should exist.")
                    .hash()
                    .expect("Hashing should succeed.")
            })
            .collect::<Vec<_>>();
        assert_eq!(prove_hashes_before.as_slice(), expected_hashes);

        let root_hash_prove_before = Hash::from_foldable(&prove_registry);

        assert_eq!(root_hash_prove_before, root_hash, "Converting to proof mode concerves root hash");

        prove_registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key_a, b"foo");
        prove_registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key_b, b"bar");

        let prove_hashes_after = (0..prove_registry.len())
            .map(|i| {
                prove_registry
                    .database(i)
                    .expect("Database should exist.")
                    .hash()
                    .expect("Hashing should succeed.")
            })
            .collect::<Vec<_>>();
        let root_hash_prove_after = Hash::from_foldable(&prove_registry);

        assert_eq!(
            prove_hashes_after, prove_hashes_before,
            "Reads must not affect Prove-mode hashes."
        );
        assert_eq!(root_hash_prove_after, root_hash_prove_before, "Reads must not affect Prove-mode hashes");

        let verify_databases = (0..prove_registry.len())
            .map(|i| to_verify::<KV>(prove_registry.database(i).expect("Database should exist.")))
            .collect();

        let verify_registry = Registry::<KV, Verify> {
            inner: VerifyImpl(PhantomData),
            databases: <Verify as VectorMode>::new(verify_databases),
        };

        let verify_hashes_before = (0..verify_registry.len())
            .map(|i| {
                verify_registry
                    .database(i)
                    .expect("Database should exist.")
                    .hash()
                    .expect("Hashing should succeed.")
            })
            .collect::<Vec<_>>();
        assert_eq!(verify_hashes_before.as_slice(), expected_hashes);

        // The same reads through the Verify-mode registry should yield the same values.
        verify_registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key_a, b"foo");

        verify_registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key_b, b"bar");

        let verify_hashes_after = (0..verify_registry.len())
            .map(|i| {
                verify_registry
                    .database(i)
                    .expect("Database should exist.")
                    .hash()
                    .expect("Hashing should succeed.")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            verify_hashes_after, verify_hashes_before,
            "Reads must not affect Verify-mode hashes."
        );
    });

    kv_test!(test_verify_clear_database, KV: BackgroundKeyValueStore, {
        let mut registry = setup_verify_size_2_registry::<KV>();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"value"))
            .expect("Setting a value should succeed");

        registry
            .clear_database(0)
            .expect("Clearing should succeed");

        assert!(
            !registry
                .database(0)
                .expect("Database should exist.")
                .exists(&key)
                .expect("Existence check should succeed"),
            "Database should be empty after clear."
        );
    });

    kv_test!(test_verify_copy_database, KV: BackgroundKeyValueStore, {
        let mut registry = setup_verify_size_2_registry::<KV>();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"src"))
            .expect("Setting a value should succeed");
        registry
            .database_mut(1)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"dst"))
            .expect("Setting a value should succeed");

        registry
            .copy_database(0, 1)
            .expect("Copying should succeed");

        registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key, b"src");
    });

    kv_test!(test_verify_database_clone_independence, KV: BackgroundKeyValueStore, {
        // Cloning via copy should produce an independent database — mutations to the source
        // after the copy must not propagate to the destination.
        let mut registry = setup_verify_size_2_registry::<KV>();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"original"))
            .expect("Setting a value should succeed");

        registry
            .copy_database(0, 1)
            .expect("Copying should succeed");

        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"mutated"))
            .expect("Setting a value should succeed");

        registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key, b"mutated");
        registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key, b"original");
    });

    kv_test!(test_verify_database_ops, KV: BackgroundKeyValueStore, {
        // Exercise read/write/delete on a verify-mode database obtained from the registry.
        let mut registry = setup_verify_size_2_registry::<KV>();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let database = registry.database_mut(0).expect("Database should exist.");
        database
            .set(key.clone(), Bytes::from_static(b"alpha"))
            .expect("Setting a value should succeed");
        database.assert_database_value(&key, b"alpha");

        database
            .delete(key.clone())
            .expect("Deleting a value should succeed");
        assert!(
            !database
                .exists(&key)
                .expect("Existence check should succeed")
        );
    });

    kv_test!(test_verify_invalid_index, KV: BackgroundKeyValueStore, {
        let mut registry = setup_verify_size_2_registry::<KV>();

        assert!(matches!(
            registry.copy_database(0, 2),
            Err(Error::InvalidArgument(
                InvalidArgumentError::DatabaseIndexOutOfBounds
            ))
        ));
        assert!(matches!(
            registry.move_database(2, 0),
            Err(Error::InvalidArgument(
                InvalidArgumentError::DatabaseIndexOutOfBounds
            ))
        ));
        assert!(matches!(
            registry.clear_database(2),
            Err(Error::InvalidArgument(
                InvalidArgumentError::DatabaseIndexOutOfBounds
            ))
        ));
    });

    kv_test!(test_verify_move_database, KV: BackgroundKeyValueStore, {
        let mut registry = setup_verify_size_2_registry::<KV>();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry
            .database_mut(0)
            .expect("Database should exist.")
            .set(key.clone(), Bytes::from_static(b"value"))
            .expect("Setting a value should succeed");

        registry
            .move_database(0, 1)
            .expect("Moving should succeed");

        assert!(
            !registry
                .database(0)
                .expect("Database should exist.")
                .exists(&key)
                .expect("Existence check should succeed"),
            "Source should be empty after move."
        );
        registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key, b"value");
    });

    kv_test!(test_verify_new, KV: BackgroundKeyValueStore, {
        let registry = setup_verify_registry::<KV>();
        assert!(registry.is_empty());
    });

    kv_test!(test_verify_resize, KV: BackgroundKeyValueStore, {
        let mut registry = setup_verify_registry::<KV>();

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
    });

    kv_test!(test_durable_storage_end_to_end, KV: BackgroundPersistentKeyValueStore,
    [
        generated in crate::test_helpers::registry_operations_strategy(1usize..100)
    ],
    {
        // Every test iteration expects an empty repo, so not setting it in a `setup` block.
        // This is because a repo preserves registry commits from previous test runs, resulting
        // in test failures when checking out a commit which isn't expected to exist succeeds.
        let (_keepalive, repo) = KV::setup_repo();

        let (keys, values, ops) = generated;
        let operations = crate::test_helpers::make_registry_operations(keys, values, ops);
        crate::test_helpers::run_operations::<KV>(repo, operations)
    });
}

#[cfg(feature = "rocksdb")]
#[cfg(test)]
mod rocksdb_tests {
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::serialisation::deserialise;

    use super::Registry;
    use super::RegistryManifest;
    use super::tests::populate_database_with_key_value;
    use super::tests::setup_registry;
    use crate::errors::Error;
    use crate::errors::OperationalError;
    use crate::storage::TestKeyValueStore;
    use crate::storage::setup_repo;

    #[test]
    fn test_registry_checkout_missing_database_commit_fails() {
        let (_keepalive, repo) = setup_repo();
        let mut registry = setup_registry::<TestKeyValueStore>(repo.clone());
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");

        populate_database_with_key_value::<TestKeyValueStore>(&mut registry, 0, &[1], b"singleton");

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
}
