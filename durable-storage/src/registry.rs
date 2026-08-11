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
use octez_riscv_data::codec;
use octez_riscv_data::components::vector::Vector;
use octez_riscv_data::components::vector::VectorMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::merkle_proof::proof::Proof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::ProvableExt;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::deserialise;
use octez_riscv_data::serialisation::serialise;
use once_cell::sync::OnceCell;
use tokio::runtime::Handle;
use tokio::runtime::Runtime;

use crate::avl::tree::Tree;
use crate::commit::CommitId;
use crate::database::Database;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::merkle_worker::BackgroundReadableKeyValueStore;
use crate::merkle_worker::BackgroundWriteableKeyValueStore;
use crate::repo::RegistryRepo;
use crate::storage::ReadOnlyKeyValueStore;
use crate::storage::ReadableKeyValueStore;

#[derive(Debug, Encode, Decode)]
/// Structure to store the result of serialising a registry.
struct RegistryManifest {
    database_hashes: Vec<CommitId>,
}

/// Registry that owns a set of databases and the repository used to manage
/// registry state.
pub struct Registry<KV: ReadableKeyValueStore, M: Mode> {
    inner: M::Select<RegistryTemplate<KV>>,
    databases: Vector<Database<KV, M>, M>,
}

impl<KV: BackgroundReadableKeyValueStore> Registry<KV, Normal> {
    /// Creates a new, empty Registry.
    ///
    /// The registry owns a register state repository. The [`Runtime`] hosting its databases'
    /// Merkle workers is built when one is first needed - see [`LazyRuntime`].
    pub fn new(repo: KV::Repo) -> Self {
        Registry {
            inner: NormalImpl {
                repo,
                runtime: LazyRuntime::default(),
            },
            databases: Vector::new(Vec::new()),
        }
    }

    /// Get a [`Handle`] to the registry's runtime.
    #[cfg(test_utils)]
    pub(crate) fn handle(&self) -> &tokio::runtime::Handle {
        self.inner
            .runtime
            .handle()
            .expect("Building the runtime should succeed")
    }
}

impl<'normal, KV> ProvableExt<'normal, 'static, OperationalError> for Registry<KV, Normal>
where
    KV: BackgroundReadableKeyValueStore,
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

impl<KV: ReadableKeyValueStore> Registry<KV, Prove<'static>> {
    /// Produce a [`Proof`] of the operations performed against this
    /// Prove-mode registry.
    ///
    /// The proof bundles a partial Merkle tree of the initial state (the
    /// read-set captured during the proving step, plus blinded hashes of any
    /// subtrees that were not touched) with the registry's current — final —
    /// state hash.
    pub fn produce_proof(&self) -> Proof {
        let merkle_proof = MerkleProof::from_foldable(self);
        let final_state_hash = Hash::from_foldable(self);
        Proof::new(merkle_proof, final_state_hash)
    }
}

impl<KV: ReadableKeyValueStore> Registry<KV, Normal>
where
    KV::Repo: RegistryRepo,
{
    fn read_checkout_manifest(
        repo: &KV::Repo,
        commit_id: &CommitId,
    ) -> Result<RegistryManifest, OperationalError> {
        let commit_bytes = repo.read_registry_commit(commit_id)?;
        deserialise(&commit_bytes).map_err(OperationalError::from)
    }

    /// Assemble a registry from databases restored from the commits the manifest of `commit_id`
    /// referenced, verifying the reconstructed registry root against `commit_id`.
    ///
    /// How much that verifies depends on where the database hashes come from. Under
    /// [`Registry::checkout`] each database has loaded its Merkle tree, so the fold is over roots
    /// computed from the data on disk. Under [`Registry::checkout_read_only`] each database holds
    /// a [`CommittedRoot`], which answers with the very commit id the manifest named: the fold then
    /// re-derives the registry root from the contents of the manifest itself, and so catches a
    /// manifest that does not match the id it is stored under - not a database that does not match
    /// its commit. As with a missing root blob, that failure surfaces when the database is read or
    /// upgraded.
    ///
    /// [`CommittedRoot`]: crate::merkle_worker::CommittedRoot
    fn from_restored_databases(
        repo: KV::Repo,
        runtime: LazyRuntime,
        databases: Vec<Database<KV, Normal>>,
        commit_id: CommitId,
    ) -> Result<Self, Error> {
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
        let runtime = LazyRuntime::default();
        let databases = Self::checkout_databases(&runtime, &repo, &manifest.database_hashes)?;

        Self::from_restored_databases(repo, runtime, databases, commit_id)
    }

    /// The database commit ids referenced by the registry committed at `commit_id`.
    #[cfg(rocksdb_test_utils)]
    pub(crate) fn database_commits(
        repo: &KV::Repo,
        commit_id: &CommitId,
    ) -> Result<Vec<CommitId>, OperationalError> {
        Ok(Self::read_checkout_manifest(repo, commit_id)?.database_hashes)
    }

    fn checkout_databases(
        runtime: &LazyRuntime,
        repo: &KV::Repo,
        database_hashes: &[CommitId],
    ) -> Result<Vec<Database<KV, Normal>>, Error> {
        let handle = runtime.handle()?;

        // TODO RV-946: Investigate parallelising the checkouts of individual databases.
        database_hashes
            .iter()
            .map(|&db_hash| Database::checkout(handle, repo, db_hash))
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

        // Recorded after the manifest, so that an interrupted commit leaves a manifest collection
        // can reclaim rather than a recorded root with nothing behind it.
        self.inner.repo.record_commit(&registry_commit)?;

        Ok(registry_commit)
    }
}

/// A registry whose databases read their commits in place, through a [`ReadOnlyKeyValueStore`].
///
/// As for [`Database`], the mutating operations - [`Registry::resize_tick`], copy/move/clear and
/// [`Registry::commit`] - are bounded on a writeable store, so they do not exist here.
///
/// Read-only databases hold no Merkle tree, so such a registry hosts no worker threads and builds
/// no async runtime - see [`LazyRuntime`].
///
/// [`ReadOnlyKeyValueStore`]: crate::storage::ReadOnlyKeyValueStore
impl<KV: ReadOnlyKeyValueStore> Registry<KV, Normal>
where
    KV::Repo: RegistryRepo,
{
    /// Restore a registry from a previously committed manifest, for reading only.
    ///
    /// Where [`Registry::checkout`] copies every database commit into a working state, this reads
    /// them where they lie.
    pub fn checkout_read_only(repo: KV::Repo, commit_id: CommitId) -> Result<Self, Error> {
        let manifest = Self::read_checkout_manifest(&repo, &commit_id)?;

        // TODO RV-946: Investigate parallelising the checkouts of individual databases.
        let databases = manifest
            .database_hashes
            .iter()
            .map(|&db_hash| Database::checkout_read_only(&repo, db_hash))
            .collect::<Result<Vec<_>, _>>()?;

        Self::from_restored_databases(repo, LazyRuntime::default(), databases, commit_id)
    }

    /// Create another registry over the same committed state.
    ///
    /// Copies nothing and cannot fail - see [`Database::clone_read_only`] - so read-only consumers
    /// can be handed a registry each for free.
    pub fn clone_read_only(&self) -> Self {
        let NormalImpl { repo, runtime } = &self.inner;

        let databases = self
            .databases
            .iter()
            .map(Database::clone_read_only)
            .collect();

        Registry {
            inner: NormalImpl {
                repo: repo.clone(),
                runtime: runtime.clone(),
            },
            databases: Vector::new(databases),
        }
    }

    /// Copy the committed state of every database into a working state - see
    /// [`Database::to_writeable`].
    pub fn to_writeable(&self) -> Result<Registry<KV::Writeable, Normal>, OperationalError>
    where
        KV::Writeable: BackgroundPersistentKeyValueStore,
    {
        let NormalImpl { repo, runtime } = &self.inner;
        let handle = runtime.handle()?;

        // TODO RV-946: Investigate parallelising the checkouts of individual databases.
        let databases = self
            .databases
            .iter()
            .map(|db| db.to_writeable(handle, repo))
            .collect::<Result<_, _>>()?;

        Ok(Registry {
            inner: NormalImpl {
                repo: repo.clone(),
                runtime: runtime.clone(),
            },
            databases: Vector::new(databases),
        })
    }
}

impl<KV: ReadableKeyValueStore, M: Mode> Registry<KV, M> {
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
        KV: BackgroundWriteableKeyValueStore,
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
        KV: BackgroundWriteableKeyValueStore,
        M: RegistryMode + VectorMode,
    {
        self.validate_index(src_index)?;
        self.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if copying to the same index.
            return Ok(());
        }

        M::copy_database(&self.inner, &mut self.databases, src_index, dst_index)?;

        Ok(())
    }

    /// Move the contents of database at `src_index` to database at `dst_index`. The source
    /// database is replaced with an empty database.
    pub fn move_database(&mut self, src_index: usize, dst_index: usize) -> Result<(), Error>
    where
        KV: BackgroundWriteableKeyValueStore,
        M: RegistryMode + VectorMode,
    {
        self.validate_index(src_index)?;
        self.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if moving to the same index.
            return Ok(());
        }

        M::move_database(&self.inner, &mut self.databases, src_index, dst_index)?;

        Ok(())
    }

    /// Clear the database at the given `index`.
    pub fn clear_database(&mut self, index: usize) -> Result<(), Error>
    where
        KV: BackgroundWriteableKeyValueStore,
        M: RegistryMode + VectorMode,
    {
        self.validate_index(index)?;
        M::clear_database(&self.inner, &mut self.databases, index)?;
        Ok(())
    }
}

impl<KV: BackgroundWriteableKeyValueStore, M: CloneRegistryMode> Registry<KV, M>
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

impl<KV: ReadableKeyValueStore, M: Mode, F: Fold> Foldable<F> for Registry<KV, M>
where
    Database<KV, M>: Foldable<F>,
    Vector<Database<KV, M>, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        self.databases.fold(builder)
    }
}

impl<KV: ReadableKeyValueStore> FromProof for Registry<KV, Verify> {
    // TODO (TZX-161): for a verify mode Registry, the resulting registry state is currently
    // only usable if `proof` is the ProofTree. Otherwise, if created using the stream
    // deserialiser, verification will fail (as this does not support capturing the owned proof).
    // Deserialising from raw bytes therefore requires two passes: a stream pass to reconstruct
    // the proof tree, then a proof-tree pass to obtain a verifiable registry.
    fn from_proof<Proof: Deserialiser<Codec = codec::Bincode>>(
        proof: Proof,
    ) -> SuspendedResult<Proof, Self> {
        let suspended = Vector::<Database<KV, Verify>, Verify>::from_proof(proof)?;
        Ok(suspended.map(|databases| Self {
            inner: VerifyImpl,
            databases,
        }))
    }
}

/// Modal template for the [`Registry`]
///
/// This is used to select the appropriate implementation for the mode.
struct RegistryTemplate<KV: ReadableKeyValueStore>(PhantomData<KV>, Infallible);

impl<KV: ReadableKeyValueStore> Modal for RegistryTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = ProveImpl<KV>;

    type Verify = VerifyImpl;
}

/// Modes that implement this support operations on [`Registry`]
#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
pub trait RegistryMode: VectorMode {
    /// Create a new database.
    fn try_new_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError>;

    /// Copy the database at `src_index` over the one at `dst_index`.
    fn copy_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError>;

    /// Move the database at `src_index` over the one at `dst_index`, leaving an empty database
    /// at `src_index`.
    fn move_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError>;

    /// Clear the database at `index`.
    fn clear_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        index: usize,
    ) -> Result<(), OperationalError>;
}

#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
impl RegistryMode for Normal {
    fn try_new_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        Database::try_new(inner.runtime.handle()?, &inner.repo)
    }

    fn copy_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError> {
        let db_copy = databases[src_index].try_clone_with(inner.runtime.handle()?, &inner.repo)?;
        databases[dst_index] = db_copy;
        Ok(())
    }

    fn move_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError> {
        let empty = Self::try_new_database(inner)?;
        let db_to_move = std::mem::replace(&mut databases[src_index], empty);
        databases[dst_index] = db_to_move;
        Ok(())
    }

    fn clear_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        index: usize,
    ) -> Result<(), OperationalError> {
        databases[index] = Self::try_new_database(inner)?;
        Ok(())
    }
}

#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
impl RegistryMode for Prove<'static> {
    fn try_new_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        let persistence = Arc::new(KV::new(&inner.repo)?);
        Ok(<Database<KV, Prove<'static>>>::empty(persistence))
    }

    // Copy, move and clear replace only the destination's *working* tree: the destination
    // database keeps its initial tree and access tracking, so the proof still encodes the
    // destination's true state at the start of the step (blinded if it was not read).
    // See TZX-170.
    //
    // TODO TZX-173: reads of the destination *after* a copy/move hit nodes that originate from
    // the source slot's initial tree, but are recorded against the destination slot — the
    // source's fold blinds them, so such proofs under-include data and fail to verify.

    fn copy_database<KV: BackgroundWriteableKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError> {
        let working_tree = databases[src_index].clone_working_tree();
        databases[dst_index].replace_working_tree(working_tree);
        Ok(())
    }

    fn move_database<KV: BackgroundWriteableKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError> {
        let working_tree = databases[src_index].replace_working_tree(Tree::default());
        databases[dst_index].replace_working_tree(working_tree);
        Ok(())
    }

    fn clear_database<KV: BackgroundWriteableKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        index: usize,
    ) -> Result<(), OperationalError> {
        databases[index].replace_working_tree(Tree::default());
        Ok(())
    }
}

#[expect(
    private_interfaces,
    reason = "This method should not be used outside of this module"
)]
impl RegistryMode for Verify {
    fn try_new_database<KV: BackgroundWriteableKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
    ) -> Result<Database<KV, Self>, OperationalError> {
        Ok(<Database<KV, Verify>>::empty())
    }

    fn copy_database<KV: BackgroundWriteableKeyValueStore>(
        _inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError> {
        let db_copy = databases[src_index].clone();
        databases[dst_index] = db_copy;
        Ok(())
    }

    fn move_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), OperationalError> {
        let empty = Self::try_new_database(inner)?;
        let db_to_move = std::mem::replace(&mut databases[src_index], empty);
        databases[dst_index] = db_to_move;
        Ok(())
    }

    fn clear_database<KV: BackgroundWriteableKeyValueStore>(
        inner: &Self::Select<RegistryTemplate<KV>>,
        databases: &mut Vector<Database<KV, Self>, Self>,
        index: usize,
    ) -> Result<(), OperationalError> {
        databases[index] = Self::try_new_database(inner)?;
        Ok(())
    }
}

/// Modes that implement this marker support cloning of the [`Registry`] type
pub trait CloneRegistryMode: Mode {
    /// See [`Registry::try_clone`]
    fn try_clone<KV: BackgroundWriteableKeyValueStore>(
        this: &Registry<KV, Self>,
    ) -> Result<Registry<KV, Self>, OperationalError>
    where
        KV::Repo: Clone;
}

impl CloneRegistryMode for Normal {
    fn try_clone<KV: BackgroundWriteableKeyValueStore>(
        this: &Registry<KV, Self>,
    ) -> Result<Registry<KV, Self>, OperationalError>
    where
        KV::Repo: Clone,
    {
        let runtime = this.inner.runtime.clone();
        let repo = this.inner.repo.clone();
        let handle = runtime.handle()?;

        let databases = this
            .databases
            .iter()
            .map(|db| db.try_clone_with(handle, &repo))
            .collect::<Result<_, _>>()?;
        let databases = Vector::new(databases);

        Ok(Registry {
            inner: NormalImpl { repo, runtime },

            databases,
        })
    }
}

/// Registry implementation for the [`Normal`] mode
struct NormalImpl<KV: ReadableKeyValueStore> {
    repo: KV::Repo,
    runtime: LazyRuntime,
}

/// The async runtime hosting the Merkle workers of a registry's databases, built when first needed.
///
/// A read-only registry has no workers to host - its databases hold a
/// [`CommittedRoot`](crate::merkle_worker::CommittedRoot), not a tree - so it never builds a
/// runtime. Only the writeable paths, including
/// [`to_writeable`](Registry::to_writeable), ask for a handle.
///
/// Clones share the runtime, so upgrading a read-only registry twice builds only one.
#[derive(Clone, Default)]
struct LazyRuntime(Arc<OnceCell<Runtime>>);

impl LazyRuntime {
    /// A handle to the runtime, building it if this is the first caller to need one.
    ///
    /// The runtime is built inside the initialiser, so exactly one is ever built. Were it built
    /// beforehand, a caller that lost the race to store it would have to drop the one it built -
    /// and dropping a runtime blocks on its worker threads, which panics when it happens inside an
    /// async context. That is precisely where a read-only registry is upgraded to a writeable one.
    fn handle(&self) -> Result<&Handle, OperationalError> {
        let runtime = self.0.get_or_try_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .build()
                .map_err(|error| OperationalError::WorkerRuntimeCreationFailed { error })
        })?;

        Ok(runtime.handle())
    }
}

/// Registry implementation for the [`Prove`] mode.
struct ProveImpl<KV: ReadableKeyValueStore> {
    repo: KV::Repo,
}

/// Registry implementation for the [`Verify`] mode.
struct VerifyImpl;

#[cfg(test)]
pub(super) mod tests {
    use bytes::Bytes;
    use octez_riscv_data::codec;
    use octez_riscv_data::components::vector::VectorMode;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_proof::proof::Proof;
    use octez_riscv_data::merkle_proof::proof::deserialise_proof;
    use octez_riscv_data::merkle_proof::proof::serialise_proof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
    use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::ProvableExt;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::mode::utils::catch_not_found;
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_data::serialisation::serialise;

    use super::LazyRuntime;
    use super::ProveImpl;
    use super::Registry;
    use super::RegistryManifest;
    use super::VerifyImpl;

    /// A losing racer must not be left holding a runtime of its own: dropping one blocks on its
    /// worker threads, which panics inside an async context - and upgrading a read-only registry
    /// to a writeable one is done from exactly there.
    ///
    /// The contention is genuine but not guaranteed on any single run; what the test can never do
    /// is fail against an implementation that builds the runtime inside the initialiser.
    #[test]
    fn test_lazy_runtime_under_contention_in_an_async_context() {
        let outer = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .expect("Building the outer runtime should succeed");

        let lazy = LazyRuntime::default();

        outer.block_on(async {
            let askers: Vec<_> = (0..8)
                .map(|_| {
                    let lazy = lazy.clone();
                    tokio::spawn(async move { lazy.handle().map(|_| ()) })
                })
                .collect();

            for asker in askers {
                asker
                    .await
                    .expect("Asking for a handle should not panic")
                    .expect("Building the runtime should succeed");
            }
        });
    }
    use crate::commit::CommitId;
    use crate::errors::Error;
    use crate::errors::InvalidArgumentError;
    use crate::errors::OperationalError;
    use crate::key::Key;
    use crate::merkle_worker::BackgroundPersistentKeyValueStore;
    use crate::merkle_worker::BackgroundWriteableKeyValueStore;
    use crate::repo::RegistryRepo;
    use crate::storage::TestKeyValueStoreSetup;
    use crate::storage::kv_test;

    pub(super) fn setup_registry<KV: BackgroundWriteableKeyValueStore>(
        repo: KV::Repo,
    ) -> Registry<KV, Normal> {
        Registry::new(repo)
    }

    pub(super) fn setup_size_2_registry<KV: BackgroundWriteableKeyValueStore>(
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

    fn setup_prove_registry<KV: BackgroundWriteableKeyValueStore>(
        repo: KV::Repo,
    ) -> Registry<KV, Prove<'static>> {
        Registry {
            inner: ProveImpl { repo },
            databases: <Prove<'static> as VectorMode>::new(Vec::new()),
        }
    }

    fn setup_prove_size_2_registry<KV: BackgroundWriteableKeyValueStore>(
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

    fn setup_verify_registry<KV: BackgroundWriteableKeyValueStore>() -> Registry<KV, Verify> {
        Registry {
            inner: VerifyImpl,
            databases: <Verify as VectorMode>::new(Vec::new()),
        }
    }

    fn setup_verify_size_2_registry<KV: BackgroundWriteableKeyValueStore>() -> Registry<KV, Verify>
    {
        let mut registry = setup_verify_registry::<KV>();
        registry
            .resize_tick(1)
            .expect("Growing the registry should succeed.");
        registry
            .resize_tick(2)
            .expect("Growing the registry should succeed.");
        registry
    }

    fn seed_copy_move<KV: BackgroundWriteableKeyValueStore>(
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

    fn assert_pairs_present<KV: BackgroundWriteableKeyValueStore>(
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

    fn assert_pairs_absent<KV: BackgroundWriteableKeyValueStore>(
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

    pub(super) fn populate_database_with_key_value<KV: BackgroundWriteableKeyValueStore>(
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

    kv_test!(test_new, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let registry = setup_registry::<KV>(repo);
        assert!(registry.is_empty());
    });

    kv_test!(test_resize, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_get_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_copy_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_database_operations_invalid_index, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_move_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_database_operations_same_index, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_clear_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_mutating_a_copy_does_not_change_the_source_root, KV: BackgroundPersistentKeyValueStore, {
        let keys =
            [0u16, 1, 2].map(|i| Key::new(&i.to_be_bytes()).expect("Sizes less than KEY_MAX_SIZE"));

        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo.clone());

        // Setup: write values in db 0, then extend the registry and copy it to slot 2.
        // Dbs 0 and 2 are identical.
        for (i, key) in keys.iter().enumerate() {
            registry.databases[0]
                .set(key.clone(), Bytes::from(format!("value-{i}")))
                .expect("Setting a key should succeed");
        }
        registry
            .resize_tick(3)
            .expect("Growing the registry should succeed.");
        registry.copy_database(0, 2).expect("Copying should succeed");
        assert_eq!(
            registry.databases[0].hash().expect("Hashing should succeed"),
            registry.databases[2].hash().expect("Hashing should succeed"),
        );
        let commit = registry.commit().expect("Commit should succeed");

        let [lhs, root, rhs] = keys;

        // Restart from checkout so nothing is resolved.
        let mut registry =
            Registry::<KV, Normal>::checkout(repo, commit).expect("Checkout should succeed");
        registry.copy_database(0, 1).expect("Copying should succeed");
        registry.databases[1]
            .delete(root.clone())
            .expect("Deleting from the copy should succeed");
        for database in [0, 2] {
            registry.databases[database]
                .set(lhs.clone(), Bytes::from_static(b"written-in-the-source"))
                .expect("Setting a key should succeed");
        }
        registry.databases[1]
            .set(rhs.clone(), Bytes::from_static(b"only-in-the-copy"))
            .expect("Setting in the copy should succeed");

        assert_eq!(
            registry.databases[0].hash().expect("Hashing should succeed"),
            registry.databases[2].hash().expect("Hashing should succeed"),
        );
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

    kv_test!(test_copied_database_commits_shared_nodes_into_its_own_store, KV: BackgroundPersistentKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_size_2_registry::<KV>(repo.clone());

        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"alpha");

        // Copying hands the destination a copy of the source's store and the *same* in-memory
        // nodes, so both databases now owe those nodes a write. Tracking "already stored" as a
        // bare flag on the node would let the first commit satisfy the obligation and the second
        // skip it, leaving a commit that refers to nodes which never reached its own store - which
        // only shows up on checkout, once the in-memory nodes are gone.
        registry.copy_database(0, 1).expect("Copy should succeed");

        let root_commit = registry.commit().expect("Commit should succeed");
        let checked_out = Registry::<KV, Normal>::checkout(repo, root_commit)
            .expect("Checkout should succeed");

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        checked_out.databases[0].assert_database_value(&key, b"alpha");
        checked_out.databases[1].assert_database_value(&key, b"alpha");
    });

    // Not `kv_test!`s: the in-memory backend copies a commit into memory either way, so it has no
    // read-only store.
    #[cfg(rocksdb)]
    #[test]
    fn test_registry_checkout_read_only() {
        use crate::persistence_layer::PersistenceLayer;
        use crate::persistence_layer::ReadOnlyPersistenceLayer;
        use crate::storage::TestKeyValueStoreSetup;

        let (_keepalive, repo) = PersistenceLayer::setup_repo();
        let mut registry = setup_size_2_registry::<PersistenceLayer>(repo.clone());

        populate_database_with_key_value::<PersistenceLayer>(&mut registry, 0, &[1], b"alpha");
        populate_database_with_key_value::<PersistenceLayer>(&mut registry, 1, &[2], b"beta");

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");

        let root_commit = registry.commit().expect("Commit should succeed");

        let read_only = Registry::<ReadOnlyPersistenceLayer, Normal>::checkout_read_only(
            repo.clone(),
            root_commit,
        )
        .expect("Read-only checkout should succeed");

        assert_eq!(read_only.len(), 2);
        read_only.databases[0].assert_database_value(&key_a, b"alpha");
        read_only.databases[1].assert_database_value(&key_b, b"beta");
        assert_eq!(
            CommitId::from(Hash::from_foldable(&read_only)),
            root_commit,
            "A read-only checkout should reconstruct the committed root"
        );

        // Clones share the committed state, so they observe exactly the same thing.
        let clone = read_only.clone_read_only();
        clone.databases[0].assert_database_value(&key_a, b"alpha");
        assert_eq!(CommitId::from(Hash::from_foldable(&clone)), root_commit);

        let mut writeable = read_only
            .to_writeable()
            .expect("Making the registry writeable should succeed");
        assert_eq!(
            CommitId::from(Hash::from_foldable(&writeable)),
            root_commit,
            "The working copies should start from the committed state"
        );

        populate_database_with_key_value::<PersistenceLayer>(&mut writeable, 0, &[3], b"gamma");
        let derived_commit = writeable.commit().expect("Commit should succeed");
        assert_ne!(derived_commit, root_commit);

        // The read-only registry - and the commit it reads - are unaffected.
        let key_c = Key::new(&[3]).expect("Size less than KEY_MAX_SIZE");
        assert!(
            !read_only.databases[0]
                .exists(&key_c)
                .expect("Existence check should succeed"),
            "The read-only registry should not observe the working copy's writes"
        );
        assert_eq!(CommitId::from(Hash::from_foldable(&read_only)), root_commit);

        let reloaded = Registry::<PersistenceLayer, Normal>::checkout(repo, root_commit)
            .expect("Checking out the original commit should succeed");
        assert_eq!(CommitId::from(Hash::from_foldable(&reloaded)), root_commit);
    }

    #[cfg(rocksdb)]
    #[test]
    fn test_registry_checkout_read_only_missing_commit_fails() {
        use crate::persistence_layer::PersistenceLayer;
        use crate::persistence_layer::ReadOnlyPersistenceLayer;
        use crate::storage::TestKeyValueStoreSetup;

        let (_keepalive, repo) = PersistenceLayer::setup_repo();
        let missing_commit = CommitId::from(Hash::hash_bytes(b"missing-registry-commit"));

        assert!(matches!(
            Registry::<ReadOnlyPersistenceLayer, Normal>::checkout_read_only(repo, missing_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));
    }

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

    kv_test!(test_hashing_prove_registry_does_not_record_reads, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_clear_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_copy_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_database_clone_independence, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_database_ops, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_invalid_index, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_move_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_prove_new, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let registry = setup_prove_registry::<KV>(repo);
        assert!(registry.is_empty());
    });

    kv_test!(test_prove_resize, KV: BackgroundWriteableKeyValueStore, {
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
    kv_test!(test_verify_replays_prove_reads, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");

        let mut registry = setup_size_2_registry::<KV>(repo);

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

        let proof = MerkleProof::from_foldable(&prove_registry);
        let verify_registry = Registry::<KV, _>::from_proof(ProofTree::present(&proof))
            .expect("from_proof should succeed")
            .into_result();

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

    kv_test!(test_verify_clear_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_verify_copy_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_verify_database_clone_independence, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_verify_database_ops, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_verify_invalid_index, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_verify_move_database, KV: BackgroundWriteableKeyValueStore, {
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

    kv_test!(test_verify_new, KV: BackgroundWriteableKeyValueStore, {
        let registry = setup_verify_registry::<KV>();
        assert!(registry.is_empty());
    });

    kv_test!(test_verify_resize, KV: BackgroundWriteableKeyValueStore, {
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

    /// Round-trip a registry [`Proof`] via bytes.
    ///
    /// Deserialising the bytes requires two passes over the registry state:
    ///
    /// 1. The stream pass parses the raw bytes — driven by the registry's [`FromProof`]
    ///    shape — and reconstructs the proof tree. The registry it returns is structurally
    ///    correct, but it cannot be hashed: the stream deserialiser cannot capture the owned
    ///    sub-proofs that verify-mode hashing folds against (TZX-161).
    /// 2. The reconstructed proof tree is deserialised again, this time with the proof-tree
    ///    deserialiser, which does capture owned sub-proofs and therefore yields a verifiable
    ///    registry.
    ///
    /// Returns the registries produced by both passes.
    fn deserialise_proof_via_bytes<KV: BackgroundWriteableKeyValueStore>(
        proof: &Proof,
    ) -> (Registry<KV, Verify>, Registry<KV, Verify>) {
        let bytes = serialise_proof(proof);

        let (reconstructed_proof, stream_registry) =
            deserialise_proof::<codec::Bincode, Registry<KV, Verify>, _>(bytes.into_iter())
                .expect("Stream deserialisation of the proof bytes should succeed");
        assert_eq!(
            &reconstructed_proof, proof,
            "The proof reconstructed from bytes should match the original"
        );

        let verify_registry =
            Registry::<KV, Verify>::from_proof(ProofTree::present(reconstructed_proof.tree()))
                .expect("from_proof should succeed")
                .into_result();

        (stream_registry, verify_registry)
    }

    /// Recompute the registry root hash of a verify-mode registry.
    ///
    /// This relies on the databases all fitting in a single level in the tree,
    /// (up to 4 currently). Additionally, the registry as a whole cannot be
    /// blinded (ie at least one database must be used).
    ///
    /// Future full e2e tests should additionally pass the whole proof to
    /// allow registry partial hash fold to function if these don't apply.
    fn registry_root_hash_small<KV: BackgroundWriteableKeyValueStore>(
        registry: &Registry<KV, Verify>,
    ) -> Hash {
        PartialHash::from_foldable(None, registry)
            .to_hash()
            .expect("Database captures owned proof, should be able to hash")
    }

    kv_test!(test_proof_via_bytes_end_to_end, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        let key_c = Key::new(&[3]).expect("Size less than KEY_MAX_SIZE");

        let mut registry = setup_size_2_registry::<KV>(repo);
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let initial_root = Hash::from_foldable(&registry);

        // The step: read a value from database 0, write a fresh key to database 1.
        let mut prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");
        prove_registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key_a, b"foo");
        prove_registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key_b, b"bar");
        prove_registry
            .database_mut(1)
            .expect("Database should exist.")
            .set(key_c.clone(), Bytes::from_static(b"baz"))
            .expect("Setting a value should succeed");

        let final_root = Hash::from_foldable(&prove_registry);
        let proof = prove_registry.produce_proof();

        assert_eq!(
            proof.initial_state_hash(),
            initial_root,
            "The proof must encode the registry's initial state"
        );
        assert_eq!(
            proof.final_state_hash(),
            final_root,
            "The proof must carry the prove-mode final state hash"
        );

        let (_stream_registry, mut verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);

        assert_eq!(
            registry_root_hash_small(&verify_registry),
            initial_root,
            "The verify-mode registry must start from the initial state hash"
        );

        // Replay the step.
        verify_registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key_a, b"foo");
        verify_registry
            .database(1)
            .expect("Database should exist.")
            .assert_database_value(&key_b, b"bar");
        verify_registry
            .database_mut(1)
            .expect("Database should exist.")
            .set(key_c, Bytes::from_static(b"baz"))
            .expect("Setting a value should succeed");

        assert_eq!(
            registry_root_hash_small(&verify_registry),
            proof.final_state_hash(),
            "Replaying the step in verify mode must reach the proof's final state hash"
        );
    });

    /// Count how many of a 16-database registry's databases are *present* (revealed as a node)
    /// in the proof, as opposed to blinded or absorbed into a blinded ancestor.
    ///
    /// Assumes the arity-4, depth-2 contents layout of a 16-element vector: root `[length, contents]`, contents
    /// node has 4 group children, each group node has 4 database children.
    fn count_present_databases_of_16(tree: &MerkleProof) -> usize {
        let MerkleProof::Node(root) = tree else {
            return 0; // whole registry blinded
        };
        let contents = &root.children[1];
        let MerkleProof::Node(contents) = contents else {
            return 0; // all databases absorbed into a blinded contents subtree
        };
        contents
            .children
            .iter()
            .map(|group| match group {
                MerkleProof::Node(group) => group
                    .children
                    .iter()
                    .filter(|db| matches!(db, MerkleProof::Node(_)))
                    .count(),
                _ => 0, // blinded group
            })
            .sum()
    }

    // Touching a single database of a 16-database registry must yield the *minimal* proof: only the
    // touched database is present, its merkle-tree siblings are blinded, and every other database
    // is absent (absorbed into a blinded ancestor). The proof must still round-trip and let the
    // verifier replay the touched read.
    kv_test!(test_touch_one_database_yields_minimal_proof, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut registry = setup_registry::<KV>(repo);
        for n in 1..=16 {
            registry.resize_tick(n).expect("resize should succeed");
        }
        for i in 0..16 {
            populate_database_with_key_value::<KV>(&mut registry, i, &[i as u8], b"val");
        }

        let root_hash = Hash::from_foldable(&registry);

        for db_n in 0..16 {
            let key_n = Key::new(&[db_n]).unwrap();
            let prove = registry
                .try_start_proof()
                .expect("Converting to prove mode should succeed");
            // Touch only database 'db_n'.
            prove
                .database(db_n as usize)
                .expect("Database should exist.")
                .assert_database_value(&key_n, b"val");

            let proof = prove.produce_proof();

            // Only database 'db_n' is present in the proof.
            assert_eq!(
                count_present_databases_of_16(proof.tree()),
                1,
                "touching one database must reveal only that database, got proof:\n{:#?}",
                proof.tree()
            );

            // The proof round-trips and the verifier can replay the touched read.
            let (_stream, verify) = deserialise_proof_via_bytes::<KV>(&proof);

            let hash = PartialHash::from_foldable(None, &verify).to_hash();
            assert_eq!(None, hash, "hashing minimal proof requires proof arg");

            let hash = PartialHash::from_foldable(Some(proof.into_tree()), &verify)
                .to_hash()
                .expect("partial hash of registry with proof succeeds");

            assert_eq!(root_hash, hash, "Verify root hash matches normal mode");

            verify
                .database(db_n as usize)
                .expect("Database should exist.")
                .assert_database_value(&key_n, b"val");
        }
    });

    // A step that touches no database must still produce a usable proof of a non-empty registry.
    // Such a proof is fully blinded — the whole registry collapses to a single blind leaf carrying
    // its root hash — rather than exposing each database (which would yield a present contents
    // subtree with no length node, rejected by the deserialiser).
    kv_test!(test_untouched_nonempty_registry_proof_is_fully_blinded, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let mut registry = setup_size_2_registry::<KV>(repo);
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let initial_root = Hash::from_foldable(&registry);

        // The step touches nothing.
        let prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");
        let proof = prove_registry.produce_proof();

        // The whole registry is a single blind leaf of its root hash.
        assert!(
            matches!(
                proof.tree(),
                MerkleProof::Leaf(MerkleProofLeaf::Blind(hash)) if *hash == initial_root
            ),
            "an untouched registry proof must be a single blind leaf of its root hash, got {:?}",
            proof.tree()
        );

        // It round-trips through bytes (previously rejected with `LengthAbsentButItemsPresent`) and
        // the resulting verify-mode registry still hashes to the initial root.
        let (_stream_registry, verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);
        let verify_root = PartialHash::from_foldable(Some(proof.tree().clone()), &verify_registry)
            .to_hash()
            .expect("a fully blinded registry must still hash to its root");
        assert_eq!(
            verify_root, initial_root,
            "verify-mode root hash of a fully blinded registry must match the initial root"
        );
    });

    // TODO (TZX-161): once the stream deserialiser can capture owned proofs, the first pass
    // should become verifiable on its own and this test should be updated.
    kv_test!(test_proof_via_bytes_requires_second_deserialisation, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut registry = setup_size_2_registry::<KV>(repo);
        // Database 0 holds several keys so that reading one of them traverses nodes whose
        // data stays unread — those fields are absent from the proof and can only be
        // resolved against the captured owned proof.
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 0, &[4], b"unread");
        populate_database_with_key_value::<KV>(&mut registry, 0, &[5], b"unread");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let db_0_initial_hash = registry
            .database(0)
            .expect("Database should exist.")
            .hash()
            .expect("Hashing should succeed.");

        // The step only reads one key from database 0.
        let prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");
        prove_registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key_a, b"foo");

        let proof = prove_registry.produce_proof();
        let (stream_registry, verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);

        // The stream-pass registry is structurally correct: recorded reads replay fine.
        stream_registry
            .database(0)
            .expect("Database should exist.")
            .assert_database_value(&key_a, b"foo");

        // ... but it cannot be hashed, because the owned proof was not captured.
        let stream_hash_attempt = catch_not_found(|| {
            stream_registry
                .database(0)
                .expect("Database should exist.")
                .hash()
        });
        assert!(
            stream_hash_attempt.is_err(),
            "Hashing a stream-deserialised database should fail: the owned proof was not captured"
        );

        // The second pass captured the owned proof, so the hash is available and matches the
        // initial state.
        assert_eq!(
            verify_registry
                .database(0)
                .expect("Database should exist.")
                .hash()
                .expect("Hashing should succeed."),
            db_0_initial_hash,
            "The proof-tree pass must recover the partially-read database's hash"
        );
        assert_eq!(
            registry_root_hash_small(&verify_registry),
            proof.initial_state_hash(),
            "The proof-tree pass must recover the registry's initial root hash"
        );
    });

    kv_test!(test_proof_database_hash_queries_replay_in_verify_mode, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let key_c = Key::new(&[3]).expect("Size less than KEY_MAX_SIZE");

        let mut registry = setup_size_2_registry::<KV>(repo);
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let mut prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");

        // Query the hash of database 0 without accessing it otherwise.
        let prove_hash_untouched = prove_registry
            .database(0)
            .expect("Database should exist.")
            .hash()
            .expect("Hashing should succeed.");

        // Mutate database 1, then query its hash mid-step.
        prove_registry
            .database_mut(1)
            .expect("Database should exist.")
            .set(key_c.clone(), Bytes::from_static(b"baz"))
            .expect("Setting a value should succeed");
        let prove_hash_after_write = prove_registry
            .database(1)
            .expect("Database should exist.")
            .hash()
            .expect("Hashing should succeed.");

        let proof = prove_registry.produce_proof();
        let (_stream_registry, mut verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);

        // Replay: the same queries must give the same answers.
        assert_eq!(
            verify_registry
                .database(0)
                .expect("Database should exist.")
                .hash()
                .expect("Hashing should succeed."),
            prove_hash_untouched,
            "The verifier must reproduce the hash of the untouched database"
        );

        verify_registry
            .database_mut(1)
            .expect("Database should exist.")
            .set(key_c, Bytes::from_static(b"baz"))
            .expect("Setting a value should succeed");
        assert_eq!(
            verify_registry
                .database(1)
                .expect("Database should exist.")
                .hash()
                .expect("Hashing should succeed."),
            prove_hash_after_write,
            "The verifier must reproduce the mid-step hash of the written database"
        );

        assert_eq!(
            registry_root_hash_small(&verify_registry),
            proof.final_state_hash(),
            "Replaying the step in verify mode must reach the proof's final state hash"
        );
    });

    kv_test!(
        test_proof_copy_database_preserves_hash_invariants, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let mut registry = setup_size_2_registry::<KV>(repo);
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let initial_root = Hash::from_foldable(&registry);

        let mut prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");
        prove_registry
            .copy_database(0, 1)
            .expect("Copying should succeed");
        let final_root = Hash::from_foldable(&prove_registry);

        // Reference: the same step in Normal mode.
        registry
            .copy_database(0, 1)
            .expect("Copying should succeed");
        assert_eq!(
            Hash::from_foldable(&registry),
            final_root,
            "Prove-mode copy must produce the Normal-mode final hash"
        );

        let proof = prove_registry.produce_proof();
        assert_eq!(
            proof.initial_state_hash(),
            initial_root,
            "The proof must encode the registry's true initial state"
        );
        assert_eq!(proof.final_state_hash(), final_root);

        // Replaying the copy in verify mode must reach the final state hash.
        let (_stream_registry, mut verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);
        assert_eq!(registry_root_hash_small(&verify_registry), initial_root);
        verify_registry
            .copy_database(0, 1)
            .expect("Copying should succeed");
        assert_eq!(
            registry_root_hash_small(&verify_registry),
            proof.final_state_hash(),
            "Replaying the copy in verify mode must reach the proof's final state hash"
        );
    });

    kv_test!(
        test_proof_move_database_preserves_hash_invariants, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let mut registry = setup_size_2_registry::<KV>(repo);
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let initial_root = Hash::from_foldable(&registry);

        let mut prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");
        prove_registry
            .move_database(0, 1)
            .expect("Moving should succeed");
        let final_root = Hash::from_foldable(&prove_registry);

        // Reference: the same step in Normal mode.
        registry
            .move_database(0, 1)
            .expect("Moving should succeed");
        assert_eq!(
            Hash::from_foldable(&registry),
            final_root,
            "Prove-mode move must produce the Normal-mode final hash"
        );

        let proof = prove_registry.produce_proof();
        assert_eq!(
            proof.initial_state_hash(),
            initial_root,
            "The proof must encode the registry's true initial state"
        );
        assert_eq!(proof.final_state_hash(), final_root);

        // Replaying the move in verify mode must reach the final state hash.
        let (_stream_registry, mut verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);
        assert_eq!(registry_root_hash_small(&verify_registry), initial_root);
        verify_registry
            .move_database(0, 1)
            .expect("Moving should succeed");
        assert_eq!(
            registry_root_hash_small(&verify_registry),
            proof.final_state_hash(),
            "Replaying the move in verify mode must reach the proof's final state hash"
        );
    });

    kv_test!(
        test_proof_clear_database_preserves_hash_invariants, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();

        let mut registry = setup_size_2_registry::<KV>(repo);
        populate_database_with_key_value::<KV>(&mut registry, 0, &[1], b"foo");
        populate_database_with_key_value::<KV>(&mut registry, 1, &[2], b"bar");

        let initial_root = Hash::from_foldable(&registry);

        let mut prove_registry = registry
            .try_start_proof()
            .expect("Converting to prove mode should succeed");
        prove_registry
            .clear_database(1)
            .expect("Clearing should succeed");
        let final_root = Hash::from_foldable(&prove_registry);

        // Reference: the same step in Normal mode.
        registry
            .clear_database(1)
            .expect("Clearing should succeed");
        assert_eq!(
            Hash::from_foldable(&registry),
            final_root,
            "Prove-mode clear must produce the Normal-mode final hash"
        );

        let proof = prove_registry.produce_proof();
        assert_eq!(
            proof.initial_state_hash(),
            initial_root,
            "The proof must encode the registry's true initial state"
        );
        assert_eq!(proof.final_state_hash(), final_root);

        // Replaying the clear in verify mode must reach the final state hash.
        let (_stream_registry, mut verify_registry) = deserialise_proof_via_bytes::<KV>(&proof);
        assert_eq!(registry_root_hash_small(&verify_registry), initial_root);
        verify_registry
            .clear_database(1)
            .expect("Clearing should succeed");
        assert_eq!(
            registry_root_hash_small(&verify_registry),
            proof.final_state_hash(),
            "Replaying the clear in verify mode must reach the proof's final state hash"
        );
    });

    kv_test!(test_durable_storage_end_to_end, KV: BackgroundPersistentKeyValueStore,
    [
        generated in <crate::test_helpers::registry::RegistryOperationView as crate::test_helpers::OperationView>::operations_commit_checkout_strategy(1usize..100, 0.1)
    ],
    {
        // Every test iteration expects an empty repo, so not setting it in a `setup` block.
        // This is because a repo preserves registry commits from previous test runs, resulting
        // in test failures when checking out a commit which isn't expected to exist succeeds.
        let (_keepalive, repo) = KV::setup_repo();

        let (keys, values, ops_a, ops_b) = generated;

        // Pick an operations vector so each backend exercises a different
        // `CommitCheckoutRoundtrip` placement against the same base operations.
        let ops = match KV::BACKEND {
            crate::storage::Backend::Persistent => ops_a,
            crate::storage::Backend::InMemory => ops_b,
        };
        let operations = crate::test_helpers::registry::make_registry_operations(
            std::num::NonZeroUsize::new(1).expect("1 > 0"),
            keys,
            values,
            ops,
        );

        crate::test_helpers::registry::run_and_prove_registry_operations::<KV>(
            repo, operations,
        )
    });

    kv_test!(test_registry_proof_regression, KV: BackgroundPersistentKeyValueStore, {
        use goldenfile::Mint;

        use crate::test_helpers::REGRESSION_EXPECTED_DIR;
        use crate::test_helpers::REGRESSION_INPUTS_DIR;
        use crate::test_helpers::registry::RegistryOperation;
        use crate::test_helpers::registry::run_and_prove_registry_operations;

        let mut inputs: Vec<_> = std::fs::read_dir(REGRESSION_INPUTS_DIR)
            .unwrap_or_else(|e| panic!("reading {REGRESSION_INPUTS_DIR} should succeed: {e}"))
            .map(|e| {
                e.unwrap_or_else(|e| {
                    panic!("reading {REGRESSION_INPUTS_DIR} entry should succeed: {e}")
                })
            })
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("registry_") && name.ends_with(".input"))
            })
            .collect();
        inputs.sort_by_key(|e| e.path());

        let mut mint = Mint::new(REGRESSION_EXPECTED_DIR);

        for input in inputs {
            let path = input.path();
            let stem = path
                .file_stem()
                .expect("input path must have a file stem")
                .to_str()
                .expect("input path must be UTF-8")
                .to_string();

            let file = std::fs::File::open(&path).expect("opening input file should succeed");
            let ops: Vec<RegistryOperation> =
                serde_json::from_reader(file).expect("decoding JSON input should succeed");

            let (_keepalive, repo) = KV::setup_repo();
            let steps = run_and_prove_registry_operations::<KV>(repo, ops);

            let mut golden = mint
                .new_goldenfile(format!("{stem}.proof-trace"))
                .expect("opening goldenfile should succeed");
            serde_json::to_writer_pretty(&mut golden, &steps)
                .expect("writing goldenfile should succeed");
        }
    });
}

#[cfg(rocksdb)]
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
