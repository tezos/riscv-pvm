// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Combined Database interface
//!
//! This module provides a database type to unify operations between the Merkle layer and the
//! key-value store.

mod traced_database;
pub(crate) mod value_ref;

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::Arc;

use bytes::BufMut;
use bytes::Bytes;
use octez_riscv_data::codec;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::ProvableExt;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;
use tokio::runtime::Handle;

#[cfg(test)]
pub(crate) use self::traced_database::Trace;
#[cfg(any(test, rocksdb_test_utils))]
pub(crate) use self::traced_database::TracedDatabase;
use crate::avl::resolver::ProveNodeId;
use crate::avl::tree::Tree;
use crate::commit::CommitId;
use crate::database::value_ref::AsRefValueRef;
use crate::database::value_ref::ValueRef;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::merkle_layer::MerkleLayer;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::merkle_worker::BackgroundReadableKeyValueStore;
use crate::merkle_worker::BackgroundWriteableKeyValueStore;
use crate::merkle_worker::CommittedRoot;
use crate::merkle_worker::MerkleHandle;
use crate::merkle_worker::MerkleWorker;
use crate::merkle_worker::TreeBackedKeyValueStore;
pub use crate::repo::DirectoryManager;
use crate::storage::PersistentKeyValueStore;
use crate::storage::ReadOnlyKeyValueStore;
use crate::storage::ReadableKeyValueStore;
use crate::storage::StoreOptions;

/// The maximum possible length of a value in durable storage.
pub const MAX_VALUE_SIZE: usize =
    const { 64_usize.checked_shl(20).expect("usize overflow on 64MiB") };

/// An isolated key-space, independent from other [`Database`]s, on which database operations can
/// be performed, e.g. read, write, delete.
///
/// This structure unifies the key-value store and Merkle layer to allow for persistent storage
/// alongside a representation which can provide a root hash.
#[perfect_derive(Clone)]
#[repr(transparent)]
pub struct Database<KV: ReadableKeyValueStore, M: Mode> {
    inner: M::Select<DatabaseTemplate<KV>>,
}

impl<KV: TreeBackedKeyValueStore> Database<KV, Normal> {
    /// Construct a new, empty database backed by `repo`.
    ///
    /// The returned database owns an isolated working state. Mutations are applied immediately to
    /// that working state and are not persisted as a named snapshot until [`Database::commit`] is
    /// called.
    pub fn try_new(handle: &Handle, repo: &KV::Repo) -> Result<Self, OperationalError>
    where
        KV: BackgroundWriteableKeyValueStore,
    {
        let persistent = KV::new(repo)?;
        let persistent = Arc::new(persistent);

        let merkle = MerkleWorker::new(handle, persistent.clone());

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// For tests, construct a database from its raw parts.
    #[cfg(test_utils)]
    pub fn new_raw(persistent: Arc<KV>, merkle: MerkleWorker<KV>) -> Self {
        Database {
            inner: NormalImpl { persistent, merkle },
        }
    }

    /// Restore a database from a previously committed snapshot.
    ///
    /// The checked-out database is isolated from the committed snapshot: subsequent mutations are
    /// applied to a working copy, not to the committed state on disk.
    pub fn checkout(handle: &Handle, repo: &KV::Repo, commit_id: CommitId) -> Result<Self, Error>
    where
        KV: BackgroundPersistentKeyValueStore,
    {
        let persistent = KV::checkout(repo, &commit_id)?;
        let persistent = Arc::new(persistent);

        let merkle = MerkleWorker::checkout(handle, persistent.clone(), commit_id)?;

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// Create a cheap clone of the current working state.
    ///
    /// The clone shares existing state efficiently with the original database and diverges on
    /// subsequent mutation. Neither database persists its state to a repository commit until
    /// [`Database::commit`] is called.
    pub fn try_clone_with(&self, handle: &Handle, repo: &KV::Repo) -> Result<Self, OperationalError>
    where
        KV: BackgroundWriteableKeyValueStore,
    {
        let persistent = self.inner.persistent.try_clone(repo)?;
        let persistent = Arc::new(persistent);

        let merkle = self.inner.merkle.clone_with(handle, persistent.clone())?;

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }

    /// Commit the current database state to the repository and return its commit identifier.
    ///
    /// The returned [`CommitId`] is derived from the Merkle root hash of the current working
    /// state. The commit can later be restored with [`Database::checkout`].
    pub fn commit(&self, repo: &KV::Repo) -> Result<crate::commit::CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        // Read before the nodes are written, so each is recorded as belonging to the commit being
        // made rather than to whatever came before it.
        let commit_options = StoreOptions::default()
            .without_node_data()
            .written_at(self.inner.persistent.next_commit_seq(repo)?);
        let commit_id = self.inner.merkle.commit(commit_options)?;
        self.inner.persistent.commit(repo, &commit_id)?;

        Ok(commit_id)
    }
}

/// A database reading a commit in place, through a [`ReadOnlyKeyValueStore`].
///
/// The mutating operations - [`Database::set`], [`Database::write`], [`Database::delete`] and
/// [`Database::commit`] - are bounded on a writeable store, so they do not exist here. Use
/// [`Database::to_writeable`] if they are required.
///
/// Such a database holds no Merkle tree: its root hash cannot change and is already known, so the
/// store pins [`ReadableKeyValueStore::Merkle`] to a [`CommittedRoot`]. Nothing here spawns a
/// worker thread, and so nothing here needs an async runtime.
///
/// [`ReadOnlyKeyValueStore`]: crate::storage::ReadOnlyKeyValueStore
impl<KV: ReadOnlyKeyValueStore> Database<KV, Normal> {
    /// Read a previously committed snapshot, without making a working copy.
    pub fn checkout_read_only(repo: &KV::Repo, commit_id: CommitId) -> Result<Self, Error> {
        let persistent = Arc::new(KV::checkout_read_only(repo, &commit_id)?);

        Ok(Database {
            inner: NormalImpl {
                persistent,
                merkle: CommittedRoot::from(commit_id),
            },
        })
    }

    /// Create another handle on the same committed state.
    ///
    /// Copies nothing and cannot fail - the store handle is shared and the root hash is a value -
    /// unlike [`Database::try_clone_with`], which copies a working state.
    pub fn clone_read_only(&self) -> Self {
        Database {
            inner: NormalImpl {
                persistent: self.inner.persistent.clone(),
                merkle: self.inner.merkle,
            },
        }
    }

    /// Copy the committed state into a working state, which can be modified.
    ///
    /// The commit is left untouched and remains readable through `self`. This is where a Merkle
    /// tree and its worker first appear: the working copy can be written to, so its root hash has
    /// to be tracked.
    pub fn to_writeable(
        &self,
        handle: &Handle,
        repo: &KV::Repo,
    ) -> Result<Database<KV::Writeable, Normal>, OperationalError>
    where
        KV::Writeable: BackgroundPersistentKeyValueStore,
    {
        let persistent = Arc::new(self.inner.persistent.to_writeable(repo)?);

        let merkle =
            MerkleWorker::checkout(handle, persistent.clone(), self.inner.merkle.commit_id())?;

        Ok(Database {
            inner: NormalImpl { persistent, merkle },
        })
    }
}

impl<KV: BackgroundReadableKeyValueStore, M: DatabaseMode> Database<KV, M> {
    /// Returns true if the provided key exists in the database, false if it does not.
    pub fn exists(&self, key: &Key) -> Result<bool, Error> {
        match self.get(key) {
            Ok(_) => Ok(true),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => Ok(false),
            Err(other_error) => Err(other_error),
        }
    }

    /// Obtain, and possibly calculate, the root hash of the database.
    pub fn hash(&self) -> Result<Hash, OperationalError> {
        M::hash(self)
    }

    /// Read a portion of the value associated with the provided key. The read data will be written
    /// into `output`. `offset` specifies from where in the associated value to start reading.
    ///
    /// Returns the number of bytes read.
    ///
    /// Fails if:
    ///  - The number of bytes to read is larger than [`MAX_FILE_CHUNK_SIZE`].
    ///  - The key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub fn read(&self, key: &Key, offset: usize, mut output: impl BufMut) -> Result<usize, Error> {
        let slice = self.read_bytes(key, offset, output.remaining_mut())?;
        let source_slice = slice.as_ref();
        output.put_slice(source_slice);
        Ok(source_slice.len())
    }

    /// Read a portion of the value associated with the provided key. The read data will be copied
    /// into the return value. `offset` specifies from where in the associated value to start
    /// reading.
    ///
    /// Fails if:
    ///  - The number of bytes to read is larger than [`MAX_FILE_CHUNK_SIZE`].
    ///  - The key does not exist.
    ///  - The offset is larger than the length of the associated value.
    pub fn read_bytes(
        &self,
        key: &Key,
        offset: usize,
        max_bytes: usize,
    ) -> Result<impl AsRef<[u8]>, Error> {
        if max_bytes > MAX_FILE_CHUNK_SIZE {
            Err(InvalidArgumentError::IoRequestTooLarge)?;
        }

        let value = self.get(key)?;
        let value_length = value.len();
        if offset > value_length {
            Err(InvalidArgumentError::OffsetTooLarge)?;
        }

        let end = offset.saturating_add(max_bytes).min(value_length);
        let mut buf = vec![0u8; end - offset];
        let written = value.read(offset, &mut buf);
        debug_assert_eq!(written, buf.len());

        Ok(buf)
    }

    /// Retrieve the length of the value associated with the provided key.
    ///
    /// Fails if:
    ///  - The key does not exist in the database.
    pub fn value_length(&self, key: &Key) -> Result<usize, Error> {
        Ok(self.get(key)?.len())
    }

    /// Retrieve the value associated with the provided key.
    ///
    /// Fails if:
    ///  - The key does not exist in the database.
    fn get(&self, key: &Key) -> Result<impl ValueRef, Error> {
        M::get(self, key)
    }
}

/// Operations which modify the database, and so require a store that can be written to.
impl<KV: BackgroundWriteableKeyValueStore, M: DatabaseMode> Database<KV, M> {
    /// Inserts the value associated with the provided key, replacing any data already associated
    /// with the key.
    ///
    /// Fails if:
    ///  - The number of bytes to write is larger than [`MAX_FILE_CHUNK_SIZE`].
    pub fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error> {
        if data.len() > MAX_FILE_CHUNK_SIZE {
            Err(InvalidArgumentError::IoRequestTooLarge)?;
        }

        const {
            assert!(
                MAX_FILE_CHUNK_SIZE <= MAX_VALUE_SIZE,
                "It must not be possible to set a value larger than MAX_VALUE_SIZE in one go."
            )
        };

        M::set(self, key, data)
    }

    /// Modify the value associated with the provided key. `offset` specifies from where to start
    /// writing within the associated value, appending if it is equal to the length. Non-existent
    /// keys have the implicit length 0, so they are writeable.
    ///
    /// Non-zero offsets require checking the existence and length of an existing value, making
    /// them more expensive.
    ///
    /// Fails if:
    ///  - The number of bytes to write is larger than [`MAX_FILE_CHUNK_SIZE`].
    ///  - The size of the value after the write would exceed [`MAX_VALUE_SIZE]`.
    ///  - The offset is non-zero and the key does not exist.
    ///  - The offset is larger than the length of the associated value.
    ///  - The offset plus the length of the data would overflow.
    pub fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error> {
        if data.len() > MAX_FILE_CHUNK_SIZE {
            Err(InvalidArgumentError::IoRequestTooLarge)?;
        }

        if offset.saturating_add(data.len()) > MAX_VALUE_SIZE {
            Err(InvalidArgumentError::ValueSizeTooLarge)?;
        }

        M::write(self, key, offset, data)
    }

    /// Remove a key from the database.
    ///
    /// Deleting a missing key succeeds and leaves the database unchanged.
    pub fn delete(&mut self, key: Key) -> Result<(), OperationalError> {
        M::delete(self, key)
    }
}

impl<KV: ReadableKeyValueStore> Foldable<HashFold> for Database<KV, Normal> {
    fn fold(&self, _builder: HashFold) -> Hash {
        self.inner.merkle.hash().expect("Hashing should not fail")
    }
}

impl<KV: ReadableKeyValueStore> Foldable<HashFold> for Database<KV, Prove<'_>> {
    fn fold(&self, _builder: HashFold) -> Hash {
        self.inner.merkle.hash()
    }
}

impl<KV: ReadableKeyValueStore> Foldable<MerkleProofFold> for Database<KV, Prove<'_>> {
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        self.inner.merkle.fold(builder)
    }
}

impl<KV: ReadableKeyValueStore> Foldable<PartialHashFold> for Database<KV, Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        self.inner.merkle.fold(builder)
    }
}

impl<'normal, KV: BackgroundReadableKeyValueStore> ProvableExt<'normal, 'static, OperationalError>
    for Database<KV, Normal>
{
    type Prover = Database<KV, Prove<'static>>;

    fn try_start_proof(&'normal self) -> Result<Self::Prover, OperationalError> {
        let NormalImpl { persistent, merkle } = &self.inner;

        let merkle = merkle.start_proof(persistent.clone())?;

        Ok(Database {
            inner: ProveImpl { merkle },
        })
    }
}

/// Modal template for the [`Database`]
///
/// This is used to select the appropriate implementation for the mode.
struct DatabaseTemplate<KV: ReadableKeyValueStore>(PhantomData<KV>, Infallible);

impl<KV: ReadableKeyValueStore> Modal for DatabaseTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = ProveImpl<KV>;

    type Verify = VerifyImpl<KV>;
}

/// Modes that support the operational API exposed by [`Database`].
pub trait DatabaseMode: Mode {
    /// See [`Database::set`]
    fn set<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        value: Bytes,
    ) -> Result<(), Error>;

    /// See [`Database::write`]
    fn write<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        offset: usize,
        value: Bytes,
    ) -> Result<usize, Error>;

    /// See [`Database::delete`]
    fn delete<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
    ) -> Result<(), OperationalError>;

    /// See [`Database::hash`]
    fn hash<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
    ) -> Result<Hash, OperationalError>;

    /// See [`Database::get`]
    fn get<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<impl ValueRef, Error>;
}

impl DatabaseMode for Normal {
    fn get<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<impl ValueRef, Error> {
        let as_ref = this.inner.persistent.get(key.as_ref())?;
        Ok(AsRefValueRef(as_ref))
    }

    fn set<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        data: Bytes,
    ) -> Result<(), Error> {
        this.inner.persistent.set(&key, &data)?;
        this.inner.merkle.set(key, data)?;
        Ok(())
    }

    fn write<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        offset: usize,
        data: Bytes,
    ) -> Result<usize, Error> {
        let written = data.len();
        this.inner.persistent.write(&key, offset, &data)?;
        this.inner.merkle.write(key, offset, data)?;
        Ok(written)
    }

    fn delete<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
    ) -> Result<(), OperationalError> {
        this.inner.persistent.delete(key.as_ref())?;
        this.inner.merkle.delete(key)?;
        Ok(())
    }

    fn hash<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
    ) -> Result<Hash, OperationalError> {
        this.inner.merkle.hash()
    }
}

/// Registry implementation for the [`Database`] mode
///
/// The Merkle representation is selected by the store: a writeable store brings a
/// [`MerkleWorker`] owning a live tree, a read-only one brings only a [`CommittedRoot`].
struct NormalImpl<KV: ReadableKeyValueStore> {
    persistent: Arc<KV>,
    merkle: KV::Merkle,
}

impl<KV: ReadableKeyValueStore> Database<KV, Prove<'static>> {
    /// An empty prove-mode database backed by the given persistence layer.
    pub(crate) fn empty(persistence: Arc<KV>) -> Self
    where
        KV: ReadableKeyValueStore,
    {
        Database {
            inner: ProveImpl {
                merkle: <MerkleLayer<KV, Prove<'static>>>::empty(persistence),
            },
        }
    }

    /// Clone of the current working tree.
    pub(crate) fn clone_working_tree(&self) -> Tree<ProveNodeId> {
        self.inner.merkle.clone_working_tree()
    }

    /// Replace the working tree with `new`, returning the outgoing tree.
    ///
    /// See [`MerkleLayer::replace_working_tree`].
    pub(crate) fn replace_working_tree(&mut self, new: Tree<ProveNodeId>) -> Tree<ProveNodeId> {
        self.inner.merkle.replace_working_tree(new)
    }
}

impl<KV: ReadableKeyValueStore> Database<KV, Verify> {
    /// An empty verify-mode database.
    pub(crate) fn empty() -> Self {
        Database {
            inner: VerifyImpl {
                merkle: <MerkleLayer<KV, Verify>>::empty(),
            },
        }
    }
}

impl<KV: ReadableKeyValueStore> FromProof for Database<KV, Verify> {
    fn from_proof<Proof: Deserialiser<Codec = codec::Bincode>>(
        proof: Proof,
    ) -> SuspendedResult<Proof, Self> {
        let suspended = <MerkleLayer<KV, Verify> as FromProof>::from_proof(proof)?;
        Ok(suspended.map(|merkle| Database {
            inner: VerifyImpl { merkle },
        }))
    }
}

/// Verify-mode implementation for the [`Database`].
#[perfect_derive(Clone)]
struct VerifyImpl<KV> {
    merkle: MerkleLayer<KV, Verify>,
}

impl DatabaseMode for Verify {
    fn get<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<impl ValueRef, Error> {
        // Wraps the return type of [`MerkleLayer::get`] to allow returning the data as an
        // [`impl ValueRef`] without allocating.
        struct Wrapper<'a>(&'a octez_riscv_data::components::bytes::Bytes<Verify>);

        impl ValueRef for Wrapper<'_> {
            fn len(&self) -> usize {
                self.0.len()
            }

            fn read(&self, offset: usize, buf: &mut [u8]) -> usize {
                let len = self.0.len();
                if offset >= len {
                    return 0;
                }
                let end = offset.saturating_add(buf.len()).min(len);
                let read_len = end - offset;
                let src = self.0.partial_slice(offset..end);
                buf[..read_len].copy_from_slice(src);
                read_len
            }
        }

        let bytes = this
            .inner
            .merkle
            .get(key)?
            .ok_or(InvalidArgumentError::KeyNotFound)?;

        Ok(Wrapper(bytes))
    }

    fn set<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        data: Bytes,
    ) -> Result<(), Error> {
        this.inner.merkle.set(&key, &data)?;
        Ok(())
    }

    fn write<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        offset: usize,
        data: Bytes,
    ) -> Result<usize, Error> {
        let written = data.len();
        this.inner.merkle.write(&key, offset, &data)?;
        Ok(written)
    }

    fn delete<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
    ) -> Result<(), OperationalError> {
        this.inner.merkle.delete(&key)
    }

    fn hash<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
    ) -> Result<Hash, OperationalError> {
        Ok(this.inner.merkle.hash())
    }
}

/// Prove-mode implementation for the [`Database`].
struct ProveImpl<KV> {
    merkle: MerkleLayer<KV, Prove<'static>>,
}

impl DatabaseMode for Prove<'static> {
    fn get<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
        key: &Key,
    ) -> Result<impl ValueRef, Error> {
        // Defers reading the bytes (and thereby recording an access against the proof) until the
        // caller actually reads from the value, and only records the requested range.
        // `value_length` and `exists` only invoke `len`, which queries the length without
        // recording a byte-range read.
        struct Wrapper<'a>(&'a octez_riscv_data::components::bytes::Bytes<Prove<'static>>);

        impl ValueRef for Wrapper<'_> {
            fn len(&self) -> usize {
                self.0.len()
            }

            fn read(&self, offset: usize, buf: &mut [u8]) -> usize {
                self.0.read(offset, buf)
            }
        }

        let bytes = this
            .inner
            .merkle
            .get(key)?
            .ok_or(InvalidArgumentError::KeyNotFound)?;

        Ok(Wrapper(bytes))
    }

    fn set<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        data: Bytes,
    ) -> Result<(), Error> {
        this.inner.merkle.set(&key, &data)?;
        Ok(())
    }

    fn write<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
        offset: usize,
        data: Bytes,
    ) -> Result<usize, Error> {
        let written = data.len();
        this.inner.merkle.write(&key, offset, &data)?;
        Ok(written)
    }

    fn delete<KV: BackgroundWriteableKeyValueStore>(
        this: &mut Database<KV, Self>,
        key: Key,
    ) -> Result<(), OperationalError> {
        this.inner.merkle.delete(&key)
    }

    fn hash<KV: BackgroundReadableKeyValueStore>(
        this: &Database<KV, Self>,
    ) -> Result<Hash, OperationalError> {
        Ok(this.inner.merkle.hash())
    }
}

#[cfg(test)]
impl<KV: BackgroundReadableKeyValueStore, M: DatabaseMode> Database<KV, M> {
    /// Assert that a database contains the expected value for a given key.
    pub(crate) fn assert_database_value(&self, key: &Key, expected: &[u8]) {
        let mut stored = vec![0; expected.len()];
        let read = self
            .read(key, 0, stored.as_mut_slice())
            .expect("Persisted value should exist");
        assert_eq!(read, stored.len());
        assert_eq!(stored.as_slice(), expected);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use bytes::Bytes;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
    use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::ProvableExt;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::mode::utils::catch_not_found;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;
    use tokio::runtime::Handle;

    use super::Database;
    use super::traced_database::TracedDatabase;
    use crate::avl::tree::Tree;
    use crate::database::MAX_VALUE_SIZE;
    use crate::errors::Error;
    use crate::errors::InvalidArgumentError;
    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;
    use crate::merkle_layer::MerkleLayer;
    use crate::merkle_worker::BackgroundPersistentKeyValueStore;
    use crate::merkle_worker::BackgroundWriteableKeyValueStore;
    use crate::merkle_worker::MerkleHandle;
    use crate::storage::TestKeyValueStoreSetup;
    use crate::storage::WriteableKeyValueStore;
    use crate::storage::kv_test;

    fn new_database<KV: BackgroundWriteableKeyValueStore>(
        handle: &Handle,
        repo: &KV::Repo,
    ) -> TracedDatabase<KV> {
        TracedDatabase::try_new(handle, repo).expect("Creating a test database should succeed")
    }

    fn new_verify_database<KV: WriteableKeyValueStore + TestKeyValueStoreSetup>(
        repo: &KV::Repo,
    ) -> TracedDatabase<KV, Verify> {
        TracedDatabase::<KV, Verify>::new_verify(repo)
    }

    fn new_prove_database<KV: WriteableKeyValueStore>(
        persistence: Arc<KV>,
    ) -> TracedDatabase<KV, Prove<'static>> {
        TracedDatabase::from(Database {
            inner: super::ProveImpl {
                merkle: MerkleLayer::from_prove_tree(persistence, Tree::default()),
            },
        })
    }

    #[cfg(rocksdb)]
    fn new_persistent_database<KV>() -> (
        tokio::runtime::Runtime,
        KV::Keepalive,
        KV::Repo,
        TracedDatabase<KV>,
    )
    where
        KV: BackgroundWriteableKeyValueStore + TestKeyValueStoreSetup,
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        (runtime, keepalive, repo, database)
    }

    kv_test!(test_database_commit_and_checkout, KV: BackgroundPersistentKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        entries in prop::collection::vec(
            (prop::collection::vec(any::<u8>(), 1..=KEY_MAX_SIZE),
             prop::collection::vec(any::<u8>(), 0..200)),
            1..50,
        ),
    ], {
        let mut database = new_database::<KV>(handle, repo);
        let expected = database.insert_entries(entries);

        let expected_hash = database.hash().expect("Hash should be calculated");
        let commit_id = database.commit(repo).expect("Commit should succeed");

        let checked_out = TracedDatabase::<KV>::checkout(handle, repo, commit_id)
            .expect("Checkout should succeed");

        prop_assert_eq!(checked_out.hash().expect("Hash should be calculated"), expected_hash);

        for (key, value) in expected {
            checked_out.assert_database_value(&key, value.as_ref());
        }

        database.into_trace()
    });

    kv_test!(test_database_delete, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
        data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        for (key, data) in keys.iter().zip(data.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
            database
                .set(key.clone(), Bytes::copy_from_slice(data))
                .expect("Writing should succeed");
            prop_assert!(database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));

            let before = database.hash().expect("Hash should be calculated");
            database.delete(key.clone()).expect("Deleting should succeed");
            let after = database.hash().expect("Hash should be calculated");
            assert_ne!(before, after);
            prop_assert!(!database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));
        }

        database.into_trace()
    });

    kv_test!(test_database_delete_nonexistent, KV: BackgroundWriteableKeyValueStore, {
        // Receiving the hash requires a separate worker thread
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();

        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        database
            .set(
                Key::new(&[1]).expect("Size less than KEY_MAX_SIZE"),
                Bytes::copy_from_slice(&[2, 3]),
            )
            .expect("Writing should succeed");

        let before = database.hash().expect("Hash should be calculated");

        // Delete a nonexistent key
        let nonexistent_key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        assert!(
            !database
                .exists(&nonexistent_key)
                .expect("There should be no other `PersistenceLayerError`s")
        );
        assert!(database.delete(nonexistent_key).is_ok());

        // Ensure the root hash is unchanged
        let after = database.hash().expect("Hash should be calculated");
        assert_eq!(before, after);

        database.into_trace()
    });

    /// Operation to perform immediately before the delete
    enum RaceConditionOp {
        Set,
        Write,
    }

    fn test_op_then_delete_race_condition<KV>(op: RaceConditionOp)
    where
        KV: TestKeyValueStoreSetup + BackgroundPersistentKeyValueStore,
    {
        // Arrange
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();

        let mut db = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let initial_hash = db.hash().expect("Hashing should succeed");

        db.set(key.clone(), Bytes::new())
            .expect("Writing should succeed");
        let commit = db.commit(&repo).expect("Commit should succeed");

        // Act
        let db =
            Database::<KV, _>::checkout(handle, &repo, commit).expect("Checkout should succeed");
        let super::NormalImpl { merkle, persistent } = db.inner;

        let new_value = vec![1, 2, 3];

        // simulate race condition: emit persistence layer operations first
        match op {
            RaceConditionOp::Set => {
                persistent
                    .set(key.as_ref(), &new_value)
                    .expect("set should succeed");
            }
            RaceConditionOp::Write => {
                persistent
                    .write(key.as_ref(), 0, &new_value)
                    .expect("write should succeed");
            }
        }

        persistent
            .delete(key.as_ref())
            .expect("Delete should succeed");

        // replay on merkle worker - the full operation order should succeed
        match op {
            RaceConditionOp::Set => {
                merkle
                    .set(key.clone(), Bytes::from(new_value))
                    .expect("worker should not have crashed");
            }
            RaceConditionOp::Write => {
                merkle
                    .write(key.clone(), 0, Bytes::from(new_value))
                    .expect("worker should not have crashed");
            }
        }
        merkle.delete(key).expect("worker should not have crashed");

        // Assert
        // - hash to ensure consistency
        let final_hash = merkle.hash().expect("worker should not have crashed");
        assert_eq!(
            initial_hash, final_hash,
            "Empty DB should always hash to the same value"
        );
    }

    // Test to exercise fix for RV-987: race condition for set-then-delete failure
    kv_test!(test_set_delete_race_condition, KV: BackgroundPersistentKeyValueStore, {
        test_op_then_delete_race_condition::<KV>(RaceConditionOp::Set);
    });

    // Test to exercise fix for RV-987: race condition for write-then-delete failure
    kv_test!(test_write_delete_race_condition, KV: BackgroundPersistentKeyValueStore, {
        test_op_then_delete_race_condition::<KV>(RaceConditionOp::Write);
    });

    // Test to exercise fix for RV-987: race condition for write-then-set failure
    //
    // This can occur when a subsequent `set` results in a value that is shorter than the
    // offset the write is trying to write to
    kv_test!(test_write_set_race_condition, KV: BackgroundPersistentKeyValueStore, {
        // Arrange
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();

        let mut db = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let original_value = Bytes::new();

        db.set(key.clone(), original_value.clone()).expect("Writing should succeed");

        let initial_hash = db.hash().expect("Hashing should succeed");

        // set to a non-zero length value, needing so we can shorten the value later
        db.set(key.clone(), Bytes::from(vec![1, 2, 3])).expect("Setting should succeed");
        let commit = db.commit(&repo).expect("Commit should succeed");

        // Act
        let db = Database::<KV, _>::checkout(handle, &repo, commit).expect("Checkout should succeed");
        let super::NormalImpl { merkle, persistent } = db.inner;

        let new_value = vec![5, 6, 7];
        let offset = 2;

        // simulate race condition: emit persistence layer operations first
        persistent.write(key.as_ref(), offset, &new_value).expect("write should succeed");
        // shorten the value - the write on the merkle layer will later fail due to OffsetTooLarge
        persistent.set(key.as_ref(), []).expect("Set should succeed");

        // replay on merkle worker - the full operation order should succeed
        merkle.write(key.clone(), offset, Bytes::from(new_value)).expect("worker should not have crashed");
        merkle.set(key, Bytes::new()).expect("worker should not have crashed");

        // Assert
        // - hash to ensure consistency
        let final_hash = merkle.hash().expect("worker should not have crashed");
        assert_eq!(initial_hash, final_hash, "DB with single identical Key-Value should always hash to the same value");
    });

    // Test to verify the fix for RV-955 (deletion after checkout failed).
    kv_test!(test_database_delete_after_checkout, KV: BackgroundPersistentKeyValueStore, {
        // Arrange
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();

        let mut db = new_database::<KV>(handle, &repo);
        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");

        let initial_hash = db.hash().expect("Hashing should succeed");

        db.set(key.clone(), Bytes::new()).expect("Writing should succeed");

        let commit = db.commit(&repo).expect("Commit should succeed");

        // Act
        let mut db = TracedDatabase::<KV>::checkout(handle, &repo, commit).expect("Checkout should succeed");

        db.delete(key).expect("Delete should succeed");

        // Assert
        // - we emit another operation to actually observe any crash.
        let final_hash = db.hash().expect("Hashing should succeed");
        assert_eq!(initial_hash, final_hash, "Empty DB should always hash to the same value");

        db.into_trace()
    });

    kv_test!(test_database_checkout_commit_creates_new_snapshot, KV: BackgroundPersistentKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();

        let mut original = new_database::<KV>(handle, &repo);

        let persisted_key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let derived_key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        original
            .set(persisted_key.clone(), Bytes::from_static(b"before"))
            .expect("Writing should succeed");

        let original_commit = original.commit(&repo).expect("Commit should succeed");

        let mut checked_out = TracedDatabase::<KV>::checkout(handle, &repo, original_commit)
            .expect("Checkout should succeed");
        checked_out
            .set(persisted_key.clone(), Bytes::from_static(b"after"))
            .expect("Writing should succeed");
        checked_out
            .set(derived_key.clone(), Bytes::from_static(b"new"))
            .expect("Writing should succeed");

        let derived_commit = checked_out.commit(&repo).expect("Commit should succeed");
        assert_ne!(derived_commit, original_commit);

        let original_reloaded = TracedDatabase::<KV>::checkout(handle, &repo, original_commit)
            .expect("Checkout should succeed");
        original_reloaded.assert_database_value(&persisted_key, b"before");
        original_reloaded.assert_traced_database_missing(&derived_key);

        let derived_reloaded = TracedDatabase::<KV>::checkout(handle, &repo, derived_commit)
            .expect("Checkout should succeed");
        derived_reloaded.assert_database_value(&persisted_key, b"after");
        derived_reloaded.assert_database_value(&derived_key, b"new");

        (original.into_trace(), checked_out.into_trace())
    });

    // Not `kv_test!`s: the in-memory backend copies a commit into memory either way, so it has no
    // read-only store.
    #[cfg(rocksdb)]
    #[test]
    fn test_database_checkout_read_only_reads_the_commit() {
        use super::Database;
        use crate::persistence_layer::PersistenceLayer;
        use crate::persistence_layer::ReadOnlyPersistenceLayer;

        let (runtime, _keepalive, repo, mut database) =
            new_persistent_database::<PersistenceLayer>();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key.clone(), Bytes::from_static(b"committed"))
            .expect("Writing should succeed");
        let expected_hash = database.hash().expect("Hashing should succeed");
        let commit_id = database.commit(&repo).expect("Commit should succeed");

        // A read-only database holds no Merkle worker, so it depends on no runtime: drop the one
        // the writeable database needed, along with the database itself.
        drop(database);
        drop(runtime);

        let read_only =
            Database::<ReadOnlyPersistenceLayer, Normal>::checkout_read_only(&repo, commit_id)
                .expect("Read-only checkout should succeed");

        read_only.assert_database_value(&key, b"committed");
        assert_eq!(
            read_only.hash().expect("Hashing should succeed"),
            expected_hash,
            "A read-only checkout should have the committed root hash"
        );

        let clone = read_only.clone_read_only();
        clone.assert_database_value(&key, b"committed");
        assert_eq!(clone.hash().expect("Hashing should succeed"), expected_hash);
    }

    #[cfg(rocksdb)]
    #[test]
    fn test_database_checkout_read_only_to_writeable() {
        use super::Database;
        use crate::persistence_layer::PersistenceLayer;
        use crate::persistence_layer::ReadOnlyPersistenceLayer;

        let (runtime, _keepalive, repo, mut database) =
            new_persistent_database::<PersistenceLayer>();
        let handle = runtime.handle();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let added_key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key.clone(), Bytes::from_static(b"committed"))
            .expect("Writing should succeed");
        let commit_id = database.commit(&repo).expect("Commit should succeed");
        let committed_hash = database.hash().expect("Hashing should succeed");

        let read_only =
            Database::<ReadOnlyPersistenceLayer, Normal>::checkout_read_only(&repo, commit_id)
                .expect("Read-only checkout should succeed");

        // Upgrading pays for the copy and yields a database which can be modified and committed.
        let mut writeable = read_only
            .to_writeable(handle, &repo)
            .expect("Making the database writeable should succeed");

        writeable.assert_database_value(&key, b"committed");
        assert_eq!(
            writeable.hash().expect("Hashing should succeed"),
            committed_hash,
            "The working copy should start from the committed state"
        );

        writeable
            .set(added_key.clone(), Bytes::from_static(b"added"))
            .expect("Writing to the working copy should succeed");
        writeable.assert_database_value(&added_key, b"added");

        let derived_commit = writeable
            .commit(&repo)
            .expect("Committing the working copy should succeed");
        assert_ne!(derived_commit, commit_id);

        // The original commit is untouched: the read-only database still sees it, and so does a
        // fresh checkout.
        assert!(
            !read_only
                .exists(&added_key)
                .expect("Existence check should succeed"),
            "The read-only database should not observe the working copy's writes"
        );
        assert_eq!(
            read_only.hash().expect("Hashing should succeed"),
            committed_hash
        );

        let reloaded = Database::<PersistenceLayer, Normal>::checkout(handle, &repo, commit_id)
            .expect("Checking out the original commit should succeed");
        assert_eq!(
            reloaded.hash().expect("Hashing should succeed"),
            committed_hash
        );
    }

    #[cfg(rocksdb)]
    #[test]
    fn test_database_checkout_read_only_unknown_commit_fails() {
        use octez_riscv_data::hash::Hash;

        use super::Database;
        use crate::commit::CommitId;
        use crate::errors::OperationalError;
        use crate::persistence_layer::PersistenceLayer;
        use crate::persistence_layer::ReadOnlyPersistenceLayer;

        let (_runtime, _keepalive, repo, _database) = new_persistent_database::<PersistenceLayer>();
        let missing_commit = CommitId::from(Hash::hash_bytes(b"missing-commit"));

        assert!(matches!(
            Database::<ReadOnlyPersistenceLayer, Normal>::checkout_read_only(&repo, missing_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));
    }

    #[cfg(rocksdb)]
    #[test]
    fn test_database_checkout_missing_root_blob_fails_operationally() {
        use super::Database;
        use crate::errors::Error;
        use crate::errors::OperationalError;
        use crate::persistence_layer::PersistenceLayer;

        let (runtime, _keepalive, repo, mut database) =
            new_persistent_database::<PersistenceLayer>();
        let handle = runtime.handle();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key, Bytes::from_static(b"value"))
            .expect("Writing should succeed");

        let commit_id = database.commit(&repo).expect("Commit should succeed");

        // Node bodies live in the repository's Merkle store rather than in the commit directory,
        // so that is where the root has to be removed from to make the commit unreadable.
        repo.merkle_store()
            .delete(commit_id.as_hash().as_ref())
            .expect("Deleting the root node should succeed");

        assert!(matches!(
            Database::<PersistenceLayer, _>::checkout(handle, &repo, commit_id),
            Err(Error::Operational(OperationalError::CommitDataMissing { root, .. }))
                if root == *commit_id.as_hash()
        ));
    }

    /// A read-only checkout never loads the Merkle root, so - unlike
    /// [`Database::checkout`], see the test above - it cannot notice that the root blob is
    /// missing. Reads are served from the store and the root hash is the commit id, so both
    /// still work; the damage only surfaces when a working copy is made and a tree is needed.
    #[cfg(rocksdb)]
    #[test]
    fn test_read_only_checkout_of_missing_root_blob_defers_the_failure() {
        use super::Database;
        use crate::errors::OperationalError;
        use crate::persistence_layer::PersistenceLayer;
        use crate::persistence_layer::ReadOnlyPersistenceLayer;

        let (runtime, _keepalive, repo, mut database) =
            new_persistent_database::<PersistenceLayer>();
        let handle = runtime.handle();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key.clone(), Bytes::from_static(b"value"))
            .expect("Writing should succeed");

        let commit_id = database.commit(&repo).expect("Commit should succeed");

        // Node bodies live in the repository's Merkle store rather than in the commit directory,
        // so that is where the root has to be removed from to make the commit unreadable.
        repo.merkle_store()
            .delete(commit_id.as_hash().as_ref())
            .expect("Deleting the root node should succeed");

        let read_only =
            Database::<ReadOnlyPersistenceLayer, Normal>::checkout_read_only(&repo, commit_id)
                .expect("Read-only checkout should succeed without loading the root");

        read_only.assert_database_value(&key, b"value");
        assert_eq!(
            read_only.hash().expect("Hashing should succeed"),
            *commit_id.as_hash(),
            "The root hash of a read-only checkout is its commit id"
        );

        assert!(matches!(
            read_only.to_writeable(handle, &repo),
            Err(OperationalError::CommitDataMissing { root, .. })
                if root == *commit_id.as_hash()
        ));
    }

    // A checked-out node no longer arrives with its value materialised, so a `set` or a `write`
    // against a committed key resolves that value itself - a `write` after the store already
    // holds it, since the store is written before the tree operation is queued.
    kv_test!(test_database_set_and_write_after_checkout, KV: BackgroundPersistentKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();

        let key = Key::new(b"probe").expect("Size less than KEY_MAX_SIZE");

        let mut database = new_database::<KV>(handle, &repo);
        database
            .set(key.clone(), Bytes::from_static(b"a value with room to write into"))
            .expect("Setting should succeed");
        let commit = database.commit(&repo).expect("Commit should succeed");

        // Checkout, then overwrite a committed key outright.
        let mut checked_out = TracedDatabase::<KV>::checkout(handle, &repo, commit)
            .expect("Checkout should succeed");
        checked_out
            .set(key.clone(), Bytes::from_static(b"replaced outright"))
            .expect("Setting a committed key after checkout should succeed");
        checked_out.assert_database_value(&key, b"replaced outright");
        let after_set = checked_out.hash().expect("Hashing after a set should succeed");

        // Checkout again, and this time write into the middle of a committed value.
        let mut checked_out = TracedDatabase::<KV>::checkout(handle, &repo, commit)
            .expect("Checkout should succeed");
        checked_out
            .write(key.clone(), 7, Bytes::from_static(b"WITH"))
            .expect("Writing into a committed value after checkout should succeed");
        checked_out.assert_database_value(&key, b"a valueWITHh room to write into");
        let after_write = checked_out.hash().expect("Hashing after a write should succeed");

        // Reads are answered by the store, so the value assertions above hold however the tree
        // resolved. Pin the tree itself against a database built from scratch with the same
        // content: a tree that folded another value commits to a different root.
        let expect_root = |value| {
            let mut expected = new_database::<KV>(handle, &repo);
            expected
                .set(key.clone(), value)
                .expect("Setting should succeed");
            expected.hash().expect("Hashing should succeed")
        };

        assert_eq!(after_set, expect_root(Bytes::from_static(b"replaced outright")));
        assert_eq!(
            after_write,
            expect_root(Bytes::from_static(b"a valueWITHh room to write into"))
        );

        (checked_out.into_trace(),)
    });

    kv_test!(test_database_checkout_unknown_commit_fails, KV: BackgroundPersistentKeyValueStore, {
        use octez_riscv_data::hash::Hash;

        use crate::commit::CommitId;
        use crate::errors::OperationalError;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        let missing_commit = CommitId::from(Hash::hash_bytes(b"missing-commit"));

        assert!(matches!(
            TracedDatabase::<KV>::checkout(handle, &repo, missing_commit),
            Err(Error::Operational(OperationalError::CommitNotFound))
        ));

        database.into_trace()
    });

    kv_test!(test_database_exists, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
        data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        let mut seen = HashSet::new();

        for (key, data) in keys.iter().zip(data.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
            let data: &[u8] = data;

            prop_assert_ne!(database.exists(&key)
                    .expect("There should be no other `PersistenceLayerError`s"),
                seen.insert(key.clone()));

            database
                .set(key.clone(), Bytes::copy_from_slice(data))
                .expect("Writing should succeed");
            prop_assert!(database.exists(&key).expect("There should be no other `PersistenceLayerError`s"));
        }

        database.into_trace()
    });

    kv_test!(test_database_hash, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            // Needs a thread for sending and a thread for receiving
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE), 0..100),
        data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..100),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        let mut seen = HashSet::new();

        for (key, data) in keys.iter().zip(data.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
            let data: &[u8] = data;

            let before = database.hash().expect("Hash should be calculated");

            database
                .set(key.clone(), Bytes::copy_from_slice(data))
                .expect("Writing should succeed");

            let after = database.hash().expect("Hash should be calculated");

            let existing_pair = !seen.insert((key, data));
            // Avoid the edge case of an identical hash from a previously seen identical
            // key-value pair, where no other keys were written to in between.
            if !existing_pair {
                prop_assert_ne!(before, after);
            }
        }

        database.into_trace()
    });

    kv_test!(test_database_hash_revert, KV: BackgroundWriteableKeyValueStore, {
        // Needs a thread for sending and a thread for receiving
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        let original_data = [1, 2, 3];
        let mutated_data = [3, 2, 1];

        database
            .set(key.clone(), Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");

        let before = database.hash().expect("Hash should be calculated");

        // Mutate the same key
        database
            .set(key.clone(), Bytes::copy_from_slice(&mutated_data))
            .expect("Writing should succeed");

        let after = database.hash().expect("Hash should be calculated");
        assert_ne!(before, after);

        // Revert the value of the same key to the original value and check the hash reverts to the
        // same value.
        database
            .set(key.clone(), Bytes::copy_from_slice(&original_data))
            .expect("Writing should succeed");
        let reverted = database.hash().expect("Hash should be calculated");
        assert_eq!(before, reverted);

        database.into_trace()
    });

    kv_test!(test_database_read, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
        data in prop::collection::vec(prop::collection::vec(any::<u8>(), 3..100), 0..100),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        for (key, data) in keys.iter().zip(data.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
            let mut read_data: [u8; 100] = [42; 100];

            let read_data_before = read_data;

            // Set the data
            database
                .set(key.clone(), Bytes::copy_from_slice(data))
                .expect("Setting should succeed");

            // The offset is bigger than the value
            prop_assert!(database.read(&key, data.len() + 1, read_data.as_mut_slice()).is_err());
            prop_assert_eq!(read_data, read_data_before);

            // Partial value write, where the output parameter is smaller than the data.
            prop_assert_eq!(
                database
                    .read(&key, 0, read_data[1..data.len()].as_mut())
                    .expect(
                        "Reading a value larger than the output parameter's size should succeed"
                    ),
                data.len() - 1
            );
            prop_assert_eq!(read_data[0], read_data_before[0]);
            prop_assert_eq!(&read_data[1..data.len()], &data[..data.len() - 1]);
            prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);
            let read_data_before = read_data;

            let read = database
                .read(&key, data.len(), read_data.as_mut_slice())
                .expect("A zero-sized write should succeed");
            prop_assert_eq!(read, 0);
            prop_assert_eq!(read_data, read_data_before);

            // Whole value write
            let read = database
                .read(&key, 0, read_data.as_mut_slice())
                .expect("Writing the whole value should succeed");
            prop_assert_eq!(read, data.len());
            prop_assert_eq!(&read_data[..data.len()], data.as_slice());
            prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);

            // Partial value write
            prop_assert_eq!(&read_data[2..data.len()], &data[2..]);
            let read = database
                .read(&key, data.len() - 1, read_data[1..2].as_mut())
                .expect("A partial write should succeed");
            prop_assert_eq!(read, 1);
            prop_assert_eq!(&read_data[1..2], &data[data.len() - 1..]);
            prop_assert_eq!(&read_data[2..data.len()], &data[2..]);
            prop_assert_eq!(&read_data[data.len()..], &read_data_before[data.len()..]);

            // Write limited by buffer
            let mut small_buffer: [u8; 3] = [0, 0 ,0];
            let read = database
                .read(&key, 0, small_buffer.as_mut_slice())
                .expect("Writing into a smaller buffer should succeed");
            prop_assert_eq!(read, small_buffer.len());
            prop_assert_eq!(&small_buffer, &data[0..3]);
        }

        database.into_trace()
    });

    kv_test!(test_database_read_bytes, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
        data in prop::collection::vec(prop::collection::vec(any::<u8>(), 3..100), 0..100),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        for (key, data) in keys.iter().zip(data.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");

            // Set the data
            database
                .set(key.clone(), Bytes::copy_from_slice(data))
                .expect("Setting should succeed");

            // The offset is bigger than the value
            prop_assert!(database.read_bytes(&key, data.len() + 1, 1).is_err());

            // Whole value read
            let result = database
                .read_bytes(&key, 0, data.len())
                .expect("Reading from offset 0 should succeed");
            prop_assert_eq!(result.as_slice(), data.as_slice());

            // Zero-sized read at end of value
            let result = database
                .read_bytes(&key, 0, 0)
                .expect("A zero-sized read should succeed");
            prop_assert_eq!(result.as_slice(), &[] as &[u8]);

            // Partial read from last byte
            let result = database
                .read_bytes(&key, data.len() - 1, 1)
                .expect("A partial read should succeed");
            prop_assert_eq!(result.as_slice(), &data[data.len() - 1..]);
        }

        database.into_trace()
    });

    kv_test!(test_database_read_bytes_no_key, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");

        // The key doesn't exist
        assert!(matches!(
            database.read_bytes(&key, 0, 1),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));

        database.into_trace()
    });

    kv_test!(test_database_read_no_key, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let mut read_data: [u8; 100] = [42; 100];
        let read_data_before = read_data;

        // The key doesn't exist
        assert!(matches!(
            database.read(&key, 0, read_data.as_mut_slice()),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
        assert_eq!(read_data_before, read_data);

        database.into_trace()
    });

    kv_test!(test_database_read_bytes_io_too_large, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");

        // The io request is too large - this takes priority
        // even though the key does not exist
        assert!(matches!(
            database.read_bytes(&key, 5, MAX_FILE_CHUNK_SIZE + 1),
            Err(Error::InvalidArgument(
                InvalidArgumentError::IoRequestTooLarge
            ))
        ));

        database.into_trace()
    });

    kv_test!(test_database_read_io_too_large, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let mut read_data: Vec<u8> = vec![42; MAX_FILE_CHUNK_SIZE + 1];
        let read_data_before = read_data.clone();

        // The io request is too large - this takes priority
        // even though the key does not exist
        assert!(matches!(
            database.read(&key, 5, read_data.as_mut_slice()),
            Err(Error::InvalidArgument(
                InvalidArgumentError::IoRequestTooLarge
            ))
        ));
        assert_eq!(read_data_before, read_data);

        database.into_trace()
    });

    kv_test!(test_database_value_length, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..100),
        data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..100), 0..100),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        for (key, data) in keys.iter().zip(data.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");
            let data = Bytes::copy_from_slice(data);

            database
                .set(key.clone(), Bytes::copy_from_slice(&data))
                .expect("Writing should succeed");
            prop_assert_eq!(
                database
                    .value_length(&key)
                    .expect("Getting the value length should succeed"),
                data.len()
            );
        }

        database.into_trace()
    });

    kv_test!(test_database_write, KV: BackgroundWriteableKeyValueStore,
        setup_runtime |handle, repo| = {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Creating a Tokio runtime should succeed");
            let handle = runtime.handle().clone();
            let (_keepalive, repo) = KV::setup_repo();
            (runtime, handle, _keepalive, repo)
        },
    [
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..KEY_MAX_SIZE), 0..10),
        offsets in prop::collection::vec(0..10usize, 0..10),
        initial_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..10),
        patch in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..200), 0..10),
    ], {
        let mut database = new_database::<KV>(handle, repo);

        for (((key, offset), initial_data), patch) in keys.iter().zip(offsets.iter()).zip(initial_data.iter()).zip(patch.iter()) {
            let key = Key::new(key).expect("Size less than KEY_MAX_SIZE");

            let initial_data = Bytes::copy_from_slice(initial_data);
            assert!(database.set(key.clone(), initial_data.clone()).is_ok());

            let patch = Bytes::copy_from_slice(patch);
            let expected_written = patch.len();
            let result = database.write(key.clone(), *offset, patch.clone());
            if *offset > initial_data.len() {
                prop_assert!(result.is_err());
            } else {
                prop_assert_eq!(result.unwrap(), expected_written);
                let expected_length = std::cmp::max(initial_data.len(), offset + patch.len());
                prop_assert_eq!(database.value_length(&key).unwrap(), expected_length);
            }
        }

        database.into_trace()
    });

    kv_test!(test_database_write_new_nonzero_offset, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[]);

        let res = database.write(key.clone(), 1, data);

        assert!(
            matches!(
                res,
                Err(Error::InvalidArgument(InvalidArgumentError::OffsetTooLarge))
            ),
            "Values that don't exist are implicitly zero in size when written. Got {res:?}"
        );

        database.into_trace()
    });

    kv_test!(test_database_write_io_too_large, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(vec![0; MAX_FILE_CHUNK_SIZE + 1].as_slice());

        let res = database.write(key.clone(), 1, data);

        // even though the offset is too large, the io error takes priority
        assert!(matches!(
            res,
            Err(Error::InvalidArgument(
                InvalidArgumentError::IoRequestTooLarge
            ))
        ));

        database.into_trace()
    });

    kv_test!(test_database_write_no_truncation, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("a long value");
        let data2 = Bytes::from("good");
        let data3 = Bytes::from("nother");

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(database.write(key.clone(), 2, data2).is_ok());
        let mut output = vec![0; data.len()];
        assert!(database.read(&key, 0, output.as_mut_slice()).is_ok());
        assert_eq!(output.as_slice(), "a good value".as_bytes());

        assert!(database.write(key.clone(), 0, data3).is_ok());
        let mut output = vec![0; data.len()];
        assert!(database.read(&key, 0, output.as_mut_slice()).is_ok());
        assert_eq!(output.as_slice(), "nother value".as_bytes());

        database.into_trace()
    });

    kv_test!(test_database_write_offset_append, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[1, 2, 3]);

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(
            database
                .write(key.clone(), data.len(), data.clone())
                .is_ok()
        );
        let mut output = vec![0; 2 * data.len()];
        assert!(database.read(&key, 0, output.as_mut_slice()).is_ok());
        assert_eq!(output.as_slice(), [1, 2, 3, 1, 2, 3]);

        database.into_trace()
    });

    kv_test!(test_database_write_oversized_offset, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(&[]);

        assert!(database.set(key.clone(), data.clone()).is_ok());
        assert!(database.write(key.clone(), data.len() + 1, data).is_err());

        database.into_trace()
    });

    kv_test!(test_database_set_io_too_large, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::copy_from_slice(vec![0; MAX_FILE_CHUNK_SIZE + 1].as_slice());

        let res = database.set(key.clone(), data);

        assert!(matches!(
            res,
            Err(Error::InvalidArgument(
                InvalidArgumentError::IoRequestTooLarge
            ))
        ));

        database.into_trace()
    });

    kv_test!(test_database_write_value_too_large, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        let can_write = 100;

        let key = Key::new(&[]).expect("Size less than KEY_MAX_SIZE");
        let bytes = Bytes::from(vec![42; MAX_VALUE_SIZE - can_write]);

        <Normal as super::DatabaseMode>::set(database.inner_mut(), key.clone(), bytes)
            .expect("Setting should succeed (bypassing API layer)");

        let value_len = database
            .value_length(&key)
            .expect("The key was written to previously");

        // Check all writes that would cause the value size to grow too large fail
        // the choice of '50' is arbitrary - mainly just chose something to prevent
        // the test running too long
        for buffer_size in ((can_write + 1)..MAX_FILE_CHUNK_SIZE)
            .step_by(50)
            .chain([MAX_FILE_CHUNK_SIZE])
        {
            let bytes = Bytes::from(vec![15; buffer_size]);
            let result = database.write(key.clone(), value_len, bytes);

            assert!(
                matches!(
                    result,
                    Err(Error::InvalidArgument(
                        InvalidArgumentError::ValueSizeTooLarge
                    ))
                ),
                "Write would cause value to exceed MAX_VALUE_SIZE"
            );

            assert_eq!(
                value_len,
                database.value_length(&key).unwrap(),
                "Value size must not have changed"
            );
        }

        // Double check writing more than MAX_FILE_CHUNK_SIZE triggers IoRequestTooLarge
        let bytes = Bytes::from(vec![15; MAX_FILE_CHUNK_SIZE + 1]);
        let result = database.write(key.clone(), value_len, bytes);

        assert!(
            matches!(
                result,
                Err(Error::InvalidArgument(
                    InvalidArgumentError::IoRequestTooLarge
                ))
            ),
            "Write would cause value to exceed MAX_VALUE_SIZE"
        );

        // Check that writing up to MAX_VALUE_SIZE is ok
        let bytes = Bytes::from(vec![15; can_write]);
        let wrote = database
            .write(key.clone(), value_len, bytes)
            .expect("Writing up to allowed value size succeeds");

        assert_eq!(
            wrote, can_write,
            "Write increasing value_length up to MAX_VALUE_SIZE succeeds"
        );
        assert_eq!(MAX_VALUE_SIZE, database.value_length(&key).unwrap());

        // Ensure we can still append 'zero bytes' to end of value
        let wrote = database
            .write(key.clone(), MAX_VALUE_SIZE, Bytes::new())
            .expect("Appending zero bytes to value of max allowed size should succeed");

        assert_eq!(
            wrote, 0,
            "Appending zero bytes to maximum length value does not change size"
        );
        assert_eq!(MAX_VALUE_SIZE, database.value_length(&key).unwrap());

        database.into_trace()
    });

    kv_test!(test_verify_database_delete, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_verify_database::<KV>(&repo);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&[]))
            .expect("Verify mode does not return `OperationError`s");
        assert!(
            database
                .exists(&key)
                .expect("Verify mode does not return `OperationError`s")
        );

        database
            .delete(key.clone())
            .expect("Verify mode does not return `OperationError`s");
        assert!(
            !database
                .exists(&key)
                .expect("Verify mode does not return `OperationError`s")
        );

        // Deleting a non-existent key should also succeed.
        let nonexistent_key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        assert!(
            !database
                .exists(&nonexistent_key)
                .expect("Verify mode does not return `OperationError`s")
        );
        database
            .delete(nonexistent_key)
            .expect("Verify mode does not return `OperationError`s");

        database.into_trace()
    });

    kv_test!(test_verify_database_set_and_read, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        data in prop::collection::vec(any::<u8>(), 0..200),
    ], {
        let mut database = new_verify_database::<KV>(repo);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&data))
            .expect("Verify mode does not return `OperationError`s");

        let result = database
            .read_bytes(&key, 0, data.len())
            .expect("Verify mode does not return `OperationError`s");
        prop_assert_eq!(result.as_slice(), data.as_slice());

        database.into_trace()
    });

    kv_test!(test_verify_database_value_length, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        data in prop::collection::vec(any::<u8>(), 0..200),
    ], {
        let mut database = new_verify_database::<KV>(repo);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&data))
            .expect("Verify mode does not return `OperationError`s");

        prop_assert_eq!(
            database
                .value_length(&key)
                .expect("Verify mode does not return `OperationError`s"),
            data.len()
        );

        database.into_trace()
    });

    kv_test!(test_verify_database_write_partial, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        initial in prop::collection::vec(any::<u8>(), 1..200),
        patch in prop::collection::vec(any::<u8>(), 0..200),
        offset_frac in 0_usize..=100,
    ], {
        let offset = offset_frac * initial.len() / 100;
        let patch_len = patch.len().min(initial.len() - offset);
        let patch = &patch[..patch_len];

        let mut database = new_verify_database::<KV>(repo);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&initial))
            .expect("Verify mode does not return `OperationError`s");

        let written = database
            .write(key.clone(), offset, Bytes::copy_from_slice(patch))
            .expect("Verify mode does not return `OperationError`s");
        prop_assert_eq!(written, patch_len);

        let result = database
            .read_bytes(&key, 0, initial.len())
            .expect("Verify mode does not return `OperationError`s");

        let mut expected = initial.clone();
        expected[offset..offset + patch_len].copy_from_slice(patch);
        prop_assert_eq!(result.as_slice(), expected.as_slice());

        database.into_trace()
    });

    fn new_persistence<KV>() -> (KV::Keepalive, Arc<KV>)
    where
        KV: BackgroundWriteableKeyValueStore + TestKeyValueStoreSetup,
    {
        let (keepalive, repo) = KV::setup_repo();
        let persistence: Arc<KV> = KV::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        (keepalive, persistence)
    }

    kv_test!(test_prove_database_delete, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, persistence) = new_persistence::<KV>();
        let mut database = new_prove_database::<KV>(persistence);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&[]))
            .expect("Setting a key in Prove mode should succeed");
        assert!(
            database
                .exists(&key)
                .expect("Existence check in Prove mode should succeed")
        );

        database
            .delete(key.clone())
            .expect("Deleting a key in Prove mode should succeed");
        assert!(
            !database
                .exists(&key)
                .expect("Existence check in Prove mode should succeed")
        );

        // Deleting a non-existent key should also succeed.
        let nonexistent_key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        assert!(
            !database
                .exists(&nonexistent_key)
                .expect("Existence check in Prove mode should succeed")
        );
        database
            .delete(nonexistent_key)
            .expect("Deleting a non-existent key in Prove mode should succeed");

        database.into_trace()
    });

    kv_test!(test_prove_database_set_and_read, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        data in prop::collection::vec(any::<u8>(), 0..200),
    ], {
        let persistence: Arc<KV> = KV::new(repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut database = new_prove_database::<KV>(persistence);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&data))
            .expect("Setting a key in Prove mode should succeed");

        let result = database
            .read_bytes(&key, 0, data.len())
            .expect("Reading a key in Prove mode should succeed");
        prop_assert_eq!(result.as_slice(), data.as_slice());

        database.into_trace()
    });

    kv_test!(test_prove_database_value_length, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        data in prop::collection::vec(any::<u8>(), 0..200),
    ], {
        let persistence: Arc<KV> = KV::new(repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut database = new_prove_database::<KV>(persistence);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&data))
            .expect("Setting a key in Prove mode should succeed");

        prop_assert_eq!(
            database
                .value_length(&key)
                .expect("Reading the length in Prove mode should succeed"),
            data.len()
        );

        database.into_trace()
    });

    kv_test!(test_prove_database_write_partial, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        initial in prop::collection::vec(any::<u8>(), 1..200),
        patch in prop::collection::vec(any::<u8>(), 0..200),
        offset_frac in 0_usize..=100,
    ], {
        let offset = offset_frac * initial.len() / 100;
        let patch_len = patch.len().min(initial.len() - offset);
        let patch = &patch[..patch_len];

        let persistence: Arc<KV> = KV::new(repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut database = new_prove_database::<KV>(persistence);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&initial))
            .expect("Setting a key in Prove mode should succeed");

        let written = database
            .write(key.clone(), offset, Bytes::copy_from_slice(patch))
            .expect("Writing a key in Prove mode should succeed");
        prop_assert_eq!(written, patch_len);

        let result = database
            .read_bytes(&key, 0, initial.len())
            .expect("Reading a key in Prove mode should succeed");

        let mut expected = initial.clone();
        expected[offset..offset + patch_len].copy_from_slice(patch);
        prop_assert_eq!(result.as_slice(), expected.as_slice());

        database.into_trace()
    });

    kv_test!(test_prove_database_missing_key, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, persistence) = new_persistence::<KV>();
        let mut database = new_prove_database::<KV>(persistence);

        let missing_key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        assert!(
            !database
                .exists(&missing_key)
                .expect("Existence check on an absent key should succeed")
        );
        assert!(matches!(
            database.value_length(&missing_key),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
        assert!(matches!(
            database.read_bytes(&missing_key, 0, 1),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
        let mut buf = [0u8; 4];
        assert!(matches!(
            database.read(&missing_key, 0, buf.as_mut_slice()),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
        database
            .delete(missing_key)
            .expect("Deleting a non-existent key in Prove mode should succeed");

        // After insert + delete, the key should once again behave as absent.
        let key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key.clone(), Bytes::copy_from_slice(b"data"))
            .expect("Setting a key in Prove mode should succeed");
        database
            .delete(key.clone())
            .expect("Deleting a key in Prove mode should succeed");

        assert!(
            !database
                .exists(&key)
                .expect("Existence check after delete should succeed")
        );
        assert!(matches!(
            database.value_length(&key),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
        assert!(matches!(
            database.read_bytes(&key, 0, 1),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));

        database.into_trace()
    });

    kv_test!(test_prove_database_read_bytes_partial, KV: BackgroundWriteableKeyValueStore,
        setup |repo| = { KV::setup_repo() },
    [
        data in prop::collection::vec(any::<u8>(), 3..200),
        offset_frac in 0_usize..=100,
        len_frac in 0_usize..=100,
    ], {
        let offset = offset_frac * data.len() / 100;
        let length = len_frac * (data.len() - offset) / 100;

        let persistence: Arc<KV> = KV::new(repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut database = new_prove_database::<KV>(persistence);
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        database
            .set(key.clone(), Bytes::copy_from_slice(&data))
            .expect("Setting a key in Prove mode should succeed");

        let result = database
            .read_bytes(&key, offset, length)
            .expect("Reading a sub-range in Prove mode should succeed");
        prop_assert_eq!(result.as_slice(), &data[offset..offset + length]);

        // Zero-sized read at the end of the value.
        let result = database
            .read_bytes(&key, data.len(), 0)
            .expect("A zero-sized read at end of value should succeed");
        prop_assert_eq!(result.as_slice(), &[] as &[u8]);

        // Single-byte read of the last byte.
        let result = database
            .read_bytes(&key, data.len() - 1, 1)
            .expect("A partial read of the last byte should succeed");
        prop_assert_eq!(result.as_slice(), &data[data.len() - 1..]);

        // Offset beyond value length is rejected.
        prop_assert!(matches!(
            database.read_bytes(&key, data.len() + 1, 1),
            Err(Error::InvalidArgument(InvalidArgumentError::OffsetTooLarge))
        ));

        // IoRequestTooLarge takes priority over offset and key checks.
        prop_assert!(matches!(
            database.read_bytes(&key, 0, MAX_FILE_CHUNK_SIZE + 1),
            Err(Error::InvalidArgument(InvalidArgumentError::IoRequestTooLarge))
        ));

        database.into_trace()
    });

    kv_test!(test_prove_database_read_bytes_records_only_accessed_range, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();

        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        // Build an 8192-byte value (two PAGE_SIZE pages)
        let value: Vec<u8> = (0..(4 * (MAX_FILE_CHUNK_SIZE as u32)))
            .map(|i| (i & 0xff) as u8)
            .collect();

        let value_iter = &mut value.iter().cloned();

        let mut normal_db = new_database::<KV>(handle, &repo);
        for i in 0..4 {
            let offset = i * MAX_FILE_CHUNK_SIZE;
            let bytes: Vec<u8> = value_iter.take(MAX_FILE_CHUNK_SIZE).collect();
            normal_db
                .write(key.clone(), offset, Bytes::from(bytes))
                .expect("Writing to the db should succeed");
        }

        // Convert to Prove and read a small range entirely contained in page 1.
        let prove_db: TracedDatabase<KV, Prove<'static>> = normal_db
            .try_start_proof()
            .expect("starting proof should succeed");

        let read_offset = 5000usize;
        let read_len = 4usize;
        let read_back = prove_db
            .read_bytes(&key, read_offset, read_len)
            .expect("Prove-mode read should succeed");
        assert_eq!(
            read_back,
            &value[read_offset..read_offset + read_len],
            "Prove-mode read should return the requested bytes"
        );

        // Generate the proof and produce a Verify-mode database from it.
        let (prove_inner, prove_trace) = prove_db.into_parts();
        let verify_ml: MerkleLayer<KV, Verify> = prove_inner.inner.merkle.to_verify();
        let verify_db: Database<KV, Verify> = Database {
            inner: super::VerifyImpl { merkle: verify_ml },
        };

        // The recorded range round-trips: the proof contains page 1, so reading [5000..5004]
        // succeeds in Verify mode.
        let verified = catch_not_found(|| {
            verify_db
                .read_bytes(&key, read_offset, read_len)
                .expect("Verify-mode read of recorded range should not error")
                .as_ref()
                .to_vec()
        })
        .expect("Verify-mode read of recorded range must not trigger not_found");
        assert_eq!(verified, &value[read_offset..read_offset + read_len]);

        // Page 0 was never accessed, so it is omitted from the proof. Reading bytes from it in
        // Verify mode panics via `not_found` — proving that the Prove-mode read recorded only the
        // requested range, not the whole value.
        let unrecorded = catch_not_found(|| {
            let _ = verify_db.read_bytes(&key, 0, 4);
        });
        assert!(
            unrecorded.is_err(),
            "Verify-mode read of an un-recorded range must trigger not_found, \
             but the read succeeded — Prove-mode `get` is over-recording"
        );

        prove_trace
    });

    kv_test!(test_prove_database_write_append, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, persistence) = new_persistence::<KV>();
        let mut database = new_prove_database::<KV>(persistence);

        // Append non-empty data at offset == value_length.
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        database
            .set(key.clone(), Bytes::copy_from_slice(b"hello"))
            .expect("Setting a key in Prove mode should succeed");

        let written = database
            .write(key.clone(), 5, Bytes::copy_from_slice(b" world"))
            .expect("Appending to an existing value in Prove mode should succeed");
        assert_eq!(written, 6);
        assert_eq!(
            database
                .value_length(&key)
                .expect("Reading the length in Prove mode should succeed"),
            11
        );
        database.assert_database_value(&key, b"hello world");

        // Writing to a non-existent key at offset 0 creates it.
        let new_key = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        let written = database
            .write(new_key.clone(), 0, Bytes::copy_from_slice(b"fresh"))
            .expect("Writing a non-existent key at offset 0 in Prove mode should succeed");
        assert_eq!(written, 5);
        database.assert_database_value(&new_key, b"fresh");

        // Writing to a non-existent key at a non-zero offset is rejected.
        let other_key = Key::new(&[3]).expect("Size less than KEY_MAX_SIZE");
        assert!(matches!(
            database.write(other_key, 1, Bytes::copy_from_slice(&[])),
            Err(Error::InvalidArgument(InvalidArgumentError::OffsetTooLarge))
        ));

        // ValueSizeTooLarge is enforced before reaching the merkle layer.
        assert!(matches!(
            database.write(key, MAX_VALUE_SIZE, Bytes::copy_from_slice(&[1])),
            Err(Error::InvalidArgument(InvalidArgumentError::ValueSizeTooLarge))
        ));

        database.into_trace()
    });

    kv_test!(test_prove_database_hash, KV: BackgroundWriteableKeyValueStore, {
        let (_keepalive, persistence) = new_persistence::<KV>();
        let mut database = new_prove_database::<KV>(persistence);

        let key = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        let original_data = [1, 2, 3];
        let mutated_data = [3, 2, 1];

        database
            .set(key.clone(), Bytes::copy_from_slice(&original_data))
            .expect("Setting a key in Prove mode should succeed");

        let before = database
            .hash()
            .expect("Hashing in Prove mode should succeed");

        // Mutate the same key.
        database
            .set(key.clone(), Bytes::copy_from_slice(&mutated_data))
            .expect("Setting a key in Prove mode should succeed");

        let after = database
            .hash()
            .expect("Hashing in Prove mode should succeed");
        assert_ne!(before, after);

        // Revert the value of the same key and check that the hash reverts to its prior value.
        database
            .set(key, Bytes::copy_from_slice(&original_data))
            .expect("Setting a key in Prove mode should succeed");
        let reverted = database
            .hash()
            .expect("Hashing in Prove mode should succeed");
        assert_eq!(before, reverted);

        database.into_trace()
    });

    kv_test!(test_empty_db_noop_proof_readable_and_writeable, KV: BackgroundWriteableKeyValueStore, {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();

        let (_keepalive, repo) = KV::setup_repo();
        let database = new_database::<KV>(handle, &repo);

        let prove = database.try_start_proof().expect("starting a proof should succeed");
        let proof = MerkleProof::from_foldable(&prove);

        assert!(
            matches!(proof, octez_riscv_data::tree::Tree::Leaf(MerkleProofLeaf::Blind(_))),
            "database should be fully blinded");

        let mut verify = Database::<KV, Verify>::from_proof(ProofTree::present(&proof))
            .expect("Can convert blinded leaf proof into blinded verify database")
            .into_result();

        let key = Key::new(b"abc").unwrap();
        let data = vec![1, 2, 3];

        assert!(matches!(verify.exists(&key), Ok(false)), "can inspect empty blinded db");
        assert!(matches!(verify.set(key.clone(), Bytes::from(data)), Ok(())), "can modify an empty blinded db");
        assert!(matches!(verify.exists(&key), Ok(true)), "can inspect modified db from empty blinded db");
    });

    #[cfg(rocksdb)]
    kv_test!(
        #[should_panic(expected = "trace mismatch")]
        test_database_trace_comparison_detects_divergence,
        KV: BackgroundWriteableKeyValueStore,
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();
        let (_keepalive, repo) = KV::setup_repo();
        let mut database = new_database::<KV>(handle, &repo);

        // Force a per-backend divergence in the recorded trace
        let value = [KV::BACKEND as u8];
        let key = Key::new(b"divergence-probe").expect("Size less than KEY_MAX_SIZE");
        database
            .set(key, Bytes::copy_from_slice(&value))
            .expect("Setting should succeed");

        database.into_trace()
    });

    kv_test!(test_database_regression, KV: BackgroundPersistentKeyValueStore, {
        use goldenfile::Mint;

        use crate::test_helpers::REGRESSION_EXPECTED_DIR;
        use crate::test_helpers::REGRESSION_INPUTS_DIR;
        use crate::test_helpers::database::DatabaseOperation;

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
                    .is_some_and(|name| name.starts_with("database_") && name.ends_with(".input"))
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

            let file =
                std::fs::File::open(&path).expect("opening input file should succeed");
            let ops: Vec<DatabaseOperation> =
                serde_json::from_reader(file).expect("decoding JSON input should succeed");

            let (_keepalive, repo) = KV::setup_repo();
            let trace =
                crate::test_helpers::database::run_database_operations::<KV>(
                    &repo, ops,
                );

            let mut golden = mint
                .new_goldenfile(format!("{stem}.trace"))
                .expect("opening goldenfile should succeed");
            serde_json::to_writer_pretty(&mut golden, &trace)
                .expect("writing goldenfile should succeed");
        }
    });

    // Like `test_database_regression`, but runs a single input through
    // `run_and_prove_database_operations` so the trace additionally captures the proof
    // recorded for every supported operation.
    //
    // Defined as a separate test so that if only proofs diverge, this test fails but `test_database_regression`
    // still passes.
    kv_test!(test_database_proof_regression, KV: BackgroundPersistentKeyValueStore, {
        use goldenfile::Mint;

        use crate::test_helpers::REGRESSION_EXPECTED_DIR;
        use crate::test_helpers::REGRESSION_INPUTS_DIR;
        use crate::test_helpers::database::DatabaseOperation;

        let path = std::path::Path::new(REGRESSION_INPUTS_DIR).join("database_00.input");
        let file = std::fs::File::open(&path).expect("opening input file should succeed");
        let ops: Vec<DatabaseOperation> =
            serde_json::from_reader(file).expect("decoding JSON input should succeed");

        let (_keepalive, repo) = KV::setup_repo();
        let trace = crate::test_helpers::database::run_and_prove_database_operations::<
            KV,
        >(&repo, ops);

        let mut mint = Mint::new(REGRESSION_EXPECTED_DIR);
        let mut golden = mint
            .new_goldenfile("database_00.proof-trace")
            .expect("opening goldenfile should succeed");
        serde_json::to_writer_pretty(&mut golden, &trace)
            .expect("writing goldenfile should succeed");

        trace
    });

    // Like `test_database_proof_regression`, but over a hand-written sequence that grows a value
    // across many pages instead of a generated input.
    //
    // The generated inputs only ever hold values of at most `MAX_FILE_CHUNK_SIZE` bytes, whose
    // page tree is a single node for any arity, so their proofs cannot show a change to the shape
    // of the `Bytes` page tree — its arity or its page size. This trace can: every proof it
    // records opens a page tree several layers deep.
    kv_test!(test_database_bytes_proof_regression, KV: BackgroundPersistentKeyValueStore, {
        use goldenfile::Mint;

        use crate::test_helpers::REGRESSION_EXPECTED_DIR;
        use crate::test_helpers::database::multi_page_value_operations;

        let (_keepalive, repo) = KV::setup_repo();
        let trace = crate::test_helpers::database::run_and_prove_database_operations::<
            KV,
        >(&repo, multi_page_value_operations());

        let mut mint = Mint::new(REGRESSION_EXPECTED_DIR);
        let mut golden = mint
            .new_goldenfile("database_bytes.proof-trace")
            .expect("opening goldenfile should succeed");
        serde_json::to_writer_pretty(&mut golden, &trace)
            .expect("writing goldenfile should succeed");

        trace
    });

    kv_test!(test_database_end_to_end, KV: BackgroundPersistentKeyValueStore,
    [
        generated in <crate::test_helpers::database::DatabaseOperationView as crate::test_helpers::OperationView>::operations_commit_checkout_strategy(1usize..100, 0.1)
    ],
    {
        // Every test iteration expects an empty repo, so not setting it in a `setup` block.
        // This is because a repo preserves database commits from previous test runs, resulting
        // in test failures when checking out a commit which isn't expected to exist succeeds.
        let (_keepalive, repo) = KV::setup_repo();

        let (keys, values, ops_a, ops_b) = generated;

        // Pick an operations vector so each backend exercises a different
        // `CommitCheckoutRoundtrip` placement against the same base operations.
        let ops = match KV::BACKEND {
            crate::storage::Backend::Persistent => ops_a,
            crate::storage::Backend::InMemory => ops_b,
        };
        let operations =
            crate::test_helpers::database::make_database_operations(keys, values, ops);

        // A proof is generated for vvery supported operation, which is then verified
        // and recorded in the trace.
        crate::test_helpers::database::run_and_prove_database_operations::<KV>(
            &repo, operations,
        )
    });
}
