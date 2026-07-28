// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Asynchronous Merkle layer worker
//!
//! This module provides a Merkle worker that processes Merkle layer commands asynchronously in a
//! background thread. It allows non-blocking `set`, `write` and `delete` operations while still
//! providing synchronous access to `hash` and `commit` operations.

use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use trait_set::trait_set;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::merkle_layer::MerkleLayer;
use crate::storage::PersistentKeyValueStore;
use crate::storage::ReadableKeyValueStore;
use crate::storage::StoreOptions;
use crate::storage::WriteableKeyValueStore;

trait_set! {
    /// [`ReadableKeyValueStore`] that can be used in a background thread
    pub trait BackgroundReadableKeyValueStore = ReadableKeyValueStore + Send + Sync + 'static;

    /// [`WriteableKeyValueStore`] that can be used in a background thread
    pub trait BackgroundWriteableKeyValueStore = WriteableKeyValueStore + BackgroundReadableKeyValueStore;

    /// [`PersistentKeyValueStore`] that can be used in a background thread
    pub trait BackgroundPersistentKeyValueStore = PersistentKeyValueStore + BackgroundWriteableKeyValueStore;
}

/// The Merkle representation held by a Normal-mode database, selected by its key-value store
/// through [`ReadableKeyValueStore::Merkle`].
///
/// A store that can be written to needs a live tree to track the root hash of the writes made
/// against it, and hence a [`MerkleWorker`] to own that tree. A store that reads a commit in place
/// cannot change its root hash, so it needs nothing beyond the hash itself - see [`CommittedRoot`].
pub trait MerkleHandle<KV> {
    /// Obtain, and possibly calculate, the root hash of the Merkle tree.
    fn hash(&self) -> Result<Hash, OperationalError>;

    /// Snapshot the tree and enter Prove mode - see [`MerkleLayer::start_proof`].
    fn start_proof(
        &self,
        store: Arc<KV>,
    ) -> Result<MerkleLayer<KV, Prove<'static>>, OperationalError>;
}

/// A store whose Merkle representation is a live tree, owned by a [`MerkleWorker`].
///
/// That is the shape of every store a database can be written through, and it is what
/// distinguishes them from the read-only stores, which hold a [`CommittedRoot`] instead.
pub trait TreeBackedKeyValueStore: ReadableKeyValueStore<Merkle = MerkleWorker<Self>> {}

impl<KV: ReadableKeyValueStore<Merkle = MerkleWorker<KV>>> TreeBackedKeyValueStore for KV {}

/// The Merkle representation of a database that reads a commit in place.
///
/// A [`CommitId`] *is* the Merkle root hash of the database committed under it - see
/// [`MerkleLayer::commit`] - so a database that can only ever read that one commit already knows
/// its root hash, and needs neither a tree nor a worker thread to answer for it. Proving requires a
/// tree, so [`MerkleHandle::start_proof`] loads one on demand.
#[derive(Debug, Clone, Copy)]
pub struct CommittedRoot(CommitId);

impl CommittedRoot {
    /// The commit this database reads.
    pub(crate) fn commit_id(&self) -> CommitId {
        self.0
    }
}

impl From<CommitId> for CommittedRoot {
    fn from(commit: CommitId) -> Self {
        Self(commit)
    }
}

impl<KV: ReadableKeyValueStore> MerkleHandle<KV> for CommittedRoot {
    fn hash(&self) -> Result<Hash, OperationalError> {
        Ok(*self.0.as_hash())
    }

    fn start_proof(
        &self,
        store: Arc<KV>,
    ) -> Result<MerkleLayer<KV, Prove<'static>>, OperationalError> {
        Ok(MerkleLayer::checkout(store, self.0)?.start_proof())
    }
}

/// Alias for the inner workings of the [`Command`] struct to make Clippy happy
type DynCommand<KV> = dyn FnOnce(&mut MerkleLayer<KV, Normal>, &mut BTreeSet<Key>) + Send;

/// Commands that will be sent to the background worker thread to manipulate the Merkle layer
///
/// # Race Conditions
///
/// As these are handled in a background thread, there are potential race conditions between the
/// background worker, and the persistence layer. The `MerkleLayer` resolves values lazily from the
/// `WriteableKeyValueStore` - and as a result can attempt to perform operations over unexpected, or
/// incorrectly shaped, values.
///
/// ## Out-of-order delete
///
/// One such race condition is that a delete operation may already have been performed in the
/// `WriteableKeyValueStore`, prior to a _previous_ write/set on that value being handled in the Merkle
/// layer. This can happen when, on the first `set/write`, the value needs to be resolved (loaded).
/// This takes time to happen - and is fully possible (if many intermediate nodes need to be resolved
/// first), that a subsequent delete operation has already been handled by the persistence layer.
///
/// This results in the write/set failing with [`OperationalError::CommitValueMissing`]. This is
/// less concerning than it first appears, however, as we know the Merkle layer will subsequently
/// handle the delete that caused the issue! Therefore, the delete does in fact restore the
/// consistency of the Merkle layer - and there is no need to crash the worker when such errors
/// occur.
///
/// ## Out-of-order set
///
/// Writes can also fail in a similar way when a subsequent `set` operation reduces the length
/// of the value stored - and the offset of the write is larger than the new value's length.
///
/// Once again, this is not as concerning as it might appear. The write may indeed fail, but the
/// subsequent set will be handled by the Merkle layer, fully overwriting any value that would have
/// been written anyway.
///
/// ## Eventual consistency
///
/// We do not wish to enforce 'full-synchronisation' on every operation - as this would lose the
/// performance gained by allowing the Merkle layer to 'catch-up' in the background thread.
///
/// The Merkle layer _will_, however, be fully-consistent at every point that matters: ie on
/// `Hash` and `Commit`. This is because synchronisation is forced on these operations - and so
/// if we encountered the above race conditions, the operation that caused (and resolves) them
/// _must_ occur before the synchronisation points. If it occurs after, then the race condition
/// cannot have happened to begin with: as all the prior operations will be fully handled first,
/// before any problematic operations can occur.
///
/// To ensure this condition is upheld, the Merkle worker tracks potentially inconsistent keys,
/// removing them once operations restoring consistency are handled. We ensure that no inconsistent
/// keys exist, when at synchronisation points.
struct Command<KV>(Box<DynCommand<KV>>);

impl<KV> Command<KV> {
    /// Apply this command to the Merkle layer.
    fn apply(self, layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>) {
        self.0(layer, consistency);
    }

    /// Construct a command that performs a [`MerkleLayer::write`].
    fn new_write(key: Key, offset: usize, value: Bytes) -> Self
    where
        KV: ReadableKeyValueStore,
    {
        Self(Box::new(
            move |layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>| match layer
                .write(&key, offset, &value)
            {
                Err(Error::Operational(OperationalError::CommitValueMissing {
                    key: missing_key,
                    source: _,
                })) if key == missing_key => {
                    // mark key as inconsistent
                    consistency.insert(key);
                }
                Err(Error::InvalidArgument(InvalidArgumentError::OffsetTooLarge)) => {
                    // mark key as inconsistent
                    consistency.insert(key);
                }
                Ok(()) => {
                    consistency.remove(&key);
                }
                Err(error) => panic!("Writing to the Merkle layer should succeed, got {error}"),
            },
        ))
    }

    /// Construct a command that performs a [`MerkleLayer::set`].
    fn new_set(key: Key, value: Bytes) -> Self
    where
        KV: ReadableKeyValueStore,
    {
        Self(Box::new(
            move |layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>| match layer
                .set(&key, &value)
            {
                Err(OperationalError::CommitValueMissing {
                    key: missing_key,
                    source: _,
                }) if key == missing_key => {
                    // mark key as inconsistent
                    consistency.insert(key);
                }
                Ok(()) => {
                    consistency.remove(&key);
                }
                Err(error) => panic!("Setting in the Merkle layer should succeed, got {error}"),
            },
        ))
    }

    /// Construct a command that performs a [`MerkleLayer::delete`].
    fn new_delete(key: Key) -> Self
    where
        KV: ReadableKeyValueStore,
    {
        Self(Box::new(
            move |layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>| {
                layer
                    .delete(&key)
                    .expect("Deleting from the Merkle layer should succeed.");

                // delete always restores consistency
                consistency.remove(&key);
            },
        ))
    }

    /// Construct a command that performs a [`MerkleLayer::clone_with`].
    fn new_clone_with(
        store: Arc<KV>,
    ) -> (
        impl FnOnce() -> Result<MerkleLayer<KV, Normal>, OperationalError>,
        Self,
    )
    where
        KV: BackgroundReadableKeyValueStore,
    {
        let (sender, receiver) = oneshot::channel();

        let this = Self(Box::new(
            move |layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>| {
                assert!(
                    consistency.is_empty(),
                    "Inconsistent layer on clone: {consistency:?}"
                );

                let result = layer.clone_with(store);
                let _ = sender.send(result);
            },
        ));

        let receive = || {
            receiver
                .blocking_recv()
                .map_err(|_error| OperationalError::WorkerThreadDied)
        };

        (receive, this)
    }

    /// Construct a command that performs a [`MerkleLayer::hash`].
    fn new_hash() -> (impl FnOnce() -> Result<Hash, OperationalError>, Self)
    where
        KV: ReadableKeyValueStore,
    {
        let (sender, receiver) = oneshot::channel();

        let this = Self(Box::new(
            move |layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>| {
                assert!(
                    consistency.is_empty(),
                    "Inconsistent layer on hash: {consistency:?}"
                );

                let result = layer.hash();
                let _ = sender.send(result);
            },
        ));

        let receive = || {
            receiver
                .blocking_recv()
                .map_err(|_error| OperationalError::WorkerThreadDied)
        };

        (receive, this)
    }

    /// Construct a command that performs a [`MerkleLayer::commit`].
    fn new_commit(
        options: StoreOptions,
    ) -> (impl FnOnce() -> Result<CommitId, OperationalError>, Self)
    where
        KV: PersistentKeyValueStore,
    {
        let (sender, receiver) = oneshot::channel();

        let this = Self(Box::new(
            move |layer: &mut MerkleLayer<KV, Normal>, consistency: &mut BTreeSet<Key>| {
                assert!(
                    consistency.is_empty(),
                    "Inconsistent layer on commit: {consistency:?}"
                );

                let result = layer.commit(&options);
                let _ = sender.send(result);
            },
        ));

        let receive = || {
            receiver
                .blocking_recv()
                .map_err(|_error| OperationalError::WorkerThreadDied)?
        };

        (receive, this)
    }
}

/// Merkle worker that processes commands asynchronously in a background thread
///
/// It works like the [`MerkleLayer`] but does not block on `set`, `write` and `delete` operations.
pub struct MerkleWorker<KV> {
    /// Send end of the command channel that is connected to the background worker thread
    sender: mpsc::UnboundedSender<Command<KV>>,
}

impl<KV> MerkleWorker<KV> {
    /// Create a new Merkle worker with an empty Merkle tree.
    ///
    /// The provided handle is used to spawn the background worker thread.
    pub fn new(async_handle: &Handle, store: Arc<KV>) -> Self
    where
        KV: BackgroundReadableKeyValueStore,
    {
        let layer = MerkleLayer::new(store);
        MerkleWorker::from_layer(async_handle, layer)
    }

    /// Create a Merkle worker from an existing Merkle layer.
    ///
    /// The provided handle is used to spawn the background worker thread.
    pub(crate) fn from_layer(async_handle: &Handle, layer: MerkleLayer<KV, Normal>) -> Self
    where
        KV: Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();

        async_handle.spawn(async move {
            let mut layer = layer;
            let mut consistency = BTreeSet::new();

            let mut receiver: mpsc::UnboundedReceiver<Command<KV>> = receiver;

            while let Some(cmd) = receiver.recv().await {
                cmd.apply(&mut layer, &mut consistency);
            }
        });

        MerkleWorker { sender }
    }

    /// See [`MerkleLayer::clone_with`].
    pub(crate) fn clone_with(
        &self,
        handle: &Handle,
        store: Arc<KV>,
    ) -> Result<Self, OperationalError>
    where
        KV: BackgroundReadableKeyValueStore,
    {
        let (receive, command) = Command::new_clone_with(store);
        self.sender
            .send(command)
            .map_err(|_error| OperationalError::WorkerThreadDied)?;

        let layer = receive()?;
        let worker = Self::from_layer(handle, layer);

        Ok(worker)
    }

    /// Non-blocking version of [`MerkleLayer::write`].
    pub(crate) fn write(
        &self,
        key: Key,
        offset: usize,
        value: Bytes,
    ) -> Result<(), OperationalError>
    where
        KV: ReadableKeyValueStore,
    {
        let command = Command::new_write(key, offset, value);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)
    }

    /// Non-blocking version of [`MerkleLayer::set`].
    pub(crate) fn set(&self, key: Key, value: Bytes) -> Result<(), OperationalError>
    where
        KV: ReadableKeyValueStore,
    {
        let command = Command::new_set(key, value);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)
    }

    /// Non-blocking version of [`MerkleLayer::delete`].
    pub(crate) fn delete(&self, key: Key) -> Result<(), OperationalError>
    where
        KV: ReadableKeyValueStore,
    {
        let command = Command::new_delete(key);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)
    }

    /// Checkout a Merkle worker from an existing commit.
    ///
    /// The provided handle is used to spawn the background worker thread.
    pub(crate) fn checkout(
        async_handle: &Handle,
        store: Arc<KV>,
        commit: CommitId,
    ) -> Result<Self, OperationalError>
    where
        KV: BackgroundReadableKeyValueStore,
    {
        let layer = MerkleLayer::checkout(store, commit)?;
        let worker = MerkleWorker::from_layer(async_handle, layer);
        Ok(worker)
    }

    /// See [`MerkleLayer::commit`].
    pub(crate) fn commit(&self, options: StoreOptions) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        let (receive, command) = Command::new_commit(options);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)?;

        receive()
    }
}

impl<KV: BackgroundReadableKeyValueStore> MerkleHandle<KV> for MerkleWorker<KV> {
    fn hash(&self) -> Result<Hash, OperationalError> {
        let (receive, command) = Command::new_hash();
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)?;

        receive()
    }

    fn start_proof(
        &self,
        store: Arc<KV>,
    ) -> Result<MerkleLayer<KV, Prove<'static>>, OperationalError> {
        let (receive, command) = Command::new_clone_with(store);
        self.sender
            .send(command)
            .map_err(|_error| OperationalError::WorkerThreadDied)?;

        let layer = receive()?;

        Ok(layer.start_proof())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use proptest::prelude::Just;
    use proptest::prelude::Strategy;
    use proptest::prop_assert_eq;
    use tokio::runtime::Handle;

    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;
    use crate::merkle_layer::MerkleLayer;
    use crate::merkle_worker::MerkleHandle;
    use crate::merkle_worker::MerkleWorker;
    use crate::storage::WriteableKeyValueStore;
    use crate::storage::kv_test;

    fn key_strategy() -> impl Strategy<Value = Key> {
        proptest::collection::vec(proptest::arbitrary::any::<u8>(), 1..KEY_MAX_SIZE).prop_map(
            |bytes| {
                Key::new(bytes.as_slice()).expect("Creating a key from valid bytes should succeed")
            },
        )
    }

    fn value_strategy() -> impl Strategy<Value = Bytes> {
        proptest::collection::vec(proptest::arbitrary::any::<u8>(), 0..8192).prop_map(Bytes::from)
    }

    #[derive(Debug, Clone)]
    enum TestCommand {
        Write {
            key: Key,
            offset: usize,
            value: Bytes,
        },
        Set {
            key: Key,
            value: Bytes,
        },
        Delete {
            key: Key,
        },
        Hash,
        #[cfg(rocksdb)]
        Commit,
        Clone,
    }

    impl TestCommand {
        fn run<KV: super::BackgroundPersistentKeyValueStore>(
            self,
            handle: &Handle,
            repo: &KV::Repo,
            worker: &mut MerkleWorker<KV>,
            layer: &mut MerkleLayer<KV, Normal>,
        ) {
            match self {
                Self::Write { key, offset, value } => {
                    layer
                        .write(&key, offset, &value)
                        .expect("Write should succeed.");
                    worker.write(key, offset, value).unwrap();
                }

                Self::Set { key, value } => {
                    layer.set(&key, &value).expect("Set should succeed.");
                    worker.set(key, value).unwrap();
                }

                Self::Delete { key } => {
                    layer.delete(&key).expect("Delete should succeed.");
                    worker.delete(key).unwrap();
                }

                Self::Hash => {
                    let hash1 = worker.hash().unwrap();
                    let hash2 = layer.hash();
                    assert_eq!(hash1, hash2);
                }

                #[cfg(rocksdb)]
                Self::Commit => {
                    let options = crate::storage::StoreOptions::default().with_node_data();
                    let commit2 = layer.commit(&options).expect("Commit should succeed");
                    let commit1 = worker.commit(options).expect("Commit should succeed");
                    assert_eq!(commit1, commit2);
                }

                Self::Clone => {
                    let persistence_layer =
                        KV::new(repo).expect("Creating a persistence layer should succeed");
                    let persistence_layer = Arc::new(persistence_layer);
                    *layer = layer.clone_with(persistence_layer);

                    let persistence_worker =
                        KV::new(repo).expect("Creating a persistence layer should succeed");
                    let persistence_worker = Arc::new(persistence_worker);
                    *worker = worker
                        .clone_with(handle, persistence_worker)
                        .expect("Cloning a Merkle worker should succeed");
                }
            }
        }

        fn strategy() -> impl Strategy<Value = Self> {
            let write = (key_strategy(), value_strategy())
                // Writing to a non-zero offset requires an existing value. Rather than bookkeeping
                // keys, we just stick to a zero offset and leave testing non-zero offsets to the
                // Merkle layer tests.
                .prop_map(|(key, value)| TestCommand::Write {
                    key,
                    offset: 0,
                    value,
                });

            let set = (key_strategy(), value_strategy())
                .prop_map(|(key, value)| TestCommand::Set { key, value });

            let delete = key_strategy().prop_map(|key| TestCommand::Delete { key });

            let hash = Just(TestCommand::Hash);

            #[cfg(rocksdb)]
            let commit = Just(TestCommand::Commit);

            let clone = Just(TestCommand::Clone);

            // The frequencies are chosen to reflect a typical workload.
            cfg_if::cfg_if! {
                if #[cfg(rocksdb)] {
                    proptest::prop_oneof![
                        250 => write,
                        250 => set,
                        50 => delete,
                        10 => clone,
                        10 => hash,
                        1 => commit,
                    ]
                } else {
                    proptest::prop_oneof![
                        25 => write,
                        25 => set,
                        5 => delete,
                        1 => clone,
                        1 => hash,
                    ]
                }
            }
        }
    }

    kv_test!(commands, KV: super::BackgroundPersistentKeyValueStore,
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
        commands in proptest::collection::vec(TestCommand::strategy(), 1..100),
    ], {
        let persistence_layer = KV::new(repo).expect("Creating a persistence layer should succeed");
        let persistence_layer = Arc::new(persistence_layer);
        let mut merkle_layer = MerkleLayer::new(persistence_layer);

        let persistence_worker = KV::new(repo).expect("Creating a persistence layer should succeed");
        let persistence_worker = Arc::new(persistence_worker);
        let mut merkle_worker = MerkleWorker::new(handle, persistence_worker);

        for command in commands {
            command.run::<KV>(handle, repo, &mut merkle_worker, &mut merkle_layer);
        }

        let layer_hash = merkle_layer.hash();
        let worker_hash = merkle_worker.hash().unwrap();
        prop_assert_eq!(layer_hash, worker_hash);
    });
}
