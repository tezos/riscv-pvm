// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Asynchronous Merkle layer worker
//!
//! This module provides a Merkle worker that processes Merkle layer commands asynchronously in a
//! background thread. It allows non-blocking `set`, `write` and `delete` operations while still
//! providing synchronous access to `hash` and `commit` operations.

use std::sync::Arc;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use trait_set::trait_set;

use crate::commit::CommitId;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::merkle_layer::MerkleLayer;
use crate::storage::KeyValueStore;
use crate::storage::PersistentKeyValueStore;

trait_set! {
    /// [`KeyValueStore`] that can be used in a background thread
    pub trait BackgroundKeyValueStore = KeyValueStore + Send + Sync + 'static;

    /// [`PersistentKeyValueStore`] that can be used in a background thread
    pub trait BackgroundPersistentKeyValueStore = PersistentKeyValueStore + BackgroundKeyValueStore;
}

/// Alias for the inner workings of the [`Command`] struct to make Clippy happy
type DynCommand<KV> = dyn FnOnce(&mut MerkleLayer<KV, Normal>) + Send;

/// Commands that will be sent to the background worker thread to manipulate the Merkle layer
struct Command<KV>(Box<DynCommand<KV>>);

impl<KV> Command<KV> {
    /// Apply this command to the Merkle layer.
    fn apply(self, layer: &mut MerkleLayer<KV, Normal>) {
        self.0(layer);
    }

    /// Construct a command that performs a [`MerkleLayer::write`].
    fn new_write(key: Key, offset: usize, value: Bytes) -> Self
    where
        KV: KeyValueStore,
    {
        Self(Box::new(move |layer: &mut MerkleLayer<KV, Normal>| {
            layer
                .write(&key, offset, &value)
                .expect("Writing to the Merkle layer should succeed.");
        }))
    }

    /// Construct a command that performs a [`MerkleLayer::set`].
    fn new_set(key: Key, value: Bytes) -> Self
    where
        KV: KeyValueStore,
    {
        Self(Box::new(move |layer: &mut MerkleLayer<KV, Normal>| {
            layer
                .set(&key, &value)
                .expect("Setting on the Merkle layer should succeed.");
        }))
    }

    /// Construct a command that performs a [`MerkleLayer::delete`].
    fn new_delete(key: Key) -> Self
    where
        KV: KeyValueStore,
    {
        Self(Box::new(move |layer: &mut MerkleLayer<KV, Normal>| {
            layer
                .delete(&key)
                .expect("Deleting from the Merkle layer should succeed.");
        }))
    }

    /// Construct a command that performs a [`MerkleLayer::try_clone_with`].
    fn new_clone_with(
        store: Arc<KV>,
    ) -> (
        impl FnOnce() -> Result<MerkleLayer<KV, Normal>, OperationalError>,
        Self,
    )
    where
        KV: Send + Sync + 'static,
    {
        let (sender, receiver) = oneshot::channel();

        let this = Self(Box::new(move |layer: &mut MerkleLayer<KV, Normal>| {
            let result = layer.try_clone_with(store);
            let _ = sender.send(result);
        }));

        let receive = || {
            receiver
                .blocking_recv()
                .map_err(|_error| OperationalError::WorkerThreadDied)
        };

        (receive, this)
    }

    /// Construct a command that performs a [`MerkleLayer::hash`].
    fn new_hash() -> (impl FnOnce() -> Result<Hash, OperationalError>, Self) {
        let (sender, receiver) = oneshot::channel();

        let this = Self(Box::new(move |layer: &mut MerkleLayer<KV, Normal>| {
            let result = layer.hash();
            let _ = sender.send(result);
        }));

        let receive = || {
            receiver
                .blocking_recv()
                .map_err(|_error| OperationalError::WorkerThreadDied)?
        };

        (receive, this)
    }

    /// Construct a command that performs a [`MerkleLayer::commit`].
    fn new_commit() -> (impl FnOnce() -> Result<CommitId, OperationalError>, Self)
    where
        KV: PersistentKeyValueStore,
    {
        let (sender, receiver) = oneshot::channel();

        let this = Self(Box::new(move |layer: &mut MerkleLayer<KV, Normal>| {
            let result = layer.commit();
            let _ = sender.send(result);
        }));

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
        KV: Send + Sync + 'static,
    {
        let layer = MerkleLayer::new(store);
        MerkleWorker::from_layer(async_handle, layer)
    }

    /// Create a Merkle worker from an existing Merkle layer.
    ///
    /// The provided handle is used to spawn the background worker thread.
    fn from_layer(async_handle: &Handle, layer: MerkleLayer<KV, Normal>) -> Self
    where
        KV: Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();

        async_handle.spawn(async move {
            let mut layer = layer;
            let mut receiver: mpsc::UnboundedReceiver<Command<KV>> = receiver;

            while let Some(cmd) = receiver.recv().await {
                cmd.apply(&mut layer);
            }
        });

        MerkleWorker { sender }
    }

    /// See [`MerkleLayer::try_clone_with`].
    pub(crate) fn clone_with(
        &self,
        handle: &Handle,
        store: Arc<KV>,
    ) -> Result<Self, OperationalError>
    where
        KV: Send + Sync + 'static,
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
        KV: KeyValueStore,
    {
        let command = Command::new_write(key, offset, value);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)
    }

    /// Non-blocking version of [`MerkleLayer::set`].
    pub(crate) fn set(&self, key: Key, value: Bytes) -> Result<(), OperationalError>
    where
        KV: KeyValueStore,
    {
        let command = Command::new_set(key, value);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)
    }

    /// Non-blocking version of [`MerkleLayer::delete`].
    pub(crate) fn delete(&self, key: Key) -> Result<(), OperationalError>
    where
        KV: KeyValueStore,
    {
        let command = Command::new_delete(key);
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)
    }

    /// See [`MerkleLayer::hash`].
    pub(crate) fn hash(&self) -> Result<Hash, OperationalError>
    where
        KV: KeyValueStore,
    {
        let (receive, command) = Command::new_hash();
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)?;

        receive()
    }

    /// Checkout a Merkle worker from an existing commit.
    ///
    /// The provided handle is used to spawn the background worker thread.
    #[expect(
        dead_code,
        reason = "Checkout functionality is currently not hooked up upstream"
    )]
    pub(crate) fn checkout(
        async_handle: &Handle,
        store: Arc<KV>,
        commit: CommitId,
    ) -> Result<Self, OperationalError>
    where
        KV: BackgroundPersistentKeyValueStore,
    {
        let layer = MerkleLayer::checkout(store, commit)?;
        let worker = MerkleWorker::from_layer(async_handle, layer);
        Ok(worker)
    }

    /// See [`MerkleLayer::commit`].
    pub(crate) fn commit(&self) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        let (receive, command) = Command::new_commit();
        self.sender
            .send(command)
            .map_err(|_| OperationalError::WorkerThreadDied)?;

        receive()
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
    use crate::merkle_worker::MerkleWorker;
    use crate::persistence_layer::PersistenceLayer;
    use crate::repo::DirectoryManager;
    use crate::storage::KeyValueStore;

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
        Commit,
        Clone,
    }

    impl TestCommand {
        fn run(
            self,
            handle: &Handle,
            dir_manager: &DirectoryManager,
            worker: &mut MerkleWorker<PersistenceLayer>,
            layer: &mut MerkleLayer<PersistenceLayer, Normal>,
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
                    let hash2 = layer
                        .hash()
                        .expect("Resolving the tree for the hash should succeed.");
                    assert_eq!(hash1, hash2);
                }

                Self::Commit => {
                    let commit1 = worker.commit().expect("Commit should succeed");
                    let commit2 = layer.commit().expect("Commit should succeed");
                    assert_eq!(commit1, commit2);
                }

                Self::Clone => {
                    let persistence_layer = PersistenceLayer::new(dir_manager)
                        .expect("Creating a persistence layer should succeed");
                    let persistence_layer = Arc::new(persistence_layer);
                    *layer = layer.try_clone_with(persistence_layer);

                    let persistence_worker = PersistenceLayer::new(dir_manager)
                        .expect("Creating a persistence layer should succeed");
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

            let commit = Just(TestCommand::Commit);

            let clone = Just(TestCommand::Clone);

            // The frequencies are chosen to reflect a typical workload.
            proptest::prop_oneof![
                250 => write,
                250 => set,
                50 => delete,
                10 => clone,
                10 => hash,
                1 => commit,
            ]
        }
    }

    #[test]
    fn commands() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .expect("Creating a Tokio runtime should succeed");
        let handle = runtime.handle();

        let tmp_dir = tempfile::tempdir().expect("Creating a temporary directory should succeed");
        let dir_manager = DirectoryManager::new(tmp_dir.path())
            .expect("Creating the directory manager should succeed");

        proptest::proptest!(|(commands in proptest::collection::vec(TestCommand::strategy(), 1..100))| {
            let persistence_layer = PersistenceLayer::new(&dir_manager).expect("Creating a persistence layer should succeed");
            let persistence_layer = Arc::new(persistence_layer);
            let mut merkle_layer = MerkleLayer::new(persistence_layer);

            let persistence_worker = PersistenceLayer::new(&dir_manager).expect("Creating a persistence layer should succeed");
            let persistence_worker = Arc::new(persistence_worker);
            let mut merkle_worker = MerkleWorker::new(handle, persistence_worker);

            for command in commands {
                command.run(handle, &dir_manager, &mut merkle_worker, &mut merkle_layer);
            }

            let layer_hash = merkle_layer.hash().expect("Resolving the tree should succeed.");
            let worker_hash = merkle_worker.hash().unwrap();
            prop_assert_eq!(layer_hash, worker_hash);
        });
    }
}
