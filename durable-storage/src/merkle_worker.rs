// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Asynchronous Merkle layer worker
//!
//! This module provides a Merkle worker that processes Merkle layer commands asynchronously in a
//! background thread. It allows non-blocking `set` and `delete` operations while still providing
//! synchronous access to `hash` and `commit` operations.

use std::sync::Arc;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::key::Key;
use crate::merkle_layer::CommitId;
use crate::merkle_layer::MerkleLayer;
use crate::merkle_layer::MerkleLayerError;
use crate::merkle_layer::node_resolver::LazyMavlNodeResolver;
use crate::persistence_layer::PersistenceLayer;

/// Commands that can be sent to the Merkle worker background thread
enum Command {
    /// Set the value associated with a key.
    Set { key: Key, value: Bytes },

    /// Delete a key-value pair.
    Delete { key: Key },

    /// Obtain the root hash of the Merkle tree.
    Hash {
        /// The background thread will write its response to this one-shot channel.
        response: oneshot::Sender<Hash>,
    },

    /// Flush the current Merkle state to the persistence layer and obtain a commit ID.
    Commit {
        /// The background thread will write its response to this one-shot channel.
        response: oneshot::Sender<Result<CommitId, MerkleLayerError>>,
    },

    /// Clone the current Merkle layer.
    Clone {
        /// The persistence layer to use for the cloned Merkle layer.
        persistence_layer: Arc<PersistenceLayer>,

        /// The background thread will write its response to this one-shot channel.
        response: oneshot::Sender<Result<MerkleLayer<LazyMavlNodeResolver>, MerkleLayerError>>,
    },
}

/// Merkle worker that processes commands asynchronously in a background thread
///
/// It works like the [`MerkleLayer`] but does not block on `set` and `delete` operations.
pub struct MerkleWorker {
    /// Send end of the command channel that is connected to the background worker thread
    sender: mpsc::UnboundedSender<Command>,
}

/// Errors that can occur when interacting with the Merkle worker
#[derive(Debug, thiserror::Error)]
pub enum MerkleWorkerError {
    #[error("Merkle layer error: {0}")]
    MerkleLayerError(#[from] MerkleLayerError),
}

impl MerkleWorker {
    /// Create a new Merkle worker with an empty Merkle tree.
    ///
    /// The provided handle is used to spawn the background worker thread.
    pub fn new(
        async_handle: &Handle,
        persistence_layer: Arc<PersistenceLayer>,
    ) -> Result<Self, MerkleWorkerError> {
        let node_resolver = Arc::new(LazyMavlNodeResolver::new(persistence_layer.clone()));
        let layer = MerkleLayer::new(persistence_layer, node_resolver);
        let worker = MerkleWorker::from_layer(async_handle, layer);
        Ok(worker)
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
        persistence_layer: Arc<PersistenceLayer>,
        hash: Hash,
    ) -> Result<Self, MerkleWorkerError> {
        let layer = MerkleLayer::checkout(persistence_layer, hash)?;
        let worker = MerkleWorker::from_layer(async_handle, layer);
        Ok(worker)
    }

    /// See [`MerkleLayer::clone_with`].
    pub(crate) fn clone_with(
        &self,
        handle: &Handle,
        persistence_layer: Arc<PersistenceLayer>,
    ) -> Result<Self, MerkleWorkerError> {
        let (sender, receiver) = oneshot::channel();

        self.sender
            .send(Command::Clone {
                persistence_layer,
                response: sender,
            })
            .expect("Merkle worker should be alive");

        let layer = receiver
            .blocking_recv()
            .expect("Merkle worker should be alive")?;

        let worker = Self::from_layer(handle, layer);
        Ok(worker)
    }

    /// Create a Merkle worker from an existing Merkle layer.
    ///
    /// The provided handle is used to spawn the background worker thread.
    fn from_layer(async_handle: &Handle, layer: MerkleLayer<LazyMavlNodeResolver>) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();

        async_handle.spawn(async move {
            let mut layer = layer;
            let mut receiver = receiver;

            while let Some(cmd) = receiver.recv().await {
                match cmd {
                    Command::Set { key, value } => layer.set(&key, value),

                    Command::Delete { key } => layer.delete(&key),

                    Command::Hash { response } => {
                        let hash = layer.hash();
                        let _ = response.send(hash);
                    }

                    Command::Commit { response } => {
                        let result = layer.commit();
                        let _ = response.send(result);
                    }

                    Command::Clone {
                        persistence_layer,
                        response,
                    } => {
                        let result = layer.clone_with(persistence_layer);
                        let _ = response.send(result);
                    }
                }
            }
        });

        MerkleWorker { sender }
    }

    /// Non-blocking version of [`MerkleLayer::set`].
    pub(crate) fn set(&self, key: Key, value: Bytes) {
        self.sender
            .send(Command::Set { key, value })
            .expect("Merkle worker should be alive");
    }

    /// Non-blocking version of [`MerkleLayer::delete`].
    pub(crate) fn delete(&self, key: Key) {
        self.sender
            .send(Command::Delete { key })
            .expect("Merkle worker should be alive");
    }

    /// See [`MerkleLayer::hash`].
    pub(crate) fn hash(&self) -> Hash {
        let (sender, receiver) = oneshot::channel();

        self.sender
            .send(Command::Hash { response: sender })
            .expect("Merkle worker should be alive");

        receiver
            .blocking_recv()
            .expect("Merkle worker should be alive")
    }

    /// See [`MerkleLayer::commit`].
    #[cfg_attr(not(test), expect(dead_code, reason = "Used in RV-827"))]
    pub(crate) fn commit(&self) -> Result<CommitId, MerkleWorkerError> {
        let (sender, receiver) = oneshot::channel();

        self.sender
            .send(Command::Commit { response: sender })
            .expect("Merkle worker should be alive");

        let result = receiver
            .blocking_recv()
            .expect("Merkle worker should be alive")?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use proptest::prelude::Just;
    use proptest::prelude::Strategy;
    use proptest::prop_assert_eq;
    use tokio::runtime::Handle;

    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;
    use crate::merkle_layer::MerkleLayer;
    use crate::merkle_layer::node_resolver::InMemoryMavlNodeResolver;
    use crate::merkle_layer::node_resolver::LazyMavlNodeResolver;
    use crate::merkle_worker::MerkleWorker;
    use crate::persistence_layer::PersistenceLayer;
    use crate::repo::DirectoryManager;

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
        Set { key: Key, value: Bytes },
        Delete { key: Key },
        Hash,
        Commit,
        Clone,
    }

    impl TestCommand {
        fn run(
            self,
            handle: &Handle,
            dir_manager: &DirectoryManager,
            worker: &mut MerkleWorker,
            layer: &mut MerkleLayer<LazyMavlNodeResolver>,
        ) {
            match self {
                Self::Set { key, value } => {
                    layer.set(&key, value.clone());
                    worker.set(key, value);
                }

                Self::Delete { key } => {
                    layer.delete(&key);
                    worker.delete(key);
                }

                Self::Hash => {
                    let hash1 = worker.hash();
                    let hash2 = layer.hash();
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
                    *layer = layer
                        .clone_with(persistence_layer)
                        .expect("Cloning a Merkle layer should succeed");

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
            let set = (key_strategy(), value_strategy())
                .prop_map(|(key, value)| TestCommand::Set { key, value });

            let delete = key_strategy().prop_map(|key| TestCommand::Delete { key });

            let hash = Just(TestCommand::Hash);

            let commit = Just(TestCommand::Commit);

            let clone = Just(TestCommand::Clone);

            // The frequencies are chosen to reflect a typical workload.
            proptest::prop_oneof![
                500 => set,
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
            let node_resolver = Arc::new(LazyMavlNodeResolver::new(persistence_layer.clone()));
            let mut merkle_layer = MerkleLayer::new(persistence_layer, node_resolver);

            let persistence_worker = PersistenceLayer::new(&dir_manager).expect("Creating a persistence layer should succeed");
            let persistence_worker = Arc::new(persistence_worker);
            let mut merkle_worker = MerkleWorker::new(handle, persistence_worker).expect("Creating a Merkle worker should succeed");

            for command in commands {
                command.run(handle, &dir_manager, &mut merkle_worker, &mut merkle_layer);
            }

            let layer_hash = merkle_layer.hash();
            let worker_hash = merkle_worker.hash();
            prop_assert_eq!(layer_hash, worker_hash);
        });
    }
}
