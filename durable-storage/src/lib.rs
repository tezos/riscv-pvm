// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of the RISC-V PVM durable storage's persistence layer.
//!
//! A durable storage is responsible for providing a crash-resistant key-value store.
//!
//! The main components of the durable storage are:
//! - **Repository**: Corresponds to a directory on disk, and acts as the storage place for the
//!   commits.
//!
//! - **Database**: The interface for a KV database implemented by the Merkle layer and Persistence
//!   layer.
//!
//! - **Database registry**: Keeps track of individual & separate KV databases. The registry
//!   provides create, delete, copy & move operations between databases. The main motivation for
//!   this is to obtain atomicity.
//!
//! - **Cache layer**: Layer providing an in-memory cache for the KV operations. The concern is to
//!   optimise for performance.
//!
//! - **Avl**: An implementation of a Merklisable AVL tree.
//!
//! - **Merkle layer**: Responsible for arranging the data associated with each database in a way
//!   such that obtaining a root hash is possible & efficient.
//!
//! - **Persistence layer**: Responsible for actually persisting the basic get, set, delete
//!   operations on disk.

pub mod avl;
pub mod commit;
pub mod database;
pub mod errors;
// The space accounting harness scans committed column families directly, so it is only available
// when `rocksdb` is enabled.
#[cfg(rocksdb_test_utils)]
pub mod gc_space;
pub mod key;
// The long-running test exercises the persistence backend directly, so it is
// only available when `rocksdb` is enabled.
#[cfg(rocksdb_test_utils)]
pub mod long_test;
mod merkle_layer;
mod merkle_worker;
pub mod persistence_layer;
pub mod registry;
pub mod repo;
pub mod storage;
pub mod test_helpers;

// The Merkle representations a key-value store selects between through
// `ReadableKeyValueStore::Merkle`. The rest of `merkle_worker` - the worker thread, its commands
// and the store trait aliases - is an implementation detail of this crate.
pub use merkle_worker::CommittedRoot;
pub use merkle_worker::MerkleHandle;
pub use merkle_worker::MerkleWorker;
