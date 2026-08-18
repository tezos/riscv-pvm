// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Error types for Durable Storage

use std::path::PathBuf;

use octez_riscv_data::hash::Hash;

use crate::key::Key;

/// Errors that are the result of an operational failure
///
/// These kinds of errors are fatal. When encountering such an error, there is no guarantee that the
/// used system is in a consistent state.
#[derive(Debug, thiserror::Error)]
pub enum OperationalError {
    #[error("Unable to locate commitment on disk")]
    CommitNotFound,

    #[error("Commit data is missing for a node or tree with hash {root:?}")]
    CommitDataMissing { root: Hash, source: Box<Error> },

    #[error("Commit is missing data for the value of key {key:?}")]
    CommitValueMissing { key: Key, source: Box<Error> },

    #[cfg(rocksdb)]
    #[error("Unable to create checkpoint: {error}")]
    CheckpointCreationFailed { error: rocksdb::Error },

    #[error("Could not create directory {path}: {error}")]
    DirCreationFailed {
        path: PathBuf,
        error: std::io::Error,
    },

    #[error("Failed to publish the commit staged at {staged} as {commit}: {error}")]
    CommitPublishFailed {
        staged: PathBuf,
        commit: PathBuf,
        error: std::io::Error,
    },

    #[error("Failed to move the incomplete commit at {commit} aside to {displaced}: {error}")]
    IncompleteCommitDisplacementFailed {
        commit: PathBuf,
        displaced: PathBuf,
        error: std::io::Error,
    },

    #[error("Failed to create directory or file in {path}: {error}")]
    TempCreationFailed {
        path: PathBuf,
        error: std::io::Error,
    },

    #[cfg(rocksdb)]
    #[error("Unable to open RocksDB: {error}")]
    OpenRocksDbFailed { error: rocksdb::Error },

    #[cfg(rocksdb)]
    #[error("Failed to create column family {name}: {error}")]
    ColumnFamilyCreationFailed { name: String, error: rocksdb::Error },

    #[cfg(rocksdb)]
    #[error("RocksDB lookup failed {key:?} in {column}: {error}")]
    GetFailed {
        column: String,
        key: Vec<u8>,
        error: rocksdb::Error,
    },

    #[cfg(rocksdb)]
    #[error("RocksDB update failed {key:?} in {column}: {error}")]
    PutFailed {
        column: String,
        key: Vec<u8>,
        error: rocksdb::Error,
    },

    #[cfg(rocksdb)]
    #[error("RocksDB delete failed {key:?} in {column}: {error}")]
    DeleteFailed {
        column: String,
        key: Vec<u8>,
        error: rocksdb::Error,
    },

    #[cfg(rocksdb)]
    #[error("RocksDB merge failed {key:?} at {offset}: {error}")]
    MergeFailed {
        key: Vec<u8>,
        offset: usize,
        error: rocksdb::Error,
    },

    #[error("Unable to create worker runtime: {error}")]
    WorkerRuntimeCreationFailed { error: std::io::Error },

    #[error("Background worker thread died")]
    WorkerThreadDied,

    #[error("Error while reading from file: {error}")]
    FileReadFailed { error: std::io::Error },

    #[error("Error while writing to file: {error}")]
    FileWriteFailed { error: std::io::Error },

    #[error("The root hash of restored registry does not match the commit id.")]
    RegistryCommitMismatch,

    /// The lazy resolver encountered an internally inconsistent identifier state.
    ///
    /// `LazyId` values must always hold either (though can hold both):
    /// - a hash (`id: Some(hash)`) when the value is not loaded yet, or
    /// - an in-memory value (`inner` set) when the hash has been consumed.
    ///
    /// This error indicates that the state does not satisfy that invariant and is treated as
    /// fatal because resolver logic can no longer proceed safely.
    #[error("Resolver invariant violated. Either the hash or the value of an ID must exist.")]
    ResolverInvariantViolated,

    #[error("Encountered a poisoned lock")]
    LockPoisoned,

    #[error("Error during decoding.")]
    Decoding(#[from] bincode::error::DecodeError),

    #[error("Error during encoding.")]
    Encoding(#[from] bincode::error::EncodeError),
}

/// Errors that occur because of incorrect usage
///
/// When a user provides arguments which are invalid with respect to the state that is being
/// operated on. These errors are not fatal. The state must be left in a coherent state.
#[derive(Debug, thiserror::Error)]
pub enum InvalidArgumentError {
    #[error("Key does not exist")]
    KeyNotFound,

    #[error("Key is too long")]
    KeyTooLong,

    #[error("IO requests cannot be larger than MAX_FILE_CHUNK_SIZE")]
    IoRequestTooLarge,

    #[error("Value offset too large")]
    OffsetTooLarge,

    #[error("Maximum value size exceeded")]
    ValueSizeTooLarge,

    #[error("Registry resize can only change size by one")]
    RegistryResizeTooLarge,

    #[error("Database index out of bounds")]
    DatabaseIndexOutOfBounds,
}

/// Errors that occur during Durable Storage operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Operational error: {0}")]
    Operational(#[from] OperationalError),

    #[error("Invalid argument error: {0}")]
    InvalidArgument(#[from] InvalidArgumentError),
}
