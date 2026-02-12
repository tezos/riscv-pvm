// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Error types for Durable Storage

use std::path::PathBuf;

/// Errors that are the result of an operational failure
///
/// These kinds of errors are fatal. When encountering such an error, there is no guarantee that the
/// used system is in a consistent state.
#[derive(Debug, thiserror::Error)]
pub enum OperationalError {
    #[error("Unable to locate commitment on disk")]
    CommitNotFound,

    #[error("Unable to create checkpoint: {error}")]
    CheckpointCreationFailed { error: rocksdb::Error },

    #[error("Could not create directory {path}: {error}")]
    DirCreationFailed {
        path: PathBuf,
        error: std::io::Error,
    },

    #[error("Failed to create directory or file in {path}: {error}")]
    TempCreationFailed {
        path: PathBuf,
        error: std::io::Error,
    },

    #[error("Unable to open RocksDB: {error}")]
    OpenRocksDbFailed { error: rocksdb::Error },

    #[error("Failed to create column family {name}: {error}")]
    ColumnFamilyCreationFailed { name: String, error: rocksdb::Error },

    #[error("RocksDB lookup failed {key:?} in {column}: {error}")]
    GetFailed {
        column: String,
        key: Vec<u8>,
        error: rocksdb::Error,
    },

    #[error("RocksDB update failed {key:?} in {column}: {error}")]
    PutFailed {
        column: String,
        key: Vec<u8>,
        error: rocksdb::Error,
    },

    #[error("RocksDB delete failed {key:?} in {column}: {error}")]
    DeleteFailed {
        column: String,
        key: Vec<u8>,
        error: rocksdb::Error,
    },

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

    #[error("Error while writing to file: {error}")]
    FileWriteFailed { error: std::io::Error },
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

    #[error("Value offset too large")]
    OffsetTooLarge,

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
