// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Long-running property-based tests for durable storage.
//!
//! [`database`] exercises a single [`Database`]; [`registry`] exercises a
//! [`Registry`] of several databases. Both drivers share the building blocks
//! in [`harness`].
//!
//! [`Database`]: crate::database::Database
//! [`Registry`]: crate::registry::Registry

pub mod database;
mod harness;
pub mod registry;

use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
pub use harness::LongTestConfig;

use crate::repo::DirectoryManager;

/// Name of the Merkle node snapshot within a failure artifact.
const MERKLE_BASE: &str = "merkle-base";

/// Save the Merkle node bodies a failure artifact needs to be replayable elsewhere.
///
/// A database snapshot holds that database's values; the nodes they are indexed by live in the
/// repository's store, so without this an artifact could only be replayed against the repository
/// that produced it.
pub(crate) fn save_merkle_base(failure_dir: &Path, repo: &DirectoryManager) -> Result<()> {
    repo.merkle_store()
        .checkpoint(&failure_dir.join(MERKLE_BASE))
        .context("writing the Merkle base snapshot")
}

/// Put the node bodies saved by [`save_merkle_base`] into the repository rooted at `repo_dir`.
///
/// The snapshot is itself a complete store, so it becomes the fresh repository's simply by being
/// put where one is expected. Takes the path rather than a [`DirectoryManager`] because it has to
/// run before one is constructed: constructing a handle opens the store, and files appearing
/// underneath an open RocksDB instance are not its own.
pub(crate) fn restore_merkle_base(failure_dir: &Path, repo_dir: &Path) -> Result<()> {
    let saved = failure_dir.join(MERKLE_BASE);
    let target = DirectoryManager::merkle_dir_in(repo_dir);

    fs::create_dir_all(&target).with_context(|| format!("creating {}", target.display()))?;

    // A RocksDB directory has no subdirectories, so a flat copy is a faithful one.
    for entry in fs::read_dir(&saved).with_context(|| format!("reading {}", saved.display()))? {
        let entry = entry.with_context(|| format!("reading an entry of {}", saved.display()))?;

        if !entry.file_type().is_ok_and(|kind| kind.is_file()) {
            continue;
        }

        let to = target.join(entry.file_name());
        fs::copy(entry.path(), &to)
            .with_context(|| format!("copying {} to {}", entry.path().display(), to.display()))?;
    }

    Ok(())
}
