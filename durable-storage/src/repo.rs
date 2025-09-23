// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::io;
use std::path::Path;
use std::path::PathBuf;

use tempfile::TempDir;

#[derive(Debug, thiserror::Error)]
pub enum DirectoryManagerError {
    #[error("Failed instantiating a new directory manager: {0}")]
    New(io::Error),

    #[error("Failed creating a new temporary directory: {0}")]
    Temporary(io::Error),
}

/// The [`DirectoryManager`] represents the root directory where commitments & internal data should
/// be stored.
pub struct DirectoryManager {
    /// The path of for the `temporary` subdirectory managed by this [`DirectoryManager`].
    temp_dir: PathBuf,
}

impl DirectoryManager {
    /// Instantiate a new [`DirectoryManager`] at the given `path`.
    ///
    /// - If the folder does not exist, it will be created and the [`DirectoryManager`] will be
    ///   empty.
    /// - If the folder exists, it will assume a valid repository for the paths managed, i.e.
    ///   `<repo>/temporary/db_<id>/checkpoint` is a valid rocksdb checkpoint.
    pub fn new(path: &Path) -> Result<Self, DirectoryManagerError> {
        let temp_dir = path.join("temporary");

        if !temp_dir.exists() {
            std::fs::create_dir_all(&temp_dir).map_err(DirectoryManagerError::New)?;
        }

        Ok(Self { temp_dir })
    }

    /// Create the new random directory for `temporary/db_<random>` under the root of the
    /// [`DirectoryManager`]'s path.
    ///
    /// Note: The folder corresponding to the returned [`TempDir`] will be deleted once the
    /// [`TempDir`] is dropped.
    pub fn new_temporary_dir(&self) -> Result<TempDir, DirectoryManagerError> {
        // Use the tempfile crate to create a random directory name.
        let mut tempdir = tempfile::Builder::new();

        let tempdir = tempdir
            .prefix("db_")
            .tempdir_in(&self.temp_dir)
            .map_err(DirectoryManagerError::Temporary)?;

        Ok(tempdir)
    }
}
