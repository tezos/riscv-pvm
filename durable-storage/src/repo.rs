// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Repository management for the Durable Storage

use std::path::Path;
use std::path::PathBuf;

use tempfile::TempDir;

use crate::commit::CommitId;
use crate::errors::OperationalError;

/// The [`DirectoryManager`] represents the root directory where commitments & internal data should
/// be stored.
pub struct DirectoryManager {
    /// Directory prefix for [`crate::database::Database`] instances (temporary)
    temp_databases_dir: PathBuf,

    /// Directory prefix for [`crate::database::Database`] commitments
    database_commits_dir: PathBuf,

    /// Directory prefix for [`crate::registry::Registry`] commitments
    registry_commits_dir: PathBuf,
}

impl DirectoryManager {
    /// Instantiate a new [`DirectoryManager`] at the given `path`.
    ///
    /// - If the folder does not exist, it will be created and the [`DirectoryManager`] will be
    ///   empty.
    /// - If the folder exists, it will assume a valid repository for the paths managed.
    pub fn new(path: &Path) -> Result<Self, OperationalError> {
        let ensure_dir_exists = |dir: &PathBuf| -> Result<(), OperationalError> {
            if !dir.exists() {
                std::fs::create_dir_all(dir).map_err(|error| {
                    OperationalError::DirCreationFailed {
                        path: dir.clone(),
                        error,
                    }
                })?;
            }

            Ok(())
        };

        let databases_dir = path.join("databases");

        // Holds temporary databases. Each database is stored as a separate directory.
        let temp_databases_dir = databases_dir.join("temporary");
        ensure_dir_exists(&temp_databases_dir)?;

        // Has the same layout as temporary databases, but for permanent databases (i.e. commits).
        let database_commits_dir = databases_dir.join("commits");
        ensure_dir_exists(&database_commits_dir)?;

        // The directory for registry commits holds files. Each file is named after the commit ID.
        let registry_commits_dir = path.join("registries").join("commits");
        ensure_dir_exists(&registry_commits_dir)?;

        Ok(Self {
            temp_databases_dir,
            database_commits_dir,
            registry_commits_dir,
        })
    }

    /// Create a temporary directory suitable for a [`crate::database::Database`].
    ///
    /// Note: The folder corresponding to the returned [`TempDir`] will be deleted once the
    /// [`TempDir`] is dropped.
    pub fn temp_database_dir(&self) -> Result<TempDir, OperationalError> {
        // Use the tempfile crate to create a random directory name.
        let mut tempdir = tempfile::Builder::new();

        let tempdir = tempdir
            .prefix("db_")
            .tempdir_in(&self.temp_databases_dir)
            .map_err(|error| OperationalError::TempCreationFailed {
                path: self.temp_databases_dir.clone(),
                error,
            })?;

        Ok(tempdir)
    }

    /// Obtain the path to the commit directory for the given commit ID.
    pub fn database_commit_dir(&self, id: &CommitId) -> PathBuf {
        self.database_commits_dir.join(id.hex_encode())
    }

    /// Generate the path for a registry commit file provided its commit ID.
    pub fn registry_commit_file(&self, id: &CommitId) -> PathBuf {
        self.registry_commits_dir.join(id.hex_encode())
    }
}
