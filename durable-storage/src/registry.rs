// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Registry of databases for the RISC-V PVM durable storage.
//!
//! This module provides the Registry struct, which is responsible for managing multiple
//! databases within the durable storage system.

use tokio::runtime::Runtime;

use crate::database::Database;
use crate::database::DatabaseError;
use crate::repo::DirectoryManager;
use crate::repo::DirectoryManagerError;

#[derive(Debug, thiserror::Error)]
/// Errors returned by [`Registry`] operations.
pub enum RegistryError {
    /// Failed to create the Tokio runtime used by the registry.
    #[error("Failed to create Tokio runtime: {0}")]
    Runtime(std::io::Error),

    /// Error reported by the directory manager.
    #[error("Directory manager error: {0}")]
    DirectoryManager(#[from] DirectoryManagerError),

    /// Error reported by an underlying database.
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),
}

/// Registry that owns a set of databases backed by a directory manager.
pub struct Registry {
    repo: DirectoryManager,
    databases: Vec<Database>,
    runtime: Runtime,
}

impl Registry {
    /// Creates a new, empty Registry.
    ///
    /// The registry owns a Tokio [`Runtime`] and a [`DirectoryManager`] rooted at
    /// `base_dir`.
    pub fn new(repo: DirectoryManager) -> Result<Self, RegistryError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .map_err(RegistryError::Runtime)?;

        Ok(Registry {
            repo,
            databases: Vec::new(),
            runtime,
        })
    }

    /// Get the number of databases held in the registry.
    pub fn size(&self) -> usize {
        self.databases.len()
    }

    /// Resize the registry to the given `new_size`.
    ///
    /// Growing the registry creates new databases, while shrinking drops
    /// databases from the end.
    pub fn resize(&mut self, new_size: usize) -> Result<(), RegistryError> {
        while self.size() < new_size {
            let database = Database::try_new(self.runtime.handle(), &self.repo)?;
            self.databases.push(database);
        }

        while self.size() > new_size {
            self.databases.pop();
        }

        Ok(())
    }

    /// Get a reference to the database at the given `index`.
    pub fn get_database_ref(&self, index: usize) -> Option<&Database> {
        self.databases.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::Registry;
    use crate::persistence_layer::utils::TestableTmpdir;
    use crate::repo::DirectoryManager;

    fn setup_registry() -> (TestableTmpdir, Registry) {
        let tmpdir = TestableTmpdir::new();
        let base_dir = tmpdir.path().join("registry");
        let repo = DirectoryManager::new(&base_dir).expect("creating manager should succeed.");
        let registry = Registry::new(repo).expect("Registry should be created");

        (tmpdir, registry)
    }

    #[test]
    fn test_new() {
        let (_tmpdir, registry) = setup_registry();

        assert!(registry.size() == 0);
    }

    #[test]
    fn test_resize() {
        let (_tmpdir, mut registry) = setup_registry();

        registry
            .resize(4)
            .expect("Growing the registry should succeed.");

        assert_eq!(registry.size(), 4);

        registry
            .resize(1)
            .expect("Shrinking the registry should succeed.");

        assert_eq!(registry.size(), 1);
    }

    #[test]
    fn test_get_database() {
        let (_tmpdir, mut registry) = setup_registry();

        registry
            .resize(3)
            .expect("Growing the registry should succeed.");

        for i in 0..3 {
            registry
                .get_database_ref(i)
                .expect("Database should exist.");
        }
    }
}
