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

    /// Error indicating an invalid database index was provided.
    #[error("Invalid database index")]
    InvalidDatabaseIndex,
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
    pub fn get_database_ref(&self, index: usize) -> Result<&Database, RegistryError> {
        self.databases
            .get(index)
            .ok_or(RegistryError::InvalidDatabaseIndex)
    }

    /// Get a mutable reference to the database at the given `index`.
    pub fn get_database_mut(&mut self, index: usize) -> Result<&mut Database, RegistryError> {
        self.databases
            .get_mut(index)
            .ok_or(RegistryError::InvalidDatabaseIndex)
    }

    /// Check the given `index` is valid for a database in the registry.
    fn validate_index(&self, index: usize) -> Result<(), RegistryError> {
        if index >= self.databases.len() {
            Err(RegistryError::InvalidDatabaseIndex)
        } else {
            Ok(())
        }
    }

    /// Copy the contents of database at `src_index` to database at `dst_index`.
    pub fn copy_database(
        &mut self,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), RegistryError> {
        self.validate_index(src_index)?;
        self.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if copying to the same index.
            return Ok(());
        }

        let db_copy =
            self.databases[src_index].try_clone_with(self.runtime.handle(), &self.repo)?;
        self.databases[dst_index] = db_copy;

        Ok(())
    }

    /// Move the contents of database at `src_index` to database at `dst_index`. The source
    /// database is replaced with an empty database.
    pub fn move_database(
        &mut self,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), RegistryError> {
        self.validate_index(src_index)?;
        self.validate_index(dst_index)?;

        if src_index == dst_index {
            // No-op if copying to the same index.
            return Ok(());
        }

        let empty = Database::try_new(self.runtime.handle(), &self.repo)?;
        let db_to_move = std::mem::replace(&mut self.databases[src_index], empty);
        self.databases[dst_index] = db_to_move;

        Ok(())
    }

    /// Clear the database at the given `index`.
    pub fn clear_database(&mut self, index: usize) -> Result<(), RegistryError> {
        self.validate_index(index)?;
        self.databases[index] = Database::try_new(self.runtime.handle(), &self.repo)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::Registry;
    use crate::key::Key;
    use crate::persistence_layer::utils::TestableTmpdir;
    use crate::repo::DirectoryManager;

    fn setup_registry() -> (TestableTmpdir, Registry) {
        let tmpdir = TestableTmpdir::new();
        let base_dir = tmpdir.path().join("registry");
        let repo = DirectoryManager::new(&base_dir).expect("creating manager should succeed.");
        let registry = Registry::new(repo).expect("Registry should be created");

        (tmpdir, registry)
    }

    fn setup_size_2_registry() -> (TestableTmpdir, Registry) {
        let (tmpdir, mut registry) = setup_registry();
        registry
            .resize(2)
            .expect("Growing the registry should succeed.");
        (tmpdir, registry)
    }

    fn seed_copy_move(
        registry: &mut Registry,
        src_index: usize,
        dst_index: usize,
    ) -> ([(Key, &'static [u8]); 2], Key) {
        // Before the copy/move, populate the source with key A and B, and the dest with key A and C.
        let key_a = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let key_b = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        let key_c = Key::new(&[3]).expect("Size less than KEY_MAX_SIZE");

        let src_pairs = [
            (key_a.clone(), b"new_a".as_slice()),
            (key_b.clone(), b"new_b".as_slice()),
        ];

        for (key, value) in src_pairs.iter() {
            registry.databases[src_index]
                .write(key.clone(), 0, Bytes::copy_from_slice(value))
                .expect("Writing to source should succeed");
        }

        // Write values to destination that should be overwritten.
        registry.databases[dst_index]
            .write(key_a, 0, Bytes::copy_from_slice(b"old_a"))
            .expect("Writing to destination should succeed");
        registry.databases[dst_index]
            .write(key_c.clone(), 0, Bytes::copy_from_slice(b"old_c"))
            .expect("Writing to destination should succeed");

        (src_pairs, key_c)
    }

    fn assert_pairs_present(registry: &Registry, db_index: usize, pairs: &[(Key, &'static [u8])]) {
        for (key, value) in pairs.iter() {
            assert!(
                registry.databases[db_index]
                    .exists(key)
                    .expect("Checking destination should succeed")
            );
            let mut buf = vec![0u8; value.len()];
            registry.databases[db_index]
                .read(key, 0, &mut buf)
                .expect("Reading from destination should succeed");
            assert_eq!(&buf, value);
        }
    }

    fn assert_pairs_absent(registry: &Registry, db_index: usize, pairs: &[(Key, &'static [u8])]) {
        for (key, _value) in pairs.iter() {
            assert!(
                !registry.databases[db_index]
                    .exists(key)
                    .expect("Checking source should succeed"),
                "Key should not exist in source after move."
            );
        }
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

    #[test]
    fn test_copy_database() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 0;
        let dst_index = 1;

        let (src_pairs, key_c) = seed_copy_move(&mut registry, src_index, dst_index);

        registry
            .copy_database(src_index, dst_index)
            .expect("Copying should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);

        assert!(
            !registry.databases[dst_index]
                .exists(&key_c)
                .expect("Checking destination should succeed"),
            "Key C should not exist in destination after copy."
        );
    }

    #[test]
    fn test_copy_same_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 0;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, src_index, 1);

        registry
            .copy_database(src_index, dst_index)
            .expect("Copying should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);
    }

    #[test]
    fn test_copy_invalid_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let result = registry.copy_database(0, 2);
        assert!(
            matches!(result, Err(super::RegistryError::InvalidDatabaseIndex)),
            "Copying to invalid index should return InvalidDatabaseIndex error."
        );

        let result = registry.copy_database(2, 0);
        assert!(
            matches!(result, Err(super::RegistryError::InvalidDatabaseIndex)),
            "Copying from invalid index should return InvalidDatabaseIndex error."
        );
    }

    #[test]
    fn test_move_database() {
        // Test that the source database is emptied and the destination database
        // has all the data, and any data previously in the destination is lost.

        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 1;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, src_index, dst_index);

        registry
            .move_database(src_index, dst_index)
            .expect("Moving should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);
        assert_pairs_absent(&registry, src_index, &src_pairs);
    }

    #[test]
    fn test_move_invalid_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let result = registry.move_database(0, 2);
        assert!(
            matches!(result, Err(super::RegistryError::InvalidDatabaseIndex)),
            "Moving to invalid index should return InvalidDatabaseIndex error."
        );

        let result = registry.move_database(2, 0);
        assert!(
            matches!(result, Err(super::RegistryError::InvalidDatabaseIndex)),
            "Moving from invalid index should return InvalidDatabaseIndex error."
        );
    }

    #[test]
    fn test_move_same_index() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let src_index = 0;
        let dst_index = 0;

        let (src_pairs, _key_c) = seed_copy_move(&mut registry, src_index, 1);

        registry
            .move_database(src_index, dst_index)
            .expect("Moving should succeed");

        assert_pairs_present(&registry, dst_index, &src_pairs);
    }

    #[test]
    fn test_clear_database() {
        let (_tmpdir, mut registry) = setup_size_2_registry();

        let db_index = 0;
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        registry.databases[db_index]
            .write(key.clone(), 0, Bytes::copy_from_slice(b"some_value"))
            .expect("Writing to database should succeed");

        registry
            .clear_database(db_index)
            .expect("Clearing the database should succeed");

        assert!(
            !registry.databases[db_index]
                .exists(&key)
                .expect("Checking database should succeed"),
            "Key should not exist after clearing the database."
        );
    }
}
