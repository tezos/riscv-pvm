// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of the RISC-V PVM durable storage's persistence layer.
//!
//! A persistence layer is tied to a repository on disk, identified by a directory.
//! Within that directory, the persistence layer needs to be able to perform basic KV operations:
//! - get
//! - set
//! - delete
//!
//! As well as repository-level operations:
//! - new
//! - clone
//! - commit (returning the commit hash)
//! - checkout a specific commit
//!
//! The folder structure of the [`DirectoryManager`] is:
//! ```
//! <repo_path>:
//!    temporary/
//!        db_<random>/checkpoint/
//!            <rocksdb internals>
//! ```

use std::mem::ManuallyDrop;

use rocksdb::MultiThreaded;
use tempfile::TempDir;

use crate::repo::DirectoryManager;
use crate::repo::DirectoryManagerError;

/// Type alias for a 32-byte hash used for identifying key-value blobs & commits.
type Hash = [u8; 32];

/// Errors encountered when interacting with the persistence layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("Directory manager error: {0}")]
    DirectoryManager(#[from] DirectoryManagerError),
}

/// Mode in which the [`PersistenceLayer`] was instantiated.
enum Mode {
    /// Either a new database, or a clone of an existing database.
    Temporary {
        /// The path to the temporary directory for the rocksdb checkpoint.
        tempdir: TempDir,
    },

    /// A database checked out from a specific commit.
    FromCommit,
}

/// These options are used for opening and closing a rocksdb instance.
///
/// Although different fields are used for opening vs. destroying a rocksdb instance, you need to
/// ensure that the options used for destroying are valid with respect to the options used when
/// opening the db. There is no concrete documentation for which options should be kept in sync for
/// open/close, may need to investigate rocksdb source code:
/// <https://github.com/facebook/rocksdb/blob/a1dad12c8c9a7a65fa19d3bc78a5f7687ce6c1bd/db/db_impl/db_impl.cc#L5185>
/// (look for the function destroying a rocksdb instance)
fn rocksdb_options() -> rocksdb::Options {
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    options.set_error_if_exists(true);
    options
}

/// Persistence layer for durable solution used by the RISC-V PVM.
///
/// Invariants:
/// - The path in `temp_initial_db_path` is unique for each instance of [`PersistenceLayer`] and is
///   assumed to not be modified / known outside of this instance.
pub struct PersistenceLayer {
    /// The underlying handle to the RocksDB instance.
    ///
    /// [`ManuallyDrop`] is used to ensure safety when dropping [`PersistenceLayer`]. Calling
    /// [`rocksdb::DB::destroy`] requires all connections to that path to be closed, which is called
    /// in [`rocksdb::DB`]'s drop method.
    db_instance: ManuallyDrop<rocksdb::DBWithThreadMode<MultiThreaded>>,

    /// What mode was the [`PersistenceLayer`] opened in.
    mode: Mode,
}

impl PersistenceLayer {
    /// Creates a new `PersistenceLayer` instance within the given `repo`.
    pub fn new(repo: &DirectoryManager) -> Result<Self, Error> {
        let tempdir = repo.new_temporary_dir()?;
        let new_db_path = tempdir.path().join("checkpoint");

        // To avoid accidentally overwriting an existing database, `error_if_exists` is set.
        let options = rocksdb_options();
        let db = rocksdb::DBWithThreadMode::open(&options, &new_db_path)?;

        Ok(Self {
            mode: Mode::Temporary { tempdir },
            db_instance: ManuallyDrop::new(db),
        })
    }
}

impl Drop for PersistenceLayer {
    /// Databases created from a new or clone operation will have `temp_initial_db_path` set. These
    /// databases do not have to be saved on disk as they have not been committed to storage.
    fn drop(&mut self) {
        let db_path = self.db_instance.path().to_path_buf();

        // Safety: This manual drop is called in this object's drop method.
        unsafe {
            ManuallyDrop::drop(&mut self.db_instance);
        }

        // SAFETY: Although marked as safe, destroy on a path requires all rocksdb connections to
        // this path to be closed. This is why we need to manual drop the db_instance first & the
        // invariants of `PersistenceLayer` to be upheld.
        if let Mode::Temporary { .. } = &self.mode {
            // Destroy the rocksdb at this path. The parent folder will be deleted by the drop
            // method of the tempdir in the mode field.

            let options = rocksdb_options();
            if let Err(e) = rocksdb::DB::destroy(&options, &db_path) {
                log::error!("Failed to destroy temporary rocksdb at {db_path:?}: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn checkpoint_db_path(db: &PersistenceLayer) -> PathBuf {
        db.db_instance.path().to_path_buf()
    }

    #[test]
    fn test_new_persistence_layer() {
        let repo = DirectoryManager::new(std::path::Path::new("/tmp/test_new_pl"))
            .expect("Failed to create directory manager");
        let db_a =
            PersistenceLayer::new(&repo).expect("Should be able to create new persistence layer");

        let db_b = PersistenceLayer::new(&repo)
            .expect("Should be able to create another persistence layer");

        let path_a = checkpoint_db_path(&db_a);
        let path_b = checkpoint_db_path(&db_b);

        // check that the directories are different
        assert!(path_a != path_b);

        drop(db_a);
        drop(db_b);

        // Check that after dropping the databases, the directories are removed - since they are not a committed database.
        assert!(!path_a.exists());
        assert!(
            !path_a
                .parent()
                .expect("Should have a db_<random> parent")
                .exists()
        );
        assert!(!path_b.exists());
        assert!(
            !path_b
                .parent()
                .expect("Should have a db_<random> parent")
                .exists()
        );
    }
}
