// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The Merkle node store shared by a whole repository.
//!
//! Node bodies used to live in a `blob` column family inside each database's own instance, which
//! meant a node reached by several databases was stored once per database, and that every commit
//! directory hard-linked the files holding every node the database had ever written. Node data is
//! the large majority of what a retained commit pins, so that cost was paid again for each commit
//! kept.
//!
//! One store per repository replaces that. Nodes are content-addressed, so the same node written
//! from any database is the same key here and is stored once. The store sits outside the per-commit
//! checkpoints, so its compaction churn is paid once rather than once per retained commit, and it
//! gives one place from which a node can actually be deleted - which directory removal can never
//! do, and which is what collecting the Merkle side will need.
//!
//! # One instance per path
//!
//! RocksDB permits a single writer per directory, while a repository is reached through
//! [`DirectoryManager`], which is cloned freely and may be constructed more than once for the same
//! path. Instances are therefore shared per canonical path through [`open_shared`], so that every
//! handle onto a repository reaches the same store rather than the second one failing to open it.
//!
//! [`DirectoryManager`]: crate::repo::DirectoryManager

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::Weak;

use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::persistence_layer::rocksdb_node_store_options;
use crate::storage::StoreId;

/// The Merkle node bodies of one repository.
///
/// Cheap to clone through the [`Arc`] handed out by [`open_shared`]; the instance itself is opened
/// once per path.
#[derive(Debug)]
pub struct MerkleStore {
    /// The underlying handle to the RocksDB instance.
    ///
    /// [`ManuallyDrop`] because closing has to happen at a controlled moment: RocksDB holds a lock
    /// on the directory for as long as the handle lives, and the entry in [`open_stores`] stops
    /// being findable as soon as the last [`Arc`] is released, which is before this field would
    /// otherwise be dropped. Closing it inside [`Drop`], while the registry is locked, keeps
    /// reopening from racing the close. See that impl.
    db: ManuallyDrop<rocksdb::DB>,

    /// Where this store was opened, so it can remove its own registry entry when it closes.
    path: PathBuf,

    /// Whether this handle may write.
    ///
    /// A store opened for reading does not hold RocksDB's single-writer lock, so any number of them
    /// can read a repository that another process is writing to.
    read_only: bool,

    /// Identifies this store, so a node can record that it has already been written here.
    ///
    /// Every database of a repository shares it, which is what lets a node written through one be
    /// recognised as already stored by the others.
    store_id: StoreId,
}

impl Drop for MerkleStore {
    /// Close the instance, then stop advertising it.
    ///
    /// In that order, and without holding the registry across the close: [`open_shared`] releases
    /// the registry before opening RocksDB, so a close that took the registry first and held it
    /// would leave an opener waiting on a directory whose lock this handle still owns.
    ///
    /// A store closing at all is the uncommon case. [`DirectoryManager`] holds its repository's
    /// store for as long as the handle lives, so the databases of one repository share an instance
    /// that stays open rather than one that closes and reopens between them.
    ///
    /// [`DirectoryManager`]: crate::repo::DirectoryManager
    fn drop(&mut self) {
        // Safety: this is the only place the handle is dropped, and it runs once.
        unsafe {
            ManuallyDrop::drop(&mut self.db);
        }

        // A poisoned registry only means another thread panicked holding it. The entry is this
        // store's own and is now stale either way, so it is still the right thing to remove.
        let mut open = open_stores()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Only if it is still ours: a new store for the same path may already have replaced it,
        // and a read-only handle never had an entry to begin with.
        if !self.read_only
            && open
                .get(&self.path)
                .is_some_and(|entry| entry.strong_count() == 0)
        {
            open.remove(&self.path);
        }
    }
}

impl MerkleStore {
    /// Read the node body stored under `key`.
    pub fn get(&self, key: &[u8]) -> Result<rocksdb::DBPinnableSlice<'_>, Error> {
        let value = self
            .db
            .get_pinned(key)
            .map_err(|error| OperationalError::GetFailed {
                column: "merkle".to_owned(),
                key: key.to_owned(),
                error,
            })?;

        match value {
            Some(value) => Ok(value),
            None => Err(InvalidArgumentError::KeyNotFound)?,
        }
    }

    /// Store `data` as the node body under `key`.
    pub fn set(&self, key: &[u8], data: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        self.db
            .put(key, data)
            .map_err(|error| OperationalError::PutFailed {
                column: "merkle".to_owned(),
                key: key.to_owned(),
                error,
            })
    }

    /// Remove the node body stored under `key`.
    pub fn delete(&self, key: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        self.db
            .delete(key)
            .map_err(|error| OperationalError::DeleteFailed {
                column: "merkle".to_owned(),
                key: key.to_owned(),
                error,
            })
    }

    /// Whether this handle may write to the store.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Fail if this handle may not write.
    fn writeable(&self) -> Result<(), OperationalError> {
        if self.read_only {
            return Err(OperationalError::RepositoryIsReadOnly);
        }

        Ok(())
    }

    /// Identity of this store, for recording that a node has already been written to it.
    pub fn store_id(&self) -> StoreId {
        self.store_id
    }

    /// Count every node in the store, and the bytes they occupy.
    ///
    /// A full scan rather than a RocksDB estimate: the point of measuring is to know how much of
    /// the store is dead, and an estimate cannot be subtracted from an exact live figure.
    #[cfg(rocksdb_test_utils)]
    pub fn totals(
        &self,
    ) -> Result<crate::persistence_layer::measurement::CfTotals, OperationalError> {
        let mut totals = crate::persistence_layer::measurement::CfTotals::default();

        for entry in self.db.iterator(rocksdb::IteratorMode::Start) {
            let (key, value) = entry.map_err(|error| OperationalError::GetFailed {
                column: "merkle".to_owned(),
                key: Vec::new(),
                error,
            })?;

            totals.entries += 1;
            totals.key_bytes += key.len() as u64;
            totals.value_bytes += value.len() as u64;
        }

        Ok(totals)
    }

    /// Bytes the store's SST files occupy on disk.
    #[cfg(rocksdb_test_utils)]
    pub fn sst_bytes(&self) -> Result<u64, OperationalError> {
        Ok(self
            .db
            .property_int_value("rocksdb.total-sst-files-size")
            .map_err(|error| OperationalError::GetFailed {
                column: "merkle".to_owned(),
                key: Vec::new(),
                error,
            })?
            .unwrap_or(0))
    }

    /// Write a self-contained copy of the store to `path`.
    ///
    /// A checkpoint hard-links the files live when it is taken, so this costs almost nothing and
    /// shares its data with the store until compaction rewrites it. The copy is a complete store: it
    /// can be opened on its own, which is what lets the node bodies belonging to a commit be carried
    /// somewhere the repository is not.
    pub fn checkpoint(&self, path: &Path) -> Result<(), OperationalError> {
        // The checkpoint object is dropped before the path is used, as elsewhere in this crate.
        rocksdb::checkpoint::Checkpoint::new(&self.db)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })?
            .create_checkpoint(path)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })
    }

    /// Put everything written so far beyond reach of a crash.
    ///
    /// A database commit checkpoints only that database's values; the nodes those values are
    /// indexed by live here, and nothing about taking that checkpoint puts them on disk. Committing
    /// therefore syncs the write-ahead log first, so a commit never refers to a node that a crash
    /// could take with it.
    pub fn sync(&self) -> Result<(), OperationalError> {
        self.writeable()?;

        self.db
            .flush_wal(true)
            .map_err(|error| OperationalError::CheckpointCreationFailed { error })
    }
}

/// Instances already open, keyed by the path they were opened at.
///
/// Held weakly: an entry does not keep a store alive, it only lets a store that is still alive be
/// found again. Entries for dropped stores are cleared when the path is next opened.
type OpenStores = Mutex<HashMap<PathBuf, Weak<MerkleStore>>>;

fn open_stores() -> &'static OpenStores {
    static OPEN_STORES: OnceLock<OpenStores> = OnceLock::new();
    OPEN_STORES.get_or_init(Default::default)
}

/// The store at `path`, opening it if it is not already open.
///
/// Two callers naming the same path receive the same instance, which is what makes the store shared
/// by a repository rather than by a handle onto one. The path is created if it does not exist.
pub fn open_shared(path: &Path) -> Result<Arc<MerkleStore>, OperationalError> {
    // The directory has to exist before it can be canonicalised, and RocksDB would create it
    // anyway; doing it here keeps the key stable no matter which caller arrives first.
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|error| OperationalError::DirCreationFailed {
            path: path.to_path_buf(),
            error,
        })?;
    }

    // Two paths naming the same directory must find the same entry, which spelling alone does not
    // guarantee.
    let key = path
        .canonicalize()
        .map_err(|error| OperationalError::FileReadFailed { error })?;

    if let Some(store) = lookup(&key)? {
        return Ok(store);
    }

    // Opened without the registry held: RocksDB takes a lock on the directory, and a store closing
    // right now needs the registry to finish getting out of the way.
    let db = rocksdb::DB::open(&rocksdb_node_store_options(), &key)
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

    let store = Arc::new(MerkleStore {
        db: ManuallyDrop::new(db),
        path: key.clone(),
        read_only: false,
        store_id: StoreId::next(),
    });

    let mut open = open_stores()
        .lock()
        .map_err(|_| OperationalError::LockPoisoned)?;

    // Another caller may have opened one for this path while this one was opening. Either instance
    // is usable, but only one may be advertised, and it must be the one they both go on to share.
    if let Some(winner) = open.get(&key).and_then(Weak::upgrade) {
        return Ok(winner);
    }

    open.insert(key, Arc::downgrade(&store));

    Ok(store)
}

/// Open the store at `path` for reading, without taking RocksDB's writer lock.
///
/// Any number of these can read a repository that another process is writing to, which is how
/// external tooling reaches a running node's Merkle nodes: the live store admits one writer, so a
/// second process cannot open it the ordinary way at all.
///
/// Not shared through [`open_stores`], and deliberately: that registry exists so the databases of
/// one repository share the single writeable instance, and a read-only handle is neither that nor
/// interchangeable with it.
///
/// The view is of the store as it stood when this was opened; later writes are not seen.
pub fn open_read_only(path: &Path) -> Result<Arc<MerkleStore>, OperationalError> {
    if !path.exists() {
        return Err(OperationalError::CommitNotFound);
    }

    let db = rocksdb::DB::open_for_read_only(&rocksdb_node_store_options(), path, false)
        .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

    Ok(Arc::new(MerkleStore {
        db: ManuallyDrop::new(db),
        read_only: true,
        // Never registered, so nothing looks this up and Drop finds no entry of its own to remove.
        path: path.to_path_buf(),
        store_id: StoreId::next(),
    }))
}

/// The store already open at `key`, if there is one.
fn lookup(key: &Path) -> Result<Option<Arc<MerkleStore>>, OperationalError> {
    Ok(open_stores()
        .lock()
        .map_err(|_| OperationalError::LockPoisoned)?
        .get(key)
        .and_then(Weak::upgrade))
}

#[cfg(test)]
mod tests {
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;

    // A node written to the store reads back from it.
    #[test]
    fn stores_and_reads_a_node() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        store.set(b"key", b"body").expect("setting should succeed");

        assert_eq!(
            store
                .get(b"key")
                .expect("the node should be there")
                .as_ref(),
            b"body"
        );
    }

    // Reading a node that was never written is a missing key, not a failure of the store.
    #[test]
    fn reading_an_absent_node_reports_it_missing() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        assert!(matches!(
            store.get(b"absent"),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
    }

    // A repository being written to can be read at the same time. The live store admits one
    // writer, so without this a second process could not reach a running node's nodes at all.
    #[test]
    fn a_reader_opens_alongside_the_writer() {
        let tmp = TestableTmpdir::new();
        let path = tmp.path().join("merkle");

        let writer = open_shared(&path).expect("the writer should open");
        writer.set(b"key", b"body").expect("setting should succeed");
        writer.sync().expect("syncing should succeed");

        let reader = open_read_only(&path).expect("a reader should open alongside it");

        assert!(reader.is_read_only());
        assert_eq!(
            reader
                .get(b"key")
                .expect("the node should be there")
                .as_ref(),
            b"body"
        );

        // And it refuses to be written through, rather than failing somewhere further down.
        assert!(matches!(
            reader.set(b"other", b"body"),
            Err(OperationalError::RepositoryIsReadOnly)
        ));

        // The writer carries on.
        writer
            .set(b"later", b"body")
            .expect("the writer should still write");
    }

    // Opening the same path twice yields one instance, so a second handle onto a repository does
    // not fail against RocksDB's single-writer lock and sees what the first wrote.
    #[test]
    fn opening_the_same_path_twice_shares_one_instance() {
        let tmp = TestableTmpdir::new();
        let path = tmp.path().join("merkle");

        let first = open_shared(&path).expect("the first open should succeed");
        let second = open_shared(&path).expect("the second open should succeed");

        assert!(Arc::ptr_eq(&first, &second), "both should be one instance");
        assert_eq!(
            first.store_id(),
            second.store_id(),
            "one instance is one store identity"
        );

        first.set(b"key", b"body").expect("setting should succeed");
        assert_eq!(
            second
                .get(b"key")
                .expect("the second handle should see it")
                .as_ref(),
            b"body"
        );
    }

    // A spelling of the path that differs only by a redundant component still finds the open
    // instance, since RocksDB's lock is on the directory rather than on the string.
    #[test]
    fn an_equivalent_path_finds_the_same_instance() {
        let tmp = TestableTmpdir::new();
        let path = tmp.path().join("merkle");

        let first = open_shared(&path).expect("the first open should succeed");
        let second =
            open_shared(&tmp.path().join(".").join("merkle")).expect("the open should succeed");

        assert!(Arc::ptr_eq(&first, &second), "both should be one instance");
    }

    // Once every handle is dropped the instance closes, and opening again reads what it wrote
    // rather than finding a stale entry or a held lock.
    #[test]
    fn reopening_after_the_last_handle_is_dropped_succeeds() {
        let tmp = TestableTmpdir::new();
        let path = tmp.path().join("merkle");

        let store = open_shared(&path).expect("opening should succeed");
        store.set(b"key", b"body").expect("setting should succeed");
        store.sync().expect("syncing should succeed");
        drop(store);

        let reopened = open_shared(&path).expect("reopening should succeed");
        assert_eq!(
            reopened
                .get(b"key")
                .expect("the node should have survived")
                .as_ref(),
            b"body"
        );
    }
}
