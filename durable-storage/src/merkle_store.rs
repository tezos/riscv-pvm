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

pub mod slots;

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::Weak;

use octez_riscv_data::hash::Hash;

use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::journal::Seq;
use crate::persistence_layer::rocksdb_node_store_options;
use crate::storage::StoreId;

/// The column family holding reverse edges, keyed by `child || parent`.
const REFS_CF: &str = "refs";

/// The column family listing nodes by the commit they were last known to be held by.
///
/// Keyed `seq || hash`, with the sequence number big-endian so that ordering by key is ordering by
/// commit. This is what makes collection proportional to the garbage rather than to the store: a
/// round at a floor scans only the entries below it, and a node known to be held by a recent commit
/// is never looked at.
const WRITTEN_CF: &str = "written";

/// Bytes of the sequence number in a [`WRITTEN_CF`] key.
const SEQ_BYTES: usize = size_of::<u64>();

/// How recently a reverse edge was known to hold its child alive.
///
/// The sequence number of the most recent root the child was proven reachable from. Collection
/// reads it before doing anything else: at or above the floor means a retained root still holds the
/// child, so the walk stops there rather than climbing to a root.
///
/// [`Stamp::UNKNOWN`] means nothing has been proven yet, so it is below every floor and always
/// forces the walk. Being wrong in that direction only costs work, which is why edges may be
/// written that way and filled in later.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Stamp(u64);

impl Stamp {
    /// No root has yet been shown to hold the child.
    pub const UNKNOWN: Self = Self(0);

    /// The stamp for a child held by the root recorded at `seq`.
    ///
    /// Offset by one so that [`Stamp::UNKNOWN`] is distinct from, and below, the first recorded
    /// commit.
    pub fn at(seq: Seq) -> Self {
        Self(seq.raw() + 1)
    }

    /// Whether this stamp shows the child is held by a root at or after `floor`.
    pub fn holds_at(self, floor: Seq) -> bool {
        self >= Self::at(floor)
    }

    fn encode(self) -> [u8; size_of::<u64>()] {
        self.0.to_le_bytes()
    }

    fn decode(bytes: &[u8]) -> Self {
        <[u8; size_of::<u64>()]>::try_from(bytes)
            .map_or(Self::UNKNOWN, |bytes| Self(u64::from_le_bytes(bytes)))
    }
}

/// The file recording that a store has collected at least once.
///
/// Beside the store rather than a key inside it, which a sweep would scan and take for a node.
fn collected_marker(store_dir: &Path) -> PathBuf {
    let mut path = store_dir.as_os_str().to_owned();
    path.push("-collected");
    PathBuf::from(path)
}

/// The key listing `hash` as held by the commit at `seq`.
///
/// The sequence number leads and is big-endian, so ordering by key is ordering by commit and a
/// range scan can stop at a floor.
fn written_key(seq: Seq, hash: &[u8]) -> [u8; SEQ_BYTES + Hash::DIGEST_SIZE] {
    let mut key = [0u8; SEQ_BYTES + Hash::DIGEST_SIZE];
    key[..SEQ_BYTES].copy_from_slice(&seq.raw().to_be_bytes());
    key[SEQ_BYTES..].copy_from_slice(hash);
    key
}

/// Read back a key written by [`written_key`].
fn split_written_key(key: &[u8]) -> Option<(Seq, Vec<u8>)> {
    if key.len() != SEQ_BYTES + Hash::DIGEST_SIZE {
        return None;
    }

    let seq = u64::from_be_bytes(key[..SEQ_BYTES].try_into().ok()?);

    Some((Seq::from_raw(seq), key[SEQ_BYTES..].to_vec()))
}

/// The key an edge from `parent` to `child` is stored under.
///
/// Child first, so that every edge into one child is contiguous and its parents can be found by
/// walking forward from the child's hash.
fn edge_key(child: &[u8], parent: &[u8]) -> [u8; 2 * Hash::DIGEST_SIZE] {
    let mut key = [0u8; 2 * Hash::DIGEST_SIZE];
    key[..Hash::DIGEST_SIZE].copy_from_slice(child);
    key[Hash::DIGEST_SIZE..].copy_from_slice(parent);
    key
}

/// The Merkle node bodies of one repository, and the edges between them.
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
    ///
    /// Replaced whenever a collection deletes anything. A node skips being written when it records
    /// this identity, and after a collection that record is no longer evidence that the body is
    /// still here - the collection may have taken it. Changing the identity makes every such record
    /// stop matching, so the next commit writes what it touches rather than skipping it. The cost is
    /// one commit's worth of rewriting after a round; the alternative is a commit silently referring
    /// to a node that was collected out from under it.
    store_id: std::sync::atomic::AtomicU64,

    /// Whether a background compaction is running.
    ///
    /// Keeps a second request from starting another rather than queueing a rewrite behind the one
    /// already under way.
    compacting: std::sync::atomic::AtomicBool,

    /// Whether anything has ever been collected from this store.
    ///
    /// Changes what an absent node means. Before a collection it can only be a bug; after one it is
    /// most likely a state being read while its nodes were reclaimed, which is worth telling apart.
    /// Persisted, since the distinction outlives the process that collected.
    collected: std::sync::atomic::AtomicBool,
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
            // Once anything has been collected here, an absent node is most likely one that was
            // reclaimed while something was still reading it, which the caller can recover from by
            // checking out again. Said plainly rather than left to look like a corrupt store.
            None if self.has_collected() => Err(OperationalError::NodeCollected {
                root: <[u8; Hash::DIGEST_SIZE]>::try_from(key)
                    .map(Hash::from)
                    .unwrap_or_else(|_| Hash::from([0u8; Hash::DIGEST_SIZE])),
            })?,
            None => Err(InvalidArgumentError::KeyNotFound)?,
        }
    }

    /// Whether anything has ever been collected from this store.
    pub fn has_collected(&self) -> bool {
        self.collected.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record that a collection has removed something from this store.
    ///
    /// Written through to disk, so that a later process reading a node the collection took still
    /// learns why it is missing.
    pub fn note_collected(&self) -> Result<(), OperationalError> {
        self.writeable()?;

        // A new identity, so that no node still in memory can claim to be stored here on the
        // strength of a write that happened before this round. See the field's documentation.
        self.store_id
            .store(StoreId::next().raw(), std::sync::atomic::Ordering::Relaxed);

        if self
            .collected
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(());
        }

        std::fs::write(collected_marker(&self.path), [])
            .map_err(|error| OperationalError::FileWriteFailed { error })
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
        StoreId::from(self.store_id.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Record that the node stored under `parent` refers to the one stored under `child`.
    ///
    /// Reverse edges rather than reference counts, because they are idempotent: writing an edge
    /// twice, or removing it twice, leaves the same state, so a collection interrupted part-way can
    /// simply be run again. A count incremented twice after a crash is silently wrong for good.
    ///
    /// The edge carries a [`Stamp`]. Written unstamped here and filled in by whatever later proves
    /// the child reachable from a root, since the sequence number of the commit being made is not
    /// settled while its nodes are being written.
    pub fn set_edge(&self, child: &[u8], parent: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        let key = edge_key(child, parent);

        self.db
            .put_cf(self.refs_cf(), key, Stamp::UNKNOWN.encode())
            .map_err(|error| OperationalError::PutFailed {
                column: REFS_CF.to_owned(),
                key: key.to_vec(),
                error,
            })
    }

    /// Every node that refers to the one stored under `child`, with the stamp of each edge.
    ///
    /// Edge keys begin with the child, so the parents of one child are contiguous and are found by
    /// walking forward from that prefix.
    pub fn parents_of(&self, child: &[u8]) -> Result<Vec<(Vec<u8>, Stamp)>, OperationalError> {
        let mut parents = Vec::new();

        let from = rocksdb::IteratorMode::From(child, rocksdb::Direction::Forward);
        for entry in self.db.iterator_cf(self.refs_cf(), from) {
            let (key, value) = entry.map_err(|error| OperationalError::GetFailed {
                column: REFS_CF.to_owned(),
                key: child.to_vec(),
                error,
            })?;

            // Ordered by key, so the first entry that is not this child's ends the run.
            if !key.starts_with(child) {
                break;
            }

            parents.push((key[child.len()..].to_vec(), Stamp::decode(&value)));
        }

        Ok(parents)
    }

    /// Note that the node under `child` is held by a root at `stamp`, through the edge to `parent`.
    ///
    /// Only ever called where the child is provably reachable from the stamped root, which is what
    /// makes the stamp safe to trust later. A stale stamp costs a walk and an over-generous one
    /// retains garbage; neither can drop something still live.
    pub fn stamp_edge(
        &self,
        child: &[u8],
        parent: &[u8],
        stamp: Stamp,
    ) -> Result<(), OperationalError> {
        self.writeable()?;

        let key = edge_key(child, parent);

        self.db
            .put_cf(self.refs_cf(), key, stamp.encode())
            .map_err(|error| OperationalError::PutFailed {
                column: REFS_CF.to_owned(),
                key: key.to_vec(),
                error,
            })
    }

    /// Remove the edge recorded from `parent` to `child`.
    pub fn delete_edge(&self, child: &[u8], parent: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        let key = edge_key(child, parent);

        self.db
            .delete_cf(self.refs_cf(), key)
            .map_err(|error| OperationalError::DeleteFailed {
                column: REFS_CF.to_owned(),
                key: key.to_vec(),
                error,
            })
    }

    /// Note that the node under `key` is held by the commit at `seq`.
    ///
    /// Written as the node is stored, so a node created by a commit that is still retained is never
    /// a candidate for collection. Re-written by a sweep that proves an older node still held, which
    /// is what moves it out of the range future rounds scan.
    pub fn log_written(&self, seq: Seq, key: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        let entry = written_key(seq, key);

        self.db
            .put_cf(self.written_cf(), entry, [])
            .map_err(|error| OperationalError::PutFailed {
                column: WRITTEN_CF.to_owned(),
                key: entry.to_vec(),
                error,
            })
    }

    /// Forget that the node under `key` was held by the commit at `seq`.
    pub fn unlog_written(&self, seq: Seq, key: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        let entry = written_key(seq, key);

        self.db
            .delete_cf(self.written_cf(), entry)
            .map_err(|error| OperationalError::DeleteFailed {
                column: WRITTEN_CF.to_owned(),
                key: entry.to_vec(),
                error,
            })
    }

    /// Up to `limit` nodes last known to be held by a commit before `floor`, starting after `after`.
    ///
    /// These are exactly the nodes whose liveness is in question at that floor: anything held by a
    /// retained commit sorts at or above it and is not returned. Batched rather than collected
    /// whole, so a sweep of a large store does not have to hold the store's worth of keys in memory.
    ///
    /// `after` is the last entry of the previous batch, or `None` to start from the oldest.
    pub fn candidates_below(
        &self,
        floor: Seq,
        limit: usize,
        after: Option<(Seq, Vec<u8>)>,
    ) -> Result<Vec<(Seq, Vec<u8>)>, OperationalError> {
        let start = match &after {
            // One past the previous batch's last entry, which sorting by key makes a suffix bump.
            Some((seq, key)) => {
                let mut start = written_key(*seq, key).to_vec();
                start.push(0);
                start
            }
            None => Vec::new(),
        };

        let limit_key = written_key(floor, &[0u8; Hash::DIGEST_SIZE]);

        let from = rocksdb::IteratorMode::From(&start, rocksdb::Direction::Forward);
        let mut candidates = Vec::new();

        for entry in self.db.iterator_cf(self.written_cf(), from) {
            let (key, _) = entry.map_err(|error| OperationalError::GetFailed {
                column: WRITTEN_CF.to_owned(),
                key: Vec::new(),
                error,
            })?;

            // Ordered by sequence number, so the first entry at or above the floor ends the range.
            if key.as_ref() >= limit_key.as_slice() {
                break;
            }

            let Some((seq, hash)) = split_written_key(&key) else {
                continue;
            };

            candidates.push((seq, hash));

            if candidates.len() >= limit {
                break;
            }
        }

        Ok(candidates)
    }

    /// How many nodes are listed in the write log.
    #[cfg(rocksdb_test_utils)]
    pub fn written_entries(&self) -> Result<u64, OperationalError> {
        let mut entries = 0;

        for entry in self
            .db
            .iterator_cf(self.written_cf(), rocksdb::IteratorMode::Start)
        {
            entry.map_err(|error| OperationalError::GetFailed {
                column: WRITTEN_CF.to_owned(),
                key: Vec::new(),
                error,
            })?;
            entries += 1;
        }

        Ok(entries)
    }

    /// The column family listing nodes by the commit holding them.
    ///
    /// Panics if it is absent, which would mean the store was not opened by this module.
    fn written_cf(&self) -> &rocksdb::ColumnFamily {
        self.db
            .cf_handle(WRITTEN_CF)
            .expect("the Merkle store always has its written column family")
    }

    /// The column family holding reverse edges.
    ///
    /// Panics if it is absent, which would mean the store was not opened by this module.
    fn refs_cf(&self) -> &rocksdb::ColumnFamily {
        self.db
            .cf_handle(REFS_CF)
            .expect("the Merkle store always has its refs column family")
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

    /// Visit the key and stored size of every node in the store.
    ///
    /// A collection has to consider every node, including ones no retained root reaches, so there
    /// is nothing to descend from - the store itself is the list.
    pub fn for_each_node(
        &self,
        mut visit: impl FnMut(&[u8], usize),
    ) -> Result<(), OperationalError> {
        for entry in self.db.iterator(rocksdb::IteratorMode::Start) {
            let (key, value) = entry.map_err(|error| OperationalError::GetFailed {
                column: "merkle".to_owned(),
                key: Vec::new(),
                error,
            })?;

            visit(&key, value.len());
        }

        Ok(())
    }

    /// Remove every edge whose child is `child`.
    pub fn delete_edges_from(&self, child: &[u8]) -> Result<(), OperationalError> {
        self.writeable()?;

        for (parent, _) in self.parents_of(child)? {
            self.delete_edge(child, &parent)?;
        }

        Ok(())
    }

    /// Count every reverse edge, and the bytes they occupy.
    #[cfg(rocksdb_test_utils)]
    pub fn edge_totals(
        &self,
    ) -> Result<crate::persistence_layer::measurement::CfTotals, OperationalError> {
        let mut totals = crate::persistence_layer::measurement::CfTotals::default();

        for entry in self
            .db
            .iterator_cf(self.refs_cf(), rocksdb::IteratorMode::Start)
        {
            let (key, value) = entry.map_err(|error| OperationalError::GetFailed {
                column: REFS_CF.to_owned(),
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
        let mut total = 0;

        for cf in [
            self.db
                .cf_handle("default")
                .expect("the Merkle store always has its default column family"),
            self.refs_cf(),
            self.written_cf(),
        ] {
            total += self
                .db
                .property_int_value_cf(cf, "rocksdb.total-sst-files-size")
                .map_err(|error| OperationalError::GetFailed {
                    column: "merkle".to_owned(),
                    key: Vec::new(),
                    error,
                })?
                .unwrap_or(0);
        }

        Ok(total)
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

    /// Rewrite the store's files without the keys that have been deleted from it.
    ///
    /// Deleting a node only writes a tombstone; the space comes back when compaction rewrites the
    /// files without it. Collection therefore has to ask for that explicitly, or what it reclaimed
    /// stays on disk until RocksDB happens to compact for its own reasons.
    pub fn compact(&self) {
        if self.read_only {
            return;
        }

        let unbounded: Option<&[u8]> = None;

        let mut options = rocksdb::CompactOptions::default();
        // RocksDB otherwise disables its own compaction for the duration and waits for whatever is
        // already running to finish. This rewrites the whole store, so excluding everything else for
        // that long would stall writes behind it for no reason: the point is to reclaim disk, not to
        // be the only thing doing so.
        options.set_exclusive_manual_compaction(false);

        self.db.compact_range_opt(unbounded, unbounded, &options);
        self.db
            .compact_range_cf_opt(self.refs_cf(), unbounded, unbounded, &options);
        // Relisting a live node writes a new entry and a tombstone for the old one, so a round
        // leaves as much behind here as it does among the nodes themselves.
        self.db
            .compact_range_cf_opt(self.written_cf(), unbounded, unbounded, &options);
    }

    /// Start reclaiming in the background, and return without waiting for it.
    ///
    /// Reclaiming rewrites the store, so it costs the store rather than the garbage and takes far
    /// longer than anything that would want to trigger it. A full commit in particular must not wait
    /// for one: taking a slot is a flush and a rename, and pairing it with a rewrite of the whole
    /// store would make the cheap operation as slow as the expensive one.
    ///
    /// Returns whether this call started one. A second call while one is running is a no-op rather
    /// than a queued rewrite, since two of these achieve nothing one does not.
    ///
    /// Reads and writes continue throughout. The store stays open until the work finishes, because
    /// the thread holds a reference to it.
    pub fn start_compaction(self: &Arc<Self>) -> bool {
        if self.read_only
            || self
                .compacting
                .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return false;
        }

        let store = Arc::clone(self);

        match std::thread::Builder::new()
            .name("merkle-compaction".to_owned())
            .spawn(move || {
                store.compact();
                store
                    .compacting
                    .store(false, std::sync::atomic::Ordering::SeqCst);
            }) {
            Ok(_) => true,
            Err(error) => {
                // Nothing is running, so the flag has to come back down or nothing ever reclaims.
                self.compacting
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                log::warn!("could not start Merkle store compaction: {error}");
                false
            }
        }
    }

    /// Whether a background compaction started by [`MerkleStore::start_compaction`] is running.
    pub fn is_compacting(&self) -> bool {
        self.compacting.load(std::sync::atomic::Ordering::SeqCst)
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
    let mut options = rocksdb_node_store_options();
    // The refs family is created with the store rather than after it, and a store opened before
    // edges existed has to gain one.
    options.create_missing_column_families(true);

    let db = rocksdb::DB::open_cf_descriptors(
        &options,
        &key,
        [
            // Node bodies. Named rather than implied, because listing any family means listing all
            // of them.
            rocksdb::ColumnFamilyDescriptor::new("default", rocksdb_node_store_options()),
            rocksdb::ColumnFamilyDescriptor::new(REFS_CF, rocksdb_node_store_options()),
            rocksdb::ColumnFamilyDescriptor::new(WRITTEN_CF, rocksdb_node_store_options()),
        ],
    )
    .map_err(|error| OperationalError::OpenRocksDbFailed { error })?;

    let store = Arc::new(MerkleStore {
        db: ManuallyDrop::new(db),
        collected: std::sync::atomic::AtomicBool::new(collected_marker(&key).exists()),
        compacting: std::sync::atomic::AtomicBool::new(false),
        path: key.clone(),
        read_only: false,
        store_id: std::sync::atomic::AtomicU64::new(StoreId::next().raw()),
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
        collected: std::sync::atomic::AtomicBool::new(collected_marker(path).exists()),
        compacting: std::sync::atomic::AtomicBool::new(false),
        // Never registered, so nothing looks this up and Drop finds no entry of its own to remove.
        path: path.to_path_buf(),
        store_id: std::sync::atomic::AtomicU64::new(StoreId::next().raw()),
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

    fn digest(byte: u8) -> [u8; Hash::DIGEST_SIZE] {
        [byte; Hash::DIGEST_SIZE]
    }

    // Every parent recorded against a child is found, and only that child's.
    #[test]
    fn finds_the_parents_of_a_child() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        store
            .set_edge(&digest(1), &digest(10))
            .expect("setting should succeed");
        store
            .set_edge(&digest(1), &digest(11))
            .expect("setting should succeed");
        store
            .set_edge(&digest(2), &digest(12))
            .expect("setting should succeed");

        let mut parents: Vec<Vec<u8>> = store
            .parents_of(&digest(1))
            .expect("reading should succeed")
            .into_iter()
            .map(|(parent, _)| parent)
            .collect();
        parents.sort();

        assert_eq!(parents, vec![digest(10).to_vec(), digest(11).to_vec()]);
    }

    // A child nothing refers to has no parents, rather than picking up its neighbours'.
    #[test]
    fn an_unreferenced_child_has_no_parents() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        store
            .set_edge(&digest(1), &digest(10))
            .expect("setting should succeed");

        assert!(
            store
                .parents_of(&digest(2))
                .expect("reading should succeed")
                .is_empty()
        );
    }

    // Writing the same edge twice leaves the same state, which is what lets an interrupted
    // collection be repeated.
    #[test]
    fn setting_an_edge_twice_is_one_edge() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        store
            .set_edge(&digest(1), &digest(10))
            .expect("setting should succeed");
        store
            .set_edge(&digest(1), &digest(10))
            .expect("setting again should succeed");

        assert_eq!(
            store
                .parents_of(&digest(1))
                .expect("reading should succeed")
                .len(),
            1
        );
    }

    // Removing an edge that is already gone succeeds, for the same reason.
    #[test]
    fn deleting_an_edge_twice_succeeds() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        store
            .set_edge(&digest(1), &digest(10))
            .expect("setting should succeed");

        store
            .delete_edge(&digest(1), &digest(10))
            .expect("deleting should succeed");
        store
            .delete_edge(&digest(1), &digest(10))
            .expect("deleting again should succeed");

        assert!(
            store
                .parents_of(&digest(1))
                .expect("reading should succeed")
                .is_empty()
        );
    }

    // A new edge carries no stamp, and stamping it records what was proven without disturbing the
    // edge itself.
    #[test]
    fn an_edge_starts_unstamped_and_can_be_stamped() {
        let tmp = TestableTmpdir::new();
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");

        store
            .set_edge(&digest(1), &digest(10))
            .expect("setting should succeed");

        assert_eq!(
            store
                .parents_of(&digest(1))
                .expect("reading should succeed")[0]
                .1,
            Stamp::UNKNOWN
        );

        let seq = Seq::FIRST.next();
        store
            .stamp_edge(&digest(1), &digest(10), Stamp::at(seq))
            .expect("stamping should succeed");

        let parents = store
            .parents_of(&digest(1))
            .expect("reading should succeed");
        assert_eq!(parents.len(), 1, "stamping should not add an edge");
        assert_eq!(parents[0].1, Stamp::at(seq));
    }

    // An unknown stamp is below every floor, so it never claims a child is held and always forces
    // the walk that settles the question.
    #[test]
    fn an_unknown_stamp_holds_at_no_floor() {
        assert!(!Stamp::UNKNOWN.holds_at(Seq::FIRST));
        assert!(!Stamp::UNKNOWN.holds_at(Seq::FIRST.next()));
    }

    // A stamp holds at its own floor and at any earlier one, and not at a later one.
    #[test]
    fn a_stamp_holds_from_its_own_floor_downwards() {
        let first = Seq::FIRST;
        let second = first.next();

        assert!(Stamp::at(second).holds_at(second));
        assert!(Stamp::at(second).holds_at(first));
        assert!(!Stamp::at(first).holds_at(second));
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
