// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Recoverable images of the Merkle store.
//!
//! A database commit checkpoints that database's values. The nodes indexing them are in the store
//! shared by the repository, which no commit checkpoints, so committing syncs it instead - enough
//! that a commit never refers to a node a crash could take with it, but it leaves the store itself
//! as the only copy of every node in the repository.
//!
//! A **full commit** takes one: a checkpoint of the store into a numbered slot. Recovery opens the
//! most recent slot. Because a slot is made by a checkpoint it is self-contained and needs no
//! write-ahead log, and because a checkpoint hard-links rather than copies, taking one costs
//! almost nothing.
//!
//! # Why slots are numbered
//!
//! A slot is created by renaming a finished directory to a name that does not exist yet, which is
//! atomic and cannot half-happen. Numbering is what makes the name fresh. The alternative - a fixed
//! name plus a `CURRENT` pointer - needs the pointer swapped separately, and renaming over an
//! existing directory is not available anyway: `rename(2)` fails with `ENOTEMPTY` when the target
//! is a non-empty directory.
//!
//! # What a slot costs
//!
//! Its hard links hold the files that were live when it was taken, so compaction rewriting those
//! files does not free them until the slot is reaped. The high-water mark is therefore the live set
//! plus roughly one full-commit period of garbage, which makes the full-commit cadence the
//! reclamation cadence.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use super::MerkleStore;
use crate::errors::OperationalError;

/// Prefix marking a slot that is still being written.
///
/// A checkpoint creates its directory as it goes, so it is built under a name no reader looks for
/// and renamed into place once it is complete. A leftover means a full commit was interrupted; it
/// is not a slot and is replaced by the next attempt.
const PENDING_PREFIX: &str = ".pending-";

/// Suffix of the file a slot's lease is taken on.
///
/// Beside the slot rather than inside it, so that taking a lease does not write into a directory
/// whose whole point is being an untouched image, and so the lock survives the slot's removal long
/// enough for the remover to hold it.
const LEASE_SUFFIX: &str = ".lease";

/// Which full commit a slot holds, counting from one.
pub type SlotId = u64;

/// A held claim on a slot, keeping it from being reaped.
///
/// Slots are what other processes read, since the live store cannot be opened by a second writer.
/// A reader takes one of these for as long as it is reading, and reaping skips any slot that is
/// leased.
///
/// The claim is an `flock`, so the kernel drops it when the process holding it dies. A reader that
/// crashes therefore cannot pin a slot forever, which a lockfile checked by hand would allow.
#[derive(Debug)]
pub struct SlotLease {
    /// Releasing the lock is this handle being dropped.
    _file: fs::File,

    /// Which slot is claimed.
    slot: SlotId,
}

impl SlotLease {
    /// Which slot this lease claims.
    pub fn slot(&self) -> SlotId {
        self.slot
    }
}

/// Claim `slot` for reading, so that reaping leaves it alone.
///
/// Fails with [`OperationalError::CommitNotFound`] if there is no such slot. Several readers may
/// hold a lease on the same slot at once; only reaping needs it to itself.
pub fn lease_slot(slots_dir: &Path, slot: SlotId) -> Result<SlotLease, OperationalError> {
    if !slot_path(slots_dir, slot).exists() {
        return Err(OperationalError::CommitNotFound);
    }

    let file = lease_file(slots_dir, slot)?;

    // Shared: readers do not exclude each other, only the reaper.
    if !try_flock(&file, libc::LOCK_SH)? {
        return Err(OperationalError::CommitNotFound);
    }

    Ok(SlotLease { _file: file, slot })
}

/// Open, creating if needed, the file a slot's lease is taken on.
fn lease_file(slots_dir: &Path, slot: SlotId) -> Result<fs::File, OperationalError> {
    let path = lease_path(slots_dir, slot);

    fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|error| OperationalError::TempCreationFailed { path, error })
}

/// Where a slot's lease is taken.
fn lease_path(slots_dir: &Path, slot: SlotId) -> PathBuf {
    slots_dir.join(format!("{slot}{LEASE_SUFFIX}"))
}

/// Take `operation` on `file` without waiting, reporting whether it was granted.
///
/// Never blocks: a reaper that waited on a reader would hold up whatever asked it to reap, and the
/// answer it wants - leave this one alone for now - is available immediately.
fn try_flock(file: &fs::File, operation: libc::c_int) -> Result<bool, OperationalError> {
    use std::os::fd::AsRawFd;

    // Safety: the descriptor is owned by `file` and outlives the call.
    if unsafe { libc::flock(file.as_raw_fd(), operation | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }

    let error = std::io::Error::last_os_error();

    match error.raw_os_error() {
        // Held by someone else, which is an answer rather than a failure.
        Some(libc::EWOULDBLOCK) => Ok(false),
        _ => Err(OperationalError::FileReadFailed { error }),
    }
}

impl MerkleStore {
    /// Take a full commit: checkpoint this store into a new slot under `slots_dir`.
    ///
    /// Returns the slot taken, which is one past the highest already there. The live store carries
    /// on unchanged - whether to check anything out again afterwards is the caller's decision, not
    /// this one's.
    pub fn take_slot(&self, slots_dir: &Path) -> Result<SlotId, OperationalError> {
        if self.is_read_only() {
            return Err(OperationalError::RepositoryIsReadOnly);
        }

        fs::create_dir_all(slots_dir).map_err(|error| OperationalError::DirCreationFailed {
            path: slots_dir.to_path_buf(),
            error,
        })?;

        let slot = latest_slot(slots_dir)?.unwrap_or(0) + 1;

        // Built aside and moved into place, so a slot directory never exists half-written.
        let pending = slots_dir.join(format!("{PENDING_PREFIX}{slot}"));
        if pending.exists() {
            fs::remove_dir_all(&pending).map_err(|error| OperationalError::DirRemovalFailed {
                path: pending.clone(),
                error,
            })?;
        }

        self.checkpoint(&pending)?;

        // The target name is fresh, so this cannot hit an existing directory and is atomic.
        fs::rename(&pending, slot_path(slots_dir, slot))
            .map_err(|error| OperationalError::FileWriteFailed { error })?;

        Ok(slot)
    }
}

/// Where the slot numbered `slot` lives.
pub fn slot_path(slots_dir: &Path, slot: SlotId) -> PathBuf {
    slots_dir.join(slot.to_string())
}

/// Every slot present under `slots_dir`, oldest first.
///
/// Entries that are not slots are ignored, which covers a `.pending-` directory left by an
/// interrupted full commit as well as anything else that finds its way in.
pub fn slots(slots_dir: &Path) -> Result<Vec<SlotId>, OperationalError> {
    let entries = match fs::read_dir(slots_dir) {
        Ok(entries) => entries,
        // A repository that has never taken a full commit has no slots, which is not a failure.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(OperationalError::FileReadFailed { error }),
    };

    let mut slots = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| OperationalError::FileReadFailed { error })?;

        if let Some(slot) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse().ok())
        {
            slots.push(slot);
        }
    }

    slots.sort_unstable();

    Ok(slots)
}

/// The most recent slot under `slots_dir`, which is the one recovery opens.
pub fn latest_slot(slots_dir: &Path) -> Result<Option<SlotId>, OperationalError> {
    Ok(slots(slots_dir)?.last().copied())
}

/// Remove all but the `keep` most recent slots, returning how many went.
///
/// Reaping is what actually returns the disk a slot was holding: until then its hard links keep the
/// files that were live when it was taken, however much compaction has rewritten since.
///
/// A slot someone holds a [`SlotLease`] on is left where it is and reported in the log, to be
/// reaped by a later round once the reader has finished. Keeping fewer than one slot is refused
/// rather than obeyed, since it would leave the repository with no image to recover from.
pub fn reap_slots(slots_dir: &Path, keep: usize) -> Result<usize, OperationalError> {
    let keep = keep.max(1);
    let slots = slots(slots_dir)?;

    let Some(drop_count) = slots.len().checked_sub(keep) else {
        return Ok(0);
    };

    let mut reaped = 0;
    for slot in slots.into_iter().take(drop_count) {
        if reap_slot(slots_dir, slot)? {
            reaped += 1;
        }
    }

    Ok(reaped)
}

/// Remove one slot if nothing is reading it, reporting whether it went.
fn reap_slot(slots_dir: &Path, slot: SlotId) -> Result<bool, OperationalError> {
    let lease = lease_file(slots_dir, slot)?;

    // Exclusive, and held for the removal: a reader taking a shared lease meanwhile would find the
    // slot half-removed otherwise.
    if !try_flock(&lease, libc::LOCK_EX)? {
        log::info!("leaving Merkle store slot {slot} in place: a reader holds a lease on it");
        return Ok(false);
    }

    let path = slot_path(slots_dir, slot);

    match fs::remove_dir_all(&path) {
        Ok(()) => {}
        // Already gone, which is what a repeated reap finds.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(OperationalError::DirRemovalFailed { path, error }),
    }

    // The lock is on this file, so it is unlinked last and while still held. A reader arriving
    // afterwards creates a fresh one and is granted a lease on a slot that is no longer there,
    // which is why taking a lease checks that the slot exists and reading it can still fail.
    if let Err(error) = fs::remove_file(lease_path(slots_dir, slot))
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(OperationalError::FileWriteFailed { error });
    }

    Ok(true)
}

/// Put the contents of `slot` back where the live store is opened from.
///
/// Recovery after the live store is lost. The slot is a complete store, so it becomes the live one
/// by being copied to where one is expected; it is copied rather than moved so the slot survives a
/// recovery that itself fails part-way.
///
/// Must run before anything opens the store: files appearing underneath an open RocksDB instance
/// are not its own. Fails if a store is already there, rather than mixing the two.
pub fn restore_from_slot(slot_dir: &Path, store_dir: &Path) -> Result<(), OperationalError> {
    if store_dir.exists() {
        return Err(OperationalError::DirCreationFailed {
            path: store_dir.to_path_buf(),
            error: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "a Merkle store is already present; remove it before restoring over it",
            ),
        });
    }

    fs::create_dir_all(store_dir).map_err(|error| OperationalError::DirCreationFailed {
        path: store_dir.to_path_buf(),
        error,
    })?;

    // A RocksDB directory has no subdirectories, so a flat copy is a faithful one.
    for entry in
        fs::read_dir(slot_dir).map_err(|error| OperationalError::FileReadFailed { error })?
    {
        let entry = entry.map_err(|error| OperationalError::FileReadFailed { error })?;

        if !entry.file_type().is_ok_and(|kind| kind.is_file()) {
            continue;
        }

        fs::copy(entry.path(), store_dir.join(entry.file_name()))
            .map_err(|error| OperationalError::FileWriteFailed { error })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use octez_riscv_test_utils::TestableTmpdir;

    use super::super::open_shared;
    use super::*;

    /// A store with one node in it, and somewhere to put slots.
    fn fixture(tmp: &TestableTmpdir) -> (std::sync::Arc<MerkleStore>, PathBuf) {
        let store = open_shared(&tmp.path().join("merkle")).expect("opening should succeed");
        store.set(b"key", b"body").expect("setting should succeed");

        (store, tmp.path().join("slots"))
    }

    // Slots are numbered from one and count upwards, so each name is fresh.
    #[test]
    fn slots_are_numbered_upwards() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        assert_eq!(
            store
                .take_slot(&slots_dir)
                .expect("the first should succeed"),
            1
        );
        assert_eq!(
            store
                .take_slot(&slots_dir)
                .expect("the second should succeed"),
            2
        );

        assert_eq!(
            slots(&slots_dir).expect("listing should succeed"),
            vec![1, 2]
        );
        assert_eq!(
            latest_slot(&slots_dir).expect("reading should succeed"),
            Some(2)
        );
    }

    // A repository that has never taken a full commit has no slots, rather than failing on the
    // missing directory.
    #[test]
    fn a_repository_without_slots_reports_none() {
        let tmp = TestableTmpdir::new();

        assert!(
            slots(&tmp.path().join("slots"))
                .expect("listing should succeed")
                .is_empty()
        );
        assert_eq!(
            latest_slot(&tmp.path().join("slots")).expect("reading should succeed"),
            None
        );
    }

    // A slot holds what the store held when it was taken, and can be opened on its own.
    #[test]
    fn a_slot_is_a_store_in_its_own_right() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        let slot = store
            .take_slot(&slots_dir)
            .expect("the slot should be taken");

        // Written after the slot was taken, so it should not appear in it.
        store
            .set(b"later", b"body")
            .expect("setting should succeed");
        drop(store);

        let opened = open_shared(&slot_path(&slots_dir, slot)).expect("the slot should open");
        assert_eq!(
            opened
                .get(b"key")
                .expect("the node should be there")
                .as_ref(),
            b"body"
        );
        assert!(
            opened.get(b"later").is_err(),
            "the slot should hold the store as it was, not as it became"
        );
    }

    // A directory left behind by an interrupted full commit is not mistaken for a slot, and the
    // next attempt replaces it.
    #[test]
    fn an_interrupted_full_commit_leaves_no_slot() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        fs::create_dir_all(slots_dir.join(format!("{PENDING_PREFIX}1")))
            .expect("creating the leftover should succeed");

        assert!(
            slots(&slots_dir)
                .expect("listing should succeed")
                .is_empty(),
            "a pending directory is not a slot"
        );

        assert_eq!(
            store.take_slot(&slots_dir).expect("taking should succeed"),
            1,
            "the interrupted attempt's number should be reused"
        );
        assert_eq!(slots(&slots_dir).expect("listing should succeed"), vec![1]);
    }

    // Reaping keeps the most recent slots and drops the rest, and repeating it changes nothing.
    #[test]
    fn reaping_keeps_the_most_recent() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        for _ in 0..3 {
            store.take_slot(&slots_dir).expect("taking should succeed");
        }

        assert_eq!(
            reap_slots(&slots_dir, 2).expect("reaping should succeed"),
            1
        );
        assert_eq!(
            slots(&slots_dir).expect("listing should succeed"),
            vec![2, 3]
        );

        assert_eq!(
            reap_slots(&slots_dir, 2).expect("reaping again should succeed"),
            0
        );
        assert_eq!(
            slots(&slots_dir).expect("listing should succeed"),
            vec![2, 3]
        );
    }

    // Reaping never leaves a repository with nothing to recover from, however few it is told to
    // keep.
    #[test]
    fn reaping_always_leaves_one() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        store.take_slot(&slots_dir).expect("taking should succeed");
        store.take_slot(&slots_dir).expect("taking should succeed");

        reap_slots(&slots_dir, 0).expect("reaping should succeed");

        assert_eq!(slots(&slots_dir).expect("listing should succeed"), vec![2]);
    }

    // A store lost entirely comes back from its most recent slot.
    #[test]
    fn a_lost_store_is_recovered_from_its_slot() {
        let tmp = TestableTmpdir::new();
        let store_dir = tmp.path().join("merkle");
        let (store, slots_dir) = fixture(&tmp);

        let slot = store.take_slot(&slots_dir).expect("taking should succeed");
        drop(store);

        fs::remove_dir_all(&store_dir).expect("removing the store should succeed");

        restore_from_slot(&slot_path(&slots_dir, slot), &store_dir)
            .expect("restoring should succeed");

        let recovered = open_shared(&store_dir).expect("the recovered store should open");
        assert_eq!(
            recovered
                .get(b"key")
                .expect("the node should have come back")
                .as_ref(),
            b"body"
        );
    }

    // A leased slot is left alone by reaping, and the one after it still goes.
    #[test]
    fn reaping_leaves_a_leased_slot_alone() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        for _ in 0..3 {
            store.take_slot(&slots_dir).expect("taking should succeed");
        }

        let lease = lease_slot(&slots_dir, 1).expect("leasing should succeed");
        assert_eq!(lease.slot(), 1);

        assert_eq!(
            reap_slots(&slots_dir, 1).expect("reaping should succeed"),
            1,
            "only the unleased older slot should go"
        );
        assert_eq!(
            slots(&slots_dir).expect("listing should succeed"),
            vec![1, 3],
            "the leased slot should still be there"
        );

        // Once the reader is done, a later round takes it.
        drop(lease);
        assert_eq!(
            reap_slots(&slots_dir, 1).expect("reaping should succeed"),
            1
        );
        assert_eq!(slots(&slots_dir).expect("listing should succeed"), vec![3]);
    }

    // Several readers can hold a slot at once; leases exclude the reaper, not each other.
    #[test]
    fn leases_do_not_exclude_each_other() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        store.take_slot(&slots_dir).expect("taking should succeed");

        let first = lease_slot(&slots_dir, 1).expect("the first lease should succeed");
        let second = lease_slot(&slots_dir, 1).expect("the second lease should succeed");

        drop((first, second));
    }

    // A slot that is not there cannot be leased, so a reader learns that rather than holding a
    // claim on nothing.
    #[test]
    fn an_absent_slot_cannot_be_leased() {
        let tmp = TestableTmpdir::new();
        let (_store, slots_dir) = fixture(&tmp);

        assert!(matches!(
            lease_slot(&slots_dir, 7),
            Err(OperationalError::CommitNotFound)
        ));
    }

    // A lease file is not mistaken for a slot.
    #[test]
    fn a_lease_file_is_not_a_slot() {
        let tmp = TestableTmpdir::new();
        let (store, slots_dir) = fixture(&tmp);

        store.take_slot(&slots_dir).expect("taking should succeed");
        let lease = lease_slot(&slots_dir, 1).expect("leasing should succeed");

        assert_eq!(slots(&slots_dir).expect("listing should succeed"), vec![1]);

        drop(lease);
    }

    // Restoring over a store that is still there is refused, rather than mixing the two.
    #[test]
    fn restoring_over_an_existing_store_is_refused() {
        let tmp = TestableTmpdir::new();
        let store_dir = tmp.path().join("merkle");
        let (store, slots_dir) = fixture(&tmp);

        let slot = store.take_slot(&slots_dir).expect("taking should succeed");
        drop(store);

        assert!(restore_from_slot(&slot_path(&slots_dir, slot), &store_dir).is_err());
    }
}
