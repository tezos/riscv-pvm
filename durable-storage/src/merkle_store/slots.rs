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

/// Which full commit a slot holds, counting from one.
pub type SlotId = u64;

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
/// Keeping fewer than one is refused rather than obeyed, since it would leave the repository with
/// no image to recover from.
pub fn reap_slots(slots_dir: &Path, keep: usize) -> Result<usize, OperationalError> {
    let keep = keep.max(1);
    let slots = slots(slots_dir)?;

    let Some(drop_count) = slots.len().checked_sub(keep) else {
        return Ok(0);
    };

    let mut reaped = 0;
    for slot in slots.into_iter().take(drop_count) {
        let path = slot_path(slots_dir, slot);

        match fs::remove_dir_all(&path) {
            Ok(()) => reaped += 1,
            // Already gone, which is what a repeated reap finds.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(OperationalError::DirRemovalFailed { path, error }),
        }
    }

    Ok(reaped)
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
