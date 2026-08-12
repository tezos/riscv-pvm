// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Repository management for the Durable Storage

use std::collections::HashSet;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use octez_riscv_data::hash::Hash;
use tempfile::TempDir;

use crate::commit::CommitId;
use crate::errors::OperationalError;
use crate::journal;
use crate::journal::JournalEntry;
use crate::journal::Seq;
#[cfg(rocksdb)]
use crate::merkle_store::MerkleStore;

/// The [`DirectoryManager`] represents the root directory where commitments & internal data should
/// be stored.
///
/// # Clone implementation
///
/// This structure stores paths internally. Cloning those is safe as this structure imposes no
/// ownership over the paths. The [`tempfile`] crate provides safe APIs to create temporary
/// directories without naming conflicts, so we don't need to worry about that here.
#[derive(Clone)]
pub struct DirectoryManager {
    /// Directory prefix for [`crate::database::Database`] instances (temporary)
    temp_databases_dir: PathBuf,

    /// Directory prefix for [`crate::database::Database`] commitments
    database_commits_dir: PathBuf,

    /// Directory prefix for [`crate::registry::Registry`] commitments
    registry_commits_dir: PathBuf,

    /// Directory holding registry-wide data, including the commit journal
    registries_dir: PathBuf,

    /// Directory holding the repository-wide Merkle node store
    merkle_dir: PathBuf,

    /// The repository's Merkle node store, held open for as long as this handle lives.
    ///
    /// Held rather than opened per use so that the databases of a repository share one instance
    /// that stays open, instead of one that closes when the last database drops and is reopened by
    /// the next. Cloning a handle shares the store, as it shares everything else here.
    #[cfg(rocksdb)]
    merkle: std::sync::Arc<MerkleStore>,
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

        let registries_dir = path.join("registries");

        // The directory for registry commits holds files. Each file is named after the commit ID.
        let registry_commits_dir = registries_dir.join("commits");
        ensure_dir_exists(&registry_commits_dir)?;

        let merkle_dir = Self::merkle_dir_in(path);

        Ok(Self {
            temp_databases_dir,
            database_commits_dir,
            registry_commits_dir,
            registries_dir,
            #[cfg(rocksdb)]
            merkle: crate::merkle_store::open_shared(&merkle_dir)?,
            merkle_dir,
        })
    }

    /// Open an existing repository for reading, without taking its writer lock.
    ///
    /// The live Merkle store admits a single writer, so a process that is not the one writing cannot
    /// construct an ordinary handle onto a repository at all. This one reads the store instead,
    /// which any number of processes may do at once, and is how external tooling reaches a running
    /// node's storage.
    ///
    /// Reading a commit through this works as it does anywhere; writing through it does not.
    /// The view is of the store as it stood when this was opened.
    #[cfg(rocksdb)]
    pub fn open_read_only(path: &Path) -> Result<Self, OperationalError> {
        let databases_dir = path.join("databases");
        let registries_dir = path.join("registries");

        // Nothing is created: a reader opens a repository that exists, and creating directories in
        // one it does not own is not its business.
        Ok(Self {
            temp_databases_dir: databases_dir.join("temporary"),
            database_commits_dir: databases_dir.join("commits"),
            registry_commits_dir: registries_dir.join("commits"),
            registries_dir,
            merkle: crate::merkle_store::open_read_only(&Self::merkle_dir_in(path))?,
            merkle_dir: Self::merkle_dir_in(path),
        })
    }

    /// Whether this handle may only read.
    #[cfg(rocksdb)]
    pub fn is_read_only(&self) -> bool {
        self.merkle.is_read_only()
    }

    /// Fail if this handle may only read.
    #[cfg(rocksdb)]
    fn writeable(&self) -> Result<(), OperationalError> {
        if self.is_read_only() {
            return Err(OperationalError::RepositoryIsReadOnly);
        }

        Ok(())
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

    /// The file recording the order in which registry commits were made.
    pub fn journal_file(&self) -> PathBuf {
        self.registries_dir.join("journal")
    }

    /// The directory holding the repository's Merkle node store.
    pub fn merkle_store_dir(&self) -> PathBuf {
        self.merkle_dir.clone()
    }

    /// Where a repository rooted at `path` keeps its Merkle node store.
    ///
    /// Available without a handle, because the store is opened when one is constructed: anything
    /// that needs to put node bodies in place has to do it before that.
    pub fn merkle_dir_in(path: &Path) -> PathBuf {
        path.join("merkle")
    }

    /// The repository's Merkle node store.
    ///
    /// Every database of the repository reads and writes nodes here, so a node reached from several
    /// of them is stored once.
    #[cfg(rocksdb)]
    pub fn merkle_store(&self) -> &std::sync::Arc<MerkleStore> {
        &self.merkle
    }

    /// Where recoverable images of the Merkle store are kept.
    pub fn merkle_slots_dir(&self) -> PathBuf {
        Self::merkle_slots_dir_in(&self.merkle_dir)
    }

    /// Where a repository whose store is at `merkle_dir` keeps its slots.
    ///
    /// Beside the store rather than inside it, so that a slot is never mistaken for part of the
    /// RocksDB directory it was taken from.
    pub fn merkle_slots_dir_in(merkle_dir: &Path) -> PathBuf {
        let mut path = merkle_dir.as_os_str().to_owned();
        path.push("-commits");
        PathBuf::from(path)
    }

    /// Take a full commit: a recoverable image of the Merkle store, in a new slot.
    ///
    /// Ordinary committing puts a database's values in a commit directory and syncs the nodes they
    /// refer to, which is enough that a commit never outlives its nodes, but it leaves the store as
    /// the only copy of them. This is the operation that takes another, and it is deliberately the
    /// caller's to schedule: taking one bounds how much a crash costs, and reaping the one before
    /// it is what returns the disk that compaction has since made redundant.
    ///
    /// The live store carries on unchanged, and so does whatever else is using the repository: a
    /// full commit is a flush and a rename, and does not wait on the compaction that returns the
    /// disk a collection freed. That is [`DirectoryManager::start_reclaim`], separately.
    #[cfg(rocksdb)]
    pub fn full_commit(&self) -> Result<crate::merkle_store::slots::SlotId, OperationalError> {
        self.merkle.take_slot(&self.merkle_slots_dir())
    }

    /// Start reclaiming the disk that collection freed, in the background.
    ///
    /// Deleting a node writes a tombstone; the bytes come back when compaction rewrites the files
    /// without it. That rewrite costs the whole store rather than the garbage, so it is deliberately
    /// neither part of collecting nor part of taking a full commit - both would then be as slow as
    /// the most expensive thing they could trigger.
    ///
    /// Returns immediately, and returns whether this call started one. Reads, writes and commits
    /// carry on throughout.
    #[cfg(rocksdb)]
    pub fn start_reclaim(&self) -> bool {
        self.merkle.start_compaction()
    }

    /// Whether a reclaim started by [`DirectoryManager::start_reclaim`] is still running.
    #[cfg(rocksdb)]
    pub fn is_reclaiming(&self) -> bool {
        self.merkle.is_compacting()
    }

    /// The most recent full commit, which is the image recovery would open.
    #[cfg(rocksdb)]
    pub fn latest_full_commit(
        &self,
    ) -> Result<Option<crate::merkle_store::slots::SlotId>, OperationalError> {
        crate::merkle_store::slots::latest_slot(&self.merkle_slots_dir())
    }

    /// Drop all but the `keep` most recent full commits, returning how many went.
    ///
    /// A full commit someone holds a lease on is left where it is, for a later round to take.
    #[cfg(rocksdb)]
    pub fn reap_full_commits(&self, keep: usize) -> Result<usize, OperationalError> {
        crate::merkle_store::slots::reap_slots(&self.merkle_slots_dir(), keep)
    }

    /// Claim a full commit for reading, so that reaping leaves it alone while the lease is held.
    ///
    /// How another process reads a repository's Merkle nodes: the live store cannot be opened by a
    /// second writer, but a full commit is an immutable image that any number of readers may open.
    #[cfg(rocksdb)]
    pub fn lease_full_commit(
        &self,
        slot: crate::merkle_store::slots::SlotId,
    ) -> Result<crate::merkle_store::slots::SlotLease, OperationalError> {
        crate::merkle_store::slots::lease_slot(&self.merkle_slots_dir(), slot)
    }

    /// The most recently recorded journal entry, if the repository has committed anything.
    ///
    /// Reads only the tail of the journal, since assigning the next sequence number is on the
    /// commit path and the journal grows with every commit until collection prunes it.
    fn last_journal_entry(&self) -> Result<Option<JournalEntry>, OperationalError> {
        let mut journal = match std::fs::File::open(self.journal_file()) {
            Ok(journal) => journal,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(OperationalError::FileReadFailed { error }),
        };

        let len = journal
            .metadata()
            .map_err(|error| OperationalError::FileReadFailed { error })?
            .len();

        // A trailing partial entry is a torn write, so the last whole entry is the one before it.
        let whole = len / journal::ENTRY_BYTES as u64;
        let Some(last) = whole.checked_sub(1) else {
            return Ok(None);
        };

        journal
            .seek(SeekFrom::Start(last * journal::ENTRY_BYTES as u64))
            .map_err(|error| OperationalError::FileReadFailed { error })?;

        let mut bytes = [0u8; journal::ENTRY_BYTES];
        journal
            .read_exact(&mut bytes)
            .map_err(|error| OperationalError::FileReadFailed { error })?;

        Ok(Some(JournalEntry::decode(&bytes)))
    }
}

/// Persistence interface for a [`crate::registry::Registry`] repo.
pub trait RegistryRepo: Clone {
    /// Read the registry manifest bytes associated with `id`.
    ///
    /// Fails with [`OperationalError::CommitNotFound`] if no manifest exists for `id`.
    fn read_registry_commit(&self, id: &CommitId) -> Result<Vec<u8>, OperationalError>;

    /// Write `bytes` as the registry manifest for `id`.
    fn write_registry_commit(&self, id: &CommitId, bytes: &[u8]) -> Result<(), OperationalError>;

    /// Record `root` as the most recently committed registry root.
    ///
    /// Called after the manifest is written, so a crash in between leaves an unrecorded manifest
    /// rather than a recorded root with nothing behind it. The commit id was never returned to the
    /// caller in that case, so nothing can reference it and collection is free to reclaim it.
    fn record_commit(&self, root: &CommitId) -> Result<Seq, OperationalError>;

    /// Every commit recorded by [`RegistryRepo::record_commit`], in the order they were recorded.
    fn commit_journal(&self) -> Result<Vec<JournalEntry>, OperationalError>;

    /// The position the next [`RegistryRepo::record_commit`] will use.
    ///
    /// Read before a commit writes its nodes, so they can be recorded as belonging to it. A commit
    /// that then fails leaves that position unused, which costs nothing: positions order commits
    /// and need not be contiguous.
    fn next_commit_seq(&self) -> Result<Seq, OperationalError>;

    /// Keep only the journal entries whose root is in `retained`, dropping the rest.
    ///
    /// Replaces the journal in one step, so an interrupted prune leaves either the old journal or
    /// the new one.
    fn prune_journal(&self, retained: &HashSet<CommitId>) -> Result<(), OperationalError>;

    /// Every registry commit the repository currently holds a manifest for.
    ///
    /// Unordered. Collection enumerates what is present rather than what the journal mentions, so
    /// that a manifest left behind by an interrupted commit or an interrupted collection is still
    /// found.
    fn registry_commits(&self) -> Result<Vec<CommitId>, OperationalError>;

    /// Every database commit the repository currently holds.
    ///
    /// Unordered, and enumerated from what is present for the same reason as
    /// [`RegistryRepo::registry_commits`].
    fn database_commits(&self) -> Result<Vec<CommitId>, OperationalError>;

    /// Remove the manifest for the registry commit `id`.
    ///
    /// Removing one that is already gone succeeds, so collection can be repeated after an
    /// interruption.
    fn remove_registry_commit(&self, id: &CommitId) -> Result<(), OperationalError>;

    /// Remove the database commit `id`.
    ///
    /// Idempotent, for the same reason as [`RegistryRepo::remove_registry_commit`].
    fn remove_database_commit(&self, id: &CommitId) -> Result<(), OperationalError>;
}

impl RegistryRepo for DirectoryManager {
    fn read_registry_commit(&self, id: &CommitId) -> Result<Vec<u8>, OperationalError> {
        let commit_path = self.registry_commit_file(id);
        std::fs::read(&commit_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                OperationalError::CommitNotFound
            } else {
                OperationalError::FileReadFailed { error }
            }
        })
    }

    fn write_registry_commit(&self, id: &CommitId, bytes: &[u8]) -> Result<(), OperationalError> {
        #[cfg(rocksdb)]
        self.writeable()?;

        let commit_path = self.registry_commit_file(id);
        std::fs::write(&commit_path, bytes)
            .map_err(|error| OperationalError::FileWriteFailed { error })
    }

    fn next_commit_seq(&self) -> Result<Seq, OperationalError> {
        Ok(match self.last_journal_entry()? {
            Some(last) => last.seq.next(),
            None => Seq::FIRST,
        })
    }

    fn record_commit(&self, root: &CommitId) -> Result<Seq, OperationalError> {
        #[cfg(rocksdb)]
        self.writeable()?;

        let seq = self.next_commit_seq()?;

        let entry = JournalEntry { seq, root: *root };

        let mut journal = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.journal_file())
            .map_err(|error| OperationalError::FileWriteFailed { error })?;

        journal
            .write_all(&entry.encode())
            .map_err(|error| OperationalError::FileWriteFailed { error })?;

        Ok(seq)
    }

    fn commit_journal(&self) -> Result<Vec<JournalEntry>, OperationalError> {
        match std::fs::read(self.journal_file()) {
            Ok(bytes) => Ok(journal::decode_entries(&bytes)),
            // A repository that has never committed has no journal, which reads as no entries
            // rather than as a failure.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(OperationalError::FileReadFailed { error }),
        }
    }

    fn prune_journal(&self, retained: &HashSet<CommitId>) -> Result<(), OperationalError> {
        #[cfg(rocksdb)]
        self.writeable()?;

        let kept: Vec<u8> = self
            .commit_journal()?
            .into_iter()
            .filter(|entry| retained.contains(&entry.root))
            .flat_map(|entry| entry.encode())
            .collect();

        // Written beside the journal and renamed over it, so the replacement is atomic: a crash
        // leaves either the journal as it was or the pruned one, never a half-written mixture.
        let pending = self.journal_file().with_extension("pending");
        std::fs::write(&pending, &kept)
            .map_err(|error| OperationalError::FileWriteFailed { error })?;
        std::fs::rename(&pending, self.journal_file())
            .map_err(|error| OperationalError::FileWriteFailed { error })
    }

    fn registry_commits(&self) -> Result<Vec<CommitId>, OperationalError> {
        commit_ids_in(&self.registry_commits_dir)
    }

    fn database_commits(&self) -> Result<Vec<CommitId>, OperationalError> {
        commit_ids_in(&self.database_commits_dir)
    }

    fn remove_registry_commit(&self, id: &CommitId) -> Result<(), OperationalError> {
        #[cfg(rocksdb)]
        self.writeable()?;

        match std::fs::remove_file(self.registry_commit_file(id)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(OperationalError::FileWriteFailed { error }),
        }
    }

    fn remove_database_commit(&self, id: &CommitId) -> Result<(), OperationalError> {
        #[cfg(rocksdb)]
        self.writeable()?;

        let dir = self.database_commit_dir(id);
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(OperationalError::DirRemovalFailed { path: dir, error }),
        }
    }
}

/// The commit ids named by the entries of `dir`.
///
/// Entries whose name is not a commit id are left alone: the durable storage does not put anything
/// else in these directories, so one that appears belongs to something else and is not ours to
/// interpret or remove.
fn commit_ids_in(dir: &Path) -> Result<Vec<CommitId>, OperationalError> {
    let entries =
        std::fs::read_dir(dir).map_err(|error| OperationalError::FileReadFailed { error })?;

    let mut ids = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| OperationalError::FileReadFailed { error })?;

        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };

        let Ok(bytes) = hex::decode(&name) else {
            continue;
        };

        let Ok(digest) = <[u8; Hash::DIGEST_SIZE]>::try_from(bytes.as_slice()) else {
            continue;
        };

        ids.push(CommitId::from(Hash::from(digest)));
    }

    Ok(ids)
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::hash::Hash;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;

    fn root(byte: u8) -> CommitId {
        CommitId::from(Hash::from([byte; Hash::DIGEST_SIZE]))
    }

    fn manager(tmp: &TestableTmpdir) -> DirectoryManager {
        DirectoryManager::new(tmp.path()).expect("creating the directory manager should succeed")
    }

    // A repository being written to can be opened for reading at the same time. The live Merkle
    // store admits a single writer, so without this a second process could not open a repository at
    // all - which is how external tooling reaches a running node.
    #[cfg(rocksdb)]
    #[test]
    fn a_reader_opens_alongside_the_writer() {
        let tmp = TestableTmpdir::new();
        let writer = manager(&tmp);

        let reader = DirectoryManager::open_read_only(tmp.path())
            .expect("a reader should open alongside the writer");

        assert!(reader.is_read_only());
        assert!(!writer.is_read_only());

        // Writing through the reader is refused rather than failing somewhere further down.
        assert!(matches!(
            reader.record_commit(&root(1)),
            Err(OperationalError::RepositoryIsReadOnly)
        ));

        // The writer carries on.
        writer
            .record_commit(&root(1))
            .expect("the writer should still record");
    }

    // A repository that has never committed reads as an empty journal rather than failing on the
    // missing file.
    #[test]
    fn a_fresh_repository_has_an_empty_journal() {
        let tmp = TestableTmpdir::new();

        assert!(
            manager(&tmp)
                .commit_journal()
                .expect("reading an absent journal should succeed")
                .is_empty()
        );
    }

    // Recorded commits are numbered from zero, in order, and read back in the order they were
    // recorded.
    #[test]
    fn commits_are_recorded_in_order() {
        let tmp = TestableTmpdir::new();
        let repo = manager(&tmp);

        let seqs: Vec<Seq> = (1..=3)
            .map(|byte| {
                repo.record_commit(&root(byte))
                    .expect("recording should succeed")
            })
            .collect();

        assert_eq!(
            seqs,
            vec![Seq::FIRST, Seq::FIRST.next(), Seq::FIRST.next().next()]
        );
        assert_eq!(
            repo.commit_journal().expect("reading should succeed"),
            seqs.into_iter()
                .zip(1..=3)
                .map(|(seq, byte)| JournalEntry {
                    seq,
                    root: root(byte),
                })
                .collect::<Vec<_>>()
        );
    }

    // The journal is on disk, so a repository reopened at the same path continues numbering where
    // it left off instead of restarting and colliding with recorded positions.
    #[test]
    fn numbering_continues_across_reopening() {
        let tmp = TestableTmpdir::new();

        manager(&tmp)
            .record_commit(&root(1))
            .expect("recording should succeed");
        let seq = manager(&tmp)
            .record_commit(&root(2))
            .expect("recording should succeed");

        assert_eq!(seq, Seq::FIRST.next());
        assert_eq!(
            manager(&tmp)
                .commit_journal()
                .expect("reading should succeed")
                .len(),
            2
        );
    }

    // A crash part-way through appending an entry leaves a partial record. Reading skips it, and
    // the next commit is numbered from the last whole entry, overwriting nothing.
    #[test]
    fn a_torn_final_entry_is_ignored_and_renumbered_over() {
        let tmp = TestableTmpdir::new();
        let repo = manager(&tmp);

        repo.record_commit(&root(1))
            .expect("recording should succeed");

        // Simulate the interrupted append of a second entry.
        let mut torn = std::fs::read(repo.journal_file()).expect("the journal should exist");
        torn.extend_from_slice(&[0xff; journal::ENTRY_BYTES - 1]);
        std::fs::write(repo.journal_file(), &torn)
            .expect("writing the torn journal should succeed");

        assert_eq!(
            repo.commit_journal().expect("reading should succeed"),
            vec![JournalEntry {
                seq: Seq::FIRST,
                root: root(1)
            }],
            "the partial entry should not be reported"
        );

        let seq = repo
            .record_commit(&root(2))
            .expect("recording after a torn write should succeed");
        assert_eq!(seq, Seq::FIRST.next());
    }
}
