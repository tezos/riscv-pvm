// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! What one measurement of the repository records.
//!
//! [`Sample`] is the whole of it; the types it is built from each answer a different question
//! about the same commit, so they are grouped here rather than beside the code that fills them
//! in.

use std::collections::HashMap;

use crate::persistence_layer::measurement::SstFile;

/// Byte totals for one committed database.
#[derive(Debug, Default, Clone, Copy)]
pub struct BlobBreakdown {
    /// Entries in the blob column family, live and dead together.
    pub entries: u64,

    /// Bytes stored in the blob column family, live and dead together.
    pub stored_bytes: u64,

    /// Distinct node bodies reachable from the committed root.
    pub live_entries: u64,

    /// Bytes of the node bodies reachable from the committed root, including their keys.
    pub live_bytes: u64,
}

impl BlobBreakdown {
    /// Bytes no longer reachable from the committed root. This is what collection could reclaim.
    pub fn dead_bytes(&self) -> u64 {
        self.stored_bytes.saturating_sub(self.live_bytes)
    }

    /// Fraction of the blob column family that is dead, between 0 and 1.
    pub fn dead_fraction(&self) -> f64 {
        if self.stored_bytes == 0 {
            return 0.0;
        }

        self.dead_bytes() as f64 / self.stored_bytes as f64
    }

    /// Fold another database's totals into these.
    pub(super) fn add(&mut self, other: &Self) {
        self.entries += other.entries;
        self.stored_bytes += other.stored_bytes;
        self.live_entries += other.live_entries;
        self.live_bytes += other.live_bytes;
    }
}

/// Disk occupancy of a directory tree, counting shared files once.
///
/// Commits are RocksDB checkpoints, which hard-link their SST files, so summing file sizes
/// massively overstates what a repository costs. `unique_bytes` counts the blocks behind each inode
/// once, which is what the filesystem actually spends.
///
/// Two figures say what that is cheaper *than*, and they are not interchangeable.
/// `apparent_bytes` is the naive sum of file lengths, so comparing it with `unique_bytes` mixes two
/// corrections that pull in opposite directions: sharing takes bytes away, while rounding each file
/// up to whole blocks puts them back. `linked_bytes` measures the same blocks as `unique_bytes` and
/// differs only in counting them once per link, so their difference is the sharing on its own.
#[derive(Debug, Default, Clone, Copy)]
pub struct DiskUsage {
    /// Sum of file lengths, counting hard-linked files once per link.
    pub apparent_bytes: u64,

    /// Blocks allocated, counting hard-linked files once per link.
    pub linked_bytes: u64,

    /// Blocks allocated, counting each inode once.
    pub unique_bytes: u64,

    /// Files encountered, counting hard-linked files once per link.
    pub files: u64,
}

impl DiskUsage {
    /// Blocks that sharing saves, being the same blocks counted per link and then per inode.
    ///
    /// Both sides are block-measured, so no rounding is folded into the answer.
    pub fn shared_bytes(&self) -> u64 {
        self.linked_bytes.saturating_sub(self.unique_bytes)
    }
}

/// Unique bytes pinned by the committed history, split by what they hold.
///
/// Retained checkpoints hard-link the files that existed when they were taken, so as compaction
/// rewrites SSTs the history ends up pinning several versions of the same data. This is what that
/// costs, attributed to the column family each file belongs to — which matters because moving the
/// Merkle side into a shared store would remove the `blob` part from every commit at a stroke.
#[derive(Debug, Default, Clone, Copy)]
pub struct PinnedBytes {
    /// Pinned bytes holding Merkle node bodies.
    pub blob: u64,

    /// Pinned bytes holding values.
    pub value: u64,

    /// Pinned bytes in files that are not SSTs, such as manifests and options.
    pub other: u64,
}

impl PinnedBytes {
    /// Total pinned bytes.
    pub fn total(&self) -> u64 {
        self.blob + self.value + self.other
    }
}

/// How much of what a commit pins it shares with the previously measured commit.
///
/// This is the measurement the whole retention story rests on. A checkpoint hard-links the files
/// that were live when it was taken, so untouched data should cost nothing to retain: the next
/// checkpoint links the same inodes. `new_bytes` is what that assumption fails by — files written
/// between the two commits, which the earlier checkpoint does not share and which therefore add to
/// the repository. It should be close to the data a commit actually changed; anything much larger
/// is compaction rewriting files that did not need to change.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sharing {
    /// Files present in both commits, so stored once.
    pub carried_files: u64,

    /// Bytes in files present in both commits.
    pub carried_bytes: u64,

    /// Files present only in the later commit, so added to the repository.
    pub new_files: u64,

    /// Bytes in files present only in the later commit.
    pub new_bytes: u64,

    /// Files present only in the earlier commit, still pinned by it.
    pub dropped_files: u64,

    /// Bytes in files present only in the earlier commit.
    pub dropped_bytes: u64,
}

impl Sharing {
    /// Fraction of the later commit's bytes that were already on disk, between 0 and 1.
    pub fn carried_fraction(&self) -> f64 {
        let total = self.carried_bytes + self.new_bytes;

        if total == 0 {
            return 0.0;
        }

        self.carried_bytes as f64 / total as f64
    }
}

/// Files and bytes at one LSM level, summed over the registry's databases.
///
/// Included because the shape of the tree explains the sharing: if a database fits inside the base
/// level, every compaction from level zero rewrites all of it.
#[derive(Debug, Clone)]
pub struct LevelSummary {
    /// LSM level.
    pub level: i32,

    /// Files at this level.
    pub files: u64,

    /// Bytes at this level.
    pub bytes: u64,
}

/// The SST files a commit pins, keyed by database index and file name.
///
/// Keyed that way because file numbering is per instance, so the same name means the same file only
/// within one database's lineage.
pub(super) type FileSet = HashMap<(usize, String), SstFile>;

/// One measurement of the repository, after a commit.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Index of the commit just made, counting from 1. Zero is the base state.
    pub commit: usize,

    /// Blob column family totals, summed over the registry's databases.
    pub blob: BlobBreakdown,

    /// Bytes stored in the value column families, summed over the registry's databases.
    pub value_stored_bytes: u64,

    /// Occupancy of the whole repository directory.
    pub disk: DiskUsage,

    /// What the committed history pins, attributed by column family.
    pub pinned: PinnedBytes,

    /// Sharing with the previously measured commit, absent for the first measurement.
    pub sharing: Option<Sharing>,

    /// Distribution of files over LSM levels.
    pub levels: Vec<LevelSummary>,

    /// Database commit directories present in the repository.
    pub commit_dirs: u64,

    /// How long the commit took.
    pub commit_ms: u64,
}
