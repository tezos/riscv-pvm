// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! One measurement of a committed registry.
//!
//! [`measure`] produces a [`Sample`]. Everything else here is one of its figures: what the
//! column families hold, what the directory tree occupies, what the history pins, and how the
//! files are spread over LSM levels.

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use octez_riscv_data::hash::Hash;

use super::sample::BlobBreakdown;
use super::sample::DiskUsage;
use super::sample::FileSet;
use super::sample::LevelSummary;
use super::sample::PinnedBytes;
use super::sample::Sample;
use super::sample::Sharing;
use crate::avl::node::walk_stored_tree;
use crate::commit::CommitId;
use crate::persistence_layer::ReadOnlyPersistenceLayer;
use crate::persistence_layer::measurement::SstOwner;
use crate::repo::DirectoryManager;
use crate::storage::ReadOnlyKeyValueStore;

/// Measure every database of a registry commit, plus the repository as a whole.
pub(super) fn measure(
    repo: &DirectoryManager,
    repo_path: &Path,
    registry_commit: &CommitId,
    commit_index: usize,
    commit_time: Duration,
    previous: Option<&FileSet>,
) -> Result<(Sample, FileSet)> {
    let started = Instant::now();

    let database_commits = crate::registry::database_commits(repo, registry_commit)
        .context("reading the registry manifest being measured")?;

    let mut value_entries = 0;
    let mut value_stored_bytes = 0;
    let mut sst_bytes = 0;
    let mut files: FileSet = HashMap::new();

    // One live set for the whole registry commit, not one per database. Nodes live in a store
    // shared by the repository, so a body reachable from two databases is one body, and counting it
    // twice would overstate what is live and understate what is dead.
    let mut live: HashSet<Hash> = HashSet::new();
    let mut live_bytes = 0;

    for (db_index, database_commit) in database_commits.iter().enumerate() {
        let committed = ReadOnlyPersistenceLayer::checkout_read_only(repo, database_commit)
            .with_context(|| {
                format!(
                    "opening database commit {} read-only",
                    database_commit.hex_encode()
                )
            })?;

        walk_live_nodes(&committed, database_commit, &mut live, &mut live_bytes)?;

        let values = committed
            .value_totals()
            .map_err(|error| anyhow::anyhow!("scanning the value column family: {error}"))?;
        value_entries += values.entries;
        value_stored_bytes += values.stored_bytes();

        sst_bytes += committed
            .sst_bytes()
            .map_err(|error| anyhow::anyhow!("reading SST file sizes: {error}"))?;

        for file in committed
            .sst_files()
            .map_err(|error| anyhow::anyhow!("listing SST files: {error}"))?
        {
            files.insert((db_index, file.name.clone()), file);
        }
    }

    // Read once for the repository, since the store is the repository's rather than any one
    // database's. Its SSTs are counted separately from the commit directories' for the same reason:
    // they are not hard-linked into any of them.
    let merkle = repo.merkle_store();
    let totals = merkle
        .totals()
        .map_err(|error| anyhow::anyhow!("scanning the Merkle store: {error}"))?;

    let blob = BlobBreakdown {
        entries: totals.entries,
        stored_bytes: totals.stored_bytes(),
        live_entries: live.len() as u64,
        live_bytes,
    };

    sst_bytes += merkle
        .sst_bytes()
        .map_err(|error| anyhow::anyhow!("reading Merkle store SST sizes: {error}"))?;

    let sample = Sample {
        commit: commit_index,
        registry_commit: registry_commit.hex_encode(),
        blob,
        value_entries,
        value_stored_bytes,
        sst_bytes,
        disk: disk_usage(repo_path).context("measuring repository disk usage")?,
        disk_commits: commits_disk_usage(repo_path).context("measuring committed history usage")?,
        pinned: pinned_by_cf(repo, repo_path)
            .context("attributing pinned bytes to column families")?,
        sharing: previous.map(|previous| sharing_between(previous, &files)),
        levels: level_summary(&files),
        commit_dirs: count_commit_dirs(repo_path)?,
        commit_ms: commit_time.as_millis() as u64,
        measure_ms: started.elapsed().as_millis() as u64,
    };

    Ok((sample, files))
}

/// Compare the files two commits pin, to see how much the later one reuses.
fn sharing_between(previous: &FileSet, current: &FileSet) -> Sharing {
    let mut sharing = Sharing::default();

    for (key, file) in current {
        if previous.contains_key(key) {
            sharing.carried_files += 1;
            sharing.carried_bytes += file.size;
        } else {
            sharing.new_files += 1;
            sharing.new_bytes += file.size;
        }
    }

    for (key, file) in previous {
        if !current.contains_key(key) {
            sharing.dropped_files += 1;
            sharing.dropped_bytes += file.size;
        }
    }

    sharing
}

/// Summarise how files are spread over LSM levels, lowest level first.
fn level_summary(files: &FileSet) -> Vec<LevelSummary> {
    let mut levels: HashMap<i32, (u64, u64)> = HashMap::new();

    for file in files.values() {
        let entry = levels.entry(file.level).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += file.size;
    }

    let mut levels: Vec<LevelSummary> = levels
        .into_iter()
        .map(|(level, (files, bytes))| LevelSummary {
            level,
            files,
            bytes,
        })
        .collect();

    levels.sort_by_key(|summary| summary.level);

    levels
}

/// Add the nodes reachable from one committed database to `live`.
///
/// Bodies are deduplicated against what is already there: content-addressed subtrees can be reached
/// by more than one path, and now that the store is shared, by more than one database.
fn walk_live_nodes(
    committed: &ReadOnlyPersistenceLayer,
    database_commit: &CommitId,
    live: &mut HashSet<Hash>,
    live_bytes: &mut u64,
) -> Result<()> {
    walk_stored_tree(committed, *database_commit.as_hash(), |hash, len| {
        if live.insert(hash) {
            // Each body is stored under its hash, so a live node costs a digest for its key too.
            *live_bytes += len as u64 + Hash::DIGEST_SIZE as u64;
        }
    })
    .with_context(|| {
        format!(
            "walking the tree committed at {}",
            database_commit.hex_encode()
        )
    })
}

/// Occupancy of a directory tree, counting each inode's blocks once.
///
/// Shared with the long tests, which report the size of their repository as it grows.
pub(crate) fn disk_usage(root: &Path) -> Result<DiskUsage> {
    let mut usage = DiskUsage::default();
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    accumulate_disk_usage(root, &mut usage, &mut seen)?;

    Ok(usage)
}

/// Occupancy of the committed history: the database commit directories and the registry manifests,
/// leaving out the working databases.
///
/// Note this is what the history *references*, not what deleting it would free: commits hard-link
/// the working database's files, so blocks counted here can be held by the working database too.
/// Use [`SpaceConfig::collect`] for what deletion actually reclaims.
pub(super) fn commits_disk_usage(repo_path: &Path) -> Result<DiskUsage> {
    let mut usage = DiskUsage::default();
    let mut seen: HashSet<(u64, u64)> = HashSet::new();

    // One `seen` set across both roots, so an inode linked from each is still counted once.
    for root in [
        repo_path.join("databases").join("commits"),
        repo_path.join("registries"),
    ] {
        accumulate_disk_usage(&root, &mut usage, &mut seen)?;
    }

    Ok(usage)
}

/// Add the occupancy of `root` to `usage`, skipping inodes already present in `seen`.
fn accumulate_disk_usage(
    root: &Path,
    usage: &mut DiskUsage,
    seen: &mut HashSet<(u64, u64)>,
) -> Result<()> {
    let mut pending = vec![root.to_path_buf()];

    while let Some(dir) = pending.pop() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("reading the directory {}", dir.display()))?;

        for entry in entries {
            let entry = entry.with_context(|| format!("reading an entry of {}", dir.display()))?;
            let metadata = entry
                .metadata()
                .with_context(|| format!("reading metadata of {}", entry.path().display()))?;

            if metadata.is_dir() {
                pending.push(entry.path());
                continue;
            }

            usage.files += 1;
            usage.apparent_bytes += metadata.len();
            usage.linked_bytes += metadata.blocks() * 512;

            // Hard-linked SSTs are shared between the working database and every checkpoint of
            // it, so their blocks must only be counted for whichever link is seen first. The gap
            // this opens against `linked_bytes` is what the sharing saves.
            if seen.insert((metadata.dev(), metadata.ino())) {
                usage.unique_bytes += metadata.blocks() * 512;
            }
        }
    }

    Ok(())
}

/// Attribute the bytes pinned by every database commit to the column family holding them.
///
/// The attribution comes from RocksDB's live-file metadata, read once per commit directory, because
/// the SSTs of both column families share a directory and nothing in their names says which is
/// which. Inodes are deduplicated across all commits, so a file hard-linked into twenty checkpoints
/// counts once: the result is what retaining this history actually costs the filesystem, split by
/// what it is holding.
fn pinned_by_cf(repo: &DirectoryManager, repo_path: &Path) -> Result<PinnedBytes> {
    let commits_dir = repo_path.join("databases").join("commits");
    let mut pinned = PinnedBytes::default();
    let mut seen: HashSet<(u64, u64)> = HashSet::new();

    let commits =
        fs::read_dir(&commits_dir).with_context(|| format!("reading {}", commits_dir.display()))?;

    for commit in commits {
        let commit =
            commit.with_context(|| format!("reading an entry of {}", commits_dir.display()))?;
        let dir = commit.path();

        if !dir.is_dir() {
            continue;
        }

        // Read the attribution before walking, and close the instance straight after: a sample can
        // visit many commits and there is no reason to hold them all open at once.
        let owners = {
            let committed = ReadOnlyPersistenceLayer::checkout_read_only_from_path(repo, &dir)
                .with_context(|| format!("opening {} read-only", dir.display()))?;

            committed
                .sst_column_families()
                .map_err(|error| anyhow::anyhow!("reading live file metadata: {error}"))?
        };

        let files = fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?;

        for file in files {
            let file = file.with_context(|| format!("reading an entry of {}", dir.display()))?;
            let metadata = file
                .metadata()
                .with_context(|| format!("reading metadata of {}", file.path().display()))?;

            if metadata.is_dir() || !seen.insert((metadata.dev(), metadata.ino())) {
                continue;
            }

            let bytes = metadata.blocks() * 512;

            match owners.get(file.file_name().to_string_lossy().as_ref()) {
                Some(SstOwner::Blob) => pinned.blob += bytes,
                Some(SstOwner::Value) => pinned.value += bytes,
                // Manifests, options, logs and locks: small, but they are part of the cost.
                None => pinned.other += bytes,
            }
        }
    }

    Ok(pinned)
}

/// Count the database commit directories in the repository.
fn count_commit_dirs(repo_path: &Path) -> Result<u64> {
    let path = repo_path.join("databases").join("commits");

    let entries = fs::read_dir(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut dirs = 0;

    for entry in entries {
        let entry = entry.with_context(|| format!("reading an entry of {}", path.display()))?;

        // A commit is a directory, so anything else the repository keeps here is not one.
        if entry
            .file_type()
            .with_context(|| format!("reading the type of {}", entry.path().display()))?
            .is_dir()
        {
            dirs += 1;
        }
    }

    Ok(dirs)
}
