// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Collection at directory granularity, and what it frees.
//!
//! At this granularity collection is no more than the removal of every commit directory that no
//! retained root reaches. What survives it is the point — see [`retain_only`].

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use octez_riscv_data::hash::Hash;

use super::measure::commits_disk_usage;
use super::measure::disk_usage;
use super::sample::DiskUsage;
use super::scenario::Reg;
use crate::commit::CommitId;
use crate::repo::DirectoryManager;

/// Delete every commit not reachable from `retained`.
///
/// The reachable database commits are read from the retained registry manifests, and every other
/// commit directory and manifest is removed. Blocks come back only where no surviving link
/// remains, which is exactly the point — the dead node data inside a retained commit survives this
/// untouched, because the retained commit still needs the files holding it.
///
/// Two callers want this for opposite reasons. Collection wants to know what it frees; a run over
/// a reused repository wants the commits earlier runs left to be gone before it starts, so that
/// what it measures does not depend on what ran before it.
pub fn retain_only(
    repo: &DirectoryManager,
    repo_path: &Path,
    retained: &[CommitId],
) -> Result<Removed> {
    let mut reachable: HashSet<CommitId> = HashSet::new();

    for commit in retained {
        reachable.extend(
            Reg::database_commits(repo, commit).context("reading a retained registry manifest")?,
        );
    }

    let mut removed = Removed::default();

    let commits_dir = repo_path.join("databases").join("commits");
    let entries =
        fs::read_dir(&commits_dir).with_context(|| format!("reading {}", commits_dir.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("reading an entry of {}", commits_dir.display()))?;

        // A commit is a directory, and `remove_dir_all` below would fail on anything else.
        if !entry
            .file_type()
            .with_context(|| format!("reading the type of {}", entry.path().display()))?
            .is_dir()
        {
            continue;
        }

        // A directory is named after its commit, so the name identifies what it holds.
        let retain = commit_id_of_name(&entry.file_name().to_string_lossy())
            .is_some_and(|id| reachable.contains(&id));

        if retain {
            continue;
        }

        fs::remove_dir_all(entry.path())
            .with_context(|| format!("removing {}", entry.path().display()))?;
        removed.databases += 1;
    }

    let registries_dir = repo_path.join("registries").join("commits");
    let entries = fs::read_dir(&registries_dir)
        .with_context(|| format!("reading {}", registries_dir.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("reading an entry of {}", registries_dir.display()))?;

        let name = entry.file_name().to_string_lossy().into_owned();

        if retained.iter().any(|commit| commit.hex_encode() == name) {
            continue;
        }

        fs::remove_file(entry.path())
            .with_context(|| format!("removing {}", entry.path().display()))?;
        removed.registries += 1;
    }

    Ok(removed)
}

/// What a collection removed.
#[derive(Debug, Default, Clone, Copy)]
pub struct Removed {
    /// Database commit directories removed.
    pub databases: u64,

    /// Registry manifests removed.
    pub registries: u64,
}

impl Removed {
    /// Commits removed in total, database directories and registry manifests together.
    pub fn total(&self) -> u64 {
        self.databases + self.registries
    }
}

/// Delete every commit not reachable from `retained`, and report what that frees.
pub(super) fn prune_unreachable(
    repo: &DirectoryManager,
    repo_path: &Path,
    retained: &[CommitId],
) -> Result<PruneOutcome> {
    let before = disk_usage(repo_path).context("measuring usage before pruning")?;
    let removed = retain_only(repo, repo_path, retained)?;

    Ok(PruneOutcome {
        before,
        after: disk_usage(repo_path).context("measuring usage after pruning")?,
        after_commits: commits_disk_usage(repo_path)
            .context("measuring committed history after pruning")?,
        databases_removed: removed.databases,
        registries_removed: removed.registries,
    })
}

/// What a simulated directory-level collection removed and freed.
#[derive(Debug, Default, Clone, Copy)]
pub struct PruneOutcome {
    /// Repository occupancy before pruning.
    pub before: DiskUsage,

    /// Repository occupancy after pruning.
    pub after: DiskUsage,

    /// Committed history occupancy after pruning.
    pub after_commits: DiskUsage,

    /// Database commit directories removed.
    pub databases_removed: u64,

    /// Registry manifests removed.
    pub registries_removed: u64,
}

impl PruneOutcome {
    /// Bytes returned to the filesystem, counted as allocated blocks.
    pub fn freed_bytes(&self) -> u64 {
        self.before
            .unique_bytes
            .saturating_sub(self.after.unique_bytes)
    }
}

/// Parse a hex commit directory or file name back into a [`CommitId`].
fn commit_id_of_name(name: &str) -> Option<CommitId> {
    let bytes = hex::decode(name).ok()?;
    let hash: [u8; Hash::DIGEST_SIZE] = bytes.as_slice().try_into().ok()?;

    Some(CommitId::from(Hash::from(hash)))
}
