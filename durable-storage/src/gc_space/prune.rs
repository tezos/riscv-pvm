// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Collection at directory granularity, and what it frees.
//!
//! At this granularity collection is no more than the removal of every commit directory that no
//! retained root reaches. What survives it is the point — see [`prune_unreachable`].

use std::path::Path;

use anyhow::Context;
use anyhow::Result;

use super::measure::commits_disk_usage;
use super::measure::disk_usage;
use super::sample::DiskUsage;
use crate::collect::collect;
use crate::commit::CommitId;
use crate::repo::DirectoryManager;

/// Collect at `target`, and report what that frees.
///
/// The collection itself is [`collect`], so this measures the shipped implementation rather than a
/// stand-in for it. Blocks come back only where no surviving link remains, which is exactly the
/// point — the dead node data inside a retained commit survives untouched, because that commit
/// still needs the files holding it.
pub(super) fn prune_unreachable(
    repo: &DirectoryManager,
    repo_path: &Path,
    target: &CommitId,
) -> Result<PruneOutcome> {
    let mut outcome = PruneOutcome {
        before: disk_usage(repo_path).context("measuring usage before pruning")?,
        ..PruneOutcome::default()
    };

    let collected = collect(repo, target).context("collecting")?;
    outcome.databases_removed = collected.database_commits as u64;
    outcome.registries_removed = collected.registry_commits as u64;

    outcome.after = disk_usage(repo_path).context("measuring usage after pruning")?;
    outcome.after_commits =
        commits_disk_usage(repo_path).context("measuring committed history after pruning")?;

    Ok(outcome)
}

/// What a directory-level collection removed and freed.
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
