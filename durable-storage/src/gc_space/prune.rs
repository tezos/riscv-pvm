// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Collection at directory granularity, and what it frees.
//!
//! At this granularity collection is no more than the removal of every commit directory that no
//! retained root reaches. What survives it is the point — see [`prune_unreachable`].

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;

use super::measure::commits_disk_usage;
use super::measure::disk_usage;
use super::measure::peak_rss_bytes;
use super::sample::DiskUsage;
use crate::collect::Suspend;
use crate::collect::collect;
use crate::collect::collect_nodes;
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

    // Timed in three parts, because they scale with different things: the commit side with the
    // number of commit directories, the sweep with the number of nodes in the store whether they
    // are dead or not, and the compaction with the bytes it has to rewrite.
    let started = Instant::now();
    let collected = collect(repo, target, &Suspend::new()).context("collecting commits")?;
    outcome.commits_ms = started.elapsed().as_millis() as u64;

    // The sweep holds a key and an answer per node in the store, so what it needs to run is worth
    // knowing alongside how long it takes: at scale that is the binding constraint, not the time.
    let before_sweep = peak_rss_bytes();

    let started = Instant::now();
    let swept = collect_nodes(repo, target, &Suspend::new()).context("collecting nodes")?;
    outcome.sweep_ms = started.elapsed().as_millis() as u64;

    outcome.rss_before_sweep = before_sweep;
    outcome.peak_rss = peak_rss_bytes();

    outcome.databases_removed = collected.database_commits as u64;
    outcome.registries_removed = collected.registry_commits as u64;
    outcome.nodes_removed = swept.nodes as u64;
    outcome.node_bytes_removed = swept.bytes;
    outcome.edges_removed = swept.edges as u64;
    outcome.nodes_examined = swept.examined as u64;

    // A delete only marks the key; the space comes back when compaction rewrites the files without
    // it. Forcing that here is what makes the freed figure below the real one rather than a promise.
    let started = Instant::now();
    repo.merkle_store().compact();
    outcome.compact_ms = started.elapsed().as_millis() as u64;

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

    /// Merkle node bodies removed.
    pub nodes_removed: u64,

    /// Bytes those bodies occupied.
    pub node_bytes_removed: u64,

    /// Reverse edges removed with them.
    pub edges_removed: u64,

    /// Nodes the round had to consider, whether or not they turned out to be dead.
    pub nodes_examined: u64,

    /// Milliseconds spent removing commit directories and manifests.
    pub commits_ms: u64,

    /// Milliseconds spent deciding which nodes are live and deleting the rest.
    pub sweep_ms: u64,

    /// Milliseconds spent compacting the store so the deletions return disk.
    pub compact_ms: u64,

    /// High-water resident memory before the sweep began.
    pub rss_before_sweep: u64,

    /// High-water resident memory after it, so the rise is what the sweep needed.
    pub peak_rss: u64,
}

impl PruneOutcome {
    /// Bytes returned to the filesystem, counted as allocated blocks.
    pub fn freed_bytes(&self) -> u64 {
        self.before
            .unique_bytes
            .saturating_sub(self.after.unique_bytes)
    }
}
