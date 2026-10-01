// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Snapshot retention for the [`Registry`] long test.
//!
//! Epoch bases are the only commits a run records, so retention is a window over them: keep the
//! `keep` most recent and hand the oldest of those to the repository's own collection. Driving
//! [`collect_all`] and [`collect`] rather than copies of them means the run exercises what ships,
//! and applying collection to both backends keeps the persistent and in-memory repositories bounded
//! the same way - which is the only place the in-memory repository's side of collection is
//! exercised at all.
//!
//! The persistent repository also has its Merkle nodes swept, and then a reclaim started that runs
//! alongside the next epoch's commits. [`ReclaimGuard`] waits for it before the repository is dropped.
//!
//! [`Registry`]: crate::registry::Registry

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use anyhow::Context;
use anyhow::Result;

use crate::collect::Round;
use crate::collect::Suspend;
use crate::collect::collect;
use crate::collect::collect_all;
use crate::commit::CommitId;
use crate::repo::DirectoryManager;
use crate::storage::in_memory::InMemoryRepo;

/// Drop epoch snapshots older than the `keep` most-recent, reclaiming what they held.
///
/// Collecting at the oldest snapshot in the window retains it and everything recorded after it,
/// which is exactly the window: a run records a commit per epoch base and nothing in between. The
/// database commits and Merkle nodes an evicted snapshot shared with a retained one survive, since
/// collection removes only those no retained root reaches.
///
/// Returns the persistent repository's round, or `None` if there was nothing to collect at.
pub(super) fn prune(
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    recent_commits: &mut VecDeque<CommitId>,
    keep: NonZeroUsize,
) -> Result<Option<Round>> {
    while recent_commits.len() > keep.get() {
        recent_commits.pop_front();
    }

    let Some(oldest) = recent_commits.front() else {
        return Ok(None);
    };

    let round = collect_all(persistent_repo, oldest, &Suspend::new())
        .context("collecting the persistent repository")?;
    collect(in_memory_repo, oldest, &Suspend::new())
        .context("collecting the in-memory repository")?;

    // A no-op if the previous epoch's reclaim is still running.
    persistent_repo.start_reclaim();

    Ok(Some(round))
}

/// Waits for a reclaim started by [`prune`] when dropped, so the store is not torn down beneath it
/// however the run ends.
pub(super) struct ReclaimGuard<'a>(pub(super) &'a DirectoryManager);

impl Drop for ReclaimGuard<'_> {
    fn drop(&mut self) {
        self.0.finish_reclaim();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::long_test::registry::tests::Fixture;
    use crate::registry::Registry;
    use crate::storage::in_memory::InMemoryKeyValueStore;

    /// Prune `recent` to a keep-1 window.
    fn prune_to_one(fixture: &Fixture, recent: &mut VecDeque<CommitId>) -> Option<Round> {
        prune(
            &fixture.persistent_repo,
            &fixture.in_memory_repo,
            recent,
            NonZeroUsize::new(1).expect("non-zero"),
        )
        .expect("pruning should succeed")
    }

    // Pruning past a keep-1 window drops the old registry manifests and the nodes only they held,
    // while keeping the retained base fully checkoutable on both backends, and keeps the database
    // commits and nodes that the retained base still shares.
    #[test]
    fn prunes_unreachable_snapshots() {
        let fixture = Fixture::new();
        let _reclaim = ReclaimGuard(&fixture.persistent_repo);

        // The first base's databases are empty and hold no nodes, so overwriting a key is what
        // leaves nodes for the sweep.
        let base0 = fixture.initial_base();
        let base1 = fixture.set(&base0, b"first");
        let base2 = fixture.set(&base1, b"second");
        assert_ne!(base1.commit, base2.commit, "bases should differ");

        let written_before = fixture
            .persistent_repo
            .merkle_store()
            .written_entries()
            .expect("counting the write log should succeed");

        let mut recent = VecDeque::from([base0.commit, base1.commit, base2.commit]);
        let round = prune_to_one(&fixture, &mut recent).expect("a full window should be collected");

        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0], base2.commit);
        assert!(!round.suspended, "an unsuspended round should finish");
        assert_eq!(round.collected.registry_commits, 2);
        assert!(
            round.swept.nodes > 0,
            "the overwritten value's nodes should be swept"
        );

        let written_after = fixture
            .persistent_repo
            .merkle_store()
            .written_entries()
            .expect("counting the write log should succeed");
        assert!(
            written_after < written_before,
            "the swept nodes should leave the write log ({written_after} >= {written_before})"
        );

        // The evicted bases' manifests are gone on both backends.
        for evicted in [base0.commit, base1.commit] {
            assert!(
                !fixture
                    .persistent_repo
                    .registry_commit_file(&evicted)
                    .exists(),
                "an evicted registry manifest should be removed"
            );
            assert!(
                Registry::<InMemoryKeyValueStore, Normal>::checkout(
                    fixture.in_memory_repo.clone(),
                    evicted
                )
                .is_err(),
                "an evicted base should no longer check out in memory"
            );
        }

        // Both journals were pruned, so neither backend will collect at an evicted base
        // again - which is what a repeated round relies on to refuse a target whose data has
        // already gone.
        assert!(
            collect(&fixture.persistent_repo, &base1.commit, &Suspend::new()).is_err(),
            "an evicted base should no longer be a persistent collection target"
        );
        assert!(
            collect(&fixture.in_memory_repo, &base1.commit, &Suspend::new()).is_err(),
            "an evicted base should no longer be an in-memory collection target"
        );

        // The retained base still checks out fully on both backends, which requires its
        // (shared) database commits and nodes to have survived.
        fixture.assert_checks_out(base2.commit);
    }

    // The reclaim a prune starts runs alongside the next epoch: advancing the base while it
    // runs commits a state that still checks out once it has finished.
    #[test]
    fn advancing_during_a_reclaim_keeps_the_base() {
        let fixture = Fixture::new();
        let _reclaim = ReclaimGuard(&fixture.persistent_repo);

        let base0 = fixture.initial_base();
        let base1 = fixture.set(&base0, b"first");
        let base2 = fixture.set(&base1, b"second");

        let mut recent = VecDeque::from([base1.commit, base2.commit]);
        prune_to_one(&fixture, &mut recent);

        let base3 = fixture.set(&base2, b"third");
        recent.push_back(base3.commit);
        fixture.persistent_repo.finish_reclaim();
        fixture.assert_checks_out(base3.commit);

        // And the next round collects on top of it.
        prune_to_one(&fixture, &mut recent);
        fixture.persistent_repo.finish_reclaim();
        fixture.assert_checks_out(base3.commit);
    }

    // A multi-epoch run with `keep_epochs: 1` retains exactly one registry
    // manifest on disk, independent of the number of epochs.
    #[test]
    fn keep_epochs_bounds_the_repo() {
        let tmp = TestableTmpdir::new();
        let out_dir = tmp.path().join("run");

        super::super::run_long_test(
            crate::long_test::LongTestConfig {
                epochs: Some(4),
                ops_per_epoch: 20,
                cases_per_epoch: 4,
                seed: None,
                time_budget: None,
                keep_epochs: Some(NonZeroUsize::new(1).expect("non-zero")),
                out_dir: Some(out_dir.clone()),
                fail_on_warning: false,
            },
            2,
            false,
        )
        .expect("the bounded run should succeed");

        let manifests = fs::read_dir(out_dir.join("repo").join("registries").join("commits"))
            .expect("the registry commits dir should exist")
            .count();
        assert_eq!(
            manifests, 1,
            "keep-epochs 1 should retain exactly one registry manifest"
        );
    }
}
