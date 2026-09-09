// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Snapshot retention for the [`Registry`] long test.
//!
//! Epoch bases are the only commits a run records, so retention is a window over them: keep the
//! `keep` most recent and hand the oldest of those to the repository's own collection. Driving
//! [`collect`] rather than a copy of it means the run exercises what ships, and applying it to
//! both backends keeps the persistent and in-memory repositories bounded the same way - which is
//! the only place the in-memory repository's side of collection is exercised at all.
//!
//! [`Registry`]: crate::registry::Registry

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use anyhow::Context;
use anyhow::Result;

use crate::collect::collect;
use crate::commit::CommitId;
use crate::repo::DirectoryManager;
use crate::storage::in_memory::InMemoryRepo;

/// Drop epoch snapshots older than the `keep` most-recent, reclaiming what they held.
///
/// Collecting at the oldest snapshot in the window retains it and everything recorded after it,
/// which is exactly the window: a run records a commit per epoch base and nothing in between. The
/// database commits an evicted snapshot shared with a retained one survive, since collection
/// removes only those no retained manifest reaches.
pub(super) fn prune(
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    recent_commits: &mut VecDeque<CommitId>,
    keep: NonZeroUsize,
) -> Result<()> {
    while recent_commits.len() > keep.get() {
        recent_commits.pop_front();
    }

    let Some(oldest) = recent_commits.front() else {
        return Ok(());
    };

    collect(persistent_repo, oldest).context("collecting the persistent repository")?;
    collect(in_memory_repo, oldest).context("collecting the in-memory repository")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::key::Key;
    use crate::long_test::registry::run_case::advance_base;
    use crate::long_test::registry::run_case::initial_base;
    use crate::persistence_layer::PersistenceLayer;
    use crate::registry::Registry;
    use crate::storage::in_memory::InMemoryKeyValueStore;
    use crate::test_helpers::database::DatabaseOperation;
    use crate::test_helpers::registry::RegistryOperation;

    const PERMANENT: usize = 2;

    // Pruning past a keep-1 window drops the old registry manifest while keeping
    // the retained base fully checkoutable on both backends, and keeps database
    // commits that the retained base still shares.
    #[test]
    fn prunes_unreachable_snapshots() {
        let tmp = TestableTmpdir::new();
        let repo_dir = tmp.path().join("repo");
        fs::create_dir_all(&repo_dir).expect("creating the repo dir should succeed");
        let persistent_repo = DirectoryManager::new(&repo_dir)
            .expect("creating the directory manager should succeed");
        let in_memory_repo = InMemoryRepo::default();

        let base0 = initial_base(&in_memory_repo, &persistent_repo, PERMANENT);
        let key = Key::new(&[1, 2, 3]).expect("the key should be valid");
        let ops = vec![RegistryOperation::Database(
            0,
            DatabaseOperation::Set(key, Bytes::from_static(b"value")),
        )];
        let base1 = advance_base(&in_memory_repo, &persistent_repo, &base0, &ops);
        assert_ne!(base0.commit, base1.commit, "bases should differ");

        let mut recent = VecDeque::from([base0.commit, base1.commit]);
        prune(
            &persistent_repo,
            &in_memory_repo,
            &mut recent,
            NonZeroUsize::new(1).expect("non-zero"),
        )
        .expect("pruning should succeed");

        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0], base1.commit);

        // The evicted base's manifest is gone on both backends.
        assert!(
            !persistent_repo.registry_commit_file(&base0.commit).exists(),
            "the evicted registry manifest should be removed"
        );
        assert!(
            Registry::<InMemoryKeyValueStore, Normal>::checkout(
                in_memory_repo.clone(),
                base0.commit
            )
            .is_err(),
            "the evicted base should no longer check out in memory"
        );

        // Both journals were pruned, so neither backend will collect at the evicted base
        // again - which is what a repeated round relies on to refuse a target whose data has
        // already gone.
        assert!(
            collect(&persistent_repo, &base0.commit).is_err(),
            "the evicted base should no longer be a persistent collection target"
        );
        assert!(
            collect(&in_memory_repo, &base0.commit).is_err(),
            "the evicted base should no longer be an in-memory collection target"
        );

        // The retained base still checks out fully on both backends, which
        // requires its (shared) database commits to have survived.
        Registry::<PersistenceLayer, Normal>::checkout(persistent_repo.clone(), base1.commit)
            .expect("the retained base should check out on the persistent backend");
        Registry::<InMemoryKeyValueStore, Normal>::checkout(in_memory_repo.clone(), base1.commit)
            .expect("the retained base should check out on the in-memory backend");
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
