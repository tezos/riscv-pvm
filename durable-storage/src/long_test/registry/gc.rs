// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Snapshot retention for the [`Registry`] long test.
//!
//! A registry commit references many shared, content-addressed database commits,
//! which can be referenced by other registry commits. When removing a registry commit,
//! its database commits are removed only if no other retained registry commit still
//! references them. Deletions apply
//! to both backends so the persistent repo and the in-memory repo
//! both stay bounded to the retention window.
//!
//! [`Registry`]: crate::registry::Registry

use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::num::NonZeroUsize;

use anyhow::Context;
use anyhow::Result;

use crate::commit::CommitId;
use crate::repo::DirectoryManager;
use crate::repo::RegistryRepo;
use crate::storage::in_memory::InMemoryRepo;

/// Drop epoch snapshots older than the `keep` most-recent, garbage-collecting
/// the database commits they no longer keep alive.
pub(super) fn prune(
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    recent_commits: &mut VecDeque<CommitId>,
    keep: NonZeroUsize,
) -> Result<()> {
    while recent_commits.len() > keep.get() {
        let old = recent_commits.pop_front().expect("non-empty");

        // Keep the snapshot if a retained epoch still references it
        if recent_commits.contains(&old) {
            continue;
        }

        prune_registry_commit(persistent_repo, in_memory_repo, &old, recent_commits)?;
    }
    Ok(())
}

/// Remove the registry commit `old` and any of its database commits not reachable
/// from a `retained` registry commit.
fn prune_registry_commit(
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    old: &CommitId,
    retained: &VecDeque<CommitId>,
) -> Result<()> {
    let mut reachable: HashSet<CommitId> = HashSet::new();
    for commit in retained {
        for db in crate::registry::database_commits(persistent_repo, commit)
            .context("reading a retained registry manifest")?
        {
            reachable.insert(db);
        }
    }

    let old_databases = crate::registry::database_commits(persistent_repo, old)
        .context("reading the evicted registry manifest")?;
    for db in old_databases {
        if reachable.contains(&db) {
            continue;
        }
        let dir = persistent_repo.database_commit_dir(&db);
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("removing database snapshot {}", dir.display()))?;
        }
        in_memory_repo
            .remove_commit(&db)
            .context("removing an in-memory database snapshot")?;
    }

    let manifest_file = persistent_repo.registry_commit_file(old);
    if manifest_file.exists() {
        fs::remove_file(&manifest_file)
            .with_context(|| format!("removing registry manifest {}", manifest_file.display()))?;
    }
    in_memory_repo
        .remove_registry_commit(old)
        .context("removing an in-memory registry manifest")?;

    Ok(())
}

#[cfg(test)]
mod tests {
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
