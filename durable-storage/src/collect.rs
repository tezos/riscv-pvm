// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Reclaiming the commits a repository no longer needs.
//!
//! Collecting at a root drops every registry commit recorded before it, together with the database
//! commits none of the surviving registry commits still reference. Database commits are shared -
//! they are content-addressed, so unrelated registry commits name the same one whenever their
//! states agree - which is why removal follows reachability rather than the order commits were
//! made in.
//!
//! This reclaims the value side of a commit. A database commit directory is a set of hard links,
//! so unlinking the last directory that references a file frees it, and values are keyed by user
//! key, so overwrites become obsolete versions that compaction discards. Merkle nodes are
//! content-addressed and every version of a node is a distinct live key, so no amount of directory
//! removal reclaims them; that needs deletion of the node keys themselves and is a separate
//! mechanism.
//!
//! # Interruption
//!
//! Collection is safe to interrupt and repeat. It runs in three steps, ordered so that everything
//! retained is intact at every point in between:
//!
//! 1. prune the journal to the retained roots,
//! 2. remove the manifests of registry commits that were dropped,
//! 3. remove the database commits no retained manifest reaches.
//!
//! Pruning first is what makes a repeat safe rather than merely possible. A target older than one
//! already collected at is no longer in the journal, so it is refused instead of being treated as
//! a floor that retains states whose data has already gone. Each later step enumerates what is
//! actually present rather than what the journal says should be, so work left unfinished by an
//! interruption is picked up by the next round.
//!
//! # Concurrency
//!
//! Collection must not run while the repository is being committed to, and two collections must
//! not overlap. Neither is enforced here.

use std::collections::HashSet;

use crate::commit::CommitId;
use crate::errors::OperationalError;
use crate::journal;
use crate::registry;
use crate::repo::RegistryRepo;

/// What a collection round reclaimed.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Collected {
    /// Registry commits whose manifests were removed.
    pub registry_commits: usize,

    /// Database commits that were removed.
    pub database_commits: usize,
}

/// Drop everything the repository no longer needs to serve `target` and the commits after it.
///
/// Retains every registry commit recorded at or after `target`, and every database commit one of
/// those still references. Fails with [`OperationalError::CollectionTargetNotRecorded`] if the
/// journal holds no entry for `target`, which is also how a target that an earlier round already
/// collected past is refused.
pub fn collect<Repo: RegistryRepo>(
    repo: &Repo,
    target: &CommitId,
) -> Result<Collected, OperationalError> {
    let retained_roots = journal::roots_to_retain(&repo.commit_journal()?, target)?;

    // Read before anything is removed. These manifests are all retained, so they would survive the
    // steps below either way, but reading them first keeps the reachable set independent of how
    // far a previous interrupted round got.
    let reachable = reachable_database_commits(repo, &retained_roots)?;

    repo.prune_journal(&retained_roots)?;

    let mut collected = Collected::default();

    for id in repo.registry_commits()? {
        if retained_roots.contains(&id) {
            continue;
        }

        repo.remove_registry_commit(&id)?;
        collected.registry_commits += 1;
    }

    for id in repo.database_commits()? {
        if reachable.contains(&id) {
            continue;
        }

        repo.remove_database_commit(&id)?;
        collected.database_commits += 1;
    }

    Ok(collected)
}

/// The database commits referenced by any of `roots`.
///
/// A root with no manifest is skipped rather than failing the round. That happens when a commit
/// was interrupted after its root was recorded but before its manifest was written, and when a
/// caller collects at a root it recorded itself but never committed; neither should stop the
/// commits that do have manifests from being collected.
fn reachable_database_commits<Repo: RegistryRepo>(
    repo: &Repo,
    roots: &HashSet<CommitId>,
) -> Result<HashSet<CommitId>, OperationalError> {
    let mut reachable = HashSet::new();

    for root in roots {
        match registry::database_commits(repo, root) {
            Ok(databases) => reachable.extend(databases),
            Err(OperationalError::CommitNotFound) => continue,
            Err(error) => return Err(error),
        }
    }

    Ok(reachable)
}

#[cfg(all(test, rocksdb_test_utils))]
mod tests {
    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::key::Key;
    use crate::persistence_layer::PersistenceLayer;
    use crate::registry::Registry;
    use crate::repo::DirectoryManager;

    /// A repository holding a registry of one database, committed once per call to `commit`.
    struct Fixture {
        _tmp: TestableTmpdir,
        repo: DirectoryManager,
        registry: Registry<PersistenceLayer, Normal>,
    }

    impl Fixture {
        fn new() -> Self {
            let tmp = TestableTmpdir::new();
            let repo = DirectoryManager::new(tmp.path())
                .expect("creating the directory manager should succeed");

            let mut registry = Registry::<PersistenceLayer, Normal>::new(repo.clone());
            registry
                .resize_tick(1)
                .expect("resizing the registry should succeed");

            Self {
                _tmp: tmp,
                repo,
                registry,
            }
        }

        /// Set `key` to `value` in the registry's first database and commit.
        fn commit(&mut self, key: &[u8], value: &[u8]) -> CommitId {
            self.commit_to(0, key, value)
        }

        /// Set `key` to `value` in database `index` and commit.
        fn commit_to(&mut self, index: usize, key: &[u8], value: &[u8]) -> CommitId {
            let key = Key::new(key).expect("the key should be valid");
            self.registry
                .database_mut(index)
                .expect("the database should exist")
                .set(key, Bytes::copy_from_slice(value))
                .expect("setting should succeed");

            self.registry.commit().expect("committing should succeed")
        }

        fn registry_commits(&self) -> HashSet<CommitId> {
            self.repo
                .registry_commits()
                .expect("listing registry commits should succeed")
                .into_iter()
                .collect()
        }

        fn database_commits(&self) -> HashSet<CommitId> {
            self.repo
                .database_commits()
                .expect("listing database commits should succeed")
                .into_iter()
                .collect()
        }
    }

    // Collecting at the newest root keeps it and drops the ones before it, and the retained root
    // still checks out - which needs its database commits to have survived.
    #[test]
    fn collecting_at_the_newest_root_drops_the_rest() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(b"a", b"1");
        let second = fixture.commit(b"a", b"2");
        let third = fixture.commit(b"a", b"3");

        let collected = collect(&fixture.repo, &third).expect("collection should succeed");

        assert_eq!(collected.registry_commits, 2);
        assert_eq!(fixture.registry_commits(), HashSet::from([third]));
        assert!(!fixture.registry_commits().contains(&first));
        assert!(!fixture.registry_commits().contains(&second));

        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), third)
            .expect("the retained root should still check out");
    }

    // A database commit shared with a retained registry commit is kept, even though the registry
    // commit that also referenced it was dropped. Committing the registry without touching the
    // database leaves the database commit unchanged, so both roots name it.
    #[test]
    fn a_shared_database_commit_survives() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(b"a", b"1");
        let shared: HashSet<CommitId> = registry::database_commits(&fixture.repo, &first)
            .expect("reading the manifest should succeed")
            .into_iter()
            .collect();

        // Write to a second database, so the registry root changes while the first database is
        // untouched and keeps the commit id the dropped root referenced.
        fixture
            .registry
            .resize_tick(2)
            .expect("resizing should succeed");
        let second = fixture.commit_to(1, b"b", b"2");

        assert!(
            registry::database_commits(&fixture.repo, &second)
                .expect("reading the manifest should succeed")
                .iter()
                .any(|id| shared.contains(id)),
            "the second root should still reference the first database's commit"
        );

        collect(&fixture.repo, &second).expect("collection should succeed");

        let present = fixture.database_commits();
        for id in &shared {
            assert!(
                present.contains(id),
                "a database commit a retained root references should survive"
            );
        }

        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), second)
            .expect("the retained root should still check out");
    }

    // Collecting at the oldest root is a no-op, since everything is at or after it.
    #[test]
    fn collecting_at_the_oldest_root_removes_nothing() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(b"a", b"1");
        fixture.commit(b"a", b"2");

        let before = (fixture.registry_commits(), fixture.database_commits());
        let collected = collect(&fixture.repo, &first).expect("collection should succeed");

        assert_eq!(collected, Collected::default());
        assert_eq!(
            (fixture.registry_commits(), fixture.database_commits()),
            before
        );
    }

    // Repeating a round changes nothing further: removals are idempotent and the second pass finds
    // nothing left to do.
    #[test]
    fn collecting_twice_at_the_same_root_is_idempotent() {
        let mut fixture = Fixture::new();

        fixture.commit(b"a", b"1");
        let second = fixture.commit(b"a", b"2");

        collect(&fixture.repo, &second).expect("the first round should succeed");
        let after_first = (fixture.registry_commits(), fixture.database_commits());

        let collected = collect(&fixture.repo, &second).expect("the second round should succeed");

        assert_eq!(collected, Collected::default());
        assert_eq!(
            (fixture.registry_commits(), fixture.database_commits()),
            after_first
        );
    }

    // A root already collected past is refused, rather than accepted as a floor that would retain
    // commits whose data has already been reclaimed.
    #[test]
    fn a_collected_target_is_refused_afterwards() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(b"a", b"1");
        let second = fixture.commit(b"a", b"2");

        collect(&fixture.repo, &second).expect("collection should succeed");

        assert!(matches!(
            collect(&fixture.repo, &first),
            Err(OperationalError::CollectionTargetNotRecorded { .. })
        ));
    }

    // Committing after a round continues to work, and the roots either side of it are retained
    // together.
    #[test]
    fn committing_after_a_round_still_works() {
        let mut fixture = Fixture::new();

        fixture.commit(b"a", b"1");
        let second = fixture.commit(b"a", b"2");

        collect(&fixture.repo, &second).expect("collection should succeed");

        let third = fixture.commit(b"a", b"3");

        assert_eq!(fixture.registry_commits(), HashSet::from([second, third]));

        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), third)
            .expect("the new root should check out");
        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), second)
            .expect("the root collected at should still check out");
    }
}
