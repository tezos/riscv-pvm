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

// The sweep works on the repository-wide Merkle store, which only exists with a rocksdb backend.
#[cfg(rocksdb)]
pub mod sweep;

use std::collections::HashSet;

#[cfg(rocksdb)]
pub use self::sweep::SweptNodes;
use crate::commit::CommitId;
use crate::errors::OperationalError;
use crate::journal;
#[cfg(rocksdb)]
use crate::journal::Seq;
use crate::registry;
#[cfg(rocksdb)]
use crate::repo::DirectoryManager;
use crate::repo::RegistryRepo;

/// A request for a collection in progress to stop early.
///
/// Collection is the longest thing the storage does, and a full commit or a reap should not have to
/// wait for one to finish. Both halves check this often enough to stop promptly, and stopping is
/// safe at any point: each step leaves everything retained intact, and enumerates what is present
/// rather than what it expected, so the next round finishes what this one left.
///
/// Cloning shares the request, so the handle given to a collection and the one kept by whoever may
/// need to interrupt it are the same signal.
#[derive(Debug, Default, Clone)]
pub struct Suspend(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Suspend {
    /// A request that has not been made.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the collection using this to stop as soon as it can.
    pub fn request(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether stopping has been asked for.
    pub fn requested(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// What a collection round reclaimed.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Collected {
    /// Registry commits whose manifests were removed.
    pub registry_commits: usize,

    /// Database commits that were removed.
    pub database_commits: usize,

    /// Whether the round stopped early because it was asked to.
    ///
    /// What it did remove is still removed; the rest is left for the next round.
    pub suspended: bool,
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
    suspend: &Suspend,
) -> Result<Collected, OperationalError> {
    let retained_roots = journal::roots_to_retain(&repo.commit_journal()?, target)?;

    // Read before anything is removed. These manifests are all retained, so they would survive the
    // steps below either way, but reading them first keeps the reachable set independent of how
    // far a previous interrupted round got.
    let reachable = reachable_database_commits(repo, &retained_roots)?;

    repo.prune_journal(&retained_roots)?;

    let mut collected = Collected::default();

    for id in repo.registry_commits()? {
        if suspend.requested() {
            collected.suspended = true;
            return Ok(collected);
        }

        if retained_roots.contains(&id) {
            continue;
        }

        repo.remove_registry_commit(&id)?;
        collected.registry_commits += 1;
    }

    for id in repo.database_commits()? {
        if suspend.requested() {
            collected.suspended = true;
            return Ok(collected);
        }

        if reachable.contains(&id) {
            continue;
        }

        repo.remove_database_commit(&id)?;
        collected.database_commits += 1;
    }

    Ok(collected)
}

/// Collect both halves of a repository at `target`.
///
/// The commits, as [`collect`] does, and then the Merkle nodes none of the surviving commits still
/// reaches - which removing commit directories can never do, because node bodies are
/// content-addressed and live in a store shared by the repository rather than in any commit.
///
/// The node sweep runs second because it decides liveness from what the commits still hold, so it
/// must see the set of commits the round settled on rather than the one it started from.
#[cfg(rocksdb)]
pub fn collect_all(
    repo: &DirectoryManager,
    target: &CommitId,
    suspend: &Suspend,
) -> Result<(Collected, SweptNodes), OperationalError> {
    let collected = collect(repo, target, suspend)?;

    // A suspended first half leaves commits the second half would treat as retained, so the sweep
    // would keep nodes that are on their way out. Correct, but wasted work: the next round does
    // both halves against a settled set.
    if collected.suspended {
        return Ok((collected, SweptNodes::default()));
    }

    let swept = collect_nodes(repo, target, suspend)?;

    Ok((collected, swept))
}

/// Delete the Merkle nodes no retained commit of `repo` still reaches.
///
/// Reads the retained roots from the journal, so a round that has already pruned it sees exactly
/// the commits that survived.
#[cfg(rocksdb)]
pub fn collect_nodes(
    repo: &DirectoryManager,
    target: &CommitId,
    suspend: &Suspend,
) -> Result<SweptNodes, OperationalError> {
    // Refused up front rather than when the first deletion is attempted, so that collecting through
    // a handle that cannot write says so whether or not this round happens to have work to do.
    if repo.is_read_only() {
        return Err(OperationalError::RepositoryIsReadOnly);
    }

    let entries = repo.commit_journal()?;
    let positions = journal::latest_positions(&entries);

    let floor =
        *positions
            .get(target)
            .ok_or_else(|| OperationalError::CollectionTargetNotRecorded {
                target: target.hex_encode(),
            })?;

    // A database commit is reached by every registry commit naming it, so it is held until the most
    // recent of them goes. Taking the highest is what stops an older reference from deciding it.
    let mut roots = sweep::RetainedRoots::new();

    for (root, seq) in positions {
        if seq < floor {
            continue;
        }

        let databases = match registry::database_commits(repo, &root) {
            Ok(databases) => databases,
            Err(OperationalError::CommitNotFound) => continue,
            Err(error) => return Err(error),
        };

        for database in databases {
            roots
                .entry(*database.as_hash())
                .and_modify(|held: &mut Seq| *held = (*held).max(seq))
                .or_insert(seq);
        }
    }

    sweep::sweep(repo.merkle_store(), &roots, floor, suspend)
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

        let collected =
            collect(&fixture.repo, &third, &Suspend::new()).expect("collection should succeed");

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

        collect(&fixture.repo, &second, &Suspend::new()).expect("collection should succeed");

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

    // A round asked to stop before it starts removes nothing and says so, leaving the repository
    // exactly as it was for the next round to finish.
    #[test]
    fn a_suspended_round_removes_nothing_and_reports_it() {
        let mut fixture = Fixture::new();

        fixture.commit(b"a", b"1");
        let second = fixture.commit(b"a", b"2");

        let before = (fixture.registry_commits(), fixture.database_commits());

        let suspend = Suspend::new();
        suspend.request();

        let collected =
            collect(&fixture.repo, &second, &suspend).expect("collection should succeed");

        assert!(collected.suspended);
        assert_eq!(collected.registry_commits, 0);
        assert_eq!(collected.database_commits, 0);
        assert_eq!(
            (fixture.registry_commits(), fixture.database_commits()),
            before
        );
    }

    // What a suspended round left is picked up by the next one, so suspending costs progress but
    // never correctness.
    #[test]
    fn a_later_round_finishes_what_a_suspended_one_left() {
        let mut fixture = Fixture::new();

        fixture.commit(b"a", b"1");
        let second = fixture.commit(b"a", b"2");

        let suspend = Suspend::new();
        suspend.request();
        collect(&fixture.repo, &second, &suspend).expect("the suspended round should succeed");

        let collected = collect(&fixture.repo, &second, &Suspend::new())
            .expect("the later round should succeed");

        assert!(!collected.suspended);
        assert_eq!(collected.registry_commits, 1);
        assert_eq!(fixture.registry_commits(), HashSet::from([second]));
    }

    // Collecting at the oldest root is a no-op, since everything is at or after it.
    #[test]
    fn collecting_at_the_oldest_root_removes_nothing() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(b"a", b"1");
        fixture.commit(b"a", b"2");

        let before = (fixture.registry_commits(), fixture.database_commits());
        let collected =
            collect(&fixture.repo, &first, &Suspend::new()).expect("collection should succeed");

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

        collect(&fixture.repo, &second, &Suspend::new()).expect("the first round should succeed");
        let after_first = (fixture.registry_commits(), fixture.database_commits());

        let collected = collect(&fixture.repo, &second, &Suspend::new())
            .expect("the second round should succeed");

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

        collect(&fixture.repo, &second, &Suspend::new()).expect("collection should succeed");

        assert!(matches!(
            collect(&fixture.repo, &first, &Suspend::new()),
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

        collect(&fixture.repo, &second, &Suspend::new()).expect("collection should succeed");

        let third = fixture.commit(b"a", b"3");

        assert_eq!(fixture.registry_commits(), HashSet::from([second, third]));

        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), third)
            .expect("the new root should check out");
        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), second)
            .expect("the root collected at should still check out");
    }
}

#[cfg(all(test, rocksdb_test_utils))]
mod node_tests {
    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::key::Key;
    use crate::persistence_layer::PersistenceLayer;
    use crate::registry::Registry;

    /// A registry of one database over several commits, with the nodes each commit left behind.
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

        /// Write several keys and commit, returning the registry root.
        fn commit(&mut self, keys: &[&[u8]], value: &[u8]) -> CommitId {
            for key in keys {
                let key = Key::new(key).expect("the key should be valid");
                self.registry
                    .database_mut(0)
                    .expect("the database should exist")
                    .set(key, Bytes::copy_from_slice(value))
                    .expect("setting should succeed");
            }

            self.registry.commit().expect("committing should succeed")
        }

        /// How many node bodies the repository's Merkle store holds.
        fn nodes(&self) -> usize {
            let mut nodes = 0;
            self.repo
                .merkle_store()
                .for_each_node(|_, _| nodes += 1)
                .expect("counting nodes should succeed");
            nodes
        }

        /// How many reverse edges it holds.
        fn edges(&self) -> u64 {
            self.repo
                .merkle_store()
                .edge_totals()
                .expect("counting edges should succeed")
                .entries
        }
    }

    // The nodes of dropped commits go, the retained root still checks out with every key readable,
    // and the edges of the deleted nodes go with them.
    #[test]
    fn collects_the_nodes_of_dropped_commits() {
        let mut fixture = Fixture::new();

        fixture.commit(&[b"a", b"b", b"c"], b"1");
        fixture.commit(&[b"a", b"b", b"c"], b"2");
        let third = fixture.commit(&[b"a", b"b", b"c"], b"3");

        let before = fixture.nodes();

        let (_, swept) =
            collect_all(&fixture.repo, &third, &Suspend::new()).expect("collection should succeed");

        assert!(swept.nodes > 0, "the earlier commits left nodes behind");
        assert!(swept.edges > 0, "their edges should go with them");
        assert_eq!(
            fixture.nodes(),
            before - swept.nodes,
            "exactly the swept nodes should be gone"
        );

        // The retained root has to be readable in full: every node it reaches must have survived.
        let restored = Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), third)
            .expect("the retained root should check out");
        for key in [b"a", b"b", b"c"] {
            let key = Key::new(key).expect("the key should be valid");
            assert_eq!(
                restored
                    .database(0)
                    .expect("the database should exist")
                    .read_bytes(&key, 0, 32)
                    .expect("the key should still be readable")
                    .as_ref(),
                b"3"
            );
        }
    }

    // A node a collection took reads back as collected, rather than as an inconsistent store.
    // The distinction is what lets a caller recover by checking out again instead of treating the
    // repository as broken.
    #[test]
    fn a_collected_node_reads_back_as_collected() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(&[b"a", b"b", b"c"], b"1");
        let second = fixture.commit(&[b"a", b"b", b"c"], b"2");

        // The tree root of the first commit's database, which the second commit supersedes.
        let superseded = *registry::database_commits(&fixture.repo, &first)
            .expect("reading the manifest should succeed")
            .first()
            .expect("the registry has a database")
            .as_hash();

        assert!(
            fixture.repo.merkle_store().get(superseded.as_ref()).is_ok(),
            "the node should be there before collecting"
        );

        let (_, swept) = collect_all(&fixture.repo, &second, &Suspend::new())
            .expect("collection should succeed");
        assert!(swept.nodes > 0, "the first commit should have left nodes");

        assert!(
            fixture.repo.merkle_store().has_collected(),
            "the store should remember that it collected"
        );

        let error = match fixture.repo.merkle_store().get(superseded.as_ref()) {
            Ok(_) => panic!("the superseded root should have been collected"),
            Err(error) => error,
        };

        assert!(
            matches!(
                error,
                crate::errors::Error::Operational(OperationalError::NodeCollected { .. })
            ),
            "expected a collected-node error, got {error:?}"
        );
    }

    // A collection gives the store a new identity, so no node still in memory can claim to be
    // stored here on the strength of a write from before the round. Without that, a working tree
    // whose lineage the round dropped would skip writing nodes the round had just deleted, and
    // commit something referring to bodies that are gone - silently, unlike a read, which errors.
    #[test]
    fn collecting_makes_stored_nodes_prove_themselves_again() {
        let mut fixture = Fixture::new();

        fixture.commit(&[b"a", b"b"], b"1");
        let second = fixture.commit(&[b"a", b"b"], b"2");

        let before = fixture.repo.merkle_store().store_id();

        let (_, swept) = collect_all(&fixture.repo, &second, &Suspend::new())
            .expect("collection should succeed");
        assert!(swept.nodes > 0, "the round should have deleted something");

        assert_ne!(
            fixture.repo.merkle_store().store_id(),
            before,
            "a round that deleted nodes should leave a store nothing can claim to be in"
        );
    }

    // A round that deletes nothing leaves the identity alone, so an ordinary collection over a
    // settled store does not cost the next commit a full rewrite.
    #[test]
    fn a_round_that_deletes_nothing_keeps_the_identity() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(&[b"a"], b"1");
        fixture.commit(&[b"a"], b"2");

        let before = fixture.repo.merkle_store().store_id();

        let (_, swept) =
            collect_all(&fixture.repo, &first, &Suspend::new()).expect("collection should succeed");
        assert_eq!(
            swept.nodes, 0,
            "collecting at the oldest root drops nothing"
        );

        assert_eq!(fixture.repo.merkle_store().store_id(), before);
    }

    // A full commit does not wait on reclaiming, and reclaiming runs alongside ordinary use: the
    // registry stays committable while the store is being rewritten underneath it.
    #[test]
    fn reclaiming_does_not_hold_up_the_registry() {
        let mut fixture = Fixture::new();

        let base: Vec<Vec<u8>> = (0..200u32).map(|i| i.to_be_bytes().to_vec()).collect();
        let base_refs: Vec<&[u8]> = base.iter().map(|k| k.as_slice()).collect();
        fixture.commit(&base_refs, b"0");
        let second = fixture.commit(&[b"\x00\x00\x00\x01"], b"1");

        collect_all(&fixture.repo, &second, &Suspend::new()).expect("collection should succeed");

        assert!(
            fixture.repo.start_reclaim(),
            "a reclaim should have started"
        );
        assert!(
            !fixture.repo.start_reclaim(),
            "a second request should not queue another rewrite"
        );

        // Taking a full commit and committing again both proceed while it runs.
        fixture
            .repo
            .full_commit()
            .expect("a full commit should not wait on reclaiming");

        let third = fixture.commit(&[b"\x00\x00\x00\x02"], b"2");

        // Wait for it, so the test does not leave a thread writing into a temporary directory.
        while fixture.repo.is_reclaiming() {
            std::thread::yield_now();
        }

        Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), third)
            .expect("the commit made during the reclaim should check out");
    }

    // A store that has never collected reports an absent node as simply absent, so the distinction
    // means something.
    #[test]
    fn an_absent_node_is_not_blamed_on_collection() {
        let fixture = Fixture::new();

        assert!(!fixture.repo.merkle_store().has_collected());

        let error = match fixture.repo.merkle_store().get(&[7u8; 32]) {
            Ok(_) => panic!("nothing was ever written under that hash"),
            Err(error) => error,
        };

        assert!(
            !matches!(
                error,
                crate::errors::Error::Operational(OperationalError::NodeCollected { .. })
            ),
            "a store that never collected should not blame collection, got {error:?}"
        );
    }

    // In steady state a round looks at the churn since the last one, not at the store. This is the
    // property that keeps collection affordable at scale, and it depends on collecting behind the
    // tip: a node settled by one round is relisted under the newest retained root, so it stays out
    // of range until the floor passes that root.
    //
    // The first round after a backlog is different, and necessarily so: every node written since
    // the last round is one whose commit is being dropped, so all of them are in question. What that
    // round settles is what later rounds no longer pay for.
    #[test]
    fn a_settled_store_costs_only_its_churn() {
        let mut fixture = Fixture::new();

        let base: Vec<Vec<u8>> = (0..400u32).map(|i| i.to_be_bytes().to_vec()).collect();
        let base_refs: Vec<&[u8]> = base.iter().map(|k| k.as_slice()).collect();
        fixture.commit(&base_refs, b"0");

        // A run of commits, so that collection can sit behind the tip the way a rollup node does.
        let mut commits = Vec::new();
        for value in [b"1", b"2", b"3", b"4", b"5", b"6"] {
            commits.push(fixture.commit(&[b"\x00\x00\x00\x01"], value));
        }

        // Catching up: settles the base, which later rounds then do not pay for.
        let catch_up = collect_nodes(&fixture.repo, &commits[1], &Suspend::new())
            .expect("the first round should succeed");
        assert!(
            catch_up.examined >= 400,
            "the first round should have had to consider the base"
        );

        let store_nodes = fixture.nodes();

        // A little churn, then a round whose floor is still behind the tip.
        fixture.commit(&[b"\x00\x00\x00\x02"], b"7");
        fixture.commit(&[b"\x00\x00\x00\x02"], b"8");

        let swept = collect_nodes(&fixture.repo, &commits[3], &Suspend::new())
            .expect("the second round should succeed");

        assert!(
            swept.examined < store_nodes / 4,
            "a round examined {} of {} nodes; it should cost the churn, not the store",
            swept.examined,
            store_nodes,
        );
    }

    // A node proved live is not looked at again by a later round at the same floor, because it is
    // relisted under the root that holds it.
    #[test]
    fn a_live_node_is_not_re_examined() {
        let mut fixture = Fixture::new();

        let base: Vec<Vec<u8>> = (0..200u32).map(|i| i.to_be_bytes().to_vec()).collect();
        let base_refs: Vec<&[u8]> = base.iter().map(|k| k.as_slice()).collect();
        fixture.commit(&base_refs, b"0");
        fixture.commit(&[b"\x00\x00\x00\x01"], b"1");
        let last = fixture.commit(&[b"\x00\x00\x00\x01"], b"2");

        let first_round = collect_nodes(&fixture.repo, &last, &Suspend::new())
            .expect("the first round should succeed");
        assert!(first_round.examined > 0);

        let second_round = collect_nodes(&fixture.repo, &last, &Suspend::new())
            .expect("the second round should succeed");

        assert_eq!(
            second_round.examined, 0,
            "everything settled by the first round should be out of the second round's range"
        );
        assert_eq!(second_round.nodes, 0);
    }

    // Collecting at the oldest root keeps every node, since every commit is retained.
    #[test]
    fn collecting_at_the_oldest_root_keeps_every_node() {
        let mut fixture = Fixture::new();

        let first = fixture.commit(&[b"a"], b"1");
        fixture.commit(&[b"a"], b"2");

        let before = (fixture.nodes(), fixture.edges());
        let (_, swept) =
            collect_all(&fixture.repo, &first, &Suspend::new()).expect("collection should succeed");

        assert_eq!(swept, SweptNodes::default());
        assert_eq!((fixture.nodes(), fixture.edges()), before);
    }

    // A second round finds nothing left to do, and leaves the store as the first did.
    #[test]
    fn a_second_sweep_finds_nothing() {
        let mut fixture = Fixture::new();

        fixture.commit(&[b"a", b"b"], b"1");
        let second = fixture.commit(&[b"a", b"b"], b"2");

        collect_all(&fixture.repo, &second, &Suspend::new())
            .expect("the first round should succeed");
        let after_first = (fixture.nodes(), fixture.edges());

        let swept = collect_nodes(&fixture.repo, &second, &Suspend::new())
            .expect("the second round should succeed");

        assert_eq!(swept, SweptNodes::default());
        assert_eq!((fixture.nodes(), fixture.edges()), after_first);
    }

    // Committing after a collection still works, and collecting again keeps the newer root whole.
    #[test]
    fn committing_after_a_sweep_still_works() {
        let mut fixture = Fixture::new();

        fixture.commit(&[b"a"], b"1");
        let second = fixture.commit(&[b"a"], b"2");

        collect_all(&fixture.repo, &second, &Suspend::new()).expect("collection should succeed");

        let third = fixture.commit(&[b"a"], b"3");
        collect_all(&fixture.repo, &third, &Suspend::new())
            .expect("the second collection should succeed");

        let restored = Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), third)
            .expect("the newest root should check out");
        let key = Key::new(b"a").expect("the key should be valid");
        assert_eq!(
            restored
                .database(0)
                .expect("the database should exist")
                .read_bytes(&key, 0, 32)
                .expect("the key should be readable")
                .as_ref(),
            b"3"
        );
    }

    // Nodes shared between a dropped and a retained commit survive: an unchanged subtree keeps the
    // same hash, so the retained root still reaches it.
    #[test]
    fn nodes_shared_with_a_retained_commit_survive() {
        let mut fixture = Fixture::new();

        // The second commit touches only one of the three keys, so most of the tree is shared.
        fixture.commit(&[b"a", b"b", b"c"], b"1");
        let second = fixture.commit(&[b"a"], b"2");

        collect_all(&fixture.repo, &second, &Suspend::new()).expect("collection should succeed");

        let restored = Registry::<PersistenceLayer, Normal>::checkout(fixture.repo.clone(), second)
            .expect("the retained root should check out");

        for (key, expected) in [
            (b"a".as_slice(), b"2".as_slice()),
            (b"b", b"1"),
            (b"c", b"1"),
        ] {
            let key = Key::new(key).expect("the key should be valid");
            assert_eq!(
                restored
                    .database(0)
                    .expect("the database should exist")
                    .read_bytes(&key, 0, 32)
                    .expect("the shared subtree should have survived")
                    .as_ref(),
                expected
            );
        }
    }
}

#[cfg(all(test, rocksdb_test_utils))]
mod full_commit_tests {
    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::key::Key;
    use crate::merkle_store::slots;
    use crate::persistence_layer::PersistenceLayer;
    use crate::registry::Registry;

    // A repository whose Merkle store is lost comes back from its last full commit, with the
    // committed state still checkoutable and readable.
    #[test]
    fn a_lost_store_recovers_from_its_last_full_commit() {
        let tmp = TestableTmpdir::new();
        let key = Key::new(b"a").expect("the key should be valid");

        let (commit, slot) = {
            let repo = DirectoryManager::new(tmp.path())
                .expect("creating the directory manager should succeed");

            let mut registry = Registry::<PersistenceLayer, Normal>::new(repo.clone());
            registry.resize_tick(1).expect("resizing should succeed");
            registry
                .database_mut(0)
                .expect("the database should exist")
                .set(key.clone(), Bytes::from_static(b"value"))
                .expect("setting should succeed");

            let commit = registry.commit().expect("committing should succeed");

            // The full commit has to come after the nodes are written, since it images them.
            let slot = repo.full_commit().expect("the full commit should succeed");
            assert_eq!(
                repo.latest_full_commit().expect("reading should succeed"),
                Some(slot)
            );

            (commit, slot)
        };

        // Lose the store, keeping the commit directories and the slot.
        let store_dir = DirectoryManager::merkle_dir_in(tmp.path());
        std::fs::remove_dir_all(&store_dir).expect("removing the store should succeed");

        let slots_dir = DirectoryManager::merkle_slots_dir_in(&store_dir);
        slots::restore_from_slot(&slots::slot_path(&slots_dir, slot), &store_dir)
            .expect("restoring should succeed");

        let repo =
            DirectoryManager::new(tmp.path()).expect("reopening the repository should succeed");
        let restored = Registry::<PersistenceLayer, Normal>::checkout(repo, commit)
            .expect("the commit should check out against the recovered store");

        assert_eq!(
            restored
                .database(0)
                .expect("the database should exist")
                .read_bytes(&key, 0, 32)
                .expect("the value should be readable")
                .as_ref(),
            b"value"
        );
    }

    // Reaping frees what an older slot was holding, while the newest still recovers.
    #[test]
    fn reaping_leaves_the_newest_recoverable() {
        let tmp = TestableTmpdir::new();
        let repo = DirectoryManager::new(tmp.path())
            .expect("creating the directory manager should succeed");

        let mut registry = Registry::<PersistenceLayer, Normal>::new(repo.clone());
        registry.resize_tick(1).expect("resizing should succeed");

        for value in [b"1", b"2", b"3"] {
            let key = Key::new(b"a").expect("the key should be valid");
            registry
                .database_mut(0)
                .expect("the database should exist")
                .set(key, Bytes::copy_from_slice(value))
                .expect("setting should succeed");
            registry.commit().expect("committing should succeed");
            repo.full_commit().expect("the full commit should succeed");
        }

        assert_eq!(
            repo.reap_full_commits(1).expect("reaping should succeed"),
            2
        );
        assert_eq!(
            repo.latest_full_commit().expect("reading should succeed"),
            Some(3),
            "the newest full commit should be the one left"
        );
    }
}
