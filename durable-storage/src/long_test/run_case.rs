// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Per-case execution: apply an operation sequence to the reference model and
//! three databases in lockstep, cross-checking results, traces, hashes, and
//! proofs.
//!
//! The three databases are:
//! - an in-memory [`TracedDatabase`]
//! - a persistent [`TracedDatabase`]
//! - a production [`Database<PersistenceLayer, Normal>`]
//!
//! Traces are compared between the two [`TracedDatabase`]s. Root hashes are
//! compared among all three. For every provable operation a proof is produced
//! from the persistent [`TracedDatabase`] and verified.

use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use tokio::runtime::Handle;

use crate::commit::CommitId;
use crate::database::Database;
use crate::database::TracedDatabase;
use crate::long_test::model::LongTestModel;
use crate::persistence_layer::PersistenceLayer;
use crate::repo::DirectoryManager;
use crate::storage::in_memory::InMemoryKeyValueStore;
use crate::storage::in_memory::InMemoryRepo;
use crate::test_helpers::database::DatabaseOperation;
use crate::test_helpers::database::DatabaseReferenceModel;
use crate::test_helpers::database::check_and_apply_value_operation;
use crate::test_helpers::prove_and_verify_operation;

/// The committed starting state shared by every case in an epoch.
#[derive(Clone)]
pub struct Base {
    /// The commit identifying the starting state (identical across backends).
    pub commit: CommitId,
    /// The reference model corresponding to `commit`.
    pub model: LongTestModel,
}

/// State carried while applying a sequence of operations to all targets.
struct Targets {
    in_memory_db: TracedDatabase<InMemoryKeyValueStore, Normal>,
    persistent_db: TracedDatabase<PersistenceLayer, Normal>,
    production_db: Database<PersistenceLayer, Normal>,
    model: LongTestModel,
}

/// Check out the shared `base` into a fresh set of targets.
fn checkout_targets(
    handle: &Handle,
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    base: &Base,
) -> Targets {
    let in_memory_db = TracedDatabase::<InMemoryKeyValueStore, Normal>::checkout(
        handle,
        in_memory_repo,
        base.commit,
    )
    .expect("checking out the in-memory base should succeed");
    let persistent_db =
        TracedDatabase::<PersistenceLayer, Normal>::checkout(handle, persistent_repo, base.commit)
            .expect("checking out the persistence base should succeed");
    let production_db =
        Database::<PersistenceLayer, Normal>::checkout(handle, persistent_repo, base.commit)
            .expect("checking out the production base should succeed");

    Targets {
        in_memory_db,
        persistent_db,
        production_db,
        model: base.model.clone(),
    }
}

/// Apply `ops` to all targets in lockstep, optionally producing and verifying a
/// proof for every provable operation.
fn apply_sequence(targets: &mut Targets, ops: &[DatabaseOperation], prove: bool) {
    for op in ops {
        // Proofs are taken over the pre-operation state, so prove first.
        if prove {
            prove_and_verify_operation(&targets.persistent_db, op);
        }

        check_and_apply_value_operation(&mut targets.in_memory_db, &targets.model, op);
        check_and_apply_value_operation(&mut targets.persistent_db, &targets.model, op);
        check_and_apply_value_operation(&mut targets.production_db, &targets.model, op);
        targets.model.apply(op);
    }
}

/// Assert the two traced databases recorded identical traces and all three
/// databases agree on the root hash.
fn check_consistency(targets: Targets) {
    let (in_memory_db, in_memory_trace) = targets.in_memory_db.into_parts();
    let (persistent_db, persistent_trace) = targets.persistent_db.into_parts();

    assert_eq!(
        in_memory_trace, persistent_trace,
        "trace mismatch between in-memory and persistence backends"
    );

    let in_memory_hash: Hash = in_memory_db
        .hash()
        .expect("hashing the in-memory database should succeed");
    let persist_hash: Hash = persistent_db
        .hash()
        .expect("hashing the persistence database should succeed");
    let production_hash: Hash = targets
        .production_db
        .hash()
        .expect("hashing the production database should succeed");

    assert_eq!(
        in_memory_hash, persist_hash,
        "root hash mismatch (in-memory vs persist)"
    );
    assert_eq!(
        persist_hash, production_hash,
        "root hash mismatch (persist vs production)"
    );
}

/// Run a single property-test case: check out the epoch base, apply `ops` with
/// proof generation, and cross-check traces and hashes. Panics on any mismatch
/// (caught by the proptest runner and shrunk).
pub fn run_case(
    handle: &Handle,
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    base: &Base,
    ops: &[DatabaseOperation],
) {
    let mut targets = checkout_targets(handle, in_memory_repo, persistent_repo, base);
    apply_sequence(&mut targets, ops, true);
    check_consistency(targets);
}

/// Advance the shared base by applying `ops` (without proofs) and committing the
/// result on every backend, returning the new committed [`Base`].
pub fn advance_base(
    handle: &Handle,
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    base: &Base,
    ops: &[DatabaseOperation],
) -> Base {
    let mut targets = checkout_targets(handle, in_memory_repo, persistent_repo, base);
    apply_sequence(&mut targets, ops, false);

    let in_memory_commit = targets
        .in_memory_db
        .commit(in_memory_repo)
        .expect("committing the in-memory base");
    let persistent_commit = targets
        .persistent_db
        .commit(persistent_repo)
        .expect("committing the persistence base");
    let production_commit = targets
        .production_db
        .commit(persistent_repo)
        .expect("committing the production base");
    assert_eq!(
        in_memory_commit, persistent_commit,
        "base commit id mismatch (in-memory vs persist)"
    );
    assert_eq!(
        persistent_commit, production_commit,
        "base commit id mismatch (persist vs production)"
    );

    Base {
        commit: persistent_commit,
        model: targets.model,
    }
}

/// Commit an empty database on every backend to obtain the initial [`Base`].
pub fn initial_base(
    handle: &Handle,
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
) -> Base {
    let in_memory_db = Database::<InMemoryKeyValueStore, Normal>::try_new(handle, in_memory_repo)
        .expect("creating the in-memory database should succeed");
    let persistent_db = Database::<PersistenceLayer, Normal>::try_new(handle, persistent_repo)
        .expect("creating the persistence database should succeed");

    let in_memory_commit = in_memory_db
        .commit(in_memory_repo)
        .expect("committing the empty in-memory database");
    let persistent_commit = persistent_db
        .commit(persistent_repo)
        .expect("committing the empty persistence database");
    assert_eq!(
        in_memory_commit, persistent_commit,
        "empty commit id mismatch across backends"
    );

    Base {
        commit: persistent_commit,
        model: LongTestModel::default(),
    }
}
