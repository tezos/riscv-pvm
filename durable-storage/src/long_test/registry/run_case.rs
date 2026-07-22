// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Per-case execution for the [`Registry`] long test: apply an operation
//! sequence to the reference model and two registries (in-memory and
//! persistence-backed) in lockstep, cross-checking operation results against
//! the model and, at the end of each case, the registry root hash between the
//! two backends.
//!
//! For every provable operation a proof is produced from the persistence-backed
//! registry and verified.
//!
//! [`Registry`]: crate::registry::Registry

use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;

use super::model::RegistryLongTestModel;
use crate::long_test::harness::Base;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::persistence_layer::PersistenceLayer;
use crate::registry::Registry;
use crate::repo::DirectoryManager;
use crate::storage::in_memory::InMemoryKeyValueStore;
use crate::storage::in_memory::InMemoryRepo;
use crate::test_helpers::database::check_and_apply_value_operation;
use crate::test_helpers::proof_size::assert_proof_size;
use crate::test_helpers::proof_size::registry_operation_proof_size_bound;
use crate::test_helpers::registry::RegistryOperation;
use crate::test_helpers::registry::grow_registry;
use crate::test_helpers::registry::prove_and_verify_registry_operation;

/// State carried while applying a sequence of operations to both backends.
struct Targets {
    in_memory: Registry<InMemoryKeyValueStore, Normal>,
    persistent: Registry<PersistenceLayer, Normal>,
    model: RegistryLongTestModel,
}

/// Apply a single operation to `registry`, checking database value operations
/// against the (read-only) `model`. Only `ShrinkRegistry` checks the model's
/// `permanent` floor here; the other registry operations rely on the strategy
/// having generated valid indices.
fn apply_op_checked<KV>(
    registry: &mut Registry<KV, Normal>,
    model: &RegistryLongTestModel,
    op: &RegistryOperation,
) where
    KV: BackgroundKeyValueStore,
{
    match op {
        RegistryOperation::Database(index, db_op) => {
            let database = registry
                .database_mut(*index)
                .expect("the database index should be in bounds");
            check_and_apply_value_operation(database, &model.databases[*index], db_op);
        }
        RegistryOperation::GrowRegistry => grow_registry(registry),
        RegistryOperation::ShrinkRegistry => {
            if registry.len() > model.permanent() {
                registry
                    .resize_tick(registry.len() - 1)
                    .expect("shrinking the registry should succeed");
            }
        }
        RegistryOperation::CopyDatabase(src, dst) => {
            registry
                .copy_database(*src, *dst)
                .expect("copying the database should succeed");
        }
        RegistryOperation::MoveDatabase(src, dst) => {
            registry
                .move_database(*src, *dst)
                .expect("moving the database should succeed");
        }
        RegistryOperation::ClearDatabase(index) => {
            registry
                .clear_database(*index)
                .expect("clearing the database should succeed");
        }
        RegistryOperation::CommitCheckoutRoundtrip => {
            unreachable!("commit/checkout operations should not be generated")
        }
    }
}

/// Check out the shared `base` into a fresh set of targets.
fn checkout_targets(
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    base: &Base<RegistryLongTestModel>,
) -> Targets {
    let in_memory =
        Registry::<InMemoryKeyValueStore, Normal>::checkout(in_memory_repo.clone(), base.commit)
            .expect("checking out the in-memory base should succeed");
    let persistent =
        Registry::<PersistenceLayer, Normal>::checkout(persistent_repo.clone(), base.commit)
            .expect("checking out the persistence base should succeed");

    Targets {
        in_memory,
        persistent,
        model: base.model.clone(),
    }
}

/// Apply `ops` to both registries and the model in lockstep, optionally
/// producing and verifying a proof for every provable operation.
/// `fail_on_warning` escalates proof size warnings into failures.
fn apply_sequence(
    targets: &mut Targets,
    ops: &[RegistryOperation],
    prove: bool,
    fail_on_warning: bool,
) {
    for op in ops {
        // Proofs are taken over the pre-operation state, so prove first. The
        // size bound is likewise computed over the pre-operation model.
        if prove {
            let bound = registry_operation_proof_size_bound(&targets.model.databases, op);
            let proof = prove_and_verify_registry_operation(&targets.persistent, op);
            if let Some(proof) = proof {
                let bound = bound.expect("provable operations have a size bound");
                assert_proof_size(op, proof.len(), bound, fail_on_warning);
            }
        }

        apply_op_checked(&mut targets.in_memory, &targets.model, op);
        apply_op_checked(&mut targets.persistent, &targets.model, op);
        targets.model.apply(op);
    }
}

/// Run a single property-test case: check out the epoch base, apply `ops` with
/// proof generation, and cross-check root hashes.
pub fn run_case(
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    base: &Base<RegistryLongTestModel>,
    ops: &[RegistryOperation],
    fail_on_warning: bool,
) {
    let mut targets = checkout_targets(in_memory_repo, persistent_repo, base);
    apply_sequence(&mut targets, ops, true, fail_on_warning);

    let in_memory_root = Hash::from_foldable(&targets.in_memory);
    let persistent_root = Hash::from_foldable(&targets.persistent);
    assert_eq!(
        in_memory_root, persistent_root,
        "registry root hash mismatch (in-memory vs persist)"
    )
}

/// Advance the shared base by applying `ops` (without proofs) and committing the
/// result on both backends, returning the new committed [`Base`].
pub fn advance_base(
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    base: &Base<RegistryLongTestModel>,
    ops: &[RegistryOperation],
) -> Base<RegistryLongTestModel> {
    let mut targets = checkout_targets(in_memory_repo, persistent_repo, base);
    apply_sequence(&mut targets, ops, false, false);

    let in_memory_commit = targets
        .in_memory
        .commit()
        .expect("committing the in-memory base");
    let persistent_commit = targets
        .persistent
        .commit()
        .expect("committing the persistence base");
    assert_eq!(
        in_memory_commit, persistent_commit,
        "base commit id mismatch (in-memory vs persist)"
    );

    Base {
        commit: persistent_commit,
        model: targets.model,
    }
}

/// Commit a registry of `permanent` empty databases on both backends to obtain
/// the initial [`Base`].
pub fn initial_base(
    in_memory_repo: &InMemoryRepo,
    persistent_repo: &DirectoryManager,
    permanent: usize,
) -> Base<RegistryLongTestModel> {
    let mut in_memory = Registry::<InMemoryKeyValueStore, Normal>::new(in_memory_repo.clone())
        .expect("creating the in-memory registry should succeed");
    let mut persistent = Registry::<PersistenceLayer, Normal>::new(persistent_repo.clone())
        .expect("creating the persistence registry should succeed");

    for size in 1..=permanent {
        in_memory
            .resize_tick(size)
            .expect("growing the in-memory registry should succeed");
        persistent
            .resize_tick(size)
            .expect("growing the persistence registry should succeed");
    }

    let in_memory_commit = in_memory
        .commit()
        .expect("committing the empty in-memory registry");
    let persistent_commit = persistent
        .commit()
        .expect("committing the empty persistence registry");
    assert_eq!(
        in_memory_commit, persistent_commit,
        "empty commit id mismatch across backends"
    );

    Base {
        commit: persistent_commit,
        model: RegistryLongTestModel::new(permanent),
    }
}
