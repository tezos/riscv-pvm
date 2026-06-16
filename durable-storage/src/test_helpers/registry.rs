// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Shared utilities for end to end durable storage property-based tests,
//! operating on a [`Registry`].
//!
//! Used by the integration test in `tests/integration_test.rs` and the in-crate
//! `kv_test!`s for `registry.rs`. Database-level utilities are reused from
//! [`super::database`].

use std::collections::HashMap;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::proof::deserialise_proof;
use octez_riscv_data::merkle_proof::proof::serialise_proof;
use octez_riscv_data::merkle_proof::proof_tree::ProofPart;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::ProvableExt;
use octez_riscv_data::mode::Verify;
use proptest::prelude::*;

use super::database::DatabaseModel;
use super::database::DatabaseOperation;
use super::database::DatabaseOperationView;
use super::database::apply_database_operation_with_model;
use super::database::apply_database_step;
use super::database::database_operation_view_strategy;
use super::database::key_strategy;
use super::database::make_database_operation;
use super::database::value_strategy;
use crate::commit::CommitId;
use crate::database::DatabaseMode;
use crate::key::Key;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::registry::Registry;
use crate::registry::RegistryMode;
use crate::repo::RegistryRepo;

/// Operations on a [`Registry`]
#[derive(Debug, Clone)]
pub enum Operation {
    Database(DatabaseOperation),
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase,
    MoveDatabase,
    ClearDatabase,
}

#[derive(Debug, Clone)]
pub enum OperationView {
    Database(DatabaseOperationView),
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase,
    MoveDatabase,
    ClearDatabase,
}

/// Turn a set of [`OperationView`]s into [`Operation`]s on the given keys
/// and values, where applicable
pub fn make_registry_operations(
    keys: Vec<Key>,
    values: Vec<Bytes>,
    ops: Vec<OperationView>,
) -> Vec<Operation> {
    ops.into_iter()
        .map(|op| match op {
            OperationView::Database(view) => {
                Operation::Database(make_database_operation(&keys, &values, view))
            }
            OperationView::GrowRegistry => Operation::GrowRegistry,
            OperationView::ShrinkRegistry => Operation::ShrinkRegistry,
            OperationView::CopyDatabase => Operation::CopyDatabase,
            OperationView::MoveDatabase => Operation::MoveDatabase,
            OperationView::ClearDatabase => Operation::ClearDatabase,
        })
        .collect()
}

pub fn registry_operations_strategy(
    length: impl Strategy<Value = usize>,
) -> impl Strategy<Value = (Vec<Key>, Vec<Bytes>, Vec<OperationView>)> {
    length.prop_flat_map(|length| {
        let count = length.div_ceil(10);

        (
            proptest::collection::vec(key_strategy(), count),
            proptest::collection::vec(value_strategy(), count),
            proptest::collection::vec(
                // The chosen frequencies emulate real workloads
                prop_oneof![
                    86 => database_operation_view_strategy().prop_map(OperationView::Database),
                    3 => Just(OperationView::GrowRegistry),
                    2 => Just(OperationView::ShrinkRegistry),
                    2 => Just(OperationView::CopyDatabase),
                    1 => Just(OperationView::MoveDatabase),
                    2 => Just(OperationView::ClearDatabase),
                ],
                length,
            ),
        )
    })
}

fn grow_registry<KV, M>(registry: &mut Registry<KV, M>)
where
    KV: BackgroundKeyValueStore,
    M: RegistryMode,
{
    let new = registry.len();

    registry
        .resize_tick(new.saturating_add(1))
        .expect("Resizing the registry should succeed");

    if let Some(previous) = new.checked_sub(1) {
        registry
            .copy_database(previous, new)
            .expect("Copying the database should succeed");
    }
}

fn grow_registry_with_model<KV>(
    registry: &mut Registry<KV, Normal>,
    registry_model: &mut Vec<DatabaseModel>,
) where
    KV: BackgroundPersistentKeyValueStore,
    KV::Repo: RegistryRepo,
{
    grow_registry(registry);

    if registry_model.is_empty() {
        registry_model.resize(1, Default::default());
    } else {
        registry_model.push(registry_model[registry_model.len() - 1].clone());
    }
}

/// Get the index of the current database, growing the registry until it has a valid index.
fn get_registry_index_with_model<KV>(
    registry: &mut Registry<KV, Normal>,
    registry_model: &mut Vec<DatabaseModel>,
) -> usize
where
    KV: BackgroundPersistentKeyValueStore,
    KV::Repo: RegistryRepo,
{
    if let Some(index) = registry.len().checked_sub(1) {
        index
    } else {
        grow_registry_with_model(registry, registry_model);
        get_registry_index_with_model(registry, registry_model)
    }
}

/// Mode-generic counterpart to [`get_registry_index_with_model`] which does not
/// update a Registry model.
fn get_registry_index<KV, M>(registry: &mut Registry<KV, M>) -> usize
where
    KV: BackgroundKeyValueStore,
    M: RegistryMode,
{
    match registry.len().checked_sub(1) {
        Some(index) => index,
        None => {
            grow_registry(registry);
            registry.len() - 1
        }
    }
}

/// Apply a single [`Operation`] to `registry`.
///
/// Returns `true` if the step was provable.
fn apply_registry_step<KV, M>(registry: &mut Registry<KV, M>, op: &Operation) -> bool
where
    KV: BackgroundKeyValueStore,
    M: RegistryMode + DatabaseMode,
{
    match op {
        Operation::Database(
            DatabaseOperation::Commit
            | DatabaseOperation::Checkout
            | DatabaseOperation::CommitCheckoutRoundtrip,
        ) => return false,
        Operation::Database(db_op) => {
            let index = get_registry_index(registry);
            let database = registry
                .database_mut(index)
                .expect("The index is in bounds");
            return apply_database_step(database, db_op).expect("applying a step should succeed");
        }
        Operation::GrowRegistry => grow_registry(registry),
        Operation::ShrinkRegistry => {
            if registry.is_empty() {
                grow_registry(registry);
            }
            let new_size = registry.len().saturating_sub(1);
            registry
                .resize_tick(new_size)
                .expect("Resizing the registry should succeed");
        }
        Operation::ClearDatabase => {
            let index = get_registry_index(registry);
            registry
                .clear_database(index)
                .expect("Clearing the database should be successful");
        }
        Operation::CopyDatabase => {
            while registry.len() < 2 {
                grow_registry(registry);
            }
            let (src, dst) = (registry.len() - 2, registry.len() - 1);
            registry
                .copy_database(src, dst)
                .expect("Copying the database should be successful");
        }
        Operation::MoveDatabase => {
            while registry.len() < 2 {
                grow_registry(registry);
            }
            let (src, dst) = (registry.len() - 2, registry.len() - 1);
            registry
                .move_database(src, dst)
                .expect("Moving the database should be successful");
        }
    }

    true
}

/// Generate and verify a proof for a single provable [`Operation`] applied to `registry`.
fn prove_and_verify_registry_operation<KV>(registry: &Registry<KV, Normal>, op: &Operation)
where
    KV: BackgroundKeyValueStore,
    KV::Repo: Clone,
{
    let pre_root = Hash::from_foldable(registry);

    let mut prover = registry
        .try_start_proof()
        .expect("Starting a proof should succeed");
    if !apply_registry_step(&mut prover, op) {
        return;
    }
    let proof = prover.produce_proof();
    assert_eq!(
        proof.initial_state_hash(),
        pre_root,
        "The proof must encode the registry's pre-operation state"
    );

    let bytes = serialise_proof(&proof);
    let (reconstructed, _stream) = deserialise_proof::<Registry<KV, Verify>, _>(bytes.into_iter())
        .expect("Stream deserialisation of the proof bytes should succeed");
    assert_eq!(
        reconstructed, proof,
        "The proof reconstructed from bytes should match the original"
    );
    let mut verify = Registry::<KV, Verify>::from_proof(ProofPart::Present(reconstructed.tree()))
        .expect("from_proof should succeed")
        .into_result();

    let verify_pre = PartialHash::from_foldable(Some(reconstructed.tree().clone()), &verify)
        .to_hash()
        .expect("Hashing the Verify registry should succeed");
    assert_eq!(
        verify_pre, pre_root,
        "The Verify-mode registry must start from the pre-operation state hash"
    );

    apply_registry_step(&mut verify, op);
    let verify_post = PartialHash::from_foldable(Some(reconstructed.tree().clone()), &verify)
        .to_hash()
        .expect("Hashing the Verify registry should succeed");
    assert_eq!(
        verify_post,
        proof.final_state_hash(),
        "Replaying the step in Verify mode must reach the proof's final state hash"
    )
}

pub fn run_and_prove_registry_operations<KV>(repo: KV::Repo, operations: Vec<Operation>)
where
    KV: BackgroundPersistentKeyValueStore,
    KV::Repo: RegistryRepo,
{
    let checkout_repo = repo.clone();
    let mut checkout_candidates: HashMap<Hash, bool> = HashMap::new();

    let mut registry: Registry<KV, Normal> =
        Registry::new(repo).expect("Creating the registry should succeed");

    let mut registry_model: Vec<DatabaseModel> = vec![];

    for operation in operations {
        prove_and_verify_registry_operation(&registry, &operation);

        match operation {
            Operation::Database(DatabaseOperation::Hash) => {
                let index = get_registry_index_with_model(&mut registry, &mut registry_model);

                let new_digest = registry
                    .database(index)
                    .expect("The index is in bounds")
                    .hash()
                    .expect("Hash should succeed");

                registry_model[index].observe_hash(new_digest);

                checkout_candidates
                    .entry(Hash::from_foldable(&registry))
                    .or_insert(false);
            }
            Operation::Database(DatabaseOperation::Commit) => {
                let commit_id = registry.commit().expect("Committing should succeed");
                checkout_candidates.insert(*commit_id.as_hash(), true);
            }
            Operation::Database(DatabaseOperation::Checkout) => {
                if !checkout_candidates.is_empty() {
                    let index = rand::random_range(0..checkout_candidates.len());
                    let (&commit_hash, &committed) = checkout_candidates
                        .iter()
                        .nth(index)
                        .expect("Index is within bounds");
                    let checkout_result = Registry::<KV, Normal>::checkout(
                        checkout_repo.clone(),
                        CommitId::from(commit_hash),
                    );

                    assert_eq!(
                        checkout_result.is_ok(),
                        committed,
                        "Checkout result did not match whether the commit id was committed"
                    );
                }
            }
            Operation::Database(op) => {
                let index = get_registry_index_with_model(&mut registry, &mut registry_model);
                let handle = registry.handle().clone();
                apply_database_operation_with_model::<KV, _>(
                    registry
                        .database_mut(index)
                        .expect("The index is in bounds"),
                    &mut registry_model[index],
                    &op,
                    &handle,
                    &checkout_repo,
                    &mut checkout_candidates,
                );
            }
            Operation::GrowRegistry => grow_registry_with_model(&mut registry, &mut registry_model),
            Operation::ShrinkRegistry => {
                if registry.is_empty() {
                    grow_registry_with_model(&mut registry, &mut registry_model);
                };

                let new_size = registry.len().saturating_sub(1);
                registry
                    .resize_tick(new_size)
                    .expect("Resizing the registry should succeed");

                registry_model.truncate(new_size);
            }
            Operation::ClearDatabase => {
                let index = get_registry_index_with_model(&mut registry, &mut registry_model);

                registry
                    .clear_database(index)
                    .expect("Clearing the database should be successful");

                registry_model[index].data.clear();
                registry_model[index].ambiguous_hash = false;
                registry_model[index].last = None;
            }
            Operation::CopyDatabase => {
                let (src, dst) = {
                    while registry.len() < 2 {
                        grow_registry_with_model(&mut registry, &mut registry_model);
                    }
                    (registry.len() - 2, registry.len() - 1)
                };

                registry
                    .copy_database(src, dst)
                    .expect("Copying the database should be successful");

                registry_model[dst] = registry_model[src].clone();
            }
            Operation::MoveDatabase => {
                let (src, dst) = {
                    while registry.len() < 2 {
                        grow_registry_with_model(&mut registry, &mut registry_model);
                    }
                    (registry.len() - 2, registry.len() - 1)
                };

                registry
                    .move_database(src, dst)
                    .expect("Moving the database should be successful");

                let empty = Default::default();
                let new_dst = std::mem::replace(&mut registry_model[src], empty);
                registry_model[dst] = new_dst;
            }
        }
    }
}
