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
use std::num::NonZeroUsize;

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
use proptest::sample::Index;

use super::database::DatabaseModel;
use super::database::DatabaseOperation;
use super::database::DatabaseOperationView;
use super::database::apply_database_operation_with_model;
use super::database::apply_database_step;
use super::database::make_database_operation;
use crate::commit::CommitId;
use crate::database::DatabaseMode;
use crate::key::Key;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::registry::Registry;
use crate::registry::RegistryMode;
use crate::repo::RegistryRepo;
use crate::test_helpers::OperationView;

/// Operations on a [`Registry`]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RegistryOperation {
    Database(usize, DatabaseOperation),
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase(usize, usize),
    MoveDatabase(usize, usize),
    ClearDatabase(usize),
    CommitCheckoutRoundtrip,
}

/// A proof recorded for a single provable [`RegistryOperation`]
#[serde_with::serde_as]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegistryProofStep {
    step: RegistryOperation,
    #[serde_as(as = "serde_with::hex::Hex")]
    proof: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum RegistryOperationView {
    Database(Index, DatabaseOperationView),
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase(Index, Index),
    MoveDatabase(Index, Index),
    ClearDatabase(Index),
    CommitCheckoutRoundtrip,
}

/// Turn a set of [`RegistryOperationView`]s into [`RegistryOperation`]s on the given keys
/// and values, where applicable.
///
/// Registry indices are resolved against the registry length computed
/// at every point in the sequence of operations. For them to be valid,
/// `initial_registry_len` must match the length of the registry on which
/// the operations will be applied.
pub fn make_registry_operations(
    initial_registry_len: NonZeroUsize,
    keys: Vec<Key>,
    values: Vec<Bytes>,
    ops: Vec<RegistryOperationView>,
) -> Vec<RegistryOperation> {
    let mut len = initial_registry_len.get();
    ops.into_iter()
        .map(|op| match op {
            RegistryOperationView::Database(idx, view) => RegistryOperation::Database(
                idx.index(len),
                make_database_operation(&keys, &values, view),
            ),
            RegistryOperationView::GrowRegistry => {
                len += 1;
                RegistryOperation::GrowRegistry
            }
            RegistryOperationView::ShrinkRegistry => {
                if len > 1 {
                    len -= 1;
                }
                RegistryOperation::ShrinkRegistry
            }
            RegistryOperationView::CopyDatabase(src, dst) => {
                RegistryOperation::CopyDatabase(src.index(len), dst.index(len))
            }
            RegistryOperationView::MoveDatabase(src, dst) => {
                RegistryOperation::MoveDatabase(src.index(len), dst.index(len))
            }
            RegistryOperationView::ClearDatabase(idx) => {
                RegistryOperation::ClearDatabase(idx.index(len))
            }
            RegistryOperationView::CommitCheckoutRoundtrip => {
                RegistryOperation::CommitCheckoutRoundtrip
            }
        })
        .collect()
}

impl OperationView for RegistryOperationView {
    fn strategy() -> impl Strategy<Value = Self> {
        // The chosen frequencies emulate real workloads
        prop_oneof![
            88 => (any::<Index>(), DatabaseOperationView::strategy())
                .prop_map(|(i, v)| RegistryOperationView::Database(i, v)),
            4 => Just(RegistryOperationView::GrowRegistry),
            2 => Just(RegistryOperationView::ShrinkRegistry),
            3 => (any::<Index>(), any::<Index>())
                .prop_map(|(src, dst)| RegistryOperationView::CopyDatabase(src, dst)),
            2 => (any::<Index>(), any::<Index>())
                .prop_map(|(src, dst)| RegistryOperationView::MoveDatabase(src, dst)),
            1 => any::<Index>().prop_map(RegistryOperationView::ClearDatabase),
        ]
    }

    fn roundtrip() -> Self {
        RegistryOperationView::CommitCheckoutRoundtrip
    }
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
}

fn grow_registry_with_model<KV>(
    registry: &mut Registry<KV, Normal>,
    registry_model: &mut Vec<DatabaseModel>,
) where
    KV: BackgroundPersistentKeyValueStore,
    KV::Repo: RegistryRepo,
{
    grow_registry(registry);

    registry_model.push(Default::default());
}

/// Apply a single [`RegistryOperation`] to `registry`.
///
/// Returns `true` if the step was provable. A no-op (e.g., a `ShrinkRegistry` when
/// `registry` has size 1) is a provable step: a proof which fully blinds the registry
/// is expected to be produced for it.
fn apply_registry_step<KV, M>(
    registry: &mut Registry<KV, M>,
    op: &RegistryOperation,
    len: usize,
) -> bool
where
    KV: BackgroundKeyValueStore,
    M: RegistryMode + DatabaseMode,
{
    match op {
        RegistryOperation::Database(
            _,
            DatabaseOperation::Commit
            | DatabaseOperation::Checkout
            | DatabaseOperation::CommitCheckoutRoundtrip,
        ) => return false,
        RegistryOperation::CommitCheckoutRoundtrip => return false,
        RegistryOperation::Database(index, db_op) => {
            let database = registry
                .database_mut(*index)
                .expect("The index is in bounds");
            return apply_database_step(database, db_op).expect("applying a step should succeed");
        }
        RegistryOperation::GrowRegistry => grow_registry(registry),
        RegistryOperation::ShrinkRegistry => {
            if len <= 1 {
                return true;
            }
            registry
                .resize_tick(len - 1)
                .expect("Resizing the registry should succeed");
        }
        RegistryOperation::ClearDatabase(index) => {
            registry
                .clear_database(*index)
                .expect("Clearing the database should be successful");
        }
        RegistryOperation::CopyDatabase(src, dst) => {
            registry
                .copy_database(*src, *dst)
                .expect("Copying the database should be successful");
        }
        RegistryOperation::MoveDatabase(src, dst) => {
            registry
                .move_database(*src, *dst)
                .expect("Moving the database should be successful");
        }
    }

    true
}

/// Generate and verify a proof for a single [`RegistryOperation`] applied to `registry`.
/// Returns the serialised proof, or `None` if `op` is not a provable step.
fn prove_and_verify_registry_operation<KV>(
    registry: &Registry<KV, Normal>,
    op: &RegistryOperation,
) -> Option<Vec<u8>>
where
    KV: BackgroundKeyValueStore,
    KV::Repo: Clone,
{
    let pre_root = Hash::from_foldable(registry);

    // Pre-operation length, read from the original registry (not the Prove-mode registry)
    // to avoid touching the registry when deciding if an operation should be a no-op.
    let len = registry.len();

    let mut prover = registry
        .try_start_proof()
        .expect("Starting a proof should succeed");
    if !apply_registry_step(&mut prover, op, len) {
        return None;
    }
    let proof = prover.produce_proof();
    assert_eq!(
        proof.initial_state_hash(),
        pre_root,
        "The proof must encode the registry's pre-operation state"
    );

    let bytes = serialise_proof(&proof);
    let (reconstructed, _stream) =
        deserialise_proof::<Registry<KV, Verify>, _>(bytes.clone().into_iter())
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

    apply_registry_step(&mut verify, op, len);
    let verify_post = PartialHash::from_foldable(Some(reconstructed.tree().clone()), &verify)
        .to_hash()
        .expect("Hashing the Verify registry should succeed");
    assert_eq!(
        verify_post,
        proof.final_state_hash(),
        "Replaying the step in Verify mode must reach the proof's final state hash"
    );

    Some(bytes)
}

/// Initialises a Normal-mode [`Registry`] in the given `repo` and applies
/// `operations` one by one. On each operation:
/// - checks the result agrees with a reference model
/// - proves and verifies a proof if the operation is provable, then
///   records the proof and the operation as a [`RegistryProofStep`]
///
/// Returns the vector of [`RegistryProofStep`]s, which can be used
/// to check that applying the same operations over registries
/// configured with different backends does not result in a state divergence.
pub fn run_and_prove_registry_operations<KV>(
    repo: KV::Repo,
    operations: Vec<RegistryOperation>,
) -> Vec<RegistryProofStep>
where
    KV: BackgroundPersistentKeyValueStore,
    KV::Repo: RegistryRepo,
{
    let checkout_repo = repo.clone();
    let mut checkout_candidates: HashMap<Hash, bool> = HashMap::new();

    let mut registry: Registry<KV, Normal> =
        Registry::new(repo).expect("Creating the registry should succeed");

    let mut registry_model: Vec<DatabaseModel> = vec![];
    let mut proof_steps: Vec<RegistryProofStep> = Vec::new();

    // Start with a size 1 registry
    grow_registry_with_model(&mut registry, &mut registry_model);

    for operation in operations {
        if let Some(proof) = prove_and_verify_registry_operation(&registry, &operation) {
            proof_steps.push(RegistryProofStep {
                step: operation.clone(),
                proof,
            });
        }

        match operation {
            RegistryOperation::Database(index, DatabaseOperation::Hash) => {
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
            RegistryOperation::Database(_, DatabaseOperation::Commit) => {
                let commit_id = registry.commit().expect("Committing should succeed");
                checkout_candidates.insert(*commit_id.as_hash(), true);
            }
            RegistryOperation::Database(_, DatabaseOperation::Checkout) => {
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
            RegistryOperation::Database(index, op) => {
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
            RegistryOperation::GrowRegistry => {
                grow_registry_with_model(&mut registry, &mut registry_model)
            }
            RegistryOperation::ShrinkRegistry => {
                // Never shrink to an empty registry. This case becomes a no-op.
                if registry.len() <= 1 {
                    continue;
                }

                let new_size = registry.len() - 1;
                registry
                    .resize_tick(new_size)
                    .expect("Resizing the registry should succeed");

                registry_model.truncate(new_size);
            }
            RegistryOperation::ClearDatabase(index) => {
                registry
                    .clear_database(index)
                    .expect("Clearing the database should be successful");

                registry_model[index].data.clear();
                registry_model[index].ambiguous_hash = false;
                registry_model[index].last = None;
            }
            RegistryOperation::CopyDatabase(src, dst) => {
                registry
                    .copy_database(src, dst)
                    .expect("Copying the database should be successful");

                if src != dst {
                    registry_model[dst] = registry_model[src].clone();
                }
            }
            RegistryOperation::MoveDatabase(src, dst) => {
                registry
                    .move_database(src, dst)
                    .expect("Moving the database should be successful");

                if src != dst {
                    registry_model[dst] = std::mem::take(&mut registry_model[src]);
                }
            }
            RegistryOperation::CommitCheckoutRoundtrip => {
                let commit_id = registry.commit().expect("Committing should succeed");
                registry = Registry::checkout(checkout_repo.clone(), commit_id)
                    .expect("Checking out the just-committed registry should succeed");
                // State is preserved, so `registry_model` is left unchanged.
                checkout_candidates.insert(*commit_id.as_hash(), true);
            }
        }
    }

    proof_steps
}
