// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Shared utilities for end to end durable storage property-based tests,
//! operating on a single [`Database`].
//!
//! Some utilities are used in both [`Database`] and [`Registry`] tests,
//! such as [`key_strategy`], [`value_strategy`], and [`DatabaseModel`].

use std::collections::HashMap;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
#[cfg(any(test, rocksdb_test_utils))]
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Normal;
#[cfg(any(test, rocksdb_test_utils))]
use octez_riscv_data::mode::ProvableExt;
#[cfg(any(test, rocksdb_test_utils))]
use octez_riscv_data::mode::Verify;
#[cfg(any(test, rocksdb_test_utils))]
use octez_riscv_data::serialisation::serialise;
use proptest::prelude::*;
use proptest::sample::Index;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;
use tokio::runtime::Handle;

use crate::commit::CommitId;
use crate::database::Database;
use crate::database::DatabaseMode;
#[cfg(test)]
use crate::database::Trace;
#[cfg(any(test, rocksdb_test_utils))]
use crate::database::TracedDatabase;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::key::KEY_MAX_SIZE;
use crate::key::Key;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::test_helpers::OperationView;
use crate::test_helpers::StepOutcome;
use crate::test_helpers::outcome_from_value;
#[cfg(test)]
use crate::test_helpers::proof_size::assert_proof_size;
#[cfg(test)]
use crate::test_helpers::proof_size::database_operation_proof_size_bound;

/// Maximum size for the value argument of a sampled operation
pub(crate) const VALUE_MAX_SIZE: usize = 10_000;

/// Operations on a single [`Database`]
#[serde_with::serde_as]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DatabaseOperation {
    Set(Key, #[serde_as(as = "serde_with::hex::Hex")] Bytes),
    Write(Key, usize, #[serde_as(as = "serde_with::hex::Hex")] Bytes),
    Read(Key, usize, usize),
    Delete(Key),
    Exists(Key),
    ValueLength(Key),
    Hash,
    Commit,
    Checkout,
    CommitCheckoutRoundtrip,
}

#[derive(Debug, Clone)]
pub enum DatabaseOperationView {
    Set(Index, Index),
    Write(Index, usize, Index),
    Read(Index, usize, usize),
    Delete(Index),
    Exists(Index),
    ValueLength(Index),
    Hash,
    Commit,
    Checkout,
    CommitCheckoutRoundtrip,
}

pub(crate) fn make_database_operation(
    keys: &[Key],
    values: &[Bytes],
    view: DatabaseOperationView,
) -> DatabaseOperation {
    match view {
        DatabaseOperationView::Set(k_idx, v_idx) => DatabaseOperation::Set(
            keys[k_idx.index(keys.len())].clone(),
            values[v_idx.index(values.len())].clone(),
        ),
        DatabaseOperationView::Write(k_idx, offset, v_idx) => DatabaseOperation::Write(
            keys[k_idx.index(keys.len())].clone(),
            offset,
            values[v_idx.index(values.len())].clone(),
        ),
        DatabaseOperationView::Read(k_idx, offset, len) => {
            DatabaseOperation::Read(keys[k_idx.index(keys.len())].clone(), offset, len)
        }
        DatabaseOperationView::Delete(idx) => {
            DatabaseOperation::Delete(keys[idx.index(keys.len())].clone())
        }
        DatabaseOperationView::Exists(idx) => {
            DatabaseOperation::Exists(keys[idx.index(keys.len())].clone())
        }
        DatabaseOperationView::ValueLength(idx) => {
            DatabaseOperation::ValueLength(keys[idx.index(keys.len())].clone())
        }
        DatabaseOperationView::Hash => DatabaseOperation::Hash,
        DatabaseOperationView::Commit => DatabaseOperation::Commit,
        DatabaseOperationView::Checkout => DatabaseOperation::Checkout,
        DatabaseOperationView::CommitCheckoutRoundtrip => {
            DatabaseOperation::CommitCheckoutRoundtrip
        }
    }
}

/// Turn a set of [`DatabaseOperationView`]s into [`DatabaseOperation`]s
/// on the given keys and values, where applicable
pub fn make_database_operations(
    keys: Vec<Key>,
    values: Vec<Bytes>,
    ops: Vec<DatabaseOperationView>,
) -> Vec<DatabaseOperation> {
    ops.into_iter()
        .map(|op| make_database_operation(&keys, &values, op))
        .collect()
}

pub(crate) fn key_strategy() -> impl Strategy<Value = Key> {
    proptest::collection::vec(any::<u8>(), 1usize..=KEY_MAX_SIZE)
        .prop_map(|bytes| Key::new(&bytes).expect("The size is less than KEY_MAX_SIZE"))
}

pub(crate) fn value_strategy() -> impl Strategy<Value = Bytes> {
    // Bias towards lengths that fit within `MAX_FILE_CHUNK_SIZE` so most
    // sampled operations exercise the success path, while also producing
    // some oversized values.
    prop_oneof![
        9 => proptest::collection::vec(any::<u8>(), 1usize..=MAX_FILE_CHUNK_SIZE),
        1 => proptest::collection::vec(any::<u8>(), (MAX_FILE_CHUNK_SIZE + 1)..=VALUE_MAX_SIZE),
    ]
    .prop_map(Bytes::from)
}

impl OperationView for DatabaseOperationView {
    fn strategy() -> impl Strategy<Value = Self> {
        let set =
            (any::<Index>(), any::<Index>()).prop_map(|(k, v)| DatabaseOperationView::Set(k, v));

        // Bias length and offset towards having most sampled operations
        // exercise the success path, while also producing some which are out of bounds.
        let read = (
            any::<Index>(),
            prop_oneof![
                5 => Just(0),
                4 => 1..=MAX_FILE_CHUNK_SIZE,
                1 => (MAX_FILE_CHUNK_SIZE + 1)..=VALUE_MAX_SIZE,
            ],
            prop_oneof![
                9 => 0..=MAX_FILE_CHUNK_SIZE,
                1 => (MAX_FILE_CHUNK_SIZE + 1)..=VALUE_MAX_SIZE,
            ],
        )
            .prop_map(|(k, off, len)| DatabaseOperationView::Read(k, off, len));

        // Writes biased towards valid offsets
        let write_valid = (
            any::<Index>(),
            prop_oneof![
                2 => Just(0),
                1 => 1..=VALUE_MAX_SIZE,
            ],
            any::<Index>(),
        )
            .prop_map(|(k, off, v)| DatabaseOperationView::Write(k, off, v));

        // Writes biased towards out-of-bounds offsets
        let write_invalid = (any::<Index>(), VALUE_MAX_SIZE..=usize::MAX, any::<Index>())
            .prop_map(|(k, off, v)| DatabaseOperationView::Write(k, off, v));

        // The chosen frequencies emulate real workloads
        prop_oneof![
            20 => set,
            20 => read,
            4 => write_valid,
            1 => write_invalid,

            10 => any::<Index>().prop_map(DatabaseOperationView::Delete),
            10 => any::<Index>().prop_map(DatabaseOperationView::Exists),
            5 => any::<Index>().prop_map(DatabaseOperationView::ValueLength),
            10 => Just(DatabaseOperationView::Hash),
            3 => Just(DatabaseOperationView::Commit),
            3 => Just(DatabaseOperationView::Checkout),
        ]
    }

    fn roundtrip() -> Self {
        DatabaseOperationView::CommitCheckoutRoundtrip
    }
}

/// A reference model of the key-value store of a [`Database`]
pub(crate) trait DatabaseReferenceModel {
    /// The modelled key-value store
    fn data(&self) -> &HashMap<Key, Bytes>;

    /// Update the model to reflect a successfully applied `operation`
    fn apply(&mut self, operation: &DatabaseOperation);

    /// The value resulting from a successful `Write`, or `None` if it should fail
    fn write_outcome(&self, key: &Key, offset: usize, data: &Bytes) -> Option<Bytes> {
        match self.data().get(key) {
            Some(existing) => {
                if offset > existing.len()
                    || offset.checked_add(data.len()).is_none()
                    || data.len() > MAX_FILE_CHUNK_SIZE
                {
                    None
                } else {
                    let mut new_value = existing.clone();
                    update_value(&mut new_value, offset, data.clone());
                    Some(new_value)
                }
            }
            None => {
                if offset > 0 || data.len() > MAX_FILE_CHUNK_SIZE {
                    None
                } else {
                    Some(data.clone())
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DatabaseModel {
    pub(crate) data: HashMap<Key, Bytes>,
    pub(crate) last: Option<(Hash, HashMap<Key, Bytes>)>,
    pub(crate) ambiguous_hash: bool,
}

impl DatabaseModel {
    /// Record an observed root hash, asserting it is consistent with the
    /// previously recorded one: equal hashes must correspond to equal contents
    /// and vice versa, unless a deletion made the hash ambiguous.
    pub(crate) fn observe_hash(&mut self, new_digest: Hash) {
        if let (Some((old_digest, old_map)), false) = (&self.last, &self.ambiguous_hash) {
            assert_eq!(new_digest == *old_digest, self.data == *old_map);
        }
        self.last = Some((new_digest, self.data.clone()));
    }
}

impl DatabaseReferenceModel for DatabaseModel {
    fn data(&self) -> &HashMap<Key, Bytes> {
        &self.data
    }

    fn apply(&mut self, operation: &DatabaseOperation) {
        match operation {
            DatabaseOperation::Set(key, data) => {
                if data.len() <= MAX_FILE_CHUNK_SIZE {
                    self.data.insert(key.clone(), data.clone());
                }
            }
            DatabaseOperation::Write(key, offset, data) => {
                if let Some(new_value) = self.write_outcome(key, *offset, data) {
                    self.data.insert(key.clone(), new_value);
                }
            }
            DatabaseOperation::Delete(key) => {
                // The hash of the `Database` can differ even if the key-value
                // pairs stored are the same, because deletion and reinsertion
                // can cause the shape of the AVL tree to change.
                if self.data.remove(key).is_some() {
                    self.ambiguous_hash = true;
                }
            }
            DatabaseOperation::Read(..)
            | DatabaseOperation::Exists(..)
            | DatabaseOperation::ValueLength(..)
            | DatabaseOperation::Hash
            | DatabaseOperation::Commit
            | DatabaseOperation::Checkout
            | DatabaseOperation::CommitCheckoutRoundtrip => {}
        }
    }
}

fn update_value(value: &mut Bytes, offset: usize, bytes: Bytes) {
    let mut new_value: Vec<u8> = value.clone().into();
    let overwrite_len = std::cmp::min(bytes.len(), new_value.len().saturating_sub(offset));
    if overwrite_len > 0 {
        new_value[offset..offset + overwrite_len].copy_from_slice(&bytes[..overwrite_len]);
    }
    if bytes.len() > overwrite_len {
        new_value.extend_from_slice(&bytes[overwrite_len..]);
    }
    *value = Bytes::copy_from_slice(&new_value);
}

/// Abstracts the interface of [`Database`] so [`apply_database_operation`]
/// can be used in both [`Registry`] (via [`Database`] references) and
/// the `Database` tests which use [`TracedDatabase`] to capture a trace.
///
/// [`Registry`]: crate::registry::Registry
pub(crate) trait DatabaseValueOps {
    fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error>;

    fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error>;

    fn delete(&mut self, key: Key) -> Result<(), OperationalError>;

    fn read(&self, key: &Key, offset: usize, output: &mut [u8]) -> Result<usize, Error>;

    fn exists(&self, key: &Key) -> Result<bool, Error>;

    fn value_length(&self, key: &Key) -> Result<usize, Error>;

    fn hash(&self) -> Result<Hash, OperationalError>;
}

/// Extends [`DatabaseValueOps`] with persistence operations.
pub(crate) trait DatabaseOps<KV: BackgroundPersistentKeyValueStore>:
    DatabaseValueOps + Sized
{
    fn commit(&self, repo: &KV::Repo) -> Result<CommitId, OperationalError>;

    fn checkout(handle: &Handle, repo: &KV::Repo, commit_id: CommitId) -> Result<Self, Error>;

    /// Commit the database, then check out the resulting commit and replace
    /// with the checked-out database. Tracing implementations must not record
    /// these in the trace. Returns the resulting [`CommitId`].
    fn commit_checkout_roundtrip(
        &mut self,
        handle: &Handle,
        repo: &KV::Repo,
    ) -> Result<CommitId, OperationalError>;
}

impl<KV: BackgroundKeyValueStore, M: DatabaseMode> DatabaseValueOps for Database<KV, M> {
    fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error> {
        Database::set(self, key, data)
    }

    fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error> {
        Database::write(self, key, offset, data)
    }

    fn delete(&mut self, key: Key) -> Result<(), OperationalError> {
        Database::delete(self, key)
    }

    fn read(&self, key: &Key, offset: usize, output: &mut [u8]) -> Result<usize, Error> {
        Database::read(self, key, offset, output)
    }

    fn exists(&self, key: &Key) -> Result<bool, Error> {
        Database::exists(self, key)
    }

    fn value_length(&self, key: &Key) -> Result<usize, Error> {
        Database::value_length(self, key)
    }

    fn hash(&self) -> Result<Hash, OperationalError> {
        Database::hash(self)
    }
}

impl<KV: BackgroundPersistentKeyValueStore> DatabaseOps<KV> for Database<KV, Normal> {
    fn commit(&self, repo: &KV::Repo) -> Result<CommitId, OperationalError> {
        Database::commit(self, repo)
    }

    fn checkout(handle: &Handle, repo: &KV::Repo, commit_id: CommitId) -> Result<Self, Error> {
        Database::checkout(handle, repo, commit_id)
    }

    fn commit_checkout_roundtrip(
        &mut self,
        handle: &Handle,
        repo: &KV::Repo,
    ) -> Result<CommitId, OperationalError> {
        let commit_id = Database::commit(self, repo)?;
        let checked_out_db = Database::checkout(handle, repo, commit_id).map_err(|e| match e {
            Error::Operational(e) => e,
            Error::InvalidArgument(e) => {
                panic!("checking out an existing commit should succeed {e:?}")
            }
        })?;
        *self = checked_out_db;
        Ok(commit_id)
    }
}

#[cfg(any(test, rocksdb_test_utils))]
impl<KV: BackgroundKeyValueStore, M: DatabaseMode> DatabaseValueOps for TracedDatabase<KV, M> {
    fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error> {
        TracedDatabase::set(self, key, data)
    }

    fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error> {
        TracedDatabase::write(self, key, offset, data)
    }

    fn delete(&mut self, key: Key) -> Result<(), OperationalError> {
        TracedDatabase::delete(self, key)
    }

    fn read(&self, key: &Key, offset: usize, output: &mut [u8]) -> Result<usize, Error> {
        TracedDatabase::read(self, key, offset, output)
    }

    fn exists(&self, key: &Key) -> Result<bool, Error> {
        TracedDatabase::exists(self, key)
    }

    fn value_length(&self, key: &Key) -> Result<usize, Error> {
        TracedDatabase::value_length(self, key)
    }

    fn hash(&self) -> Result<Hash, OperationalError> {
        TracedDatabase::hash(self)
    }
}

#[cfg(any(test, rocksdb_test_utils))]
impl<KV: BackgroundPersistentKeyValueStore> DatabaseOps<KV> for TracedDatabase<KV, Normal> {
    fn commit(&self, repo: &KV::Repo) -> Result<CommitId, OperationalError> {
        TracedDatabase::commit(self, repo)
    }

    fn checkout(handle: &Handle, repo: &KV::Repo, commit_id: CommitId) -> Result<Self, Error> {
        TracedDatabase::checkout(handle, repo, commit_id)
    }

    fn commit_checkout_roundtrip(
        &mut self,
        handle: &Handle,
        repo: &KV::Repo,
    ) -> Result<CommitId, OperationalError> {
        // Keep the trace but replace the inner database
        self.inner_mut().commit_checkout_roundtrip(handle, repo)
    }
}

/// Apply a single operation to `database` and assert its observable outcome matches what the
/// reference `model` predicts. The model itself is not mutated. Callers need to advance it
/// separately via [`DatabaseReferenceModel::apply`]. This is in order to allow callers to check
/// and apply the same operation to multiple databases at the same time.
///
/// Returns the observable [`StepOutcome`] for provable operations (so callers can additionally
/// compare it against a prove/verify-mode outcome), and `None` for the persistence operations,
/// which are not handled here.
///
/// The operation is applied via [`apply_database_step`] so that the captured outcome is
/// constructed identically to the one produced in Prove/Verify mode, making the two directly
/// comparable.
pub(crate) fn check_and_apply_value_operation<D, M>(
    database: &mut D,
    model: &M,
    op: &DatabaseOperation,
) -> Option<StepOutcome>
where
    D: DatabaseValueOps,
    M: DatabaseReferenceModel,
{
    let outcome = apply_database_step(database, op).expect("applying a step should succeed")?;
    assert_outcome_matches_model(&outcome, model, op);
    Some(outcome)
}

/// Assert that an operation's observable [`StepOutcome`] agrees with the reference `model`.
fn assert_outcome_matches_model<M: DatabaseReferenceModel>(
    outcome: &StepOutcome,
    model: &M,
    op: &DatabaseOperation,
) {
    match (op, outcome) {
        (DatabaseOperation::Set(_, bytes), StepOutcome::Unit(result)) => {
            if bytes.len() <= MAX_FILE_CHUNK_SIZE {
                assert!(result.is_ok(), "Set should have succeeded: {result:?}");
            } else {
                assert!(result.is_err(), "Set should have failed but succeeded");
            }
        }
        (DatabaseOperation::Write(key, offset, bytes), StepOutcome::Length(result)) => {
            let should_succeed = model.write_outcome(key, *offset, bytes).is_some();
            assert_eq!(
                should_succeed,
                result.is_ok(),
                "Write outcome disagrees with the model: {result:?}"
            );

            if let Ok(length) = result {
                assert_eq!(
                    *length,
                    bytes.len(),
                    "num bytes_written disagreement between model and application"
                );
            }
        }
        (DatabaseOperation::Read(key, offset, len), StepOutcome::Read(result)) => {
            if let Some(map_value) = model.data().get(key) {
                if *offset > map_value.len() || *len > MAX_FILE_CHUNK_SIZE {
                    assert!(result.is_err(), "Read should have failed but succeeded");
                } else {
                    let bytes = result.as_ref().expect("Read should have succeeded");
                    let expected_len = std::cmp::min(*len, map_value.len() - offset);
                    assert!(bytes.len() >= expected_len);
                    assert_eq!(
                        &bytes[..expected_len],
                        &map_value[*offset..*offset + expected_len]
                    );
                }
            } else {
                assert!(result.is_err(), "Read of a missing key should have failed");
            }
        }
        (DatabaseOperation::Delete(_), StepOutcome::Unit(result)) => {
            assert!(result.is_ok(), "Delete should have succeeded: {result:?}");
        }
        (DatabaseOperation::Exists(key), StepOutcome::Exists(result)) => {
            let in_database = *result.as_ref().expect("Exists should have succeeded");
            assert_eq!(in_database, model.data().contains_key(key));
        }
        (DatabaseOperation::ValueLength(key), StepOutcome::Length(result)) => {
            match (result, model.data().get(key)) {
                (Ok(length), Some(map_value)) => assert_eq!(*length, map_value.len()),
                (Err(_), None) => (),
                _ => panic!("The value exists in one map but not the other"),
            }
        }
        (DatabaseOperation::Hash, StepOutcome::Hash(result)) => {
            assert!(result.is_ok(), "Hash should have succeeded: {result:?}");
        }
        _ => panic!("outcome {outcome:?} does not correspond to operation {op:?}"),
    }
}

/// Apply `op` to `database`, keeping `model` and `checkout_candidates` in sync and asserting
/// observable results against the model.
///
/// Returns the observable [`StepOutcome`] for provable operations (so callers can compare it
/// against a prove/verify-mode outcome), and `None` for the persistence operations.
pub(crate) fn apply_database_operation_with_model<KV, D>(
    database: &mut D,
    model: &mut DatabaseModel,
    op: &DatabaseOperation,
    handle: &Handle,
    repo: &KV::Repo,
    checkout_candidates: &mut HashMap<Hash, bool>,
) -> Option<StepOutcome>
where
    KV: BackgroundPersistentKeyValueStore,
    D: DatabaseOps<KV>,
{
    match op {
        DatabaseOperation::Hash => {
            let outcome = check_and_apply_value_operation(database, model, op)
                .expect("Hash operations produce an outcome");
            let StepOutcome::Hash(Ok(new_digest)) = outcome else {
                panic!("Hash operation should produce a hash outcome, got {outcome:?}");
            };
            model.observe_hash(new_digest);
            checkout_candidates.entry(new_digest).or_insert(false);
            Some(StepOutcome::Hash(Ok(new_digest)))
        }
        DatabaseOperation::Commit => {
            let commit_id = database.commit(repo).expect("Committing should succeed");
            checkout_candidates.insert(*commit_id.as_hash(), true);
            None
        }
        DatabaseOperation::Checkout => {
            if !checkout_candidates.is_empty() {
                let index = rand::random_range(0..checkout_candidates.len());
                let (&commit_hash, &committed) = checkout_candidates
                    .iter()
                    .nth(index)
                    .expect("Index is within bounds");
                let checkout_result = D::checkout(handle, repo, CommitId::from(commit_hash));

                assert_eq!(
                    checkout_result.is_ok(),
                    committed,
                    "Checkout result did not match whether the commit id was committed"
                );
            }
            None
        }
        DatabaseOperation::CommitCheckoutRoundtrip => {
            let commit_id = database
                .commit_checkout_roundtrip(handle, repo)
                .expect("Commit-checkout roundtrip should succeed");
            // Register the resulting commit so a later `Checkout` operation
            // does not see an unexpected success against this hash.
            checkout_candidates.insert(*commit_id.as_hash(), true);
            None
        }
        op => {
            let outcome = check_and_apply_value_operation(database, model, op);
            model.apply(op);
            outcome
        }
    }
}

/// Run a sequence of [`DatabaseOperation`]s against a single [`TracedDatabase`],
/// asserting against a reference in-memory model after each step, and return the recorded [`Trace`].
#[cfg(test)]
pub(crate) fn run_database_operations<KV>(
    repo: &KV::Repo,
    mut operations: Vec<DatabaseOperation>,
) -> Trace
where
    KV: BackgroundPersistentKeyValueStore,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .expect("Building the runtime should succeed");
    let handle = runtime.handle();

    let mut database = TracedDatabase::<KV, Normal>::try_new(handle, repo)
        .expect("Creating the database should succeed");
    let mut model = DatabaseModel::default();
    let mut checkout_candidates: HashMap<Hash, bool> = HashMap::new();

    // Force a final hash to be recorded in the trace
    operations.push(DatabaseOperation::Hash);

    for op in operations {
        apply_database_operation_with_model::<KV, _>(
            &mut database,
            &mut model,
            &op,
            handle,
            repo,
            &mut checkout_candidates,
        );
    }

    database.into_trace()
}

/// Apply a value `operation` to `database`, capturing its observable outcome.
///
/// Operational errors are propagated (they indicate a harness or implementation bug, not an
/// observable result); invalid-argument failures and `Ok` values are captured in the returned
/// [`StepOutcome`]. Returns `None` for the persistence operations, which are not provable
/// steps.
pub(crate) fn apply_database_step<D: DatabaseValueOps>(
    database: &mut D,
    operation: &DatabaseOperation,
) -> Result<Option<StepOutcome>, OperationalError> {
    let outcome = match operation {
        DatabaseOperation::Set(key, data) => {
            outcome_from_value(database.set(key.clone(), data.clone()), StepOutcome::Unit)?
        }
        DatabaseOperation::Write(key, offset, data) => outcome_from_value(
            database.write(key.clone(), *offset, data.clone()),
            StepOutcome::Length,
        )?,
        DatabaseOperation::Read(key, offset, len) => {
            let mut buffer = vec![0; *len];
            let mut cursor = 0;
            loop {
                match database.read(key, offset + cursor, &mut buffer[cursor..]) {
                    Ok(0) => break StepOutcome::Read(Ok(buffer[..cursor].to_vec())),
                    Ok(read) => cursor += read,
                    Err(Error::InvalidArgument(error)) => {
                        break StepOutcome::Read(Err(format!("{error:?}")));
                    }
                    Err(Error::Operational(error)) => return Err(error),
                }
            }
        }
        DatabaseOperation::Delete(key) => {
            database.delete(key.clone())?;
            StepOutcome::Unit(Ok(()))
        }
        DatabaseOperation::Exists(key) => {
            outcome_from_value(database.exists(key), StepOutcome::Exists)?
        }
        DatabaseOperation::ValueLength(key) => {
            outcome_from_value(database.value_length(key), StepOutcome::Length)?
        }
        DatabaseOperation::Hash => StepOutcome::Hash(Ok(database.hash()?)),
        DatabaseOperation::Commit
        | DatabaseOperation::Checkout
        | DatabaseOperation::CommitCheckoutRoundtrip => {
            return Ok(None);
        }
    };

    Ok(Some(outcome))
}

/// Generate and verify a proof for a single [`DatabaseOperation`] applied to `database`.
///
/// Returns the serialised proof together with the operation's observable [`StepOutcome`], or
/// `None` if `operation` is not a provable step. The Prove- and Verify-mode outcomes are
/// asserted to be equal before returning; the returned outcome is the (identical) Prove-mode one.
#[cfg(any(test, rocksdb_test_utils))]
pub(crate) fn prove_and_verify_database_operation<KV: BackgroundKeyValueStore>(
    database: &Database<KV, Normal>,
    operation: &DatabaseOperation,
) -> Option<(Vec<u8>, StepOutcome)> {
    use octez_riscv_data::hash::PartialHash;

    let pre_root_hash = Hash::from_foldable(database);

    // Produce a proof and record the trace of applying `operation`. The prover is wrapped in a
    // `TracedDatabase` so its per-step trace can be compared against the Verify-mode replay below.
    let mut prover = TracedDatabase::from(
        database
            .try_start_proof()
            .expect("starting a proof should succeed"),
    );

    // Nothing to record or compare if the step was not provable
    let prove_outcome =
        apply_database_step(&mut prover, operation).expect("applying a step should succeed")?;

    let post_root_hash = Hash::from_foldable(&prover);
    let proof = MerkleProof::from_foldable(&prover);
    let proof_step_trace = prover.into_trace();
    let proof_bytes = serialise(&proof).expect("serialising the proof should succeed");

    // Construct the Verify-mode database from the proof and verify
    let mut verify_db = TracedDatabase::from(
        <Database<KV, Verify> as octez_riscv_data::merkle_proof::FromProof>::from_proof(
            octez_riscv_data::merkle_proof::proof_tree::ProofTree::Present(&proof),
        )
        .expect("proof should be valid")
        .into_result(),
    );
    assert_eq!(
        PartialHash::from_foldable(None, &verify_db)
            .to_hash()
            .expect("hashing the Verify database should succeed"),
        pre_root_hash,
        "the proof must reconstruct the pre-operation root hash"
    );
    let verify_outcome = apply_database_step(&mut verify_db, operation)
        .expect("applying a step should succeed")
        .expect("a provable step in Prove mode must be provable in Verify mode");
    let verify_post_root_hash = PartialHash::from_foldable(None, &verify_db)
        .to_hash()
        .expect("hashing the Verify database should succeed");
    let verify_step_trace = verify_db.into_trace();

    assert_eq!(
        prove_outcome, verify_outcome,
        "Prove- and Verify-mode operations must produce the same observable result"
    );
    assert_eq!(
        verify_step_trace, proof_step_trace,
        "Prove- and Verify-mode execution traces should match"
    );
    assert_eq!(
        verify_post_root_hash, post_root_hash,
        "Prove- and Verify-mode root hashes should match"
    );

    Some((proof_bytes, prove_outcome))
}

/// A proof recorded for a single provable [`DatabaseOperation`], together with the operation's
/// observable outcome (asserted equal across Normal, Prove and Verify mode).
#[cfg(test)]
#[serde_with::serde_as]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct DatabaseProofStep {
    step: DatabaseOperation,
    #[serde_as(as = "serde_with::hex::Hex")]
    proof: Vec<u8>,
    outcome: StepOutcome,
}

/// Like [`run_database_operations`], but additionally generates and verifies a proof for every
/// provable operation, returning the sequence of recorded proofs and their observable outcomes.
///
/// Unlike the interleaved [`Trace`] returned by [`run_database_operations`], each provable
/// operation contributes a single [`DatabaseProofStep`] carrying the operation, its proof and its
/// outcome — so the operation is recorded once rather than duplicated alongside a separate proof
/// entry.
#[cfg(test)]
pub(crate) fn run_and_prove_database_operations<KV>(
    repo: &KV::Repo,
    mut operations: Vec<DatabaseOperation>,
) -> Vec<DatabaseProofStep>
where
    KV: BackgroundPersistentKeyValueStore,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .expect("Building the runtime should succeed");
    let handle = runtime.handle();

    let mut database = Database::<KV, Normal>::try_new(handle, repo)
        .expect("Creating the database should succeed");
    let mut model = DatabaseModel::default();
    let mut checkout_candidates: HashMap<Hash, bool> = HashMap::new();
    let mut proof_steps = Vec::new();

    // Force a final hash so a proof is generated for the terminal state.
    operations.push(DatabaseOperation::Hash);

    for operation in operations {
        // Provable operations are proven over their pre-operation state, so prove before applying.
        // The size bound is likewise computed over the pre-operation model.
        let bound = database_operation_proof_size_bound(&model, &operation);
        let proof_and_outcome = prove_and_verify_database_operation(&database, &operation);
        if let Some((proof, _)) = &proof_and_outcome {
            let bound = bound.expect("provable operations have a size bound");
            assert_proof_size(&operation, proof.len(), bound, false);
        }

        let normal_outcome = apply_database_operation_with_model::<KV, _>(
            &mut database,
            &mut model,
            &operation,
            handle,
            repo,
            &mut checkout_candidates,
        );

        // The proof is generated over the pre-operation state, so the Prove/Verify-mode result
        // must match the result the same operation produces in Normal mode.
        if let Some((proof, prove_outcome)) = proof_and_outcome {
            let normal_outcome =
                normal_outcome.expect("a provable operation must produce a Normal-mode outcome");
            assert_eq!(
                prove_outcome, normal_outcome,
                "Prove/Verify-mode result must match the Normal-mode result"
            );
            proof_steps.push(DatabaseProofStep {
                step: operation.clone(),
                proof,
                outcome: prove_outcome,
            });
        }
    }

    proof_steps
}
