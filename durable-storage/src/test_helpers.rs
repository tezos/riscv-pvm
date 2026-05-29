// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

#![cfg(any(test, feature = "unstable-test-utils"))]

//! Shared utilities for end to end durable storage property-based tests
//!
//! Used by the integration test in `tests/integration_test.rs` and in-crate
//! `kv_test!`s for `registry.rs` and `database.rs`.

use std::collections::HashMap;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
#[cfg(test)]
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Normal;
#[cfg(test)]
use octez_riscv_data::mode::ProvableExt;
#[cfg(test)]
use octez_riscv_data::mode::Verify;
#[cfg(test)]
use octez_riscv_data::serialisation::serialise;
use proptest::prelude::*;
use proptest::sample::Index;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;
use tokio::runtime::Handle;

use crate::commit::CommitId;
use crate::database::Database;
#[cfg(test)]
use crate::database::DatabaseMode;
#[cfg(test)]
use crate::database::Trace;
#[cfg(test)]
use crate::database::TracedDatabase;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::key::KEY_MAX_SIZE;
use crate::key::Key;
#[cfg(test)]
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
use crate::registry::Registry;
use crate::repo::RegistryRepo;

/// Maximum size for the value argument of a sampled operation
pub const VALUE_MAX_SIZE: usize = 10_000;

/// Path to regression test inputs relative to the crate root
pub const REGRESSION_INPUTS_DIR: &str = "tests/inputs";

/// Path to regression test expected outputs relative to the crate root
pub const REGRESSION_EXPECTED_DIR: &str = "tests/expected";

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

#[derive(Debug, Clone)]
pub enum OperationView {
    Database(DatabaseOperationView),
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase,
    MoveDatabase,
    ClearDatabase,
}

fn make_database_operation(
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

fn key_strategy() -> impl Strategy<Value = Key> {
    proptest::collection::vec(any::<u8>(), 1usize..=KEY_MAX_SIZE)
        .prop_map(|bytes| Key::new(&bytes).expect("The size is less than KEY_MAX_SIZE"))
}

fn value_strategy() -> impl Strategy<Value = Bytes> {
    // Bias towards lengths that fit within `MAX_FILE_CHUNK_SIZE` so most
    // sampled operations exercise the success path, while also producing
    // some oversized values.
    prop_oneof![
        9 => proptest::collection::vec(any::<u8>(), 1usize..=MAX_FILE_CHUNK_SIZE),
        1 => proptest::collection::vec(any::<u8>(), (MAX_FILE_CHUNK_SIZE + 1)..=VALUE_MAX_SIZE),
    ]
    .prop_map(Bytes::from)
}

fn database_operation_view_strategy() -> impl Strategy<Value = DatabaseOperationView> {
    let set = (any::<Index>(), any::<Index>()).prop_map(|(k, v)| DatabaseOperationView::Set(k, v));

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

pub fn database_operations_strategy(
    length: impl Strategy<Value = usize>,
) -> impl Strategy<Value = (Vec<Key>, Vec<Bytes>, Vec<DatabaseOperationView>)> {
    length.prop_flat_map(|length| {
        let count = length.div_ceil(10);
        (
            proptest::collection::vec(key_strategy(), count),
            proptest::collection::vec(value_strategy(), count),
            proptest::collection::vec(database_operation_view_strategy(), length),
        )
    })
}

/// Produces `Some(CommitCheckoutRoundtrip)` with the given probability, `None` otherwise.
fn maybe_roundtrip_strategy(prob: f32) -> impl Strategy<Value = Option<DatabaseOperationView>> {
    assert!(
        (0.0..=1.0).contains(&prob),
        "expected a probability, got {prob}"
    );
    let (yes, no) = proptest::strategy::float_to_weight(prob.into());
    prop_oneof![
        yes => Just(Some(DatabaseOperationView::CommitCheckoutRoundtrip)),
        no => Just(None),
    ]
}

/// Like [`database_operations_strategy`] but produces two operation vectors sharing
/// identical base operations, with independently sampled
/// [`DatabaseOperationView::CommitCheckoutRoundtrip`]. Intended to check that 2 test runs
/// with differently-placed commit - checkout roundtrips are observationally equivalent.
pub fn database_operations_commit_checkout_strategy(
    length: impl Strategy<Value = usize>,
    roundtrip_probability: f32,
) -> impl Strategy<
    Value = (
        Vec<Key>,
        Vec<Bytes>,
        Vec<DatabaseOperationView>,
        Vec<DatabaseOperationView>,
    ),
> {
    length.prop_flat_map(move |length| {
        let count = length.div_ceil(10);
        (
            proptest::collection::vec(key_strategy(), count),
            proptest::collection::vec(value_strategy(), count),
            proptest::collection::vec(
                (
                    maybe_roundtrip_strategy(roundtrip_probability),
                    maybe_roundtrip_strategy(roundtrip_probability),
                    database_operation_view_strategy(),
                ),
                length,
            ),
        )
            .prop_map(|(keys, values, ops)| {
                let mut ops_a = Vec::with_capacity(ops.len() * 2);
                let mut ops_b = Vec::with_capacity(ops.len() * 2);
                for (pre_a, pre_b, op) in ops {
                    if let Some(r) = pre_a {
                        ops_a.push(r);
                    }
                    if let Some(r) = pre_b {
                        ops_b.push(r);
                    }
                    ops_a.push(op.clone());
                    ops_b.push(op);
                }
                (keys, values, ops_a, ops_b)
            })
    })
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

#[derive(Clone, Debug, Default)]
struct DatabaseModel {
    data: HashMap<Key, Bytes>,
    last: Option<(Hash, HashMap<Key, Bytes>)>,
    ambiguous_hash: bool,
}

fn grow_registry<KV>(registry: &mut Registry<KV, Normal>, registry_model: &mut Vec<DatabaseModel>)
where
    KV: BackgroundPersistentKeyValueStore,
    KV::Repo: RegistryRepo,
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

    if registry_model.is_empty() {
        registry_model.resize(1, Default::default());
    } else {
        registry_model.push(registry_model[registry_model.len() - 1].clone());
    }
}

/// Get the index of the current database, growing the registry until it has a valid index.
fn get_index<KV>(
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
        grow_registry(registry, registry_model);
        get_index(registry, registry_model)
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
trait DatabaseOps<KV: BackgroundPersistentKeyValueStore>: Sized {
    fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error>;

    fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error>;

    fn delete(&mut self, key: Key) -> Result<(), OperationalError>;

    fn read(&self, key: &Key, offset: usize, output: &mut [u8]) -> Result<usize, Error>;

    fn exists(&self, key: &Key) -> Result<bool, Error>;

    fn value_length(&self, key: &Key) -> Result<usize, Error>;

    fn hash(&self) -> Result<Hash, OperationalError>;

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

impl<KV: BackgroundPersistentKeyValueStore> DatabaseOps<KV> for Database<KV, Normal> {
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

#[cfg(test)]
impl<KV: BackgroundPersistentKeyValueStore> DatabaseOps<KV> for TracedDatabase<KV, Normal> {
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

fn apply_database_operation<KV, D>(
    database: &mut D,
    model: &mut DatabaseModel,
    op: DatabaseOperation,
    handle: &Handle,
    repo: &KV::Repo,
    checkout_candidates: &mut HashMap<Hash, bool>,
) where
    KV: BackgroundPersistentKeyValueStore,
    D: DatabaseOps<KV>,
{
    match op {
        DatabaseOperation::Set(key, bytes) => {
            let data = Bytes::copy_from_slice(&bytes);
            let result = database.set(key.clone(), data);

            if bytes.len() <= MAX_FILE_CHUNK_SIZE {
                assert!(
                    result.is_ok(),
                    "Set should have succeeded but failed: {:?}",
                    result.err()
                );

                model.data.insert(key, bytes.clone());
            } else {
                assert!(result.is_err(), "Set should have failed but succeeded");
            }
        }
        DatabaseOperation::Write(key, offset, bytes) => {
            let data = Bytes::copy_from_slice(&bytes);
            let result = database.write(key.clone(), offset, data);

            let should_succeed = if let Some(map_value) = model.data.get_mut(&key) {
                if offset > map_value.len()
                    || offset.checked_add(bytes.len()).is_none()
                    || bytes.len() > MAX_FILE_CHUNK_SIZE
                {
                    false
                } else {
                    update_value(map_value, offset, bytes);
                    true
                }
            } else if offset > 0 || bytes.len() > MAX_FILE_CHUNK_SIZE {
                false
            } else {
                model.data.insert(key, bytes);
                true
            };

            if should_succeed {
                assert!(
                    result.is_ok(),
                    "Write should have succeeded but failed: {:?}",
                    result.err()
                );
            } else {
                assert!(result.is_err(), "Write should have failed but succeeded");
            }
        }
        DatabaseOperation::Read(key, offset, len) => {
            let mut database_value = vec![0; len];

            let mut cursor = 0;
            let mut result = database.read(&key, offset + cursor, &mut database_value[cursor..]);

            while let Ok(read) = result {
                if read == 0 {
                    break;
                }
                cursor += read;
                result = database.read(&key, offset + cursor, &mut database_value[cursor..])
            }

            if let Some(map_value) = model.data.get(&key) {
                if offset > map_value.len() || len > MAX_FILE_CHUNK_SIZE {
                    assert!(result.is_err());
                } else {
                    let expected_len = std::cmp::min(len, map_value.len() - offset);
                    assert!(cursor >= expected_len);
                    assert_eq!(
                        &database_value[..expected_len],
                        &map_value[offset..offset + expected_len]
                    );
                }
            } else {
                assert!(result.is_err());
            }
        }
        DatabaseOperation::Delete(key) => {
            // The hash of the `Database` can differ even if the key-value pairs stored are
            // the same, because deletion and reinsertion can cause the shape of the AVL
            // tree to change.
            let deleted = model.data.remove(&key).is_some();
            if deleted {
                model.ambiguous_hash = true;
            }

            database.delete(key).expect("Deleting should succeed");
        }
        DatabaseOperation::Exists(key) => {
            let in_database = database.exists(&key).expect("Writing should succeed");
            let in_map = model.data.contains_key(&key);
            assert_eq!(in_database, in_map);
        }
        DatabaseOperation::ValueLength(key) => {
            let database_length = database.value_length(&key);
            let map_value = model.data.get(&key);

            match (database_length, map_value) {
                (Ok(database_length), Some(map_value)) => {
                    assert_eq!(database_length, map_value.len())
                }
                (Err(_), None) => (),
                _ => panic!("The value exists in one map but not the other"),
            }
        }
        DatabaseOperation::Hash => {
            let new_digest = database.hash().expect("Hash should succeed");

            if let (Some((old_digest, old_map)), false) = (&model.last, &model.ambiguous_hash) {
                assert_eq!(new_digest == *old_digest, model.data == *old_map);
            }

            model.last = Some((new_digest, model.data.clone()));

            checkout_candidates.entry(new_digest).or_insert(false);
        }
        DatabaseOperation::Commit => {
            let commit_id = database.commit(repo).expect("Committing should succeed");
            checkout_candidates.insert(*commit_id.as_hash(), true);
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
        }
        DatabaseOperation::CommitCheckoutRoundtrip => {
            let commit_id = database
                .commit_checkout_roundtrip(handle, repo)
                .expect("Commit-checkout roundtrip should succeed");
            // Register the resulting commit so a later `Checkout` operation
            // does not see an unexpected success against this hash.
            checkout_candidates.insert(*commit_id.as_hash(), true);
        }
    }
}

pub fn run_operations<KV>(repo: KV::Repo, operations: Vec<Operation>)
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
        match operation {
            Operation::Database(DatabaseOperation::Hash) => {
                let index = get_index(&mut registry, &mut registry_model);

                let new_digest = registry
                    .database(index)
                    .expect("The index is in bounds")
                    .hash()
                    .expect("Hash should succeed");

                if let (Some((old_digest, old_map)), false) = (
                    &registry_model[index].last,
                    &registry_model[index].ambiguous_hash,
                ) {
                    assert_eq!(
                        new_digest == *old_digest,
                        registry_model[index].data == *old_map
                    );
                }

                registry_model[index].last = Some((new_digest, registry_model[index].data.clone()));

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
                let index = get_index(&mut registry, &mut registry_model);
                let handle = registry.handle().clone();
                apply_database_operation::<KV, _>(
                    registry
                        .database_mut(index)
                        .expect("The index is in bounds"),
                    &mut registry_model[index],
                    op,
                    &handle,
                    &checkout_repo,
                    &mut checkout_candidates,
                );
            }
            Operation::GrowRegistry => grow_registry(&mut registry, &mut registry_model),
            Operation::ShrinkRegistry => {
                // Make sure there's a database to drop
                if registry.is_empty() {
                    grow_registry(&mut registry, &mut registry_model);
                };

                let new_size = registry.len().saturating_sub(1);
                registry
                    .resize_tick(new_size)
                    .expect("Resizing the registry should succeed");

                registry_model.truncate(new_size);
            }
            Operation::ClearDatabase => {
                let index = get_index(&mut registry, &mut registry_model);

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
                        grow_registry(&mut registry, &mut registry_model);
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
                        grow_registry(&mut registry, &mut registry_model);
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
        apply_database_operation::<KV, _>(
            &mut database,
            &mut model,
            op,
            handle,
            repo,
            &mut checkout_candidates,
        );
    }

    database.into_trace()
}

/// Apply `operation` to a traced `database`, recording its [`TraceEntry`].
///
/// Returns `true` if the operation was a provable step.
#[cfg(test)]
fn apply_step<KV: BackgroundKeyValueStore, M: DatabaseMode>(
    database: &mut TracedDatabase<KV, M>,
    operation: &DatabaseOperation,
) -> Result<bool, OperationalError> {
    fn result_operational<T>(result: Result<T, Error>) -> Result<(), OperationalError> {
        match result {
            Ok(_) | Err(Error::InvalidArgument(_)) => Ok(()),
            Err(Error::Operational(e)) => Err(e),
        }
    }

    match operation {
        DatabaseOperation::Set(key, data) => {
            result_operational(database.set(key.clone(), data.clone()))?;
        }
        DatabaseOperation::Write(key, offset, data) => {
            result_operational(database.write(key.clone(), *offset, data.clone()))?;
        }
        DatabaseOperation::Read(key, offset, len) => {
            result_operational(database.read_bytes(key, *offset, *len))?;
        }
        DatabaseOperation::Delete(key) => {
            database.delete(key.clone())?;
        }
        DatabaseOperation::Exists(key) => {
            result_operational(database.exists(key))?;
        }
        DatabaseOperation::ValueLength(key) => {
            result_operational(database.value_length(key))?;
        }
        DatabaseOperation::Hash => {
            database.hash()?;
        }
        DatabaseOperation::Commit
        | DatabaseOperation::Checkout
        | DatabaseOperation::CommitCheckoutRoundtrip => {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Generate and verify a proof for a single [`DatabaseOperation`] applied to `database`
#[cfg(test)]
fn prove_and_verify_operation<KV: BackgroundKeyValueStore>(
    database: &TracedDatabase<KV, Normal>,
    operation: &DatabaseOperation,
) {
    let pre_root_hash = Hash::from_foldable(database);

    // Produce a proof and record the trace of applying `operation`
    let mut prover = database
        .try_start_proof()
        .expect("starting a proof should succeed");

    // Nothing to record or compare if the step was not provable
    if !apply_step(&mut prover, operation).expect("applying a step should succeed") {
        return;
    }

    let post_root_hash = Hash::from_foldable(&prover);
    let proof = MerkleProof::from_foldable(&prover);
    let proof_step_trace = prover.into_trace();
    let proof_bytes = serialise(&proof).expect("serialising the proof should succeed");

    // Construct the Verify-mode database from the proof and verify
    let mut verify_db = TracedDatabase::from(
        Database::<KV, Verify>::from_proof(proof).expect("proof should be valid"),
    );
    assert_eq!(
        Hash::from_foldable(&verify_db),
        pre_root_hash,
        "the proof must reconstruct the pre-operation root hash"
    );
    apply_step(&mut verify_db, operation).expect("applying a step should succeed");
    let verify_post_root_hash = Hash::from_foldable(&verify_db);
    let verify_step_trace = verify_db.into_trace();

    assert_eq!(
        verify_step_trace, proof_step_trace,
        "Prove- and Verify-mode execution traces should match"
    );
    assert_eq!(
        verify_post_root_hash, post_root_hash,
        "Prove- and Verify-mode root hashes should match"
    );

    database.record_proof(operation.clone(), proof_bytes)
}

/// Like [`run_database_operations`], but additionally generates and verifies a proof for
/// every supported operation, recording the serialised proof in the trace.
#[cfg(test)]
pub(crate) fn run_and_prove_database_operations<KV>(
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

    for operation in operations {
        // Provable operations are proven over their pre-operation state, so prove before applying.
        prove_and_verify_operation::<KV>(&database, &operation);

        apply_database_operation::<KV, _>(
            &mut database,
            &mut model,
            operation,
            handle,
            repo,
            &mut checkout_candidates,
        );
    }

    database.into_trace()
}
