// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Shared test helpers for prove/verify proof round-trip tests.
//!
//! Hosts `apply_*` helpers that consume the [`Operation`] enum defined in [`test_helpers`].
//! Lives behind `#[cfg(test)]` (rather than the `unstable-test-utils` feature like
//! [`test_helpers`]) because it depends on the test-only [`TracedDatabase`] wrapper.
//!
//! [`test_helpers`]: crate::test_helpers
//! [`Operation`]: crate::test_helpers::Operation
//! [`TracedDatabase`]: crate::database::traced_database::TracedDatabase

use octez_riscv_data::components::bytes::BytesMode;

use crate::database::DatabaseMode;
use crate::database::traced_database::TracedDatabase;
use crate::merkle_layer::MerkleLayer;
use crate::merkle_layer::MerkleLayerMode;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::storage::KeyValueStore;
use crate::test_helpers::Operation;

/// Derive a setup-time data length in the range `[1, 8]` from key bytes. Used to seed setup
/// values with varying length so proof tests cover non-uniform initial state.
pub(crate) fn setup_data_len(key_bytes: &[u8; 2]) -> usize {
    (key_bytes[0] as usize % 8) + 1
}

/// Apply `op` against a [`MerkleLayer`] in any mode that implements [`MerkleLayerMode`] and
/// [`BytesMode`]. Returns `Some(bytes)` for [`Operation::Read`] and `None` for mutating ops.
///
/// Only the prove/verify-relevant subset of [`Operation`] is supported — variants that target
/// registry- or commit-level state are unreachable in a single proof step. Read clamps to the
/// existing data length and returns zero bytes for a missing key.
pub(crate) fn apply_to_merkle_layer<KV, M>(
    ml: &mut MerkleLayer<KV, M>,
    op: &Operation,
) -> Option<Vec<u8>>
where
    KV: KeyValueStore,
    M: MerkleLayerMode + BytesMode,
{
    match op {
        Operation::Set(key, data) => {
            ml.set(key, data).expect("set should succeed");
            None
        }
        Operation::Delete(key) => {
            ml.delete(key).expect("delete should succeed");
            None
        }
        Operation::Write(key, offset, data) => {
            ml.write(key, *offset, data).expect("write should succeed");
            None
        }
        Operation::Read(key, offset, count) => {
            let mut buf = vec![0u8; *count];
            let n = ml
                .get(key)
                .expect("get should succeed")
                .map_or(0, |data| data.read(*offset, &mut buf));
            buf.truncate(n);
            Some(buf)
        }
        Operation::Exists(_)
        | Operation::ValueLength(_)
        | Operation::Hash
        | Operation::Commit
        | Operation::Checkout
        | Operation::GrowRegistry
        | Operation::ShrinkRegistry
        | Operation::CopyDatabase
        | Operation::MoveDatabase
        | Operation::ClearDatabase => {
            unimplemented!("{op:?} is not supported when applying to a MerkleLayer")
        }
    }
}

/// Apply `op` against a [`TracedDatabase`] in any mode that implements [`DatabaseMode`].
/// Returns `Some(bytes)` for [`Operation::Read`] and `None` for mutating ops.
///
/// Only the prove/verify-relevant subset of [`Operation`] is supported.
/// `Database::read_bytes` requires the key to exist and the offset to be `<= value_length`;
/// callers must constrain `Read` ops accordingly.
pub(crate) fn apply_to_database<KV, M>(
    db: &mut TracedDatabase<KV, M>,
    op: &Operation,
) -> Option<Vec<u8>>
where
    KV: BackgroundKeyValueStore,
    M: DatabaseMode,
{
    match op {
        Operation::Set(key, data) => {
            db.set(key.clone(), data.clone())
                .expect("set should succeed");
            None
        }
        Operation::Delete(key) => {
            db.delete(key.clone()).expect("delete should succeed");
            None
        }
        Operation::Write(key, offset, data) => {
            db.write(key.clone(), *offset, data.clone())
                .expect("write should succeed");
            None
        }
        Operation::Read(key, offset, max_bytes) => Some(
            db.read_bytes(key, *offset, *max_bytes)
                .expect("read_bytes should succeed"),
        ),
        Operation::Exists(_)
        | Operation::ValueLength(_)
        | Operation::Hash
        | Operation::Commit
        | Operation::Checkout
        | Operation::GrowRegistry
        | Operation::ShrinkRegistry
        | Operation::CopyDatabase
        | Operation::MoveDatabase
        | Operation::ClearDatabase => {
            unimplemented!("{op:?} is not supported when applying to a TracedDatabase")
        }
    }
}
