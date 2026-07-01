// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Model-guided operation strategy for the long-running [`Database`] test.
//!
//! The strategy is parameterised by a snapshot of the reference model's key
//! pools ([`KeyPools`]), so generated operations favour recently used and
//! recently deleted keys while still introducing fresh keys. Because the
//! snapshot is fixed for the duration of an epoch, this is an ordinary
//! `proptest` [`Strategy`] and retains shrinking.
//!
//! [`Database`]: crate::database::Database

use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

use super::model::KeyPools;
use crate::key::Key;
use crate::test_helpers::database::DatabaseOperation;
use crate::test_helpers::database::VALUE_MAX_SIZE;
use crate::test_helpers::database::key_strategy;
use crate::test_helpers::database::value_strategy;

/// Which of the model's key pools a sampled key should be drawn from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KeyPoolSelector {
    /// A fresh random key.
    Fresh,
    /// A recently written or read key.
    Hot,
    /// A currently present key.
    Existing,
    /// A recently deleted key.
    Deleted,
}

/// Blend weights matching the original pooled key strategy: mostly fresh keys,
/// with occasional draws from the hot, existing, and recently-deleted pools.
pub(crate) fn pool_selector_strategy() -> impl Strategy<Value = KeyPoolSelector> {
    prop_oneof![
        90 => Just(KeyPoolSelector::Fresh),
        5 => Just(KeyPoolSelector::Hot),
        3 => Just(KeyPoolSelector::Existing),
        2 => Just(KeyPoolSelector::Deleted),
    ]
}

/// Resolve a sampled `(fresh, selector, index)` triple against `pools`.
///
/// Falls back to `fresh` when the selected pool is empty, so an empty pool
/// simply behaves like a fresh key.
pub(crate) fn resolve_pooled_key(
    pools: &KeyPools,
    selector: KeyPoolSelector,
    index: proptest::sample::Index,
    fresh: Key,
) -> Key {
    let pool = match selector {
        KeyPoolSelector::Fresh => return fresh,
        KeyPoolSelector::Hot => &pools.hot,
        KeyPoolSelector::Existing => &pools.existing,
        KeyPoolSelector::Deleted => &pools.deleted,
    };
    if pool.is_empty() {
        fresh
    } else {
        pool[index.index(pool.len())].clone()
    }
}

/// A key strategy that blends fresh random keys with samples drawn from the
/// model's hot, existing, and recently-deleted pools.
fn pooled_key_strategy(pools: &KeyPools) -> BoxedStrategy<Key> {
    let pools = pools.clone();
    (
        key_strategy(),
        pool_selector_strategy(),
        any::<proptest::sample::Index>(),
    )
        .prop_map(move |(fresh, selector, index)| {
            resolve_pooled_key(&pools, selector, index, fresh)
        })
        .boxed()
}

// Distribution is based on that of `<DatabaseOperationView as OperationView>::view_strategy`
pub(crate) fn database_op_strategy(pools: &KeyPools) -> BoxedStrategy<DatabaseOperation> {
    let set = (pooled_key_strategy(pools), value_strategy())
        .prop_map(|(k, v)| DatabaseOperation::Set(k, v));

    let read = (
        pooled_key_strategy(pools),
        prop_oneof![
            5 => Just(0usize),
            4 => 1..=MAX_FILE_CHUNK_SIZE,
            1 => (MAX_FILE_CHUNK_SIZE + 1)..=VALUE_MAX_SIZE,
        ],
        prop_oneof![
            9 => 0..=MAX_FILE_CHUNK_SIZE,
            1 => (MAX_FILE_CHUNK_SIZE + 1)..=VALUE_MAX_SIZE,
        ],
    )
        .prop_map(|(k, off, len)| DatabaseOperation::Read(k, off, len));

    let write_valid = (
        pooled_key_strategy(pools),
        prop_oneof![
            2 => Just(0),
            1 => 1..=VALUE_MAX_SIZE,
        ],
        value_strategy(),
    )
        .prop_map(|(k, off, v)| DatabaseOperation::Write(k, off, v));

    let write_invalid = (
        pooled_key_strategy(pools),
        VALUE_MAX_SIZE..=usize::MAX,
        value_strategy(),
    )
        .prop_map(|(k, off, v)| DatabaseOperation::Write(k, off, v));

    prop_oneof![
        20 => set,
        20 => read,
        4 => write_valid,
        1 => write_invalid,
        10 => pooled_key_strategy(pools).prop_map(DatabaseOperation::Delete),
        10 => pooled_key_strategy(pools).prop_map(DatabaseOperation::Exists),
        5 => pooled_key_strategy(pools).prop_map(DatabaseOperation::ValueLength),
        10 => Just(DatabaseOperation::Hash),
    ]
    .boxed()
}

pub(super) fn ops_strategy(
    pools: &KeyPools,
    length: usize,
) -> impl Strategy<Value = Vec<DatabaseOperation>> + use<> {
    let length = length.max(1);
    proptest::collection::vec(database_op_strategy(pools), 1..=length)
}
