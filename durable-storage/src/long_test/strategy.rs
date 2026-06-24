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
use proptest::sample::select;
use proptest::strategy::BoxedStrategy;
use proptest::strategy::Union;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

use crate::key::Key;
use crate::long_test::model::KeyPools;
use crate::test_helpers::database::DatabaseOperation;
use crate::test_helpers::database::VALUE_MAX_SIZE;
use crate::test_helpers::database::key_strategy;
use crate::test_helpers::database::value_strategy;

/// A key strategy that blends fresh random keys with samples drawn from the
/// model's hot, existing, and recently-deleted pools.
fn pooled_key_strategy(pools: &KeyPools) -> BoxedStrategy<Key> {
    let mut arms: Vec<(u32, BoxedStrategy<Key>)> = Vec::new();

    arms.push((90, key_strategy().boxed()));

    if !pools.hot.is_empty() {
        arms.push((5, select(pools.hot.clone()).boxed()));
    }
    if !pools.deleted.is_empty() {
        arms.push((2, select(pools.deleted.clone()).boxed()));
    }
    if !pools.existing.is_empty() {
        arms.push((3, select(pools.existing.clone()).boxed()));
    }

    Union::new_weighted(arms).boxed()
}

// Distribution is based on that of `<DatabaseOperationView as OperationView>::view_strategy`
fn database_op_strategy(pools: &KeyPools) -> BoxedStrategy<DatabaseOperation> {
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

pub fn long_test_ops_strategy(
    pools: &KeyPools,
    length: usize,
) -> impl Strategy<Value = Vec<DatabaseOperation>> + use<> {
    let length = length.max(1);
    proptest::collection::vec(database_op_strategy(pools), 1..=length)
}
