// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

#![cfg(test_utils)]

//! Shared utilities for end to end durable storage property-based tests
//!
//! Split into [`database`] (single-[`Database`] operations and
//! utilities shared between both) and [`registry`]
//! ([`Registry`] operations).
//!
//! [`Database`]: crate::database::Database
//! [`Registry`]: crate::registry::Registry

pub mod database;
pub mod registry;

use bytes::Bytes;
use database::key_strategy;
use database::value_strategy;
use proptest::prelude::*;

use crate::key::Key;

/// A proptest-sampleable view of an operation which can be applied to an NDS component
pub trait OperationView: Clone + std::fmt::Debug + 'static {
    /// Strategy for a single operation, excluding the commit/checkout-roundtrip variant.
    fn strategy() -> impl Strategy<Value = Self>;

    /// Keys, values, and a base sequence of `length` operations
    fn operations_strategy(
        length: impl Strategy<Value = usize>,
    ) -> impl Strategy<Value = (Vec<Key>, Vec<Bytes>, Vec<Self>)> {
        length.prop_flat_map(|length| {
            let count = length.div_ceil(10);
            (
                proptest::collection::vec(key_strategy(), count),
                proptest::collection::vec(value_strategy(), count),
                proptest::collection::vec(Self::strategy(), length),
            )
        })
    }

    /// The commit/checkout-roundtrip variant.
    fn roundtrip() -> Self;

    /// `Some` commit/checkout roundtrip with the given probability, `None` otherwise.
    fn maybe_roundtrip_strategy(prob: f32) -> impl Strategy<Value = Option<Self>> {
        assert!(
            (0.0..=1.0).contains(&prob),
            "expected a probability, got {prob}"
        );
        let (yes, no) = proptest::strategy::float_to_weight(prob.into());
        prop_oneof![
            yes => Just(Some(Self::roundtrip())),
            no => Just(None),
        ]
    }

    /// Two operation sequences sharing identical base operations, with independently
    /// sampled commit/checkout roundtrips. Intended to check that 2 runs with
    /// differently-placed roundtrips are observationally equivalent.
    fn operations_commit_checkout_strategy(
        length: impl Strategy<Value = usize>,
        roundtrip_probability: f32,
    ) -> impl Strategy<Value = (Vec<Key>, Vec<Bytes>, Vec<Self>, Vec<Self>)> {
        length.prop_flat_map(move |length| {
            let count = length.div_ceil(10);
            (
                proptest::collection::vec(key_strategy(), count),
                proptest::collection::vec(value_strategy(), count),
                proptest::collection::vec(
                    (
                        Self::maybe_roundtrip_strategy(roundtrip_probability),
                        Self::maybe_roundtrip_strategy(roundtrip_probability),
                        Self::strategy(),
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
}
