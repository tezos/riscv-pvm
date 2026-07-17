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
use octez_riscv_data::hash::Hash;
use proptest::prelude::*;

use crate::errors::Error;
use crate::errors::OperationalError;
use crate::key::Key;

/// Path to regression test inputs relative to the crate root
#[cfg(test)]
pub(crate) const REGRESSION_INPUTS_DIR: &str = "tests/inputs";

/// Path to regression test expected outputs relative to the crate root
#[cfg(test)]
pub(crate) const REGRESSION_EXPECTED_DIR: &str = "tests/expected";

/// The observable outcome of applying a single provable operation.
///
/// Captured so the result computed in [`Normal`], [`Prove`] and [`Verify`] mode can be
/// compared for equality — the prove/verify harnesses otherwise only constrain state
/// (root hashes), never the value an operation returns.
///
/// The representation is deliberately mode- and backend-independent: errors are stringified
/// (matching the `{:?}` convention used by the operation traces) and byte payloads are owned,
/// so the type implements [`PartialEq`], [`Serialize`] and [`Deserialize`].
///
/// [`Normal`]: octez_riscv_data::mode::Normal
/// [`Prove`]: octez_riscv_data::mode::Prove
/// [`Verify`]: octez_riscv_data::mode::Verify
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[serde_with::serde_as]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) enum StepOutcome {
    /// Result of `set` or `delete`.
    Unit(Result<(), String>),
    /// Result of `write` (bytes written) or `value_length`.
    Length(Result<usize, String>),
    /// Result of `read`: the bytes actually returned.
    Read(
        #[serde_as(as = "Result<serde_with::hex::Hex, serde_with::Same>")] Result<Vec<u8>, String>,
    ),
    /// Result of `exists`.
    Exists(Result<bool, String>),
    /// Result of `hash`: the root hash.
    Hash(Result<Hash, String>),
}

/// Capture the observable outcome of an [`Error`]-returning operation.
///
/// Operational errors are propagated: they indicate a harness or implementation bug rather
/// than an observable result. `Ok` values and invalid-argument errors are the operation's
/// observable outcome and are captured via `wrap` (e.g. [`StepOutcome::Unit`]).
pub(crate) fn outcome_from_value<T>(
    result: Result<T, Error>,
    wrap: impl FnOnce(Result<T, String>) -> StepOutcome,
) -> Result<StepOutcome, OperationalError> {
    match result {
        Ok(value) => Ok(wrap(Ok(value))),
        Err(Error::InvalidArgument(error)) => Ok(wrap(Err(format!("{error:?}")))),
        Err(Error::Operational(error)) => Err(error),
    }
}

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
