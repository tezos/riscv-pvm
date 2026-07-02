// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Model-guided operation strategy for the long-running [`Registry`] test.
//!
//! [`Registry`]: crate::registry::Registry

use proptest::prelude::*;
use proptest::sample::Index;

use crate::long_test::database::model::KeyPools;
use crate::long_test::database::strategy::KeyPoolSelector;
use crate::long_test::database::strategy::database_op_strategy;
use crate::long_test::database::strategy::pool_selector_strategy;
use crate::long_test::database::strategy::resolve_pooled_key;
use crate::test_helpers::database::DatabaseOperation;
use crate::test_helpers::registry::RegistryOperation;

/// One sampled operation before its indices and key are resolved.
#[derive(Clone, Debug)]
enum Draw {
    /// A database operation carrying its target-db index, a fresh-keyed
    /// operation, and a key selector `(selector, key_index)` used to redraw the
    /// key from the resolved database's pool.
    Database {
        db: Index,
        op: DatabaseOperation,
        selector: KeyPoolSelector,
        key_index: Index,
    },
    Grow,
    Shrink,
    Copy {
        src: Index,
        dst: Index,
    },
    Move {
        src: Index,
        dst: Index,
    },
    Clear {
        index: Index,
    },
}

/// A single skeleton draw. Weights mirror `<RegistryOperationView as OperationView>::strategy`.
///
/// With `keep_stable_size`, the Grow and Shrink weights are set to `permanent : (permanent + 1)`.
/// The remaining weights are scaled by `2 * permanent + 1` so their relative frequencies match the
/// default strategy and only the Grow/Shrink split changes.
///
/// This means that the number of databases follows a biased random walk with `permanent` as a
/// lower bound (we won't allow the registry to shrink below that). The stationary distribution for
/// this random walk is geometric with parameter `permanent / permanent + 1`. Therefore in the long
/// run, the mean number of databases is `2 * permanent`, while the run will typically spend rather
/// more than half its time with fewer than that---the distribution is skewed towards lower values.
fn draw_strategy(permanent: usize, keep_stable_size: bool) -> impl Strategy<Value = Draw> {
    let (p_database, p_grow, p_shrink, p_copy, p_move, p_clear) = if keep_stable_size {
        let permanent = permanent as u32;
        let scale = 2 * permanent + 1;
        (
            88 * scale,
            6 * permanent,
            6 * (permanent + 1),
            3 * scale,
            2 * scale,
            scale,
        )
    } else {
        (88, 4, 2, 3, 2, 1)
    };

    prop_oneof![
        p_database => (
            any::<Index>(),
            database_op_strategy(&KeyPools::default()),
            pool_selector_strategy(),
            any::<Index>(),
        )
            .prop_map(|(db, op, selector, key_index)| Draw::Database {
                db,
                op,
                selector,
                key_index,
            }),
        p_grow => Just(Draw::Grow),
        p_shrink => Just(Draw::Shrink),
        p_copy => (any::<Index>(), any::<Index>()).prop_map(|(src, dst)| Draw::Copy { src, dst }),
        p_move => (any::<Index>(), any::<Index>()).prop_map(|(src, dst)| Draw::Move { src, dst }),
        p_clear => any::<Index>().prop_map(|index| Draw::Clear { index }),
    ]
}

/// Replace `op`'s key with one drawn from `pools` when the selector picks a
/// non-empty pool; otherwise keep the operation's original (fresh) key.
/// Operations without a key are returned unchanged.
fn with_resolved_key(
    op: DatabaseOperation,
    pools: &KeyPools,
    selector: KeyPoolSelector,
    key_index: Index,
) -> DatabaseOperation {
    let resolve = |k| resolve_pooled_key(pools, selector, key_index, k);
    match op {
        DatabaseOperation::Set(k, v) => DatabaseOperation::Set(resolve(k), v),
        DatabaseOperation::Write(k, off, v) => DatabaseOperation::Write(resolve(k), off, v),
        DatabaseOperation::Read(k, off, len) => DatabaseOperation::Read(resolve(k), off, len),
        DatabaseOperation::Delete(k) => DatabaseOperation::Delete(resolve(k)),
        DatabaseOperation::Exists(k) => DatabaseOperation::Exists(resolve(k)),
        DatabaseOperation::ValueLength(k) => DatabaseOperation::ValueLength(resolve(k)),
        other => other,
    }
}

/// Resolve draws into concrete [`RegistryOperation`]s.
///
/// `pools` is the per-database pool snapshot at the start of the sequence; its
/// length is the initial registry length. `permanent` databases occupy indices
/// `[0, permanent)`.
fn resolve(pools: &[KeyPools], permanent: usize, draws: Vec<Draw>) -> Vec<RegistryOperation> {
    let mut pools_sim = pools.to_vec();
    let mut len = pools_sim.len();
    let mut out = Vec::with_capacity(draws.len());

    for draw in draws {
        match draw {
            Draw::Database {
                db,
                op,
                selector,
                key_index,
            } => {
                let i = db.index(len);
                let op = with_resolved_key(op, &pools_sim[i], selector, key_index);
                out.push(RegistryOperation::Database(i, op));
            }
            Draw::Grow => {
                pools_sim.push(KeyPools::default());
                len += 1;
                out.push(RegistryOperation::GrowRegistry);
            }
            Draw::Shrink => {
                // Only shrink above the permanent floor
                if len > permanent {
                    pools_sim.pop();
                    len -= 1;
                    out.push(RegistryOperation::ShrinkRegistry);
                }
            }
            Draw::Copy { src, dst } => {
                // The destination must be non-permanent.
                if len > permanent {
                    let s = src.index(len);
                    let d = permanent + dst.index(len - permanent);
                    pools_sim[d] = pools_sim[s].clone();
                    out.push(RegistryOperation::CopyDatabase(s, d));
                }
            }
            Draw::Move { src, dst } => {
                // Both endpoints must be non-permanent (a move empties its source).
                if len > permanent {
                    let s = permanent + src.index(len - permanent);
                    let d = permanent + dst.index(len - permanent);
                    if s != d {
                        pools_sim[d] = std::mem::take(&mut pools_sim[s]);
                    }
                    out.push(RegistryOperation::MoveDatabase(s, d));
                }
            }
            Draw::Clear { index } => {
                if len > permanent {
                    let i = permanent + index.index(len - permanent);
                    pools_sim[i] = KeyPools::default();
                    out.push(RegistryOperation::ClearDatabase(i));
                }
            }
        }
    }

    out
}

/// A model-guided strategy for a sequence of up to `length` registry operations,
/// given the per-database key `pools` snapshot and the `permanent` count.
pub(super) fn ops_strategy(
    pools: &[KeyPools],
    permanent: usize,
    keep_stable_size: bool,
    length: usize,
) -> impl Strategy<Value = Vec<RegistryOperation>> + use<> {
    let length = length.max(1);
    let pools = pools.to_vec();
    proptest::collection::vec(draw_strategy(permanent, keep_stable_size), 1..=length)
        .prop_map(move |draws| resolve(&pools, permanent, draws))
}
