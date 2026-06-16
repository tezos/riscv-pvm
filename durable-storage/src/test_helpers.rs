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

// Cross-cutting entry points used directly via `test_helpers::…`; everything
// else is reached through its owning submodule.
#[cfg(any(test, rocksdb_test_utils))]
pub(crate) use database::prove_and_verify_operation;
pub use registry::run_operations;
