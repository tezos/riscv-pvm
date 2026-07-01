// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Long-running property-based tests for durable storage.
//!
//! [`database`] exercises a single [`Database`]; [`registry`] exercises a
//! [`Registry`] of several databases. Both drivers share the building blocks
//! in [`harness`].
//!
//! [`Database`]: crate::database::Database
//! [`Registry`]: crate::registry::Registry

pub mod database;
mod harness;
pub mod registry;

pub use harness::LongTestConfig;
