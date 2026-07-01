// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Long-running property-based tests for durable storage.
//!
//! [`database`] exercises a single [`Database`]. The subject-agnostic building
//! blocks live in [`harness`].
//!
//! [`Database`]: crate::database::Database

pub mod database;
mod harness;

pub use harness::LongTestConfig;
