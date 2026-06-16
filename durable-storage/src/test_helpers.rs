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
