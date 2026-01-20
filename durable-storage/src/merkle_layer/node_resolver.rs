// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! # Node resolver utilities
//!
//! This module provides Resolver classes for resolving potentially blind nodes. In
//! memory and lazy mode is supported at the moment.
//!
//! ## Structure
//!
//! The [`MavlNodeResolver`] trait defines the structure. The resolver trait has an
//! associated [`MavlNodeWrapper`] type at [`MavlNodeResolver::NodeWrapper`]. A one to
//! one relationship between wrapper and resolver implementations are enforced on the
//! trait level. The trait supports a single method [`MavlNodeResolver::resolve`] which
//! is used for resolving child nodes which hasn't been resolved yet.

use std::fmt::Debug;

use super::node_wrapper::InMemoryMavlNodeWrapper;
use super::node_wrapper::MavlNodeWrapper;

pub(crate) trait MavlNodeResolver: Clone + Debug + Default {
    type NodeWrapper: MavlNodeWrapper<Resolver = Self>;

    fn resolve(&self, node_wrapper: &Self::NodeWrapper);
}

/// [`struct@InMemoryMavlNodeResolver`] is a blank resolver since the associated wrapper
/// already owns the underlying data, so there is nothing to resolve here.
#[derive(Debug, Default, Clone)]
pub(crate) struct InMemoryMavlNodeResolver {}

impl MavlNodeResolver for InMemoryMavlNodeResolver {
    type NodeWrapper = InMemoryMavlNodeWrapper;

    fn resolve(&self, _node_wrapper: &Self::NodeWrapper) {}
}
