// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for resolving data from IDs of [`Tree`] and [`Node`] objects.
//!
//! [`Tree`]: crate::avl::tree::Tree
//! [`Node`]: crate::avl::node::Node

use std::sync::Arc;

use octez_riscv_data::hash::Hash;

use super::node::Node;
use crate::errors::OperationalError;

/// Trait for resolving identifiers to values.
pub trait Resolver<Id, Value> {
    /// Retrieve the hash using the identifier.
    ///
    /// This will not perform any resolution. If the identifier is for a blinded node, then the
    /// identity is the hash. In case the node is not blinded, there is no resolution needed
    /// anyway.
    fn hash<'a>(&self, id: &'a Id) -> &'a Hash;

    /// Resolve an identifier to a value.
    fn resolve<'a>(&self, id: &'a Id) -> Result<&'a Value, OperationalError>;

    /// Resolve an identifier to a mutable value.
    fn resolve_mut<'a>(&mut self, id: &'a mut Id) -> Result<&'a mut Value, OperationalError>;
}

/// ID for a node that is always present
#[derive(Debug, Clone, derive_more::From)]
#[from(Node<Self>)]
pub struct ArcNodeId(Arc<Node<Self>>);

/// Provide values identified by an [`Arc`].
#[derive(Clone, Debug)]
pub struct ArcResolver;

impl Resolver<ArcNodeId, Node<ArcNodeId>> for ArcResolver {
    fn hash<'a>(&self, id: &'a ArcNodeId) -> &'a Hash {
        id.0.hash(self)
    }

    fn resolve<'a>(&self, id: &'a ArcNodeId) -> Result<&'a Node<ArcNodeId>, OperationalError> {
        Ok(id.0.as_ref())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut ArcNodeId,
    ) -> Result<&'a mut Node<ArcNodeId>, OperationalError> {
        Ok(Arc::make_mut(&mut id.0))
    }
}
