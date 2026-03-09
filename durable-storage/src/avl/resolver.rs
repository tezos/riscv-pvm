// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for resolving data from IDs of [`Tree`] and [`Node`] objects.
//!
//! [`Tree`]: crate::avl::tree::Tree
//! [`Node`]: crate::avl::node::Node

use std::sync::Arc;

use octez_riscv_data::hash::Hash;
use trait_set::trait_set;

use super::node::Node;
use super::tree::Tree;
use crate::errors::OperationalError;

/// Trait for resolving identifiers to values.
pub trait Resolver<Id, Value> {
    /// Retrieve the hash using the identifier.
    ///
    /// Depending on the implementation, this may compute or fetch the hash without resolving the
    /// full value.
    fn hash(&self, id: &Id) -> Hash;

    /// Resolve an identifier to a value.
    fn resolve<'a>(&self, id: &'a Id) -> Result<&'a Value, OperationalError>;

    /// Resolve an identifier to a mutable value.
    fn resolve_mut<'a>(&mut self, id: &'a mut Id) -> Result<&'a mut Value, OperationalError>;
}

trait_set! {
    /// Specialised [`Resolver`] for MAVL nodes
    pub trait NodeResolver<NodeId, TreeId> = Resolver<NodeId, Node<TreeId>>;

    /// Specialised [`Resolver`] for MAVL trees
    pub trait TreeResolver<NodeId, TreeId> = Resolver<TreeId, Tree<NodeId>>;

    /// Specialised [`Resolver`] for MAVL nodes and trees
    pub trait AvlResolver<NodeId, TreeId> = NodeResolver<NodeId, TreeId> + TreeResolver<NodeId, TreeId>;
}

/// ID for a node that is always present
#[derive(Debug, Clone, derive_more::From)]
#[from(Node<ArcTreeId>)]
pub struct ArcNodeId(Arc<Node<ArcTreeId>>);

/// ID for a tree that is always present
#[derive(Debug, Clone, derive_more::From, Default)]
pub struct ArcTreeId(Tree<ArcNodeId>);

/// Provide values identified by an [`Arc`].
#[derive(Clone, Debug)]
pub struct ArcResolver;

impl Resolver<ArcNodeId, Node<ArcTreeId>> for ArcResolver {
    fn hash(&self, id: &ArcNodeId) -> Hash {
        *id.0.hash(self)
    }

    fn resolve<'a>(&self, id: &'a ArcNodeId) -> Result<&'a Node<ArcTreeId>, OperationalError> {
        Ok(id.0.as_ref())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut ArcNodeId,
    ) -> Result<&'a mut Node<ArcTreeId>, OperationalError> {
        Ok(Arc::make_mut(&mut id.0))
    }
}

impl Resolver<ArcTreeId, Tree<ArcNodeId>> for ArcResolver {
    fn hash(&self, id: &ArcTreeId) -> Hash {
        id.0.hash(self)
    }

    fn resolve<'a>(&self, id: &'a ArcTreeId) -> Result<&'a Tree<ArcNodeId>, OperationalError> {
        Ok(&id.0)
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut ArcTreeId,
    ) -> Result<&'a mut Tree<ArcNodeId>, OperationalError> {
        Ok(&mut id.0)
    }
}
