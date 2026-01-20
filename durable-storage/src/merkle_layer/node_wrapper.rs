// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! # Node wrapper utilities
//!
//! This module provides wrapper classes to abstract the parent-child relationship
//! between [`MavlNode`] types. The reason for this is to support multiple ways to load
//! nodes.
//!
//! ## Structure
//!
//! The [`MavlNodeWrapper`] trait defines the structure. The wrapper has an associated
//! [`MavlNodeResolver`] at [`MavlNodeWrapper::Resolver`] and there is a one to one
//! relationship between the implementors of this trait and the resolver. The role of
//! the implementors of this trait is to hold a reference to a child node of a node.
//! The traits implement both [`std::ops::Deref`] and [`std::ops::DerefMut`] because the
//! child nodes can be accessed as references.

use std::convert::Infallible;
use std::fmt::Debug;
use std::sync::Arc;

use super::node::MavlNode;
use super::node_resolver::InMemoryMavlNodeResolver;
use super::node_resolver::MavlNodeResolver;

pub(crate) trait MavlNodeWrapper: Clone + Debug + Default {
    type Resolver: MavlNodeResolver<NodeWrapper = Self>;
    type BorrowError: std::error::Error;

    fn new(value: Option<Arc<MavlNode<Self::Resolver>>>) -> Self;

    fn try_borrow(&self) -> Result<&Option<Arc<MavlNode<Self::Resolver>>>, &Self::BorrowError>;

    fn try_borrow_mut(
        &mut self,
    ) -> Result<&mut Option<Arc<MavlNode<Self::Resolver>>>, &Self::BorrowError>;
}

/// [`struct@InMemoryMavlNodeWrapper`] defines a wrapper which owns the underlying
/// data. There is no need for resolving. This is mainly used for testing purposes.
#[derive(Debug, Clone, Default)]
pub(crate) struct InMemoryMavlNodeWrapper {
    pub node: Option<Arc<MavlNode<InMemoryMavlNodeResolver>>>,
}

impl MavlNodeWrapper for InMemoryMavlNodeWrapper {
    type Resolver = InMemoryMavlNodeResolver;
    type BorrowError = Infallible;

    fn new(value: Option<Arc<MavlNode<Self::Resolver>>>) -> Self {
        Self { node: value }
    }

    fn try_borrow(&self) -> Result<&Option<Arc<MavlNode<Self::Resolver>>>, &Self::BorrowError> {
        Ok(&self.node)
    }

    fn try_borrow_mut(
        &mut self,
    ) -> Result<&mut Option<Arc<MavlNode<Self::Resolver>>>, &Self::BorrowError> {
        Ok(&mut self.node)
    }
}
