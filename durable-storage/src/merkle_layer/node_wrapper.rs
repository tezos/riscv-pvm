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
use std::sync::OnceLock;

use octez_riscv_data::hash::Hash;

use super::node::MavlNode;
use super::node_resolver::InMemoryMavlNodeResolver;
use super::node_resolver::LazyMavlNodeResolver;
use super::node_resolver::MavlNodeResolver;
use super::node_resolver_error::MavlNodeResolverError;

pub(crate) trait MavlNodeWrapper: Clone + Debug {
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

/// [`struct@LazyMavlNodeWrapper`] can store the hash of the key and once succesfully
/// resolved it stores the node itself, otherwise it will store the cause of the error
/// that happened during resolving the node.
#[derive(Clone, Debug)]
pub(crate) struct LazyMavlNodeWrapper {
    pub node:
        OnceLock<Result<Option<Arc<MavlNode<LazyMavlNodeResolver>>>, Arc<MavlNodeResolverError>>>,
    pub commited_hash: Option<Hash>,
}

impl Default for LazyMavlNodeWrapper {
    fn default() -> Self {
        let node = OnceLock::new();
        node.set(Ok(None)).expect("This operation will not fail");
        Self {
            node,
            commited_hash: None,
        }
    }
}

impl LazyMavlNodeWrapper {
    pub(crate) fn blinded(commited_hash: Hash) -> Self {
        Self {
            node: OnceLock::new(),
            commited_hash: Some(commited_hash),
        }
    }
}

impl MavlNodeWrapper for LazyMavlNodeWrapper {
    type Resolver = LazyMavlNodeResolver;
    type BorrowError = MavlNodeResolverError;

    fn new(value: Option<Arc<MavlNode<Self::Resolver>>>) -> Self {
        let node = OnceLock::new();
        node.set(Ok(value)).expect("This can't fail");
        Self {
            node,
            commited_hash: None,
        }
    }

    fn try_borrow(&self) -> Result<&Option<Arc<MavlNode<Self::Resolver>>>, &Self::BorrowError> {
        match self.node.get() {
            Some(ref result) => match result {
                Ok(node) => Ok(node),
                Err(err) => Err(err),
            },
            None => Err(&MavlNodeResolverError::Unresolved),
        }
    }

    fn try_borrow_mut(
        &mut self,
    ) -> Result<&mut Option<Arc<MavlNode<Self::Resolver>>>, &Self::BorrowError> {
        match self.node.get_mut() {
            Some(result) => match result {
                Ok(node) => Ok(node),
                Err(err) => Err(err),
            },
            None => Err(&MavlNodeResolverError::Unresolved),
        }
    }
}
