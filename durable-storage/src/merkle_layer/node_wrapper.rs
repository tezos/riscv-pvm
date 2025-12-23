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
//!
//! ## In memory wrapper
//!
//! The [`InMemoryMavlNodeWrapper`] essentially owns the child node. There is no
//! resolution needed since everything is already in memory. The associated
//! resolver implementation is [`InMemoryMavlNodeResolver`]
//!
//! ## Lazy wrapper
//!
//! The [`LazyMavlNodeWrapper`] can hold both blinded nodes and nodes that are loaded
//! into memory. The associated resolved type is [`LazyMavlNodeResolver`].
//!
//! ### Blinded nodes
//!
//! In this case the [`LazyMavlNodeWrapper::commited_hash`] is not `None` and refers to
//! a node stored in `RocksDB`. By resolving this we store the decoded value in
//! [`LazyMavlNodeWrapper::node`]. Note that the child nodes of the newly loaded node
//! are blinded.
//!
//! ### Loaded nodes
//!
//! In case a node is loaded we set the value of [`LazyMavlNodeWrapper::node`] directly
//! and the value of [`LazyMavlNodeWrapper::commited_hash`] is irrelevant and never be
//! observed.

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
