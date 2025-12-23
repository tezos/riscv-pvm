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
//!
//! ## In memory resolver
//!
//! [`InMemoryMavlNodeResolver`] defines a resolver where it's assumed that the whole
//! tree is in memory and resolving a node is a blank operation. Its associated wrapper
//! type is [`InMemoryMavlNodeWrapper`].
//!
//! ## Lazy resolver
//!
//! [`LazyMavlNodeResolver`] defines a resolver where nodes can be blinded. A blinded
//! node is resolved by loading the node data based on the
//! [`LazyMavlNodeWrapper::commited_hash`]. Once the data is decoded into a [`MavlNode`]
//! this node is stored in the [`LazyMavlNodeWrapper::node`]. Note that when a new node
//! is loaded, both of its children are blinded. The associated  wrapper type is
//! [`LazyMavlNodeWrapper`].

use std::sync::Arc;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::serialisation::borrow_decode;

use super::node_resolver_error::MavlNodeResolverError;
use super::node_wrapper::InMemoryMavlNodeWrapper;
use super::node_wrapper::MavlNodeWrapper;
use crate::merkle_layer::Key;
use crate::merkle_layer::node::MavlNode;
use crate::merkle_layer::node::MavlNodeHashRepresentation;
use crate::merkle_layer::node_wrapper::LazyMavlNodeWrapper;
use crate::persistence_layer::PersistenceLayer;

pub(crate) trait MavlNodeResolver: Clone + std::fmt::Debug {
    type NodeWrapper: MavlNodeWrapper<Resolver = Self>;

    fn resolve(&self, node_wrapper: &Self::NodeWrapper);
}

#[derive(Debug, Default, Clone)]
pub(crate) struct InMemoryMavlNodeResolver {}

impl MavlNodeResolver for InMemoryMavlNodeResolver {
    type NodeWrapper = InMemoryMavlNodeWrapper;

    fn resolve(&self, _node_wrapper: &Self::NodeWrapper) {}
}

#[derive(Debug, Clone)]
pub(crate) struct LazyMavlNodeResolver {
    persistence_layer: Arc<PersistenceLayer>,
}

impl LazyMavlNodeResolver {
    /// Creates an instance of [`MavlNodeResolver`].
    pub(crate) fn new(persistence_layer: Arc<PersistenceLayer>) -> Self {
        Self { persistence_layer }
    }
}

fn parse_hash(node: Option<&[u8]>) -> Result<Option<Hash>, MavlNodeResolverError> {
    match node {
        Some(bytes) => {
            let array: &[u8; 32] = match bytes.try_into() {
                Ok(bytes) => bytes,
                Err(err) => {
                    return Err(MavlNodeResolverError::ChildHasWrongFormat(err));
                }
            };
            Ok(Some(Hash::hash_bytes(array)))
        }
        None => Ok(None),
    }
}

fn parse_key(node: &[u8]) -> Result<Key, MavlNodeResolverError> {
    match Key::new(node) {
        Ok(res) => Ok(res),
        Err(err) => Err(MavlNodeResolverError::KeyHasWrongFormat(err)),
    }
}

impl MavlNodeResolver for LazyMavlNodeResolver {
    type NodeWrapper = LazyMavlNodeWrapper;

    fn resolve(&self, node_wrapper: &Self::NodeWrapper) {
        node_wrapper.node.get_or_init(|| {
            let hash = match node_wrapper.commited_hash {
                Some(commited_hash) => commited_hash,
                None => return Err(MavlNodeResolverError::MissingCommitHash.into()),
            };
            let serialized_data = match self.persistence_layer.blob_get(&hash) {
                Ok(some_data) => some_data,
                Err(err) => {
                    return Err(MavlNodeResolverError::MissingInKeyValueStore(err).into());
                }
            };
            // This panics if the key is not present in RocksDB
            let node_representation: MavlNodeHashRepresentation =
                match borrow_decode(serialized_data.as_ref()) {
                    Ok(intermediate_representation) => intermediate_representation,
                    Err(err) => {
                        return Err(
                            MavlNodeResolverError::FailedParsingIntermediateRepresentation(err)
                                .into(),
                        );
                    }
                };

            let left_node = match parse_hash(node_representation.left)? {
                None => LazyMavlNodeWrapper::new(None),
                Some(hash) => LazyMavlNodeWrapper::blinded(hash),
            };
            let right_node = match parse_hash(node_representation.right)? {
                None => LazyMavlNodeWrapper::new(None),
                Some(hash) => LazyMavlNodeWrapper::blinded(hash),
            };
            let key = parse_key(node_representation.key)?;

            Ok(Some(Arc::new(MavlNode::decode(
                node_representation,
                hash,
                key,
                left_node,
                right_node,
            ))))
        });
    }
}
