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
use std::sync::Arc;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::serialisation::deserialise_borrowed;

use super::node::MavlNodeHashRepresentation;
use super::node_resolver_error::MavlNodeResolverError;
use super::node_wrapper::InMemoryMavlNodeWrapper;
use super::node_wrapper::MavlNodeWrapper;
use crate::merkle_layer::Key;
use crate::merkle_layer::node::MavlNode;
use crate::merkle_layer::node_wrapper::LazyMavlNodeWrapper;
use crate::persistence_layer::PersistenceLayer;

pub(crate) trait MavlNodeResolver: Clone + Debug {
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
                match deserialise_borrowed(serialized_data.as_ref()) {
                    Ok(intermediate_representation) => intermediate_representation,
                    Err(err) => {
                        return Err(
                            MavlNodeResolverError::FailedParsingIntermediateRepresentation(err)
                                .into(),
                        );
                    }
                };

            let key = parse_key(node_representation.key)?;

            Ok(Some(Arc::new(MavlNode::decode(
                node_representation,
                hash,
                key,
                None,
                None,
            ))))
        });
    }
}
