// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::OnceLock;

use super::hash;
use bincode::Encode;
use bytes::Bytes;
use octez_riscv_data::serialisation::serialise;

use super::Key;

/// A node that supports rebalancing and Merklisation.
#[derive(Clone, Default, Debug)]
pub(crate) struct MavlNode {
    pub(crate) key: Key,
    pub(crate) data: Bytes,
    pub(crate) left: Option<Arc<Self>>,
    pub(crate) right: Option<Arc<Self>>,

    /// A cache for the hash of this node. This uses `OnceLock` so that updating the cache is a
    /// non-mutating operation.
    ///
    /// An uninitialised hash is a hash that has not been set or has been dirtied.
    pub(crate) hash: OnceLock<blake3::Hash>,

    /// The difference in heights between child branches (right - left).
    pub(crate) balance_factor: i64,
}

#[derive(Encode)]
/// A serialisable representation of [`MavlNode`].
struct MavlNodeHashRepresentation<'a> {
    key: &'a Key,
    data: &'a [u8],
    // The bytes of the hash of an optional left child
    left: Option<&'a [u8; blake3::OUT_LEN]>,
    // The bytes of the hash of an optional right child
    right: Option<&'a [u8; blake3::OUT_LEN]>,
    balance_factor: i64,
}

impl MavlNode {
    /// The difference in heights between child branches.
    #[cfg(test)]
    pub(super) fn balance_factor(&self) -> i64 {
        self.balance_factor
    }

    /// The data stored in the node.
    pub(super) fn data(&self) -> &Bytes {
        &self.data
    }

    /// The key used for determining the node.
    pub(super) fn key(&self) -> &Key {
        &self.key
    }

    /// A mutable reference to the left branch.
    pub(super) fn left_mut(&mut self) -> &mut Option<Arc<Self>> {
        self.invalidate_hash();
        &mut self.left
    }

    /// An immutable reference to the left branch.
    pub(super) fn left_ref(&self) -> &Option<Arc<Self>> {
        &self.left
    }

    /// Create a new leaf node from the given key and data.
    pub(super) fn new(key: Key, data: Bytes) -> Self {
        MavlNode {
            key,
            data,
            balance_factor: 0,
            ..Default::default()
        }
    }

    /// A mutable reference to the right branch.
    pub(super) fn right_mut(&mut self) -> &mut Option<Arc<Self>> {
        self.invalidate_hash();
        &mut self.right
    }

    /// An immutable reference to the right branch.
    pub(super) fn right_ref(&self) -> &Option<Arc<Self>> {
        &self.right
    }

    /// Converts the node to an encoded, serialisable representation, potentially re-hashing
    /// uncached nodes.
    pub(super) fn to_encode(&self) -> impl Encode + '_ {
        MavlNodeHashRepresentation {
            key: &self.key,
            data: &self.data,

            // Recursively hashes any left child and its children
            left: self.left.as_ref().map(hash).map(|h| h.as_bytes()),

            // Recursively hashes any right child and its children
            right: self.right.as_ref().map(hash).map(|h| h.as_bytes()),

            balance_factor: self.balance_factor,
        }
    }

    pub(crate) fn encode_to_vec(&self) -> Vec<u8> {
        serialise(self.to_encode()).expect("Serialisation of a MavlNode should not fail")
    }

    /// Mark the hash of this node as dirty.
    pub(crate) fn invalidate_hash(&mut self) {
        self.hash = OnceLock::new();
    }

    pub(crate) fn get_key(&self) -> &Key {
        &self.key
    }
}
