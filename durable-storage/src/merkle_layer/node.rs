// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fmt::Debug;

use super::Key;

/// A node which supports rebalancing operations
pub(super) trait AvlNode: BinaryTreeNode + BinaryTreeNodeInvalidating {
    /// Delete the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn delete(_root: &mut Option<Box<Self>>, _key: &Key) {
        todo!()
    }

    /// Return the height of the branch including this node.
    #[cfg(test)] // Currently only used in a test
    fn height(&self) -> u32;

    /// Rebalance the node so that the difference in height between child branches is in the range
    /// of -1..=1.
    #[expect(dead_code, reason = "Not used")]
    fn rebalance(_node: &mut Box<Self>) {
        todo!()
    }

    /// Set the value of the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn set(_root: &mut Option<Box<Self>>, _key: &Key, _data: Vec<u8>) {
        todo!()
    }
}

/// A node with basic immutable binary tree traversal.
pub(super) trait BinaryTreeNode {
    /// An immutable reference to the left branch.
    fn left_ref(&self) -> &Option<Box<Self>>;

    /// An immutable reference to the right branch.
    fn right_ref(&self) -> &Option<Box<Self>>;
}

/// A node with basic mutable binary tree traversal and mutation.
pub(super) trait BinaryTreeNodeInvalidating: NodeData + NodeDataInvalidating {
    /// Delete the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn delete(root: &mut Option<Box<Self>>, key: &Key);

    /// A mutable reference to the left branch.
    #[expect(dead_code, reason = "Not implemented")]
    fn left_mut(&mut self) -> &mut Option<Box<Self>>;

    /// A mutable reference to the right branch.
    #[expect(dead_code, reason = "Not implemented")]
    fn right_mut(&mut self) -> &mut Option<Box<Self>>;

    /// Set the value of the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn set(root: &mut Option<Box<Self>>, key: &Key, data: Vec<u8>);
}

/// A node that supports rebalancing and Merklisation
#[derive(Clone, Default, Debug)]
pub struct MavlNode {
    key: Key,
    data: Vec<u8>,
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,

    // A [`None`] hash is a hash that has not been set or has been dirtied
    hash: Option<blake3::Hash>,
    #[cfg(test)] // Currently only used in a test
    height: u32,
}

impl AvlNode for MavlNode {
    #[cfg(test)] // Currently only used in a test
    fn height(&self) -> u32 {
        self.height
    }
}

impl BinaryTreeNode for MavlNode {
    fn left_ref(&self) -> &Option<Box<Self>> {
        &self.left
    }

    fn right_ref(&self) -> &Option<Box<Self>> {
        &self.right
    }
}

impl BinaryTreeNodeInvalidating for MavlNode {
    fn delete(root: &mut Option<Box<Self>>, key: &Key) {
        AvlNode::delete(root, key)
    }

    fn left_mut(&mut self) -> &mut Option<Box<Self>> {
        self.invalidate_hash();
        &mut self.left
    }

    fn right_mut(&mut self) -> &mut Option<Box<Self>> {
        self.invalidate_hash();
        &mut self.right
    }

    fn set(root: &mut Option<Box<Self>>, key: &Key, data: Vec<u8>) {
        AvlNode::set(root, key, data)
    }
}

impl MerkleNode for MavlNode {
    fn hash(&mut self) -> blake3::Hash {
        todo!()
    }

    fn invalidate_hash(&mut self) {
        self.hash = None;
    }
}

impl NodeData for MavlNode {
    fn data(&self) -> &Vec<u8> {
        &self.data
    }

    fn key(&self) -> &Key {
        &self.key
    }

    #[cfg(test)] // Currently only used in a test
    fn new(key: Key, data: Vec<u8>) -> Self {
        MavlNode {
            key,
            data,
            height: 1,
            ..Default::default()
        }
    }
}

impl NodeDataInvalidating for MavlNode {
    fn data_mut(&mut self) -> &mut Vec<u8> {
        self.invalidate_hash();
        &mut self.data
    }

    fn key_mut(&mut self) -> &mut Key {
        self.invalidate_hash();
        &mut self.key
    }
}

/// A node that can be Merklised
pub(super) trait MerkleNode {
    /// Return the hash of the node.
    ///
    /// This may trigger re-hashing if the hash of this node is dirty.
    #[expect(dead_code, reason = "Not implemented")]
    fn hash(&mut self) -> blake3::Hash;

    /// Mark the hash of this node as dirty
    fn invalidate_hash(&mut self);
}

/// A key-value store node with immutable access.
pub(super) trait NodeData: Default + Sized {
    /// An immutable reference to the data stored in the node.
    fn data(&self) -> &Vec<u8>;

    /// An immutable reference to the key used for finding the node.
    fn key(&self) -> &Key;

    /// Create a new node from the given key and data.
    #[cfg(test)] // Currently only used in a test
    fn new(key: Key, data: Vec<u8>) -> Self;
}

/// Mutable access to a key-value store node.
pub(super) trait NodeDataInvalidating {
    /// A mutable reference to the data stored in the node.
    #[expect(dead_code, reason = "Not implemented")]
    fn data_mut(&mut self) -> &mut Vec<u8>;

    /// A mutable reference to the key used for finding the node.
    #[expect(dead_code, reason = "Not implemented")]
    fn key_mut(&mut self) -> &mut Key;
}
