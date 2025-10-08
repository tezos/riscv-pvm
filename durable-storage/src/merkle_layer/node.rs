// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;

use super::Key;
use super::NONE_HASH;

/// A node which supports rebalancing operations
pub trait AvlNode: BinaryTreeNode {
    /// Rebalance the node so that the difference in height between child branches is in the range
    /// of -2..=2.
    fn rebalance(node: &mut Box<Self>);
}

/// A node with basic immutable binary tree traversal.
pub trait BinaryTreeNode {
    /// An immutable reference to the left branch.
    fn left_ref(&self) -> &Option<Box<Self>>;

    /// An immutable reference to the right branch.
    fn right_ref(&self) -> &Option<Box<Self>>;
}

/// A node with basic mutable binary tree traversal and mutation.
pub trait BinaryTreeNodeInvalidating: NodeData + NodeDataInvalidating {
    /// Delete the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn delete(root: &mut Option<Box<Self>>, key: &Key);

    /// A mutable reference to the left branch.
    fn left_mut(&mut self) -> &mut Option<Box<Self>>;

    /// A mutable reference to the right branch.
    fn right_mut(&mut self) -> &mut Option<Box<Self>>;

    /// Set the value of the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn set(root: &mut Option<Box<Self>>, key: &Key, data: Vec<u8>);

    /// Swap this node with its successor.
    ///
    /// This method assumes this node has a successor.
    fn trade_successor(root: &mut Option<Box<Self>>);
}

/// A node that supports rebalancing and Merklisation
#[derive(Clone, Default, Debug)]
pub struct MavlNode {
    key: Key,
    data: Vec<u8>,
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
    hash: Option<blake3::Hash>,
    _height: u32,
}

impl AvlNode for MavlNode {
    fn rebalance(_node: &mut Box<Self>) {
        todo!()
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
        let mut stack = Vec::new();
        let mut current = root as *mut Option<Box<Self>>;

        unsafe {
            while let Some(node) = &mut *current {
                stack.push(node as *mut Box<Self>);
                match key.cmp(node.key()) {
                    Ordering::Equal => {
                        match (node.left_ref().as_ref(), node.right_ref().as_ref()) {
                            (None, None) => *current = None,
                            (Some(_), None) => *current = node.left_mut().take(),
                            (None, Some(_)) => *current = node.right_mut().take(),
                            (Some(_), Some(_)) => Self::trade_successor(&mut *current),
                        }
                        return;
                    }

                    Ordering::Greater => {
                        current = (&mut *current)
                            .as_mut()
                            .expect("current is Some(_) in this loop")
                            .right_mut()
                    }

                    Ordering::Less => {
                        current = (&mut *current)
                            .as_mut()
                            .expect("current is Some(_) in this loop")
                            .left_mut()
                    }
                }
            }

            for &node_ptr in stack.iter().rev().skip(1) {
                let node = &mut *node_ptr;
                Self::rebalance(node);
            }
        }
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
        let mut stack = Vec::new();
        let mut current = root as *mut Option<Box<Self>>;

        unsafe {
            while let Some(node) = &mut *current {
                stack.push(node as *mut Box<Self>);
                match node.key().cmp(key) {
                    std::cmp::Ordering::Equal => {
                        *node.key_mut() = key.clone();
                        *node.data_mut() = data;
                        return;
                    }
                    std::cmp::Ordering::Greater => current = &mut node.left as *mut _,
                    std::cmp::Ordering::Less => current = &mut node.right as *mut _,
                }
            }

            *current = Some(Box::new(Self::new(key.clone(), data)));
            if let Some(node) = &mut *current {
                stack.push(node as *mut Box<Self>);
            }
        }
    }

    fn trade_successor(_root: &mut Option<Box<Self>>) {
        todo!()
    }
}

impl MerkleNode for MavlNode {
    fn hash(&mut self) -> blake3::Hash {
        if let Some(hash) = self.hash {
            hash
        } else {
            let mut hasher = blake3::Hasher::new();
            let (l, r) = match (&mut self.left, &mut self.right) {
                (Some(l), Some(r)) => (l.hash(), r.hash()),
                (Some(l), None) => (l.hash(), NONE_HASH),
                (None, Some(r)) => (NONE_HASH, r.hash()),
                (None, None) => (NONE_HASH, NONE_HASH),
            };

            hasher.update(l.as_bytes());
            hasher.update(r.as_bytes());
            let hash = hasher.finalize();
            self.hash = Some(hash);
            hash
        }
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

    fn new(key: Key, data: Vec<u8>) -> Self {
        MavlNode {
            key,
            data,
            _height: 1,
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
pub trait MerkleNode {
    /// Return the hash of the node.
    ///
    /// This may trigger re-hashing if the hash of this node is dirty.
    fn hash(&mut self) -> blake3::Hash;

    /// Mark the hash of this node as dirty
    fn invalidate_hash(&mut self);
}

/// A key-value store node with immutable access.
pub trait NodeData: Default + Sized {
    /// An immutable reference to the data stored in the node.
    fn data(&self) -> &Vec<u8>;

    /// An immutable reference to the key used for finding the node.
    fn key(&self) -> &Key;

    /// Create a new node from the given key and data.
    fn new(key: Key, data: Vec<u8>) -> Self;
}

/// Mutable access to a key-value store node.
pub trait NodeDataInvalidating {
    /// A mutable reference to the data stored in the node.
    fn data_mut(&mut self) -> &mut Vec<u8>;

    /// A mutable reference to the key used for finding the node.
    fn key_mut(&mut self) -> &mut Key;
}
