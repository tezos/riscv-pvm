// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;
use std::fmt::Debug;

use super::Key;
use super::NONE_BYTE;
use super::SOME_BYTE;

/// A node which supports rebalancing operations
pub(super) trait AvlNode: BinaryTreeNode + BinaryTreeNodeInvalidating + Debug {
    fn balance_factor(&self) -> i64 {
        let left_height = self.left_ref().as_ref().map(|l| l.height()).unwrap_or(0);
        let right_height = self.right_ref().as_ref().map(|r| r.height()).unwrap_or(0);
        left_height as i64 - right_height as i64
    }

    /// Delete the node with a given key.
    ///
    /// If it does not exist, do nothing.
    fn delete(root: &mut Option<Box<Self>>, key: &Key) {
        let mut stack: Vec<*mut Box<Self>> = Vec::new();
        let mut current = root as *mut Option<Box<Self>>;

        #[cfg(debug_assertions)]
        let height = root.as_ref().map(|r| r.height()).unwrap_or(0);

        unsafe {
            while let Some(node) = &mut *current {
                #[cfg(debug_assertions)]
                debug_assert!(node.is_balanced(), "Node not balanced? {node:?}");
                match key.cmp(node.key()) {
                    Ordering::Equal => {
                        match (node.left_ref().as_ref(), node.right_ref().as_ref()) {
                            (None, None) => *current = None,
                            (Some(_), None) => *current = node.left_mut().take(),
                            (None, Some(_)) => *current = node.right_mut().take(),
                            (Some(_), Some(_)) => Self::trade_successor(&mut *current),
                        }

                        for &node_ptr in stack.iter().rev() {
                            if node_ptr.is_null() {
                                continue;
                            }
                            let node = &mut *node_ptr;
                            Self::rebalance(node);
                            #[cfg(debug_assertions)]
                            debug_assert!(node.is_balanced(), "Node not balanced? {node:?}");
                        }

                        #[cfg(debug_assertions)]
                        {
                            let new_height = root.as_ref().map(|r| r.height()).unwrap_or(0);
                            debug_assert!(
                                new_height == height - 1 || new_height == height,
                                "Height should be {:?} or {new_height:?} but is {height:?}",
                                new_height - 1
                            );
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
                stack.push(node);
            }
        }

        #[cfg(debug_assertions)]
        {
            let new_height = root.as_ref().map(|r| r.height()).unwrap_or(0);
            debug_assert!(
                new_height == height,
                "Height should be {new_height:?} but is {height:?}"
            );
        }
    }

    /// Return the height of the branch including this node.
    fn height(&self) -> u32;

    #[cfg(debug_assertions)]
    fn is_balanced(&self) -> bool {
        (-1..=1).contains(&self.balance_factor())
    }

    /// Rebalance the node so that the difference in height between child branches is in the range
    /// of -2..=2.
    fn rebalance(node: &mut Box<Self>) {
        node.update_height();
        #[cfg(debug_assertions)]
        {
            let left_height = node.left_ref().as_ref().map(|l| l.height()).unwrap_or(0);
            let right_height = node.right_ref().as_ref().map(|r| r.height()).unwrap_or(0);

            debug_assert!(
                (-2..=2).contains(&node.balance_factor()),
                "Node is imbalanced: {node:?}\nleft height: {left_height:?} | right height: {right_height:?}"
            );
        }

        match node.balance_factor() {
            ..=-2 => {
                let right_balance = node
                    .right_ref()
                    .as_ref()
                    .map(|r| {
                        let lh = r.left_ref().as_ref().map(|x| x.height()).unwrap_or(0);
                        let rh = r.right_ref().as_ref().map(|x| x.height()).unwrap_or(0);
                        lh as i64 - rh as i64
                    })
                    .unwrap_or(0);

                if right_balance <= 0 {
                    Self::rotate_left(node);
                } else {
                    if let Some(right) = node.right_mut() {
                        Self::rotate_right(right);
                    }
                    Self::rotate_left(node);
                }
            }
            -1..=1 => (/* Already balanced */),
            2.. => {
                let left_balance = node
                    .left_ref()
                    .as_ref()
                    .map(|l| {
                        let lh = l.left_ref().as_ref().map(|x| x.height()).unwrap_or(0);
                        let rh = l.right_ref().as_ref().map(|x| x.height()).unwrap_or(0);
                        lh as i64 - rh as i64
                    })
                    .unwrap_or(0);

                if left_balance >= 0 {
                    Self::rotate_right(node);
                } else {
                    if let Some(left) = node.left_mut() {
                        Self::rotate_left(left);
                    }
                    Self::rotate_right(node);
                }
            }
        }
    }

    /// Rotate this node left.
    ///
    /// For example:
    ///
    /// ```text
    ///     BEFORE          AFTER
    /// self                  A
    ///     \               /   \
    ///       A          self    B
    ///         \
    ///           B
    /// ```
    fn rotate_left(node: &mut Box<Self>) {
        let mut right = node
            .right_mut()
            .take()
            .expect("To rotate left there should be a right child");
        *node.right_mut() = right.left_mut().take();
        right.update_height();
        node.update_height();

        let new_left = std::mem::replace(node, right);
        *node.left_mut() = Some(new_left);
        node.update_height();
    }

    /// Rotate this node right.
    ///
    /// For example:
    ///
    /// ```text
    ///     BEFORE          AFTER
    ///          self         A
    ///         /           /   \
    ///       A          self    B
    ///     /
    ///   B
    /// ```
    fn rotate_right(node: &mut Box<Self>) {
        let mut left = node
            .left_mut()
            .take()
            .expect("To rotate right there should be a left child");
        *node.left_mut() = left.right_mut().take();
        left.update_height();
        node.update_height();

        let new_right = std::mem::replace(node, left);
        *node.right_mut() = Some(new_right);
        node.update_height();
    }

    fn set(root: &mut Option<Box<Self>>, key: &Key, data: Vec<u8>) {
        let mut stack: Vec<*mut Box<Self>> = Vec::new();
        let mut current = root as *mut Option<Box<Self>>;

        let do_rebalancing = |stack: Vec<*mut Box<Self>>| {
            for &node_ptr in stack.iter().rev() {
                unsafe {
                    let node = &mut *node_ptr;
                    Self::rebalance(node);
                }
            }
        };

        unsafe {
            while let Some(node) = &mut *current {
                stack.push(node);
                match node.key().cmp(key) {
                    std::cmp::Ordering::Equal => {
                        *node.key_mut() = key.clone();
                        *node.data_mut() = data;
                        do_rebalancing(stack);
                        return;
                    }
                    std::cmp::Ordering::Greater => {
                        stack.push(node);
                        current = node.left_mut() as *mut _
                    }
                    std::cmp::Ordering::Less => {
                        stack.push(node);
                        current = node.right_mut() as *mut _
                    }
                }
            }

            *current = Some(Box::new(Self::new(key.clone(), data)));
            do_rebalancing(stack);
        }
    }

    /// Update the height stored in this node by adding 1 to the greatest height of any child
    /// branches.
    fn update_height(&mut self);
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
    height: u32,
}

impl AvlNode for MavlNode {
    fn height(&self) -> u32 {
        self.height
    }

    fn update_height(&mut self) {
        let left_height = self.left_ref().as_ref().map(|l| l.height()).unwrap_or(0);
        let right_height = self.right_ref().as_ref().map(|r| r.height()).unwrap_or(0);
        self.height = 1 + std::cmp::max(left_height, right_height);
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

    fn trade_successor(root: &mut Option<Box<Self>>) {
        debug_assert!(root.is_some());
        let mut successor = root
            .as_mut()
            .expect("Early return if `root` is None")
            .right_mut();
        debug_assert!(successor.is_some());

        let mut stack: Vec<*mut Box<Self>> = Vec::new();
        while successor
            .as_ref()
            .expect("`right` has been asserted to be `Some(_)`")
            .left_ref()
            .is_some()
        {
            stack.push(successor.as_mut().expect("In is_some() loop"));
            successor = &mut *successor
                .as_mut()
                .expect("`right` has been asserted to be `Some(_)`")
                .left_mut()
        }

        let mut min = successor
            .take()
            .expect("The while loop checks this is `Some(_)`");
        min.as_mut().update_height();
        *successor = min.right;

        if let Some(s) = successor.as_mut() {
            s.update_height();
        }

        for &node_ptr in stack.iter().rev() {
            if node_ptr.is_null() {
                continue;
            }
            unsafe {
                let node = &mut *node_ptr;
                Self::rebalance(node);
                #[cfg(debug_assertions)]
                debug_assert!(node.is_balanced(), "Node not balanced? {node:?}");
            }
        }

        *root
            .as_mut()
            .expect("Early return if `root` is None")
            .data_mut() = min.data;

        *root
            .as_mut()
            .expect("Early return if `root` is None")
            .key_mut() = min.key;

        Self::rebalance(root.as_mut().expect("Early return if `root` is None"));
    }
}

impl MerkleNode for MavlNode {
    fn hash(&mut self) -> blake3::Hash {
        if let Some(hash) = self.hash {
            hash
        } else {
            let mut hasher = blake3::Hasher::new();
            let l = self.left.as_mut().map(|l| l.hash());
            let r = self.right.as_mut().map(|r| r.hash());

            match l {
                Some(digest) => {
                    hasher.update(&[SOME_BYTE]);
                    hasher.update(digest.as_bytes())
                }
                _ => hasher.update(&[NONE_BYTE]),
            };

            match r {
                Some(digest) => {
                    hasher.update(&[SOME_BYTE]);
                    hasher.update(digest.as_bytes())
                }
                _ => hasher.update(&[NONE_BYTE]),
            };

            hasher.update(&self.key.0);
            hasher.update(self.data());
            hasher.update(&self.balance_factor().to_le_bytes());

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
    fn new(key: Key, data: Vec<u8>) -> Self;
}

/// Mutable access to a key-value store node.
pub(super) trait NodeDataInvalidating {
    /// A mutable reference to the data stored in the node.
    fn data_mut(&mut self) -> &mut Vec<u8>;

    /// A mutable reference to the key used for finding the node.
    fn key_mut(&mut self) -> &mut Key;
}
