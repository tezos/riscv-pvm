// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use super::Key;

/// A node with basic immutable binary tree traversal.
pub trait BinaryTreeNode {
    /// An immutable reference to the left branch.
    fn left_ref(&self) -> &Option<Box<Self>>;

    /// An immutable reference to the right branch.
    fn right_ref(&self) -> &Option<Box<Self>>;
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
