// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;
use std::fmt::Debug;

use super::Key;
use super::NONE_HASH;
use super::node::AvlNode;
use super::node::BinaryTreeNode;
use super::node::BinaryTreeNodeInvalidating;
use super::node::MerkleNode;
use super::node::NodeData;
use super::node::NodeDataInvalidating;

/// An immutable key-value store tree with left and right nodes that supports traversal and value
/// retrieval.
#[derive(Clone, Default, Debug)]
pub struct Avl<Node: BinaryTreeNode> {
    root: Option<Box<Node>>,
}

impl<Node: BinaryTreeNode + Default + NodeData> BinaryTree for Avl<Node> {
    type Node = Node;

    fn root(&self) -> &Option<Box<Self::Node>> {
        &self.root
    }
}

impl<Node: BinaryTreeNode + BinaryTreeNodeInvalidating> BinaryTreeMutating for Avl<Node> {
    type Node = Node;

    fn root_mut(&mut self) -> &mut Option<Box<Self::Node>> {
        &mut self.root
    }
}

impl<Node: AvlNode + BinaryTreeNode + BinaryTreeNodeInvalidating + Debug> BinaryTreeUnbalancing
    for Avl<Node>
{
    type Node = Node;

    fn set(&mut self, key: &Key, data: Vec<u8>) {
        let root = self.root_mut();
        Node::set(root, key, data);
    }

    fn delete(&mut self, key: &Key) {
        let root = self.root_mut();
        Node::delete(root, key);
    }
}

impl<Node: AvlNode + BinaryTreeNode + BinaryTreeNodeInvalidating + Debug + Default + MerkleNode>
    MerkleBinaryTree for Avl<Node>
{
    type Node = Node;
}

/// A tree of nodes with basic immutable binary tree traversal.
pub(super) trait BinaryTree: Default {
    type Node: BinaryTreeNode + NodeData;

    /// An immutable reference the data stored in a node in the tree with a given key.
    fn get(&self, key: &Key) -> Option<&Vec<u8>> {
        let mut node = self.root().as_deref()?;
        loop {
            match node.key().cmp(key) {
                Ordering::Equal => return Some(node.data()),
                Ordering::Greater => node = node.left_ref().as_deref()?,
                Ordering::Less => node = node.right_ref().as_deref()?,
            }
        }
    }

    /// An immutable reference to the root node of the tree.
    fn root(&self) -> &Option<Box<Self::Node>>;
}

/// A mutable key-value store tree.
pub(super) trait BinaryTreeMutating {
    type Node: BinaryTreeNodeInvalidating + NodeDataInvalidating;

    /// A mutable reference to the root node of the tree.
    fn root_mut(&mut self) -> &mut Option<Box<Self::Node>>;
}

/// A mutable key-value store binary tree that supports operations that can cause it to become unbalanced.
pub(super) trait BinaryTreeUnbalancing:
    BinaryTree<Node = <Self as BinaryTreeUnbalancing>::Node>
    + BinaryTreeMutating<Node = <Self as BinaryTreeUnbalancing>::Node>
{
    type Node;

    /// Set the value of a node in the tree with a given key.
    ///
    /// If it does not exist, do nothing.
    fn set(&mut self, key: &Key, data: Vec<u8>);

    /// Delete the node in the tree with a given key.
    ///
    /// If it does not exist, do nothing.
    fn delete(&mut self, key: &Key);
}

/// A mutable key-value store binary tree with nodes that can be Merklised.
pub(super) trait MerkleBinaryTree:
    BinaryTreeUnbalancing<Node = <Self as MerkleBinaryTree>::Node>
{
    type Node: MerkleNode + NodeData + NodeDataInvalidating;

    fn hash(&mut self) -> blake3::Hash {
        if let Some(root) = self.root_mut() {
            root.hash()
        } else {
            NONE_HASH
        }
    }
}
