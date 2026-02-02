// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

/// Tree node
#[derive(Clone, Debug, PartialEq)]
pub struct Node<Data, Tree> {
    /// Data contained in the node
    pub data: Data,

    /// Children of the node
    pub children: Vec<Tree>,
}

impl<Tree> Node<(), Tree> {
    /// Create a new node without data.
    pub fn new_without_data(children: Vec<Tree>) -> Self {
        Node { data: (), children }
    }
}

/// Generic tree structure used to model the [`crate::merkle_proof::proof_tree::MerkleProof`],
/// as well as the full & partial shapes of a [`crate::merkle_tree::MerkleTree`].
#[derive(Clone, Debug, PartialEq)]
pub enum Tree<LeafData, NodeData = ()> {
    Node(Node<NodeData, Self>),
    Leaf(LeafData),
}

impl<LeafData> Tree<LeafData, ()> {
    /// Create a new node without data.
    pub fn node_without_data(children: Vec<Self>) -> Self {
        let node = Node::new_without_data(children);
        Tree::Node(node)
    }
}
