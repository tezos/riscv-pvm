// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Merkle tree data structure and utilities

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::hash::Hash;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::tree::Node;
use crate::tree::Tree;

/// Struct which holds data for the leafs of a [`MerkleTree`].
#[derive(Debug, Clone)]
pub struct MerkleTreeLeafData {
    /// The hash of the leaf.
    pub hash: Hash,

    /// Whether the leaf was accessed.
    pub access_info: bool,

    /// The data associated with the leaf node.
    pub data: Vec<u8>,
}

/// Merkle node data
#[derive(Debug, Clone)]
pub struct MerkleNodeData {
    /// Hash of the Merkle node
    pub hash: Hash,
}

/// A variable-width Merkle tree with access metadata for leaves.
///
/// Values of this type are produced by the proof-generating backend to capture
/// a snapshot of the machine state along with access information for leaves
/// which hold data that was used in a particular evaluation step.
pub type MerkleTree = Tree<MerkleTreeLeafData, MerkleNodeData>;

impl MerkleTree {
    /// Returns the precalculated root hash of the node.
    ///
    /// # Example
    ///
    /// ```
    /// use octez_riscv_data::merkle_tree::MerkleTree;
    /// use octez_riscv_data::hash::Hash;
    ///
    /// let data = vec![1, 2, 3];
    /// let merkle_tree = MerkleTree::make_merkle_leaf(data.clone(), false);
    /// let root_hash = merkle_tree.root_hash();
    /// let hash = Hash::hash_bytes(&data);
    ///
    /// assert_eq!(hash, root_hash);
    /// ```
    pub fn root_hash(&self) -> Hash {
        match self {
            Self::Node(node) => node.data.hash,
            Self::Leaf(leaf) => leaf.hash,
        }
    }

    /// Creates a merkle tree which is a single leaf
    pub fn make_merkle_leaf(data: Vec<u8>, access_info: bool) -> Self {
        let hash = Hash::hash_bytes(&data);
        MerkleTree::Leaf(MerkleTreeLeafData {
            hash,
            access_info,
            data,
        })
    }

    /// Takes a list of children nodes and creates a
    /// new parent node on top of them.
    pub fn make_merkle_node(children: Vec<Self>) -> Self {
        let children_hashes = children.iter().map(|t| t.root_hash());
        let node_hash = Hash::combine_hashes(children_hashes);
        MerkleTree::Node(Node {
            data: MerkleNodeData { hash: node_hash },
            children,
        })
    }

    /// Recomputes the hashes for the whole tree
    /// and checks whether the previous values match
    /// the calculated ones
    pub fn check_root_hash(&self) -> bool {
        let mut deque = std::collections::VecDeque::new();
        deque.push_back(self);

        while let Some(node) = deque.pop_front() {
            let is_valid_hash = match node {
                Self::Leaf(MerkleTreeLeafData { hash, data, .. }) => {
                    &Hash::hash_bytes(data) == hash
                }
                Self::Node(node) => {
                    let children_hashes: Vec<Hash> = node
                        .children
                        .iter()
                        .map(|child| {
                            deque.push_back(child);
                            child.root_hash()
                        })
                        .collect();

                    Hash::combine_hashes(children_hashes) == node.data.hash
                }
            };
            if !is_valid_hash {
                return false;
            }
        }
        true
    }

    /// Extract the Merkle tree from a foldable structure.
    pub fn from_foldable(foldable: &impl Foldable<MerkleTreeFold>) -> MerkleTree {
        foldable.fold(MerkleTreeFold)
    }

    /// Compress a [`MerkleTree`] into a [`MerkleProof`].
    ///
    /// If a leaf was not accessed, it will be compressed as a blinded leaf.
    /// If all children of a node are blinded, compress it as a blinded node.
    pub fn compress(self) -> MerkleProof {
        match self {
            Tree::Node(node) => compress_merkle_node(node.data.hash, node.children),
            Tree::Leaf(leaf) => compress_merkle_leaf(leaf),
        }
    }
}

impl Foldable<MerkleTreeFold> for MerkleTree {
    fn fold(&self, _builder: MerkleTreeFold) -> MerkleTree {
        self.clone()
    }
}

/// [`Fold`] implementation for Merkle trees
pub struct MerkleTreeFold;

impl Fold for MerkleTreeFold {
    type Folded = MerkleTree;

    type NodeFold = MerkleNodeFold;

    fn into_node_fold(self) -> Self::NodeFold {
        MerkleNodeFold::default()
    }
}

/// [`NodeFold`] implementation for Merkle trees
///
/// It accumulates child [`MerkleTree`] node and constructs a parent node when [`NodeFold::done`] is
/// called.
#[derive(Default)]
pub struct MerkleNodeFold {
    /// Accumulated child nodes
    children: Vec<MerkleTree>,
}

impl NodeFold for MerkleNodeFold {
    type Parent = MerkleTreeFold;

    fn add<F: Foldable<MerkleTreeFold>>(&mut self, child: &F) {
        let folded_child = child.fold(MerkleTreeFold);
        self.children.push(folded_child);
    }

    fn done(self) -> MerkleTree {
        MerkleTree::make_merkle_node(self.children)
    }
}

/// Compress a Merkle tree leaf.
fn compress_merkle_leaf(leaf: MerkleTreeLeafData) -> MerkleProof {
    if !leaf.access_info {
        return Tree::Leaf(MerkleProofLeaf::Blind(leaf.hash));
    }

    Tree::Leaf(MerkleProofLeaf::Read(leaf.data))
}

/// Compress a Merkle tree node.
fn compress_merkle_node(hash: Hash, children: Vec<MerkleTree>) -> MerkleProof {
    let children = children
        .into_iter()
        .map(MerkleTree::compress)
        .collect::<Vec<_>>();

    let not_accessed = children
        .iter()
        .all(|child| matches!(child, Tree::Leaf(MerkleProofLeaf::Blind(_))));

    if not_accessed {
        return Tree::Leaf(MerkleProofLeaf::Blind(hash));
    }

    Tree::node_without_data(children)
}

#[cfg(test)]
mod tests;
