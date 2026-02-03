// SPDX-FileCopyrightText: 2025-2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Merkle tree data structure and utilities

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::hash::Hash;

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

/// A variable-width Merkle tree with access metadata for leaves.
///
/// Values of this type are produced by the proof-generating backend to capture
/// a snapshot of the machine state along with access information for leaves
/// which hold data that was used in a particular evaluation step.
#[derive(Debug, Clone)]
pub enum MerkleTree {
    Leaf(MerkleTreeLeafData),
    Node(Hash, Vec<Self>),
}

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
            Self::Node(hash, _) => *hash,
            Self::Leaf(MerkleTreeLeafData { hash, .. }) => *hash,
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
        MerkleTree::Node(node_hash, children)
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
                Self::Node(hash, children) => {
                    let children_hashes: Vec<Hash> = children
                        .iter()
                        .map(|child| {
                            deque.push_back(child);
                            child.root_hash()
                        })
                        .collect();

                    &Hash::combine_hashes(children_hashes) == hash
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
