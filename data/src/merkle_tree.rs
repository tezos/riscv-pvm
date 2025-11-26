// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Merkle tree data structure and utilities

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::hash::Hash;
use crate::tree::Tree;

#[derive(Debug, Clone, PartialEq)]
pub struct MerkleTreeLeafData {
    pub hash: Hash,
    pub access_info: bool,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MerkleTreeNodeData {
    pub hash: Hash,
}

/// A variable-width Merkle tree with access metadata for leaves.
///
/// Values of this type are produced by the proof-generating backend to capture
/// a snapshot of the machine state along with access information for leaves
/// which hold data that was used in a particular evaluation step.
pub type MerkleTree = Tree<MerkleTreeLeafData, MerkleTreeNodeData>;

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
    /// let hash = Hash::blake3_hash_bytes(&data);
    ///
    /// assert_eq!(hash, root_hash);
    /// ```
    pub fn root_hash(&self) -> Hash {
        match self {
            Self::Node {
                data: MerkleTreeNodeData { hash },
                ..
            } => *hash,
            Self::Leaf {
                data: MerkleTreeLeafData { hash, .. },
            } => *hash,
        }
    }

    /// Creates a merkle tree which is a single leaf
    pub fn make_merkle_leaf(data: Vec<u8>, access_info: bool) -> Self {
        let hash = Hash::blake3_hash_bytes(&data);
        MerkleTree::Leaf {
            data: MerkleTreeLeafData {
                hash,
                access_info,
                data,
            },
        }
    }

    /// Takes a list of children nodes and creates a
    /// new parent node on top of them.
    pub fn make_merkle_node(children: Vec<Self>) -> Self {
        let children_hashes = children.iter().map(|t| t.root_hash());
        let hash = Hash::combine(children_hashes);
        MerkleTree::Node {
            data: MerkleTreeNodeData { hash },
            children,
        }
    }

    /// Recomputes the hashes for the whole tree
    /// and checks whether the previous values match
    /// the calculated ones
    pub fn check_root_hash(&self) -> bool {
        let mut deque = std::collections::VecDeque::new();
        deque.push_back(self);

        while let Some(node) = deque.pop_front() {
            let is_valid_hash = match node {
                Self::Leaf {
                    data:
                        MerkleTreeLeafData {
                            hash,
                            data: node_data,
                            ..
                        },
                } => &Hash::blake3_hash_bytes(node_data) == hash,
                Self::Node {
                    data: MerkleTreeNodeData { hash },
                    children,
                } => {
                    let children_hashes: Vec<Hash> = children
                        .iter()
                        .map(|child| {
                            deque.push_back(child);
                            child.root_hash()
                        })
                        .collect();

                    &Hash::combine(children_hashes) == hash
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

pub struct MerkleTreeFold;

impl Fold for MerkleTreeFold {
    type Folded = MerkleTree;

    type NodeFold = MerkleNodeFold;

    fn into_node_fold(self) -> Self::NodeFold {
        MerkleNodeFold::default()
    }
}

#[derive(Default)]
pub struct MerkleNodeFold {
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
