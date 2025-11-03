// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Merkle tree data structure and utilities

use crate::hash::Hash;

/// A variable-width Merkle tree with access metadata for leaves.
///
/// Values of this type are produced by the proof-generating backend to capture
/// a snapshot of the machine state along with access information for leaves
/// which hold data that was used in a particular evaluation step.
#[derive(Debug, Clone)]
pub enum MerkleTree {
    Leaf(Hash, bool, Vec<u8>),
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
    /// let hash = Hash::blake3_hash_bytes(&data);
    ///
    /// assert_eq!(hash, root_hash);
    /// ```
    pub fn root_hash(&self) -> Hash {
        match self {
            Self::Node(hash, _) => *hash,
            Self::Leaf(hash, _, _) => *hash,
        }
    }

    /// Creates a merkle tree which is a single leaf
    pub fn make_merkle_leaf(data: Vec<u8>, access_info: bool) -> Self {
        let hash = Hash::blake3_hash_bytes(&data);
        MerkleTree::Leaf(hash, access_info, data)
    }

    /// Takes a list of children nodes and creates a
    /// new parent node on top of them.
    pub fn make_merkle_node(children: Vec<Self>) -> Self {
        let children_hashes = children.iter().map(|t| t.root_hash());
        let node_hash = Hash::combine(children_hashes);
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
                Self::Leaf(hash, _, data) => &Hash::blake3_hash_bytes(data) == hash,
                Self::Node(hash, children) => {
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
}
