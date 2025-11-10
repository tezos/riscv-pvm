// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Merkle trees used for proof generation by the PVM

use std::num::NonZeroUsize;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::Hasher;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
use octez_riscv_data::merkle_tree::MerkleTree;
use octez_riscv_data::merkle_tree::MerkleTreeLeafData;
use octez_riscv_data::tree::Tree;

// TODO RV-322: Choose optimal Merkleisation parameters for main memory.
/// Size of the Merkle leaf used for Merkleising [`DynArrays`].
///
/// [`DynArrays`]: [`crate::state_backend::layout::DynArray`]
pub const MERKLE_LEAF_SIZE: NonZeroUsize = NonZeroUsize::new(4096).unwrap();

// TODO RV-322: Choose optimal Merkleisation parameters for main memory.
/// Arity of the Merkle tree used for Merkleising [`DynArrays`].
///
/// [`DynArrays`]: [`crate::state_backend::layout::DynArray`]
pub const MERKLE_ARITY: usize = 4;

/// Turns a [MerkleTree] into a [CompressedMerkleTree]
pub(crate) fn merkle_tree_to_compressed_merkle_tree(
    merkle_tree: MerkleTree,
) -> CompressedMerkleTree {
    let mut nodes: Vec<(MerkleTree, usize)> = vec![(merkle_tree, 0)];
    let mut compressed_nodes: Vec<(CompressedMerkleTree, usize)> = vec![];

    while let Some((node, parent_index)) = nodes.pop() {
        match node {
            MerkleTree::Leaf(leaf_data) => {
                compressed_nodes.push((CompressedMerkleTree::leaf(leaf_data), parent_index));
            }
            MerkleTree::Node(hash, children) => {
                compressed_nodes.push((CompressedMerkleTree::Node(hash, vec![]), parent_index));
                let new_parent_index = compressed_nodes.len() - 1;
                for child in children {
                    nodes.push((child, new_parent_index));
                }
            }
        }
    }

    while compressed_nodes.len() > 1 {
        let (compressed_node, parent_index) = compressed_nodes
            .pop()
            .expect("compressed_nodes can't be empty");
        if let (CompressedMerkleTree::Node(_, children), _) = &mut compressed_nodes[parent_index] {
            match compressed_node {
                leaf @ CompressedMerkleTree::Leaf(_, _) => children.push(leaf),
                CompressedMerkleTree::Node(hash, node_children) => {
                    children.push(CompressedMerkleTree::node(hash, node_children))
                }
            }
        } else {
            unreachable!("The parent of a node must exist and must not be a leaf");
        }
    }

    compressed_nodes
        .pop()
        .expect("compressed_nodes can't be empty")
        .0
}

/// Intermediary representation obtained when compressing a [`MerkleTree`].
///
/// For the compressed tree, we only care about the data in the non-blinded leaves.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressedMerkleTree {
    Leaf(Hash, CompressedAccessInfo),
    Node(Hash, Vec<Self>),
}

impl CompressedMerkleTree {
    /// Turns a [`CompressedMerkleTree`] into a [`MerkleProof`]
    pub fn to_proof(self) -> MerkleProof {
        match self {
            CompressedMerkleTree::Leaf(hash, compressed_access_info) => {
                match compressed_access_info {
                    CompressedAccessInfo::NoAccess => Tree::Leaf(MerkleProofLeaf::Blind(hash)),
                    CompressedAccessInfo::ReadWrite(data) => {
                        Tree::Leaf(MerkleProofLeaf::Read(data))
                    }
                }
            }
            CompressedMerkleTree::Node(_, children) => {
                Tree::Node(children.into_iter().map(|child| child.to_proof()).collect())
            }
        }
    }

    fn leaf(leaf: MerkleTreeLeafData) -> CompressedMerkleTree {
        if leaf.access_info {
            CompressedMerkleTree::Leaf(leaf.hash, CompressedAccessInfo::ReadWrite(leaf.data))
        } else {
            CompressedMerkleTree::Leaf(leaf.hash, CompressedAccessInfo::NoAccess)
        }
    }

    fn node(hash: Hash, children: Vec<CompressedMerkleTree>) -> CompressedMerkleTree {
        let mut none_accessed = true;
        let mut hasher = Hasher::default();
        for child in children.iter() {
            match child {
                CompressedMerkleTree::Leaf(leaf_hash, access_info) => {
                    hasher.update_with_hash(leaf_hash);
                    if let CompressedAccessInfo::ReadWrite(_) = access_info {
                        none_accessed = false;
                    }
                }
                CompressedMerkleTree::Node(node_hash, _) => {
                    none_accessed = false;
                    hasher.update_with_hash(node_hash);
                }
            }
        }
        if none_accessed {
            CompressedMerkleTree::Leaf(hasher.to_hash(), CompressedAccessInfo::NoAccess)
        } else {
            CompressedMerkleTree::Node(hash, children)
        }
    }
}

/// Type of access associated with leaves in a [`CompressedMerkleTree`].
///
/// If a subtree only contains leaves which have not been accessed, it can be
/// compressed into a blinded leaf. Leaves which have been accessed also hold
/// the leaf data.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressedAccessInfo {
    /// A leaf which has not been accessed
    NoAccess,
    /// A leaf which has been accessed
    ReadWrite(Vec<u8>),
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::HashError;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
    use octez_riscv_data::merkle_tree::MerkleTreeLeafData;
    use proptest::prelude::*;

    use super::CompressedAccessInfo;
    use super::CompressedMerkleTree;
    use super::MerkleTree;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_compressed_merkle_tree;

    impl CompressedMerkleTree {
        /// Get the root hash of a compressed Merkle tree
        fn root_hash(&self) -> Hash {
            match self {
                Self::Node(hash, _) => *hash,
                Self::Leaf(hash, _) => *hash,
            }
        }

        /// Check the validity of the Merkle root by recomputing all hashes
        fn check_root_hash(&self) -> bool {
            let mut deque = std::collections::VecDeque::new();
            deque.push_back(self);

            while let Some(node) = deque.pop_front() {
                let is_valid_hash = match node {
                    Self::Leaf(hash, access_info) => match access_info {
                        CompressedAccessInfo::NoAccess => true,
                        CompressedAccessInfo::ReadWrite(data) => &Hash::hash_bytes(data) == hash,
                    },
                    Self::Node(hash, children) => {
                        let children_hashes = children.iter().map(|child| {
                            deque.push_back(child);
                            child.root_hash()
                        });
                        &Hash::combine_hashes(children_hashes) == hash
                    }
                };
                if !is_valid_hash {
                    return false;
                }
            }
            true
        }
    }

    fn m_l(data: &[u8], access: bool) -> Result<MerkleTree, HashError> {
        let hash = Hash::hash_bytes(data);
        Ok(MerkleTree::Leaf(MerkleTreeLeafData {
            hash,
            access_info: access,
            data: data.to_vec(),
        }))
    }

    fn m_t(left: MerkleTree, right: MerkleTree) -> MerkleTree {
        MerkleTree::make_merkle_node(vec![left, right])
    }

    #[test]
    fn test_compression() {
        let test = |l: Vec<Vec<u8>>| -> Result<_, HashError> {
            // The LHS leaf will be blinded
            let single_leaves_t = m_t(m_l(&l[0], false)?, m_l(&l[1], true)?);

            // The whole subtree will be blinded and compressed
            let no_access_t = m_t(
                m_l(&l[2], false)?,
                m_t(m_l(&l[3], false)?, m_l(&l[4], false)?),
            );

            // No leaf will be blinded, the tree will not be compressed
            let read_write_3_t = m_t(m_t(m_l(&l[5], true)?, m_l(&l[6], true)?), m_l(&l[7], true)?);

            // No leaf will be blinded, the tree will not be compressed
            let read_write_4_t = m_t(
                m_t(m_l(&l[8], true)?, m_l(&l[9], true)?),
                m_t(m_l(&l[10], true)?, m_l(&l[11], true)?),
            );

            let combine_isolated_t = m_t(m_t(no_access_t, read_write_3_t), read_write_4_t);

            let mix_t = m_t(
                // The whole subtree will be compressed
                m_t(
                    m_l(&l[12], false)?,
                    m_t(
                        m_l(&l[13], false)?,
                        m_t(m_l(&l[14], false)?, m_l(&l[15], false)?),
                    ),
                ),
                m_t(
                    m_t(
                        m_l(&l[16], true)?,
                        // Only the non-accessed leaves will be compressed
                        m_t(m_l(&l[17], false)?, m_l(&l[18], false)?),
                    ),
                    m_l(&l[19], true)?,
                ),
            );

            let merkle_tree = m_t(single_leaves_t, m_t(combine_isolated_t, mix_t));

            let merkle_proof_leaf =
                |data: &Vec<u8>, access: bool| -> Result<MerkleProof, HashError> {
                    let hash = Hash::hash_bytes(data);
                    Ok(MerkleProof::Leaf(if access {
                        MerkleProofLeaf::Read(data.clone())
                    } else {
                        MerkleProofLeaf::Blind(hash)
                    }))
                };

            let proof_single_leaves = MerkleProof::Node(vec![
                merkle_proof_leaf(&l[0], false)?,
                merkle_proof_leaf(&l[1], true)?,
            ]);

            // The structure of the original subtree is compressed into a single leaf.
            let proof_no_access =
                MerkleProof::Leaf(MerkleProofLeaf::Blind(Hash::combine_hashes([
                    Hash::hash_bytes(&l[2]),
                    Hash::combine_hashes([Hash::hash_bytes(&l[3]), Hash::hash_bytes(&l[4])]),
                ])));

            let proof_read_write_3 = MerkleProof::Node(vec![
                MerkleProof::Node(vec![
                    merkle_proof_leaf(&l[5], true)?,
                    merkle_proof_leaf(&l[6], true)?,
                ]),
                merkle_proof_leaf(&l[7], true)?,
            ]);

            let proof_read_write_4 = MerkleProof::Node(vec![
                MerkleProof::Node(vec![
                    merkle_proof_leaf(&l[8], true)?,
                    merkle_proof_leaf(&l[9], true)?,
                ]),
                MerkleProof::Node(vec![
                    merkle_proof_leaf(&l[10], true)?,
                    merkle_proof_leaf(&l[11], true)?,
                ]),
            ]);

            let proof_combine_isolated = MerkleProof::Node(vec![
                MerkleProof::Node(vec![proof_no_access, proof_read_write_3]),
                proof_read_write_4,
            ]);

            let proof_mix = MerkleProof::Node(vec![
                // The structure of the original subtree is compressed into a single leaf.
                MerkleProof::Leaf(MerkleProofLeaf::Blind(Hash::combine_hashes([
                    Hash::hash_bytes(&l[12]),
                    Hash::combine_hashes([
                        Hash::hash_bytes(&l[13]),
                        Hash::combine_hashes([Hash::hash_bytes(&l[14]), Hash::hash_bytes(&l[15])]),
                    ]),
                ]))),
                MerkleProof::Node(vec![
                    MerkleProof::Node(vec![
                        merkle_proof_leaf(&l[16], true)?,
                        MerkleProof::Leaf(MerkleProofLeaf::Blind(Hash::combine_hashes([
                            Hash::hash_bytes(&l[17]),
                            Hash::hash_bytes(&l[18]),
                        ]))),
                    ]),
                    merkle_proof_leaf(&l[19], true)?,
                ]),
            ]);

            let proof = MerkleProof::Node(vec![
                proof_single_leaves,
                MerkleProof::Node(vec![proof_combine_isolated, proof_mix]),
            ]);

            let merkle_tree_root_hash = merkle_tree.root_hash();

            let compressed_merkle_tree = merkle_tree_to_compressed_merkle_tree(merkle_tree.clone());
            assert!(compressed_merkle_tree.check_root_hash());
            assert_eq!(compressed_merkle_tree.root_hash(), merkle_tree_root_hash);

            let compressed_merkle_proof = compressed_merkle_tree.clone().to_proof();
            assert_eq!(compressed_merkle_proof, proof);

            assert_eq!(
                merkle_tree_to_compressed_merkle_tree(merkle_tree).to_proof(),
                proof
            );
            assert_eq!(compressed_merkle_proof.root_hash(), merkle_tree_root_hash);

            Ok(())
        };

        // this whole proptest macro delegates to a pure rust function in order to have easy access to formatting
        proptest!(|(l in prop::collection::vec(
            prop::collection::vec(0u8..255, 0..100),
            20
        ))| {
            test(l).expect("Unexpected Hashing error");
        });
    }

    fn check_tree_proof_consistency(
        compressed_merkle_tree: CompressedMerkleTree,
        merkle_proof: MerkleProof,
    ) {
        let proof_from_compressed_merkle_tree = compressed_merkle_tree.to_proof();
        assert_eq!(proof_from_compressed_merkle_tree, merkle_proof);
    }

    #[test]
    fn transform_compressed_merkle_tree_to_proof() {
        use CompressedAccessInfo::*;

        let gen_hash_data = || {
            let data = rand::random::<[u8; 12]>().to_vec();
            let hash = Hash::hash_bytes(&data);
            (data, hash)
        };

        let (data, hash) = gen_hash_data();

        // Check leaves
        check_tree_proof_consistency(
            CompressedMerkleTree::Leaf(hash, NoAccess),
            MerkleProof::Leaf(MerkleProofLeaf::Blind(hash)),
        );
        check_tree_proof_consistency(
            CompressedMerkleTree::Leaf(hash, ReadWrite(data.clone())),
            MerkleProof::Leaf(MerkleProofLeaf::Read(data.clone())),
        );

        // Check nodes
        let [d0, d1, d2, d3, d4, d5, d6, d7, d8] = [0; 9].map(|_| gen_hash_data());
        let l0 = MerkleProof::Leaf(MerkleProofLeaf::Read(d0.0.clone()));
        let l1 = MerkleProof::Leaf(MerkleProofLeaf::Read(d1.0.clone()));
        let l2 = MerkleProof::Leaf(MerkleProofLeaf::Read(d2.0.clone()));
        let l3 = MerkleProof::Leaf(MerkleProofLeaf::Blind(d3.1));
        let l4 = MerkleProof::Leaf(MerkleProofLeaf::Blind(d4.1));
        let l5 = MerkleProof::Leaf(MerkleProofLeaf::Blind(d5.1));

        let n6 = MerkleProof::Node(vec![l0, l1, l3]);
        let n7 = MerkleProof::Node(vec![l4, l2, l5]);
        let root = MerkleProof::Node(vec![n6, n7]);

        let t0 = CompressedMerkleTree::Leaf(d0.1, ReadWrite(d0.0));
        let t1 = CompressedMerkleTree::Leaf(d1.1, ReadWrite(d1.0));
        let t2 = CompressedMerkleTree::Leaf(d2.1, ReadWrite(d2.0));
        let t3 = CompressedMerkleTree::Leaf(d3.1, NoAccess);
        let t4 = CompressedMerkleTree::Leaf(d4.1, NoAccess);
        let t5 = CompressedMerkleTree::Leaf(d5.1, NoAccess);

        let t6 = CompressedMerkleTree::Node(d6.1, vec![t0, t1, t3]);
        let t7 = CompressedMerkleTree::Node(d7.1, vec![t4, t2, t5]);
        let t_root = CompressedMerkleTree::Node(d8.1, vec![t6, t7]);

        check_tree_proof_consistency(t_root, root);
    }
}
