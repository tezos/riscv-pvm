// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use crate::hash::Hash;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::merkle_tree::MerkleTree;
use crate::merkle_tree::MerkleTreeLeafData;
use crate::merkle_tree::MerkleTreeNodeData;
use crate::tree::Tree;

// TODO RV-322: Choose optimal Merkleisation parameters for main memory.
/// Size of the Merkle leaf used for Merkleising [`DynArrays`].
///
/// [`DynArrays`]: [`crate::state_backend::layout::DynArray`]
pub const MERKLE_LEAF_SIZE: NonZeroUsize = NonZeroUsize::new(4096).unwrap();

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

impl CompressedAccessInfo {
    pub fn from_access_info(access_info: bool, data: Vec<u8>) -> Self {
        if access_info {
            Self::ReadWrite(data)
        } else {
            Self::NoAccess
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressedMerkleTreeLeafData {
    pub hash: Hash,
    pub compressed_access_info: CompressedAccessInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressedMerkleTreeNodeData {
    pub hash: Hash,
}

/// Intermediary representation obtained when compressing a [`MerkleTree`].
///
/// For the compressed tree, we only care about the data in the non-blinded leaves.
pub type CompressedMerkleTree = Tree<CompressedMerkleTreeLeafData, CompressedMerkleTreeNodeData>;

impl CompressedMerkleTree {
    pub fn to_proof(self) -> MerkleProof {
        match self {
            CompressedMerkleTree::Leaf {
                data:
                    CompressedMerkleTreeLeafData {
                        hash,
                        compressed_access_info,
                    },
            } => match compressed_access_info {
                CompressedAccessInfo::NoAccess => Tree::Leaf {
                    data: MerkleProofLeaf::Blind(hash),
                },
                CompressedAccessInfo::ReadWrite(data) => Tree::Leaf {
                    data: MerkleProofLeaf::Read(data),
                },
            },
            CompressedMerkleTree::Node { children, .. } => Tree::Node {
                data: Default::default(),
                children: children.into_iter().map(|child| child.to_proof()).collect(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct MerkleTreeCompressionError;

fn maybe_compress_node(hash: Hash, children: Vec<CompressedMerkleTree>) -> CompressedMerkleTree {
    let mut none_accessed = true;
    let mut hasher = blake3::Hasher::new();
    for child in children.iter() {
        match child {
            CompressedMerkleTree::Leaf {
                data:
                    CompressedMerkleTreeLeafData {
                        hash: leaf_hash,
                        compressed_access_info,
                    },
            } => {
                hasher.update(leaf_hash.as_ref());
                match compressed_access_info {
                    CompressedAccessInfo::ReadWrite(_) => none_accessed = false,
                    _ => {}
                };
            }
            CompressedMerkleTree::Node {
                data: CompressedMerkleTreeNodeData { hash: node_hash },
                ..
            } => {
                none_accessed = false;
                hasher.update(node_hash.as_ref());
            }
        }
    }
    if none_accessed {
        CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: Hash::new(hasher.finalize().into()),
                compressed_access_info: CompressedAccessInfo::NoAccess,
            },
        }
    } else {
        CompressedMerkleTree::Node {
            data: CompressedMerkleTreeNodeData { hash },
            children,
        }
    }
}

fn merkle_child_to_compressed_child(
    child: MerkleTree,
) -> Result<CompressedMerkleTree, MerkleTreeCompressionError> {
    if let MerkleTree::Leaf {
        data:
            MerkleTreeLeafData {
                hash,
                access_info,
                data,
            },
    } = child
    {
        if access_info {
            Ok(CompressedMerkleTree::Leaf {
                data: CompressedMerkleTreeLeafData {
                    hash,
                    compressed_access_info: CompressedAccessInfo::ReadWrite(data),
                },
            })
        } else {
            Ok(CompressedMerkleTree::Leaf {
                data: CompressedMerkleTreeLeafData {
                    hash,
                    compressed_access_info: CompressedAccessInfo::NoAccess,
                },
            })
        }
    } else {
        Err(MerkleTreeCompressionError)
    }
}

/// Turns a [MerkleTree] into a [CompressedMerkleTree]
pub fn merkle_tree_to_compressed_merkle_tree(
    merkle_tree: MerkleTree,
) -> Result<CompressedMerkleTree, MerkleTreeCompressionError> {
    let mut nodes: Vec<(MerkleTree, usize)> = vec![(merkle_tree, 0)];
    let mut compressed_nodes: Vec<(CompressedMerkleTree, usize)> = vec![];

    while let Some((node, parent_index)) = nodes.pop() {
        match node {
            leaf @ MerkleTree::Leaf { .. } => {
                compressed_nodes.push((merkle_child_to_compressed_child(leaf)?, parent_index));
            }
            MerkleTree::Node {
                data: MerkleTreeNodeData { hash },
                children,
            } => {
                compressed_nodes.push((
                    CompressedMerkleTree::Node {
                        data: CompressedMerkleTreeNodeData { hash },
                        children: vec![],
                    },
                    parent_index,
                ));
                let new_parent_index = compressed_nodes.len() - 1;
                for child in children {
                    nodes.push((child, new_parent_index));
                }
            }
        }
    }

    while compressed_nodes.len() > 1 {
        let (compressed_node, parent_index) = compressed_nodes.pop().unwrap();
        if let (CompressedMerkleTree::Node { children, .. }, _) =
            &mut compressed_nodes[parent_index]
        {
            match compressed_node {
                leaf @ CompressedMerkleTree::Leaf { .. } => children.push(leaf),
                CompressedMerkleTree::Node {
                    data: CompressedMerkleTreeNodeData { hash },
                    children: node_children,
                } => children.push(maybe_compress_node(hash, node_children)),
            }
        } else {
            panic!("This should not happen");
        }
    }

    Ok(compressed_nodes.pop().unwrap().0)
}

/// Turns a [MerkleTree] into a [MerkleProof]
pub fn merkle_tree_to_merkle_proof(merkle_tree: MerkleTree) -> MerkleProof {
    merkle_tree_to_compressed_merkle_tree(merkle_tree)
        .expect("This conversion should not fail")
        .to_proof()
}

/// Helper function which allows iterating over chunks of a dynamic array
/// and writing them to a writer. The last chunk may be smaller than the
/// Merkle leaf size. The implementations of [`HashState`] and
/// [`ProofLayout`] both use it, ensuring consistency between the two.
///
/// [`HashState`]: octez_riscv_data::hash::HashState
/// [`ProofLayout`]: crate::state_backend::proof_layout::ProofLayout
pub fn chunks_to_writer<T: std::io::Write, F: Fn(usize) -> [u8; MERKLE_LEAF_SIZE.get()]>(
    writer: &mut T,
    len: usize,
    read: F,
) -> Result<(), std::io::Error> {
    let merkle_leaf_size = MERKLE_LEAF_SIZE.get();
    assert!(len >= merkle_leaf_size);

    let mut address = 0;

    while address + merkle_leaf_size <= len {
        writer.write_all(read(address).as_slice())?;
        address += merkle_leaf_size;
    }

    // When the last chunk is smaller than `MERKLE_LEAF_SIZE`,
    // read the last `MERKLE_LEAF_SIZE` bytes and pass a subslice containing
    // only the bytes not previously read to the writer.
    if address != len {
        address += merkle_leaf_size;
        let buffer = read(len.saturating_sub(merkle_leaf_size));
        writer.write_all(&buffer[address.saturating_sub(len)..])?;
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use proptest::prelude::*;

    use super::CompressedAccessInfo;
    use super::CompressedMerkleTree;
    use super::CompressedMerkleTreeLeafData;
    use super::CompressedMerkleTreeNodeData;
    use super::MERKLE_LEAF_SIZE;
    use super::chunks_to_writer;
    use super::merkle_tree_to_compressed_merkle_tree;
    use super::merkle_tree_to_merkle_proof;
    use crate::hash::Hash;
    use crate::hash::HashError;
    use crate::merkle_proof::proof_tree::MerkleProof;
    use crate::merkle_proof::proof_tree::MerkleProofLeaf;
    use crate::merkle_tree::MerkleTree;
    use crate::merkle_tree::MerkleTreeLeafData;

    impl CompressedMerkleTree {
        /// Get the root hash of a compressed Merkle tree
        fn root_hash(&self) -> Hash {
            match self {
                Self::Node {
                    data: CompressedMerkleTreeNodeData { hash },
                    ..
                } => *hash,
                Self::Leaf {
                    data: CompressedMerkleTreeLeafData { hash, .. },
                } => *hash,
            }
        }

        /// Check the validity of the Merkle root by recomputing all hashes
        fn check_root_hash(&self) -> bool {
            let mut deque = std::collections::VecDeque::new();
            deque.push_back(self);

            while let Some(node) = deque.pop_front() {
                let is_valid_hash = match node {
                    Self::Leaf {
                        data:
                            CompressedMerkleTreeLeafData {
                                hash,
                                compressed_access_info,
                            },
                    } => match compressed_access_info {
                        CompressedAccessInfo::NoAccess => true,
                        CompressedAccessInfo::ReadWrite(data) => {
                            &Hash::blake3_hash_bytes(data) == hash
                        }
                    },
                    Self::Node {
                        data: CompressedMerkleTreeNodeData { hash },
                        children,
                    } => {
                        let children_hashes = children.iter().map(|child| {
                            deque.push_back(child);
                            child.root_hash()
                        });
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

    fn m_l(data: &[u8], access: bool) -> Result<MerkleTree, HashError> {
        let hash = Hash::blake3_hash_bytes(data);
        Ok(MerkleTree::Leaf {
            data: MerkleTreeLeafData {
                hash,
                access_info: access,
                data: data.to_vec(),
            },
        })
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
                    let hash = Hash::blake3_hash_bytes(data);
                    Ok(MerkleProof::Leaf {
                        data: if access {
                            MerkleProofLeaf::Read(data.clone())
                        } else {
                            MerkleProofLeaf::Blind(hash)
                        },
                    })
                };

            let proof_single_leaves = MerkleProof::Node {
                data: Default::default(),
                children: vec![
                    merkle_proof_leaf(&l[0], false)?,
                    merkle_proof_leaf(&l[1], true)?,
                ],
            };

            // The structure of the original subtree is compressed into a single leaf.
            let proof_no_access = MerkleProof::Leaf {
                data: MerkleProofLeaf::Blind(Hash::combine([
                    Hash::blake3_hash_bytes(&l[2]),
                    Hash::combine([
                        Hash::blake3_hash_bytes(&l[3]),
                        Hash::blake3_hash_bytes(&l[4]),
                    ]),
                ])),
            };

            let proof_read_write_3 = MerkleProof::Node {
                data: Default::default(),
                children: vec![
                    MerkleProof::Node {
                        data: Default::default(),
                        children: vec![
                            merkle_proof_leaf(&l[5], true)?,
                            merkle_proof_leaf(&l[6], true)?,
                        ],
                    },
                    merkle_proof_leaf(&l[7], true)?,
                ],
            };

            let proof_read_write_4 = MerkleProof::Node {
                data: Default::default(),
                children: vec![
                    MerkleProof::Node {
                        data: Default::default(),
                        children: vec![
                            merkle_proof_leaf(&l[8], true)?,
                            merkle_proof_leaf(&l[9], true)?,
                        ],
                    },
                    MerkleProof::Node {
                        data: Default::default(),
                        children: vec![
                            merkle_proof_leaf(&l[10], true)?,
                            merkle_proof_leaf(&l[11], true)?,
                        ],
                    },
                ],
            };

            let proof_combine_isolated = MerkleProof::Node {
                data: Default::default(),
                children: vec![
                    MerkleProof::Node {
                        data: Default::default(),
                        children: vec![proof_no_access, proof_read_write_3],
                    },
                    proof_read_write_4,
                ],
            };

            let proof_mix = MerkleProof::Node {
                data: Default::default(),
                children: vec![
                    // The structure of the original subtree is compressed into a single leaf.
                    MerkleProof::Leaf {
                        data: MerkleProofLeaf::Blind(Hash::combine([
                            Hash::blake3_hash_bytes(&l[12]),
                            Hash::combine([
                                Hash::blake3_hash_bytes(&l[13]),
                                Hash::combine([
                                    Hash::blake3_hash_bytes(&l[14]),
                                    Hash::blake3_hash_bytes(&l[15]),
                                ]),
                            ]),
                        ])),
                    },
                    MerkleProof::Node {
                        data: Default::default(),
                        children: vec![
                            MerkleProof::Node {
                                data: Default::default(),
                                children: vec![
                                    merkle_proof_leaf(&l[16], true)?,
                                    MerkleProof::Leaf {
                                        data: MerkleProofLeaf::Blind(Hash::combine([
                                            Hash::blake3_hash_bytes(&l[17]),
                                            Hash::blake3_hash_bytes(&l[18]),
                                        ])),
                                    },
                                ],
                            },
                            merkle_proof_leaf(&l[19], true)?,
                        ],
                    },
                ],
            };

            let proof = MerkleProof::Node {
                data: Default::default(),
                children: vec![proof_single_leaves, MerkleProof::Node {
                    data: Default::default(),
                    children: vec![proof_combine_isolated, proof_mix],
                }],
            };

            let merkle_tree_root_hash = merkle_tree.root_hash();

            let compressed_merkle_tree = merkle_tree_to_compressed_merkle_tree(merkle_tree.clone());
            assert!(compressed_merkle_tree.check_root_hash());
            assert_eq!(compressed_merkle_tree.root_hash(), merkle_tree_root_hash);

            let compressed_merkle_proof = compressed_merkle_tree.clone().to_proof();
            assert_eq!(compressed_merkle_proof, proof);

            assert_eq!(merkle_tree_to_merkle_proof(merkle_tree), proof);
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

    fn check(compressed_merkle_tree: CompressedMerkleTree, merkle_proof: MerkleProof) {
        let proof_from_compressed_merkle_tree = compressed_merkle_tree.to_proof();
        assert_eq!(proof_from_compressed_merkle_tree, merkle_proof);
    }

    #[test]
    fn transform_compressed_merkle_tree_to_proof() {
        use CompressedAccessInfo::*;

        let gen_hash_data = || {
            let data = rand::random::<[u8; 12]>().to_vec();
            let hash = Hash::blake3_hash_bytes(&data);
            (data, hash)
        };

        let (data, hash) = gen_hash_data();

        // Check leaves
        check(
            CompressedMerkleTree::Leaf {
                data: CompressedMerkleTreeLeafData {
                    hash,
                    compressed_access_info: NoAccess,
                },
            },
            MerkleProof::Leaf {
                data: MerkleProofLeaf::Blind(hash),
            },
        );
        check(
            CompressedMerkleTree::Leaf {
                data: CompressedMerkleTreeLeafData {
                    hash,
                    compressed_access_info: ReadWrite(data.clone()),
                },
            },
            MerkleProof::Leaf {
                data: MerkleProofLeaf::Read(data.clone()),
            },
        );

        // Check nodes
        let [d0, d1, d2, d3, d4, d5, d6, d7, d8] = [0; 9].map(|_| gen_hash_data());
        let l0 = MerkleProof::Leaf {
            data: MerkleProofLeaf::Read(d0.0.clone()),
        };
        let l1 = MerkleProof::Leaf {
            data: MerkleProofLeaf::Read(d1.0.clone()),
        };
        let l2 = MerkleProof::Leaf {
            data: MerkleProofLeaf::Read(d2.0.clone()),
        };
        let l3 = MerkleProof::Leaf {
            data: MerkleProofLeaf::Blind(d3.1),
        };
        let l4 = MerkleProof::Leaf {
            data: MerkleProofLeaf::Blind(d4.1),
        };
        let l5 = MerkleProof::Leaf {
            data: MerkleProofLeaf::Blind(d5.1),
        };

        let n6 = MerkleProof::Node {
            data: Default::default(),
            children: vec![l0, l1, l3],
        };
        let n7 = MerkleProof::Node {
            data: Default::default(),
            children: vec![l4, l2, l5],
        };
        let root = MerkleProof::Node {
            data: Default::default(),
            children: vec![n6, n7],
        };

        let t0 = CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: d0.1,
                compressed_access_info: ReadWrite(d0.0),
            },
        };
        let t1 = CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: d1.1,
                compressed_access_info: ReadWrite(d1.0),
            },
        };
        let t2 = CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: d2.1,
                compressed_access_info: ReadWrite(d2.0),
            },
        };
        let t3 = CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: d3.1,
                compressed_access_info: NoAccess,
            },
        };
        let t4 = CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: d4.1,
                compressed_access_info: NoAccess,
            },
        };
        let t5 = CompressedMerkleTree::Leaf {
            data: CompressedMerkleTreeLeafData {
                hash: d5.1,
                compressed_access_info: NoAccess,
            },
        };

        let t6 = CompressedMerkleTree::Node {
            data: CompressedMerkleTreeNodeData { hash: d6.1 },
            children: vec![t0, t1, t3],
        };
        let t7 = CompressedMerkleTree::Node {
            data: CompressedMerkleTreeNodeData { hash: d7.1 },
            children: vec![t4, t2, t5],
        };
        let t_root = CompressedMerkleTree::Node {
            data: CompressedMerkleTreeNodeData { hash: d8.1 },
            children: vec![t6, t7],
        };

        check(t_root, root);
    }

    const LENS: [usize; 4] = [0, 1, 535, 4095];
    const _: () = {
        if MERKLE_LEAF_SIZE.get() != 4096 {
            panic!(
                "Test values in `LENS` assume a specific MERKLE_LEAF_SIZE, change them accordingly"
            );
        }
    };

    // Test `chunks_to_writer` with a variety of lengths
    macro_rules! generate_test_chunks_to_writer {
        ( $name:ident, $i:expr ) => {
            proptest! {
                #[test]
                fn $name(
                    data in proptest::collection::vec(any::<u8>(), 4 * MERKLE_LEAF_SIZE.get() + LENS[$i])
                ) {
                    const LEN: usize = 4 * MERKLE_LEAF_SIZE.get() + LENS[$i];

                    let read = |pos: usize| {
                        assert!(pos + MERKLE_LEAF_SIZE.get() <= LEN);
                        data[pos..pos + MERKLE_LEAF_SIZE.get()].try_into().unwrap()
                    };

                    let mut writer = Cursor::new(Vec::new());
                    chunks_to_writer::< _, _>(&mut writer, LEN, read).unwrap();
                    assert_eq!(writer.into_inner(), data);
                }
            }
        }
    }

    generate_test_chunks_to_writer!(test_chunks_to_writer_0, 0);
    generate_test_chunks_to_writer!(test_chunks_to_writer_1, 1);
    generate_test_chunks_to_writer!(test_chunks_to_writer_2, 2);
    generate_test_chunks_to_writer!(test_chunks_to_writer_3, 3);
}
