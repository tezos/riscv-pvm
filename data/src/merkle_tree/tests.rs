// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Tests for Merkle trees

use proptest::prelude::*;

use crate::hash::Hash;
use crate::hash::HashError;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::merkle_tree::MerkleTree;
use crate::merkle_tree::MerkleTreeLeafData;

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

        let merkle_proof_leaf = |data: &Vec<u8>, access: bool| -> Result<MerkleProof, HashError> {
            let hash = Hash::hash_bytes(data);
            Ok(MerkleProof::Leaf(if access {
                MerkleProofLeaf::Read(data.clone())
            } else {
                MerkleProofLeaf::Blind(hash)
            }))
        };

        let proof_single_leaves = MerkleProof::node_without_data(vec![
            merkle_proof_leaf(&l[0], false)?,
            merkle_proof_leaf(&l[1], true)?,
        ]);

        // The structure of the original subtree is compressed into a single leaf.
        let proof_no_access = MerkleProof::Leaf(MerkleProofLeaf::Blind(Hash::combine_hashes([
            Hash::hash_bytes(&l[2]),
            Hash::combine_hashes([Hash::hash_bytes(&l[3]), Hash::hash_bytes(&l[4])]),
        ])));

        let proof_read_write_3 = MerkleProof::node_without_data(vec![
            MerkleProof::node_without_data(vec![
                merkle_proof_leaf(&l[5], true)?,
                merkle_proof_leaf(&l[6], true)?,
            ]),
            merkle_proof_leaf(&l[7], true)?,
        ]);

        let proof_read_write_4 = MerkleProof::node_without_data(vec![
            MerkleProof::node_without_data(vec![
                merkle_proof_leaf(&l[8], true)?,
                merkle_proof_leaf(&l[9], true)?,
            ]),
            MerkleProof::node_without_data(vec![
                merkle_proof_leaf(&l[10], true)?,
                merkle_proof_leaf(&l[11], true)?,
            ]),
        ]);

        let proof_combine_isolated = MerkleProof::node_without_data(vec![
            MerkleProof::node_without_data(vec![proof_no_access, proof_read_write_3]),
            proof_read_write_4,
        ]);

        let proof_mix = MerkleProof::node_without_data(vec![
            // The structure of the original subtree is compressed into a single leaf.
            MerkleProof::Leaf(MerkleProofLeaf::Blind(Hash::combine_hashes([
                Hash::hash_bytes(&l[12]),
                Hash::combine_hashes([
                    Hash::hash_bytes(&l[13]),
                    Hash::combine_hashes([Hash::hash_bytes(&l[14]), Hash::hash_bytes(&l[15])]),
                ]),
            ]))),
            MerkleProof::node_without_data(vec![
                MerkleProof::node_without_data(vec![
                    merkle_proof_leaf(&l[16], true)?,
                    MerkleProof::Leaf(MerkleProofLeaf::Blind(Hash::combine_hashes([
                        Hash::hash_bytes(&l[17]),
                        Hash::hash_bytes(&l[18]),
                    ]))),
                ]),
                merkle_proof_leaf(&l[19], true)?,
            ]),
        ]);

        let proof = MerkleProof::node_without_data(vec![
            proof_single_leaves,
            MerkleProof::node_without_data(vec![proof_combine_isolated, proof_mix]),
        ]);

        let merkle_tree_root_hash = merkle_tree.root_hash();

        let compressed_merkle_proof = merkle_tree.clone().compress();
        assert_eq!(compressed_merkle_proof.root_hash(), merkle_tree_root_hash);
        assert_eq!(compressed_merkle_proof, proof);

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
