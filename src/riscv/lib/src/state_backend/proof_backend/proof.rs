// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! ## Module for handling proofs
//!
//! - Serialise & Deserialise a [`Proof`]
//!
//!   Structure of serialisation:
//!   * Final hash state
//!   * Tags which dictate the shape of the proof (a partial Merkle tree)
//!   * Leaf contents
//!
//! - Convert [`octez_riscv_data::merkle_tree::MerkleTree`] to [`MerkleProof`]

use bincode::Encode;
use octez_riscv_data::hash::DIGEST_SIZE;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::serialise;

use crate::pvm::node_pvm::NodePvm;
use crate::state_backend::OwnedProofPart;
use crate::state_backend::ProofError;

pub mod deserialise_owned;
pub mod deserialise_stream;
pub mod deserialiser;

/// Structure of a proof transitioning from state A to state B.
///
/// The proof needs to be able to:
/// - Contain enough information to be able to run a single step on A
/// - Obtain the hash of the state after the step
#[derive(Clone, Debug, PartialEq, Encode)]
pub struct Proof {
    /// State of the final state B
    final_state_hash: Hash,
    /// Partial Merkle tree representation of the initial state
    partial_tree: MerkleProof,
}

impl Proof {
    /// Create a proof from a partial Merkle tree and the final state hash.
    /// The initial state hash is the root hash of the partial Merkle tree so it is not included separately.
    pub fn new(proof: MerkleProof, final_state_hash: Hash) -> Self {
        Self {
            final_state_hash,
            partial_tree: proof,
        }
    }

    /// Get the proof tree.
    pub fn tree(&self) -> &MerkleProof {
        &self.partial_tree
    }

    /// Convert the proof into a Merkle proof tree.
    pub fn into_tree(self) -> MerkleProof {
        self.partial_tree
    }

    /// Get the initial state hash of the proof.
    pub fn initial_state_hash(&self) -> Hash {
        self.partial_tree.root_hash()
    }

    /// Get the final state hash of the proof.
    pub fn final_state_hash(&self) -> Hash {
        self.final_state_hash
    }
}

/// Serialise a [`Proof`] to an array of bytes.
///
/// In the encoding, lengths are not necessary, but tags are,
/// since the tags depend on runtime information and events
pub fn serialise_proof(proof: &Proof) -> Vec<u8> {
    serialise(proof).expect("Serialisation of Merkle proof should not fail")
}

/// Serialise just the proof tree part of a general [`Proof`] object.
///
/// Useful for testing
pub fn serialise_merkle_tree(tree: &MerkleProof) -> Vec<u8> {
    serialise(tree).expect("Serialisation of Merkle tree should not fail")
}

/// When parsing, not enough bytes were provided to successfully complete the operation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Not enough bytes")]
pub struct NotEnoughBytesError;

fn deserialise_final_hash(
    bytes: &mut impl Iterator<Item = u8>,
) -> Result<Hash, NotEnoughBytesError> {
    let mut digest = [0; DIGEST_SIZE];
    for b in digest.iter_mut() {
        match bytes.next() {
            None => return Err(NotEnoughBytesError),
            Some(byte) => *b = byte,
        }
    }
    Ok(Hash::from(digest))
}

/// Deserialise a [`Proof`] from an iterator of bytes.
///
/// Obtain a [`Proof`] and the associated [`NodePvm<Verify>`].
pub fn deserialise_proof<I: Iterator<Item = u8>>(
    mut bytes: I,
) -> deserialiser::Result<(
    Proof,
    // TODO RV-849: use `EmptyPageCache` for verify mode
    NodePvm<Verify>,
)> {
    let final_state_hash =
        deserialise_final_hash(&mut bytes).map_err(|e| ProofError::TagDeserialise(e.into()))?;

    let (pvm, proof_tree) = deserialise_stream::deserialise(bytes.collect::<Vec<u8>>().as_slice())?;

    let merkle_tree = match proof_tree {
        OwnedProofPart::Absent => return Err(ProofError::AbsentProof),
        OwnedProofPart::Present(tree) => tree,
    };

    let pvm = NodePvm::wrap(pvm);
    Ok((Proof::new(merkle_tree, final_state_hash), pvm))
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::hash::DIGEST_SIZE;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
    use octez_riscv_data::merkle_proof::tag::TAG_BLIND;
    use octez_riscv_data::merkle_proof::tag::TAG_NODE;
    use octez_riscv_data::merkle_proof::tag::TAG_READ;
    use proptest::proptest;
    use rand::Fill;

    use super::serialise_proof;
    use crate::state_backend::proof_backend::proof::Proof;

    /// Utility struct that computes the bounds of a [`MerkleProof`] serialisation
    /// based on total number of nodes in the tree and total size of raw data in the leaves.
    struct SerialisationBound {
        nodes_count: u64,
        content_size: u64,
    }

    impl SerialisationBound {
        /// Compute equivalent bound of a node having the children given from the argument.
        fn node_combine(children: Vec<SerialisationBound>) -> SerialisationBound {
            children
                .into_iter()
                // The base fold value has node count 1 to account for the node combining the children
                .fold(
                    SerialisationBound {
                        nodes_count: 1,
                        content_size: 0,
                    },
                    |a, b| SerialisationBound {
                        nodes_count: a.nodes_count + b.nodes_count,
                        content_size: a.content_size + b.content_size,
                    },
                )
        }

        fn expected_serialisation_length(&self) -> usize {
            let hashes_size = 2 * DIGEST_SIZE as u64;

            // Each node tag occupies 1 byte.
            let tags_size = self.nodes_count;

            (hashes_size + tags_size + self.content_size) as usize
        }

        /// Generate the serialisation bound for a [`MerkleProof`] leaf.
        fn from_merkle_leaf(leaf: &MerkleProof) -> Self {
            match leaf {
                MerkleProof::Node(_) => panic!("Expected a Merkle proof leaf"),
                MerkleProof::Leaf(MerkleProofLeaf::Blind(hash)) => Self {
                    nodes_count: 1,
                    content_size: hash.as_ref().len() as u64,
                },
                MerkleProof::Leaf(MerkleProofLeaf::Read(data)) => Self {
                    nodes_count: 1,
                    content_size: data.len() as u64,
                },
            }
        }
    }

    fn generate_rand_leaf() -> MerkleProof {
        let is_leaf_read = rand::random::<bool>();
        let length: usize = rand::random::<u64>() as usize % 100 + 1;

        let mut raw_array = vec![0; length];
        Fill::fill(raw_array.as_mut_slice(), &mut rand::rng());
        let blind_hash: Hash = Hash::blake3_hash_bytes(&raw_array);

        match is_leaf_read {
            true => MerkleProof::Leaf(MerkleProofLeaf::Read(raw_array)),
            false => MerkleProof::Leaf(MerkleProofLeaf::Blind(blind_hash)),
        }
    }

    fn check_serialisation(tree: MerkleProof, tree_correct_bytes: &[u8]) {
        let final_state_hash = Hash::blake3_hash_bytes(&rand::random::<[u8; 10]>());
        let proof = Proof::new(tree, final_state_hash);

        let ser_bytes: Vec<u8> = serialise_proof(&proof);
        assert_eq!(
            ser_bytes.as_slice(),
            &[final_state_hash.as_ref(), tree_correct_bytes].concat()
        );
    }

    #[test]
    fn serialise_leaf_trees() {
        // Check serialisation of all leaf variants

        let raw_array: [u8; 10] = rand::random();

        let rleaf = MerkleProof::Leaf(MerkleProofLeaf::Read(raw_array.to_vec()));
        check_serialisation(rleaf, &[&[TAG_READ], raw_array.as_slice()].concat());

        let hash = Hash::blake3_hash_bytes(&raw_array);
        check_serialisation(
            MerkleProof::Leaf(MerkleProofLeaf::Blind(hash)),
            &[&[TAG_BLIND], hash.as_ref()].concat(),
        );
    }

    #[test]
    fn serialise_1_level() {
        // Check serialisation of a node containing some leaves.

        let h1 = Hash::blake3_hash_bytes(&[1, 2, 3]);
        let h2 = Hash::blake3_hash_bytes(&[20, 30, 1, 5, 6]);

        let n1 = MerkleProof::Leaf(MerkleProofLeaf::Read(vec![12, 15, 30, 40]));
        let n2 = MerkleProof::Leaf(MerkleProofLeaf::Blind(h1));
        let n3 = MerkleProof::Leaf(MerkleProofLeaf::Blind(h2));
        let n4 = MerkleProof::Leaf(MerkleProofLeaf::Read(vec![123, 234, 42, 1, 2, 3]));

        let root = MerkleProof::Node(vec![n1.clone()]);
        check_serialisation(root, &[TAG_NODE, TAG_READ, 12, 15, 30, 40]);

        let root = MerkleProof::Node(vec![n1.clone(), n2.clone()]);
        check_serialisation(
            root,
            &[
                [TAG_NODE].as_ref(),
                [TAG_READ].as_ref(),
                &[12, 15, 30, 40],
                [TAG_BLIND].as_ref(),
                h1.as_ref(),
            ]
            .concat(),
        );

        let root = MerkleProof::Node(vec![n1.clone(), n2.clone(), n3.clone()]);
        check_serialisation(
            root,
            &[
                [TAG_NODE].as_ref(),
                [TAG_READ].as_ref(),
                &[12, 15, 30, 40],
                [TAG_BLIND].as_ref(),
                h1.as_ref(),
                [TAG_BLIND].as_ref(),
                h2.as_ref(),
            ]
            .concat(),
        );

        let root = MerkleProof::Node(vec![n1.clone(), n2.clone(), n4.clone(), n3.clone()]);
        check_serialisation(
            root,
            &[
                [TAG_NODE].as_ref(),
                [TAG_READ].as_ref(),
                &[12, 15, 30, 40],
                [TAG_BLIND].as_ref(),
                h1.as_ref(),
                [TAG_READ].as_ref(),
                &[123, 234, 42, 1, 2, 3],
                [TAG_BLIND].as_ref(),
                h2.as_ref(),
            ]
            .concat(),
        )
    }

    fn check_bounds(tree: MerkleProof, bound: &SerialisationBound) {
        let final_state_hash = Hash::blake3_hash_bytes(&rand::random::<[u8; 10]>());
        let proof = Proof::new(tree, final_state_hash);

        let serialisation: Vec<_> = super::serialise_proof(&proof);
        assert!(serialisation.len() <= bound.expected_serialisation_length());
    }

    #[test]
    fn bounds_1_level() {
        // Check size of the serialisations for a node having a number of leaves.

        for i in 1..20 {
            let children: Vec<_> = (0..i).map(|_| generate_rand_leaf()).collect();
            let bounds = children
                .iter()
                .map(SerialisationBound::from_merkle_leaf)
                .collect();

            let root = MerkleProof::Node(children);
            let bound = SerialisationBound::node_combine(bounds);

            check_bounds(root, &bound);
        }
    }

    #[test]
    fn bounds_n_levels() {
        // Check size of serialisation of a randomly generated Merkle tree.

        // Starting from an array of leaves, combine a randomly generated number of consecutive nodes
        // into a node for the level above, This will continue until only a node is left, which is the root.

        proptest!(|(total_length in 100..300)| {
            let mut nodes: Vec<_> = (0..total_length).map(|_| {
                let leaf = generate_rand_leaf();
                let bound = SerialisationBound::from_merkle_leaf(&leaf);
                (leaf, bound)
            }).collect();

            while nodes.len() > 1 {
                let mut new_nodes = vec![];
                let mut iter = nodes.into_iter();

                let mut nothing_taken = false;
                while !nothing_taken {
                    // wanted number of children of a node is between 2 and 10.
                    // (Note, if only a node is left, then there will be only one child)
                    let nr_children = rand::random::<u64>() % 8 + 2;

                    let mut new_children = vec![];
                    let mut new_bounds = vec![];
                    for _ in 0..nr_children {
                        if let Some((child, bound)) = iter.next() {
                            new_children.push(child);
                            new_bounds.push(bound);
                        }
                    }

                    nothing_taken = new_bounds.is_empty();

                    if !nothing_taken {
                        let node = MerkleProof::Node(new_children);
                        let bound = SerialisationBound::node_combine(new_bounds);

                        new_nodes.push((node, bound));
                    }
                }

                nodes = new_nodes;
            }

            let (root, bound) = &nodes[0];
            check_bounds(root.clone(), bound);
        });
    }
}
