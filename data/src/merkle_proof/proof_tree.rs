// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

use bincode::enc::write::Writer;

use super::tag::LeafTag;
use super::tag::Tag;
use crate::hash::Hash;
use crate::tree::Tree;

/// Merkle proof tree structure.
///
/// Leaves can be read and/or written to.
/// If a read was done, the content will be stored in the proof.
/// If a write was done, the written content is not necessary since the semantics of running the step will
/// deduce the written contents.
///
/// A proof will have the shape of a subtree of a [`MerkleTree`].
/// The structure of the full [`MerkleTree`] is known statically (since it represents the whole state of the PVM)
/// so the number of children of a node and the sizes of the leaves
/// do not need to be stored in either the proof or its encoding.
///
/// [`MerkleTree`]: crate::merkle_tree::MerkleTree
pub type MerkleProof = Tree<MerkleProofLeaf>;

impl bincode::Encode for MerkleProof {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let mut nodes = vec![self];

        while let Some(node) = nodes.pop() {
            match node {
                Self::Node(node) => {
                    Tag::Node.encode(encoder)?;

                    // We add the children in reverse order so that when we pop them from the
                    // `nodes` stack, they are in the original order.
                    nodes.extend(node.children.iter().rev());
                }

                Self::Leaf(MerkleProofLeaf::Read(data)) => {
                    Tag::Leaf(LeafTag::Read).encode(encoder)?;

                    // We want to write the raw data, and avoid the bincode length prefix. The decoder
                    // will know how many bytes to read.
                    encoder.writer().write(data.as_slice())?;
                }

                Self::Leaf(MerkleProofLeaf::Blind(hash)) => {
                    Tag::Leaf(LeafTag::Blind).encode(encoder)?;
                    hash.encode(encoder)?;
                }
            }
        }

        Ok(())
    }
}

/// Type used to describe the leaves of a [`MerkleProof`].
/// For more details see the documentation of [`MerkleProof`].
#[derive(Clone, Debug, PartialEq)]
pub enum MerkleProofLeaf {
    /// A leaf that is not read. It may be written.
    /// Contains the hash of the contents from initial state.
    ///
    /// Note: a blinded leaf can correspond to a blinded subtree
    /// in a [`crate::merkle_tree::MerkleTree`] due to compression.
    Blind(Hash),
    /// A leaf that is read. It may also be written.
    /// Contains the read data from the initial state.
    /// The initial hash can be deduced based on the read data.
    Read(Vec<u8>),
}

/// [`enum@HashState`] is associated with the state of hashing a [`MerkleProof`].
/// We record whether the node is a leaf or an internal node, the index of its parent(
/// see [`MerkleProof::root_hash`] for more details) and the hashes of its children
/// if it's a node and its own hash if its a leaf.
enum HashState {
    Node {
        parent_index: usize,
        hashes: Vec<Hash>,
    },
    Leaf {
        parent_index: usize,
        hash: Hash,
    },
}

impl HashState {
    fn new_leaf(parent_index: usize, hash: Hash) -> Self {
        HashState::Leaf { parent_index, hash }
    }

    fn new_node(parent_index: usize) -> Self {
        HashState::Node {
            parent_index,
            hashes: vec![],
        }
    }

    /// Push a hash to node's hash list.
    ///
    /// # Panics
    ///
    /// Panics if the hash state is a Leaf.
    fn push(&mut self, hash: Hash) {
        match self {
            HashState::Node { hashes, .. } => hashes.push(hash),
            _ => unreachable!("A leaf node must not have children"),
        }
    }

    fn hash(&self) -> Hash {
        match self {
            HashState::Node { hashes, .. } => Hash::combine_hashes(hashes),
            HashState::Leaf { hash, .. } => *hash,
        }
    }

    fn parent_index(&self) -> usize {
        match self {
            HashState::Node { parent_index, .. } | HashState::Leaf { parent_index, .. } => {
                *parent_index
            }
        }
    }
}

impl MerkleProof {
    /// Create a new Merkle proof as a read leaf.
    pub fn leaf_read(data: Vec<u8>) -> Self {
        MerkleProof::Leaf(MerkleProofLeaf::Read(data))
    }

    /// Create a new Merkle proof as a blind leaf.
    pub fn leaf_blind(hash: Hash) -> Self {
        MerkleProof::Leaf(MerkleProofLeaf::Blind(hash))
    }

    /// Compute the root hash of the Merkle proof.
    pub fn root_hash(&self) -> Hash {
        // Child nodes are stored in normal order in `nodes`.
        let mut nodes: Vec<(&MerkleProof, usize)> = vec![(self, 0)];
        // Child nodes are stored in reverse order in `hash_states`.
        let mut hash_states: Vec<HashState> = vec![];

        while let Some((node, parent_index)) = nodes.pop() {
            match node {
                Tree::Leaf(MerkleProofLeaf::Blind(hash)) => {
                    hash_states.push(HashState::new_leaf(parent_index, *hash));
                }
                Tree::Leaf(MerkleProofLeaf::Read(data)) => {
                    hash_states.push(HashState::new_leaf(
                        parent_index,
                        Hash::hash_bytes(data.as_slice()),
                    ));
                }
                Tree::Node(node) => {
                    hash_states.push(HashState::new_node(parent_index));
                    let new_parent_index = hash_states.len() - 1;
                    for child in node.children.iter() {
                        nodes.push((child, new_parent_index));
                    }
                }
            }
        }

        while hash_states.len() > 1 {
            let hash_state = hash_states
                .pop()
                .expect("hash_states can't be empty at this point");
            // Note that child hashes are added in normal order to `hash_states`.
            hash_states[hash_state.parent_index()].push(hash_state.hash());
        }

        // Hash states is not empty at this point.
        hash_states[0].hash()
    }
}

impl From<&MerkleProof> for Tag {
    fn from(value: &MerkleProof) -> Self {
        match value {
            MerkleProof::Node(_) => Tag::Node,
            MerkleProof::Leaf(MerkleProofLeaf::Blind(_)) => Tag::Leaf(LeafTag::Blind),
            MerkleProof::Leaf(MerkleProofLeaf::Read(_)) => Tag::Leaf(LeafTag::Read),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialisation::bincode_default_config;

    #[test]
    fn merkle_proofs_can_be_encoded() {
        let merkle_proofs = [
            MerkleProof::leaf_read([1, 2, 3].to_vec()),
            MerkleProof::leaf_blind(Hash::hash_bytes(&[1, 3, 4])),
            Tree::node_without_data(
                [
                    MerkleProof::leaf_read([1, 2, 3].to_vec()),
                    MerkleProof::leaf_blind(Hash::hash_bytes(&[1, 3, 4])),
                ]
                .to_vec(),
            ),
        ];
        for merkle_proof in merkle_proofs.iter() {
            bincode::encode_to_vec(merkle_proof, bincode_default_config())
                .expect("Failed to encode the merkle proof");
        }
    }

    #[test]
    fn we_can_take_the_merkle_proof_of_the_root_hash() {
        let node = Tree::node_without_data(
            [
                MerkleProof::leaf_read([1, 2, 3].to_vec()),
                MerkleProof::leaf_blind(Hash::hash_bytes(&[1, 3, 4])),
            ]
            .to_vec(),
        );
        let _ = node.root_hash();
    }

    #[test]
    fn child_node_hashes_are_pushed_back_in_normal_order() {
        let merkle_proof = Tree::node_without_data(
            [
                Tree::node_without_data(
                    [
                        MerkleProof::leaf_read([1, 2, 3].to_vec()),
                        MerkleProof::leaf_blind(Hash::hash_bytes(&[4, 5, 6])),
                    ]
                    .to_vec(),
                ),
                MerkleProof::leaf_blind(Hash::hash_bytes(&[7, 8, 9])),
            ]
            .to_vec(),
        );

        let calculated_root_hash = merkle_proof.root_hash();

        let mut first_child_node = HashState::new_node(0);
        first_child_node.push(Hash::hash_bytes(&[1, 2, 3]));
        first_child_node.push(Hash::hash_bytes(&[4, 5, 6]));

        let mut root_node = HashState::new_node(0);
        root_node.push(first_child_node.hash());
        root_node.push(Hash::hash_bytes(&[7, 8, 9]));

        assert_eq!(root_node.hash(), calculated_root_hash);
    }
}
