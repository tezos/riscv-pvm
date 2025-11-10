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
                Self::Node(trees) => {
                    Tag::Node.encode(encoder)?;

                    // We add the children in reverse order so that when we pop them from the
                    // `nodes` stack, they are in the original order.
                    nodes.extend(trees.iter().rev());
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

enum NodeLeaf {
    Node,
    Leaf,
}

struct HashState {
    node_leaf: NodeLeaf,
    parent_index: usize,
    digests: Vec<[u8; 32]>,
}

impl HashState {
    pub fn new(node_leaf: NodeLeaf, parent_index: usize) -> Self {
        Self {
            node_leaf,
            parent_index,
            digests: vec![],
        }
    }

    pub fn new_with_digest(node_leaf: NodeLeaf, parent_index: usize, digest: [u8; 32]) -> Self {
        Self {
            node_leaf,
            parent_index,
            digests: vec![digest],
        }
    }

    pub fn push(&mut self, digest: [u8; 32]) {
        self.digests.push(digest);
    }

    pub fn get_digest(&self) -> [u8; 32] {
        match self.node_leaf {
            NodeLeaf::Node => {
                let mut hasher = blake3::Hasher::new();

                for digest in self.digests.iter() {
                    hasher.update(digest);
                }

                hasher.finalize().into()
            }
            NodeLeaf::Leaf => self.digests[0],
        }
    }

    pub fn get_parent_index(&self) -> usize {
        self.parent_index
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
        let mut nodes: Vec<(&MerkleProof, usize)> = vec![(self, 0)];
        let mut hashes: Vec<HashState> = vec![];

        while let Some((node, parent_index)) = nodes.pop() {
            match node {
                Tree::Leaf(MerkleProofLeaf::Blind(hash)) => {
                    hashes.push(HashState::new_with_digest(
                        NodeLeaf::Leaf,
                        parent_index,
                        hash.digest,
                    ));
                }
                Tree::Leaf(MerkleProofLeaf::Read(data)) => {
                    hashes.push(HashState::new_with_digest(
                        NodeLeaf::Leaf,
                        parent_index,
                        Hash::blake3_hash_bytes(data.as_slice()).digest,
                    ));
                }
                Tree::Node(children) => {
                    hashes.push(HashState::new(NodeLeaf::Node, parent_index));
                    let new_parent_index = hashes.len() - 1;
                    for child in children.iter() {
                        nodes.push((&child, new_parent_index));
                    }
                }
            }
        }

        while hashes.len() > 1 {
            let hash_state = hashes.pop().unwrap();
            hashes[hash_state.get_parent_index()].push(hash_state.get_digest());
        }

        Hash::new(hashes[0].get_digest())
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

    fn get_bincode_config() -> bincode::config::Configuration {
        bincode::config::standard()
    }

    #[test]
    fn merkle_proofs_can_be_encoded() {
        let merkle_proofs = [
            MerkleProof::leaf_read([1, 2, 3].to_vec()),
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[1, 3, 4])),
            Tree::Node(
                [
                    MerkleProof::leaf_read([1, 2, 3].to_vec()),
                    MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[1, 3, 4])),
                ]
                .to_vec(),
            ),
        ];
        for merkle_proof in merkle_proofs.iter() {
            bincode::encode_to_vec(merkle_proof, get_bincode_config())
                .expect("Failed to encode the merkle proof");
        }
    }

    #[test]
    fn we_can_take_the_merkle_proof_of_the_root_hash() {
        let node = Tree::Node(
            [
                MerkleProof::leaf_read([1, 2, 3].to_vec()),
                MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[1, 3, 4])),
            ]
            .to_vec(),
        );
        let _ = node.root_hash();
    }
}
