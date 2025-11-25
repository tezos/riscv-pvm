// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Hashing

use std::borrow::Borrow;
use std::collections::VecDeque;

use bincode::Decode;
use bincode::Encode;
use bincode::error::EncodeError;
use thiserror::Error;

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::serialisation as binary;
use crate::tree::Tree;

#[derive(Error, Debug)]
pub enum HashError {
    #[error("Encoding error: {0}")]
    Encode(#[from] EncodeError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("The input buffer was expected to be non-empty")]
    NonEmptyBufferExpected,
}

/// Size of digest produced by the underlying hash function
pub const DIGEST_SIZE: usize = 32;

/// A value of type [struct@Hash] indicates that the enclosed array is a digest
/// produced by a preset hash function, currently BLAKE2b. It can be obtained
/// by either hashing data directly or after hashing by converting from
/// a suitably sized byte slice or vector.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Encode,
    Decode,
    Hash,
    PartialOrd,
    Ord,
    derive_more::From,
    derive_more::Debug,
)]
#[debug("{}", self)]
pub struct Hash {
    digest: [u8; DIGEST_SIZE],
}

impl Hash {
    /// Hash a slice of bytes
    pub fn blake3_hash_bytes(bytes: &[u8]) -> Self {
        let digest = blake3::hash(bytes).into();
        Hash { digest }
    }

    /// Get the hash of a value that can be serialised by hashing its serialisation
    pub fn blake3_hash<T: Encode>(data: T) -> Result<Self, EncodeError> {
        let mut hasher = blake3::Hasher::new();
        binary::serialise_into(&data, &mut hasher)?;

        let digest = hasher.finalize().into();
        Ok(Hash { digest })
    }

    /// Combine multiple [`struct@Hash`] values into a single one.
    ///
    /// The hashes are combined by concatenating them, then hashing the result.
    /// Pre-image resistance is not compromised because the concatenation is not
    /// ambiguous, with hashes having a fixed size ([`DIGEST_SIZE`]).
    pub fn combine<H: Borrow<Hash>, HS: IntoIterator<Item = H>>(hashes: HS) -> Hash {
        let mut hasher = blake3::Hasher::new();

        for hash in hashes {
            let hash: &Hash = hash.borrow();
            hasher.update(hash.as_ref());
        }

        let digest = hasher.finalize().into();
        Hash { digest }
    }

    /// Like [`Self::combine`], but the iterator can yield errors.
    pub fn try_combine<H: Borrow<Hash>, E, HS: IntoIterator<Item = Result<H, E>>>(
        hashes: HS,
    ) -> Result<Hash, E> {
        let mut hasher = blake3::Hasher::new();

        for hash in hashes {
            let hash = hash?;
            hasher.update(hash.borrow().as_ref());
        }

        let digest = hasher.finalize().into();
        Ok(Hash { digest })
    }

    /// Hash the underlying state of a foldable structure.
    pub fn from_foldable(foldable: &impl Foldable<HashFold>) -> Self {
        foldable.fold(HashFold)
    }
}

impl std::fmt::Display for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(self.digest).fmt(f)
    }
}

impl From<Hash> for [u8; DIGEST_SIZE] {
    fn from(value: Hash) -> Self {
        value.digest
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.digest
    }
}

impl Foldable<HashFold> for Hash {
    fn fold(&self, _builder: HashFold) -> Hash {
        *self
    }
}

pub struct HashFold;

impl Fold for HashFold {
    type Folded = Hash;

    type NodeFold = HashNodeFold;

    fn into_node_fold(self) -> Self::NodeFold {
        HashNodeFold::default()
    }
}

#[derive(Default)]
pub struct HashNodeFold {
    hasher: blake3::Hasher,
}

impl NodeFold for HashNodeFold {
    type Parent = HashFold;

    fn add<F: Foldable<HashFold>>(&mut self, child: &F) {
        let folded_child = child.fold(HashFold);
        self.hasher.update(folded_child.as_ref());
    }

    fn done(self) -> Hash {
        let digest = self.hasher.finalize().into();
        Hash { digest }
    }
}

/// Result of hashing a potentially partial state
///
/// This type may not contain the hash itself. However, it indicates whether the hash can be
/// recovered from the compressed partial Merkle tree used to instantiate the state.
///
/// It may also indicate whether such recovery is not possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartialHash {
    /// State hash could not be produced due to an invalid proof
    ///
    /// This can happen when mixing [`PartialHash::Previous`] and [`PartialHash::Present`] in the
    /// same node.
    ///
    /// Effectively, this indicates that the compressed partial Merkle tree which was used to
    /// instantiate the state did not contain the right information to compute the state hash after
    /// modifications to the state.
    InvalidProof,

    /// State is unchanged compared to the previous state
    ///
    /// Indicates that the state is the same as in the previous state. The component that produced
    /// this variant may not have access to the previous state hash. This is a way to defer to the
    /// parent component to provide the previous hash.
    Previous,

    /// Current state hash is available
    ///
    /// Note, this does not indicate that the state has changed.
    Present(Hash),
}

impl PartialHash {
    /// Get the hash from a [`PartialHash::Present`] variant, if possible. Otherwise, panic.
    pub fn to_hash(self) -> Option<Hash> {
        match self {
            PartialHash::Present(hash) => Some(hash),
            PartialHash::InvalidProof | PartialHash::Previous => None,
        }
    }

    /// Compute a [`PartialHash`] from a foldable structure.
    pub fn from_foldable<'tree>(
        proof: Option<&'tree MerkleProof>,
        foldable: &impl Foldable<PartialHashFold<'tree>>,
    ) -> PartialHash {
        foldable.fold(PartialHashFold { proof })
    }
}

impl Foldable<PartialHashFold<'_>> for PartialHash {
    fn fold(&self, builder: PartialHashFold) -> Self {
        match self {
            PartialHash::InvalidProof => PartialHash::InvalidProof,
            PartialHash::Previous => builder.previous(),
            PartialHash::Present(hash) => builder.present(*hash),
        }
    }
}

/// [`Fold`] implementation for computing the [`PartialHash`] of a state
pub struct PartialHashFold<'tree> {
    /// Original proof which is the source of previous hashes
    proof: Option<&'tree MerkleProof>,
}

impl<'tree> PartialHashFold<'tree> {
    /// Mark the state as present with the given hash.
    pub fn present(self, hash: Hash) -> PartialHash {
        PartialHash::Present(hash)
    }

    /// Mark the state as unchanged, thereby deferring to the previous hash if available.
    pub fn previous(self) -> PartialHash {
        match self.proof {
            None => PartialHash::Previous,
            Some(tree) => {
                let hash = tree.root_hash();
                PartialHash::Present(hash)
            }
        }
    }
}

impl<'tree> Fold for PartialHashFold<'tree> {
    type Folded = PartialHash;

    type NodeFold = PartialHashNodeFold<'tree>;

    fn into_node_fold(self) -> Self::NodeFold {
        let Some(tree) = self.proof else {
            return PartialHashNodeFold {
                node_hash: None,
                children: VecDeque::new(),
                child_hashes: VecDeque::new(),
            };
        };

        match tree {
            Tree::Node(children) => PartialHashNodeFold {
                node_hash: None,
                children: VecDeque::from_iter(children),
                child_hashes: VecDeque::new(),
            },

            Tree::Leaf(MerkleProofLeaf::Read(_)) => PartialHashNodeFold {
                node_hash: None,
                children: VecDeque::new(),
                child_hashes: VecDeque::new(),
            },

            Tree::Leaf(MerkleProofLeaf::Blind(hash)) => PartialHashNodeFold {
                node_hash: Some(hash),
                children: VecDeque::new(),
                child_hashes: VecDeque::new(),
            },
        }
    }
}

/// [`NodeFold`] implementation for computing the [`PartialHash`] of a state
pub struct PartialHashNodeFold<'tree> {
    /// Previous hash for this node, if available
    node_hash: Option<&'tree Hash>,

    /// Proof for each remaining child of the node
    children: VecDeque<&'tree MerkleProof>,

    /// Hash of each child seen so far
    child_hashes: VecDeque<PartialHash>,
}

impl<'tree> NodeFold for PartialHashNodeFold<'tree> {
    type Parent = PartialHashFold<'tree>;

    fn add<F: Foldable<Self::Parent>>(&mut self, child: &F) {
        let hash = match self.children.pop_front() {
            Some(tree) => {
                let prev_hash = tree.root_hash();
                let hash = child.fold(PartialHashFold { proof: Some(tree) });

                // If the child is absent but we have the previous hash, we can use it here.
                match hash {
                    PartialHash::Previous => PartialHash::Present(prev_hash),
                    other => other,
                }
            }

            None => child.fold(PartialHashFold { proof: None }),
        };

        self.child_hashes.push_back(hash);
    }

    fn done(self) -> PartialHash {
        let mut saw_absent_child = false;
        let mut hasher = blake3::Hasher::new();

        for child_hash in self.child_hashes {
            match child_hash {
                PartialHash::InvalidProof => {
                    // Any invalid child makes the whole node invalid.
                    return PartialHash::InvalidProof;
                }

                PartialHash::Previous => {
                    if hasher.count() > 0 {
                        // There was at least one present child before. Mixing absent and present
                        // makes it invalid.
                        return PartialHash::InvalidProof;
                    }

                    // Ensure that encountering any further present child will make the whole node
                    // invalid, as that would mix absent and present children.
                    saw_absent_child = true;
                }

                PartialHash::Present(hash) => {
                    // Are we mixing absent and present children?
                    if saw_absent_child {
                        return PartialHash::InvalidProof;
                    }

                    hasher.update(hash.as_ref());
                }
            }
        }

        if saw_absent_child {
            return self
                .node_hash
                .cloned()
                .map(PartialHash::Present)
                .unwrap_or(PartialHash::Previous);
        }

        let digest: [u8; 32] = hasher.finalize().into();
        let hash = Hash::from(digest);
        PartialHash::Present(hash)
    }
}
