// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Hashing

use std::borrow::Borrow;
use std::collections::VecDeque;
use std::ops::Deref;

use bincode::Decode;
use bincode::Encode;
use bincode::error::EncodeError;
use thiserror::Error;

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::serialisation::serialise_into;
use crate::tree::Tree;

/// Errors that can occur during hashing operations
#[derive(Error, Debug)]
pub enum HashError {
    #[error("Encoding error: {0}")]
    Encode(#[from] EncodeError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("The input buffer was expected to be non-empty")]
    NonEmptyBufferExpected,
}

/// A value of type [struct@Hash] indicates that the enclosed array is a digest
/// produced by a preset hash function, currently BLAKE3. It can be obtained
/// by either hashing data directly or after hashing by converting from
/// a suitably sized byte slice or vector.
#[derive(Clone, Copy, PartialEq, Eq, Encode, Decode, Hash, PartialOrd, Ord, derive_more::Debug)]
#[debug("{}", self)]
pub struct Hash {
    digest: [u8; Hash::DIGEST_SIZE],
}

impl Hash {
    /// Size of digest produced by the underlying hash function
    pub const DIGEST_SIZE: usize = 32;

    /// Hashes a byte slice into a [`struct@Hash`].
    pub fn hash_bytes(bytes: &[u8]) -> Self {
        let digest = blake3::hash(bytes).into();
        Hash { digest }
    }

    /// Creates a [`struct@Hash`] from something that implements the
    /// [`bincode::enc::Encode`] trait.
    pub fn hash_encodable<T: Encode>(data: T) -> Result<Self, EncodeError> {
        let mut hasher = blake3::Hasher::new();
        serialise_into(&data, &mut hasher)?;

        let digest = hasher.finalize().into();
        Ok(Hash { digest })
    }

    /// Creates a [`struct@Hash`] from a collection of iterables that can be
    /// [`Deref`]ed as a [`struct@Hash`]. Note that this method  is rehashing the
    /// hashes.
    pub fn combine_hashes<H: Borrow<Hash>, HS: IntoIterator<Item = H>>(hashes: HS) -> Hash {
        let mut hasher = blake3::Hasher::new();

        for hash in hashes {
            let hash: &Hash = hash.borrow();
            // Pre-image resistence is guaranteed by the constant hash digest length.
            hasher.update(hash.as_ref());
        }

        let digest = hasher.finalize().into();
        Hash { digest }
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

impl From<[u8; Hash::DIGEST_SIZE]> for Hash {
    fn from(value: [u8; Hash::DIGEST_SIZE]) -> Self {
        Self { digest: value }
    }
}

impl From<Hash> for [u8; Hash::DIGEST_SIZE] {
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

impl Deref for Hash {
    type Target = [u8; Hash::DIGEST_SIZE];

    fn deref(&self) -> &Self::Target {
        &self.digest
    }
}

/// [struct@Hasher] can be dynamically updated with byte arrays and [`struct@Hash`]s and
/// can be turned into a [`struct@Hash`].
#[derive(Default)]
pub struct Hasher {
    hasher: blake3::Hasher,
}

impl Hasher {
    /// Updates the [`Hasher`] with some bytes
    pub fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    /// Updates the [`Hasher`] with the digest of a [`struct@Hash`]
    pub fn update_with_hash(&mut self, hash: &Hash) {
        self.hasher.update(hash.deref());
    }

    /// Turns the [`Hasher`] into a [`struct@Hash`]
    pub fn to_hash(self) -> Hash {
        let digest: [u8; Hash::DIGEST_SIZE] = self.hasher.finalize().into();
        Hash { digest }
    }

    /// Returns the number of bytes hashed so far.
    pub fn count(&self) -> u64 {
        self.hasher.count()
    }
}

/// [`Fold`] implementation producing a [`struct@Hash`]
pub struct HashFold;

impl Fold for HashFold {
    type Folded = Hash;

    type NodeFold = HashNodeFold;

    fn into_node_fold(self) -> Self::NodeFold {
        HashNodeFold::default()
    }
}

/// [`NodeFold`] implementation producing a [`struct@Hash`]
///
/// It collects the hashes of all children and then combines them by hashing the concatenation of
/// children's [`struct@Hash`] bytes.
#[derive(Default)]
pub struct HashNodeFold {
    /// Hasher used to combine children's hashes
    hasher: Hasher,
}

impl NodeFold for HashNodeFold {
    type Parent = HashFold;

    fn add<F: Foldable<HashFold>>(&mut self, child: &F) {
        let folded_child = child.fold(HashFold);
        self.hasher.update(folded_child.as_ref());
    }

    fn done(self) -> Hash {
        self.hasher.to_hash()
    }
}

/// Result of hashing a potentially partial state
///
/// This type might not contain the hash itself. However, it indicates whether the hash can be
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
    /// this variant might not have access to the previous state hash. This is a way to defer to the
    /// parent component to provide the previous hash.
    Previous,

    /// Current state hash is available
    ///
    /// Note, this does not indicate that the state has changed.
    Present(Hash),
}

impl PartialHash {
    /// Get the hash from a [`PartialHash::Present`] variant, if possible. Otherwise, return `None`.
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
            PartialHash::Previous => builder.previous(),
            PartialHash::Present(hash) => builder.present(*hash),
            PartialHash::InvalidProof => PartialHash::InvalidProof,
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
            Tree::Node(node) => PartialHashNodeFold {
                node_hash: None,
                children: VecDeque::from_iter(&node.children),
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
        let mut hasher = Hasher::default();

        for child_hash in self.child_hashes {
            match child_hash {
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

                    hasher.update_with_hash(&hash);
                }

                PartialHash::InvalidProof => {
                    // Any invalid child makes the whole node invalid.
                    return PartialHash::InvalidProof;
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

        PartialHash::Present(hasher.to_hash())
    }
}

#[cfg(test)]
mod tests {
    use bincode::Encode;

    use super::Hash;
    use super::Hasher;
    use crate::serialisation::bincode_default_config;

    #[derive(Clone, Encode)]
    struct Encodable {
        a: u32,
    }

    impl Encodable {
        fn new(a: u32) -> Self {
            Self { a }
        }
    }

    #[test]
    fn test_hash_bytes_works_as_blake3_hashing() {
        let bytes = [1, 2, 3];
        let hash = Hash::hash_bytes(&bytes);
        let hash_digest: [u8; Hash::DIGEST_SIZE] = hash.into();
        let blake3_digest: [u8; 32] = blake3::hash(&bytes).into();
        assert_eq!(hash_digest, blake3_digest);
    }

    #[test]
    fn test_hash_encodable_can_hash_encodables() {
        let encodable = Encodable::new(12);
        let bytes = bincode::encode_to_vec(encodable.clone(), bincode_default_config())
            .expect("Should work");
        let encodable_hash_digest: blake3::Hash = blake3::hash(bytes.as_slice());
        let hash_digest: [u8; Hash::DIGEST_SIZE] =
            Hash::hash_encodable(encodable).expect("Should work").into();
        assert_eq!(encodable_hash_digest, hash_digest);
    }

    #[test]
    fn test_hash_combines_can_combine_hashes_to_a_new_hash() {
        let coll = vec![Hash::hash_bytes(&[1, 2, 3]), Hash::hash_bytes(&[4, 5, 6])];
        let mut hasher = blake3::Hasher::new();
        for elem in coll.iter() {
            hasher.update(elem.as_slice());
        }
        let hasher_digest: [u8; Hash::DIGEST_SIZE] = hasher.finalize().into();
        let hash = Hash::combine_hashes(coll);
        let hash_digest: [u8; Hash::DIGEST_SIZE] = hash.into();
        assert_eq!(hash_digest, hasher_digest);

        let hash_from_combined = Hash::hash_bytes(&[1, 2, 3, 4, 5, 6]);

        assert_ne!(hash, hash_from_combined);
    }

    #[test]
    fn test_hasher_update_with_bytes_is_the_same_as_hash_bytes() {
        let elems: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let mut hasher: Hasher = Hasher::default();

        for elem in elems.iter() {
            hasher.update(elem.as_slice());
        }

        let flattened_elems: Vec<u8> = elems.into_iter().flatten().collect();
        let hash = Hash::hash_bytes(flattened_elems.as_slice());

        assert_eq!(hasher.to_hash(), hash);
    }

    #[test]
    fn test_combine_hashes_is_the_same_as_update_with_hash() {
        let elems: Vec<Hash> = vec![Hash::hash_bytes(&[1, 2, 3]), Hash::hash_bytes(&[4, 5, 6])];

        let hash = Hash::combine_hashes(elems.clone());

        let mut hasher = Hasher::default();

        for elem in elems.into_iter() {
            hasher.update_with_hash(&elem);
        }

        assert_eq!(hash, hasher.to_hash());
    }
}
