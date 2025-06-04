// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Module for defining traits which facilitate desserialising a tree structure.
//! More precisely, our usecase if for deserialising a Merkle tree either from a [`ProofTree`] or
//! from the raw bytes of a serialisation.
//!
//! Due to the nature of the deserialisation, the structure of the tree is not known until part of the
//! deserialisation is already parsed and the shape is known. This introduces the need for the
//! [`Suspended`] trait which abstracts over a computation to be obtained after parsing just enough from
//! the serialisation to deduce the shape of the tree.
//!
//! [`ProofTree`]: crate::state_backend::ProofTree

use serde::de::DeserializeOwned;

use crate::state_backend::FromProofError;
use crate::state_backend::OwnedProofPart;
use crate::state_backend::hash::Hash;
use crate::state_backend::proof_backend::merkle::MERKLE_LEAF_SIZE;

/// Error used when deserialising using [`Deserialiser`] methods
pub type DeserError = FromProofError;

/// Result type used when deserialising using [`Deserialiser`] methods.
pub type Result<R, E = DeserError> = std::result::Result<R, E>;

/// Possible outcomes when parsing a node or a leaf from a Merkle proof
/// where the leaf is assumed to have type `T`.
#[derive(Clone)]
pub enum Partial<T> {
    /// The leaf / node is altogether absent from the proof.
    Absent,
    /// A blinded subtree and its [`struct@Hash`] is provided.
    Blinded(Hash),
    /// Data successfully parsed and its type is `T`.
    Present(T),
}

impl<T> Partial<T> {
    /// Map the present result of a [`Partial<T>`] into [`Partial<R>`].
    pub fn map_present<R>(self, f: impl FnOnce(T) -> R) -> Partial<R> {
        match self {
            Partial::Absent => Partial::Absent,
            Partial::Blinded(hash) => Partial::Blinded(hash),
            Partial::Present(data) => Partial::Present(f(data)),
        }
    }

    /// Same as [`Partial::map_present`] but can fail.
    pub fn map_present_fallible<R, E>(
        self,
        f: impl FnOnce(T) -> Result<R, E>,
    ) -> Result<Partial<R>, E> {
        match self {
            Partial::Absent => Ok(Partial::Absent),
            Partial::Blinded(hash) => Ok(Partial::Blinded(hash)),
            Partial::Present(data) => Ok(Partial::Present(f(data)?)),
        }
    }
}

impl Partial<Vec<u8>> {
    /// Convert a [`Partial<Vec<u8>>`] into an owned proof part.
    pub fn into_leaf_proof_tree(self) -> OwnedProofPart {
        OwnedProofPart::leaf_from_partial(self, |data| data)
    }
}

impl Partial<Box<[u8; MERKLE_LEAF_SIZE.get()]>> {
    /// Convert a [`Partial<Box<[u8; MERKLE_LEAF_SIZE]>>`] into an owned proof part.
    pub fn into_leaf_proof_tree(self) -> OwnedProofPart {
        OwnedProofPart::leaf_from_partial(self, |data| data.to_vec())
    }
}

impl<A, B> Partial<(A, B)> {
    /// Split a [`Partial<(A, B)>`] into [`Partial<A>`] and [`Partial<B>`].
    pub fn split(self) -> (Partial<A>, Partial<B>) {
        match self {
            Partial::Absent => (Partial::Absent, Partial::Absent),
            Partial::Blinded(hash) => (Partial::Blinded(hash), Partial::Blinded(hash)),
            Partial::Present((a, b)) => (Partial::Present(a), Partial::Present(b)),
        }
    }
}

/// The main trait used for deserialising a proof.
///
/// Having an object of this trait is equivalent to having a proof and being able to deserialise it.
///
/// A proof can be interpreted in 3 cases:
/// 1. [`Deserialiser::into_leaf_raw`] The proof is a leaf and raw bytes are obtained.
/// 2. [`Deserialiser::into_leaf<T>`] The proof is a leaf and the type `T` is parsed.
/// 3. [`Deserialiser::into_node`] The proof is a node in the tree.
pub trait Deserialiser {
    /// After deserialising a proof, a [`Suspended<R>`] computation is obtained.
    type Suspended<R>: Suspended<Output = R, Parent = Self>;

    /// In case the proof is a node, [`Deserialiser::DeserialiserNode`] is the deserialiser for the branch case.
    type DeserialiserNode<R>: DeserialiserNode<R, Parent = Self>;

    /// It is expected for the proof to be a leaf. Obtain the raw bytes from that leaf.
    fn into_leaf_raw<const LEN: usize>(self) -> Result<Self::Suspended<Partial<Box<[u8; LEN]>>>>;

    /// It is expected for the proof to be a leaf. Parse the raw bytes of that leaf into a type `T`.
    #[expect(
        clippy::type_complexity,
        reason = "Adding an alias for Partial<(T, Vec<u8>)> would only decrease readability"
    )]
    fn into_leaf<T: DeserializeOwned + 'static>(
        self,
    ) -> Result<Self::Suspended<Partial<(T, Vec<u8>)>>>;

    /// It is expected for the proof to be a node. Obtain the deserialiser for the branch case.
    fn into_node(self) -> Result<Self::DeserialiserNode<Partial<()>>>;
}

/// The trait used for deserialising a proof's node.
/// Having an object of this trait is equivalent to knowing the current proof is a node.
/// Deserialisers for each of its branches are expected to be provided to continue the deserialisation.
pub trait DeserialiserNode<R> {
    type Parent: Deserialiser;

    /// The next branch of the current node is deserialised using the provided deserialiser `br_deser`.
    fn next_branch<T>(
        self,
        branch_deserialiser: impl FnOnce(
            Self::Parent,
        )
            -> Result<<Self::Parent as Deserialiser>::Suspended<T>>,
    ) -> Result<<Self::Parent as Deserialiser>::DeserialiserNode<(R, T)>>
    where
        T: 'static,
        R: 'static;

    /// Helper for mapping the current result into a new type.
    fn map<T>(
        self,
        f: impl FnOnce(R) -> T + 'static,
    ) -> <Self::Parent as Deserialiser>::DeserialiserNode<T>
    where
        T: 'static,
        R: 'static;

    /// Signal the end of deserialisation of the node's branches.
    /// Call this method after all calls to [`DeserialiserNode::next_branch`] have been made.
    fn done(self) -> Result<<Self::Parent as Deserialiser>::Suspended<R>>;
}

/// The trait represents a computation function obtained after deserialising a proof.
pub trait Suspended {
    /// End result of the computation.
    type Output;

    type Parent: Deserialiser;

    /// Helper to map the current result into a new type.
    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T + 'static,
    ) -> <Self::Parent as Deserialiser>::Suspended<T>
    where
        Self::Output: 'static;
}

#[cfg(test)]
mod tests {

    use std::cell::RefCell;
    use std::rc::Rc;

    use super::Deserialiser;
    use super::DeserialiserNode;
    use super::Partial;
    use super::Result;
    use super::Suspended;
    use crate::state_backend::ProofTree;
    use crate::state_backend::proof_backend::proof::DeserialiseError;
    use crate::state_backend::proof_backend::proof::MerkleProof;
    use crate::state_backend::proof_backend::proof::MerkleProofLeaf;
    use crate::state_backend::proof_backend::proof::TAG_BLIND;
    use crate::state_backend::proof_backend::proof::TAG_NODE;
    use crate::state_backend::proof_backend::proof::TAG_READ;
    use crate::state_backend::proof_backend::proof::Tag;
    use crate::state_backend::proof_backend::proof::deserialise_owned::ProofTreeDeserialiser;
    use crate::state_backend::proof_backend::proof::deserialise_stream::StreamDeserialiser;
    use crate::state_backend::proof_backend::proof::deserialise_stream::StreamParserComb;
    use crate::state_backend::proof_backend::proof::deserialise_stream::TagIter;
    use crate::state_backend::proof_backend::proof::deserialiser::DeserError;
    use crate::state_backend::proof_backend::proof::serialise_raw_tags;
    use crate::state_backend::proof_backend::proof::tag_offset;
    use crate::storage::DIGEST_SIZE;
    use crate::storage::Hash;

    fn computation<D: Deserialiser>(proof: D) -> Result<<D as Deserialiser>::Suspended<i32>> {
        // The tree structure:
        // Node (root)
        // ├── Leaf (type: Hash)
        // └── Node
        //     └── Leaf (type: i32)

        // Computation: return the value of the nested leaf

        let ctx = proof.into_node()?;
        let r = ctx
            .next_branch(|br_proof| br_proof.into_leaf::<Hash>())?
            .map(|(_node_parse, br)| br)
            .next_branch(|br_proof| {
                br_proof
                    .into_node()?
                    .next_branch(|pr| pr.into_leaf::<i32>())?
                    .map(|(_node_parse, br)| br)
                    .done()
            })?
            .done()?;

        Ok(r.map(|(_left, right)| match right {
            Partial::Absent => 0,
            // This blinded hash can be of the nested leaf or the root
            Partial::Blinded(_hash) => -1,
            Partial::Present((nr, _)) => nr,
        }))
    }

    fn computation_2<D: Deserialiser>(proof: D) -> Result<<D as Deserialiser>::Suspended<i32>> {
        // The tree structure
        // Node (root)
        // ├── Leaf 1 (type: i32)
        // ├── Leaf 2 (type: i32)
        // ├── Leaf 3 (type: i32)
        // └── Leaf 4 (type: i32)

        // Computation: sum the non-blinded leaves

        let mut ctx = proof
            .into_node()?
            .map(|data| data.map_present(|_| Vec::<i32>::new()));

        for _ in 0..4 {
            ctx = ctx
                .next_branch(|br_proof| br_proof.into_leaf::<i32>())?
                .map(|(acc, val)| {
                    acc.map_present(|mut acc| {
                        if let Partial::Present((val, _)) = val {
                            acc.push(val);
                        }
                        acc
                    })
                })
        }

        Ok(ctx.done()?.map(|data| match data {
            Partial::Absent => 0,
            Partial::Blinded(_hash) => -1,
            Partial::Present(data) => data.into_iter().sum(),
        }))
    }

    /// Nested results are used to distinguish between deserialisation and parsing leaves stages
    fn run_stream_deserialiser<'t>(
        deser: impl FnOnce(StreamDeserialiser<'t>) -> Result<StreamParserComb<'t, i32>>,
        bytes: &'t [u8],
    ) -> Result<Result<i32>> {
        let tags = Rc::new(RefCell::new(TagIter::new(bytes)));
        let comp_fn = deser(StreamDeserialiser::new_present(tags.clone()));
        comp_fn.map(|f| f.into_result(&mut tags.borrow().remaining_to_stream_input()))
    }

    #[test]
    fn test_absent_computation() {
        // Root is absent already
        let proof: ProofTreeDeserialiser = ProofTree::Absent.into();
        let comp_fn = computation(proof).unwrap();
        assert_eq!(comp_fn.into_result(), 0);

        // We expect to get the Absent case since the father of the nested node is blinded
        let merkle_proof = MerkleProof::Node(vec![
            MerkleProof::Leaf(MerkleProofLeaf::Read(
                // Note, this is a Read leaf, not a blinded one
                Hash::blake2b_hash_bytes(&[0, 1, 2])
                    .unwrap()
                    .as_ref()
                    .to_vec(),
            )),
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[3, 4, 5]).unwrap(),
            )),
        ]);
        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkle_proof).into();
        let comp_fn = computation(proof).unwrap();
        assert_eq!(comp_fn.into_result(), 0);
    }

    #[test]
    fn test_absent_computation_stream() {
        // Root is absent already
        let proof: StreamDeserialiser = StreamDeserialiser::Absent;
        let comp_fn = computation(proof).unwrap();
        assert_eq!(
            comp_fn
                .into_result(&mut TagIter::new(&[]).remaining_to_stream_input())
                .unwrap(),
            0
        );

        // Expect absent case in the computed result
        let tag_bytes = [TAG_NODE << 6 | TAG_READ << 4 | TAG_BLIND << 2];
        let leaf_read: [u8; DIGEST_SIZE] = [12; 32];
        let leaf_blind: [u8; DIGEST_SIZE] = Hash::blake2b_hash_bytes(&[3, 4, 5]).unwrap().into();
        let proof_bytes = [tag_bytes.as_ref(), leaf_read.as_ref(), leaf_blind.as_ref()].concat();
        let res = run_stream_deserialiser(computation, &proof_bytes);
        assert_eq!(res.unwrap().unwrap(), 0);
    }

    #[test]
    fn test_blind_computation() {
        // The nested leaf is blinded
        let absent_shape = MerkleProof::Node(vec![
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap(),
            )),
            MerkleProof::Node(vec![MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap(),
            ))]),
        ]);
        let comp_fn =
            computation::<ProofTreeDeserialiser>(ProofTree::Present(&absent_shape).into());

        let res = comp_fn.unwrap().into_result();

        assert_eq!(res, -1);

        // For computation_2, the provided merkle proof will resolve as blinded
        // since root is blinded
        let merkle_proof = MerkleProof::Leaf(MerkleProofLeaf::Blind(
            Hash::blake2b_hash_bytes(&[6, 7, 8]).unwrap(),
        ));
        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkle_proof).into();
        let comp_fn = computation_2(proof).unwrap();
        assert_eq!(comp_fn.into_result(), -1);
    }

    fn raw_tags_to_bytes<const LEN: usize>(tags: [u8; LEN]) -> Vec<u8> {
        serialise_raw_tags(tags.into_iter().map(|tag| Tag::try_from(tag).unwrap()))
    }

    #[test]
    fn test_blind_computation_stream() {
        // The nested leaf is blinded
        let raw_bytes_tags = raw_tags_to_bytes([TAG_NODE, TAG_BLIND, TAG_NODE, TAG_BLIND]);
        let b1: [u8; DIGEST_SIZE] = Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap().into();
        let b2: [u8; DIGEST_SIZE] = Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap().into();
        let raw_bytes_content = [raw_bytes_tags.as_ref(), b1.as_ref(), b2.as_ref()].concat();

        let rc = Rc::new(RefCell::new(TagIter::new(&raw_bytes_content)));

        let comp_fn =
            computation::<StreamDeserialiser>(StreamDeserialiser::new_present(rc.clone()));

        let res = comp_fn
            .unwrap()
            .into_result(&mut rc.borrow_mut().remaining_to_stream_input())
            .unwrap();

        assert_eq!(res, -1);

        // For computation_2, the provided merkle proof will resolve as blinded
        // since root is blinded
        let merkle_proof = MerkleProof::Leaf(MerkleProofLeaf::Blind(
            Hash::blake2b_hash_bytes(&[6, 7, 8]).unwrap(),
        ));
        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkle_proof).into();
        let comp_fn = computation_2(proof).unwrap();
        assert_eq!(comp_fn.into_result(), -1);
    }

    #[test]
    fn test_bad_structure() {
        let bad_shape_1 = MerkleProof::Node(vec![]);
        let bad_shape_2 = MerkleProof::Node(vec![
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap(),
            )),
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap(),
            )),
            MerkleProof::Node(vec![]),
            MerkleProof::Node(vec![]),
            MerkleProof::Node(vec![]),
        ]);
        let bad_shape_3 = MerkleProof::Node(vec![
            MerkleProof::Node(vec![]),
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap(),
            )),
        ]);
        let bad_shape_4 = MerkleProof::Node(vec![
            MerkleProof::Leaf(MerkleProofLeaf::Read([42_u8; 32].to_vec())),
            MerkleProof::Leaf(MerkleProofLeaf::Read(100_i32.to_le_bytes().to_vec())),
        ]);

        // Tree is missing branches
        let comp_fn = computation::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_1).into());
        assert!(comp_fn.is_err_and(|e| matches!(e, DeserError::BadNumberOfBranches { .. })));

        // First 2 children of root are ok in shape (blinded) but the total number of children does not correspond
        // Ideally, we would like to have expected: 2, got: 5, but the implemenetation for `ProofTreeDeserialiser`
        // does not track this information (the original number of chilren)
        let comp_fn = computation::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_2).into());
        assert!(comp_fn.is_err_and(|e| {
            println!("{e:?}");
            matches!(e, DeserError::BadNumberOfBranches {
                expected: 0,
                got: 3
            })
        }));

        // The first child is a node, but is expected to be a leaf
        let comp_fn = computation::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_3).into());
        assert!(comp_fn.is_err_and(|e| matches!(e, DeserError::UnexpectedNode)));

        // The second child is a leaf, but is expected to be a node
        let comp_fn = computation::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_4).into());
        assert!(comp_fn.is_err_and(|e| { matches!(e, DeserError::UnexpectedLeaf) }));
    }

    #[test]
    fn test_bad_structure_stream() {
        let hash: [u8; DIGEST_SIZE] = Hash::blake2b_hash_bytes(&[0, 1, 2]).unwrap().into();
        // Place an invalid second tag
        let tag_shape_1 = [TAG_NODE << tag_offset(0) | 0b01 << tag_offset(1)];
        let tag_shape_2 =
            raw_tags_to_bytes([TAG_NODE, TAG_BLIND, TAG_BLIND, TAG_NODE, TAG_NODE, TAG_NODE]);
        let tag_shape_3 = raw_tags_to_bytes([TAG_NODE, TAG_NODE, TAG_BLIND]);
        let tag_shape_4 = raw_tags_to_bytes([TAG_NODE, TAG_READ, TAG_READ]);

        let data_shape_1 = [];
        let data_shape_2 = [hash.as_ref(), hash.as_ref()].concat();
        let data_shape_3 = &hash;
        let data_shape_4 = [hash.as_ref(), hash.as_ref()].concat();

        // Bad tag introduced after the first node
        let res = run_stream_deserialiser(
            computation,
            &[tag_shape_1.as_ref(), data_shape_1.as_ref()].concat(),
        );
        assert!(matches!(
            res,
            Err(DeserError::TagDeserialise(DeserialiseError::InvalidTag))
        ));

        // First 2 children of root are ok in shape (blinded) but the total number of children does not correspond
        let bytes = &[tag_shape_2.as_slice(), data_shape_2.as_ref()].concat();
        let res = run_stream_deserialiser(computation, bytes);
        assert!(matches!(res, Ok(Err(DeserError::RemainingBytes))));

        // The first child is a node, but is expected to be a leaf
        let res = run_stream_deserialiser(
            computation,
            &[tag_shape_3.as_ref(), data_shape_3.as_ref()].concat(),
        );
        assert!(matches!(res, Err(DeserError::UnexpectedNode)));

        // The second child is a read leaf, but is expected to be a node
        let res = run_stream_deserialiser(
            computation,
            &[tag_shape_4.as_slice(), data_shape_4.as_ref()].concat(),
        );
        assert!(matches!(res, Err(DeserError::UnexpectedLeaf)));
    }

    #[test]
    fn test_valid_computation() {
        let merkleproof = MerkleProof::Node(vec![
            MerkleProof::Leaf(MerkleProofLeaf::Read(
                0x140A_0000_i32.to_le_bytes().to_vec(),
            )),
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[3, 4, 5]).unwrap(),
            )),
            MerkleProof::Leaf(MerkleProofLeaf::Read(0xC0005_i32.to_le_bytes().to_vec())),
            MerkleProof::Leaf(MerkleProofLeaf::Blind(
                Hash::blake2b_hash_bytes(&[9, 10, 11]).unwrap(),
            )),
        ]);

        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkleproof).into();
        let comp_fn = computation_2(proof).unwrap();
        assert_eq!(comp_fn.into_result(), 0x140A_0000 + 0xC0005);
    }

    #[test]
    fn test_valid_computation_stream() {
        let h1 = 0x140A_0000_i32.to_le_bytes();
        let h2: [u8; DIGEST_SIZE] = Hash::blake2b_hash_bytes(&[3, 4, 5]).unwrap().into();
        let h3 = 0xC0005_i32.to_le_bytes();
        let h4: [u8; DIGEST_SIZE] = Hash::blake2b_hash_bytes(&[9, 10, 11]).unwrap().into();

        let tags = raw_tags_to_bytes([TAG_NODE, TAG_READ, TAG_BLIND, TAG_READ, TAG_BLIND]);

        let res = run_stream_deserialiser(
            computation_2,
            &[
                tags.as_ref(),
                h1.as_ref(),
                h2.as_ref(),
                h3.as_ref(),
                h4.as_ref(),
            ]
            .concat(),
        );
        assert_eq!(res.unwrap().unwrap(), 0x140A_0000 + 0xC0005);
    }
}
