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

use bincode::Decode;

use crate::state_backend::OwnedProofPart;
use crate::state_backend::ProofError;
use crate::state_backend::hash::Hash;
use crate::state_backend::proof_backend::merkle::MERKLE_LEAF_SIZE;

/// Result type used when deserialising a proof - including both the layout and contents of the
/// proof.
pub type Result<T, E = ProofError> = std::result::Result<T, E>;

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

    /// Convert a [`Partial<T>`] into an [`Option<T>`], discarding blinded and absent cases.
    pub fn to_present(self) -> Option<T> {
        match self {
            Partial::Present(data) => Some(data),
            Partial::Absent | Partial::Blinded(_) => None,
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
    type DeserialiserNode: DeserialiserNode<Parent = Self>;

    /// It is expected for the proof to be a leaf. Obtain the raw bytes from that leaf.
    fn into_leaf_raw<const LEN: usize>(self) -> Result<Self::Suspended<Partial<Box<[u8; LEN]>>>>;

    /// It is expected for the proof to be a leaf. Parse the raw bytes of that leaf into a type `T`.
    fn into_leaf<T: Decode<()>>(self) -> Result<Self::Suspended<Partial<T>>>;

    /// It is expected for the proof to be a node. Obtain the deserialiser for the branch case.
    fn into_node(self) -> Result<Self::DeserialiserNode>;
}

/// The trait used for deserialising a proof's node.
/// Having an object of this trait is equivalent to knowing the current proof is a node.
/// Deserialisers for each of its branches are expected to be provided to continue the deserialisation.
pub trait DeserialiserNode: Sized {
    type Parent: Deserialiser;

    /// Get the presence information for the node that is being parsed.
    fn presence(&self) -> Partial<()>;

    /// The next branch of the current node is deserialised using the provided deserialiser `br_deser`.
    fn next_branch<T>(
        self,
        branch_deserialiser: impl FnOnce(
            Self::Parent,
        )
            -> Result<<Self::Parent as Deserialiser>::Suspended<T>>,
    ) -> Result<(Self, T)>;

    /// Signal the end of deserialisation of the node's branches.
    /// Call this method after all calls to [`DeserialiserNode::next_branch`] have been made.
    fn done<T>(self, value: T) -> Result<<Self::Parent as Deserialiser>::Suspended<T>>;
}

/// The trait represents a computation function obtained after deserialising a proof.
pub trait Suspended {
    /// End result of the computation.
    type Output;

    type Parent: Deserialiser;

    /// Helper to map the current result into a new type.
    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T,
    ) -> <Self::Parent as Deserialiser>::Suspended<T>;
}

#[cfg(test)]
mod tests {
    use bincode::Decode;

    use super::Deserialiser;
    use super::DeserialiserNode;
    use super::Partial;
    use super::Result;
    use crate::state_backend::ProofError;
    use crate::state_backend::ProofTree;
    use crate::state_backend::proof_backend::proof::InvalidTagError;
    use crate::state_backend::proof_backend::proof::MerkleProof;
    use crate::state_backend::proof_backend::proof::TAG_BLIND;
    use crate::state_backend::proof_backend::proof::TAG_NODE;
    use crate::state_backend::proof_backend::proof::TAG_READ;
    use crate::state_backend::proof_backend::proof::deserialise_owned::OwnedParserComb;
    use crate::state_backend::proof_backend::proof::deserialise_owned::ProofTreeDeserialiser;
    use crate::state_backend::proof_backend::proof::deserialise_stream::StreamDeserialiser;
    use crate::state_backend::proof_backend::proof::deserialise_stream::StreamInput;
    use crate::state_backend::proof_backend::proof::deserialise_stream::StreamParserComb;
    use crate::storage::DIGEST_SIZE;
    use crate::storage::Hash;

    fn generic_computation<T: Into<i32> + Decode<()>, D: Deserialiser>(
        proof: D,
    ) -> Result<<D as Deserialiser>::Suspended<i32>> {
        // The tree structure:
        // Node (root)
        // ├── Leaf (type: Hash)
        // └── Node
        //     └── Leaf (type: T)

        // Computation: return the value of the nested leaf

        let ctx = proof.into_node()?;
        let (ctx, _left) = ctx.next_branch(|br_proof| br_proof.into_leaf::<Hash>())?;
        let (ctx, right) = ctx.next_branch(|br_ctx| {
            let ctx = br_ctx.into_node()?;
            let (ctx, result) = ctx.next_branch(|pr| pr.into_leaf::<T>())?;
            ctx.done(result)
        })?;

        ctx.done(match right {
            Partial::Present(nr) => nr.into(),
            Partial::Absent => 0,
            Partial::Blinded(_hash) => -1,
        })
    }

    fn computation_i16<D: Deserialiser>(proof: D) -> Result<<D as Deserialiser>::Suspended<i32>> {
        generic_computation::<i16, D>(proof)
    }

    fn computation_bool<D: Deserialiser>(proof: D) -> Result<<D as Deserialiser>::Suspended<i32>> {
        generic_computation::<bool, D>(proof)
    }

    fn computation_leaves<D: Deserialiser>(
        proof: D,
    ) -> Result<<D as Deserialiser>::Suspended<i32>> {
        // The tree structure
        // Node (root)
        // ├── Leaf 1 (type: i32)
        // ├── Leaf 2 (type: i32)
        // ├── Leaf 3 (type: i32)
        // └── Leaf 4 (type: i32)

        // Computation: sum the non-blinded leaves

        let ctx = proof.into_node()?;

        match ctx.presence() {
            Partial::Absent => return ctx.done(0),
            Partial::Blinded(_) => return ctx.done(-1),
            Partial::Present(_) => {}
        }

        let mut data = Vec::new();

        let ctx = (0..4).try_fold(ctx, |ctx, _| -> Result<_> {
            let (ctx, val) = ctx.next_branch(|br_proof| br_proof.into_leaf::<i32>())?;

            if let Partial::Present(val) = val {
                data.push(val);
            }

            Ok(ctx)
        })?;

        ctx.done(data.into_iter().sum())
    }

    /// Execute a deserialising computation over an owned Merkle proof.
    fn run_owned_deserialiser<'t>(
        deser: impl FnOnce(ProofTreeDeserialiser<'t>) -> Result<OwnedParserComb<'t, i32>>,
        merkle_proof: &'t MerkleProof,
    ) -> Result<i32> {
        let proof: ProofTreeDeserialiser = ProofTree::Present(merkle_proof).into();
        let parsed_result = deser(proof)?;
        Ok(parsed_result.into_result())
    }

    /// Execute a deserialising computation over raw bytes.
    fn run_stream_deserialiser<'t>(
        deser: impl FnOnce(StreamDeserialiser<'t>) -> Result<StreamParserComb<'t, i32>>,
        bytes: &'t [u8],
    ) -> Result<i32> {
        let input = StreamInput::new(bytes);
        let comp_fn = deser(StreamDeserialiser::new_present(input))?;
        comp_fn.into_result().map(|(ret, _)| ret)
    }

    #[test]
    fn test_absent_computation() {
        // Root is absent already
        let proof: ProofTreeDeserialiser = ProofTree::Absent.into();
        let comp_fn = computation_i16(proof).unwrap();
        assert_eq!(comp_fn.into_result(), 0);

        // We expect to get the Absent case since the father of the nested node is blinded
        let merkle_proof = MerkleProof::Node(vec![
            MerkleProof::leaf_read(Hash::blake3_hash_bytes(&[0, 1, 2]).as_ref().to_vec()),
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[3, 4, 5])),
        ]);
        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkle_proof).into();
        let comp_fn = computation_i16(proof).unwrap();
        assert_eq!(comp_fn.into_result(), 0);
    }

    #[test]
    fn test_absent_computation_stream() {
        // Root is absent already
        let proof: StreamDeserialiser = StreamDeserialiser::new_absent();
        let comp_fn = computation_i16(proof).unwrap();
        assert_eq!(comp_fn.into_result().unwrap().0, 0);

        // Expect absent case in the computed result
        let leaf_read: [u8; DIGEST_SIZE] = [12; 32];
        let leaf_blind: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[3, 4, 5]).into();
        let proof_bytes = [
            [TAG_NODE, TAG_READ].as_ref(),
            leaf_read.as_ref(),
            [TAG_BLIND].as_ref(),
            leaf_blind.as_ref(),
        ]
        .concat();
        let res = run_stream_deserialiser(computation_i16, &proof_bytes);
        assert_eq!(res.unwrap(), 0);
    }

    #[test]
    fn test_not_enough_bytes_error() {
        // For the streaming case if the data is incomplete we will actually get a bincode::Error
        // due to eof being reached. So to test for NotEnoughBytes we are just going to provide less tags
        let hash_read: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[0, 1, 2]).into();
        let bool_read = [1u8];

        // Note the truncated hash
        let raw_bytes_content = [
            [TAG_NODE, TAG_READ].as_ref(),
            hash_read[0..5].as_ref(),
            [TAG_NODE, TAG_READ].as_ref(),
            bool_read.as_ref(),
        ]
        .concat();
        let res = run_stream_deserialiser(computation_bool, &raw_bytes_content).unwrap_err();

        // Corresponds to a bincode::Error & std::io::Error because the hash deserialisation is done by
        // serde/bincode.
        if let ProofError::Deserialise(bincode::error::DecodeError::Io {
            inner: io_err,
            additional: 32,
        }) = res
        {
            assert_eq!(io_err.kind(), std::io::ErrorKind::UnexpectedEof);
        } else {
            panic!("Expected a bincode::Error due to EOF");
        }

        // We don't include the `TAG_READ` that should follow the `TAG_NODE`.
        let raw_bytes_content = [TAG_NODE];
        let res = run_stream_deserialiser(computation_bool, &raw_bytes_content).unwrap_err();

        // In this case, the error happens earlier, at the tag deserialisation, so it is an error
        // thrown by our own `Deserialiser` traits.
        if let ProofError::Deserialise(bincode::error::DecodeError::Io {
            inner: io_err,
            additional: 1,
        }) = res
        {
            assert_eq!(io_err.kind(), std::io::ErrorKind::UnexpectedEof);
        } else {
            panic!("Expected a bincode::Error due to EOF: {res:?}");
        }

        // We omit the contents associated with the `TAG_READ`. This should trigger an error.
        let raw_bytes_content = [TAG_NODE, TAG_READ];
        let res = run_stream_deserialiser(computation_bool, &raw_bytes_content).unwrap_err();

        // In this case, the error happens earlier, at the tag deserialisation, so it is an error
        // thrown by our own `Deserialiser` traits.
        if let ProofError::Deserialise(bincode::error::DecodeError::Io {
            inner: io_err,
            additional: 32,
        }) = res
        {
            assert_eq!(io_err.kind(), std::io::ErrorKind::UnexpectedEof);
        } else {
            panic!("Expected a bincode::Error due to EOF: {res:?}");
        }

        // the same test for the OwnedDeserialiser
        let merkle_proof = MerkleProof::Node(vec![
            MerkleProof::leaf_read(hash_read[0..5].to_vec()),
            MerkleProof::Node(vec![MerkleProof::leaf_read(bool_read.to_vec())]),
        ]);

        let res = run_owned_deserialiser(computation_bool, &merkle_proof);

        // Corresponds to a bincode::Error only because the deserialisation will throw an EOF error.
        eprintln!("Result: {res:?}");
        assert!(
            matches!(
                res,
                Err(ProofError::Deserialise(
                    bincode::error::DecodeError::UnexpectedEnd { additional: 27 }
                ))
            ),
            "{res:?}"
        )
    }

    #[test]
    fn test_bad_bincode() {
        let hash_read: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[0, 1, 2]).into();
        let bad_bool_bincode = [42_u8; 1];

        let raw_bytes_content = [
            [TAG_NODE, TAG_READ].as_ref(),
            hash_read.as_ref(),
            [TAG_NODE, TAG_READ].as_ref(),
            bad_bool_bincode.as_ref(),
        ]
        .concat();

        let res = run_stream_deserialiser(computation_bool, &raw_bytes_content);

        assert!(matches!(res, Err(ProofError::Deserialise(_))));

        let merkle_proof = MerkleProof::Node(vec![
            MerkleProof::leaf_read(hash_read.to_vec()),
            MerkleProof::Node(vec![MerkleProof::leaf_read(bad_bool_bincode.to_vec())]),
        ]);
        let res = run_owned_deserialiser(computation_bool, &merkle_proof);
        eprintln!("Result: {res:?}");
        assert!(matches!(res, Err(ProofError::Deserialise(_))));
    }

    #[test]
    fn test_too_many_bytes_error() {
        let tag_bytes = [TAG_NODE, TAG_READ, TAG_NODE, TAG_READ];
        let hash_read: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[0, 1, 2]).into();
        let bool_read = [1u8];

        // Note the extra byte at the end
        let raw_bytes_content = [
            tag_bytes.as_ref(),
            hash_read.as_ref(),
            bool_read.as_ref(),
            &[42_u8],
        ]
        .concat();

        // This test only makes sense for the stream deserialiser.
        let res = run_stream_deserialiser(computation_bool, &raw_bytes_content);

        matches!(res, Err(ProofError::RemainingBytes));
    }

    #[test]
    fn test_blind_computation() {
        // The nested leaf is blinded
        let absent_shape = MerkleProof::Node(vec![
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[0, 1, 2])),
            MerkleProof::Node(vec![MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[
                0, 1, 2,
            ]))]),
        ]);
        let comp_fn =
            computation_i16::<ProofTreeDeserialiser>(ProofTree::Present(&absent_shape).into());

        assert_eq!(comp_fn.unwrap().into_result(), -1);

        // For computation_2, the provided merkle proof will resolve as blinded
        // since root is blinded
        let merkle_proof = MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[6, 7, 8]));
        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkle_proof).into();
        let comp_fn = computation_leaves(proof).unwrap();
        assert_eq!(comp_fn.into_result(), -1);
    }

    #[test]
    fn test_blind_computation_stream() {
        // The nested leaf is blinded
        let b1: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[0, 1, 2]).into();
        let b2: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[0, 1, 2]).into();
        let raw_bytes_content = [
            [TAG_NODE, TAG_BLIND].as_ref(),
            b1.as_ref(),
            [TAG_NODE, TAG_BLIND].as_ref(),
            b2.as_ref(),
        ]
        .concat();

        let input = StreamInput::new(&raw_bytes_content);
        let comp_fn = computation_i16::<StreamDeserialiser>(StreamDeserialiser::new_present(input));
        let (res, _) = comp_fn.unwrap().into_result().unwrap();

        assert_eq!(res, -1);

        // For computation_2, the provided merkle proof will resolve as blinded
        // since root is blinded
        let merkle_proof = MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[6, 7, 8]));
        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkle_proof).into();
        let comp_fn = computation_leaves(proof).unwrap();
        assert_eq!(comp_fn.into_result(), -1);
    }

    #[test]
    fn test_bad_structure() {
        let bad_shape_1 = MerkleProof::Node(vec![]);
        let bad_shape_2 = MerkleProof::Node(vec![
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[0, 1, 2])),
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[0, 1, 2])),
            MerkleProof::Node(vec![]),
            MerkleProof::Node(vec![]),
            MerkleProof::Node(vec![]),
        ]);
        let bad_shape_3 = MerkleProof::Node(vec![
            MerkleProof::Node(vec![]),
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[0, 1, 2])),
        ]);
        let bad_shape_4 = MerkleProof::Node(vec![
            MerkleProof::leaf_read([42_u8; 32].to_vec()),
            MerkleProof::leaf_read(100_i32.to_le_bytes().to_vec()),
        ]);

        // Tree is missing branches
        let comp_fn =
            computation_i16::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_1).into());
        assert!(comp_fn.is_err_and(|e| matches!(e, ProofError::BadNumberOfBranches { .. })));

        // First 2 children of root are ok in shape (blinded) but the total number of children does not correspond
        // Ideally, we would like to have expected: 2, got: 5, but the implementation for `ProofTreeDeserialiser`
        // does not track this information (the original number of children)
        let comp_fn =
            computation_i16::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_2).into());
        assert!(comp_fn.is_err_and(|e| {
            println!("{e:?}");
            matches!(e, ProofError::BadNumberOfBranches {
                expected: 0,
                got: 3
            })
        }));

        // The first child is a node, but is expected to be a leaf
        let comp_fn =
            computation_i16::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_3).into());
        assert!(comp_fn.is_err_and(|e| matches!(e, ProofError::UnexpectedNode)));

        // The second child is a leaf, but is expected to be a node
        let comp_fn =
            computation_i16::<ProofTreeDeserialiser>(ProofTree::Present(&bad_shape_4).into());
        assert!(comp_fn.is_err_and(|e| { matches!(e, ProofError::UnexpectedLeaf) }));
    }

    #[test]
    fn test_bad_structure_stream() {
        let hash: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[0, 1, 2]).into();
        // Place an invalid second tag
        // Bad tag introduced after the first node
        let res = run_stream_deserialiser(computation_i16, [TAG_NODE, 0b01].as_ref());

        if let ProofError::Deserialise(bincode::error::DecodeError::OtherString(message)) =
            res.unwrap_err()
        {
            assert_eq!(message, InvalidTagError.to_string());
        } else {
            panic!("Expected an InvalidTagError");
        }

        // First 2 children of root are ok in shape (blinded) but because the extra byte in tags
        // will be counted towards the blinded hashes a RemainingBytes error will occur.
        let bytes = &[
            [TAG_NODE, TAG_BLIND].as_ref(),
            hash.as_ref(),
            [TAG_BLIND].as_ref(),
            hash.as_ref(),
            [TAG_NODE, TAG_NODE, TAG_NODE].as_ref(),
        ]
        .concat();
        let res = run_stream_deserialiser(computation_i16, bytes);
        assert!(matches!(res, Err(ProofError::RemainingBytes)));

        // The first child is a node, but is expected to be a leaf
        let res = run_stream_deserialiser(
            computation_i16,
            &[[TAG_NODE, TAG_NODE, TAG_BLIND].as_ref(), hash.as_ref()].concat(),
        );
        assert!(matches!(res, Err(ProofError::UnexpectedNode)));

        // The second child is a read leaf, but is expected to be a node
        let res = run_stream_deserialiser(
            computation_i16,
            &[
                [TAG_NODE, TAG_READ].as_ref(),
                hash.as_ref(),
                [TAG_READ].as_ref(),
                hash.as_ref(),
            ]
            .concat(),
        );
        assert!(matches!(res, Err(ProofError::UnexpectedLeaf)));
    }

    #[test]
    fn test_valid_computation() {
        let merkleproof = MerkleProof::Node(vec![
            MerkleProof::leaf_read(0x140A_0000_i32.to_le_bytes().to_vec()),
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[3, 4, 5])),
            MerkleProof::leaf_read(0xC0005_i32.to_le_bytes().to_vec()),
            MerkleProof::leaf_blind(Hash::blake3_hash_bytes(&[9, 10, 11])),
        ]);

        let proof: ProofTreeDeserialiser = ProofTree::Present(&merkleproof).into();
        let comp_fn = computation_leaves(proof).unwrap();
        assert_eq!(comp_fn.into_result(), 0x140A_0000 + 0xC0005);
    }

    #[test]
    fn test_valid_computation_stream() {
        let h1 = 0x140A_0000_i32.to_le_bytes();
        let h2: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[3, 4, 5]).into();
        let h3 = 0xC0005_i32.to_le_bytes();
        let h4: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[9, 10, 11]).into();

        let res = run_stream_deserialiser(
            computation_leaves,
            &[
                [TAG_NODE, TAG_READ].as_ref(),
                h1.as_ref(),
                [TAG_BLIND].as_ref(),
                h2.as_ref(),
                [TAG_READ].as_ref(),
                h3.as_ref(),
                [TAG_BLIND].as_ref(),
                h4.as_ref(),
            ]
            .concat(),
        );
        assert_eq!(res.unwrap(), 0x140A_0000 + 0xC0005);
    }
}
