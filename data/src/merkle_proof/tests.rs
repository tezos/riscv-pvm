// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

//! Tests for Merkle proofs

use std::num::NonZeroUsize;

use bincode::Decode;
use bincode::Encode;
use proptest::proptest;
use proptest::test_runner::TestCaseResult;

use super::proof_tree::MerkleProofFold;
use super::proof_tree::MinimumPresence;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::hash::Hash;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::DeserialiserNode;
use crate::merkle_proof::Partial;
use crate::merkle_proof::ProofError;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::descend_tree;
use crate::merkle_proof::proof_binary::StreamDeserialiser;
use crate::merkle_proof::proof_binary::StreamInput;
use crate::merkle_proof::proof_binary::StreamParserComb;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::ProofTree;
use crate::merkle_proof::proof_tree::ProofTreeResult;
use crate::merkle_proof::tag::InvalidTagError;
use crate::merkle_proof::tag::TAG_BLIND;
use crate::merkle_proof::tag::TAG_NODE;
use crate::merkle_proof::tag::TAG_READ;
use crate::serialisation::serialise;

fn generic_computation<T: Into<i32> + Decode<()>, D: Deserialiser>(
    proof: D,
) -> Result<<D as Deserialiser>::Suspended<i32>, D::Error> {
    // The tree structure:
    // Node (root)
    // ├── Leaf (type: Hash)
    // └── Node
    //     └── Leaf (type: T)

    // Computation: return the value of the nested leaf

    let ctx = proof.into_node()?;
    let (ctx, _left) = ctx.next_branch_with(|br_proof| br_proof.into_leaf::<Hash>())?;
    let (ctx, right) = ctx.next_branch_with(|br_ctx| {
        let ctx = br_ctx.into_node()?;
        let (ctx, result) = ctx.next_branch_with(|pr| pr.into_leaf::<T>())?;
        ctx.done(result)
    })?;

    ctx.done(match right {
        Partial::Present(nr) => nr.into(),
        Partial::Absent => 0,
        Partial::Blinded(_hash) => -1,
    })
}

fn computation_i16<D: Deserialiser>(
    proof: D,
) -> Result<<D as Deserialiser>::Suspended<i32>, D::Error> {
    generic_computation::<i16, D>(proof)
}

fn computation_bool<D: Deserialiser>(
    proof: D,
) -> Result<<D as Deserialiser>::Suspended<i32>, D::Error> {
    generic_computation::<bool, D>(proof)
}

fn computation_leaves<D: Deserialiser>(
    proof: D,
) -> Result<<D as Deserialiser>::Suspended<i32>, D::Error> {
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

    let ctx = (0..4).try_fold(ctx, |ctx, _| -> Result<_, D::Error> {
        let (ctx, val) = ctx.next_branch_with(|br_proof| br_proof.into_leaf::<i32>())?;

        if let Partial::Present(val) = val {
            data.push(val);
        }

        Ok(ctx)
    })?;

    ctx.done(data.into_iter().sum())
}

/// Execute a deserialising computation over an owned Merkle proof.
fn run_owned_deserialiser<'t>(
    deser: impl FnOnce(ProofTree<'t>) -> Result<ProofTreeResult<'t, i32>, ProofError>,
    merkle_proof: &'t MerkleProof,
) -> Result<i32, ProofError> {
    let proof = ProofTree::Present(merkle_proof);
    let parsed_result = deser(proof)?;
    Ok(parsed_result.into_result())
}

/// Execute a deserialising computation over raw bytes.
fn run_stream_deserialiser<'t>(
    deser: impl FnOnce(StreamDeserialiser<'t>) -> Result<StreamParserComb<'t, i32>, ProofError>,
    bytes: &'t [u8],
) -> Result<i32, ProofError> {
    let input = StreamInput::new(bytes);
    let comp_fn = deser(StreamDeserialiser::new_present(input))?;
    comp_fn.into_result().map(|(ret, _)| ret)
}

#[test]
fn test_absent_computation() {
    // Root is absent already
    let proof = ProofTree::Absent;
    let comp_fn = computation_i16(proof).unwrap();
    assert_eq!(comp_fn.into_result(), 0);

    // We expect to get the Absent case since the father of the nested node is blinded
    let merkle_proof = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read(Hash::hash_bytes(&[0, 1, 2]).as_ref().to_vec()),
        MerkleProof::leaf_blind(Hash::hash_bytes(&[3, 4, 5])),
    ]);
    let proof = ProofTree::Present(&merkle_proof);
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
    let leaf_read: [u8; Hash::DIGEST_SIZE] = [12; Hash::DIGEST_SIZE];
    let leaf_blind: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[3, 4, 5]).into();
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
    let hash_read: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[0, 1, 2]).into();
    let hash_read_raw: [u8; Hash::DIGEST_SIZE] = hash_read;
    let bool_read = [1u8];

    // Note the truncated hash
    let raw_bytes_content = [
        [TAG_NODE, TAG_READ].as_ref(),
        hash_read_raw[0..5].as_ref(),
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
    let merkle_proof = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read(hash_read_raw[0..5].to_vec()),
        MerkleProof::node_without_data(vec![MerkleProof::leaf_read(bool_read.to_vec())]),
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
    let hash_read: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[0, 1, 2]).into();
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

    let merkle_proof = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read(hash_read.to_vec()),
        MerkleProof::node_without_data(vec![MerkleProof::leaf_read(bad_bool_bincode.to_vec())]),
    ]);
    let res = run_owned_deserialiser(computation_bool, &merkle_proof);
    eprintln!("Result: {res:?}");
    assert!(matches!(res, Err(ProofError::Deserialise(_))));
}

#[test]
fn test_too_many_bytes_error() {
    let tag_bytes = [TAG_NODE, TAG_READ, TAG_NODE, TAG_READ];
    let hash_read: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[0, 1, 2]).into();
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
    let absent_shape = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_blind(Hash::hash_bytes(&[0, 1, 2])),
        MerkleProof::node_without_data(vec![MerkleProof::leaf_blind(Hash::hash_bytes(&[0, 1, 2]))]),
    ]);
    let comp_fn = computation_i16::<ProofTree>(ProofTree::Present(&absent_shape));

    assert_eq!(comp_fn.unwrap().into_result(), -1);

    // For computation_2, the provided merkle proof will resolve as blinded
    // since root is blinded
    let merkle_proof = MerkleProof::leaf_blind(Hash::hash_bytes(&[6, 7, 8]));
    let proof = ProofTree::Present(&merkle_proof);
    let comp_fn = computation_leaves(proof).unwrap();
    assert_eq!(comp_fn.into_result(), -1);
}

#[test]
fn test_blind_computation_stream() {
    // The nested leaf is blinded
    let b1: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[0, 1, 2]).into();
    let b2: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[0, 1, 2]).into();
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
    let merkle_proof = MerkleProof::leaf_blind(Hash::hash_bytes(&[6, 7, 8]));
    let proof = ProofTree::Present(&merkle_proof);
    let comp_fn = computation_leaves(proof).unwrap();
    assert_eq!(comp_fn.into_result(), -1);
}

#[test]
fn test_bad_structure() {
    let bad_shape_1 = MerkleProof::node_without_data(vec![]);
    let bad_shape_2 = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_blind(Hash::hash_bytes(&[0, 1, 2])),
        MerkleProof::leaf_blind(Hash::hash_bytes(&[0, 1, 2])),
        MerkleProof::node_without_data(vec![]),
        MerkleProof::node_without_data(vec![]),
        MerkleProof::node_without_data(vec![]),
    ]);
    let bad_shape_3 = MerkleProof::node_without_data(vec![
        MerkleProof::node_without_data(vec![]),
        MerkleProof::leaf_blind(Hash::hash_bytes(&[0, 1, 2])),
    ]);
    let bad_shape_4 = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read([42_u8; 32].to_vec()),
        MerkleProof::leaf_read(100_i32.to_le_bytes().to_vec()),
    ]);

    // Tree is missing branches
    let comp_fn = computation_i16::<ProofTree>(ProofTree::Present(&bad_shape_1));
    assert!(comp_fn.is_err_and(|e| matches!(e, ProofError::BadNumberOfBranches { .. })));

    // First 2 children of root are ok in shape (blinded) but the total number of children does not correspond
    // Ideally, we would like to have expected: 2, got: 5, but the implementation for `ProofTree`
    // does not track this information (the original number of children)
    let comp_fn = computation_i16::<ProofTree>(ProofTree::Present(&bad_shape_2));
    assert!(comp_fn.is_err_and(|e| {
        println!("{e:?}");
        matches!(
            e,
            ProofError::BadNumberOfBranches {
                expected: 0,
                got: 3
            }
        )
    }));

    // The first child is a node, but is expected to be a leaf
    let comp_fn = computation_i16::<ProofTree>(ProofTree::Present(&bad_shape_3));
    assert!(comp_fn.is_err_and(|e| matches!(e, ProofError::UnexpectedNode)));

    // The second child is a leaf, but is expected to be a node
    let comp_fn = computation_i16::<ProofTree>(ProofTree::Present(&bad_shape_4));
    assert!(comp_fn.is_err_and(|e| { matches!(e, ProofError::UnexpectedLeaf) }));
}

#[test]
fn test_bad_structure_stream() {
    let hash: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[0, 1, 2]).into();
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
    let merkleproof = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read(0x140A_0000_i32.to_le_bytes().to_vec()),
        MerkleProof::leaf_blind(Hash::hash_bytes(&[3, 4, 5])),
        MerkleProof::leaf_read(0xC0005_i32.to_le_bytes().to_vec()),
        MerkleProof::leaf_blind(Hash::hash_bytes(&[9, 10, 11])),
    ]);

    let proof = ProofTree::Present(&merkleproof);
    let comp_fn = computation_leaves(proof).unwrap();
    assert_eq!(comp_fn.into_result(), 0x140A_0000 + 0xC0005);
}

#[test]
fn test_valid_computation_stream() {
    let h1 = 0x140A_0000_i32.to_le_bytes();
    let h2: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[3, 4, 5]).into();
    let h3 = 0xC0005_i32.to_le_bytes();
    let h4: [u8; Hash::DIGEST_SIZE] = Hash::hash_bytes(&[9, 10, 11]).into();

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

struct TestLeaf<T>(T);

impl<T: Encode> Foldable<MerkleProofFold> for TestLeaf<T> {
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        let data = serialise(&self.0).expect("Serialising u8 should not fail");
        builder.into_leaf(MinimumPresence::Present, data)
    }
}

#[test]
fn test_descend_tree_trailing_remainder() {
    let arity = 4;
    let leaves = 17;
    let generator = |idx: usize| TestLeaf(idx);
    let seq_as_tree = IndexableSeqAsTree::new(leaves, arity, &generator);
    let proof = MerkleProof::from_foldable(&seq_as_tree);

    let MerkleProof::Node(root) = &proof else {
        panic!("The sequence proof root should be a node")
    };
    assert_eq!(root.children.len(), 2);
    let MerkleProof::Node(right_depth_one) = &root.children[1] else {
        panic!("The right subtree should start with a node")
    };
    assert_eq!(right_depth_one.children.len(), 1);
    assert!(matches!(right_depth_one.children[0], MerkleProof::Node(_)));

    let mut visited = Vec::new();

    descend_tree(
        ProofTree::Present(&proof),
        arity,
        leaves,
        &mut |idx, proof| {
            let leaf = proof.into_leaf::<usize>()?;
            Ok(leaf.map(|leaf| {
                let Partial::Present(value) = leaf else {
                    panic!("Expected a present leaf in this proof")
                };
                visited.push((idx, value));
            }))
        },
    )
    .map(ProofTreeResult::into_result)
    .expect("descend_tree should parse a 17-leaf arity-4 tree with trailing remainder");

    let expected: Vec<_> = (0..leaves).map(|idx| (idx, idx)).collect();
    assert_eq!(visited, expected);
}

#[test]
fn round_trip_descend_tree_indexable_seq_as_tree() {
    fn test(data: Vec<usize>, arity: NonZeroUsize) -> TestCaseResult {
        let get_item = |idx: usize| TestLeaf(data[idx]);
        let seq_as_tree = IndexableSeqAsTree::new(data.len(), arity.get(), &get_item);
        let proof = MerkleProof::from_foldable(&seq_as_tree);

        descend_tree(
            ProofTree::Present(&proof),
            arity.get(),
            data.len(),
            &mut |idx, proof| {
                let leaf = proof.into_leaf::<usize>()?;
                Ok(leaf.map(|leaf| {
                    let Partial::Present(value) = leaf else {
                        panic!("Expected a present leaf in this proof")
                    };
                    assert_eq!(value, data[idx]);
                }))
            },
        )
        .unwrap()
        .into_result();

        Ok(())
    }

    proptest!(|(data: Vec<usize>, arity in 1..17usize)| {
        test(data, NonZeroUsize::new(arity).unwrap())?;
    });
}
