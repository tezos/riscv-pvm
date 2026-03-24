// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Tests for the [`Vector`] component

use proptest::collection::vec;
use proptest::prelude::Just;
use proptest::prelude::Strategy;
use proptest::prelude::any;
use proptest::prop_assert_eq;
use proptest::prop_oneof;
use proptest::proptest;
use proptest::test_runner::TestCaseResult;

use super::NODE_ARITY;
use super::Vector;
use crate::components::atom::Atom;
use crate::components::atom::tests::AtomMutOp;
use crate::components::atom::tests::AtomOp;
use crate::components::bytes::Bytes;
use crate::components::bytes::BytesMode;
use crate::components::bytes::tests::BytesMutOp;
use crate::components::bytes::tests::BytesOp;
use crate::components::vector::VectorMode;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::Unfoldable;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::foldable::tests::TestFolder;
use crate::foldable::tests::TestTree;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::proof_binary;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofFold;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode_test;
use crate::serialisation::serialise;

impl<T: Foldable<TestFolder>> Foldable<TestFolder> for Vector<T, Normal> {
    fn fold(&self, builder: TestFolder) -> TestTree {
        let length = self.len();
        let length_node = TestTree::Leaf(serialise(length as u64).unwrap());

        let get_item = |idx: usize| &self[idx];
        let seq_as_tree = IndexableSeqAsTree::new(length, NODE_ARITY, &get_item);

        let mut node = builder.into_node_fold();
        node.add(&length_node);
        node.add(&seq_as_tree);
        node.done()
    }
}

// Test that the Vector doesn't drop any values on construction.
mode_test!(len_and_is_empty_match_initial_values, F, {
    proptest!(|(initial_values in vec(any::<u64>(), 0..64))| {
        let vector: Vector<Atom<u64, F>, F> =
            Vector::new(initial_values.iter().copied().map(Atom::new).collect());
        prop_assert_eq!(vector.len(), initial_values.len());
        prop_assert_eq!(vector.is_empty(), initial_values.is_empty());
    });
});

/// Operations to be issued against an immutable [`Vector`] state component
#[derive(Debug, Clone)]
enum VectorOp<RefOp> {
    Len,
    Index { idx: usize, op: RefOp },
}

impl<RefOp> VectorOp<RefOp> {
    /// Run an operation against an immutable [`Vector`] state component.
    fn run<T, M: VectorMode, R>(
        &self,
        vector: &Vector<T, M>,
        run_ref_op: impl Fn(&RefOp, &T) -> R,
    ) -> VectorOpResult<R> {
        match self {
            Self::Len => VectorOpResult::Len { len: vector.len() },
            Self::Index { idx, op } => {
                let len = vector.len();

                // We need to cap the length to avoid out-of-bounds panics, given the vector can be
                // resized.
                let Some(idx) = idx.checked_rem(len) else {
                    return VectorOpResult::Void;
                };

                let item = &vector[idx];
                let result = run_ref_op(op, item);
                VectorOpResult::Inner { result }
            }
        }
    }
}

/// Operations to be issued against a mutable [`Vector`] state component
#[derive(Debug, Clone)]
enum VectorMutOp<RefOp, RefMutOp> {
    Immutable { op: VectorOp<RefOp> },
    IndexMut { idx: usize, op: RefMutOp },
    Resize { new_len: usize },
}

impl<RefOp, RefMutOp> VectorMutOp<RefOp, RefMutOp> {
    /// Run the operation against the [`Vector`] state component.
    fn run<T: Default, M: VectorMode, R>(
        &self,
        vector: &mut Vector<T, M>,
        run_ref_op: impl Fn(&RefOp, &T) -> R,
        run_ref_mut_op: impl Fn(&RefMutOp, &mut T) -> R,
    ) -> VectorOpResult<R> {
        match self {
            Self::Immutable { op } => op.run(vector, run_ref_op),
            Self::IndexMut { idx, op } => {
                let len = vector.len();

                // We need to cap the length to avoid out-of-bounds panics, given the vector can be
                // resized.
                let Some(idx) = idx.checked_rem(len) else {
                    return VectorOpResult::Void;
                };

                let item = &mut vector[idx];
                let result = run_ref_mut_op(op, item);
                VectorOpResult::Inner { result }
            }
            Self::Resize { new_len } => {
                vector.resize_with(*new_len, T::default);
                VectorOpResult::Void
            }
        }
    }
}

/// Results of operations issued against the [`Vector`] state component
#[derive(Debug, PartialEq, Eq)]
enum VectorOpResult<R> {
    Void,
    Len { len: usize },
    Inner { result: R },
}

/// Strategy for generating operations for a [`Vector`].
fn vector_mut_ops<RefOp, RefMutOp>(
    initial_len: usize,
    ref_op: impl Strategy<Value = RefOp> + 'static,
    ref_mut_op: impl Strategy<Value = RefMutOp> + 'static,
) -> impl Strategy<Value = Vec<VectorMutOp<RefOp, RefMutOp>>>
where
    RefOp: Clone + std::fmt::Debug + 'static,
    RefMutOp: Clone + std::fmt::Debug + 'static,
{
    let range_end = initial_len.saturating_mul(2).saturating_add(1);

    let len_op = Just(VectorMutOp::Immutable { op: VectorOp::Len });

    let index_op = (0usize..range_end, ref_op).prop_map(move |(idx, op)| VectorMutOp::Immutable {
        op: VectorOp::Index { idx, op },
    });

    let index_mut_op = (0usize..range_end, ref_mut_op)
        .prop_map(move |(idx, op)| VectorMutOp::IndexMut { idx, op });

    let resize_op = (0usize..=range_end).prop_map(|new_len| VectorMutOp::Resize { new_len });

    vec(
        prop_oneof![len_op, index_op, index_mut_op, resize_op],
        1..20,
    )
}

/// Test case for the vector implementation
#[derive(Debug, Clone)]
struct VectorCase<InitElem, RefOp, RefMutOp> {
    initial_values: Vec<InitElem>,
    ops: Vec<VectorMutOp<RefOp, RefMutOp>>,
}

/// Strategy for generating test cases for the vector implementation
fn vector_case<InitElem, RefOp, RefMutOp>(
    initial_values: impl Strategy<Value = InitElem> + 'static,
    ref_op: impl Strategy<Value = RefOp> + Clone + 'static,
    ref_mut_op: impl Strategy<Value = RefMutOp> + Clone + 'static,
) -> impl Strategy<Value = VectorCase<InitElem, RefOp, RefMutOp>>
where
    InitElem: Clone + std::fmt::Debug + 'static,
    RefOp: Clone + std::fmt::Debug + 'static,
    RefMutOp: Clone + std::fmt::Debug + 'static,
{
    vec(initial_values, 0..20).prop_flat_map(move |initial_values| {
        let ops = vector_mut_ops(initial_values.len(), ref_op.clone(), ref_mut_op.clone());
        (Just(initial_values), ops).prop_map(|(initial_values, ops)| VectorCase {
            initial_values,
            ops,
        })
    })
}

/// Strategy for generating test cases for the vector implementation using [`Bytes`] operations
fn vector_bytes_case() -> impl Strategy<Value = VectorCase<Vec<u8>, BytesOp, BytesMutOp>> {
    vector_case(vec(any::<u8>(), 0..32), BytesOp::any(), BytesMutOp::any())
}

/// Strategy for generating test cases for the vector implementation using [`Atom`] operations
fn vector_atom_u64_case() -> impl Strategy<Value = VectorCase<u64, AtomOp, AtomMutOp<u64>>> {
    vector_case(any::<u64>(), AtomOp::any(), AtomMutOp::<u64>::any())
}

/// Shared harness asserting the same vector behavior in all modes
fn run_vector_is_same_across_modes<RefOp, RefMutOp, InnerNormal, InnerProve, InnerVerify, Res>(
    initial_normal: Vec<InnerNormal>,
    initial_prove: Vec<InnerProve>,
    initial_verify: Vec<InnerVerify>,
    ops: &[VectorMutOp<RefOp, RefMutOp>],
    mut run_normal: impl FnMut(
        &VectorMutOp<RefOp, RefMutOp>,
        &mut Vector<InnerNormal, Normal>,
    ) -> VectorOpResult<Res>,
    mut run_prove: impl FnMut(
        &VectorMutOp<RefOp, RefMutOp>,
        &mut Vector<InnerProve, Prove>,
    ) -> VectorOpResult<Res>,
    mut run_verify: impl FnMut(
        &VectorMutOp<RefOp, RefMutOp>,
        &mut Vector<InnerVerify, Verify>,
    ) -> VectorOpResult<Res>,
) -> TestCaseResult
where
    Res: std::fmt::Debug + Eq,
    InnerNormal: Default + Foldable<HashFold>,
    InnerProve: Default + Foldable<HashFold> + Foldable<MerkleProofFold>,
    InnerVerify: Default + Foldable<PartialHashFold>,
{
    let mut vector_normal = Vector::<InnerNormal, Normal>::new(initial_normal);
    let hash_initial_normal = Hash::from_foldable(&vector_normal);

    let results_normal = ops
        .iter()
        .map(|op| run_normal(op, &mut vector_normal))
        .collect::<Vec<_>>();
    let hash_normal = Hash::from_foldable(&vector_normal);

    let mut vector_prove = Vector::<InnerProve, Prove>::new(initial_prove);

    let hash_initial_prove = Hash::from_foldable(&vector_prove);
    prop_assert_eq!(hash_initial_normal, hash_initial_prove);

    let results_prove = ops
        .iter()
        .map(|op| run_prove(op, &mut vector_prove))
        .collect::<Vec<_>>();
    prop_assert_eq!(&results_normal, &results_prove);

    let hash_prove = Hash::from_foldable(&vector_prove);
    prop_assert_eq!(hash_normal, hash_prove);

    let merkle_proof = MerkleProof::from_foldable(&vector_prove);

    let mut vector_verify = Vector::<InnerVerify, Verify>::new(initial_verify);

    let hash_initial_verify =
        PartialHash::from_foldable(Some(merkle_proof.clone()), &vector_verify)
            .to_hash()
            .unwrap();
    prop_assert_eq!(hash_initial_normal, hash_initial_verify);

    let results_verify = ops
        .iter()
        .map(|op| run_verify(op, &mut vector_verify))
        .collect::<Vec<_>>();
    prop_assert_eq!(&results_normal, &results_verify);

    let hash_verify = PartialHash::from_foldable(Some(merkle_proof), &vector_verify)
        .to_hash()
        .unwrap();
    prop_assert_eq!(hash_normal, hash_verify);

    Ok(())
}

fn bytes_from_data<M: BytesMode>(data: &[u8]) -> Bytes<M> {
    let mut bytes = Bytes::<M>::default();
    bytes.set(data);
    bytes
}

#[test]
fn vector_of_bytes_is_same_across_modes() {
    proptest!(|(case in vector_bytes_case())| {
        let initial_normal = case.initial_values
            .iter()
            .map(|data| bytes_from_data::<Normal>(data))
            .collect::<Vec<_>>();

        let initial_prove = case.initial_values
            .iter()
            .map(|data| bytes_from_data::<Prove>(data))
            .collect::<Vec<_>>();

        let initial_verify = case.initial_values
            .iter()
            .map(|data| bytes_from_data::<Verify>(data))
            .collect::<Vec<_>>();

        run_vector_is_same_across_modes(
            initial_normal,
            initial_prove,
            initial_verify,
            &case.ops,
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, bytes| inner_op.run(bytes),
                    |inner_op, bytes| inner_op.run(bytes),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, bytes| inner_op.run(bytes),
                    |inner_op, bytes| inner_op.run(bytes),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, bytes| inner_op.run(bytes),
                    |inner_op, bytes| inner_op.run(bytes),
                )
            },
        )?;
    });
}

#[test]
fn vector_of_atom_u64_is_same_across_modes() {
    proptest!(|(case in vector_atom_u64_case())| {
        let initial_normal = case.initial_values
            .iter()
            .copied()
            .map(Atom::<u64, Normal>::new)
            .collect::<Vec<_>>();

        let initial_prove = case.initial_values
            .iter()
            .copied()
            .map(Atom::<u64, Prove>::new)
            .collect::<Vec<_>>();

        let initial_verify = case.initial_values
            .iter()
            .copied()
            .map(Atom::<u64, Verify>::new)
            .collect::<Vec<_>>();

        run_vector_is_same_across_modes(
            initial_normal,
            initial_prove,
            initial_verify,
            &case.ops,
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, atom| inner_op.run(atom),
                    |inner_op, atom| inner_op.run(atom),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, atom| inner_op.run(atom),
                    |inner_op, atom| inner_op.run(atom),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, atom| inner_op.run(atom),
                    |inner_op, atom| inner_op.run(atom),
                )
            },
        )?;
    });
}

fn proof_round_trip<Res, Op, OpMut, InnerNormal, InnerVerify>(
    mut vector_normal: Vector<InnerNormal, Normal>,
    ops: Vec<VectorMutOp<Op, OpMut>>,
    run_normal: impl Fn(
        &VectorMutOp<Op, OpMut>,
        &mut Vector<InnerNormal, Normal>,
    ) -> VectorOpResult<Res>,
    run_prove: impl for<'normal> Fn(
        &VectorMutOp<Op, OpMut>,
        &mut Vector<<InnerNormal as Provable<'normal>>::Prover, Prove<'normal>>,
    ) -> VectorOpResult<Res>,
    run_verify: impl Fn(
        &VectorMutOp<Op, OpMut>,
        &mut Vector<InnerVerify, Verify>,
    ) -> VectorOpResult<Res>,
) -> TestCaseResult
where
    Res: std::fmt::Debug + Eq,
    InnerNormal: Foldable<HashFold> + for<'normal> Provable<'normal>,
    for<'normal> <InnerNormal as Provable<'normal>>::Prover:
        Foldable<HashFold> + Foldable<MerkleProofFold>,
    InnerVerify: FromProof + Foldable<PartialHashFold>,
{
    for op in ops {
        // We use a block expression to clearly delimit the lifetimes. Otherwise, the compiler will
        // freak out over using the `vector_normal` while `vector_prove` is still alive.
        let (verify_result, after_verify_hash) = {
            let mut vector_prove = vector_normal.start_proof();

            // The initial hash of the Prove mode should match the Normal mode hash.
            let init_normal_hash = Hash::from_foldable(&vector_normal);
            let init_prove_hash = Hash::from_foldable(&vector_prove);
            prop_assert_eq!(init_normal_hash, init_prove_hash);

            // Run the operation which we would like to prove.
            let prove_result = run_prove(&op, &mut vector_prove);

            // The post-operation hash is later used to compare against the Normal mode hash.
            let after_proof_hash = Hash::from_foldable(&vector_prove);

            // The Merkle proof tree should match the state hash before the operation was applied.
            let proof_tree = MerkleProof::from_foldable(&vector_prove);
            prop_assert_eq!(init_prove_hash, proof_tree.root_hash());

            // We want to serialise the proof to its binary format to make this test more realistic.
            let proof_bytes = serialise(proof_tree).unwrap();

            // Parsing the proof so we can see if the proof generation worked.
            let (mut vector_verify, parsed_proof_tree) =
                proof_binary::deserialise::<Vector<InnerVerify, Verify>>(&proof_bytes).unwrap();
            let parsed_proof_tree = parsed_proof_tree.into_present();

            // The parsed state should have a state hash equal to that of the initial Normal/Prove state.
            let init_verify_hash =
                PartialHash::from_foldable(parsed_proof_tree.clone(), &vector_verify)
                    .to_hash()
                    .unwrap();
            prop_assert_eq!(init_verify_hash, init_prove_hash);

            // Run the operation which we would like to verify.
            let verify_result = run_verify(&op, &mut vector_verify);
            prop_assert_eq!(&verify_result, &prove_result);

            // The post-operation hash should match the Normal mode hash.
            let after_verify_hash = PartialHash::from_foldable(parsed_proof_tree, &vector_verify)
                .to_hash()
                .unwrap();
            prop_assert_eq!(after_verify_hash, after_proof_hash);

            (verify_result, after_verify_hash)
        };

        // Finally advance the Normal mode state as well.
        let normal_result = run_normal(&op, &mut vector_normal);
        prop_assert_eq!(&normal_result, &verify_result);

        // The Normal mode hash should match the post-operation hash that we proved and verified.
        let after_normal_hash = Hash::from_foldable(&vector_normal);
        prop_assert_eq!(after_normal_hash, after_verify_hash);
    }

    Ok(())
}

#[test]
fn proof_round_trip_atom_u64() {
    proptest!(|(case in vector_atom_u64_case())| {
        let vector_normal = Vector::<Atom<u64, Normal>, Normal>::new(
            case.initial_values
                .into_iter()
                .map(Atom::<u64, Normal>::new)
                .collect(),
        );

        proof_round_trip(
            vector_normal,
            case.ops,
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, atom| inner_op.run(atom),
                    |inner_op, atom| inner_op.run(atom),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, atom| inner_op.run(atom),
                    |inner_op, atom| inner_op.run(atom),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, atom| inner_op.run(atom),
                    |inner_op, atom| inner_op.run(atom),
                )
            }
        )?;
    });
}

#[test]
fn proof_round_trip_bytes() {
    proptest!(|(case in vector_bytes_case())| {
        let vector_normal = Vector::<Bytes<Normal>, Normal>::new(
            case.initial_values
                .into_iter()
                .map(|vec| bytes_from_data(&vec))
                .collect(),
        );

        proof_round_trip(
            vector_normal,
            case.ops,
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, bytes| inner_op.run(bytes),
                    |inner_op, bytes| inner_op.run(bytes),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, bytes| inner_op.run(bytes),
                    |inner_op, bytes| inner_op.run(bytes),
                )
            },
            |op, vector| {
                op.run(
                    vector,
                    |inner_op, bytes| inner_op.run(bytes),
                    |inner_op, bytes| inner_op.run(bytes),
                )
            }
        )?;
    });
}

#[test]
fn fold_unfold_atom_u64() {
    proptest!(|(values in vec(any::<u64>(), 0..64))| {
        let vector: Vector<Atom<u64, Normal>, Normal> = Vector::new(
            values.into_iter().map(Atom::new).collect(),
        );

        let tree = vector.fold(TestFolder);
        let unfolded: Vector<Atom<u64, Normal>, Normal> = Vector::unfold(tree).unwrap();

        prop_assert_eq!(vector, unfolded);
    });
}

#[test]
fn fold_unfold_bytes() {
    proptest!(|(values in vec(vec(any::<u8>(), 0..32), 0..64))| {
        let vector: Vector<Bytes<Normal>, Normal> = Vector::new(
            values.iter().map(|data| bytes_from_data::<Normal>(data)).collect(),
        );

        let tree = vector.fold(TestFolder);
        let unfolded: Vector<Bytes<Normal>, Normal> = Vector::unfold(tree).unwrap();

        prop_assert_eq!(vector, unfolded);
    });
}
