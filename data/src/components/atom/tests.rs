// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`Atom`]

use proptest::arbitrary::Arbitrary;
use proptest::collection::vec;
use proptest::prelude::Just;
use proptest::prelude::any;
use proptest::prop_assert;
use proptest::prop_assert_eq;
use proptest::prop_oneof;
use proptest::proptest;
use proptest::strategy::Strategy;

use crate::components::atom::Atom;
use crate::components::atom::AtomMode;
use crate::foldable::Foldable;
use crate::foldable::Unfoldable;
use crate::foldable::tests::TestFolder;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::merkle_proof::proof_tree;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::merkle_proof::proof_tree::ProofTree;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::catch_not_found;
use crate::mode_test;
use crate::serialisation::deserialise;
use crate::serialisation::serialise;

mode_test!(init, F, {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct MyFoo(u64);

    impl Default for MyFoo {
        fn default() -> Self {
            MyFoo(1337)
        }
    }

    assert_eq!(Atom::<MyFoo, F>::default().read(), MyFoo::default());
});

#[test]
fn verify_present() {
    proptest!(|(reg: [u64; 32])| {
        let mut atoms: Atom<[_; 32], Verify> = Atom::new(reg);

        for i in 0..32 {
            let value = catch_not_found(|| atoms[i]).ok();
            proptest::prop_assert_eq!(value, Some(reg[i]));

            let new_value = rand::random();
            atoms[i] = new_value;

            let read_value = atoms[i];
            proptest::prop_assert_eq!(read_value, new_value);
        }
    });
}

#[test]
fn verify_absent() {
    let cells: Atom<[u64; 32], Verify> = Atom::absent();

    for i in 0..32 {
        let value = catch_not_found(|| cells[i]).ok();
        assert_eq!(value, None);
    }
}

#[test]
fn partial_hash_absent() {
    let verify_cell: Atom<u64, Verify> = Atom::absent();
    let proof = None;

    let hash = PartialHash::from_foldable(proof, &verify_cell);
    assert_eq!(hash, PartialHash::Previous);
}

#[test]
fn partial_hash_absent_written() {
    let mut verify_cell: Atom<u64, Verify> = Atom::absent();
    let proof = None;

    let written_value = 1337;
    verify_cell.write(written_value);

    let value_hash = Hash::hash_encodable(written_value).unwrap();
    let expected_state_hash = PartialHash::Present(value_hash);
    let hash = PartialHash::from_foldable(proof, &verify_cell);
    assert_eq!(hash, expected_state_hash);
}

#[test]
fn partial_hash_present_written() {
    let mut verify_cell: Atom<u64, Verify> = Atom::new(42);
    let proof = None;

    let written_value = 1337;
    verify_cell.write(written_value);

    let value_hash = Hash::hash_encodable(written_value).unwrap();
    let expected_state_hash = PartialHash::Present(value_hash);
    let hash = PartialHash::from_foldable(proof, &verify_cell);
    assert_eq!(hash, expected_state_hash);
}

#[test]
fn example_struct_read_write_serialise() {
    struct Example<M: Mode> {
        first: Atom<u64, M>,
        second: Atom<[u32; 4], M>,
    }

    let first_value: u64 = rand::random();
    let second_value: [u32; 4] = rand::random();

    let mut instance: Example<Normal> = Example {
        first: Atom::default(),
        second: Atom::default(),
    };

    instance.first.write(first_value);
    assert_eq!(instance.first.read(), first_value);

    instance.second.write(second_value);
    assert_eq!(instance.second.read(), second_value);

    let first_value_read =
        u64::from_le_bytes(serialise(instance.first).unwrap().try_into().unwrap());
    assert_eq!(first_value_read, first_value);

    let second_value_read = {
        let data = serialise(instance.second).unwrap();
        assert_eq!(data.len(), std::mem::size_of::<[u32; 4]>());

        let (values, &[]) = data
            .as_slice()
            .as_chunks::<{ std::mem::size_of::<u32>() }>()
        else {
            panic!("Unexpected extra bytes in serialisation");
        };

        <[[u8; 4]; 4]>::try_from(values)
            .expect("Expected exactly 16 bytes")
            .map(u32::from_le_bytes)
    };
    assert_eq!(second_value_read, second_value);
}

#[test]
fn serialise_correctly() {
    proptest!(|(a: u64, b: u64, c: u64)| {
        let atoms: Atom<_, Normal> = Atom::new([a, b, c]);
        let bytes = serialise(&atoms).unwrap();

        let atoms_after: Atom<[u64; 3], Normal> = deserialise(&bytes).unwrap();

        assert_eq!(atoms.read(), atoms_after.read());

        for i in 0..3 {
            assert_eq!(atoms[i], atoms_after[i]);
        }

        let bytes_after = serialise(&atoms_after).unwrap();
        assert_eq!(bytes, bytes_after);

        // Serialisation is consistent with that of the `Prove` mode.
        let proof_atoms: Atom<_, Prove> = atoms.start_proof();
        let proof_bytes = serialise(&proof_atoms).unwrap();
        assert_eq!(bytes, proof_bytes);
    });
}

#[test]
fn serialise_like_value() {
    proptest!(|(value: u64)| {
        let cell: Atom<u64, Normal> = Atom::new(value);
        let binary_value = serialise(cell).unwrap();
        let expected_binary_value = serialise(value).unwrap();
        assert_eq!(binary_value, expected_binary_value);
    });
}

#[test]
fn proof_gen() {
    const CELLS_SIZE: usize = 32;

    proptest!(|(value_before: u64, value_after: u64, i in 0..CELLS_SIZE)| {
        let data_before = [value_before; CELLS_SIZE];
        let data_after = [value_after; CELLS_SIZE];

        // A read followed by a write
        let mut atoms: Atom<[u64; CELLS_SIZE], Prove> = Atom::new(data_before);
        prop_assert!(!atoms.was_accessed());
        let value = atoms[i];
        prop_assert_eq!(value, value_before);
        prop_assert!(atoms.was_accessed());
        atoms[i] = value_after;
        prop_assert!(atoms.was_accessed());

        // A write followed by a read
        let mut atoms: Atom<[u64; CELLS_SIZE], Prove> = Atom::new(data_before);
        prop_assert!(!atoms.was_accessed());
        atoms[i] = value_after;
        prop_assert!(atoms.was_accessed());
        let value = atoms[i];
        prop_assert_eq!(value, value_after);
        prop_assert!(atoms.was_accessed());

        // A read_all followed by a write_all
        let mut atoms: Atom<[u64; CELLS_SIZE], Prove> = Atom::new(data_before);
        prop_assert!(!atoms.was_accessed());
        let values = atoms.read();
        prop_assert_eq!(values.as_slice(), data_before);
        prop_assert!(atoms.was_accessed());
        atoms.write(data_after);
        prop_assert!(atoms.was_accessed());

        // A write_all followed by a read_all
        let mut atoms: Atom<[u64; CELLS_SIZE], Prove> = Atom::new(data_before);
        prop_assert!(!atoms.was_accessed());
        atoms.write(data_after);
        prop_assert!(atoms.was_accessed());
        let values = atoms.read();
        prop_assert_eq!(values.as_slice(), data_after);
        prop_assert!(atoms.was_accessed());

        // Check correct Merkleisation
        let atoms: Atom<[u64; CELLS_SIZE], Normal> = Atom::new(data_before);
        let initial_root_hash = Hash::from_foldable(&atoms);

        let mut proof_atoms: Atom<[u64; CELLS_SIZE], Prove> = atoms.start_proof();
        proof_atoms[i] = value_after;

        let merkle_tree = MerkleProof::from_foldable(&proof_atoms);
        prop_assert_eq!(merkle_tree.root_hash(), initial_root_hash);
        prop_assert!(matches!(merkle_tree, MerkleProof::Leaf(MerkleProofLeaf::Read(_))));
    });
}

#[test]
fn proof_blinding() {
    const ATOMS_SIZE: usize = 32;

    type TestState<M> = (Atom<[u64; ATOMS_SIZE], M>, Atom<[u64; ATOMS_SIZE], M>);

    proptest!(|(value_before: u64, value_after: u64, i in 0..ATOMS_SIZE)| {
        let data_before = [value_before; ATOMS_SIZE];

        // Bind `Prove` atoms and write to one index.
        let mut proof_atoms1: Atom<[u64; ATOMS_SIZE], Prove> = Atom::new(data_before);
        proof_atoms1[i] = value_after;

        // Bind `Prove` atoms and do not access them.
        let proof_atoms2: Atom<[u64; ATOMS_SIZE], Prove> = Atom::new(data_before);

        let proof_state = (proof_atoms1, proof_atoms2);

        let merkle_proof = MerkleProof::from_foldable(&proof_state);

        let verifier_state =
            proof_tree::deserialise::<TestState<Verify>>(ProofTree::Present(&merkle_proof)).unwrap();

        // The first component of the state was present in the proof, can be
        // fully read, and contains the initial state.
        prop_assert_eq!(verifier_state.0.0.read(), [value_before; ATOMS_SIZE]);

        // The second component of the state is fully blinded: no values can
        // be read from the array.
        for i in 0..ATOMS_SIZE {
            prop_assert!(catch_not_found(|| verifier_state.0.1[i]).is_err());
        }

        let partial_hash = PartialHash::from_foldable(Some(merkle_proof), &verifier_state.0);
        prop_assert!(partial_hash.to_hash().is_some());
    })
}

/// Operations to be issued against an immutable Atom state component
#[derive(Debug, Clone)]
pub(crate) enum AtomOp {
    Read,
}

impl AtomOp {
    /// Strategy for generating operations to be issued against the Atom state component
    pub(crate) fn any() -> impl Strategy<Value = Self> + Clone {
        Just(Self::Read)
    }

    /// Run an operation against an immutable Atom state component.
    pub(crate) fn run<M: AtomMode, T: Copy + 'static>(&self, atom: &Atom<T, M>) -> AtomOpResult<T> {
        match self {
            Self::Read => AtomOpResult::Read { value: atom.read() },
        }
    }
}

/// Operations to be issued against a mutable Atom state component
#[derive(Debug, Clone)]
pub(crate) enum AtomMutOp<T> {
    Write { value: T },
    Immutable { op: AtomOp },
}

impl<T: Copy + 'static> AtomMutOp<T> {
    /// Run the operation against the Atom state component.
    pub(crate) fn run<M: AtomMode>(&self, atom: &mut Atom<T, M>) -> AtomOpResult<T> {
        match self {
            Self::Write { value } => {
                atom.write(*value);
                AtomOpResult::Void
            }

            Self::Immutable { op } => op.run(atom),
        }
    }
}

impl<T: Copy + Arbitrary + 'static> AtomMutOp<T> {
    /// Strategy for generating operations to be issued against the Atom state component
    pub(crate) fn any() -> impl Strategy<Value = Self> + Clone {
        prop_oneof![
            any::<T>().prop_map(|value| Self::Write { value }),
            AtomOp::any().prop_map(|op| Self::Immutable { op }),
        ]
    }
}

/// Results of operations issued against the Atom state component
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AtomOpResult<T> {
    Read { value: T },
    Void,
}

/// Atom behaves the same across different modes
#[test]
fn atom_is_same_across_modes() {
    proptest!(|(initial in any::<u64>(), ops in vec(AtomMutOp::<u64>::any(), 1..20))| {
        let mut atom_normal = Atom::<u64, Normal>::new(initial);
        let results_normal = ops.iter().map(|op| op.run(&mut atom_normal)).collect::<Vec<_>>();
        let hash_normal = Hash::from_foldable(&atom_normal);

        let mut atom_prove = Atom::<u64, Prove>::new(initial);
        let results_prove = ops.iter().map(|op| op.run(&mut atom_prove)).collect::<Vec<_>>();
        prop_assert_eq!(&results_normal, &results_prove);

        let hash_prove = Hash::from_foldable(&atom_prove);
        prop_assert_eq!(hash_normal, hash_prove);

        let merkle_proof = MerkleProof::from_foldable(&atom_prove);

        let mut atom_verify = Atom::<u64, Verify>::new(initial);
        let results_verify = ops.iter().map(|op| op.run(&mut atom_verify)).collect::<Vec<_>>();
        prop_assert_eq!(results_normal, results_verify);

        let hash_verify =
            PartialHash::from_foldable(Some(merkle_proof), &atom_verify)
                .to_hash()
                .unwrap();
        prop_assert_eq!(hash_normal, hash_verify);
    });
}

#[test]
fn fold_unfold() {
    let atom: Atom<String, Normal> = Atom::new("Hello world!".to_string());

    let tree = atom.fold(TestFolder);
    let unfolded = Atom::<String, Normal>::unfold(tree).unwrap();

    assert_eq!(atom, unfolded);
}
