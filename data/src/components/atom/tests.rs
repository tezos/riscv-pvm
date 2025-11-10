// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`Atom`]

use proptest::prop_assert;
use proptest::prop_assert_eq;
use proptest::proptest;

use crate::components::atom::Atom;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::merkle_tree::MerkleTree;
use crate::merkle_tree::MerkleTreeLeafData;
use crate::mode::Mode;
use crate::mode::Normal;
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

    let second_value_read = unsafe {
        let data = serialise(instance.second).unwrap();
        data.as_ptr().cast::<[u32; 4]>().read().map(u32::from_le)
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

        let merkle_tree = MerkleTree::from_foldable(&proof_atoms);
        merkle_tree.check_root_hash();
        match merkle_tree {
            MerkleTree::Leaf(MerkleTreeLeafData {
                hash,
                access_info,
                ..
            }) => {
                prop_assert_eq!(hash, initial_root_hash);
                prop_assert!(access_info);
            }
            _ => panic!("Expected Merkle tree to contain a single written leaf"),
        }
    });
}
