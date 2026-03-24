// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`DataSpace`]

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::num::NonZeroUsize;

use bincode::Encode;
use bincode::enc::Encoder;
use bincode::error::EncodeError;
use proptest::array;
use proptest::prop_assert;
use proptest::prop_assert_eq;
use proptest::proptest;

use super::PAGE_SIZE;
use crate::components::data_space::DataSpace;
use crate::components::data_space::NODE_ARITY;
use crate::foldable::Foldable;
use crate::foldable::Unfoldable;
use crate::foldable::tests::TestFolder;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::merkle_proof::proof_tree;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::merkle_proof::proof_tree::OwnedProofTree;
use crate::merkle_proof::proof_tree::ProofTree;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::assert_eq_found;
use crate::mode::utils::assert_not_found;
use crate::serialisation::deserialise;
use crate::serialisation::elem::Elem;
use crate::serialisation::serialise;

/// Dummy type that helps us implement custom serialisation via [`Elem`]
#[repr(C, packed)]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
struct Flipper {
    a: u8,
    b: u8,
}

impl Encode for Flipper {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.b.encode(encoder)?;
        self.a.encode(encoder)?;
        Ok(())
    }
}

impl Elem for Flipper {
    const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(2).unwrap();

    unsafe fn read_unaligned(source: *const u8) -> Self {
        unsafe {
            Self {
                a: source.add(1).read(),
                b: source.read(),
            }
        }
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        unsafe {
            dest.add(1).write(self.a);
            dest.write(self.b);
        }
    }
}

/// Test that writing out of bounds in data space panics
#[test]
#[should_panic]
fn out_of_bounds() {
    const LEN: usize = 4096;

    let mut state = DataSpace::<Normal>::new(LEN);

    // This should panic because we are trying to write an element at the address which
    // corresponds to the end of the buffer.
    unsafe {
        state.write(LEN * Flipper::STORED_SIZE.get(), Flipper { a: 1, b: 2 });
    }
}

/// Test that data space properly converts values to stored format
#[test]
fn stored_format() {
    // Writing to one item of the data space must convert to stored format.
    let mut data_space = DataSpace::<Normal>::new(4096);

    unsafe {
        data_space.write(0, Flipper { a: 13, b: 37 });
        assert_eq!(data_space.read::<Flipper>(0), Flipper { a: 13, b: 37 });
    }

    let buffer = unsafe { data_space.read::<[u8; 2]>(0) };
    assert_eq!(buffer, [37, 13]);

    // Writing to the entire data space must convert properly to stored format.
    data_space.write_all::<Flipper>(
        0,
        &[
            Flipper { a: 11, b: 22 },
            Flipper { a: 13, b: 24 },
            Flipper { a: 15, b: 26 },
            Flipper { a: 17, b: 28 },
        ],
    );

    let mut buff = [Flipper::default(); 4];
    data_space.read_all::<Flipper>(0, &mut buff);
    assert_eq!(
        buff,
        [
            Flipper { a: 11, b: 22 },
            Flipper { a: 13, b: 24 },
            Flipper { a: 15, b: 26 },
            Flipper { a: 17, b: 28 },
        ]
    );

    let buffer = unsafe { data_space.read::<[u8; 8]>(0) };
    assert_eq!(buffer, [22, 11, 24, 13, 26, 15, 28, 17]);
}

/// Ensure [`DataSpace`] can be serialised and deserialised in a consistent way.
#[test]
fn serialise_consistently() {
    proptest!(|(address in (0usize..120), value: u64)| {
        let mut space: DataSpace<Normal> = DataSpace::new(128);

        unsafe {
            space.write(address, value);
        }

        let bytes = serialise(&space).unwrap();

        let space_after: DataSpace<Normal> = deserialise(&bytes).unwrap();
        for i in 0..128 {
            unsafe {
                assert_eq!(space.read::<u8>(i), space_after.read::<u8>(i));
            }
        }

        let bytes_after = serialise(&space_after).unwrap();
        assert_eq!(bytes, bytes_after);

        // Serialisation is consistent with that of the `Prove` mode.
        let proof_space: DataSpace<Prove> = space.start_proof();
        let proof_bytes = serialise(&proof_space).unwrap();
        assert_eq!(bytes, proof_bytes);
    });
}

/// Test proof generation for data spaces with various access patterns
#[test]
fn generate_proof() {
    const MERKLE_LEAF_SIZE: usize = 4096;
    const LEAVES: usize = 8;
    const DATA_SPACE_SIZE: usize = MERKLE_LEAF_SIZE * LEAVES;
    const ELEM_SIZE: usize = u64::STORED_SIZE.get();

    if ELEM_SIZE > MERKLE_LEAF_SIZE {
        unreachable!("This test assumes that a single element does not span more than 2 leaves");
    }

    let address_range = 0..DATA_SPACE_SIZE - ELEM_SIZE;

    // Check that writing to an address in the proof data space makes subsequent reads return
    // the overwritten value.
    proptest!(|(
        byte_before: u8,
        bytes_after: [u8; ELEM_SIZE],
        write_address in &address_range,
    )| {
        let mut space = DataSpace::<Normal>::new(DATA_SPACE_SIZE);
        space.fill(byte_before);
        let mut proof_space: DataSpace<Prove> = space.start_proof();

        // Perform static memory accesses
        let value_before = u64::from_le_bytes([byte_before; ELEM_SIZE]);
        let value_after = u64::from_le_bytes(bytes_after);

        let value: u64 = unsafe { proof_space.read(write_address) };
        assert_eq!(value, value_before);
        unsafe { proof_space.write(write_address, value_after); }
        let value: u64 = unsafe { proof_space.read(write_address) };
        assert_eq!(value, value_after);

        let mut space = DataSpace::<Normal>::new(DATA_SPACE_SIZE);
        space.fill(byte_before);
        let mut proof_space: DataSpace<Prove> = space.start_proof();

        // Perform dynamic memory accesses as `u16`
        let value_before = [u16::from_le_bytes([byte_before; 2]); ELEM_SIZE / 2];
        let value_after = [
            u16::from_le_bytes([bytes_after[0], bytes_after[1]]),
            u16::from_le_bytes([bytes_after[2], bytes_after[3]]),
            u16::from_le_bytes([bytes_after[4], bytes_after[5]]),
            u16::from_le_bytes([bytes_after[6], bytes_after[7]]),
        ];

        let mut value = [0u16; ELEM_SIZE / 2];
        proof_space.read_all(write_address, &mut value);
        assert_eq!(value, value_before);
        proof_space.write_all(write_address, &value_after);
        proof_space.read_all(write_address, &mut value);
        assert_eq!(value, value_after);

        let mut space = DataSpace::<Normal>::new(DATA_SPACE_SIZE);
        space.fill(byte_before);
        let mut proof_space: DataSpace<Prove> = space.start_proof();

        // Perform dynamic memory accesses as bytes
        let value_before = [byte_before; ELEM_SIZE];

        let mut value = [0u8; ELEM_SIZE];
        proof_space.read_all(write_address, &mut value);
        assert_eq!(value, value_before);
        proof_space.write_all(write_address, &bytes_after);
        proof_space.read_all(write_address, &mut value);
        assert_eq!(value, bytes_after);
    });

    // Check correct Merkleisation of a data space which was read from and written to
    proptest!(|(
        byte_before: u8,
        bytes_after: [u8; ELEM_SIZE],
        reads in array::uniform2(&address_range),
        writes in array::uniform2(&address_range),
    )| {
        let mut space = DataSpace::<Normal>::new(DATA_SPACE_SIZE);
        space.fill(byte_before);
        let initial_root_hash = Hash::from_foldable(&space);

        let mut proof_space: DataSpace<Prove> = space.start_proof();

        // Perform memory accesses
        let value_before = [byte_before; ELEM_SIZE];
        reads.iter().try_for_each(|i| {
            let mut value = [0u8; ELEM_SIZE];
            proof_space.read_all(*i, &mut value);
            prop_assert_eq!(value, value_before);
            Ok::<(), proptest::test_runner::TestCaseError>(())
        })?;
        writes.iter().for_each(|i| {
            proof_space.write_all(*i, &bytes_after);
        });

        // Build the Merkle tree and check that it has the root hash of the
        // initial data space.
        let merkle_tree = MerkleProof::from_foldable(&proof_space);
        prop_assert_eq!(merkle_tree.root_hash(), initial_root_hash);

        // Compute expected access info for each leaf, assuming that an access
        // cannot span more than 2 leaves.
        let expected_leaves = |accesses: &[usize]| {
            let mut leaves = BTreeSet::<usize>::new();
            for i in accesses {
                leaves.insert(*i / MERKLE_LEAF_SIZE);
                leaves.insert((i + ELEM_SIZE - 1) / MERKLE_LEAF_SIZE);
            }
            leaves
        };
        let read_leaves = expected_leaves(&reads);
        let written_leaves = expected_leaves(&writes);

        // Traverse the generated Merkle tree and check each leaf's access log
        let mut queue = VecDeque::with_capacity(LEAVES + 1);

        let pages_tree = match merkle_tree {
            MerkleProof::Leaf(_) => panic!("Did not expect leaf"),
            MerkleProof::Node(mut node) => {
                // The node for the pages is the second child.
                node.children.remove(1)
            },
        };
        queue.push_back((0usize, 0usize, pages_tree));

        let max_depth = LEAVES.saturating_sub(1).checked_ilog(NODE_ARITY).unwrap_or(0) as usize + 1;
        while let Some((depth, pos, node)) = queue.pop_front() {
            match node {
                MerkleProof::Node(node) => queue.extend(
                    node.children
                        .into_iter()
                        .enumerate()
                        .map(|(id, child)| (depth + 1, pos * NODE_ARITY + id, child))
                ),
                MerkleProof::Leaf(leaf) => {
                    let width = NODE_ARITY.pow((max_depth - depth) as u32);
                    let start = pos * width;

                    // The tree might not be perfectly balanced, so the right-most leaf might have
                    // a smaller width. Capping the theoretical end of the leaf to the total number
                    // of leaves ensures we don't go out of bounds.
                    let end = LEAVES.min(start + width);

                    // Each leaf represents a contiguous range of pages in the data space. This is
                    // because as the Merkle proof tree gets compressed, multiple leaves might get
                    // compressed into one blinded leaf.
                    let range = start..end;

                    match leaf {
                        MerkleProofLeaf::Blind(_) => {
                            for idx in range {
                                prop_assert!(!read_leaves.contains(&idx) && !written_leaves.contains(&idx));
                            }
                        },
                        MerkleProofLeaf::Read(_) => {
                            for idx in range {
                                prop_assert!(read_leaves.contains(&idx) || written_leaves.contains(&idx));
                            }
                        },
                    }
                }
            }
        }
    });
}

/// Test the proof generation and verification for a computation against a data space.
///
/// # Safety
///
/// The `test_proof` and `test_verify` function must be the same function instantiated to
/// different modes.
///
/// Due to Rust's limitation on higher-ranked polymorphism, we can't accept
/// a single function and instantiate it within the function body with the respective modes
/// `Prove<_>` and `Verify`. One could work around this restriction by using a trait to
/// simulate the rank-2-ness, but that means you can't provide closures as the implementation
/// any more. If any of the given `test_proof` or `test_verify` capture an environment, this
/// would no longer work.
unsafe fn test_data_space_with_funs(
    len: usize,
    test_proof: impl FnOnce(&mut DataSpace<Prove>),
    test_verify: impl FnOnce(&mut DataSpace<Verify>),
) {
    let owned_cell = DataSpace::new(len);

    // We require the initial hash to ensure that the generated proof, but also the
    // instantiated state from the proof match the "before" state.
    let init_hash = Hash::from_foldable(&owned_cell);

    // The `ProofWrapper` transformer ensures the resulting data space is
    // setup for proof generation. You can think of this as starting the recording for a proof.
    let mut proof_cell = owned_cell.start_proof();

    test_proof(&mut proof_cell);

    // The post-hash is required to ensure that the verifier's final state matches the prover's
    // final state.
    let post_hash = Hash::from_foldable(&proof_cell);

    let proof_tree = MerkleProof::from_foldable(&proof_cell);
    assert_eq!(proof_tree.root_hash(), init_hash);

    // Instantiating the verifier state allows us to replay the computation and verify it does
    // the right things.
    let (mut verify_cell, out_proof) =
        proof_tree::deserialise::<DataSpace<Verify>>(ProofTree::Present(&proof_tree)).unwrap();

    let OwnedProofTree::Present(out_proof) = out_proof else {
        panic!("Expected present proof");
    };
    assert_eq!(&proof_tree, &out_proof);

    // The initial verifier state must match that of the initial state against which we
    // produced the proof.
    let verifier_init_hash = PartialHash::from_foldable(Some(out_proof.clone()), &verify_cell)
        .to_hash()
        .unwrap();
    assert_eq!(verifier_init_hash, init_hash);

    test_verify(&mut verify_cell);

    // Once we're doing replaying the computation on the verifier side, the final state must
    // match that of the prover's. If not, that means we produced a proof that results in a
    // transition that we did not intend to prove.
    let verifier_post_hash = PartialHash::from_foldable(Some(out_proof), &verify_cell)
        .to_hash()
        .unwrap();
    assert_eq!(verifier_post_hash, post_hash);
}

/// Generate a test for data spaces using a given size and closure which operates on the
/// [`DataSpace`]. This effectively demonstrates that the actions performed by the given closure
/// can be proven and verified correctly.
macro_rules! test_data_space_with {
    ($len:literal, | $param:ident | { $($body:tt)* }) => {
        {
            let test_proof = |$param: &mut DataSpace<Prove>| {
                $($body)*
            };

            let test_verify = |$param: &mut DataSpace<Verify>| {
                $($body)*
            };

            // SAFETY: This function is intended to be used only in this macro.
            unsafe {
                test_data_space_with_funs($len, test_proof, test_verify);
            }
        }
    };
}

#[test]
fn test_data_space_proofs_not_power_of_arity() {
    proptest!(|(addr in 0..28664usize, val: u64)| {
        // We use 28672 because it is 7 * 4096 (a data space with 7 pages).
        test_data_space_with!(28672, |cell| {
            unsafe {
                cell.write::<u64>(addr, val);
            }
        });
    });
}

#[test]
fn test_data_space_proofs_nothing() {
    test_data_space_with!(65536, |_cell| {});
}

#[test]
fn test_data_space_proofs_read() {
    proptest!(|(addr in 0..65528usize)| {
        test_data_space_with!(65536, |cell| {
            unsafe {
                cell.read::<u64>(addr);
            }
        });
    });
}

#[test]
fn test_data_space_proofs_write() {
    proptest!(|(addr in 0..65528usize, val: u64)| {
        test_data_space_with!(65536, |cell| {
            unsafe {
                cell.write::<u64>(addr, val);
            }
        });
    });
}

#[test]
fn test_data_space_proofs_len() {
    test_data_space_with!(65536, |cell| {
        cell.len();
    });
}

#[test]
fn test_data_space_proofs_read_and_len() {
    proptest!(|(addr in 0..65528usize)| {
        test_data_space_with!(65536, |cell| {
            unsafe {
                cell.read::<u64>(addr);
            }

            cell.len();
        });
    });
}

#[test]
fn test_data_space_proofs_write_and_len() {
    proptest!(|(addr in 0..65528usize, val: u64)| {
        test_data_space_with!(65536, |cell| {
            unsafe {
                cell.write::<u64>(addr, val);
            }

            cell.len();
        });
    });
}

#[test]
fn test_data_space_proofs_read_and_write() {
    proptest!(|(addr in 0..65528usize, val: u64)| {
        test_data_space_with!(65536, |cell| {
            unsafe {
                let x = cell.read::<u64>(addr);
                cell.write(addr, x.wrapping_add(val));
            }
        });
    });
}

#[test]
fn test_data_space_proofs_read_and_write_and_len() {
    proptest!(|(addr in 0..65528usize, val: u64)| {
        test_data_space_with!(65536, |cell| {
            unsafe {
                let x = cell.read::<u64>(addr);
                cell.write(addr, x.wrapping_add(val));
            }

            cell.len();
        });
    });
}

/// Check the read functionality of a data space that has no gaps between its pages.
#[test]
fn verify_no_gaps() {
    let mut dyn_cells = DataSpace::absent(3 * PAGE_SIZE);
    dyn_cells.populate_pages_with_bytes(
        0,
        [1u8, 3, 3, 7]
            .into_iter()
            .cycle()
            .take(PAGE_SIZE)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    dyn_cells.populate_pages_with_bytes(
        PAGE_SIZE,
        [11u8, 14, 14, 15]
            .into_iter()
            .cycle()
            .take(PAGE_SIZE)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    // Read things that are contained in the first leaf.
    unsafe {
        assert_eq_found!(dyn_cells.read::<[u8; 4]>(0), [1, 3, 3, 7]);
        assert_eq_found!(dyn_cells.read::<[u8; 4]>(1), [3, 3, 7, 1]);
        assert_eq_found!(dyn_cells.read::<[u8; 4]>(PAGE_SIZE - 4), [1, 3, 3, 7]);
    }

    // Read things that span the first and second leaf.
    unsafe {
        assert_eq_found!(dyn_cells.read::<[u8; 4]>(PAGE_SIZE - 2), [3, 7, 11, 14]);
    }

    // Read things that are contained in the second leaf.
    unsafe {
        assert_eq_found!(dyn_cells.read::<[u8; 4]>(PAGE_SIZE), [11, 14, 14, 15]);
        assert_eq_found!(dyn_cells.read::<[u8; 4]>(PAGE_SIZE + 1), [14, 14, 15, 11]);
    }

    // Read more than is available.
    unsafe {
        assert_not_found!(dyn_cells.read::<[u8; PAGE_SIZE * 3 + 1]>(0));
    }

    // Read at an offset that is out of bounds.
    unsafe {
        assert_not_found!(dyn_cells.read::<u8>(PAGE_SIZE * 2));
    }

    // Add more to the third leaf.
    unsafe {
        assert_not_found!(dyn_cells.clone().write(PAGE_SIZE * 2, [255u8, 0]));
    }

    unsafe {
        assert_not_found!(dyn_cells.read::<[u8; 4]>(PAGE_SIZE * 2));
        assert_not_found!(dyn_cells.read::<[u8; 2]>(PAGE_SIZE * 2 + 2));
    }

    // Read at an offset that is out of bounds.
    unsafe {
        assert_not_found!(dyn_cells.read::<u8>(PAGE_SIZE * 3));
    }
}

/// Check the functionality of a data space that has gaps between its pages.
#[test]
fn verify_with_gaps() {
    let mut dyn_cells = DataSpace::absent(3 * PAGE_SIZE);
    dyn_cells.populate_pages_with_bytes(
        0,
        [7u8, 3, 3]
            .into_iter()
            .cycle()
            .take(PAGE_SIZE)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    dyn_cells.populate_pages_with_bytes(
        PAGE_SIZE * 2,
        [42u8, 41]
            .into_iter()
            .cycle()
            .take(PAGE_SIZE)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    unsafe {
        assert_eq_found!(dyn_cells.read::<[u8; 3]>(0), [7, 3, 3]);
        assert_eq_found!(dyn_cells.read::<[u8; 2]>(1), [3, 3]);
        assert_eq_found!(dyn_cells.read::<[u8; 1]>(PAGE_SIZE * 2), [42]);
        assert_eq_found!(dyn_cells.read::<[u8; 1]>(PAGE_SIZE * 2 + 1), [41]);
    }

    // Read a range that covers a gap.
    unsafe {
        assert_not_found!(dyn_cells.read::<[u8; PAGE_SIZE + 4]>(PAGE_SIZE - 2));
        assert_not_found!(dyn_cells.read::<[u8; PAGE_SIZE]>(PAGE_SIZE));
    }

    // Write within the gap. That should fail as these kinds of writes need to read first.
    unsafe {
        assert_not_found!(dyn_cells.clone().write(PAGE_SIZE - 1, [1u8, 1, 3]));
    }

    unsafe {
        assert_not_found!(dyn_cells.read::<[u8; 6]>(PAGE_SIZE - 1));
        assert_not_found!(dyn_cells.read::<[u8; 4]>(PAGE_SIZE));
    }
}

#[test]
fn fold_unfold() {
    proptest::proptest!(|(pages in 1usize..=17)| {
        let length = pages * PAGE_SIZE;
        let mut space: DataSpace<Normal> = DataSpace::new(length);
        space.fill(42);

        let tree = space.fold(TestFolder);
        let unfolded = DataSpace::unfold(tree).unwrap();

        assert!(space == unfolded);
    });
}

#[test]
fn unfold_error_on_invalid_length() {
    let length = 5000;
    let mut space: DataSpace<Normal> = DataSpace::new(length);
    space.fill(42);

    let tree = space.fold(TestFolder);
    let unfold_result = DataSpace::unfold(tree.clone());

    assert!(unfold_result == Err("InvalidLength(5000)".to_string()));
}
