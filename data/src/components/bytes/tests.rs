// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Tests for [`Bytes`]

use std::panic::AssertUnwindSafe;

use proptest::collection::vec;
use proptest::prop_assert;
use proptest::prop_assert_eq;
use proptest::proptest;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

use super::test_utils::BytesOp;
use super::test_utils::MAX_PROOF_OFFSETS;
use crate::codec::Bincode;
use crate::codec::LeafEncode;
use crate::components::bytes::Bytes;
use crate::components::bytes::NODE_ARITY;
use crate::components::bytes::PAGE_SIZE;
use crate::components::bytes::test_utils::BytesMutOp;
use crate::components::bytes::test_utils::MAX_PROOF_LENGTH;
use crate::components::bytes::test_utils::MAX_WRITE_PROOF_LENGTH;
use crate::components::bytes::test_utils::NDS_BYTES_LENGTH;
use crate::foldable::Foldable;
use crate::foldable::Unfoldable;
use crate::foldable::seq_tree::tree_depth;
use crate::foldable::tests::TestFolder;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::merkle_proof::proof_binary;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::assert_eq_found;
use crate::mode::utils::assert_not_found;
use crate::mode::utils::catch_not_found;
use crate::mode_test;
use crate::serialisation::serialise;

// Bytes should be empty after creation
mode_test!(new_is_empty, F, {
    let bytes = Bytes::<F>::default();
    assert_eq!(bytes.len(), 0);
    assert!(bytes.is_empty());
});

// Reading from empty bytes always returns 0, regardless of offset or buffer size
mode_test!(read_from_empty_bytes_returns_zero, F, {
    proptest!(|(offset: usize, buffer_size in 0usize..50)| {
        let bytes = Bytes::<F>::default();
        let mut buffer = vec![0u8; buffer_size];
        prop_assert_eq!(bytes.read(offset, &mut buffer), 0);
    });
});

// Reading at offset >= len always returns 0
mode_test!(read_past_end_returns_zero, F, {
    proptest!(|(size in 0usize..100, extra: usize, buffer_size in 0usize..50)| {
        // Out of bounds offset
        let offset = size.saturating_add(extra);

        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);

        let mut buffer = vec![0u8; buffer_size];
        prop_assert_eq!(bytes.read(offset, &mut buffer), 0);
    });
});

// Read doesn't return more bytes than are available
mode_test!(read_returns_correct_byte_count, F, {
    proptest!(|(size in 0usize..100, offset: usize, buffer_size in 0usize..50)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);

        let mut buffer = vec![0u8; buffer_size];
        let read = bytes.read(offset, &mut buffer);
        let expected = size.saturating_sub(offset).min(buffer_size);
        prop_assert_eq!(read, expected);
    });
});

// Writing to empty bytes always returns 0, regardless of offset or data
mode_test!(write_to_empty_bytes_returns_zero, F, {
    proptest!(|(offset: usize, data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();
        prop_assert_eq!(bytes.write(offset, &data), 0);
        prop_assert!(bytes.is_empty());
        prop_assert_eq!(bytes.read(0, &mut vec![0u8; data.len()]), 0);
    });
});

// Writing at offset >= len always returns 0
mode_test!(write_past_end_returns_zero, F, {
    proptest!(|(size in 0usize..100, extra: usize, data: Vec<u8>)| {
        let offset = size.saturating_add(extra);
        let mut bytes = Bytes::<F>::default();

        bytes.resize(size);

        prop_assert_eq!(bytes.write(offset, &data), 0);
        prop_assert_eq!(bytes.read(offset, &mut vec![0u8; data.len()]), 0);

        let mut buffer = vec![0u8; size];
        prop_assert_eq!(bytes.read(0, &mut buffer), size);
        prop_assert!(buffer.iter().all(|&b| b == 0));
    });
});

// Write doesn't update more than there are bytes to be updated
mode_test!(write_returns_correct_byte_count, F, {
    proptest!(|(size in 0usize..100, offset in 0usize..200, data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);

        let written = bytes.write(offset, &data);
        let expected = data.len().min(size.saturating_sub(offset));
        prop_assert_eq!(written, expected);
    });
});

// Write followed by read at same offset returns the written data
mode_test!(write_read_roundtrip, F, {
    proptest!(|(size in 0usize..100, start in 0usize..100, data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);

        let written = bytes.write(start, &data);

        // Write is truncated to available space
        let expected_written = data.len().min(size.saturating_sub(start));
        prop_assert_eq!(written, expected_written);

        // Reading back should return exactly what was written
        let mut buffer = vec![0u8; expected_written];
        let read = bytes.read(start, &mut buffer);
        prop_assert_eq!(read, expected_written);
        prop_assert_eq!(&buffer, &data[..expected_written]);
    });
});

// Multiple overlapping writes result in last-write-wins semantics
mode_test!(overlapping_writes_last_write_wins, F, {
    // Sequence of (start, data) writes
    let writes_strat = vec((0usize..100, vec(0u8..=255, 0..10)), 0..20);

    proptest!(|(size in 1usize..50, writes in writes_strat)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);

        // Track expected state by applying same writes to a reference buffer
        let mut expected = vec![0u8; size];

        for (start, data) in writes {
            let start = start % size;
            let written = bytes.write(start, &data);

            // Mirror the write to our reference buffer
            let write_len = data.len().min(size - start);
            expected[start..][..write_len].copy_from_slice(&data[..write_len]);
            prop_assert_eq!(written, write_len);
        }

        // Final state should match reference buffer exactly
        let mut buffer = vec![0u8; size];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, size);
        prop_assert_eq!(buffer, expected);
    });
});

// Resize always results in len() == new_size
mode_test!(resize_sets_correct_length, F, {
    proptest!(|(initial_size in 0usize..100, new_size in 0usize..100)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(initial_size);
        prop_assert_eq!(bytes.len(), initial_size);
        prop_assert_eq!(bytes.is_empty(), initial_size == 0);

        bytes.resize(new_size);
        prop_assert_eq!(bytes.len(), new_size);
        prop_assert_eq!(bytes.is_empty(), new_size == 0);
    });
});

// Resize up fills new space with zeros
mode_test!(resize_up_fills_with_zeros, F, {
    proptest!(|(initial_size in 0usize..50, grow_by in 1usize..50)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(initial_size);

        // Initial resize should yield zeros
        let mut buffer = vec![0u8; initial_size];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, initial_size);
        prop_assert!(buffer.iter().all(|&b| b == 0));

        bytes.resize(initial_size + grow_by);

        // Another resize should yield zeros in the new region
        let mut buffer = vec![255u8; grow_by];
        let read = bytes.read(initial_size, &mut buffer);
        prop_assert_eq!(read, grow_by);
        prop_assert!(buffer.iter().all(|&b| b == 0));
    });
});

// Resize down preserves prefix data
mode_test!(resize_down_preserves_prefix, F, {
    proptest!(|(initial_size in 10usize..50, shrink_to in 0usize..10, data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(initial_size);

        let written = bytes.write(0, &data);

        bytes.resize(shrink_to);
        let preserved_len = written.min(shrink_to);

        let mut buffer = vec![0u8; preserved_len];
        let read = bytes.read(0, &mut buffer);

        prop_assert_eq!(read, preserved_len);
        prop_assert_eq!(buffer, &data[..preserved_len]);
    });
});

// Resize preserves written data when growing
mode_test!(resize_up_preserves_written_data, F, {
    proptest!(|(
        initial_size in 10usize..50,
        data: Vec<u8>,
        grow_by in 1usize..20
    )| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(initial_size);

        let written = bytes.write(0, &data);
        prop_assert_eq!(written, data.len().min(initial_size));

        // Growing should not affect existing data
        bytes.resize(initial_size + grow_by);

        let mut buffer = vec![0u8; written];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, written);
        prop_assert_eq!(buffer, &data[..written]);
    });
});

// Resize down then up clears truncated region (fills with zeros)
mode_test!(resize_down_then_up_clears_truncated_region, F, {
    proptest!(|(
        initial_size in 10usize..50,
        shrink_to in 1usize..10,
        grow_extra in 1usize..20,
        data: Vec<u8>,
    )| {
        let grow_to = shrink_to + grow_extra;

        let mut bytes = Bytes::<F>::default();
        bytes.resize(initial_size);
        bytes.write(0, &data);

        // Shrink discards data beyond shrink_to
        bytes.resize(shrink_to);

        // Grow re-adds space, but old data is gone
        bytes.resize(grow_to);

        // Re-grown region must be zeros, not old data
        let check_len = grow_to - shrink_to;
        let mut buffer = vec![255u8; check_len];
        let read = bytes.read(shrink_to, &mut buffer);
        prop_assert_eq!(read, check_len);
        prop_assert!(buffer.iter().all(|&b| b == 0));

        // Prefix that was never truncated remains intact
        let preserved_len = shrink_to.min(data.len()).min(initial_size);
        let mut buffer = vec![0u8; preserved_len];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, preserved_len);
        prop_assert_eq!(&buffer, &data[..preserved_len]);
    });
});

// Resize to same size is idempotent (no-op)
mode_test!(resize_to_same_size_is_idempotent, F, {
    proptest!(|(size in 0usize..50, data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);
        bytes.write(0, &data);

        let mut before = vec![0u8; size];
        bytes.read(0, &mut before);

        // Resizing to current size should be a no-op
        bytes.resize(size);

        prop_assert_eq!(bytes.len(), size);
        let mut after = vec![0u8; size];
        bytes.read(0, &mut after);
        prop_assert_eq!(before, after);
    });
});

// Zero-length operations are always safe and return 0
mode_test!(zero_length_operations, F, {
    proptest!(|(size in 0usize..50, offset: usize)| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);

        // Empty writes and reads should succeed regardless of offset validity
        let empty: [u8; 0] = [];
        prop_assert_eq!(bytes.write(offset, &empty), 0);

        let mut empty_buf: [u8; 0] = [];
        prop_assert_eq!(bytes.read(offset, &mut empty_buf), 0);
    });
});

// Resize to zero clears all data
mode_test!(resize_to_zero_clears_data, F, {
    proptest!(|(size in 0usize..50, data in vec(0u8..=255, 1..20))| {
        let mut bytes = Bytes::<F>::default();
        bytes.resize(size);
        let written_len = bytes.write(0, &data);

        // Verify data was written
        let mut buffer = vec![0u8; written_len];
        prop_assert_eq!(bytes.read(0, &mut buffer), buffer.len());
        prop_assert_eq!(&buffer, &data[..written_len]);

        // Resize to zero
        bytes.resize(0);
        prop_assert_eq!(bytes.len(), 0);
        prop_assert!(bytes.is_empty());

        // Reading from empty bytes returns 0
        let mut buffer = [0u8; 10];
        prop_assert_eq!(bytes.read(0, &mut buffer), 0);

        // Re-growing should yield zeros, not old data
        bytes.resize(size);
        let mut buffer = vec![0u8; size];
        bytes.read(0, &mut buffer);
        prop_assert!(buffer.iter().all(|&b| b == 0));
    });
});

// Set overwrites entire contents with new data
mode_test!(set_overwrites_contents, F, {
    proptest!(|(initial_data: Vec<u8>, new_data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();

        // Set initial data
        bytes.set(&initial_data);
        prop_assert_eq!(bytes.len(), initial_data.len());

        let mut buffer = vec![0u8; initial_data.len()];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, initial_data.len());
        prop_assert_eq!(&buffer, &initial_data);

        // Set new data (potentially different size)
        bytes.set(&new_data);
        prop_assert_eq!(bytes.len(), new_data.len());

        let mut buffer = vec![0u8; new_data.len()];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, new_data.len());
        prop_assert_eq!(&buffer, &new_data);
    });
});

// Append adds data to the end of the byte array
mode_test!(append_adds_to_end, F, {
    proptest!(|(initial_data: Vec<u8>, append_data: Vec<u8>)| {
        let mut bytes = Bytes::<F>::default();
        bytes.set(&initial_data);

        let appended = bytes.append(&append_data);
        prop_assert_eq!(appended, append_data.len());
        prop_assert_eq!(bytes.len(), initial_data.len() + append_data.len());

        // Verify initial data is preserved
        let mut buffer = vec![0u8; initial_data.len()];
        let read = bytes.read(0, &mut buffer);
        prop_assert_eq!(read, initial_data.len());
        prop_assert_eq!(&buffer, &initial_data);

        // Verify appended data is at the end
        let mut buffer = vec![0u8; append_data.len()];
        let read = bytes.read(initial_data.len(), &mut buffer);
        prop_assert_eq!(read, append_data.len());
        prop_assert_eq!(&buffer, &append_data);
    });
});

// Bytes behaves the same across different modes
#[test]
fn bytes_are_same_across_modes() {
    proptest!(|(ops in vec(BytesMutOp::any(64 * PAGE_SIZE), 1..20))| {
        let mut bytes_normal = Bytes::<Normal>::default();
        let results_normal = ops.iter().map(|op| op.run(&mut bytes_normal)).collect::<Vec<_>>();
        let hash_normal = Hash::from_foldable(&bytes_normal);

        let mut bytes_prove = Bytes::<Prove>::default();
        let results_prove = ops.iter().map(|op| op.run(&mut bytes_prove)).collect::<Vec<_>>();
        prop_assert_eq!(&results_normal, &results_prove);

        let hash_prove = Hash::from_foldable(&bytes_prove);
        prop_assert_eq!(hash_normal, hash_prove);

        let merkle_proof = MerkleProof::from_foldable(&bytes_prove);

        let mut bytes_verify = Bytes::<Verify>::default();
        let results_verify = ops.iter().map(|op| op.run(&mut bytes_verify)).collect::<Vec<_>>();
        prop_assert_eq!(results_normal, results_verify);

        let hash_verify =
            PartialHash::from_foldable(Some(merkle_proof), &bytes_verify)
                .to_hash()
                .unwrap();
        prop_assert_eq!(hash_normal, hash_verify);
    });
}

#[test]
fn proof_round_trip() {
    proptest!(|(ops in vec(BytesMutOp::any(64 * PAGE_SIZE), 1..20))| {
        let mut bytes_normal = Bytes::<Normal>::default();

        for op in ops {
            let mut bytes_prove = bytes_normal.start_proof();

            // The initial hash of the Prove mode should match the Normal mode hash.
            let init_normal_hash = Hash::from_foldable(&bytes_normal);
            let init_prove_hash = Hash::from_foldable(&bytes_prove);
            prop_assert_eq!(init_normal_hash, init_prove_hash);

            // Run the operation which we would like to prove.
            let prove_result = op.run(&mut bytes_prove);

            // The post-operation hash is later used to compare against the Normal mode hash.
            let after_proof_hash = Hash::from_foldable(&bytes_prove);

            // The Merkle proof tree should match the state hash before the operation was applied.
            let proof_tree = MerkleProof::from_foldable(&bytes_prove);
            prop_assert_eq!(init_prove_hash, proof_tree.root_hash());

            // We want to serialise the proof to its binary format to make this test more realistic.
            let proof_bytes = serialise(proof_tree).unwrap();

            // Parsing the proof so we can see if the proof generation worked.
            let (mut bytes_verify, parsed_proof_tree) = proof_binary::deserialise(&proof_bytes).unwrap();
            let parsed_proof_tree = parsed_proof_tree.into_present();

            // The parsed state should have a state hash equal to that of the initial Normal/Prove state
            let init_verify_hash = PartialHash::from_foldable(parsed_proof_tree.clone(), &bytes_verify).to_hash().unwrap();
            prop_assert_eq!(init_verify_hash, init_prove_hash);

            // Run the operation which we would like to verify.
            let verify_result = op.run(&mut bytes_verify);
            prop_assert_eq!(&verify_result, &prove_result);

            // The post-operation hash should match the Normal mode hash.
            let after_verify_hash = PartialHash::from_foldable(parsed_proof_tree, &bytes_verify).to_hash().unwrap();
            prop_assert_eq!(after_verify_hash, after_proof_hash);

            // Finally advance the Normal mode state as well
            let normal_result = op.run(&mut bytes_normal);
            prop_assert_eq!(&normal_result, &verify_result);

            // The Normal mode hash should match the post-operation hash that we proved and verified.
            let after_normal_hash = Hash::from_foldable(&bytes_normal);
            prop_assert_eq!(after_normal_hash, after_verify_hash);
        }
    });
}

// `partial_slice` of an empty range, even when the underlying region is undefined.
#[test]
fn partial_slice_empty_range() {
    let bytes = Bytes::<Verify>::absent(10);
    assert_eq_found!(bytes.partial_slice(0..0), [].as_slice());
    assert_eq_found!(bytes.partial_slice(5..5), [].as_slice());
}

// `partial_slice` at the exact boundaries of a defined entry within a partial state.
#[test]
fn partial_slice_entry_boundaries() {
    let mut bytes = Bytes::<Verify>::absent(20);
    bytes.write(5, &[1, 2, 3, 4, 5]);

    // Exact match for the entry's range.
    assert_eq_found!(bytes.partial_slice(5..10), [1, 2, 3, 4, 5].as_slice());
    // Sub-ranges fully contained inside the entry.
    assert_eq_found!(bytes.partial_slice(6..9), [2, 3, 4].as_slice());
    assert_eq_found!(bytes.partial_slice(5..6), [1].as_slice());
    assert_eq_found!(bytes.partial_slice(9..10), [5].as_slice());
}

// `partial_slice` when the range is fully contained within a single defined entry.
#[test]
fn partial_slice_fully_defined() {
    // `Bytes::<Verify>::new` lays out the whole length as a single contiguous defined entry.
    let mut bytes = Bytes::<Verify>::new(20);
    bytes.write(
        0,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11,
        ],
    );

    assert_eq_found!(
        bytes.partial_slice(0..20),
        [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11
        ]
        .as_slice()
    );
    assert_eq_found!(bytes.partial_slice(5..10), [60, 70, 80, 90, 100].as_slice());
    assert_eq_found!(bytes.partial_slice(0..1), [10].as_slice());
    assert_eq_found!(bytes.partial_slice(19..20), [11].as_slice());
}

// `partial_slice` when the requested range extends past the byte array's length.
#[test]
fn partial_slice_range_exceeds_length() {
    let bytes = Bytes::<Verify>::new(10);
    assert_not_found!(bytes.partial_slice(0..11));
    assert_not_found!(bytes.partial_slice(5..15));
    assert_not_found!(bytes.partial_slice(10..11));
    // Empty ranges always succeed, even when out-of-bounds: this matches `slice[len..len]` in std
    // and lets callers that already validated their bounds (e.g. via `min(len, ..)`) ask for a
    // zero-length read at the end without tripping `not_found`.
    assert_eq_found!(bytes.partial_slice(11..11), [].as_slice());
}

// `partial_slice` when the range crosses a gap between two separately defined entries.
#[test]
fn partial_slice_range_spans_gap() {
    let mut bytes = Bytes::<Verify>::absent(20);
    // Define [0, 3) and [10, 13), leaving an undefined gap at [3, 10).
    bytes.write(0, &[1, 2, 3]);
    bytes.write(10, &[4, 5, 6]);

    assert_not_found!(bytes.partial_slice(2..11));
    assert_not_found!(bytes.partial_slice(0..13));
    assert_not_found!(bytes.partial_slice(3..10));
}

// `partial_slice` when the range straddles a defined/undefined boundary.
#[test]
fn partial_slice_range_straddles_boundary() {
    let mut bytes = Bytes::<Verify>::absent(20);
    bytes.write(5, &[1, 2, 3, 4, 5]);

    // Starts in undefined, ends in defined.
    assert_not_found!(bytes.partial_slice(3..7));
    // Starts in defined, ends in undefined.
    assert_not_found!(bytes.partial_slice(7..12));
    // Starts before and ends after the defined region.
    assert_not_found!(bytes.partial_slice(0..15));
}

// `partial_slice` when no part of the requested range is defined.
#[test]
fn partial_slice_range_undefined() {
    // `absent` leaves the underlying `PartialVec` empty, so any non-empty range is undefined.
    let bytes = Bytes::<Verify>::absent(10);
    assert_not_found!(bytes.partial_slice(0..1));
    assert_not_found!(bytes.partial_slice(0..10));
    assert_not_found!(bytes.partial_slice(3..7));
}

#[test]
fn fold_unfold() {
    // Test with lengths of up to around 25 pages, including non-multiples of page size.
    proptest::proptest!(|(length in 0usize..=100000)| {
        let v = [4u8, 0, 89, 251, 3].iter().copied().cycle().take(length).collect::<Vec<_>>();
        let bytes: Bytes<Normal> = Bytes::from(&v[..]);

        let tree = bytes.fold(TestFolder);
        let unfolded = Bytes::unfold(tree).unwrap();

        assert!(bytes == unfolded);
    });
}

/// At each of the `MAX_PROOF_OFFSETS` a read or write of `MAX_FILE_CHUNK_SIZE` results in a
/// maximally long proof for its kind. This returns all four such operations, each paired with the
/// length its proof is expected to have.
fn max_proof_ops() -> Vec<(BytesMutOp, usize)> {
    let mut v = vec![];

    for offset in MAX_PROOF_OFFSETS {
        v.push((
            BytesMutOp::Immutable {
                op: BytesOp::Read {
                    offset,
                    size: MAX_FILE_CHUNK_SIZE,
                },
            },
            MAX_PROOF_LENGTH,
        ));
        v.push((
            BytesMutOp::Write {
                offset,
                data: vec![0; MAX_FILE_CHUNK_SIZE],
            },
            MAX_WRITE_PROOF_LENGTH,
        ));
    }

    v
}

#[test]
fn test_bytes_largest_valid_proof_nds() {
    let v = vec![0; 1024 * 1024 * 64];
    let bytes: Bytes<Normal> = Bytes::from(&v[..]);

    for (op, expected) in max_proof_ops() {
        let mut bytes_prove = bytes.start_proof();
        let _result = op.run(&mut bytes_prove);
        let proof_tree = MerkleProof::from_foldable(&bytes_prove);
        let proof = serialise(proof_tree).unwrap();

        assert_eq!(
            proof.len(),
            expected,
            "expect maximum proof size to be {expected} for {op:?}, but got {}",
            proof.len()
        );
    }

    proptest::proptest!(|(op in BytesMutOp::any(NDS_BYTES_LENGTH))| {
        let mut bytes_prove = bytes.start_proof();
        let _result = op.run(&mut bytes_prove);
        let proof_tree = MerkleProof::from_foldable(&bytes_prove);
        let proof = serialise(proof_tree).unwrap();

        assert!(proof.len() <= MAX_PROOF_LENGTH);
    });
}

/// Run `ops` in `Prove` mode against `normal`, round trip the proof through its binary form, replay
/// the same operations in `Verify` mode and check that the state hash they arrive at is the one
/// `Prove` mode arrived at. Returns the serialised proof, so callers can assert on its size.
fn proof_for(normal: &Bytes<Normal>, ops: &[BytesMutOp]) -> Vec<u8> {
    let mut prove = normal.start_proof();
    for op in ops {
        op.run(&mut prove);
    }
    let after_prove = Hash::from_foldable(&prove);

    let proof_tree = MerkleProof::from_foldable(&prove);
    assert_eq!(
        Hash::from_foldable(normal),
        proof_tree.root_hash(),
        "the proof must describe the state before the operations"
    );

    let proof = serialise(proof_tree).unwrap();
    let (verify, parsed_proof_tree): (Bytes<Verify>, _) =
        proof_binary::deserialise(&proof).unwrap();
    let parsed_proof_tree = parsed_proof_tree.into_present();

    let verify = catch_not_found(AssertUnwindSafe(|| {
        let mut verify = verify;
        for op in ops {
            op.run(&mut verify);
        }
        verify
    }))
    .expect("the proof must hold everything the operations access");

    let after_verify = PartialHash::from_foldable(parsed_proof_tree, &verify)
        .to_hash()
        .expect("Verify mode must be able to recompute the state hash");
    assert_eq!(
        after_prove, after_verify,
        "Prove and Verify mode must agree on the state after the operations"
    );

    proof
}

// A `set` replaces every page, so the proof holds none of the previous value: its size does not
// depend on how large that value was.
#[test]
fn set_proof_omits_previous_pages() {
    let set = [BytesMutOp::Set {
        data: (0..PAGE_SIZE + 7).map(|i| i as u8).collect(),
    }];

    let small = Bytes::<Normal>::from(&vec![0xffu8; 4 * PAGE_SIZE][..]);
    let large = Bytes::<Normal>::from(&vec![0xffu8; NDS_BYTES_LENGTH][..]);

    let small_proof = proof_for(&small, &set);
    let large_proof = proof_for(&large, &set);

    assert_eq!(
        small_proof.len(),
        large_proof.len(),
        "a set proof must not depend on the size of the value it replaces"
    );
    assert!(
        small_proof.len() < 100,
        "a set proof holds the length leaf and one blinded page tree, but was {} bytes",
        small_proof.len()
    );
}

// A page read before a `set` is still needed in full: `Verify` mode replays that read and has no
// other source for the bytes it returns.
#[test]
fn set_proof_keeps_read_pages() {
    let normal = Bytes::<Normal>::from(&vec![0xffu8; 4 * PAGE_SIZE][..]);
    let set = BytesMutOp::Set {
        data: vec![7u8; PAGE_SIZE + 7],
    };
    let read = BytesMutOp::Immutable {
        op: BytesOp::Read {
            offset: 3 * PAGE_SIZE,
            size: 8,
        },
    };

    // `proof_for` replays the read in Verify mode, so it fails if the page was dropped.
    let with_read = proof_for(&normal, &[read, set.clone()]);
    let without_read = proof_for(&normal, &[set]);

    assert!(
        with_read.len() > without_read.len() + PAGE_SIZE,
        "the page that was read must still be in the proof in full"
    );
}

// A page a write covers in full is blinded rather than carried; a partially written page is not.
#[test]
fn write_covering_page_is_blinded() {
    let normal = Bytes::<Normal>::from(&vec![0xffu8; 4 * PAGE_SIZE][..]);

    let whole_page = proof_for(
        &normal,
        &[BytesMutOp::Write {
            offset: PAGE_SIZE,
            data: vec![1u8; PAGE_SIZE],
        }],
    );
    let half_page = proof_for(
        &normal,
        &[BytesMutOp::Write {
            offset: PAGE_SIZE,
            data: vec![1u8; PAGE_SIZE / 2],
        }],
    );

    assert!(
        whole_page.len() < PAGE_SIZE,
        "a fully covered page must be blinded, but the proof was {} bytes",
        whole_page.len()
    );
    assert!(
        half_page.len() > PAGE_SIZE,
        "a partially written page must be carried in full, but the proof was {} bytes",
        half_page.len()
    );
}

/// Folding a verify-mode value must cost what the proof contains, not what it claims to describe.
///
/// The length reaching the fold is recovered from the proof, and nothing bounds what it may assert:
/// a proof describing a value of any size can still hash correctly. Nothing has been read or
/// written here, so every page below the items node stands unchanged and that node contributes
/// exactly the hash the proof already carries for it. Walking the pages to discover as much would
/// take time proportional to the claimed length, which at this size is not a wait anyone will sit
/// through.
#[test]
fn verify_fold_skips_pages_holding_no_state() {
    // Four tebibytes, or a little over four billion pages.
    let length = 1usize << 42;

    let length_leaf = MerkleProof::leaf_read(
        LeafEncode::<Bincode>::leaf_encode(&(length as u64)).expect("Encoding length should work"),
    );
    let pages = MerkleProof::leaf_blind(Hash::hash_bytes(b"untouched pages"));
    let proof = MerkleProof::node_without_data(vec![length_leaf, pages]);

    let bytes = Bytes::<Verify>::absent(length);

    assert_eq!(
        PartialHash::from_foldable(Some(proof.clone()), &bytes),
        PartialHash::Present(proof.root_hash()),
        "An untouched value should re-hash to exactly what the proof committed to"
    );
}

/// A page the step wrote to must be descended into and re-hashed, not folded away.
///
/// The proof spells out the path down to page 0 and blinds every sibling along the way, which is
/// what a prover emits for a step touching one page of a large value. The whole page is written,
/// since leaving it partly defined would be rejected as incoherent before any of this was reached.
///
/// Note what this does and does not pin down. Every node on the path is present in the proof, so
/// `skip_unchanged_subtree` declines on those regardless of what `has_state` reports - the
/// shortcut could not fire here even if the predicate were wrong. What this covers is the opposite
/// risk: that an honest proof of a touched page still folds to a hash rather than being rejected.
/// The predicate itself is pinned by
/// [`verify_fold_rejects_a_partial_write_into_a_blinded_subtree`].
#[test]
fn verify_fold_descends_into_pages_holding_state() {
    let length = 1usize << 42;
    let pages = length.div_ceil(PAGE_SIZE);

    // Path to page 0, with the sibling at each level blinded.
    let mut items = MerkleProof::leaf_read(vec![0u8; 4]);
    for level in 0..tree_depth(pages, NODE_ARITY) {
        items = MerkleProof::node_without_data(vec![
            items,
            MerkleProof::leaf_blind(Hash::hash_bytes(&[level as u8, 0x5a])),
        ]);
    }

    let length_leaf = MerkleProof::leaf_read(
        LeafEncode::<Bincode>::leaf_encode(&(length as u64)).expect("Encoding length should work"),
    );
    let proof = MerkleProof::node_without_data(vec![length_leaf, items]);

    let mut bytes = Bytes::<Verify>::absent(length);
    bytes.write(0, &[1u8; PAGE_SIZE]);

    let hash = PartialHash::from_foldable(Some(proof.clone()), &bytes);

    let PartialHash::Present(hash) = hash else {
        panic!("A coherent state over a well-formed proof should hash, got {hash:?}")
    };
    assert_ne!(
        hash,
        proof.root_hash(),
        "A written page must not be folded away"
    );
}

/// A write covering only part of a blinded subtree must be rejected.
///
/// This is the case the `has_state` predicate exists to catch. The proof carries a single blind for
/// the whole page sequence and one page underneath it is written, so the old contents of the
/// remaining pages are still needed and are not there: the written page hashes to something, its
/// neighbours defer to a proof that says nothing about them, and mixing the two is what
/// `InvalidProof` reports. Were the predicate to under-report, the fold would answer from the blind
/// and quietly accept a state the proof cannot support.
///
/// Note that blinding alone is not what makes this fail - overwriting the region in full succeeds,
/// per [`verify_fold_accepts_a_full_overwrite_of_a_blinded_subtree`]. It fails because the write is
/// partial, leaving pages whose previous contents nothing can supply.
#[test]
fn verify_fold_rejects_a_partial_write_into_a_blinded_subtree() {
    let length = 1usize << 42;

    let length_leaf = MerkleProof::leaf_read(
        LeafEncode::<Bincode>::leaf_encode(&(length as u64)).expect("Encoding length should work"),
    );
    let pages = MerkleProof::leaf_blind(Hash::hash_bytes(b"untouched pages"));
    let proof = MerkleProof::node_without_data(vec![length_leaf, pages]);

    let mut bytes = Bytes::<Verify>::absent(length);
    bytes.write(0, &[1u8; PAGE_SIZE]);

    assert_eq!(
        PartialHash::from_foldable(Some(proof), &bytes),
        PartialHash::InvalidProof,
        "A page written underneath a blind cannot be re-hashed from that blind"
    );
}

/// A write beneath a node the proof says nothing about must be rejected too.
///
/// Unlike the blinded case there is no hash to fall back on at all: the written page hashes to
/// something, the page beside it defers to a proof carrying nothing for it, and `InvalidProof` is
/// what mixing the two reports.
///
/// This pins the rejection rather than the shortcut. The mixture arises among the leaves, beside
/// the shortcut rather than through it, so the outcome holds however `skip_unchanged_subtree`
/// answers for an absent subtree - unlike
/// [`verify_fold_rejects_a_partial_write_into_a_blinded_subtree`], which does discriminate the predicate.
/// What both guard is the present/absent mixing rule in `PartialHashNodeFold::done`, without which
/// this write would be quietly accepted.
#[test]
fn verify_fold_rejects_a_write_under_an_absent_node() {
    let length = 1usize << 42;

    // The proof carries the length and nothing else - the page sequence is absent from it entirely.
    let length_leaf = MerkleProof::leaf_read(
        LeafEncode::<Bincode>::leaf_encode(&(length as u64)).expect("Encoding length should work"),
    );
    let proof = MerkleProof::node_without_data(vec![length_leaf]);

    let mut bytes = Bytes::<Verify>::absent(length);
    bytes.write(0, &[1u8; PAGE_SIZE]);

    assert_eq!(
        PartialHash::from_foldable(Some(proof), &bytes),
        PartialHash::InvalidProof,
        "A page written where the proof carries nothing cannot be re-hashed"
    );
}

/// Overwriting a blinded region in full must succeed: none of the old contents are needed to
/// re-hash it.
///
/// This is the counterpart to [`verify_fold_rejects_a_partial_write_into_a_blinded_subtree`]. Every
/// page under the blind is defined, so each hashes from what the state now holds and the subtree
/// re-hashes from those - the blind is never consulted. Asserting against the hash of a concrete
/// [`Normal`] value of the same contents pins that it reproduces the real hash, rather than merely
/// avoiding rejection.
#[test]
fn verify_fold_accepts_a_full_overwrite_of_a_blinded_subtree() {
    let contents = [7u8; 4 * PAGE_SIZE];

    let length_leaf = MerkleProof::leaf_read(
        LeafEncode::<Bincode>::leaf_encode(&(contents.len() as u64))
            .expect("Encoding length should work"),
    );
    let pages = MerkleProof::leaf_blind(Hash::hash_bytes(b"the contents being replaced"));
    let proof = MerkleProof::node_without_data(vec![length_leaf, pages]);

    let mut bytes = Bytes::<Verify>::absent(contents.len());
    bytes.write(0, &contents);

    let expected = Hash::from_foldable(&Bytes::<Normal>::from(&contents[..]));

    assert_eq!(
        PartialHash::from_foldable(Some(proof), &bytes),
        PartialHash::Present(expected),
        "A fully overwritten blinded region should re-hash to what its contents actually hash to"
    );
}

/// A resize that crosses a power-of-arity boundary changes the page tree's depth, so the reference
/// proof no longer has the shape the state does and `DepthAdjustedSeqAsTree` re-scopes it. That has
/// to stay bounded when the claimed length is huge, since the claim is the proof's to make.
///
/// Here a value claiming a tebibyte grows by one byte, taking the page count from `2^30` to
/// `2^30 + 1` and the depth from 30 to 31, so the proof is padded with a dummy layer. Only the page
/// holding the new byte is descended into; the rest of the tree is skipped from the blind the proof
/// carries. Walking it instead would mean a billion pages.
#[test]
fn verify_fold_stays_bounded_across_a_depth_adjusting_resize() {
    let original_length = 1usize << 40;

    let length_leaf = MerkleProof::leaf_read(
        LeafEncode::<Bincode>::leaf_encode(&(original_length as u64))
            .expect("Encoding length should work"),
    );
    let pages = MerkleProof::leaf_blind(Hash::hash_bytes(b"a tebibyte of untouched pages"));
    let proof = MerkleProof::node_without_data(vec![length_leaf, pages]);

    let mut bytes = Bytes::<Verify>::absent(original_length);
    bytes.resize(original_length + 1);

    assert_eq!(
        tree_depth(original_length.div_ceil(PAGE_SIZE), NODE_ARITY) + 1,
        tree_depth((original_length + 1).div_ceil(PAGE_SIZE), NODE_ARITY),
        "The resize should be the one that pushes the page tree a level deeper"
    );

    let hash = PartialHash::from_foldable(Some(proof), &bytes);

    assert!(
        matches!(hash, PartialHash::Present(_)),
        "The grown page is defined and every other page is unchanged, so this should hash; got \
         {hash:?}"
    );
}
