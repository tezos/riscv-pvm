// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`Bytes`]

use proptest::collection::vec;
use proptest::prelude::Just;
use proptest::prelude::Strategy;
use proptest::prelude::any;
use proptest::prop_assert;
use proptest::prop_assert_eq;
use proptest::prop_oneof;
use proptest::proptest;

use crate::components::bytes::Bytes;
use crate::components::bytes::BytesMode;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::merkle_tree::MerkleTree;
use crate::mode::Normal;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode_test;

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

/// Operations to be issued against an immutable Bytes state component
#[derive(Debug, Clone)]
pub(crate) enum BytesOp {
    Read { offset: usize, size: usize },
    Len,
}

impl BytesOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub(crate) fn any() -> impl Strategy<Value = Self> {
        prop_oneof![
            (0usize..100, 0usize..50).prop_map(|(offset, size)| Self::Read { offset, size }),
            Just(Self::Len),
        ]
    }

    /// Run an operation against an immutable Bytes state component.
    pub(crate) fn run<M: BytesMode>(&self, bytes: &Bytes<M>) -> BytesOpResult {
        match self {
            Self::Read { offset, size } => {
                let mut data = vec![0u8; *size];
                let read = bytes.read(*offset, &mut data);
                BytesOpResult::Read { read, data }
            }

            Self::Len => BytesOpResult::Len { len: bytes.len() },
        }
    }
}

/// Operations to be issued against a mutable Bytes state component
#[derive(Debug, Clone)]
pub(crate) enum BytesMutOp {
    Write { offset: usize, data: Vec<u8> },
    Resize { new_size: usize },
    Immutable { op: BytesOp },
}

impl BytesMutOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub(crate) fn any() -> impl Strategy<Value = Self> {
        prop_oneof![
            (0usize..100, vec(any::<u8>(), 0..50))
                .prop_map(|(offset, data)| Self::Write { offset, data }),
            (0usize..150).prop_map(|new_size| Self::Resize { new_size }),
            BytesOp::any().prop_map(|op| Self::Immutable { op }),
        ]
    }

    /// Run the operation against the Bytes state component.
    pub(crate) fn run<M: BytesMode>(&self, bytes: &mut Bytes<M>) -> BytesOpResult {
        match self {
            Self::Write { offset, data } => {
                let wrote = bytes.write(*offset, data);
                BytesOpResult::Wrote { wrote }
            }

            Self::Resize { new_size } => {
                bytes.resize(*new_size);
                BytesOpResult::Void
            }

            Self::Immutable { op } => op.run(bytes),
        }
    }
}

/// Results of operations issued against the Bytes state component
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BytesOpResult {
    Read { read: usize, data: Vec<u8> },
    Wrote { wrote: usize },
    Len { len: usize },
    Void,
}

// Bytes behaves the same across different modes
#[test]
fn bytes_are_same_across_modes() {
    proptest!(|(ops in vec(BytesMutOp::any(), 1..20))| {
        let mut bytes_normal = Bytes::<Normal>::default();
        let results_normal = ops.iter().map(|op| op.run(&mut bytes_normal)).collect::<Vec<_>>();
        let hash_normal = Hash::from_foldable(&bytes_normal);

        let mut bytes_prove = Bytes::<Prove>::default();
        let results_prove = ops.iter().map(|op| op.run(&mut bytes_prove)).collect::<Vec<_>>();
        prop_assert_eq!(&results_normal, &results_prove);

        let hash_prove = Hash::from_foldable(&bytes_prove);
        prop_assert_eq!(hash_normal, hash_prove);

        let merkle_tree = MerkleTree::from_foldable(&bytes_prove);
        let merkle_proof = merkle_tree.compress();

        let mut bytes_verify = Bytes::<Verify>::default();
        let results_verify = ops.iter().map(|op| op.run(&mut bytes_verify)).collect::<Vec<_>>();
        prop_assert_eq!(results_normal, results_verify);

        let hash_verify =
            PartialHash::from_foldable(Some(&merkle_proof), &bytes_verify)
                .to_hash()
                .unwrap();
        prop_assert_eq!(hash_normal, hash_verify);
    });
}
