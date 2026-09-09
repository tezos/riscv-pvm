// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for the length-committed BLAKE3 hashing scheme.

use blake3::CHUNK_LEN;
use proptest::prelude::*;

use super::hash_len;
use super::hash_value;
use crate::hash::Hash;

/// Sizes chosen to exercise the empty value, a sub-chunk value, an exact chunk, a chunk plus
/// one byte, multi-chunk values, exact powers of two, and ragged right spines.
pub(super) fn interesting_len() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(0usize),
        1usize..CHUNK_LEN,
        Just(CHUNK_LEN),
        Just(CHUNK_LEN + 1),
        Just(2 * CHUNK_LEN),
        Just(3 * CHUNK_LEN),
        Just(4 * CHUNK_LEN),
        Just(5 * CHUNK_LEN + 7),
        (0usize..=8 * CHUNK_LEN),
    ]
}

pub(super) fn bytes_of_len(len: usize) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), len..=len)
}

proptest! {
    /// I8: the committed hash is `combine(H_len, blake3::hash(bytes))` — the length
    /// commitment — and is NOT bare `blake3::hash(bytes)`. A value hash equal to the bare
    /// BLAKE3 hash would be the length-free scheme this replaces.
    #[test]
    fn hash_value_commits_length(data in bytes_of_len(3 * CHUNK_LEN + 11)) {
        let expected = Hash::combine_hashes([hash_len(data.len()), Hash::hash_bytes(&data)]);
        prop_assert_eq!(hash_value(&data), expected);
        prop_assert_ne!(hash_value(&data), Hash::hash_bytes(&data));
    }

    /// I8: a value and its truncation hash differently even though one's bytes are a prefix
    /// of the other's. This is the property the length leaf exists to provide.
    #[test]
    fn truncation_changes_value_hash(
        data in bytes_of_len(2 * CHUNK_LEN),
        cut in 1usize..2 * CHUNK_LEN,
    ) {
        let short = &data[..data.len() - cut];
        prop_assert_ne!(hash_value(&data), hash_value(short));
    }

    /// I8: the length leaf does not swallow the bytes — two values of equal length that
    /// differ in one byte hash differently.
    #[test]
    fn content_changes_value_hash(
        data in bytes_of_len(2 * CHUNK_LEN + 5),
        idx in 0usize..(2 * CHUNK_LEN + 5),
    ) {
        let mut other = data.clone();
        other[idx] ^= 0xFF;
        prop_assert_ne!(hash_value(&data), hash_value(&other));
    }

    /// The empty value has a hash of its own, and no non-empty value shares it. It is a legal
    /// value rather than an absent one, so every mode has to agree on it.
    #[test]
    fn empty_value_hashes_distinctly(len in interesting_len()) {
        let data = vec![0u8; len];
        prop_assert_eq!(hash_value(&data) == hash_value(&[]), len == 0);
    }
}

/// A multi-chunk value hashes by the same formula as one that fits in a single chunk. The
/// bytes half is BLAKE3's own tree, so this pins that we never re-derive it ourselves above
/// some size threshold.
#[test]
fn large_value_matches_reference_formula() {
    for len in [255usize * 1024, 256 * 1024, 512 * 1024 + 7, 2 * 1024 * 1024] {
        let data: Vec<u8> = (0..len)
            .map(|i| i.wrapping_mul(0x9E3779B1usize) as u8)
            .collect();
        let reference = Hash::combine_hashes([hash_len(len), Hash::hash_bytes(&data)]);
        assert_eq!(
            hash_value(&data),
            reference,
            "hash_value drift at len {len}"
        );
    }
}
