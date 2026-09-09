// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for the length-committed BLAKE3 hashing scheme.

#![expect(
    clippy::single_range_in_vec_init,
    reason = "accessed ranges are passed as one-element slices throughout, which is what these tests mean; the lint reads them as a mistyped range"
)]

use std::ops::Range;

use blake3::CHUNK_LEN;
use blake3::hazmat::HasherExt;
use blake3::hazmat::Mode as B3Mode;
use blake3::hazmat::left_subtree_len;
use blake3::hazmat::merge_subtrees_non_root;
use blake3::hazmat::merge_subtrees_root;
use proptest::prelude::*;

use super::Blake3Proof;
use super::ProofError;
use super::ProofTree;
use super::canonical_cv;
use super::hash_len;
use super::hash_value;
use super::prove;
use super::verify_root;
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

// -- Reference canonical decomposition: the algorithm `prove` and `verify_root` mirror. --
//
// Written out independently of the implementation so a bug in the recursion cannot hide by
// being made twice; the property tests below check the two agree with `blake3::hash`.

/// Chaining value of the canonical subtree covering `input`, which *starts* at byte `offset`
/// (a chunk boundary). Non-root.
fn subtree_cv(input: &[u8], offset: u64) -> [u8; 32] {
    if input.len() <= CHUNK_LEN {
        return blake3::Hasher::new()
            .set_input_offset(offset)
            .update(input)
            .finalize_non_root();
    }
    let left_len = left_subtree_len(input.len() as u64) as usize;
    let (left, right) = input.split_at(left_len);
    let left_cv = subtree_cv(left, offset);
    let right_cv = subtree_cv(right, offset + left_len as u64);
    merge_subtrees_non_root(&left_cv, &right_cv, B3Mode::Hash)
}

/// Root of the canonical BLAKE3 tree of `input`.
fn ref_root(input: &[u8]) -> [u8; 32] {
    if input.len() <= CHUNK_LEN {
        // A single chunk finalised with ROOT is exactly `blake3::hash`.
        return *blake3::hash(input).as_bytes();
    }
    let left_len = left_subtree_len(input.len() as u64) as usize;
    let (left, right) = input.split_at(left_len);
    let left_cv = subtree_cv(left, 0);
    let right_cv = subtree_cv(right, left_len as u64);
    *merge_subtrees_root(&left_cv, &right_cv, B3Mode::Hash).as_bytes()
}

/// The root with the whole left subtree replaced by its precomputed chaining value — the shape
/// a blinded subtree produces.
fn root_with_left_blinded(input: &[u8]) -> [u8; 32] {
    assert!(input.len() > CHUNK_LEN);
    let left_len = left_subtree_len(input.len() as u64) as usize;
    let (left, right) = input.split_at(left_len);
    let blinded_left = subtree_cv(left, 0);
    let right_cv = subtree_cv(right, left_len as u64);
    *merge_subtrees_root(&blinded_left, &right_cv, B3Mode::Hash).as_bytes()
}

fn data_and_access() -> impl Strategy<Value = (Vec<u8>, Vec<Range<usize>>)> {
    interesting_len().prop_flat_map(|len| {
        let data = bytes_of_len(len);
        let ranges = if len == 0 {
            Just(Vec::new()).boxed()
        } else {
            proptest::collection::vec(
                (0usize..len)
                    .prop_flat_map(move |a| (Just(a), (a + 1)..=len))
                    .prop_map(|(a, b)| a..b),
                0..=4,
            )
            .boxed()
        };
        (data, ranges)
    })
}

proptest! {
    /// I1: the canonical decomposition this module encodes is BLAKE3's own, so walking it by
    /// hand reproduces `blake3::hash`. Everything else rests on this.
    #[test]
    fn canonical_decomposition_agrees_with_blake3(len in interesting_len()) {
        let data = vec![0xA5u8; len];
        prop_assert_eq!(ref_root(&data), *blake3::hash(&data).as_bytes());
    }

    /// I1/I6: replacing an aligned subtree by its chaining value leaves the root unchanged.
    /// This is what makes blinding possible at all.
    #[test]
    fn blinding_a_subtree_preserves_the_root(len in (CHUNK_LEN + 1)..=8 * CHUNK_LEN) {
        let data = vec![0x5Au8; len];
        prop_assert_eq!(root_with_left_blinded(&data), *blake3::hash(&data).as_bytes());
    }

    /// I8: prove-then-verify reproduces the committed hash, for any access set — including the
    /// empty one, where the whole value collapses to a single blind.
    #[test]
    fn prove_then_verify_reproduces_hash((data, access) in data_and_access()) {
        let proof = prove(&data, &access);
        prop_assert_eq!(verify_root(&proof).unwrap(), hash_value(&data));
    }

    /// I2/I3: tampering with the claimed length makes verification reject or diverge from the
    /// committed root.
    #[test]
    fn tampered_length_rejected((data, access) in data_and_access()) {
        let committed = hash_value(&data);
        let mut proof = prove(&data, &access);
        proof.total_len = proof.total_len.wrapping_add(CHUNK_LEN);
        match verify_root(&proof) {
            Err(_) => {}
            Ok(root) => prop_assert_ne!(root, committed),
        }
    }

    /// Substituting a blinded chaining value changes the recomputed root, or is rejected.
    #[test]
    fn tampered_blind_detected(data in bytes_of_len(4 * CHUNK_LEN)) {
        let committed = hash_value(&data);
        let mut proof = prove(&data, &[0..1]);
        fn tamper(t: &mut ProofTree) -> bool {
            match t {
                ProofTree::Blind(cv) => { cv[0] ^= 1; true }
                ProofTree::Node(l, r) => tamper(l) || tamper(r),
                ProofTree::Chunk(_) => false,
            }
        }
        prop_assume!(tamper(&mut proof.data));
        match verify_root(&proof) {
            Err(_) => {}
            Ok(root) => prop_assert_ne!(root, committed),
        }
    }

    /// I5: a present chunk larger than `CHUNK_LEN` is rejected. Verify decodes attacker input,
    /// so this is a memory-safety bound, not only a canonicity one.
    #[test]
    fn oversized_chunk_rejected(extra in 1usize..64) {
        let proof = Blake3Proof {
            total_len: CHUNK_LEN + extra,
            data: ProofTree::Chunk(vec![0u8; CHUNK_LEN + extra]),
        };
        prop_assert_eq!(verify_root(&proof), Err(ProofError::ChunkTooLarge));
    }

    /// Tampering with a *present* chunk's bytes changes the recomputed root, or is rejected.
    /// This is the guard against a `verify_root` that never hashes present chunk contents at
    /// all — such an implementation passes every other test here.
    #[test]
    fn tampered_present_chunk_detected(data in bytes_of_len(4 * CHUNK_LEN)) {
        let committed = hash_value(&data);
        let mut proof = prove(&data, &[0..data.len()]);
        fn flip_first_chunk(t: &mut ProofTree) -> bool {
            match t {
                ProofTree::Chunk(b) if !b.is_empty() => { b[0] ^= 1; true }
                ProofTree::Chunk(_) => false,
                ProofTree::Node(l, r) => flip_first_chunk(l) || flip_first_chunk(r),
                ProofTree::Blind(_) => false,
            }
        }
        prop_assume!(flip_first_chunk(&mut proof.data));
        match verify_root(&proof) {
            Err(_) => {}
            Ok(root) => prop_assert_ne!(root, committed),
        }
    }

    /// Two different values of the same length verify to different roots, each equal to its own
    /// `hash_value`. Guards against a `verify_root` that returns a constant, echoes the length,
    /// or otherwise ignores content.
    #[test]
    fn distinct_values_verify_to_distinct_roots(
        data in bytes_of_len(2 * CHUNK_LEN + 5),
        idx in 0usize..(2 * CHUNK_LEN + 5),
    ) {
        let a = data.clone();
        let mut b = data;
        b[idx] ^= 0xFF;
        let ra = verify_root(&prove(&a, &[0..a.len()])).unwrap();
        let rb = verify_root(&prove(&b, &[0..b.len()])).unwrap();
        prop_assert_eq!(ra, hash_value(&a));
        prop_assert_eq!(rb, hash_value(&b));
        prop_assert_ne!(ra, rb);
    }
}

/// I2: a structurally non-canonical proof is rejected. A two-chunk value's canonical shape is
/// `Node(Chunk, Chunk)`, so a bare single `Chunk` claiming `total_len = 2 * CHUNK_LEN` is not
/// one the shape allows.
#[test]
fn non_canonical_shape_rejected() {
    let proof = Blake3Proof {
        total_len: 2 * CHUNK_LEN,
        data: ProofTree::Chunk(vec![0u8; CHUNK_LEN]),
    };
    assert_eq!(verify_root(&proof), Err(ProofError::NonCanonicalShape));
}

/// The cross-span graft that broke the length-free predecessor, now rejected.
///
/// The honest value is three full chunks, so its data root is `merge_root(cv[0..2], cv2)`.
/// Forging `total_len = 4096` splits `2048 | 2048`, and the right slot accepts a `Blind`:
/// planting the genuine chunk-2 chaining value there reconstructs the *identical* honest data
/// root. Under the old scheme that reproduced the honest value hash exactly. Here the length
/// commitment differs, so the value hash does too.
#[test]
fn cross_span_graft_rejected() {
    let data = vec![0xABu8; 3 * CHUNK_LEN];
    let committed = hash_value(&data);

    let cv2 = canonical_cv(&data[2 * CHUNK_LEN..3 * CHUNK_LEN], (2 * CHUNK_LEN) as u64);
    let forged = Blake3Proof {
        total_len: 4 * CHUNK_LEN,
        data: ProofTree::Node(
            Box::new(ProofTree::Node(
                Box::new(ProofTree::Chunk(data[0..CHUNK_LEN].to_vec())),
                Box::new(ProofTree::Chunk(data[CHUNK_LEN..2 * CHUNK_LEN].to_vec())),
            )),
            Box::new(ProofTree::Blind(cv2)),
        ),
    };

    match verify_root(&forged) {
        Err(_) => {}
        Ok(root) => assert_ne!(root, committed, "cross-span graft forged the length"),
    }
}

/// The strongest constructed graft, over a wide range of forged lengths: build the canonical
/// shape for `L'` and fill every slot with the true value's genuine chaining values and bytes.
/// Verification must always error or diverge from the committed hash.
#[test]
fn constructed_graft_at_any_length_rejected() {
    for chunks in 1usize..=12 {
        let data = vec![0x5Au8; chunks * CHUNK_LEN];
        let committed = hash_value(&data);
        for lp_chunks in 1usize..=16 {
            let l_prime = lp_chunks * CHUNK_LEN;
            if l_prime == data.len() {
                continue;
            }
            let forged = Blake3Proof {
                total_len: l_prime,
                data: forge_graft(&data, 0, l_prime, true),
            };
            match verify_root(&forged) {
                Err(_) => {}
                Ok(root) => assert_ne!(
                    root,
                    committed,
                    "graft forged {} bytes as {l_prime}",
                    data.len()
                ),
            }
        }
    }
}

/// The strongest attacker's proof: the canonical shape for `L'`, every position filled with the
/// true value's genuine chaining value or bytes for that byte range — cross-span wherever the
/// spans differ.
fn forge_graft(honest: &[u8], offset: usize, span: usize, is_root: bool) -> ProofTree {
    let filled = |off: usize, sp: usize| -> Vec<u8> {
        let mut buf = vec![0u8; sp];
        let end = (off + sp).min(honest.len());
        if off < end {
            buf[..end - off].copy_from_slice(&honest[off..end]);
        }
        buf
    };
    if !is_root {
        return ProofTree::Blind(canonical_cv(&filled(offset, span), offset as u64));
    }
    if span <= CHUNK_LEN {
        ProofTree::Chunk(filled(offset, span))
    } else {
        let ll = left_subtree_len(span as u64) as usize;
        ProofTree::Node(
            Box::new(forge_graft(honest, offset, ll, false)),
            Box::new(forge_graft(honest, offset + ll, span - ll, false)),
        )
    }
}

/// I8: the empty value's proof round-trips like any other.
#[test]
fn empty_value_roundtrips() {
    let data: &[u8] = &[];
    let proof = prove(data, &[]);
    assert_eq!(verify_root(&proof).unwrap(), hash_value(data));
}
