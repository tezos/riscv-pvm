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

use super::Blake3Bytes;
use super::Blake3Proof;
use super::MAX_PROOF_TREE_DEPTH;
use super::ProofError;
use super::ProofTree;
use super::canonical_cv;
use super::hash_len;
use super::hash_value;
use super::present_ranges;
use super::proof_read;
use super::prove;
use super::verify_root;
use crate::hash::Hash;
use crate::mode::Normal;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::catch_not_found;
use crate::serialisation::deserialise;
use crate::serialisation::serialise;

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

proptest! {
    /// I4: a range that was never accessed is blinded, and reading it faults rather than
    /// returning zeroes. A verifier that invented data for unproven regions would pass every
    /// round-trip test while being unsound.
    #[test]
    fn unaccessed_read_faults(
        start in (2 * CHUNK_LEN)..(8 * CHUNK_LEN),
        len in 1usize..=CHUNK_LEN,
    ) {
        let data = vec![7u8; 8 * CHUNK_LEN];
        let proof = prove(&data, &[0..1]);
        let end = (start + len).min(data.len());
        prop_assume!(end > start);
        prop_assert!(matches!(proof_read(&proof, start..end), Err(ProofError::Blinded)));
    }

    /// Accessed ranges are present and read back exactly the bytes that were proved.
    #[test]
    fn accessed_ranges_read_back((data, access) in data_and_access()) {
        prop_assume!(!access.is_empty());
        let proof = prove(&data, &access);
        for range in &access {
            let got = proof_read(&proof, range.clone()).unwrap();
            prop_assert_eq!(got.as_slice(), &data[range.clone()]);
        }
    }

    /// I4: reading past the claimed length faults rather than returning padding.
    #[test]
    fn out_of_bounds_read_faults(over in 1usize..64) {
        let data = vec![3u8; 2 * CHUNK_LEN];
        let proof = prove(&data, &[0..data.len()]);
        let start = data.len();
        prop_assert!(matches!(
            proof_read(&proof, start..(start + over)),
            Err(ProofError::OutOfBounds)
        ));
    }

    /// The ranges reported as present are exactly those that can be read back, so a caller can
    /// use them to decide what it may ask for.
    #[test]
    fn present_ranges_are_readable((data, access) in data_and_access()) {
        let proof = prove(&data, &access);
        for range in present_ranges(&proof) {
            let got = proof_read(&proof, range.clone()).unwrap();
            prop_assert_eq!(got.as_slice(), &data[range]);
        }
    }
}

proptest! {
    /// A proof survives the wire: what comes back verifies to the same committed hash.
    #[test]
    fn proof_roundtrips_through_the_wire((data, access) in data_and_access()) {
        let proof = prove(&data, &access);
        let bytes = serialise(&proof).expect("proof serialises");
        let decoded = deserialise::<Blake3Proof>(&bytes).expect("proof decodes");
        prop_assert_eq!(&decoded, &proof);
        prop_assert_eq!(verify_root(&decoded).unwrap(), hash_value(&data));
    }
}

/// I5: a present chunk larger than `CHUNK_LEN` is refused by the decoder, before the bytes are
/// allocated rather than after.
#[test]
fn decode_rejects_oversized_chunk() {
    let proof = Blake3Proof {
        total_len: CHUNK_LEN + 5,
        data: ProofTree::Chunk(vec![0u8; CHUNK_LEN + 5]),
    };
    let encoded = serialise(&proof).unwrap();
    assert!(deserialise::<Blake3Proof>(&encoded).is_err());
}

/// A proof nested beyond the depth bound is rejected by the decoder rather than overflowing
/// the stack. The tree is built and serialised iteratively, so only the decode side is under
/// test.
#[test]
fn overly_deep_proof_tree_rejected() {
    let mut tree = ProofTree::Blind([0u8; 32]);
    for _ in 0..(MAX_PROOF_TREE_DEPTH + 50) {
        tree = ProofTree::Node(Box::new(tree), Box::new(ProofTree::Blind([0u8; 32])));
    }
    let proof = Blake3Proof {
        total_len: 4 * CHUNK_LEN,
        data: tree,
    };
    let bytes = serialise(&proof).expect("encode is iterative, so it does not overflow");
    assert!(
        deserialise::<Blake3Proof>(&bytes).is_err(),
        "decoder accepted an over-deep proof tree instead of rejecting it"
    );
}

/// The depth bound does not reject well-formed proofs: a fully present multi-level value still
/// round-trips.
#[test]
fn deep_but_canonical_proof_roundtrips() {
    let data = vec![0x33u8; 64 * CHUNK_LEN];
    let proof = prove(&data, &[0..data.len()]);
    let bytes = serialise(&proof).expect("serialises");
    let decoded = deserialise::<Blake3Proof>(&bytes).expect("well-formed proof decodes");
    assert_eq!(verify_root(&decoded).unwrap(), hash_value(&data));
}

/// The serialised size of a proof for a 2 KiB access is `O(depth)` and does not depend on
/// where in the value the access falls. Position independence is what unrestricted blinding
/// buys: there is no spine or boundary that has to be materialised.
#[test]
fn proof_size_is_o_depth_and_position_independent() {
    /// Present-chunk and blinded-chaining-value counts in a proof tree.
    fn counts(tree: &ProofTree) -> (usize, usize) {
        match tree {
            ProofTree::Chunk(_) => (1, 0),
            ProofTree::Blind(_) => (0, 1),
            ProofTree::Node(l, r) => {
                let (a, b) = counts(l);
                let (c, d) = counts(r);
                (a + c, b + d)
            }
        }
    }

    for mib in [1usize, 16, 64] {
        let n = mib << 20;
        let depth = (n / CHUNK_LEN).next_power_of_two().trailing_zeros() as usize;
        let data = vec![0x5Au8; n];

        let mut wires = Vec::new();
        for access in [0..2048, n / 2..n / 2 + 2048, n - 2048..n] {
            let proof = prove(&data, &[access]);
            assert_eq!(
                verify_root(&proof).expect("honest proof verifies"),
                hash_value(&data)
            );
            let wire = serialise(&proof).expect("proof serialises").len();
            let (present, blinds) = counts(&proof.data);

            // A 2 KiB access spans at most three chunks, and the blinds are the
            // authentication-path siblings, one per level plus a small constant.
            assert!(present <= 4, "present {present} at {mib} MiB");
            assert!(blinds <= 2 * depth + 4, "blinds {blinds} (depth {depth})");
            wires.push(wire);
        }

        let min = *wires.iter().min().unwrap();
        let max = *wires.iter().max().unwrap();
        assert!(
            max - min <= 2 * CHUNK_LEN,
            "position-dependent wire size at {mib} MiB: min {min} max {max}"
        );
    }
}

/// Read a range through a verify-mode value, returning `None` where the read faults.
fn verify_read(value: &Blake3Bytes<Verify>, range: Range<usize>) -> Option<Vec<u8>> {
    catch_not_found(|| {
        let mut buf = vec![0u8; range.len()];
        let read = value.read(range.start, &mut buf);
        buf.truncate(read);
        buf
    })
    .ok()
}

/// The mode-generic operations behave the same way a plain byte buffer would, including that
/// bytes exposed by a grow read back as zero.
#[test]
fn mode_ops_basic() {
    let mut value = Blake3Bytes::<Normal>::new(10);
    assert_eq!(value.len(), 10);
    assert_eq!(value.write(2, &[1, 2, 3]), 3);
    let mut buf = [0u8; 3];
    assert_eq!(value.read(2, &mut buf), 3);
    assert_eq!(buf, [1, 2, 3]);

    value.resize(4);
    assert_eq!(value.len(), 4);
    value.resize(6);
    assert_eq!(value.len(), 6);
    let mut tail = [9u8; 2];
    value.read(4, &mut tail);
    assert_eq!(tail, [0, 0]);
}

proptest! {
    /// A read-only transition: the proof the prover emits verifies to the value's committed
    /// hash, and every range the transition read is readable through the reconstructed verify
    /// view. This is the whole prove/verify contract at the level a caller sees it.
    #[test]
    fn read_transition_reconstructs_the_view((data, access) in data_and_access()) {
        let committed = hash_value(&data);
        let prover = Blake3Bytes::<Prove>::from_raw_source(&data);
        for range in &access {
            let mut buf = vec![0u8; range.len()];
            let read = prover.read(range.start, &mut buf);
            prop_assert_eq!(read, range.len());
            prop_assert_eq!(&buf[..], &data[range.clone()]);
        }

        let proof = prover.value_proof();
        prop_assert_eq!(verify_root(&proof).unwrap(), committed);
        prop_assert_eq!(prover.hash(), committed);

        let verify = Blake3Bytes::<Verify>::from_proof_checked(proof, committed).unwrap();
        for range in &access {
            let got = verify_read(&verify, range.clone())
                .expect("a range the transition read must be present");
            prop_assert_eq!(got.as_slice(), &data[range.clone()]);
        }
    }

    /// A writing transition reaches the same post-transition hash in prove mode as a normal
    /// value written the same way. Prove mode never materialises the post value for the caller,
    /// so this is what pins that its bookkeeping reconstructs it correctly.
    #[test]
    fn write_transition_hashes_like_a_normal_value(
        data in bytes_of_len(3 * CHUNK_LEN),
        at in 0usize..(3 * CHUNK_LEN - 16),
    ) {
        let patch = [0xEEu8; 16];

        let mut normal = Blake3Bytes::<Normal>::from(data.as_slice());
        normal.write(at, &patch);

        let mut prover = Blake3Bytes::<Prove>::from_raw_source(&data);
        prover.write(at, &patch);

        prop_assert_eq!(prover.hash(), normal.hash());
    }

    /// A transition that resizes reaches the same post-transition hash as a normal value
    /// resized the same way, whether it grows or shrinks.
    #[test]
    fn resize_transition_hashes_like_a_normal_value(
        data in bytes_of_len(2 * CHUNK_LEN),
        new_len in 0usize..(4 * CHUNK_LEN),
    ) {
        let mut normal = Blake3Bytes::<Normal>::from(data.as_slice());
        normal.resize(new_len);

        let mut prover = Blake3Bytes::<Prove>::from_raw_source(&data);
        prover.resize(new_len);

        prop_assert_eq!(prover.hash(), normal.hash());
    }
}

/// A value that shrinks and then regrows must read back zeros in the regrown region, not the
/// bytes that were there before. Prove mode still holds the pre-transition bytes, so without
/// the low-water mark it would hash the stale ones and disagree with a normal value.
#[test]
fn shrink_then_regrow_drops_the_old_bytes() {
    let data = vec![0xAAu8; 2 * CHUNK_LEN];

    let mut normal = Blake3Bytes::<Normal>::from(data.as_slice());
    normal.resize(0);
    normal.resize(2 * CHUNK_LEN);

    let mut prover = Blake3Bytes::<Prove>::from_raw_source(&data);
    prover.resize(0);
    prover.resize(2 * CHUNK_LEN);

    assert_eq!(prover.hash(), normal.hash());
    assert_eq!(prover.hash(), hash_value(&vec![0u8; 2 * CHUNK_LEN]));
}
