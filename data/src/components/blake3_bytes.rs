// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! BLAKE3-direct hashing scheme for a byte array (**stub**).
//!
//! This is the replacement for the page-tree scheme in [`super::bytes`]. A value's hash
//! becomes `blake3::hash(bytes)` directly; within-value partial proofs ride BLAKE3's own
//! internal 1024-byte chunk tree via the [`blake3::hazmat`] guts API instead of a
//! hand-rolled arity-4 page tree.
//!
//! **Everything here is stubbed with [`todo!`].** The full design, the security
//! properties this must uphold, and the numbered invariants (`I1`..`I8`) referenced
//! throughout are in `data/docs/blake3-bytes-hashing.md`. The property tests at the
//! bottom of this file are the executable form of those invariants and are the spec the
//! implementing session must turn green.
//!
//! ## What the implementer must build
//!
//! - [`hash_value`] — Normal-mode hash. MUST equal `blake3::hash(bytes)` (invariant I8).
//! - [`prove`] — build a [`Blake3Proof`] capturing the accessed byte ranges of the
//!   pre-transition value, blinding every untouched *legal* BLAKE3 subtree to its 32-byte
//!   chaining value (invariants I1, I5).
//! - [`verify_root`] — recompute the committed root from a proof, canonically from the
//!   claimed length, rejecting non-canonical shapes and illegal offsets (invariants I2,
//!   I3, I6).
//! - [`proof_read`] — read a byte range from a verified proof; faults on blinded/absent
//!   regions (invariant I4).
//!
//! Wiring the mode-generic [`Blake3Bytes`] into the fold pipelines (`HashFold`,
//! `MerkleProofFold`, `PartialHashFold`) so it can replace [`super::bytes::Bytes`] as an
//! AVL value is the integration step and is intentionally left as `todo!` here — keep the
//! value-proof encoding structurally disjoint from the AVL-node encoding (invariant I7).

use std::ops::Range;

use crate::hash::Hash;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;

/// BLAKE3 chunk length: the leaf granularity of the internal chunk tree (1024 bytes).
pub const CHUNK_LEN: usize = blake3::CHUNK_LEN;

// ---------------------------------------------------------------------------------------
// Normal-mode hash
// ---------------------------------------------------------------------------------------

/// Normal-mode hash of a byte array under the BLAKE3-direct scheme.
///
/// MUST satisfy, for every input, `hash_value(bytes) == Hash::hash_bytes(bytes)`
/// (invariant I8). No length prefix, no page tree, no wrapping node — just BLAKE3 of the
/// raw bytes.
pub fn hash_value(bytes: &[u8]) -> Hash {
    let _ = bytes;
    todo!("hash_value: return Hash::hash_bytes(bytes); see docs §3")
}

// ---------------------------------------------------------------------------------------
// Within-value proof representation
// ---------------------------------------------------------------------------------------

/// A within-value Merkle proof mirroring BLAKE3's internal binary chunk tree.
///
/// The shape is the *canonical* BLAKE3 decomposition for [`Blake3Proof::total_len`]:
/// splits follow [`blake3::hazmat::left_subtree_len`], leaf spans are 1024-byte chunks,
/// and the single top node is finalized with the `ROOT` flag. Verify reconstructs the
/// shape from `total_len` and rejects any proof whose structure deviates (invariant I2).
///
/// This is a *suggested* representation to anchor the tests; the implementer may refine
/// it, but the invariants in the docs must still hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blake3Proof {
    /// Claimed total byte length of the value. Drives the canonical shape and is
    /// validated transitively by the root equality check in [`verify_root`] (invariant
    /// I3) — it is never trusted on its own.
    pub total_len: usize,

    /// Root of the partial chunk tree.
    pub root: ProofTree,
}

/// A node in a [`Blake3Proof`]'s partial chunk tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofTree {
    /// A present chunk carrying its real bytes. Length MUST be `<= CHUNK_LEN` (invariant
    /// I6). Its chaining value is recomputed via
    /// `Hasher::set_input_offset(chunk_index * CHUNK_LEN).update(bytes).finalize_non_root()`.
    Chunk(Vec<u8>),

    /// A blinded, aligned, *legal* subtree, represented only by its 32-byte chaining
    /// value. Its span must satisfy `span <= max_subtree_len(offset)` (invariant I5).
    Blind([u8; 32]),

    /// An internal binary node: `(left, right)`. Non-root nodes merge via
    /// `merge_subtrees_non_root`; the single top node merges via `merge_subtrees_root`.
    Node(Box<ProofTree>, Box<ProofTree>),
}

/// Why a proof was rejected or a read failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofError {
    /// The tree shape does not match the canonical BLAKE3 decomposition of `total_len`.
    NonCanonicalShape,
    /// A subtree offset/span violates the `max_subtree_len` rule.
    IllegalSubtree,
    /// A present chunk exceeds `CHUNK_LEN`.
    ChunkTooLarge,
    /// The recomputed root does not equal the committed root.
    RootMismatch,
    /// The requested read touches a blinded/absent region.
    Blinded,
    /// The requested range is outside `0..total_len`.
    OutOfBounds,
}

// ---------------------------------------------------------------------------------------
// Prove / Verify contract
// ---------------------------------------------------------------------------------------

/// Build a proof for `data` that keeps exactly the chunks overlapping `accessed` present
/// and blinds every other *legal* aligned subtree to its chaining value.
///
/// Blinding is recursive: an untouched aligned subtree of many chunks collapses to a
/// single 32-byte CV (invariants I1, I5). The rightmost (non-power-of-two) region must be
/// split into legal subtrees, never blinded as one illegal subtree.
pub fn prove(data: &[u8], accessed: &[Range<usize>]) -> Blake3Proof {
    let _ = (data, accessed);
    todo!("prove: canonical BLAKE3 shape, present accessed chunks, blind the rest; docs §4")
}

/// Recompute the committed root from `proof`.
///
/// MUST reconstruct the tree shape canonically from `proof.total_len`, reject
/// non-canonical shapes/illegal offsets/oversized chunks (invariants I2, I6), merge
/// present-chunk CVs and blinded CVs per the shape, and return the resulting `ROOT`
/// output. For an honest proof of `data`, this equals `hash_value(data)` (invariant I8).
pub fn verify_root(proof: &Blake3Proof) -> Result<Hash, ProofError> {
    let _ = proof;
    todo!("verify_root: canonical reconstruction + ROOT finalize; docs §4, §5 P3")
}

/// Read `range` from a verified proof.
///
/// Returns the bytes if `range` lies entirely within present chunks; faults with
/// [`ProofError::Blinded`] if any part was blinded, or [`ProofError::OutOfBounds`] if
/// `range` exceeds `total_len` (invariant I4). This never returns zeroes for absent data.
pub fn proof_read(proof: &Blake3Proof, range: Range<usize>) -> Result<Vec<u8>, ProofError> {
    let _ = (proof, range);
    todo!("proof_read: present -> bytes, blinded -> Err(Blinded); docs §5 P5")
}

/// The byte ranges materialized (present) in `proof`, in ascending order. Reading any
/// sub-range of these via [`proof_read`] succeeds; reading outside them faults.
pub fn present_ranges(proof: &Blake3Proof) -> Vec<Range<usize>> {
    let _ = proof;
    todo!("present_ranges: ascending, coalesced ranges backed by Chunk leaves")
}

// ---------------------------------------------------------------------------------------
// Mode-generic component identity (stub facade)
// ---------------------------------------------------------------------------------------

/// Modal template for [`Blake3Bytes`]. The per-mode representations are placeholders; the
/// implementer fills them in when wiring the fold pipelines.
pub enum Blake3BytesTemplate {}

impl Modal for Blake3BytesTemplate {
    type Normal = NormalRepr;
    type Prove<'normal> = ProveRepr<'normal>;
    type Verify = VerifyRepr;
}

/// Normal-mode representation: the full byte array. (Stub.)
#[derive(Debug, Default)]
pub struct NormalRepr {
    /// Backing bytes.
    pub bytes: Vec<u8>,
}

/// Prove-mode representation: borrows the pre-transition bytes and records accesses.
/// (Stub — mirror [`super::bytes`]'s `ProveImpl`: `previous`, read set, write set.)
#[derive(Debug)]
pub struct ProveRepr<'normal> {
    /// Pre-transition source data.
    pub previous: &'normal [u8],
}

/// Verify-mode representation: a sparse view populated from a proof. (Stub.)
#[derive(Debug, Default)]
pub struct VerifyRepr {
    /// Claimed length recovered from the proof.
    pub length: usize,
}

/// Byte-array state component under the BLAKE3-direct hashing scheme (**stub**).
///
/// Public surface mirrors [`super::bytes::Bytes`]. Hashing delegates to [`hash_value`]
/// (Normal) and to [`prove`]/[`verify_root`] (Prove/Verify) once the fold wiring lands.
pub struct Blake3Bytes<M: Mode> {
    /// Mode-specific representation. Public only so the stub compiles without a
    /// dispatch trait; not the final API.
    pub repr: M::Select<Blake3BytesTemplate>,
}

impl<M: Mode> Blake3Bytes<M> {
    /// Create a zero-initialised byte array of the given length.
    pub fn new(len: usize) -> Self {
        let _ = len;
        todo!("Blake3Bytes::new")
    }

    /// Read into `buffer` starting at `start`; returns bytes read.
    pub fn read(&self, start: usize, buffer: &mut [u8]) -> usize {
        let _ = (start, buffer);
        todo!("Blake3Bytes::read")
    }

    /// Write from `buffer` starting at `start`; returns bytes written.
    pub fn write(&mut self, start: usize, buffer: &[u8]) -> usize {
        let _ = (start, buffer);
        todo!("Blake3Bytes::write")
    }

    /// Number of bytes held.
    pub fn len(&self) -> usize {
        todo!("Blake3Bytes::len")
    }

    /// Is the length zero?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Change the length, zero-filling growth and truncating shrinkage.
    pub fn resize(&mut self, new_len: usize) {
        let _ = new_len;
        todo!("Blake3Bytes::resize")
    }

    /// Overwrite the entire contents.
    pub fn set(&mut self, buffer: &[u8]) {
        self.resize(buffer.len());
        self.write(0, buffer);
    }
}

impl Blake3Bytes<Normal> {
    /// Normal-mode hash of this value. MUST equal `hash_value(self.as_bytes())`.
    pub fn hash(&self) -> Hash {
        hash_value(self.as_bytes())
    }

    /// Borrow the backing bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.repr.bytes
    }
}

// ---------------------------------------------------------------------------------------
// Property tests — the executable spec (see data/docs/blake3-bytes-hashing.md).
// ---------------------------------------------------------------------------------------
//
// Two groups:
//   * Group A (RUN NOW): pure-BLAKE3 assumption tests. They prove the foundations the
//     design leans on and double as a reference implementation of the canonical
//     decomposition the implementer must encode in `prove`/`verify_root`.
//   * Group B (`#[ignore]` until implemented): contract tests against the stubbed
//     functions. Remove `#[ignore]` as each piece lands. Each is tagged with the
//     invariant(s) it pins.
//
// Group B deliberately mixes POSITIVE and NEGATIVE tests, and that pairing is the point:
// a positive-only suite is satisfied by a `verify_root` that accepts anything, and a
// negative-only suite by one that rejects everything. The positives (`*_reproduces_hash`,
// `accessed_ranges_read_back`, `*_roundtrips`) pin that honest proofs verify to the right
// root and honest reads succeed; the negatives (`tampered_*`, `*_tamper_detected`,
// `*_faults`, `*_rejected`, `distinct_values_distinct_roots`) pin that mutated proofs,
// mutated content, and out-of-range/unproven reads are caught. Together they leave little
// room for a wrong-but-passing implementation: any single mutation to an honest proof
// (present bytes, blinded CV, length, or shape) must either error or change the root, and
// no read may return data that was not proven. When adding an implementation, do NOT
// weaken a negative test to make it pass — fix the implementation.
#[cfg(test)]
mod tests {
    use blake3::CHUNK_LEN;
    use blake3::hazmat::HasherExt;
    use blake3::hazmat::Mode as B3Mode;
    use blake3::hazmat::left_subtree_len;
    use blake3::hazmat::max_subtree_len;
    use blake3::hazmat::merge_subtrees_non_root;
    use blake3::hazmat::merge_subtrees_root;
    use proptest::prelude::*;

    use super::*;

    // -- Reference canonical decomposition (the algorithm `prove`/`verify_root` mirror) --

    /// Chaining value of the canonical subtree covering `input`, which *starts* at byte
    /// `offset` (a chunk boundary). Non-root.
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

    /// Root of the canonical BLAKE3 tree of `input`. Reference for `verify_root`.
    fn ref_root(input: &[u8]) -> [u8; 32] {
        if input.len() <= CHUNK_LEN {
            // A single chunk finalized with ROOT is exactly `blake3::hash`.
            return *blake3::hash(input).as_bytes();
        }
        let left_len = left_subtree_len(input.len() as u64) as usize;
        let (left, right) = input.split_at(left_len);
        let left_cv = subtree_cv(left, 0);
        let right_cv = subtree_cv(right, left_len as u64);
        *merge_subtrees_root(&left_cv, &right_cv, B3Mode::Hash).as_bytes()
    }

    /// Reconstruct the root but with the whole left subtree replaced by its precomputed
    /// (blinded) CV — models a blinded subtree. Must still equal `blake3::hash`.
    fn root_with_left_blinded(input: &[u8]) -> [u8; 32] {
        assert!(input.len() > CHUNK_LEN);
        let left_len = left_subtree_len(input.len() as u64) as usize;
        let (left, right) = input.split_at(left_len);
        let blinded_left = subtree_cv(left, 0); // precomputed, "blinded"
        let right_cv = subtree_cv(right, left_len as u64);
        *merge_subtrees_root(&blinded_left, &right_cv, B3Mode::Hash).as_bytes()
    }

    // Sizes chosen to exercise: empty, sub-chunk, exact chunk, chunk+1, multi-chunk,
    // exact power-of-two chunk counts, and non-power-of-two right spines.
    fn interesting_len() -> impl Strategy<Value = usize> {
        prop_oneof![
            Just(0usize),
            1usize..CHUNK_LEN,
            Just(CHUNK_LEN),
            Just(CHUNK_LEN + 1),
            Just(2 * CHUNK_LEN),
            Just(3 * CHUNK_LEN),      // right spine is a single chunk
            Just(4 * CHUNK_LEN),      // perfect power of two
            Just(5 * CHUNK_LEN + 7),  // ragged multi-level
            (0usize..=8 * CHUNK_LEN), // fuzz across levels
        ]
    }

    fn bytes_of_len(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(any::<u8>(), len..=len)
    }

    // ------------------------------- Group A: RUN NOW -------------------------------

    proptest! {
        /// Foundation for P0/I1: the canonical hazmat decomposition equals `blake3::hash`.
        #[test]
        fn hazmat_root_agrees_with_blake3_hash(len in interesting_len()) {
            let data = vec![0xA5u8; len];
            prop_assert_eq!(ref_root(&data), *blake3::hash(&data).as_bytes());
        }

        /// P3/P1: distinct lengths (via truncation) yield distinct hashes.
        #[test]
        fn truncation_changes_hash(data in bytes_of_len(2 * CHUNK_LEN), cut in 1usize..2 * CHUNK_LEN) {
            let short = &data[..data.len() - cut];
            prop_assert_ne!(blake3::hash(&data), blake3::hash(short));
        }

        /// P3/P1: appending changes the hash.
        #[test]
        fn extension_changes_hash(data in bytes_of_len(CHUNK_LEN + 5), extra in 1u8..=255) {
            let mut longer = data.clone();
            longer.push(extra);
            prop_assert_ne!(blake3::hash(&data), blake3::hash(&longer));
        }

        /// P5: blinding an aligned subtree (replacing it by its CV) preserves the root.
        #[test]
        fn blinding_a_subtree_preserves_root(len in (CHUNK_LEN + 1)..=8 * CHUNK_LEN) {
            let data = vec![0x5Au8; len];
            prop_assert_eq!(root_with_left_blinded(&data), *blake3::hash(&data).as_bytes());
        }

        /// I5: every canonical left-subtree offset/span is a legal subtree.
        #[test]
        fn canonical_left_subtree_is_legal(len in (CHUNK_LEN + 1)..=16 * CHUNK_LEN) {
            let left_len = left_subtree_len(len as u64);
            // The left subtree starts at offset 0 (unbounded) and the right subtree
            // starts at `left_len`, whose max legal span must cover the remainder-or-more.
            let right_offset = left_len;
            let right_span = len as u64 - left_len;
            if let Some(max) = max_subtree_len(right_offset) {
                // The right child may itself split; its *first* legal subtree must not be
                // larger than `max`. This just checks the offset is a lawful boundary.
                prop_assert!(right_span == 0 || max >= CHUNK_LEN as u64);
            }
        }
    }

    // ---------------------- Group B: pending implementation ----------------------

    fn arb_data_and_access() -> impl Strategy<Value = (Vec<u8>, Vec<Range<usize>>)> {
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
        /// I8/P0: Normal-mode hash equals `blake3::hash`.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn hash_value_equals_blake3(data in bytes_of_len(3 * CHUNK_LEN + 11)) {
            let expected = Hash::hash_bytes(&data);
            prop_assert_eq!(hash_value(&data), expected);
        }

        /// I8/P0: prove-then-verify reproduces the Normal-mode root, for any access set.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn prove_then_verify_reproduces_hash((data, access) in arb_data_and_access()) {
            let proof = prove(&data, &access);
            prop_assert_eq!(verify_root(&proof).unwrap(), hash_value(&data));
        }

        /// P3/I2/I3: tampering with the claimed length makes verification reject or
        /// diverge from the committed root.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn tampered_length_rejected((data, access) in arb_data_and_access()) {
            let committed = hash_value(&data);
            let mut proof = prove(&data, &access);
            proof.total_len = proof.total_len.wrapping_add(CHUNK_LEN);
            match verify_root(&proof) {
                Err(_) => {}
                Ok(root) => prop_assert_ne!(root, committed),
            }
        }

        /// I4/P5: reading a range that was never accessed (and is blinded) faults.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn blinded_read_faults(data in bytes_of_len(4 * CHUNK_LEN)) {
            // Access only the first chunk; the rest is blinded.
            let proof = prove(&data, &[0..1]);
            let err = proof_read(&proof, (3 * CHUNK_LEN)..(4 * CHUNK_LEN));
            prop_assert!(matches!(err, Err(ProofError::Blinded)));
        }

        /// P0/P5: accessed ranges are present and read back exactly.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn accessed_ranges_read_back((data, access) in arb_data_and_access()) {
            prop_assume!(!access.is_empty());
            let proof = prove(&data, &access);
            for r in &access {
                let got = proof_read(&proof, r.clone()).unwrap();
                prop_assert_eq!(got.as_slice(), &data[r.clone()]);
            }
        }

        /// P5: substituting a blinded CV changes the recomputed root (or is rejected).
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn tampered_blind_cv_detected(data in bytes_of_len(4 * CHUNK_LEN)) {
            let committed = hash_value(&data);
            let mut proof = prove(&data, &[0..1]);
            // Flip a bit in the first blinded CV we can find.
            fn tamper(t: &mut ProofTree) -> bool {
                match t {
                    ProofTree::Blind(cv) => { cv[0] ^= 1; true }
                    ProofTree::Node(l, r) => tamper(l) || tamper(r),
                    ProofTree::Chunk(_) => false,
                }
            }
            prop_assume!(tamper(&mut proof.root));
            match verify_root(&proof) {
                Err(_) => {}
                Ok(root) => prop_assert_ne!(root, committed),
            }
        }

        /// I6: a present chunk larger than CHUNK_LEN must be rejected (memory safety).
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn oversized_chunk_rejected(extra in 1usize..64) {
            let proof = Blake3Proof {
                total_len: CHUNK_LEN + extra,
                root: ProofTree::Chunk(vec![0u8; CHUNK_LEN + extra]),
            };
            prop_assert_eq!(verify_root(&proof), Err(ProofError::ChunkTooLarge));
        }

        /// P0/P1: tampering with PRESENT chunk bytes changes the recomputed root (or is
        /// rejected). This is the guard against a `verify_root` that never actually hashes
        /// present chunk contents — such an implementation would pass every *other* test
        /// (round-trip, blinded-CV tamper, length tamper) while being completely wrong.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn present_chunk_tamper_detected(data in bytes_of_len(4 * CHUNK_LEN)) {
            let committed = hash_value(&data);
            let mut proof = prove(&data, &[0..data.len()]); // every chunk present
            fn flip_first_chunk(t: &mut ProofTree) -> bool {
                match t {
                    ProofTree::Chunk(b) if !b.is_empty() => { b[0] ^= 1; true }
                    ProofTree::Chunk(_) => false,
                    ProofTree::Node(l, r) => flip_first_chunk(l) || flip_first_chunk(r),
                    ProofTree::Blind(_) => false,
                }
            }
            prop_assume!(flip_first_chunk(&mut proof.root));
            match verify_root(&proof) {
                Err(_) => {}
                Ok(root) => prop_assert_ne!(root, committed),
            }
        }

        /// I4/P5: reading ANY range wholly outside the accessed set faults with `Blinded`
        /// — the property-level generalisation of `blinded_read_faults`. Guards against a
        /// verify that silently invents data (e.g. zero-fills) for unproven regions.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn unaccessed_read_faults(a in (2 * CHUNK_LEN)..(8 * CHUNK_LEN), len in 1usize..=CHUNK_LEN) {
            let data = vec![7u8; 8 * CHUNK_LEN];
            let proof = prove(&data, &[0..1]); // only chunk 0 present; rest blinded
            let end = (a + len).min(data.len());
            prop_assume!(end > a);
            prop_assert!(matches!(proof_read(&proof, a..end), Err(ProofError::Blinded)));
        }

        /// I4: reading past `total_len` faults with `OutOfBounds`; never returns padding.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn out_of_bounds_read_faults(over in 1usize..64) {
            let data = vec![3u8; 2 * CHUNK_LEN];
            let proof = prove(&data, &[0..data.len()]);
            let start = data.len();
            prop_assert!(matches!(
                proof_read(&proof, start..(start + over)),
                Err(ProofError::OutOfBounds)
            ));
        }

        /// P1 anti-collapse: two different values of the same length verify to DIFFERENT
        /// roots, each equal to its own `hash_value`. Guards against a `verify_root` that
        /// returns a constant, echoes `total_len`, or otherwise ignores content.
        #[test]
        #[ignore = "pending blake3 scheme implementation"]
        fn distinct_values_distinct_roots(
            data in bytes_of_len(2 * CHUNK_LEN + 5),
            idx in 0usize..(2 * CHUNK_LEN + 5),
        ) {
            let a = data.clone();
            let mut b = data;
            b[idx] ^= 0xFF; // one byte differs => different value, identical length
            let ra = verify_root(&prove(&a, &[0..a.len()])).unwrap();
            let rb = verify_root(&prove(&b, &[0..b.len()])).unwrap();
            prop_assert_eq!(ra, hash_value(&a));
            prop_assert_eq!(rb, hash_value(&b));
            prop_assert_ne!(ra, rb);
        }
    }

    /// I2: a structurally non-canonical proof is rejected. A two-chunk value's canonical
    /// shape is `Node(Chunk, Chunk)`; a bare single `Chunk` claiming `total_len =
    /// 2*CHUNK_LEN` is non-canonical.
    #[test]
    #[ignore = "pending blake3 scheme implementation"]
    fn non_canonical_shape_rejected() {
        let proof = Blake3Proof {
            total_len: 2 * CHUNK_LEN,
            root: ProofTree::Chunk(vec![0u8; CHUNK_LEN]),
        };
        assert_eq!(verify_root(&proof), Err(ProofError::NonCanonicalShape));
    }

    /// I8/P0 edge case: the empty value hashes like `blake3::hash(&[])` and its proof
    /// round-trips.
    #[test]
    #[ignore = "pending blake3 scheme implementation"]
    fn empty_value_roundtrips() {
        let data: &[u8] = &[];
        let proof = prove(data, &[]);
        assert_eq!(verify_root(&proof).unwrap(), hash_value(data));
        assert_eq!(hash_value(data), Hash::hash_bytes(data));
    }
}
