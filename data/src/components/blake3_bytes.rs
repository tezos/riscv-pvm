// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Length-committed BLAKE3 hashing scheme for a byte array.
//!
//! A value's hash is `H(v) = combine(H_len, blake3::hash(bytes))` — the length committed as its
//! own leaf, combined with the BLAKE3 hash of the raw bytes. Within-value partial proofs ride
//! BLAKE3's own internal 1024-byte chunk tree via the [`blake3::hazmat`] guts API (over the
//! `blake3::hash(bytes)` part), with *unrestricted* blinding: because the committed `H_len` pins
//! the length, and hence the canonical shape, any untouched subtree collapses to a single
//! chaining value. The full design, the security argument, and the numbered invariants
//! (`I1`..`I8`) are in `docs/state-framework/blake3-bytes.mdx`.
//!
//! ## The audited free functions
//!
//! - [`hash_value`] — the committed value hash (invariants I8, S0).
//! - [`prove`] — build a [`Blake3Proof`] capturing accessed byte ranges, blinding every
//!   untouched canonical subtree (I1, I6).
//! - [`verify_root`] — recompute the committed value hash canonically from the claimed length,
//!   rejecting non-canonical shapes and oversized chunks (I2, I3, I5).
//!
//! The proof also has its own self-contained wire format ([`Blake3Proof`]'s [`Encode`] and
//! [`Decode`]), kept structurally disjoint from the generic Merkle-proof encoding (I7).
//!
//! ## The component
//!
//! [`Blake3Bytes`] is the mode-generic byte-array component built on those functions. Normal
//! mode holds the bytes; Prove mode borrows the pre-transition bytes and records what the
//! transition touched; Verify mode holds the sparse view a proof reconstructs, faulting on a
//! read it cannot answer. The folds that carry it through the state pipelines follow.

use std::borrow::Borrow;
use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Index;
use std::ops::Range;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use blake3::hazmat::HasherExt;
use blake3::hazmat::Mode as B3Mode;
use blake3::hazmat::left_subtree_len;
use blake3::hazmat::merge_subtrees_non_root;
use blake3::hazmat::merge_subtrees_root;
use perfect_derive::perfect_derive;

use crate::clone::CloneState;
use crate::hash::Hash;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::Source;
use crate::mode::utils::not_found;
use crate::partial_vec::PartialVec;

/// BLAKE3 chunk length: the leaf granularity of the internal chunk tree (1024 bytes).
pub const CHUNK_LEN: usize = blake3::CHUNK_LEN;

/// Commit a value's length as its own leaf: `H_len = blake3(len_le_bytes)`.
///
/// The length is hashed as its `u64` little-endian encoding (no bincode framing) so it is a
/// fixed, consensus-relevant preimage. This is the leaf that pins the value's length in the
/// committed hash (see [`hash_value`]); dropping it is what let the predecessor length-free
/// scheme be forged by a cross-span graft (invariant I3).
fn hash_len(len: usize) -> Hash {
    Hash::hash_bytes(&(len as u64).to_le_bytes())
}

/// Normal-mode hash of a byte array under the length-committed BLAKE3 scheme.
///
/// `H(v) = combine(H_len, blake3::hash(bytes))` — the length leaf [`hash_len`] combined with
/// the BLAKE3 hash of the raw bytes (invariant I8). The explicit length commitment pins the
/// value's length, which is what makes within-value partial proofs with *unrestricted* blinding
/// secure. This is **not** equal to `blake3::hash(bytes)`: it wraps it in one more combine.
pub fn hash_value(bytes: &[u8]) -> Hash {
    Hash::combine_hashes([hash_len(bytes.len()), Hash::hash_bytes(bytes)])
}

// ---------------------------------------------------------------------------------------
// Within-value proof representation
// ---------------------------------------------------------------------------------------

/// A within-value proof: a claimed length plus a partial view of BLAKE3's internal binary
/// chunk tree over the value bytes.
///
/// The committed value hash is `combine(H_len, H_data)` where `H_len = hash_len(total_len)`
/// and `H_data` is the BLAKE3 root of the bytes (see [`hash_value`]). This proof carries the
/// length explicitly (as its own leaf, [`Blake3Proof::total_len`]) and a [`ProofTree`] for the
/// `H_data` chunk tree, in which any untouched canonical subtree collapses to a single blinded
/// chaining value.
///
/// The [`data`](Blake3Proof::data) shape is the *canonical* BLAKE3 decomposition for
/// `total_len`: splits follow [`blake3::hazmat::left_subtree_len`], leaf spans are 1024-byte
/// chunks, and the single top node is finalized with the `ROOT` flag. Verify reconstructs the
/// shape from `total_len` and rejects any proof whose structure deviates (invariant I2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blake3Proof {
    /// Claimed total byte length of the value — the length leaf. It is authenticated by being
    /// hashed into `H_len` and folded into the committed value hash by [`verify_root`]
    /// (invariant I3): a lie about `total_len` changes `H_len` and so the recomputed value
    /// hash, and is rejected. It is never trusted on its own.
    pub total_len: usize,

    /// Partial view of the BLAKE3 chunk tree for `blake3::hash(bytes)` (the `H_data` leaf of
    /// the committed value hash). A top-level [`ProofTree::Blind`] means the whole value was
    /// untouched (only the length was read) and carries `H_data` directly.
    pub data: ProofTree,
}

/// A node in a [`Blake3Proof`]'s partial chunk tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofTree {
    /// A present chunk carrying its real bytes. Length MUST be `<= CHUNK_LEN` (invariant
    /// I6). Its chaining value is recomputed via
    /// `Hasher::set_input_offset(chunk_index * CHUNK_LEN).update(bytes).finalize_non_root()`.
    Chunk(Vec<u8>),

    /// An untouched canonical subtree, collapsed to a single 32-byte chaining value.
    ///
    /// Blinding is *unrestricted*: because `total_len` pins the canonical shape, every node's
    /// `(offset, span)` is fixed, so any untouched subtree — at any offset (including 0) and
    /// any span (including the ragged right spine and partial final chunks) — may be blinded.
    /// No maximal/interior/spine rule is needed; the length commitment (`H_len`) provides the
    /// security instead (invariant I6).
    ///
    /// A **nested** blind (inside a [`ProofTree::Node`]) is the *non-root* chaining value of
    /// its subtree; a **top-level** blind (the whole [`Blake3Proof::data`]) is `H_data` itself
    /// (a `ROOT` output). Verify distinguishes them by position.
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
    /// A present chunk exceeds `CHUNK_LEN`.
    ChunkTooLarge,
    /// The recomputed root does not equal the committed root.
    RootMismatch,
    /// The requested read touches a blinded/absent region.
    Blinded,
    /// The requested range is outside `0..total_len`.
    OutOfBounds,
}

/// Build a proof for `data` that keeps exactly the chunks overlapping `accessed` present
/// and blinds every other canonical subtree to its chaining value.
///
/// Blinding is recursive and *unrestricted* (invariants I1, I6): an untouched canonical subtree
/// of many chunks — at any offset or span — collapses to a single 32-byte CV. If nothing was
/// accessed, the whole data tree collapses to a single top-level [`ProofTree::Blind`] carrying
/// `H_data` (used when only the value's length was read; the length still travels in
/// [`Blake3Proof::total_len`]).
pub fn prove(data: &[u8], accessed: &[Range<usize>]) -> Blake3Proof {
    Blake3Proof {
        total_len: data.len(),
        data: build_proof_tree(data, 0, data.len(), accessed, true),
    }
}

/// Non-root chaining value of the canonical BLAKE3 subtree covering `data`, which *starts*
/// at byte `offset` (a chunk boundary). This is the reference decomposition from the docs:
/// split at [`left_subtree_len`], recurse, and merge non-root; a lone chunk finalises
/// non-root at its input offset.
fn canonical_cv(data: &[u8], offset: u64) -> [u8; 32] {
    if data.len() <= CHUNK_LEN {
        return blake3::Hasher::new()
            .set_input_offset(offset)
            .update(data)
            .finalize_non_root();
    }

    let left_len = left_subtree_len(data.len() as u64) as usize;
    let (left, right) = data.split_at(left_len);
    let left_cv = canonical_cv(left, offset);
    let right_cv = canonical_cv(right, offset + left_len as u64);
    merge_subtrees_non_root(&left_cv, &right_cv, B3Mode::Hash)
}

/// Does any accessed range overlap the span `[offset, offset + len)`?
fn range_is_accessed(offset: usize, len: usize, accessed: &[Range<usize>]) -> bool {
    let end = offset + len;
    accessed.iter().any(|r| r.start < end && offset < r.end)
}

/// Recursively build the proof tree for the subtree at `[offset, offset + len)`.
///
/// An untouched subtree collapses to a single blinded chaining value — with **no**
/// blindability restriction, because `total_len` (committed via `H_len`) pins the shape
/// (invariants I1, I6). At the top (`is_root`), an untouched whole value collapses to
/// `Blind(H_data)` — `blake3::hash(data)`, a `ROOT` output; a nested untouched subtree
/// collapses to `Blind(canonical_cv(...))`, a non-root chaining value. Accessed regions
/// descend to present [`ProofTree::Chunk`] leaves.
///
/// `is_root` is "the whole value was untouched by byte accesses": if the transition only read
/// the length, `accessed` is empty and the whole data tree becomes `Blind(H_data)`.
fn build_proof_tree(
    data: &[u8],
    offset: usize,
    len: usize,
    accessed: &[Range<usize>],
    is_root: bool,
) -> ProofTree {
    let touched = if is_root {
        !accessed.is_empty()
    } else {
        range_is_accessed(offset, len, accessed)
    };

    if !touched {
        // Collapse the whole untouched subtree to one chaining value (invariants I1, I6).
        return if is_root {
            // The whole value is untouched: its `ROOT` output, used directly as `H_data`.
            ProofTree::Blind(*blake3::hash(data).as_bytes())
        } else {
            ProofTree::Blind(canonical_cv(&data[offset..offset + len], offset as u64))
        };
    }

    if len <= CHUNK_LEN {
        ProofTree::Chunk(data[offset..offset + len].to_vec())
    } else {
        let left_len = left_subtree_len(len as u64) as usize;
        let left = build_proof_tree(data, offset, left_len, accessed, false);
        let right = build_proof_tree(data, offset + left_len, len - left_len, accessed, false);
        ProofTree::Node(Box::new(left), Box::new(right))
    }
}

/// Recompute the committed value hash from `proof`.
///
/// Returns `combine(H_len, H_data)` where `H_len = hash_len(proof.total_len)` and `H_data` is
/// the BLAKE3 root reconstructed from the [`data`](Blake3Proof::data) tree (invariant I3). The
/// caller compares the result to the committed value hash: a lie about `total_len` changes
/// `H_len` (hence the returned hash) and a wrong present chunk / blinded CV changes `H_data`,
/// so neither is accepted absent a BLAKE3 collision. For an honest proof of `data`, this equals `hash_value(data)` (invariant I8).
pub fn verify_root(proof: &Blake3Proof) -> Result<Hash, ProofError> {
    let h_data = data_root(&proof.data, proof.total_len)?;
    Ok(Hash::combine_hashes([
        hash_len(proof.total_len),
        Hash::from(h_data),
    ]))
}

/// Recompute `H_data` — the `ROOT`-finalised BLAKE3 root of the value bytes — from the data
/// tree, reconstructing the shape canonically from `length` (invariants I2, I5).
///
/// A top-level [`ProofTree::Blind`] carries `H_data` directly (the whole value was untouched).
/// A single [`ProofTree::Chunk`] (`length <= CHUNK_LEN`) is finalised with `ROOT`
/// (= `blake3::hash`). A [`ProofTree::Node`] merges its children's non-root chaining values
/// with `merge_subtrees_root`.
fn data_root(tree: &ProofTree, length: usize) -> Result<[u8; 32], ProofError> {
    match tree {
        // Whole value untouched: the blind IS `H_data` (a ROOT output), used as-is. Any lie
        // is caught by the outer `combine(H_len, H_data)` equality check in `verify_root`.
        ProofTree::Blind(h_data) => Ok(*h_data),

        // A value of at most one chunk hashes as `blake3::hash(bytes)` (the chunk finalised
        // with the ROOT flag), so the bytes must be present.
        ProofTree::Chunk(bytes) => {
            if bytes.len() > CHUNK_LEN {
                return Err(ProofError::ChunkTooLarge);
            }
            if length > CHUNK_LEN || bytes.len() != length {
                return Err(ProofError::NonCanonicalShape);
            }
            Ok(*blake3::hash(bytes).as_bytes())
        }

        ProofTree::Node(left, right) => {
            if length <= CHUNK_LEN {
                return Err(ProofError::NonCanonicalShape);
            }
            let left_len = left_subtree_len(length as u64) as usize;
            let left_cv = compute_cv(left, 0, left_len)?;
            let right_cv = compute_cv(right, left_len as u64, length - left_len)?;
            Ok(*merge_subtrees_root(&left_cv, &right_cv, B3Mode::Hash).as_bytes())
        }
    }
}

/// Recompute the non-root chaining value of the subtree `tree` is claimed to represent,
/// which canonically covers `[offset, offset + len)`.
///
/// The tree shape is reconstructed from `len` alone (invariant I2): a [`ProofTree::Node`] is
/// only legal where the canonical decomposition has an internal node (`len > CHUNK_LEN`) and
/// splits at [`left_subtree_len`]; a [`ProofTree::Chunk`] is only legal where it has a leaf
/// (`len <= CHUNK_LEN`) and must carry exactly `len` bytes. A [`ProofTree::Blind`] needs **no**
/// blindability check: its span is fixed by the canonical recursion, and the length commitment
/// (`H_len`) — not a span/offset gate — is what prevents forgery, so any wrong CV changes
/// `H_data` (invariant I6).
fn compute_cv(tree: &ProofTree, offset: u64, len: usize) -> Result<[u8; 32], ProofError> {
    match tree {
        ProofTree::Chunk(bytes) => {
            if bytes.len() > CHUNK_LEN {
                return Err(ProofError::ChunkTooLarge);
            }
            if len > CHUNK_LEN || bytes.len() != len {
                return Err(ProofError::NonCanonicalShape);
            }
            Ok(blake3::Hasher::new()
                .set_input_offset(offset)
                .update(bytes)
                .finalize_non_root())
        }

        // The span is pinned by the canonical shape; no blindability check is needed (I6).
        ProofTree::Blind(cv) => Ok(*cv),

        ProofTree::Node(left, right) => {
            if len <= CHUNK_LEN {
                return Err(ProofError::NonCanonicalShape);
            }
            let left_len = left_subtree_len(len as u64) as usize;
            let left_cv = compute_cv(left, offset, left_len)?;
            let right_cv = compute_cv(right, offset + left_len as u64, len - left_len)?;
            Ok(merge_subtrees_non_root(&left_cv, &right_cv, B3Mode::Hash))
        }
    }
}

/// Collect the absolute `(offset, bytes)` of every present chunk leaf, descending the
/// canonical shape derived from `len`.
fn collect_present(tree: &ProofTree, offset: usize, len: usize, out: &mut Vec<(usize, Vec<u8>)>) {
    match tree {
        ProofTree::Chunk(bytes) => out.push((offset, bytes.clone())),
        ProofTree::Blind(_) => {}
        ProofTree::Node(left, right) => {
            let left_len = left_subtree_len(len as u64) as usize;
            collect_present(left, offset, left_len, out);
            collect_present(right, offset + left_len, len - left_len, out);
        }
    }
}

/// Read `range` from a verified proof.
///
/// Returns the bytes if `range` lies entirely within present chunks; faults with
/// [`ProofError::Blinded`] if any part was blinded, or [`ProofError::OutOfBounds`] if
/// `range` exceeds `total_len` (invariant I4). This never returns zeroes for absent data.
pub fn proof_read(proof: &Blake3Proof, range: Range<usize>) -> Result<Vec<u8>, ProofError> {
    if range.start > range.end || range.end > proof.total_len {
        return Err(ProofError::OutOfBounds);
    }
    if range.is_empty() {
        return Ok(Vec::new());
    }

    let mut present = Vec::new();
    collect_present(&proof.data, 0, proof.total_len, &mut present);

    let mut out = vec![0u8; range.len()];
    let mut covered = vec![false; range.len()];

    for (chunk_offset, bytes) in &present {
        let seg_start = (*chunk_offset).max(range.start);
        let seg_end = (chunk_offset + bytes.len()).min(range.end);
        for pos in seg_start..seg_end {
            out[pos - range.start] = bytes[pos - chunk_offset];
            covered[pos - range.start] = true;
        }
    }

    if covered.iter().all(|&c| c) {
        Ok(out)
    } else {
        // Any byte of the requested range fell in a blinded/absent subtree.
        Err(ProofError::Blinded)
    }
}

/// The byte ranges materialized (present) in `proof`, in ascending order. Reading any
/// sub-range of these via [`proof_read`] succeeds; reading outside them faults.
pub fn present_ranges(proof: &Blake3Proof) -> Vec<Range<usize>> {
    let mut present = Vec::new();
    collect_present(&proof.data, 0, proof.total_len, &mut present);
    present.sort_by_key(|(offset, _)| *offset);

    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (offset, bytes) in present {
        let start = offset;
        let end = offset + bytes.len();
        if start == end {
            continue;
        }
        match ranges.last_mut() {
            Some(last) if last.end == start => last.end = end,
            _ => ranges.push(start..end),
        }
    }
    ranges
}

// ---------------------------------------------------------------------------------------
// Proof serialisation (within-value proof wire format)
// ---------------------------------------------------------------------------------------
//
// The value proof is encoded structurally disjointly from the AVL-node encoding (invariant
// I7): it is a self-contained format, decided by the parse position (only the `data` slot of
// an AVL node is ever parsed as a `Blake3Proof`). It is *never* merged into the generic
// `crate::merkle_proof` Node/Read/Blind wire, whose node-combine (`combine_hashes`) and
// leaf-hash (`hash_bytes`) do not match BLAKE3's internal `merge_subtrees` / raw-`blake3::hash`
// scheme. See the module docs, and invariant I7 in `docs/state-framework/blake3-bytes.mdx`.

/// Wire tag for a [`ProofTree::Chunk`] node.
const TREE_TAG_CHUNK: u8 = 0;

/// Wire tag for a [`ProofTree::Blind`] node.
const TREE_TAG_BLIND: u8 = 1;

/// Wire tag for a [`ProofTree::Node`] node.
const TREE_TAG_NODE: u8 = 2;

impl Encode for ProofTree {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // Encoded iteratively to avoid unbounded recursion on adversarial depth.
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            match node {
                ProofTree::Chunk(bytes) => {
                    TREE_TAG_CHUNK.encode(encoder)?;
                    // Length-prefixed so the decoder can bound-check before allocating (I6).
                    (bytes.len() as u64).encode(encoder)?;
                    encoder.writer().write(bytes)?;
                }
                ProofTree::Blind(cv) => {
                    TREE_TAG_BLIND.encode(encoder)?;
                    encoder.writer().write(cv)?;
                }
                ProofTree::Node(left, right) => {
                    TREE_TAG_NODE.encode(encoder)?;
                    // Push right first so left is popped (and thus decoded) first.
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        Ok(())
    }
}

/// Maximum `ProofTree` nesting the decoder accepts. The canonical BLAKE3 tree for a value of up to
/// `usize::MAX` (≤ `u64::MAX`) bytes has depth `ceil(log2(ceil(len / CHUNK_LEN))) <= 54`, so this
/// bound never rejects a well-formed proof. It exists solely to stop an adversarially deeply-nested
/// decoded proof from overflowing the verifier's stack (a DoS) before the shape is validated: any
/// tree this deep is non-canonical and would be rejected by [`verify_root`] anyway. This guards
/// both the standalone `Blake3Proof` wire and the AVL-node `Blake3` leaf (which decodes via the
/// same path).
const MAX_PROOF_TREE_DEPTH: usize = 64;

impl<Context> Decode<Context> for ProofTree {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        decode_proof_tree(decoder, 0)
    }
}

/// Depth-bounded recursive decode of a [`ProofTree`] (see [`MAX_PROOF_TREE_DEPTH`]).
fn decode_proof_tree<Context, D: Decoder<Context = Context>>(
    decoder: &mut D,
    depth: usize,
) -> Result<ProofTree, DecodeError> {
    if depth > MAX_PROOF_TREE_DEPTH {
        return Err(DecodeError::OtherString(format!(
            "BLAKE3 proof-tree nesting exceeds max depth {MAX_PROOF_TREE_DEPTH}"
        )));
    }
    let tag = u8::decode(decoder)?;
    match tag {
        TREE_TAG_CHUNK => {
            let len = u64::decode(decoder)?;
            // Bound-check the present-chunk length before allocating (invariant I6).
            if len > CHUNK_LEN as u64 {
                return Err(DecodeError::OtherString(format!(
                    "BLAKE3 proof chunk length {len} exceeds CHUNK_LEN {CHUNK_LEN}"
                )));
            }
            let mut bytes = vec![0u8; len as usize];
            decoder.reader().read(&mut bytes)?;
            Ok(ProofTree::Chunk(bytes))
        }
        TREE_TAG_BLIND => {
            let mut cv = [0u8; 32];
            decoder.reader().read(&mut cv)?;
            Ok(ProofTree::Blind(cv))
        }
        TREE_TAG_NODE => {
            let left = decode_proof_tree(decoder, depth + 1)?;
            let right = decode_proof_tree(decoder, depth + 1)?;
            Ok(ProofTree::Node(Box::new(left), Box::new(right)))
        }
        other => Err(DecodeError::OtherString(format!(
            "invalid BLAKE3 proof-tree tag {other}"
        ))),
    }
}

impl Encode for Blake3Proof {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        (self.total_len as u64).encode(encoder)?;
        self.data.encode(encoder)
    }
}

impl<Context> Decode<Context> for Blake3Proof {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let total_len = u64::decode(decoder)? as usize;
        let data = ProofTree::decode(decoder)?;
        Ok(Blake3Proof { total_len, data })
    }
}

// ---------------------------------------------------------------------------------------
// Mode-generic component
// ---------------------------------------------------------------------------------------

/// Modal template for [`Blake3Bytes`].
pub enum Blake3BytesTemplate {}

impl Modal for Blake3BytesTemplate {
    type Normal = NormalRepr;
    type Prove<'normal> = ProveRepr<'normal>;
    type Verify = VerifyRepr;
}

/// Normal-mode representation: the full byte array.
#[derive(Clone, Debug, Default)]
pub struct NormalRepr {
    /// Backing bytes.
    pub bytes: Vec<u8>,
}

impl Borrow<[u8]> for NormalRepr {
    fn borrow(&self) -> &[u8] {
        &self.bytes
    }
}

/// Prove-mode representation: borrows (or owns) the pre-transition bytes and records the byte
/// ranges accessed during the transition.
///
/// Mirrors [`super::bytes`]'s `ProveImpl`. The `previous` bytes are the pre-transition state
/// captured for proof generation; `reads`/`writes` record the accessed regions that must stay
/// present in the value proof (docs §4).
#[perfect_derive(Clone, Debug)]
pub struct ProveRepr<'normal> {
    /// Pre-transition source data.
    previous: Source<'normal, Blake3Bytes<Normal>, [u8]>,
    /// Current (post-transition) length.
    length: usize,
    /// Whether the length was read during the transition.
    did_access_length: Cell<bool>,
    /// The lowest length the value ever had during the transition (the "low-water mark"). Bytes
    /// at or above it were dropped by some resize, so if the value later regrew they must read
    /// back as zero rather than as the pre-transition bytes `previous` still holds. Used by
    /// [`Self::post_bytes`] to zero that region; the verifier tracks its own copy (see
    /// [`VerifyRepr::low_water`]).
    low_water: Cell<usize>,
    /// Ranges that were read (may overlap; overlaps are harmless for proof generation).
    reads: RefCell<Vec<Range<usize>>>,
    /// Writes recorded against the pre-transition state.
    writes: PartialVec<u8>,
}

/// Verify-mode representation: a sparse view populated from a [`Blake3Proof`].
///
/// `data` holds the present (proven) byte ranges; reads of absent/blinded regions fault via
/// [`not_found`] (invariant I4). `proof` retains the committed pre-transition proof so the
/// post-transition root can be recomputed after writes.
#[derive(Clone, Debug, Default)]
pub struct VerifyRepr {
    /// Current length, if known.
    length: Option<usize>,
    /// The lowest length the value has had since reconstruction (see [`ProveRepr::low_water`]).
    /// The verifier replays the same resizes as the prover, so the two derive the same mark and
    /// agree on whether the committed shape still describes the value. Meaningless while
    /// `length` is `None` (an absent value).
    low_water: usize,
    /// Present byte ranges recovered from the proof (and any writes).
    data: PartialVec<u8>,
    /// The committed within-value proof, if the value participated in the proof.
    proof: Option<Blake3Proof>,
}

/// Byte-array state component under the BLAKE3-direct hashing scheme.
///
/// Public surface mirrors [`super::bytes::Bytes`]. The value hash is `hash_value(bytes)`
/// (see [`hash_value`]); within-value partial proofs ride BLAKE3's internal chunk tree via
/// [`prove`] / [`verify_root`].
#[perfect_derive(Debug)]
pub struct Blake3Bytes<M: Mode> {
    repr: M::Select<Blake3BytesTemplate>,
}

impl<M: Blake3BytesMode> Clone for Blake3Bytes<M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

impl<M: Blake3BytesMode> Blake3Bytes<M> {
    /// Create a zero-initialised byte array of the given length.
    pub fn new(len: usize) -> Self {
        M::new(len)
    }

    /// Read into `buffer` starting at `start`; returns bytes read.
    pub fn read(&self, start: usize, buffer: &mut [u8]) -> usize {
        M::read(self, start, buffer)
    }

    /// Write from `buffer` starting at `start`; returns bytes written.
    pub fn write(&mut self, start: usize, buffer: &[u8]) -> usize {
        M::write(self, start, buffer)
    }

    /// Number of bytes held.
    pub fn len(&self) -> usize {
        M::len(self)
    }

    /// Is the length zero?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Change the length, zero-filling growth and truncating shrinkage.
    pub fn resize(&mut self, new_len: usize) {
        M::resize(self, new_len)
    }

    /// Overwrite the entire contents.
    pub fn set(&mut self, buffer: &[u8]) {
        self.resize(buffer.len());
        self.write(0, buffer);
    }
}

impl<M: Blake3BytesMode> Default for Blake3Bytes<M> {
    fn default() -> Self {
        M::new(0)
    }
}

impl<M: Blake3BytesMode> CloneState for Blake3Bytes<M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl Blake3Bytes<Normal> {
    /// Normal-mode hash of this value. Equals `hash_value(self.as_bytes())` (invariant I8).
    pub fn hash(&self) -> Hash {
        hash_value(self.as_bytes())
    }

    /// Borrow the backing bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.repr.bytes
    }

    /// Convert this value into [`Prove`] mode, taking ownership of the pre-transition data.
    pub fn into_proof(self) -> Blake3Bytes<Prove<'static>> {
        let length = self.repr.bytes.len();
        Blake3Bytes {
            repr: ProveRepr {
                previous: Source::owned(self),
                length,
                did_access_length: Cell::new(false),
                low_water: Cell::new(length),
                reads: RefCell::new(Vec::new()),
                writes: PartialVec::default(),
            },
        }
    }
}

impl Borrow<[u8]> for Blake3Bytes<Normal> {
    fn borrow(&self) -> &[u8] {
        &self.repr.bytes
    }
}

impl Index<Range<usize>> for Blake3Bytes<Normal> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &[u8] {
        &self.repr.bytes[range]
    }
}

impl From<&[u8]> for Blake3Bytes<Normal> {
    fn from(slice: &[u8]) -> Self {
        Blake3Bytes {
            repr: NormalRepr {
                bytes: slice.to_vec(),
            },
        }
    }
}

impl From<bytes::Bytes> for Blake3Bytes<Normal> {
    fn from(bytes: bytes::Bytes) -> Self {
        Blake3Bytes::from(bytes.as_ref())
    }
}

impl<T: AsRef<[u8]>, M: Blake3BytesMode> PartialEq<T> for Blake3Bytes<M> {
    fn eq(&self, other: &T) -> bool {
        let other = other.as_ref();
        let len = self.len();

        if len != other.len() {
            return false;
        }

        let mut chunk = vec![0u8; 4096];
        for start in (0..len).step_by(chunk.len()) {
            let read = self.read(start, &mut chunk);
            if chunk[..read] != other[start..][..read] {
                return false;
            }
        }

        true
    }
}

impl<M: Blake3BytesMode, N: Blake3BytesMode> PartialEq<Blake3Bytes<N>> for Blake3Bytes<M> {
    fn eq(&self, other: &Blake3Bytes<N>) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        let mut chunk_lhs = vec![0u8; 4096];
        let mut chunk_rhs = chunk_lhs.clone();

        for offset in (0..len).step_by(chunk_lhs.len()) {
            let read = self.read(offset, &mut chunk_lhs);
            if read != other.read(offset, &mut chunk_rhs) {
                return false;
            }
            if chunk_lhs[..read] != chunk_rhs[..read] {
                return false;
            }
        }

        true
    }
}

impl<M: Blake3BytesMode> Eq for Blake3Bytes<M> {}

impl<'normal> Provable<'normal> for Blake3Bytes<Normal> {
    type Prover = Blake3Bytes<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        Blake3Bytes {
            repr: ProveRepr {
                previous: Source::borrowed(&self.repr.bytes),
                length: self.repr.bytes.len(),
                did_access_length: Cell::new(false),
                low_water: Cell::new(self.repr.bytes.len()),
                reads: RefCell::new(Vec::new()),
                writes: PartialVec::default(),
            },
        }
    }
}

impl<'a> Blake3Bytes<Prove<'a>> {
    /// Construct the component in [`Prove`] mode from raw pre-transition source data.
    pub fn from_raw_source(source: &'a [u8]) -> Self {
        Blake3Bytes {
            repr: ProveRepr {
                previous: Source::borrowed(source),
                length: source.len(),
                did_access_length: Cell::new(false),
                low_water: Cell::new(source.len()),
                reads: RefCell::new(Vec::new()),
                writes: PartialVec::default(),
            },
        }
    }

    /// The byte ranges of the pre-transition value the proof has to carry.
    ///
    /// Reads need their bytes so the verifier can return the same ones. Writes need them too: the
    /// verifier recomputes the post-transition root by substituting written bytes into the
    /// committed shape, and a chunk it holds only part of cannot be hashed.
    ///
    /// A length change needs the whole pre-transition value, because the canonical shape is a
    /// function of the length: at the new length the committed chaining values stand for spans
    /// that no longer exist, so none of them can be reused and the root has to be recomputed from
    /// the bytes. That makes a resize cost the size of the value, which a later commit reduces.
    fn accessed_ranges(&self) -> Vec<Range<usize>> {
        let prev_len = self.repr.previous.len();
        let clamp = |start: usize, end: usize| -> Option<Range<usize>> {
            let start = start.min(prev_len);
            let end = end.min(prev_len);
            (start < end).then_some(start..end)
        };

        let mut ranges: Vec<Range<usize>> = self
            .repr
            .reads
            .borrow()
            .iter()
            .filter_map(|r| clamp(r.start, r.end))
            .collect();

        for (offset, bytes) in self.repr.writes.defined_range(0..self.repr.length) {
            if let Some(range) = clamp(offset, offset + bytes.len()) {
                ranges.push(range);
            }
        }

        if self.repr.length != prev_len || self.repr.low_water.get() != self.repr.length {
            ranges.extend(clamp(0, prev_len));
        }

        ranges
    }

    /// Build the within-value [`Blake3Proof`] capturing the pre-transition state, keeping the
    /// accessed ranges present and blinding every untouched canonical subtree.
    pub fn value_proof(&self) -> Blake3Proof {
        prove(&self.repr.previous, &self.accessed_ranges())
    }

    /// Reconstruct the post-transition bytes: the pre-transition data, resized to the current
    /// length, with the recorded writes overlaid.
    ///
    /// Any position at or beyond the low-water mark was dropped by a shrink at some point in the
    /// transition, so if the value later regrew past it those bytes must read back as **zero**,
    /// not the stale pre-transition data that `previous` still holds. For a plain append the
    /// low-water mark is the pre length, so the zero-fill only touches the appended tail, which
    /// is already zero.
    fn post_bytes(&self) -> Vec<u8> {
        let length = self.repr.length;
        let mut data = self.repr.previous.to_vec();
        data.resize(length, 0);
        let low_water = self.repr.low_water.get();
        if low_water < length {
            data[low_water..length].fill(0);
        }
        for (offset, chunk) in self.repr.writes.defined_range(0..length) {
            data[offset..][..chunk.len()].copy_from_slice(chunk);
        }
        data
    }

    /// Hash of the post-transition value.
    pub fn hash(&self) -> Hash {
        hash_value(&self.post_bytes())
    }
}

impl Blake3Bytes<Verify> {
    /// Reconstruct a [`Blake3Bytes<Verify>`] from a within-value proof, after checking the
    /// proof reconstructs to `committed_root`.
    ///
    /// The present chunk bytes populate the sparse view; blinded regions stay absent and
    /// faulting on read (invariant I4). Returns [`ProofError`] if the proof is malformed or
    /// does not reconstruct to `committed_root` (invariants I2, I3).
    pub fn from_proof_checked(
        proof: Blake3Proof,
        committed_root: Hash,
    ) -> Result<Self, ProofError> {
        let root = verify_root(&proof)?;
        if root != committed_root {
            return Err(ProofError::RootMismatch);
        }
        Ok(Self::from_proof_unchecked(proof))
    }

    /// Reconstruct a [`Blake3Bytes<Verify>`] from a proof without checking it against a
    /// committed root. The root check is performed later by the [`PartialHashFold`] fold.
    pub fn from_proof_unchecked(proof: Blake3Proof) -> Self {
        let mut data = PartialVec::empty();
        for range in present_ranges(&proof) {
            let start = range.start;
            let bytes = proof_read(&proof, range).unwrap_or_default();
            if !bytes.is_empty() {
                data.define(start, bytes);
            }
        }

        Blake3Bytes {
            repr: VerifyRepr {
                length: Some(proof.total_len),
                low_water: proof.total_len,
                data,
                proof: Some(proof),
            },
        }
    }

    /// The committed (pre-transition) root recovered from the retained proof, if any.
    pub fn committed_root(&self) -> Option<Hash> {
        self.repr.proof.as_ref().and_then(|p| verify_root(p).ok())
    }

    /// Return the given range as a contiguous byte slice, faulting (via [`not_found`]) if the
    /// range is out of bounds or touches an unproven/blinded region (invariant I4).
    pub fn partial_slice(&self, range: Range<usize>) -> &[u8] {
        if range.is_empty() {
            return &[];
        }
        if range.end > self.len() {
            // SAFETY: called only in `Verify` mode.
            unsafe { not_found() }
        }
        match self.repr.data.contiguous_range(range) {
            Some(slice) => slice,
            // SAFETY: called only in `Verify` mode.
            None => unsafe { not_found() },
        }
    }
}

/// Construct `start..start+len` clamped so it does not extend beyond `total_len`.
fn clamp_range(total_len: usize, start: usize, len: usize) -> Range<usize> {
    let end = start.saturating_add(len).min(total_len);
    start..end
}

// ---------------------------------------------------------------------------------------
// Per-mode operations
// ---------------------------------------------------------------------------------------

/// Mode types that support the common [`Blake3Bytes`] operations.
///
/// Mirrors [`super::bytes::BytesMode`], adapted to the BLAKE3-direct scheme.
pub trait Blake3BytesMode: Mode {
    /// See [`Blake3Bytes::new`].
    fn new(len: usize) -> Blake3Bytes<Self>;
    /// See [`Blake3Bytes::read`].
    fn read(this: &Blake3Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize;
    /// See [`Blake3Bytes::write`].
    fn write(this: &mut Blake3Bytes<Self>, start: usize, buffer: &[u8]) -> usize;
    /// See [`Blake3Bytes::len`].
    fn len(this: &Blake3Bytes<Self>) -> usize;
    /// See [`Blake3Bytes::resize`].
    fn resize(this: &mut Blake3Bytes<Self>, new_len: usize);
    /// Clone the whole component (including mode-specific bookkeeping).
    fn clone(this: &Blake3Bytes<Self>) -> Blake3Bytes<Self>;
}

impl Blake3BytesMode for Normal {
    fn new(len: usize) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: NormalRepr {
                bytes: vec![0u8; len],
            },
        }
    }

    fn read(this: &Blake3Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        if start >= this.repr.bytes.len() {
            return 0;
        }
        let range = clamp_range(this.repr.bytes.len(), start, buffer.len());
        let len = range.len();
        buffer[..len].copy_from_slice(&this.repr.bytes[range]);
        len
    }

    fn write(this: &mut Blake3Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        if start >= this.repr.bytes.len() {
            return 0;
        }
        let range = clamp_range(this.repr.bytes.len(), start, buffer.len());
        let len = range.len();
        this.repr.bytes[range].copy_from_slice(&buffer[..len]);
        len
    }

    fn len(this: &Blake3Bytes<Self>) -> usize {
        this.repr.bytes.len()
    }

    fn resize(this: &mut Blake3Bytes<Self>, new_len: usize) {
        this.repr.bytes.resize(new_len, 0);
    }

    fn clone(this: &Blake3Bytes<Self>) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: this.repr.clone(),
        }
    }
}

impl Blake3BytesMode for Prove<'_> {
    fn new(len: usize) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: ProveRepr {
                previous: Source::owned(Blake3Bytes::new(len)),
                length: len,
                did_access_length: Cell::new(false),
                low_water: Cell::new(len),
                reads: RefCell::new(Vec::new()),
                writes: PartialVec::default(),
            },
        }
    }

    fn read(this: &Blake3Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        // Go through `len` so the length access is recorded: a read depends on the value's length
        // (to bound it), so the proof must commit the length — otherwise an out-of-bounds read
        // records nothing, the value is blinded, and the verify-mode replay faults instead of
        // returning 0. Mirrors `super::bytes::Bytes`.
        let total = Self::len(this);
        if start >= total {
            return 0;
        }
        let range = clamp_range(total, start, buffer.len());
        let len = range.len();

        // Record the read so it stays present in the value proof (docs §4).
        this.repr.reads.borrow_mut().push(range.clone());

        let buffer = &mut buffer[..len];
        let previous_len = this.repr.previous.len();
        let from_prev_start = range.start.min(previous_len);
        let from_prev_end = range.end.min(previous_len);
        let from_prev_len = from_prev_end - from_prev_start;

        buffer[..from_prev_len]
            .copy_from_slice(&this.repr.previous[from_prev_start..from_prev_end]);
        buffer[from_prev_len..].fill(0);

        // `defined_range` yields offsets relative to `range.start`, so they index `buffer`
        // (the range-sized slice) directly — matching `super::bytes::Bytes`.
        for (offset, bytes) in this.repr.writes.defined_range(range.clone()) {
            buffer[offset..][..bytes.len()].copy_from_slice(bytes);
        }

        len
    }

    fn write(this: &mut Blake3Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        // Route through `len` so the length access is recorded (see the note on `read`).
        let total = Self::len(this);
        if start >= total {
            return 0;
        }
        let range = clamp_range(total, start, buffer.len());
        let len = range.len();
        this.repr.writes.define(range.start, buffer[..len].to_vec());
        len
    }

    fn len(this: &Blake3Bytes<Self>) -> usize {
        this.repr.did_access_length.set(true);
        this.repr.length
    }

    fn resize(this: &mut Blake3Bytes<Self>, new_len: usize) {
        // Route through `len` so the length access is recorded — a resize commits the value's
        // length into the proof, without which the verify-mode replay cannot reconstruct the
        // resized value. Mirrors `super::bytes::Bytes`.
        let prev_len = Self::len(this);

        // Drop the low-water mark. Everything at or above it was dropped by this resize, so a
        // later regrow must read back zeros there rather than the pre-transition bytes.
        if prev_len != new_len {
            this.repr
                .low_water
                .set(this.repr.low_water.get().min(new_len));
        }

        if new_len < prev_len {
            this.repr.writes.truncate(new_len);
        }
        this.repr.length = new_len;
    }

    fn clone(this: &Blake3Bytes<Self>) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: this.repr.clone(),
        }
    }
}

impl Blake3BytesMode for Verify {
    fn new(len: usize) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: VerifyRepr {
                length: Some(len),
                low_water: len,
                data: PartialVec::from(vec![0u8; len]),
                proof: None,
            },
        }
    }

    fn read(this: &Blake3Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        let total = Verify::len(this);
        if start >= total {
            return 0;
        }
        let range = clamp_range(total, start, buffer.len());
        let len = range.len();

        let buffer = &mut buffer[..len];
        let Some(chunks) = this.repr.data.continuous_defined_range(range) else {
            // SAFETY: called only in `Verify` mode (invariant I4).
            unsafe { not_found() }
        };
        let mut offset = 0;
        for chunk in chunks {
            buffer[offset..][..chunk.len()].copy_from_slice(chunk);
            offset += chunk.len();
        }
        len
    }

    fn write(this: &mut Blake3Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        let total = Verify::len(this);
        if start >= total {
            return 0;
        }
        let range = clamp_range(total, start, buffer.len());
        let len = range.len();
        this.repr.data.define(start, buffer[..len].to_vec());
        len
    }

    fn len(this: &Blake3Bytes<Self>) -> usize {
        match this.repr.length {
            Some(len) => len,
            // SAFETY: called only in `Verify` mode (invariant I4).
            None => unsafe { not_found() },
        }
    }

    fn resize(this: &mut Blake3Bytes<Self>, new_len: usize) {
        let prev_len = Verify::len(this);
        if new_len > prev_len {
            this.repr
                .data
                .define(prev_len, vec![0u8; new_len - prev_len]);
        }
        if new_len < prev_len {
            this.repr.data.truncate(new_len);
        }
        this.repr.length = Some(new_len);
        // Track the low-water mark identically to prove mode, so the two agree on whether the
        // committed shape still describes the value.
        this.repr.low_water = this.repr.low_water.min(new_len);
    }

    fn clone(this: &Blake3Bytes<Self>) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: this.repr.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
