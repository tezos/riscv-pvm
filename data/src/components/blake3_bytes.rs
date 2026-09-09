// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Length-committed BLAKE3 hashing scheme for a byte array.
//!
//! This is the replacement for the page-tree scheme in [`super::bytes`]. A value's hash is
//! `H(v) = combine(H_len, blake3::hash(bytes))` — the length committed as its own leaf,
//! combined with the BLAKE3 hash of the raw bytes. Within-value partial proofs ride BLAKE3's
//! own internal 1024-byte chunk tree via the [`blake3::hazmat`] guts API (over the
//! `blake3::hash(bytes)` part), with *unrestricted* blinding: because the committed `H_len`
//! pins the length (hence the canonical shape), any untouched subtree collapses to a single
//! chaining value. The full design, security properties, and the numbered invariants
//! (`I1`..`I8`) are in `data/docs/blake3-bytes-length-committed.md`.
//!
//! The predecessor "no length node" scheme (`hash_value = blake3::hash(bytes)`,
//! `data/docs/blake3-bytes-hashing.md`) was **broken by a cross-span graft** that forged a
//! value's length; committing the length is the fix (see `cross_span_graft_rejected`).
//!
//! ## Audited free functions (the executable spec)
//!
//! - [`hash_value`] — Normal-mode hash `combine(H_len, blake3::hash(bytes))` (invariants I8, S0).
//! - [`prove`] — build a [`Blake3Proof`] capturing accessed byte ranges of the
//!   pre-transition value, blinding every untouched canonical subtree (I1, I6).
//! - [`verify_root`] — recompute the committed value hash canonically from the claimed length,
//!   rejecting non-canonical shapes / oversized chunks (I2, I3, I5).
//! - [`proof_read`] — read a range from a verified proof; faults on blinded/absent (I4).
//!
//! ## Fold-pipeline wiring (this integration step)
//!
//! The mode-generic [`Blake3Bytes`] is wired into the fold pipelines on top of those
//! functions:
//!
//! - [`Foldable<HashFold>`](Foldable) for [`Normal`] and [`Prove`] — the committed value hash
//!   `combine(H_len, blake3::hash(bytes))` (Prove folds the post-transition state; I8).
//! - [`Foldable<PartialHashFold>`](Foldable) for [`Verify`] — recomputes the root from the
//!   partial proof (present bytes + committed chaining values; I3, I8).
//! - [`Unfoldable`] / [`Foldable<BlobStoreFold>`](Foldable) for [`Normal`] — PVM-state store
//!   persistence (see the note on that impl about store-hash vs value-hash).
//!
//! The within-value proof also has its own self-contained wire format ([`Blake3Proof`]'s
//! [`Encode`]/[`Decode`]), used when the value is standalone.
//!
//! ## AVL proof wire (`MerkleProofFold` / `FromProof`)
//!
//! `Blake3Bytes` is a drop-in value for the durable-storage **proof** pipeline: in an AVL node's
//! `data` slot it is carried by a dedicated [`crate::merkle_proof::proof_tree::MerkleProofLeaf::Blake3`]
//! leaf (wire tag [`crate::merkle_proof::tag::TAG_BLAKE3`] = `0b01`) holding the length-committed
//! [`Blake3Proof`]. That leaf's generic Merkle `root_hash` is exactly the committed value hash
//! `combine(H_len, blake3::hash(bytes))` (via [`proof_root_hash`]), matching [`hash_value`]. This
//! is a genuine hard fork from the old page-tree value: the generic [`crate::merkle_proof`] tree
//! hashes a `Read` leaf as `hash_bytes(content)` and combines nodes with `combine_hashes`, neither
//! of which matches `blake3::hash(bytes)` nor BLAKE3's internal `merge_subtrees` compression — so
//! the value needs its own leaf kind rather than the `Read`/`Blind` wire. The two encodings stay
//! structurally disjoint (invariant I7): the parse position and tag decide the decoder, and only
//! `Blake3Bytes` ever emits or parses the `Blake3` leaf (see [`Foldable<MerkleProofFold>`](Foldable)
//! for [`Prove`] and [`FromProof`] for [`Verify`] below).
//!
//! **Still pending (a separate step):** switching the durable-storage / PVM AVL value type from
//! the old [`super::bytes::Bytes`] to `Blake3Bytes`, and the attendant goldenfile regeneration.
//! Until that lands, nothing in production emits a `Blake3` leaf.

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
use crate::foldable::Fold;
use crate::foldable::FoldLeaf;
use crate::foldable::Foldable;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::foldable::Unfoldable;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::Partial;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::SuspendedResult;
use crate::merkle_proof::proof_tree::MerkleProofFold;
use crate::merkle_proof::proof_tree::MinimumPresence;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::Source;
use crate::mode::utils::not_found;
use crate::partial_vec::PartialVec;
use crate::store::BlobStore;
use crate::store::fold::BlobStoreFold;

/// BLAKE3 chunk length: the leaf granularity of the internal chunk tree (1024 bytes).
pub const CHUNK_LEN: usize = blake3::CHUNK_LEN;

// ---------------------------------------------------------------------------------------
// Normal-mode hash
// ---------------------------------------------------------------------------------------

/// Precomputed [`hash_len`] for every length that fits in a single BLAKE3 chunk
/// (`0..=CHUNK_LEN`), which is the overwhelmingly common value size. `hash_len` is a pure function
/// of the length, so this is a transparent cache: a table hit returns the identical hash a fresh
/// `blake3` call would, but skips one 8-byte BLAKE3 compression — a meaningful fraction of the
/// fixed cost of hashing a small value (`combine(H_len, H_data)` over ≤64 B is dominated by the
/// three compressions; this removes one). Built once, lazily, on first use (1025 hashes, ~tens of
/// µs). `Box`ed to keep the 32 KiB table off the stack during initialisation.
static SMALL_LEN_HASHES: std::sync::LazyLock<Box<[Hash; CHUNK_LEN + 1]>> =
    std::sync::LazyLock::new(|| {
        Box::new(std::array::from_fn(|len| {
            Hash::hash_bytes(&(len as u64).to_le_bytes())
        }))
    });

/// Commit a value's length as its own leaf: `H_len = blake3(len_le_bytes)`.
///
/// The length is hashed as its `u64` little-endian encoding (no bincode framing) so it is a
/// fixed, consensus-relevant preimage. This is the leaf that pins the value's length in the
/// committed hash (see [`hash_value`]); dropping it is what let the predecessor "no length
/// node" scheme be forged by a cross-span graft (`data/docs/blake3-bytes-length-committed.md`
/// §1, §5 S2).
///
/// Single-chunk lengths (the common case) are served from the [`SMALL_LEN_HASHES`] cache; larger
/// lengths are computed directly (there the value-bytes hash dominates anyway).
fn hash_len(len: usize) -> Hash {
    match SMALL_LEN_HASHES.get(len) {
        Some(hash) => *hash,
        None => Hash::hash_bytes(&(len as u64).to_le_bytes()),
    }
}

/// Value sizes at or above this use the rayon-parallel BLAKE3 hash (only with the
/// `parallel-hashing` feature). Below it the single-thread SIMD hash is faster than the
/// thread-hand-off overhead (measured crossover ~128-256 KiB).
#[cfg(feature = "parallel-hashing")]
const PARALLEL_HASH_THRESHOLD: usize = 256 * 1024;

/// BLAKE3 hash of the raw value bytes — the `H_data` half of [`hash_value`].
///
/// With the `parallel-hashing` feature, large values are hashed across a rayon pool
/// ([`blake3::Hasher::update_rayon`]); the result is byte-identical to the single-threaded
/// [`blake3::hash`] (BLAKE3's tree is the same however it is evaluated), so this never changes any
/// committed hash — it is purely a wall-clock optimisation. Small values (the common case) always
/// take the single-threaded path.
#[inline]
fn hash_data(bytes: &[u8]) -> Hash {
    #[cfg(feature = "parallel-hashing")]
    if bytes.len() >= PARALLEL_HASH_THRESHOLD {
        let mut hasher = blake3::Hasher::new();
        hasher.update_rayon(bytes);
        return Hash::from(*hasher.finalize().as_bytes());
    }
    Hash::hash_bytes(bytes)
}

/// Normal-mode hash of a byte array under the length-committed BLAKE3 scheme.
///
/// `H(v) = combine(H_len, blake3::hash(bytes))` — the length leaf [`hash_len`] combined with
/// the BLAKE3 hash of the raw bytes (invariant I8). The explicit length commitment pins the
/// value's length, which is what makes within-value partial proofs with *unrestricted*
/// blinding secure (`data/docs/blake3-bytes-length-committed.md` §2). This is **not** equal to
/// `blake3::hash(bytes)`: it wraps it in one more combine.
pub fn hash_value(bytes: &[u8]) -> Hash {
    Hash::combine_hashes([hash_len(bytes.len()), hash_data(bytes)])
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
    /// security instead (`data/docs/blake3-bytes-length-committed.md` §3, §5 S2; invariant I6).
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

// ---------------------------------------------------------------------------------------
// Prove / Verify contract
// ---------------------------------------------------------------------------------------

/// Build a proof for `data` that keeps exactly the chunks overlapping `accessed` present
/// and blinds every other canonical subtree to its chaining value.
///
/// Blinding is recursive and *unrestricted* (invariant I1, I6): an untouched canonical subtree
/// of many chunks — at any offset or span — collapses to a single 32-byte CV. If nothing was
/// accessed, the whole data tree collapses to a single top-level [`ProofTree::Blind`] carrying
/// `H_data` (used when only the value's length was read; the length still travels in
/// [`Blake3Proof::total_len`]).
pub fn prove(data: &[u8], accessed: &[Range<usize>]) -> Blake3Proof {
    prove_for_reshape(data, accessed, &[])
}

/// [`prove`], additionally keeping the shape open wherever a reshape verifier needs to reach into
/// the unchanged prefix.
///
/// Presence of *bytes* and openness of *shape* are different requirements, and a bare [`prove`] can
/// only express the first. A reshape verifier recovers the unchanged prefix from aligned
/// power-of-two chunk subtree CVs, and can fetch one only by an exact descent - so no coarser blind
/// may swallow it. `reused` lists those subtrees (see [`reused_prefix_subtrees`]); their ancestors
/// are kept as nodes even when nothing in them was accessed, while the subtrees themselves stay
/// blinded, which is precisely the reuse. The extra cost is one CV per level.
///
/// The root is kept open whenever `reused` is non-empty: a *top-level* blind carries the
/// `ROOT`-finalised `H_data`, which is not a chaining value and so can never serve as one. Left
/// open, its children recombine to the right non-root CV instead.
///
/// Pass an empty slice when the transition does not reshape the value.
pub fn prove_for_reshape(
    data: &[u8],
    accessed: &[Range<usize>],
    reused: &[(usize, usize)],
) -> Blake3Proof {
    let total_len = data.len();
    let tree = build_proof_tree(data, 0, total_len, accessed, true, reused);
    Blake3Proof {
        total_len,
        data: tree,
    }
}

/// The aligned prefix subtrees a reshape verifier will fetch from the pre-proof, as
/// `(offset, len)` pairs.
///
/// Mirrors [`reshape_data_root`]/[`reshape_cv`]'s descent over `canonical(lq)`, so the answer is
/// exactly what that descent will ask [`pre_subtree_cv`] for - which is why the granularity cannot
/// be derived from `safe_full` alone: the descent splits the post root first, so a small `lq`
/// bottoms out at single chunks even when the prefix is large.
/// The maximal subtrees of the pre tree whose bytes the replay writes in full.
///
/// These have to stay individually blinded rather than be swallowed by a coarser blind. A blind is
/// re-derived only where the verifier holds the whole span, and what it holds is exactly what the
/// replay wrote: a blind spanning written and unwritten bytes is one it can neither trust nor
/// recompute, and it would silently keep the pre value instead.
///
/// This is the same distinction the page-tree `Bytes` component draws with
/// `MinimumPresence::MayBlind` against `MayOmit` - written in full means no pre-image is needed,
/// not that the subtree can vanish. Only the whole value being written is `MayOmit`, and that is
/// handled by the caller.
fn written_subtrees(total_len: usize, writes: &PartialVec<u8>) -> Vec<(usize, usize)> {
    fn collect(offset: usize, len: usize, writes: &PartialVec<u8>, out: &mut Vec<(usize, usize)>) {
        if writes.contiguous_range(offset..offset + len).is_some() {
            out.push((offset, len));
            return;
        }

        // A partly written chunk keeps its pre-image instead, recorded by `accessed_ranges`.
        if len <= CHUNK_LEN {
            return;
        }

        let left_len = left_subtree_len(len as u64) as usize;
        collect(offset, left_len, writes, out);
        collect(offset + left_len, len - left_len, writes, out);
    }

    let mut out = Vec::new();

    if total_len == 0 {
        return out;
    }

    collect(0, total_len, writes, &mut out);
    out
}

fn reused_prefix_subtrees(lq: usize, safe_full: usize) -> Vec<(usize, usize)> {
    fn collect(offset: usize, len: usize, safe_full: usize, out: &mut Vec<(usize, usize)>) {
        if offset + len <= safe_full && is_aligned_subtree(offset, len) {
            out.push((offset, len));
            return;
        }
        // At or past the prefix the verifier hashes its own bytes; a leaf cannot be split further.
        if offset >= safe_full || len <= CHUNK_LEN {
            return;
        }
        let left_len = left_subtree_len(len as u64) as usize;
        collect(offset, left_len, safe_full, out);
        collect(offset + left_len, len - left_len, safe_full, out);
    }

    let mut out = Vec::new();
    // A single-chunk post value is hashed live in full, and an empty prefix reuses nothing.
    if safe_full == 0 || lq <= CHUNK_LEN {
        return out;
    }
    let left_len = left_subtree_len(lq as u64) as usize;
    collect(0, left_len, safe_full, &mut out);
    collect(left_len, lq - left_len, safe_full, &mut out);
    out
}

/// Must the subtree `[offset, offset + len)` stay open (a node) so a reshape verifier can reach the
/// CVs it reuses, even though nothing in it was accessed?
///
/// Yes exactly when it *strictly* contains one of the reused subtrees. A node that is itself reused
/// may stay blinded - that blind is the CV the verifier wants.
fn must_stay_open(offset: usize, len: usize, reused: &[(usize, usize)]) -> bool {
    reused
        .iter()
        .any(|&(po, pl)| po >= offset && po + pl <= offset + len && (po, pl) != (offset, len))
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
    reused: &[(usize, usize)],
) -> ProofTree {
    let touched = if is_root {
        !accessed.is_empty() || !reused.is_empty()
    } else {
        range_is_accessed(offset, len, accessed) || must_stay_open(offset, len, reused)
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
        // Reached by presence of bytes only: `must_stay_open` never fires at chunk granularity, so
        // a chunk is present here because it really was accessed.
        ProofTree::Chunk(data[offset..offset + len].to_vec())
    } else {
        let left_len = left_subtree_len(len as u64) as usize;
        let left = build_proof_tree(data, offset, left_len, accessed, false, reused);
        let right = build_proof_tree(
            data,
            offset + left_len,
            len - left_len,
            accessed,
            false,
            reused,
        );
        ProofTree::Node(Box::new(left), Box::new(right))
    }
}

/// Recompute the committed value hash from `proof`.
///
/// Returns `combine(H_len, H_data)` where `H_len = hash_len(proof.total_len)` and `H_data` is
/// the BLAKE3 root reconstructed from the [`data`](Blake3Proof::data) tree (invariant I3). The
/// caller compares the result to the committed value hash: a lie about `total_len` changes
/// `H_len` (hence the returned hash) and a wrong present chunk / blinded CV changes `H_data`,
/// so neither is accepted absent a BLAKE3 collision (`data/docs/blake3-bytes-length-committed.md`
/// §5). For an honest proof of `data`, this equals `hash_value(data)` (invariant I8).
pub fn verify_root(proof: &Blake3Proof) -> Result<Hash, ProofError> {
    let h_data = data_root(&proof.data, proof.total_len)?;
    Ok(Hash::combine_hashes([
        hash_len(proof.total_len),
        Hash::from(h_data),
    ]))
}

/// Infallible committed value hash of a proof, for the generic Merkle-proof machinery.
///
/// The BLAKE3 within-value leaf ([`crate::merkle_proof::proof_tree::MerkleProofLeaf::Blake3`])
/// hashes to [`verify_root`]. Callers of [`crate::merkle_proof::proof_tree::MerkleProof::root_hash`]
/// need a total function, so a malformed proof (only reachable from an adversarial, decoded proof)
/// yields a fixed domain-separated poison hash instead of erroring. This is sound: the poison can
/// never equal an honest value hash (which is a `combine`, not `hash_bytes` of this literal), so a
/// proof whose root hashes to the poison simply fails the caller's root-equality check.
pub fn proof_root_hash(proof: &Blake3Proof) -> Hash {
    verify_root(proof).unwrap_or_else(|_| Hash::hash_bytes(b"octez-riscv:invalid-blake3-proof"))
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
/// `H_data` (`data/docs/blake3-bytes-length-committed.md` §4.2, §5 S2; invariant I6).
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
// scheme. See the module docs and `data/docs/blake3-bytes-hashing.md` §5 P4 / I7.

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
    /// below it were never dropped by a resize, so they retain their pre-transition value — hence
    /// `[0, low_water)` is exactly the prefix whose subtree CVs the verifier may reuse across a
    /// reshape. Here it is used only by [`Self::post_bytes`] to zero the regrown region; the
    /// verifier tracks its own copy (see [`VerifyRepr::low_water`]). See `resize`.
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
    /// The verifier replays the same resizes as the prover, so it derives the same low-water
    /// mark and hence agrees on whether a reshape is "clean" (prefix reusable) or must fall back
    /// to a full re-hash. Meaningless while `length` is `None` (absent value).
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

    /// The accessed byte ranges (reads unioned with writes), clamped to the pre-transition
    /// length.
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

        // Bytes that survive the transition: past this point the pre value is either gone (a
        // shrink) or was never there, so its pre-image cannot be part of any post chunk.
        let live = prev_len.min(self.repr.length);
        let clamp_live = |start: usize, end: usize| -> Option<Range<usize>> {
            let start = start.min(live);
            let end = end.min(live);
            (start < end).then_some(start..end)
        };

        // A write needs no pre-image where it covers a *whole* chunk: the verifier replays the
        // same write, so it can hash that chunk's post bytes itself and never has to know what
        // stood there before (the pre root stays authenticated, by the blinded CV). Only the
        // partially covered boundary chunks need their pre-image, and only for the part the write
        // does not itself supply - `build_proof_tree` emits the whole chunk once any byte of it
        // is accessed, so recording that remainder brings the full chunk into the proof. The
        // verifier side of this is the `Blind` arm of `rebuild_tree` / `subtree_cv_from_view`,
        // which prefer materialised bytes over a blinded CV.
        for (offset, bytes) in self.repr.writes.defined_range(0..self.repr.length) {
            let end = offset + bytes.len();

            let head = (offset / CHUNK_LEN) * CHUNK_LEN;
            if let Some(range) = clamp_live(head, offset) {
                ranges.push(range);
            }

            let tail = end.div_ceil(CHUNK_LEN) * CHUNK_LEN;
            if let Some(range) = clamp_live(end, tail) {
                ranges.push(range);
            }
        }

        // On a reshape the verifier reuses length-independent aligned subtree CVs for
        // `[0, safe_full)` and rebuilds `[low_water, length)` from its own replay, so the only pre
        // bytes it still needs are the prefix's partial boundary chunk, `[safe_full, low_water)` -
        // the one span neither mechanism covers. The exception is a post value of a single chunk,
        // which `reshape_data_root` hashes live in one piece: there every surviving pre byte is
        // needed. Either way, bytes a write already supplies are not. (Keeping the prefix's *shape*
        // reachable is a separate requirement, handled by `prove_for_reshape`.)
        if let Some((safe_full, low_water)) = self.reshape_prefix() {
            let needed = if self.repr.length <= CHUNK_LEN {
                0..self.repr.length.min(prev_len)
            } else {
                safe_full..low_water
            };
            let supplied = self.repr.writes.contiguous_range(needed.clone()).is_some();
            if !supplied && let Some(range) = clamp(needed.start, needed.end) {
                ranges.push(range);
            }
        }

        ranges
    }

    /// The reshape geometry, or `None` when the transition does not reshape the value.
    ///
    /// `low_water` is the lowest length the value held (clamped to the pre length): its bytes were
    /// never dropped, so they are unchanged pre->post and their aligned subtree CVs are reusable.
    /// `safe_full` rounds that down to a whole chunk, since only whole chunks have such CVs.
    fn reshape_prefix(&self) -> Option<(usize, usize)> {
        let prev_len = self.repr.previous.len();
        let low_water = self.repr.low_water.get().min(prev_len);
        let reshaped = self.repr.length != prev_len || low_water != self.repr.length;
        reshaped.then(|| ((low_water / CHUNK_LEN) * CHUNK_LEN, low_water))
    }

    /// Build the within-value [`Blake3Proof`] capturing the pre-transition state, keeping the
    /// accessed ranges present and blinding every untouched canonical subtree (docs §3, §4).
    ///
    /// This reuses the audited free function [`prove`].
    pub fn value_proof(&self) -> Blake3Proof {
        let accessed = self.accessed_ranges();

        // Open the shape only for the prefix subtrees the verifier cannot rebuild itself: one whose
        // bytes a write already supplies is hashed from the replay (`pre_subtree_cv`), so opening
        // the path to it would cost a CV per level and buy nothing.
        let previous_len = self.repr.previous.len();

        // The whole value written is the one case a single coarse blind serves: the verifier holds
        // all of it, so it re-derives the root itself and nothing has to stay open.
        let all_written = previous_len > 0
            && self
                .repr
                .writes
                .contiguous_range(0..previous_len)
                .is_some();

        let mut reused: Vec<_> = self
            .reshape_prefix()
            .map(|(safe_full, _)| reused_prefix_subtrees(self.repr.length, safe_full))
            .unwrap_or_default();

        // Subtrees the replay writes in full, so that no coarser blind swallows them. A prefix
        // subtree whose bytes a write supplies is one of these, which is why it is no longer
        // filtered out here: rebuilding it from the replay only works if the blind is at a
        // granularity the verifier holds whole, and dropping it from this list is what let a
        // coarser one take its place.
        if !all_written {
            reused.extend(written_subtrees(previous_len, &self.repr.writes));
        }

        reused.sort_unstable();
        reused.dedup();

        prove_for_reshape(&self.repr.previous, &accessed, &reused)
    }

    /// Reconstruct the post-transition bytes (pre-transition data overlaid with recorded
    /// writes, resized to the current length).
    ///
    /// Any position at or beyond the low-water mark was dropped by a shrink at some point in the
    /// transition, so if the value later regrew past it those bytes must read back as **zero**, not
    /// the stale pre-transition data that `previous` still holds. We zero `[low_water, length)`
    /// before overlaying the surviving writes (writes past a shrink boundary were already dropped
    /// by `resize`'s `writes.truncate`). This matches verify mode, which zero-fills the newly
    /// exposed region on every grow. For a plain append `low_water == previous.len()`, so the
    /// zero-fill only touches the appended tail (already zero) and this is a no-op.
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

    /// Recompute the current (post-transition) value root from the sparse verify view.
    ///
    /// Returns `None` only when the value is completely absent (length unknown — blinded at the
    /// AVL layer), in which case the fold defers to the previous hash. Cases:
    ///
    /// - **Retained proof, unchanged length:** rebuild the committed BLAKE3 shape, substituting
    ///   current (possibly written) bytes for present chunks and keeping blinded chaining values.
    /// - **Retained proof, reshape:** reuse the pre-proof's length-independent prefix subtree CVs
    ///   for the reusable prefix `[0, low_water)` (bytes never dropped by a shrink, hence unchanged)
    ///   and hash the changed region `[low_water, length)` — which the verifier reconstructed by
    ///   replaying the transition (zero-fill on grow + writes) — from the sparse view. This keeps
    ///   the recompute O(depth) for *every* reshape, including shrink-below-then-regrow (see
    ///   [`reshape_data_root`]).
    /// - **No retained proof** (a value freshly created or wholly `set` in verify mode): recompute
    ///   `hash_value` from the fully materialised bytes.
    fn current_root(&self) -> Option<Hash> {
        let length = self.repr.length?;

        // A present but empty value always hashes to `hash_value(&[])`, independent of any
        // retained proof or reshape history. Handle it up front — the sparse-view accessors below
        // return `None` for the degenerate `0..0` range.
        if length == 0 {
            return Some(hash_value(&[]));
        }

        if let Some(proof) = self.repr.proof.as_ref() {
            let lp = proof.total_len;

            // Fast path: the length is unchanged AND the value never shrank below it
            // (`low_water == length`, since `low_water <= length` always). Only then is every
            // blinded chunk genuinely untouched, so the committed shape can be reused directly,
            // substituting current bytes for present (written) chunks. NOTE the `low_water` guard:
            // a shrink-below-then-regrow back to the same length (e.g. 8192 -> 0 -> 8192) leaves
            // `length == lp` but replaces the "blinded" chunks with zeros — it must NOT hit this
            // path, or it would reuse stale pre CVs. It falls through to the reshape recompute.
            if length == lp && self.repr.low_water == length {
                let data = rebuild_tree(&proof.data, 0, length, &self.repr.data, true);
                return verify_root(&Blake3Proof {
                    total_len: length,
                    data,
                })
                .ok();
            }

            // Reshape (length changed, or a shrink-below-then-regrow returned to the same length).
            // The reusable prefix is `[0, low_water)`: bytes below the low-water mark were never
            // dropped by a shrink, so they are byte-identical pre->post and their aligned subtree
            // CVs (length-independent) can be reused at the new shape. The changed region
            // `[low_water, length)` was rebuilt by the verifier replaying the same transition, so
            // it is materialised in the sparse view and hashed live — nothing extra needs to be
            // present in the proof. `low_water` is replayed identically to the prover.
            let prefix = self.repr.low_water.min(length).min(lp);
            if let Some(h_data) =
                reshape_data_root(&proof.data, lp, length, prefix, &self.repr.data)
            {
                return Some(Hash::combine_hashes([hash_len(length), Hash::from(h_data)]));
            }
            // Reshape recompute could not complete from the sparse view (only reachable from a
            // malformed/adversarial proof): fall through to the full-materialisation attempt,
            // which will itself return `None` and let the fold's root check reject the proof.
        }

        // No retained proof: recompute from fully materialised bytes.
        let full = self.repr.data.contiguous_range(0..length)?;
        Some(hash_value(full))
    }
}

/// Recompute `H_data` (the `ROOT`-finalised BLAKE3 root of the reshaped post value) for a **clean
/// length reshape**, reusing the pre-proof's length-independent prefix subtree chaining values.
///
/// `pre` is the retained pre-transition proof tree (canonical shape for `lp`); `lq` is the new
/// length; `common = min(lp, lq)` is the unchanged prefix. Only *whole chunks* below the prefix
/// are reused (`safe_full = floor(common / CHUNK) * CHUNK`): an aligned power-of-two chunk subtree
/// is a node of both `canonical(lp)` and `canonical(lq)` and carries the same (length-independent)
/// CV (F1/F2 in `data/docs/blake3-bytes-reshape-optimisation.md`). The boundary chunk and the
/// changed region `[safe_full, lq)` are hashed from the *current* materialised bytes. Returns
/// `None` if the sparse view lacks bytes it needs (adversarial/malformed proof).
fn reshape_data_root(
    pre: &ProofTree,
    lp: usize,
    lq: usize,
    common: usize,
    data: &PartialVec<u8>,
) -> Option<[u8; 32]> {
    let safe_full = (common / CHUNK_LEN) * CHUNK_LEN;

    if lq <= CHUNK_LEN {
        // The whole post value is a single ROOT-finalised chunk; it is materialised (boundary or
        // changed region), so hash it live.
        let bytes = data.contiguous_range(0..lq)?;
        return Some(*blake3::hash(bytes).as_bytes());
    }

    let left_len = left_subtree_len(lq as u64) as usize;
    let left = reshape_cv(pre, lp, 0, left_len, safe_full, data)?;
    let right = reshape_cv(pre, lp, left_len, lq - left_len, safe_full, data)?;
    Some(*merge_subtrees_root(&left, &right, B3Mode::Hash).as_bytes())
}

/// Non-root CV of the `canonical(lq)` subtree `[offset, offset + len)` during a clean reshape.
///
/// Three cases, in order (offsets are always chunk-aligned in the canonical recursion, and
/// `safe_full` is chunk-aligned, so a leaf is never split across `safe_full`):
/// - **Entirely in the reusable aligned prefix** (`offset + len <= safe_full`): take the CV from
///   the pre-proof via [`pre_subtree_cv`] — it is length-independent and thus valid at the new
///   shape. This is the load-bearing R3 rule: reuse only fully within the unchanged prefix.
/// - **At or beyond the prefix** (`offset >= safe_full`): the boundary chunk or the changed
///   region — hash the current materialised bytes canonically.
/// - **Straddling `safe_full`** (an internal node only): descend.
fn reshape_cv(
    pre: &ProofTree,
    lp: usize,
    offset: usize,
    len: usize,
    safe_full: usize,
    data: &PartialVec<u8>,
) -> Option<[u8; 32]> {
    // Reuse only an *aligned power-of-two chunk* subtree that lies entirely in the unchanged
    // prefix: such a node is common to `canonical(lp)` and `canonical(lq)` and carries the same
    // (length-independent) CV (F2). A *ragged* subtree can lie within the prefix yet exist only in
    // `canonical(lq)` (e.g. `[4,7)` is a node of `canonical(7)` but not `canonical(10)`), so it is
    // NOT directly reusable — it is descended into its aligned pieces below.
    if offset + len <= safe_full && is_aligned_subtree(offset, len) {
        return pre_subtree_cv(pre, 0, lp, offset, len, data);
    }
    if offset >= safe_full {
        let bytes = data.contiguous_range(offset..offset + len)?;
        return Some(canonical_cv(bytes, offset as u64));
    }
    // Straddles `safe_full`, or is a ragged node within the prefix: descend. Left children of a
    // canonical split are always aligned, so the aligned prefix pieces are reached at the first
    // branch above.
    let left_len = left_subtree_len(len as u64) as usize;
    let left = reshape_cv(pre, lp, offset, left_len, safe_full, data)?;
    let right = reshape_cv(pre, lp, offset + left_len, len - left_len, safe_full, data)?;
    Some(merge_subtrees_non_root(&left, &right, B3Mode::Hash))
}

/// Is `[offset, offset + len)` an aligned power-of-two **chunk** subtree — i.e. a node that occurs
/// in *every* canonical BLAKE3 tree large enough to contain it (F1/F2)? Requires `len` to be a
/// power-of-two multiple of `CHUNK_LEN` and `offset` a multiple of `len`. (Within the reusable
/// prefix every leaf is a full chunk, so `len` is always a whole number of chunks here.)
fn is_aligned_subtree(offset: usize, len: usize) -> bool {
    len.is_multiple_of(CHUNK_LEN) && {
        let chunks = len / CHUNK_LEN;
        chunks.is_power_of_two() && offset.is_multiple_of(len)
    }
}

/// CV of the `canonical(lp)` subtree `(t_offset, t_len)`, navigating the pre-proof `pre` (whose
/// shape over `[cur_offset, cur_offset + cur_len)` is `canonical(lp)`). The target is always a
/// genuine canonical node (a set-bit-of-`safe_full` prefix piece), so the descent reaches it
/// exactly. Its CV is then taken from the pre-proof by [`subtree_cv_from_view`]: a blind yields its
/// stored CV, a present chunk/node is re-hashed from the *current* bytes (so prefix writes are
/// reflected). Returns `None` if a coarser blind swallows the target (cannot happen for an honest
/// pre-proof) — the caller then falls back to a full re-hash.
fn pre_subtree_cv(
    pre: &ProofTree,
    cur_offset: usize,
    cur_len: usize,
    t_offset: usize,
    t_len: usize,
    data: &PartialVec<u8>,
) -> Option<[u8; 32]> {
    if cur_offset == t_offset && cur_len == t_len {
        return subtree_cv_from_view(pre, cur_offset, cur_len, data);
    }
    match pre {
        ProofTree::Node(left, right) => {
            let left_len = left_subtree_len(cur_len as u64) as usize;
            if t_offset < cur_offset + left_len {
                pre_subtree_cv(left, cur_offset, left_len, t_offset, t_len, data)
            } else {
                pre_subtree_cv(
                    right,
                    cur_offset + left_len,
                    cur_len - left_len,
                    t_offset,
                    t_len,
                    data,
                )
            }
        }
        // The target sits inside a coarser blind, which cannot be decomposed - unless the verifier
        // materialised the target's bytes itself (it replayed writes covering them), in which case
        // it needs nothing from the proof here. This is what lets a full overwrite omit the prefix
        // shape entirely rather than paying a CV per level to open it.
        _ => data
            .contiguous_range(t_offset..t_offset + t_len)
            .map(|bytes| canonical_cv(bytes, t_offset as u64)),
    }
}

/// CV of the subtree `pre` (canonically covering `[offset, offset + len)`), reading present chunks
/// from the *current* materialised bytes so writes to the unchanged prefix are reflected. Mirrors
/// [`compute_cv`] but sources present bytes from `data` rather than the (pre-transition) proof.
fn subtree_cv_from_view(
    pre: &ProofTree,
    offset: usize,
    len: usize,
    data: &PartialVec<u8>,
) -> Option<[u8; 32]> {
    match pre {
        // As in `rebuild_tree`: if the verifier materialised this whole span itself (replayed
        // writes inside the reusable prefix), its CV comes from those bytes, not from the stale
        // pre CV. Never the root here - `pre_subtree_cv` only targets proper subtrees of a
        // reshaped value, whose blinds are non-root CVs.
        ProofTree::Blind(cv) => Some(match data.contiguous_range(offset..offset + len) {
            Some(current) => canonical_cv(current, offset as u64),
            None => *cv,
        }),
        ProofTree::Chunk(_) => {
            let bytes = data.contiguous_range(offset..offset + len)?;
            Some(
                blake3::Hasher::new()
                    .set_input_offset(offset as u64)
                    .update(bytes)
                    .finalize_non_root(),
            )
        }
        ProofTree::Node(left, right) => {
            let left_len = left_subtree_len(len as u64) as usize;
            let l = subtree_cv_from_view(left, offset, left_len, data)?;
            let r = subtree_cv_from_view(right, offset + left_len, len - left_len, data)?;
            Some(merge_subtrees_non_root(&l, &r, B3Mode::Hash))
        }
    }
}

/// Rebuild a proof tree over `[offset, offset + len)` from the committed shape, substituting the
/// current (possibly written) bytes wherever the sparse view has them.
///
/// A blinded span whose current bytes the verifier holds in full - because it replayed writes
/// covering it - is re-blinded to the CV of *those* bytes rather than keeping the stale pre CV.
/// That is what lets the prover omit the pre-image of a fully overwritten chunk. The span stays a
/// `Blind` (blinding is unrestricted once the length is committed), so this costs one CV rather
/// than materialising the chunk subtree. `is_root` distinguishes the whole value, whose blind
/// carries the `ROOT`-finalised `H_data` instead of a non-root chaining value.
fn rebuild_tree(
    node: &ProofTree,
    offset: usize,
    len: usize,
    data: &PartialVec<u8>,
    is_root: bool,
) -> ProofTree {
    match node {
        ProofTree::Chunk(original) => match data.contiguous_range(offset..offset + len) {
            Some(current) => ProofTree::Chunk(current.to_vec()),
            None => ProofTree::Chunk(original.clone()),
        },
        ProofTree::Blind(cv) => match data.contiguous_range(offset..offset + len) {
            Some(current) if is_root => ProofTree::Blind(*blake3::hash(current).as_bytes()),
            Some(current) => ProofTree::Blind(canonical_cv(current, offset as u64)),
            None => ProofTree::Blind(*cv),
        },
        ProofTree::Node(left, right) => {
            let left_len = left_subtree_len(len as u64) as usize;
            ProofTree::Node(
                Box::new(rebuild_tree(left, offset, left_len, data, false)),
                Box::new(rebuild_tree(
                    right,
                    offset + left_len,
                    len - left_len,
                    data,
                    false,
                )),
            )
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

        // A length change reshapes BLAKE3's chunk tree, but the prefix that was never dropped is
        // recoverable cheaply: its aligned power-of-two subtree chaining values are
        // *length-independent*, so the verifier reuses the pre-proof's blinded prefix CVs at the
        // new shape (see `reshape_data_root` and `data/docs/blake3-bytes-reshape-optimisation.md`).
        // All this call has to do is drop the low-water mark; `accessed_ranges` derives what the
        // proof must carry from the *final* mark, which is where the verifier actually reuses up
        // to. The changed region above it is rebuilt by the verifier's own replay and is never
        // carried in the proof. This keeps *every* reshape O(depth), never O(value).
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
        // Track the low-water mark identically to prove mode so the reshape recompute agrees on
        // whether the common prefix is intact.
        this.repr.low_water = this.repr.low_water.min(new_len);
    }

    fn clone(this: &Blake3Bytes<Self>) -> Blake3Bytes<Self> {
        Blake3Bytes {
            repr: this.repr.clone(),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Fold pipelines
// ---------------------------------------------------------------------------------------

impl Foldable<HashFold> for Blake3Bytes<Normal> {
    /// Normal-mode value hash: `combine(H_len, blake3::hash(bytes))` (invariant I8).
    fn fold(&self, _builder: HashFold) -> Hash {
        hash_value(self.as_bytes())
    }
}

impl Foldable<HashFold> for Blake3Bytes<Prove<'_>> {
    /// Prove-mode value hash over the *post-transition* state (matches Normal mode after the
    /// same writes — invariant I8).
    fn fold(&self, _builder: HashFold) -> Hash {
        self.hash()
    }
}

impl Foldable<PartialHashFold> for Blake3Bytes<Verify> {
    /// Verify-mode value hash recomputed from the partial proof (invariants I3, I8).
    ///
    /// Defers to the previous hash when the value is completely absent (blinded at the AVL
    /// layer); otherwise recomputes the current root from present bytes + committed CVs.
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        match self.current_root() {
            Some(hash) => builder.present(hash),
            None => builder.previous(),
        }
    }
}

// ---------------------------------------------------------------------------------------
// AVL proof wire (Prove: MerkleProofFold; Verify: FromProof)
// ---------------------------------------------------------------------------------------
//
// A byte value occupies a single child slot (the `data` slot) of an AVL node. In a proof it is a
// single BLAKE3 within-value leaf (`MerkleProofLeaf::Blake3`) carrying the length-committed
// `Blake3Proof`, whose Merkle-proof `root_hash` is the committed value hash
// `combine(H_len, blake3::hash(bytes))` — matching the `HashFold` value hash exactly. This keeps
// the value-proof encoding structurally disjoint from the generic Read/Blind node encoding
// (invariant I7): the wire tag `LeafTag::Blake3` selects the decoder, and only `Blake3Bytes` ever
// emits or parses it. See `data/docs/blake3-bytes-length-committed.md` §7.3.

impl Foldable<MerkleProofFold> for Blake3Bytes<Prove<'_>> {
    /// Emit the value's AVL-wire proof, capturing the *pre-transition* state (proofs always
    /// describe the state at the start of proof recording).
    ///
    /// If any byte range was accessed, or the length was read, the value is [`MinimumPresence::Present`]
    /// and carries a [`Blake3Proof`] with the accessed chunks present and every untouched subtree
    /// blinded. If nothing was touched, it is blinded to the pre-transition value hash so the parent
    /// node can omit it.
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        let accessed = self.accessed_ranges();
        let needs_present = self.repr.did_access_length.get() || !accessed.is_empty();

        if !needs_present {
            // Whole value untouched: blind to the pre-transition committed value hash.
            return builder.into_blind(hash_value(&self.repr.previous));
        }

        builder.into_blake3_leaf(MinimumPresence::Present, self.value_proof())
    }
}

impl FromProof for Blake3Bytes<Verify> {
    /// Reconstruct a verify-mode value from the AVL-wire proof's `data` slot.
    ///
    /// A present [`MerkleProofLeaf::Blake3`] populates the sparse view from its proven chunks (reads
    /// of blinded/absent regions fault, invariant I4); a blinded or absent value node yields a
    /// completely-absent value whose [`PartialHashFold`] defers to the previous (blinded) hash.
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let suspended = proof.into_blake3_leaf()?;
        Ok(suspended.map(|partial| match partial {
            Partial::Present(value_proof) => {
                Blake3Bytes::<Verify>::from_proof_unchecked(value_proof)
            }
            Partial::Blinded(_) | Partial::Absent => Blake3Bytes {
                repr: VerifyRepr::default(),
            },
        }))
    }
}

// ---------------------------------------------------------------------------------------
// Store fold / unfold (PVM state tree persistence)
// ---------------------------------------------------------------------------------------
//
// NOTE: unlike [`super::bytes::Bytes`], the store representation here is a single
// length-prefixed leaf rather than a page tree, so the *store* content hash is
// `hash_bytes(bincode(bytes))`, which differs from the *value* hash `blake3::hash(bytes)`
// used by `HashFold`. This is fine for the durable-storage AVL (which persists values via
// `Storable`/`DataLoadable`, not this fold-store) but means a PVM-state switch-over would
// want a variable-length raw-leaf store representation so store-hash == value-hash. See the
// report / docs.

impl<BS: BlobStore> Foldable<BlobStoreFold<BS>> for Blake3Bytes<Normal> {
    fn fold(&self, builder: BlobStoreFold<BS>) -> <BlobStoreFold<BS> as Fold>::Folded {
        let bytes = self.as_bytes().to_vec();
        builder
            .fold_leaf(&bytes)
            .expect("Serialising bytes should not fail")
    }
}

impl Unfoldable for Blake3Bytes<Normal> {
    fn unfold<U: Unfold>(source: U) -> Result<Self, UnfoldError> {
        let bytes = source.into_leaf::<Vec<u8>>()?;
        Ok(Blake3Bytes::from(bytes.as_slice()))
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
#[expect(
    clippy::single_range_in_vec_init,
    reason = "Accessed ranges are passed as one-element slices throughout, which is what these tests mean; the lint reads them as a mistyped range"
)]
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
        /// I8/S0: Normal-mode hash is `combine(H_len, blake3::hash(bytes))` — the length
        /// commitment — and is NOT bare `blake3::hash(bytes)`.
        #[test]
        fn hash_value_commits_length(data in bytes_of_len(3 * CHUNK_LEN + 11)) {
            let expected =
                Hash::combine_hashes([hash_len(data.len()), Hash::hash_bytes(&data)]);
            prop_assert_eq!(hash_value(&data), expected);
            prop_assert_ne!(hash_value(&data), Hash::hash_bytes(&data));
        }

        /// I8/P0: prove-then-verify reproduces the Normal-mode root, for any access set.
        #[test]
        fn prove_then_verify_reproduces_hash((data, access) in arb_data_and_access()) {
            let proof = prove(&data, &access);
            prop_assert_eq!(verify_root(&proof).unwrap(), hash_value(&data));
        }

        /// P3/I2/I3: tampering with the claimed length makes verification reject or
        /// diverge from the committed root.
        #[test]
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
        fn blinded_read_faults(data in bytes_of_len(4 * CHUNK_LEN)) {
            // Access only the first chunk; the rest is blinded.
            let proof = prove(&data, &[0..1]);
            let err = proof_read(&proof, (3 * CHUNK_LEN)..(4 * CHUNK_LEN));
            prop_assert!(matches!(err, Err(ProofError::Blinded)));
        }

        /// P0/P5: accessed ranges are present and read back exactly.
        #[test]
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
            prop_assume!(tamper(&mut proof.data));
            match verify_root(&proof) {
                Err(_) => {}
                Ok(root) => prop_assert_ne!(root, committed),
            }
        }

        /// I5: a present chunk larger than CHUNK_LEN must be rejected (memory safety).
        #[test]
        fn oversized_chunk_rejected(extra in 1usize..64) {
            let proof = Blake3Proof {
                total_len: CHUNK_LEN + extra,
                data: ProofTree::Chunk(vec![0u8; CHUNK_LEN + extra]),
            };
            prop_assert_eq!(verify_root(&proof), Err(ProofError::ChunkTooLarge));
        }

        /// P0/P1: tampering with PRESENT chunk bytes changes the recomputed root (or is
        /// rejected). This is the guard against a `verify_root` that never actually hashes
        /// present chunk contents — such an implementation would pass every *other* test
        /// (round-trip, blinded-CV tamper, length tamper) while being completely wrong.
        #[test]
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
            prop_assume!(flip_first_chunk(&mut proof.data));
            match verify_root(&proof) {
                Err(_) => {}
                Ok(root) => prop_assert_ne!(root, committed),
            }
        }

        /// I4/P5: reading ANY range wholly outside the accessed set faults with `Blinded`
        /// — the property-level generalisation of `blinded_read_faults`. Guards against a
        /// verify that silently invents data (e.g. zero-fills) for unproven regions.
        #[test]
        fn unaccessed_read_faults(a in (2 * CHUNK_LEN)..(8 * CHUNK_LEN), len in 1usize..=CHUNK_LEN) {
            let data = vec![7u8; 8 * CHUNK_LEN];
            let proof = prove(&data, &[0..1]); // only chunk 0 present; rest blinded
            let end = (a + len).min(data.len());
            prop_assume!(end > a);
            prop_assert!(matches!(proof_read(&proof, a..end), Err(ProofError::Blinded)));
        }

        /// I4: reading past `total_len` faults with `OutOfBounds`; never returns padding.
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

        /// P1 anti-collapse: two different values of the same length verify to DIFFERENT
        /// roots, each equal to its own `hash_value`. Guards against a `verify_root` that
        /// returns a constant, echoes `total_len`, or otherwise ignores content.
        #[test]
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

    /// `hash_value` on a value above the parallel-hashing threshold must equal the reference
    /// `combine(H_len, blake3::hash(bytes))`. This guards the rayon path (feature on) and the
    /// single-thread path (feature off) alike: `update_rayon` must be byte-identical to `hash`.
    #[test]
    fn hash_value_large_matches_reference() {
        for len in [255usize * 1024, 256 * 1024, 512 * 1024 + 7, 2 * 1024 * 1024] {
            let data: Vec<u8> = (0..len)
                .map(|i| i.wrapping_mul(0x9E3779B1usize) as u8)
                .collect();
            let reference =
                Hash::combine_hashes([hash_len(len), Hash::from(*blake3::hash(&data).as_bytes())]);
            assert_eq!(
                hash_value(&data),
                reference,
                "hash_value drift at len {len}"
            );
        }
    }

    /// The small-length cache must return exactly what a fresh `blake3` of the LE `u64` length
    /// would, across the whole cached range, its boundary, and beyond (cache miss path). A drift
    /// here would silently change every value hash — a consensus break.
    #[test]
    fn hash_len_cache_matches_direct() {
        for len in (0..=CHUNK_LEN + 2)
            .chain([4096, 65536, 1_000_000, usize::MAX])
            .collect::<Vec<_>>()
        {
            let direct = Hash::hash_bytes(&(len as u64).to_le_bytes());
            assert_eq!(hash_len(len), direct, "hash_len cache drift at len {len}");
        }
    }

    /// DoS hardening: a proof with `ProofTree` nesting beyond [`MAX_PROOF_TREE_DEPTH`] must be
    /// rejected by the decoder (returning an error) rather than overflowing the stack. Builds and
    /// serialises the deep tree iteratively (the `Encode` impl is non-recursive), then checks that
    /// decoding it errors cleanly. Guards both the standalone and AVL-wire decode paths.
    #[test]
    fn overly_deep_proof_tree_rejected() {
        use crate::serialisation::deserialise;
        use crate::serialisation::serialise;

        let mut tree = ProofTree::Blind([0u8; 32]);
        for _ in 0..(MAX_PROOF_TREE_DEPTH + 50) {
            tree = ProofTree::Node(Box::new(tree), Box::new(ProofTree::Blind([0u8; 32])));
        }
        let proof = Blake3Proof {
            total_len: 4 * CHUNK_LEN,
            data: tree,
        };
        let bytes = serialise(&proof).expect("encode is iterative, does not overflow");
        assert!(
            deserialise::<Blake3Proof>(&bytes).is_err(),
            "decoder accepted an over-deep proof tree instead of rejecting it"
        );
    }

    /// A canonical proof at a realistic depth still round-trips through the decoder (the depth cap
    /// does not reject well-formed proofs).
    #[test]
    fn deep_but_canonical_proof_roundtrips() {
        use crate::serialisation::deserialise;
        use crate::serialisation::serialise;
        // 2^20 chunks = 1 GiB value => canonical depth 20, far below the cap; every chunk present.
        let data = vec![0x33u8; 64 * CHUNK_LEN];
        let proof = prove(&data, &[0..data.len()]);
        let bytes = serialise(&proof).expect("serialises");
        let decoded = deserialise::<Blake3Proof>(&bytes).expect("well-formed proof decodes");
        assert_eq!(verify_root(&decoded).unwrap(), hash_value(&data));
    }

    /// I2: a structurally non-canonical proof is rejected. A two-chunk value's canonical
    /// shape is `Node(Chunk, Chunk)`; a bare single `Chunk` claiming `total_len =
    /// 2*CHUNK_LEN` is non-canonical.
    #[test]
    fn non_canonical_shape_rejected() {
        let proof = Blake3Proof {
            total_len: 2 * CHUNK_LEN,
            data: ProofTree::Chunk(vec![0u8; CHUNK_LEN]),
        };
        assert_eq!(verify_root(&proof), Err(ProofError::NonCanonicalShape));
    }

    /// S2 regression — the **cross-span graft** that broke the predecessor "no length node"
    /// scheme is now rejected. Honest value: 3 full chunks, so the honest data root is
    /// `merge_root(cv[0..2], cv2)`. Forge `total_len = 4096` (4 chunks); `canonical(4096)`
    /// splits `2048 | 2048`, and the right `[2048, 4096)` slot accepts a `Blind`. Plant the
    /// genuine chunk-2 CV there: `data_root` reconstructs the *identical* honest data root — but
    /// the length commitment differs (`H_len(4096) != H_len(3072)`), so the value hash differs
    /// and the forgery is rejected. Under the old scheme this reproduced the honest root exactly.
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

        // The data tree reconstructs the honest data root, but the committed value hash binds
        // the length, so the forged length yields a different hash.
        match verify_root(&forged) {
            Err(_) => {}
            Ok(root) => assert_ne!(root, committed, "cross-span graft forged the length!"),
        }
    }

    /// S2: for a *wide* range of forged lengths built as the strongest constructed graft
    /// (canonical shape for `L'`, every slot filled with the true value's genuine CV/bytes),
    /// verification always errors or diverges from the committed hash.
    #[test]
    fn constructed_graft_any_length_rejected() {
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

    /// The strongest attacker's proof: build `canonical(L')` and fill every position with the
    /// true value's genuine CV/bytes for that byte range (cross-span where the spans differ).
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

    /// I8/S0 edge case: the empty value hashes to `combine(H_len(0), blake3::hash(&[]))` and
    /// its proof round-trips.
    #[test]
    fn empty_value_roundtrips() {
        let data: &[u8] = &[];
        let proof = prove(data, &[]);
        assert_eq!(verify_root(&proof).unwrap(), hash_value(data));
        assert_eq!(
            hash_value(data),
            Hash::combine_hashes([hash_len(0), Hash::hash_bytes(data)])
        );
    }
}

// ---------------------------------------------------------------------------------------
// Fold-pipeline wiring tests — the deferred integration step.
//
// These exercise the mode-generic `Blake3Bytes` through the fold traits, demonstrating the
// full Normal == Prove == Verify round-trip *with partial proofs*, that the value proof
// serialises/deserialises and survives tampering, that verify-mode reads fault on blinded
// regions (I4), and that the value contributes `blake3::hash(bytes)` when embedded as an
// AVL-node child (docs §3). They complement (do not replace) the free-function property
// tests above.
// ---------------------------------------------------------------------------------------
#[cfg(test)]
mod fold_tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use blake3::CHUNK_LEN;
    use proptest::prelude::*;

    use super::*;
    use crate::hash::HashedData;
    use crate::merkle_proof::proof_binary;
    use crate::merkle_proof::proof_tree;
    use crate::merkle_proof::proof_tree::MerkleProof;
    use crate::mode::utils::catch_not_found;
    use crate::serialisation::deserialise;
    use crate::serialisation::serialise;
    use crate::store::BlobStoreError;
    use crate::store::unfold::BlobStoreUnfold;

    /// Minimal in-crate content-addressed store for the fold/unfold test (the crate's own
    /// `InMemoryBlobStore::new` is private to the `store` module).
    #[derive(Default)]
    struct TestStore(Mutex<HashMap<Hash, Vec<u8>>>);

    impl BlobStore for TestStore {
        fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
            self.0
                .lock()
                .expect("lock poisoned")
                .get(&key)
                .cloned()
                .ok_or(BlobStoreError::NotFound(key))
        }

        fn blob_set<Data: AsRef<[u8]>>(
            &self,
            blob: &HashedData<Data>,
        ) -> Result<(), BlobStoreError> {
            self.0
                .lock()
                .expect("lock poisoned")
                .insert(blob.hash(), blob.data().to_vec());
            Ok(())
        }

        fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError> {
            self.0.lock().expect("lock poisoned").remove(&key);
            Ok(())
        }
    }

    fn interesting_len() -> impl Strategy<Value = usize> {
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

    fn bytes_of_len(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(any::<u8>(), len..=len)
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

    /// Read `range` from a verify-mode value, catching the `not_found` fault.
    fn verify_read(value: &Blake3Bytes<Verify>, range: Range<usize>) -> Option<Vec<u8>> {
        catch_not_found(|| {
            let mut buf = vec![0u8; range.len()];
            let read = value.read(range.start, &mut buf);
            buf.truncate(read);
            buf
        })
        .ok()
    }

    /// I8/§2: `HashFold` on a Normal value is the committed value hash
    /// `combine(H_len, blake3::hash(bytes))`.
    #[test]
    fn normal_hashfold_commits_length() {
        let data = vec![0xABu8; 3 * CHUNK_LEN + 17];
        let value = Blake3Bytes::<Normal>::from(data.as_slice());
        assert_eq!(Hash::from_foldable(&value), hash_value(&data));
    }

    /// §2: embedded as an AVL-node child, the value contributes exactly its committed value
    /// hash `hash_value(bytes)`. A node folds via `combine_hashes(child_hashes)`, so a
    /// `(key_hash, data)` node must hash to `combine_hashes([key_hash, hash_value(bytes)])`.
    #[test]
    fn value_contributes_value_hash_to_node_hash() {
        let data = vec![7u8; 2 * CHUNK_LEN + 3];
        let key_hash = Hash::hash_bytes(b"some-key");
        let value = Blake3Bytes::<Normal>::from(data.as_slice());

        let node_hash = Hash::from_foldable(&(key_hash, value));
        let expected = Hash::combine_hashes([key_hash, hash_value(&data)]);
        assert_eq!(node_hash, expected);
    }

    proptest! {
        /// I8/P0: for a read-only transition, Normal == Prove == Verify roots agree, and every
        /// accessed range reads back exactly through both the raw proof and the verify view.
        #[test]
        fn normal_prove_verify_agree((data, access) in data_and_access()) {
            let normal = Blake3Bytes::<Normal>::from(data.as_slice());
            let normal_root = Hash::from_foldable(&normal);

            // Prove mode: replay the reads, then fold + capture the value proof.
            let prover = normal.start_proof();
            for r in &access {
                let mut buf = vec![0u8; r.len()];
                let read = prover.read(r.start, &mut buf);
                prop_assert_eq!(read, r.len());
                prop_assert_eq!(&buf[..], &data[r.clone()]);
            }
            let prove_root = Hash::from_foldable(&prover);
            let proof = prover.value_proof();
            prop_assert_eq!(prove_root, normal_root);
            prop_assert_eq!(verify_root(&proof).unwrap(), normal_root);

            // Verify mode: reconstruct from the proof and fold.
            let verify = Blake3Bytes::<Verify>::from_proof_checked(proof, normal_root).unwrap();
            let verify_root_hash = PartialHash::from_foldable(None, &verify)
                .to_hash()
                .expect("verify fold should produce a present hash");
            prop_assert_eq!(verify_root_hash, normal_root);

            // Accessed ranges read back through the verify view.
            for r in &access {
                let got = verify_read(&verify, r.clone())
                    .expect("accessed range must be present in verify mode");
                prop_assert_eq!(got.as_slice(), &data[r.clone()]);
            }
        }

        /// The value proof serialises and deserialises, reconstructing the committed root.
        #[test]
        fn value_proof_serialisation_roundtrip((data, access) in data_and_access()) {
            let root = hash_value(&data);
            let prover = Blake3Bytes::<Prove>::from_raw_source(&data);
            for r in &access {
                let mut buf = vec![0u8; r.len()];
                prover.read(r.start, &mut buf);
            }
            let proof = prover.value_proof();

            let encoded = serialise(&proof).unwrap();
            let decoded: Blake3Proof = deserialise(&encoded).unwrap();
            prop_assert_eq!(&decoded, &proof);
            prop_assert_eq!(verify_root(&decoded).unwrap(), root);
        }

        /// Tampering with a serialised proof either fails to decode or diverges from the root.
        #[test]
        fn tampered_serialised_proof_detected(
            data in bytes_of_len(4 * CHUNK_LEN),
            idx in 0usize..64,
        ) {
            let root = hash_value(&data);
            let prover = Blake3Bytes::<Prove>::from_raw_source(&data);
            let mut buf = [0u8; 8];
            prover.read(0, &mut buf); // access chunk 0 only; rest blinded
            let proof = prover.value_proof();

            let mut encoded = serialise(&proof).unwrap();
            let pos = idx % encoded.len();
            encoded[pos] ^= 0x80;

            match deserialise::<Blake3Proof>(&encoded) {
                Err(_) => {}
                Ok(tampered) => match verify_root(&tampered) {
                    Err(_) => {}
                    Ok(got) => prop_assert_ne!(got, root),
                },
            }
        }
    }

    /// I4/P5: a verify-mode read of a blinded (never-accessed) region faults.
    #[test]
    fn verify_blinded_read_faults() {
        let data = vec![9u8; 4 * CHUNK_LEN];
        let root = hash_value(&data);
        let prover = Blake3Bytes::<Prove>::from_raw_source(&data);
        let mut buf = [0u8; 1];
        prover.read(0, &mut buf); // only chunk 0 present
        let proof = prover.value_proof();

        let verify = Blake3Bytes::<Verify>::from_proof_checked(proof, root).unwrap();
        // Chunk 0 is present.
        assert!(verify_read(&verify, 0..1).is_some());
        // Chunk 3 was blinded.
        assert!(verify_read(&verify, (3 * CHUNK_LEN)..(3 * CHUNK_LEN + 1)).is_none());
    }

    /// I6: decoding a proof whose present chunk exceeds `CHUNK_LEN` is rejected before
    /// allocation.
    #[test]
    fn decode_rejects_oversized_chunk() {
        let proof = Blake3Proof {
            total_len: CHUNK_LEN + 5,
            data: ProofTree::Chunk(vec![0u8; CHUNK_LEN + 5]),
        };
        let encoded = serialise(&proof).unwrap();
        assert!(deserialise::<Blake3Proof>(&encoded).is_err());
    }

    /// A write-carrying transition: Prove (post) == Verify (post) == a Normal value written
    /// the same way. Exercises `current_root` recomputation over written present chunks.
    #[test]
    fn write_transition_agrees() {
        let original = vec![1u8; 3 * CHUNK_LEN];
        let patch = vec![0xEEu8; 16];
        let write_at = CHUNK_LEN; // chunk 1

        // Reference post-transition value in Normal mode.
        let mut normal_post = Blake3Bytes::<Normal>::from(original.as_slice());
        normal_post.write(write_at, &patch);
        let post_root = normal_post.hash();

        // Prove mode over the pre-transition data.
        let mut prover = Blake3Bytes::<Prove>::from_raw_source(&original);
        prover.write(write_at, &patch);
        assert_eq!(Hash::from_foldable(&prover), post_root);

        let pre_root = hash_value(&original);
        let proof = prover.value_proof();
        assert_eq!(verify_root(&proof).unwrap(), pre_root);

        // Verify mode: reconstruct from the pre-transition proof, replay the write.
        let mut verify = Blake3Bytes::<Verify>::from_proof_checked(proof, pre_root).unwrap();
        let written = verify.write(write_at, &patch);
        assert_eq!(written, patch.len());
        let verify_post = PartialHash::from_foldable(None, &verify)
            .to_hash()
            .expect("verify post fold should be present");
        assert_eq!(verify_post, post_root);
    }

    /// Store fold/unfold round-trips the byte content (PVM-state persistence path).
    #[test]
    fn store_fold_unfold_roundtrip() {
        use std::sync::Arc;

        let data = vec![0x33u8; 5 * CHUNK_LEN + 9];
        let value = Blake3Bytes::<Normal>::from(data.as_slice());

        let store = Arc::new(TestStore::default());
        let root = value.fold(BlobStoreFold::from(Arc::clone(&store))).unwrap();

        let unfolded = Blake3Bytes::<Normal>::unfold(BlobStoreUnfold::new(store, root)).unwrap();
        assert_eq!(unfolded.as_bytes(), data.as_slice());
    }

    /// Basic mode operations behave consistently.
    #[test]
    fn mode_ops_basic() {
        let mut n = Blake3Bytes::<Normal>::new(10);
        assert_eq!(n.len(), 10);
        assert_eq!(n.write(2, &[1, 2, 3]), 3);
        let mut buf = [0u8; 3];
        assert_eq!(n.read(2, &mut buf), 3);
        assert_eq!(buf, [1, 2, 3]);
        n.resize(4);
        assert_eq!(n.len(), 4);
        n.resize(6);
        assert_eq!(n.len(), 6);
        let mut tail = [9u8; 2];
        n.read(4, &mut tail);
        assert_eq!(tail, [0, 0]); // grown bytes are zero
    }

    // ------------------------------ AVL proof-wire round-trips ------------------------------
    //
    // These drive the value through the *generic* `MerkleProof` wire (the AVL `data` slot),
    // not just its own `Blake3Proof` serialisation: Prove `MerkleProofFold` → serialise →
    // both deserialisers → Verify `FromProof` + `PartialHashFold`. See docs §7.3.

    proptest! {
        /// Round-trip through the binary (stream) wire: Normal == proof root == verify hash, and
        /// accessed ranges read back through the reconstructed verify view.
        #[test]
        fn avl_wire_stream_roundtrip((data, access) in data_and_access()) {
            let normal = Blake3Bytes::<Normal>::from(data.as_slice());
            let normal_root = Hash::from_foldable(&normal);

            let prover = normal.start_proof();
            for r in &access {
                let mut buf = vec![0u8; r.len()];
                prover.read(r.start, &mut buf);
            }

            let proof_tree = MerkleProof::from_foldable(&prover);
            prop_assert_eq!(proof_tree.root_hash(), normal_root);

            let bytes = serialise(&proof_tree).unwrap();
            let (verify, parsed) =
                proof_binary::deserialise::<crate::codec::Bincode, Blake3Bytes<Verify>>(&bytes).unwrap();
            let parsed = parsed.into_present();

            let verify_hash = PartialHash::from_foldable(parsed, &verify)
                .to_hash()
                .expect("verify fold should be present");
            prop_assert_eq!(verify_hash, normal_root);

            for r in &access {
                let got = verify_read(&verify, r.clone()).expect("accessed range present");
                prop_assert_eq!(got.as_slice(), &data[r.clone()]);
            }
        }

        /// Same round-trip via the in-memory proof-tree deserialiser (`ProofTree::into_blake3_leaf`).
        #[test]
        fn avl_wire_in_memory_roundtrip((data, access) in data_and_access()) {
            let normal = Blake3Bytes::<Normal>::from(data.as_slice());
            let normal_root = Hash::from_foldable(&normal);

            let prover = normal.start_proof();
            for r in &access {
                let mut buf = vec![0u8; r.len()];
                prover.read(r.start, &mut buf);
            }
            let proof_tree = MerkleProof::from_foldable(&prover);

            let (verify, parsed) =
                proof_tree::deserialise::<crate::codec::Bincode, Blake3Bytes<Verify>>(crate::merkle_proof::proof_tree::ProofTree::present(&proof_tree))
                    .unwrap();
            let parsed = parsed.into_present();
            let verify_hash = PartialHash::from_foldable(parsed, &verify).to_hash().unwrap();
            prop_assert_eq!(verify_hash, normal_root);
            for r in &access {
                let got = verify_read(&verify, r.clone()).expect("accessed range present");
                prop_assert_eq!(got.as_slice(), &data[r.clone()]);
            }
        }
    }

    /// A completely untouched value collapses to a single blind of its committed value hash.
    #[test]
    fn avl_wire_untouched_value_is_blinded() {
        let data = vec![7u8; 5 * CHUNK_LEN + 3];
        let root = hash_value(&data);

        let normal = Blake3Bytes::<Normal>::from(data.as_slice());
        let prover = normal.start_proof(); // no access at all
        let proof_tree = MerkleProof::from_foldable(&prover);
        assert!(proof_tree.is_blind());
        assert_eq!(proof_tree.root_hash(), root);

        let bytes = serialise(&proof_tree).unwrap();
        let (verify, parsed) =
            proof_binary::deserialise::<crate::codec::Bincode, Blake3Bytes<Verify>>(&bytes)
                .unwrap();
        let verify_hash = PartialHash::from_foldable(parsed.into_present(), &verify)
            .to_hash()
            .unwrap();
        assert_eq!(verify_hash, root);
    }

    /// Reading only the length keeps the value present (the length must be recoverable in verify
    /// mode) while the data stays fully blinded.
    #[test]
    fn avl_wire_length_only_access_keeps_length() {
        let data = vec![9u8; 4 * CHUNK_LEN + 10];
        let root = hash_value(&data);

        let normal = Blake3Bytes::<Normal>::from(data.as_slice());
        let prover = normal.start_proof();
        let _ = prover.len(); // only the length is read

        let proof_tree = MerkleProof::from_foldable(&prover);
        assert!(!proof_tree.is_blind()); // present Blake3 leaf, not blinded
        assert_eq!(proof_tree.root_hash(), root);

        let bytes = serialise(&proof_tree).unwrap();
        let (verify, parsed) =
            proof_binary::deserialise::<crate::codec::Bincode, Blake3Bytes<Verify>>(&bytes)
                .unwrap();
        assert_eq!(verify.len(), data.len()); // length available in verify mode
        let verify_hash = PartialHash::from_foldable(parsed.into_present(), &verify)
            .to_hash()
            .unwrap();
        assert_eq!(verify_hash, root);
        // The data itself was blinded, so reading it faults.
        assert!(verify_read(&verify, 0..1).is_none());
    }

    /// A value written under an AVL-style node (a two-child tuple): the written child is a present
    /// BLAKE3 leaf, the untouched sibling is blinded, and pre/post hashes agree across
    /// Normal / Prove / Verify — exercising the node combine and blinded-sibling deferral.
    #[test]
    fn avl_wire_node_write_and_blinded_sibling() {
        let a0 = vec![1u8; 3 * CHUNK_LEN];
        let b0 = vec![2u8; 2 * CHUNK_LEN + 5];
        let patch = vec![0xEEu8; 16];
        let at = CHUNK_LEN; // chunk 1 of `a`

        let pre_root = Hash::from_foldable(&(
            Blake3Bytes::<Normal>::from(a0.as_slice()),
            Blake3Bytes::<Normal>::from(b0.as_slice()),
        ));

        // Prove: write into `a`, leave `b` untouched.
        let mut prover = (
            Blake3Bytes::<Prove>::from_raw_source(&a0),
            Blake3Bytes::<Prove>::from_raw_source(&b0),
        );
        prover.0.write(at, &patch);

        let post_prove_root = Hash::from_foldable(&prover);
        let proof_tree = MerkleProof::from_foldable(&prover);
        assert_eq!(proof_tree.root_hash(), pre_root);

        let bytes = serialise(&proof_tree).unwrap();
        let (mut verify, parsed) = proof_binary::deserialise::<
            crate::codec::Bincode,
            (Blake3Bytes<Verify>, Blake3Bytes<Verify>),
        >(&bytes)
        .unwrap();
        let parsed = parsed.into_present();

        let pre_verify = PartialHash::from_foldable(parsed.clone(), &verify)
            .to_hash()
            .unwrap();
        assert_eq!(pre_verify, pre_root);

        verify.0.write(at, &patch);
        let post_verify = PartialHash::from_foldable(parsed, &verify)
            .to_hash()
            .unwrap();
        assert_eq!(post_verify, post_prove_root);

        // Cross-check against Normal mode applying the same write.
        let mut a_post = Blake3Bytes::<Normal>::from(a0.as_slice());
        a_post.write(at, &patch);
        let post_normal =
            Hash::from_foldable(&(a_post, Blake3Bytes::<Normal>::from(b0.as_slice())));
        assert_eq!(post_verify, post_normal);
    }
}
