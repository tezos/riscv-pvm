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
//! The modal component that folds these into the state pipelines follows.

use std::ops::Range;

use blake3::hazmat::HasherExt;
use blake3::hazmat::Mode as B3Mode;
use blake3::hazmat::left_subtree_len;
use blake3::hazmat::merge_subtrees_non_root;
use blake3::hazmat::merge_subtrees_root;

use crate::hash::Hash;

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

#[cfg(test)]
mod tests;
