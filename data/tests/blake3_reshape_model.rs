// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT
//! Reference model + proof campaign for the **length-reshape proof-size optimisation** of the
//! length-committed BLAKE3 `Bytes` scheme (design: `data/docs/blake3-bytes-reshape-optimisation.md`).
//!
//! Problem: today a length-*changing* transition (grow / shrink) forces the whole pre-transition
//! value present in the proof (see `data/tests/blake3_reshape_proof_size.rs`), because
//! `current_root`'s length-changed branch re-hashes from full bytes. This model validates that it
//! can instead stay `O(depth)` by reusing the pre-proof's blinded prefix subtree chaining values —
//! which are **length-independent** — when recomputing the post-transition root.
//!
//! This file is self-contained (independent of the production `blake3_bytes.rs`) so the campaign
//! runs BEFORE the optimisation is implemented. It is the executable form of the design's
//! invariants. Run hard with `PROPTEST_CASES=100000`.
//!
//! Structure mirrors `blake3_length_committed_model.rs`:
//!   * Group A (runnable now): the pure-BLAKE3 facts the optimisation leans on.
//!   * Group B (the model + campaign): the optimised prove / reshape-recompute, round-trip, the
//!     O(depth) size property, and the length-forgery campaign (reusing CVs across lengths must not
//!     forge). Some Group B items are the spec the implementing session must satisfy.

use std::collections::BTreeMap;

use blake3::CHUNK_LEN;
use blake3::hazmat::HasherExt;
use blake3::hazmat::Mode as B3;
use blake3::hazmat::left_subtree_len;
use blake3::hazmat::merge_subtrees_non_root;
use blake3::hazmat::merge_subtrees_root;
use proptest::prelude::*;

// ----------------------------------------------------------------------------------------
// Shared helpers (identical scheme to blake3_length_committed_model.rs; verify is UNCHANGED)
// ----------------------------------------------------------------------------------------

fn h_len(n: usize) -> [u8; 32] {
    *blake3::hash(&(n as u64).to_le_bytes()).as_bytes()
}
fn combine(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(a);
    h.update(b);
    *h.finalize().as_bytes()
}
fn value_hash(bytes: &[u8]) -> [u8; 32] {
    combine(&h_len(bytes.len()), blake3::hash(bytes).as_bytes())
}

/// Non-root chaining value of the canonical subtree over `data` starting at byte `offset` (a chunk
/// boundary). Depends ONLY on `(data, offset)` — never on any enclosing value's total length. This
/// length-independence is the whole basis of the optimisation.
fn canonical_cv(data: &[u8], offset: u64) -> [u8; 32] {
    if data.len() <= CHUNK_LEN {
        return blake3::Hasher::new()
            .set_input_offset(offset)
            .update(data)
            .finalize_non_root();
    }
    let ll = left_subtree_len(data.len() as u64) as usize;
    let (l, r) = data.split_at(ll);
    merge_subtrees_non_root(
        &canonical_cv(l, offset),
        &canonical_cv(r, offset + ll as u64),
        B3::Hash,
    )
}

// ----------------------------------------------------------------------------------------
// Group A: the pure-BLAKE3 foundation (RUN NOW). These must pass before the design is trusted.
// ----------------------------------------------------------------------------------------

/// Record `(offset_chunks, span_chunks) -> cv` for every node of `canonical(total_chunks)`, over a
/// full-chunk value of `total_chunks` chunks filled from `data` (len == total_chunks*CHUNK_LEN).
fn node_cvs(
    data: &[u8],
    off_chunks: usize,
    span_chunks: usize,
    out: &mut BTreeMap<(usize, usize), [u8; 32]>,
) {
    let off = off_chunks * CHUNK_LEN;
    let span = span_chunks * CHUNK_LEN;
    let cv = canonical_cv(&data[off..off + span], off as u64);
    out.insert((off_chunks, span_chunks), cv);
    if span_chunks > 1 {
        let ll = (left_subtree_len(span as u64) as usize) / CHUNK_LEN;
        node_cvs(data, off_chunks, ll, out);
        node_cvs(data, off_chunks + ll, span_chunks - ll, out);
    }
}

/// The aligned power-of-two subtrees that tile `[0, p)` chunks: one per set bit of `p`, MSB→LSB.
/// This is `canonical(p)`'s own left-spine decomposition and has `popcount(p)` pieces.
fn set_bit_subtrees(p_chunks: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut offset = 0;
    let mut remaining = p_chunks;
    // Largest power of two <= remaining, descending.
    for bit in (0..usize::BITS).rev() {
        let s = 1usize << bit;
        if remaining & s != 0 {
            pieces.push((offset, s));
            offset += s;
            remaining -= s;
        }
    }
    pieces
}

fn buf(chunks: usize) -> Vec<u8> {
    (0..chunks * CHUNK_LEN)
        .map(|i| (i.wrapping_mul(2654435761)) as u8)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 4000, ..ProptestConfig::default() })]

    /// FOUNDATION 1 — length independence: the CV of an aligned prefix subtree `[0, s)` is the same
    /// whether it sits inside a value of `s`, `p`, or `n` chunks. (`canonical_cv` proves it by
    /// construction; this pins it against differently-sized enclosing trees.)
    #[test]
    fn aligned_prefix_cv_is_length_independent(bits in 1usize..=6, extra in 0usize..40) {
        let s = 1usize << bits;                 // aligned power-of-two span (chunks)
        let n = s + extra;                      // a larger enclosing value
        let data = buf(n);
        let standalone = canonical_cv(&data[..s * CHUNK_LEN], 0);
        let mut nodes = BTreeMap::new();
        node_cvs(&data, 0, n, &mut nodes);
        // [0, s) is a node of canonical(n) and carries the standalone CV.
        prop_assert_eq!(nodes.get(&(0, s)).copied(), Some(standalone));
    }

    /// FOUNDATION 2 — the set-bit decomposition of the *unchanged prefix* `[0, p)` is a common
    /// sub-forest of `canonical(n)` for EVERY `n >= p`. So blinding the prefix at set-bit-of-`p`
    /// granularity yields CVs reusable at any post-transition length. This is what makes the reshape
    /// proof O(depth): `popcount(p)` prefix blinds serve both the pre (length `Lp`) and post
    /// (length `Lq`) shapes for `p = min(Lp, Lq)`.
    #[test]
    fn prefix_setbit_subtrees_are_nodes_of_every_larger_tree(p in 1usize..=48, extra in 0usize..48) {
        let n = p + extra;
        let data = buf(n);
        let mut nodes = BTreeMap::new();
        node_cvs(&data, 0, n, &mut nodes);
        let pieces = set_bit_subtrees(p);
        // O(depth): at most ceil(log2(p))+1 pieces.
        prop_assert!(pieces.len() as u32 <= (usize::BITS - p.leading_zeros()));
        // Each prefix piece is a node of canonical(n) with the length-independent CV.
        for (off, span) in pieces {
            let expected = canonical_cv(&data[off * CHUNK_LEN..(off + span) * CHUNK_LEN], (off * CHUNK_LEN) as u64);
            prop_assert_eq!(
                nodes.get(&(off, span)).copied(),
                Some(expected),
                "prefix subtree ({},{}) not a reusable node of canonical({})",
                off,
                span,
                n
            );
        }
    }

    /// FOUNDATION 3 — reconstructing `canonical(Lq)` from the prefix set-bit blinds `[0, min)` plus
    /// the changed region reproduces `blake3::hash` of the reshaped bytes. Models the verifier's
    /// post-transition recompute for a GROW (append zeros); the prefix pieces are supplied as their
    /// (blinded) CVs, the new chunks are hashed live.
    #[test]
    fn grow_recompute_from_prefix_blinds_matches_hash(lp in 1usize..=24, add in 1usize..=24) {
        let lq = lp + add;
        let pre = buf(lp);
        // Post value: pre bytes followed by `add` zero chunks.
        let mut post = pre.clone();
        post.resize(lq * CHUNK_LEN, 0);

        // Verifier's inputs: prefix pieces as CVs (from the pre proof), new region as live bytes.
        let prefix_pieces: BTreeMap<(usize, usize), [u8; 32]> = set_bit_subtrees(lp)
            .into_iter()
            .map(|(o, s)| ((o, s), canonical_cv(&pre[o * CHUNK_LEN..(o + s) * CHUNK_LEN], (o * CHUNK_LEN) as u64)))
            .collect();

        let got = recompute_canonical(lq, &post, &prefix_pieces, lp);
        prop_assert_eq!(got, *blake3::hash(&post).as_bytes());
        // And the committed value hash matches end to end (H_len over the byte length).
        prop_assert_eq!(combine(&h_len(post.len()), &got), value_hash(&post));
    }

    /// FOUNDATION 3b — same for SHRINK (truncate): reconstruct canonical(Lq<Lp) from the prefix
    /// set-bit blinds of the *surviving* prefix `[0, Lq)`.
    #[test]
    fn shrink_recompute_from_prefix_blinds_matches_hash(lq in 1usize..=24, drop in 1usize..=24) {
        let lp = lq + drop;
        let pre = buf(lp);
        let post = pre[..lq * CHUNK_LEN].to_vec();

        let prefix_pieces: BTreeMap<(usize, usize), [u8; 32]> = set_bit_subtrees(lq)
            .into_iter()
            .map(|(o, s)| ((o, s), canonical_cv(&pre[o * CHUNK_LEN..(o + s) * CHUNK_LEN], (o * CHUNK_LEN) as u64)))
            .collect();

        let got = recompute_canonical(lq, &post, &prefix_pieces, lq);
        prop_assert_eq!(got, *blake3::hash(&post).as_bytes());
    }
}

/// Reference recompute: build `canonical(total_chunks)`'s ROOT output, taking a subtree's CV from
/// `prefix_pieces` when it is an unchanged-prefix node (offset+span <= `unchanged_chunks`), else
/// hashing the live bytes in `post`. This is the shape the optimised verifier `current_root` must
/// implement for the length-changed case (full chunks only in this model).
fn recompute_canonical(
    total_chunks: usize,
    post: &[u8],
    prefix_pieces: &BTreeMap<(usize, usize), [u8; 32]>,
    unchanged_chunks: usize,
) -> [u8; 32] {
    fn cv(
        post: &[u8],
        off_c: usize,
        span_c: usize,
        pieces: &BTreeMap<(usize, usize), [u8; 32]>,
        unchanged: usize,
    ) -> [u8; 32] {
        // Reuse a blinded prefix CV when this exact node lies fully in the unchanged prefix.
        if off_c + span_c <= unchanged
            && let Some(cv) = pieces.get(&(off_c, span_c))
        {
            return *cv;
        }
        let off = off_c * CHUNK_LEN;
        let span = span_c * CHUNK_LEN;
        if span_c == 1 {
            return blake3::Hasher::new()
                .set_input_offset(off as u64)
                .update(&post[off..off + span])
                .finalize_non_root();
        }
        let ll = (left_subtree_len(span as u64) as usize) / CHUNK_LEN;
        merge_subtrees_non_root(
            &cv(post, off_c, ll, pieces, unchanged),
            &cv(post, off_c + ll, span_c - ll, pieces, unchanged),
            B3::Hash,
        )
    }

    if total_chunks == 1 {
        return *blake3::hash(&post[..CHUNK_LEN]).as_bytes();
    }
    let ll = (left_subtree_len((total_chunks * CHUNK_LEN) as u64) as usize) / CHUNK_LEN;
    *merge_subtrees_root(
        &cv(post, 0, ll, prefix_pieces, unchanged_chunks),
        &cv(post, ll, total_chunks - ll, prefix_pieces, unchanged_chunks),
        B3::Hash,
    )
    .as_bytes()
}

// ----------------------------------------------------------------------------------------
// Group B: the optimised prove / reshape-recompute model + campaign.
//
// SPEC FOR THE IMPLEMENTING SESSION — extend the above full-chunk foundation into the production
// shape and prove the following. These are written against helpers the implementer will add; they
// are the acceptance criteria, not yet all runnable (see the doc §Test plan).
//
//   B1 round-trip (grow): for random pre value + grow-by-append (+ writes in the new/old region),
//       optimised-prove → verify(pre)==pre_root, then reshape-recompute==value_hash(post), with
//       proof size O(depth) (NOT O(value)).
//   B2 round-trip (shrink): as B1 for truncation; prefix refined to set-bit-of-Lq granularity.
//   B3 partial-final-chunk: pre and/or post have a ragged last chunk; the boundary chunk is kept
//       present; recompute still matches blake3::hash(post).
//   B4 size: assert present-chunks <= a+O(1), blinds <= c*depth for grow/shrink of 1/16/64 MiB
//       values with a <=3-chunk access — matching the in-place numbers (flip
//       `blake3_reshape_proof_size.rs` from "grow >= len" to the O(depth) bound).
//   B5 length-forgery campaign (SECURITY): reusing prefix CVs across lengths must NOT enable a
//       forged length. For random pre and every L' != Lp, the constructed strongest graft (as in
//       blake3_length_committed_model.rs) that additionally REUSES prefix blinds across the reshape
//       must still verify-fail or diverge from the committed root. The load-bearing invariant to
//       attack: a reused blind whose span exceeds min(Lp,Lq) (i.e. straddles changed bytes) must be
//       rejected / never reused (else a stale CV reproduces a wrong-but-accepted root).
// ----------------------------------------------------------------------------------------

#[test]
fn group_b_is_implemented_against_production() {
    // Group A (above) proves the pure-BLAKE3 foundation. Group B (B1..B5) is now implemented as a
    // runnable campaign against the production optimisation in `data/tests/blake3_reshape_campaign.rs`
    // (round-trip grow/shrink/partial-chunk + multi-resize, O(depth) size, and the tamper / R3
    // security tests), and the O(depth) size guard is `data/tests/blake3_reshape_proof_size.rs`.
    // This foundation lemma stays here as the executable spec of the set-bit decomposition both rely
    // on: popcount(0b1011) == 3 aligned pieces.
    assert_eq!(set_bit_subtrees(0b1011).len(), 3);
}
