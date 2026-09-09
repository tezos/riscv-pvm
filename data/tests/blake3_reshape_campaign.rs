// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT
//! Group B of the length-reshape proof-size optimisation: the **production** acceptance +
//! security campaign (the foundation, Group A, lives in `data/tests/blake3_reshape_model.rs`;
//! design: `data/docs/blake3-bytes-reshape-optimisation.md`).
//!
//! Everything here drives the real `Blake3Bytes` through the same Prove -> value-proof ->
//! Verify-replay pipeline the durable-storage AVL uses. Each test pairs a POSITIVE property
//! (honest reshape verifies to the true post hash, and the proof stays O(depth)) with a NEGATIVE
//! one (tampering the proof, or a shape the reuse must NOT trust, is rejected or diverges).
//!
//!   * B1/B2 round-trip (grow / shrink), incl. writes in the old and new regions.
//!   * B3 partial-final-chunk (ragged pre and/or post lengths).
//!   * Multi-resize: the reusable prefix is `[0, low_water)` (bytes never dropped). A
//!     shrink-*below*-then-regrow is the load-bearing hazard — `min(Lp,Lq)` overstates what is
//!     actually unchanged, so reuse is bounded by the low-water mark and the changed region is
//!     rebuilt from the verifier's replay (keeping it O(depth), no full re-hash). A net-length-
//!     unchanged reshape (e.g. 8192->0->8192) must NOT hit the unchanged-length fast path.
//!   * B4 size: a reshape of a large value is O(depth), not O(value).
//!   * B5 security: any single-bit tamper of an accepted reshape proof is rejected or diverges;
//!     and the R3 rule (never reuse a CV straddling the reshape boundary) holds — a proof that
//!     omits the boundary chunk cannot reproduce the honest post root.

use octez_riscv_data::components::blake3_bytes::Blake3Bytes;
use octez_riscv_data::components::blake3_bytes::Blake3BytesMode;
use octez_riscv_data::components::blake3_bytes::Blake3Proof;
use octez_riscv_data::components::blake3_bytes::ProofTree;
use octez_riscv_data::components::blake3_bytes::hash_value;
use octez_riscv_data::components::blake3_bytes::verify_root;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::serialise;
use proptest::prelude::*;

const CHUNK_LEN: usize = 1024;

/// A step of a transition, replayed identically in prove and verify mode.
#[derive(Clone, Debug)]
enum Op {
    Read(usize, usize),
    Write(usize, Vec<u8>),
    Resize(usize),
}

/// Apply the ops to a mode-generic value (prover or verifier).
fn apply<M: Blake3BytesMode>(value: &mut Blake3Bytes<M>, ops: &[Op]) {
    for op in ops {
        match op {
            Op::Read(start, len) => {
                let mut buf = vec![0u8; *len];
                value.read(*start, &mut buf);
            }
            Op::Write(start, data) => {
                value.write(*start, data);
            }
            Op::Resize(new_len) => value.resize(*new_len),
        }
    }
}

/// The exact post value the ops produce, computed independently of `Blake3Bytes` (the oracle).
/// Mirrors the component semantics: a write past the current length is a no-op and otherwise
/// clamps to it; resize zero-fills growth and truncates shrinkage.
fn oracle(pre: &[u8], ops: &[Op]) -> Vec<u8> {
    let mut v = pre.to_vec();
    for op in ops {
        match op {
            Op::Read(..) => {}
            Op::Write(start, data) => {
                if *start < v.len() {
                    let end = (start + data.len()).min(v.len());
                    v[*start..end].copy_from_slice(&data[..end - start]);
                }
            }
            Op::Resize(new_len) => v.resize(*new_len, 0),
        }
    }
    v
}

/// Run the full Prove -> value-proof -> Verify-replay pipeline and return
/// `(post_hash, verify_root, proof)`. `post_hash` is the prover's HashFold over the post state;
/// `verify_root` is what the verifier recomputes after replaying the same ops on the state it
/// reconstructed from the (pre-state) proof.
fn run(pre: &[u8], ops: &[Op]) -> (Hash, Option<Hash>, Blake3Proof) {
    let pre_root = hash_value(pre);

    let normal = Blake3Bytes::<Normal>::from(pre);
    let mut prover = normal.start_proof();
    apply(&mut prover, ops);
    let post_hash = Hash::from_foldable(&prover);

    // The within-value proof captures the PRE state and must reconstruct the committed pre root.
    let proof = prover.value_proof();
    assert_eq!(
        verify_root(&proof).expect("pre proof verifies"),
        pre_root,
        "value proof did not reconstruct the committed pre root"
    );

    // Reconstruct the verifier state from the proof (checked against the pre root), replay, fold.
    let mut verify = Blake3Bytes::<Verify>::from_proof_checked(proof.clone(), pre_root)
        .expect("proof checks against pre root");
    apply(&mut verify, ops);
    let verify_root = PartialHash::from_foldable(None, &verify).to_hash();

    (post_hash, verify_root, proof)
}

/// Positive round-trip: the verifier reproduces the true post hash (== oracle), for any ops.
fn assert_roundtrip(pre: &[u8], ops: &[Op]) {
    let (post_hash, verify_root, _) = run(pre, ops);
    assert_eq!(
        post_hash,
        hash_value(&oracle(pre, ops)),
        "prover post hash disagrees with the oracle"
    );
    assert_eq!(
        verify_root,
        Some(post_hash),
        "verifier reshape root != true post hash (pre={} ops={:?})",
        pre.len(),
        ops
    );
}

fn filled(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i.wrapping_mul(2654435761) ^ 0x5A) as u8)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// B1/B2/B3 — targeted round-trips across every interesting reshape shape.
// ---------------------------------------------------------------------------------------------

#[test]
fn grow_shrink_across_boundaries() {
    // Chunk counts around powers of two and ragged multi-level trees, plus sub-chunk sizes.
    let lens = [
        0usize,
        1,
        100,
        1023,
        1024,
        1025,
        2048,
        3000,
        4096,
        5 * 1024 + 7,
        8192,
        16384 + 13,
    ];
    for &pre_len in &lens {
        let pre = filled(pre_len);
        for &post_len in &lens {
            // Pure resize (append zeros / truncate).
            assert_roundtrip(&pre, &[Op::Resize(post_len)]);
            // Resize + a write in the (new) tail region.
            if post_len > 0 {
                let w = post_len.saturating_sub(20).min(post_len - 1);
                assert_roundtrip(&pre, &[Op::Resize(post_len), Op::Write(w, filled(30))]);
            }
            // A write in the surviving prefix, then resize (the prefix chunk must stay present).
            if pre_len > 40 {
                assert_roundtrip(
                    &pre,
                    &[Op::Write(pre_len / 2, filled(17)), Op::Resize(post_len)],
                );
            }
        }
    }
}

#[test]
fn grow_from_zero_and_shrink_to_zero() {
    assert_roundtrip(&[], &[Op::Resize(4096), Op::Write(0, filled(100))]);
    assert_roundtrip(&filled(4096), &[Op::Resize(0)]);
    assert_roundtrip(&filled(4096), &[Op::Resize(0), Op::Resize(2048)]);
}

// ---------------------------------------------------------------------------------------------
// Multi-resize — including the shrink-below-then-regrow hazard.
// ---------------------------------------------------------------------------------------------

#[test]
fn clean_multi_resize_reuses_prefix() {
    let pre = filled(10 * CHUNK_LEN);
    // grow then grow, shrink then shrink, grow then shrink above/below original: all "clean"
    // (low-water == min(Lp, final)).
    assert_roundtrip(
        &pre,
        &[Op::Resize(12 * CHUNK_LEN), Op::Resize(14 * CHUNK_LEN)],
    );
    assert_roundtrip(
        &pre,
        &[Op::Resize(6 * CHUNK_LEN), Op::Resize(3 * CHUNK_LEN)],
    );
    assert_roundtrip(
        &pre,
        &[Op::Resize(12 * CHUNK_LEN), Op::Resize(7 * CHUNK_LEN)],
    );
    assert_roundtrip(
        &pre,
        &[Op::Resize(12 * CHUNK_LEN), Op::Resize(2 * CHUNK_LEN)],
    );
}

/// The shrink-below-then-regrow reshape (the case that used to force the whole value present) is
/// now O(depth): the reusable prefix is `[0, low_water)` and the regrown region is rebuilt by the
/// verifier's replay, so the proof carries only boundary chunks — not the value.
#[test]
fn non_clean_reshape_is_o_depth() {
    let len = 4 * 1024 * 1024; // 4 MiB, depth 12
    let pre = filled(len);
    let depth = (len / CHUNK_LEN).next_power_of_two().trailing_zeros() as usize;
    // shrink to 3 chunks (drops almost everything), then grow back near the original length.
    let ops = vec![Op::Resize(3 * CHUNK_LEN), Op::Resize(len - 4096)];
    let (post_hash, verify_root, proof) = run(&pre, &ops);
    assert_eq!(verify_root, Some(post_hash));
    assert_eq!(post_hash, hash_value(&oracle(&pre, &ops)));
    let size = serialise(&proof).expect("serialises").len();
    let bound = 4 * CHUNK_LEN + 40 * (depth + 1);
    assert!(
        size <= bound,
        "non-clean reshape proof {size} B exceeded O(depth) bound {bound} B (was ~O(value)={len})"
    );
    assert!(
        size < len / 100,
        "proof {size} still scales with value {len}"
    );
}

#[test]
fn shrink_below_then_regrow_is_correct() {
    // THE hazard: shrink to 3 chunks (dropping chunks 3..10), then grow to 8 chunks (chunks
    // 3..8 come back as ZEROS). The naive min(Lp=10, Lq=8)=8-chunk "unchanged prefix" is wrong —
    // chunks 3..8 changed. The reuse boundary is low_water = 3; chunks 3..8 are rebuilt from the
    // verifier's replay (zeros), so the round-trip holds and the proof stays O(depth).
    let pre = filled(10 * CHUNK_LEN);
    assert_roundtrip(
        &pre,
        &[Op::Resize(3 * CHUNK_LEN), Op::Resize(8 * CHUNK_LEN)],
    );
    // With a write landing in the regrown (zero) region and in the surviving prefix.
    assert_roundtrip(
        &pre,
        &[
            Op::Resize(3 * CHUNK_LEN),
            Op::Resize(8 * CHUNK_LEN),
            Op::Write(5 * CHUNK_LEN, filled(50)),
            Op::Write(CHUNK_LEN, filled(50)),
        ],
    );
    // Ragged variant.
    assert_roundtrip(&pre, &[Op::Resize(2500), Op::Resize(9000)]);
    // NET-LENGTH-UNCHANGED reshape: shrink to 0 (or below) then regrow to the SAME length. This
    // returns `length == Lp` yet the value is now zeros, so it must NOT hit the unchanged-length
    // fast path (which would reuse stale pre CVs). Found by the 100k fuzzer.
    assert_roundtrip(&pre, &[Op::Resize(0), Op::Resize(10 * CHUNK_LEN)]);
    assert_roundtrip(
        &pre,
        &[Op::Resize(4 * CHUNK_LEN), Op::Resize(10 * CHUNK_LEN)],
    );
    assert_roundtrip(
        &filled(8192),
        &[Op::Resize(0), Op::Resize(8192), Op::Write(100, filled(50))],
    );
    // Grow-above-then-shrink-back stays unchanged (chunks below `length` never dropped).
    assert_roundtrip(
        &pre,
        &[Op::Resize(14 * CHUNK_LEN), Op::Resize(10 * CHUNK_LEN)],
    );
}

// ---------------------------------------------------------------------------------------------
// B4 — a reshape of a large value is O(depth), not O(value).
// ---------------------------------------------------------------------------------------------

#[test]
fn reshape_proof_is_o_depth_not_o_value() {
    for &kib in &[64usize, 256, 1024] {
        let len = kib * 1024;
        let pre = filled(len);
        let depth = (len / CHUNK_LEN).next_power_of_two().trailing_zeros() as usize;
        let bound = 3 * CHUNK_LEN + 40 * (depth + 1);

        for ops in [
            vec![Op::Resize(len + 16), Op::Write(len, filled(16))], // grow
            vec![Op::Resize(len - 16)],                             // shrink
        ] {
            let (_, verify_root, proof) = run(&pre, &ops);
            assert_eq!(verify_root, Some(hash_value(&oracle(&pre, &ops))));
            let size = serialise(&proof).expect("serialises").len();
            assert!(
                size <= bound,
                "{kib}KiB reshape proof {size} B exceeded O(depth) bound {bound} B (ops={ops:?})"
            );
            assert!(
                size < len / 4,
                "reshape proof {size} scales with value {len}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// B5 — security: tamper detection on the reshape path, and the R3 boundary rule.
// ---------------------------------------------------------------------------------------------

/// Flip a bit in the first reachable node of the given kind; returns whether one was flipped.
fn tamper_first_blind(tree: &mut ProofTree) -> bool {
    match tree {
        ProofTree::Blind(cv) => {
            cv[0] ^= 1;
            true
        }
        ProofTree::Node(l, r) => tamper_first_blind(l) || tamper_first_blind(r),
        ProofTree::Chunk(_) => false,
    }
}

/// Total bytes of present chunks in a proof tree - the part of a proof that scales with the value
/// rather than with its depth.
fn chunk_bytes(tree: &ProofTree) -> usize {
    match tree {
        ProofTree::Chunk(b) => b.len(),
        ProofTree::Node(l, r) => chunk_bytes(l) + chunk_bytes(r),
        ProofTree::Blind(_) => 0,
    }
}

/// A reshape whose low-water mark is chunk-aligned needs no pre bytes: the prefix is recovered from
/// length-independent aligned subtree CVs and the changed region from the verifier's own replay. It
/// must therefore carry no present chunk - only O(depth) blinds.
#[test]
fn aligned_reshape_carries_no_value_bytes() {
    for ops in [
        vec![Op::Resize(20 * CHUNK_LEN)],
        vec![Op::Resize(4 * CHUNK_LEN)],
        vec![Op::Resize(20 * CHUNK_LEN), Op::Resize(8 * CHUNK_LEN)],
    ] {
        let pre = filled(16 * CHUNK_LEN);
        let (_, _, proof) = run(&pre, &ops);
        assert_eq!(
            chunk_bytes(&proof.data),
            0,
            "aligned reshape carried value bytes for {ops:?}"
        );
    }
}

/// A write that covers a whole chunk needs no pre-image: the verifier replays the same write, so it
/// hashes that chunk itself. A full overwrite therefore carries no value bytes either, in place or
/// combined with a reshape.
#[test]
fn full_overwrite_carries_no_pre_image() {
    for (pre_len, ops) in [
        // In place, chunk-aligned and ragged.
        (4 * CHUNK_LEN, vec![Op::Write(0, filled(4 * CHUNK_LEN))]),
        (
            4 * CHUNK_LEN + 7,
            vec![Op::Write(0, filled(4 * CHUNK_LEN + 7))],
        ),
        // Grown, then written in full - the shape a `set` of a longer value takes.
        (
            4 * CHUNK_LEN,
            vec![
                Op::Resize(6 * CHUNK_LEN + 9),
                Op::Write(0, filled(6 * CHUNK_LEN + 9)),
            ],
        ),
        // Shrunk, then written in full.
        (
            8 * CHUNK_LEN,
            vec![
                Op::Resize(3 * CHUNK_LEN + 5),
                Op::Write(0, filled(3 * CHUNK_LEN + 5)),
            ],
        ),
    ] {
        let pre = filled(pre_len);
        let (_, _, proof) = run(&pre, &ops);
        assert_eq!(
            chunk_bytes(&proof.data),
            0,
            "full overwrite carried a pre-image for pre={pre_len} {ops:?}"
        );
    }
}

fn tamper_first_chunk(tree: &mut ProofTree) -> bool {
    match tree {
        ProofTree::Chunk(b) if !b.is_empty() => {
            b[0] ^= 1;
            true
        }
        ProofTree::Chunk(_) => false,
        ProofTree::Node(l, r) => tamper_first_chunk(l) || tamper_first_chunk(r),
        ProofTree::Blind(_) => false,
    }
}

/// Tampering a prefix blind CV in an otherwise-honest reshape proof must break verification: it no
/// longer reconstructs the committed pre root (so a real verifier rejects it up-front), and if one
/// were to force the reshape recompute with it, the post root diverges from the honest one.
#[test]
fn tampered_prefix_blind_is_rejected_on_reshape() {
    let pre = filled(16 * CHUNK_LEN);
    let ops = vec![
        Op::Resize(20 * CHUNK_LEN),
        Op::Write(16 * CHUNK_LEN, filled(16)),
    ];
    let pre_root = hash_value(&pre);
    let (honest_post, honest_verify, mut proof) = run(&pre, &ops);
    assert_eq!(honest_verify, Some(honest_post));

    // A tampered blind fails the pre-root reconstruction (the guard that validates every reused
    // CV before the reshape recompute trusts it).
    assert!(tamper_first_blind(&mut proof.data));
    match verify_root(&proof) {
        Err(_) => {}
        Ok(root) => assert_ne!(root, pre_root, "tampered blind still matched the pre root"),
    }
    // And a real verifier rejects at reconstruction time.
    assert!(
        Blake3Bytes::<Verify>::from_proof_checked(proof, pre_root).is_err(),
        "verifier accepted a proof with a tampered prefix blind"
    );
}

#[test]
fn tampered_boundary_chunk_diverges_on_reshape() {
    // A *partial* boundary chunk, i.e. an unaligned low-water mark - an aligned one leaves the
    // prefix wholly reusable, so the proof carries no chunk to tamper with (see
    // `aligned_reshape_carries_no_value_bytes`).
    let pre = filled(16 * CHUNK_LEN + 500);
    let ops = vec![Op::Resize(20 * CHUNK_LEN)];
    let pre_root = hash_value(&pre);
    let (_honest_post, _, mut proof) = run(&pre, &ops);

    // Flip the present boundary chunk: it changes the reconstructed pre root, so the proof is
    // rejected before the reshape recompute ever runs.
    assert!(tamper_first_chunk(&mut proof.data));
    assert!(
        Blake3Bytes::<Verify>::from_proof_checked(proof, pre_root).is_err(),
        "verifier accepted a proof with a tampered boundary chunk"
    );
}

/// R3 defence in depth: if the boundary chunk is *omitted* (blinded instead of present) from a
/// reshape proof, the verifier cannot hash the changed region live, so it cannot reproduce the
/// honest post root — it must diverge or fail, never accept.
#[test]
fn reshape_without_boundary_chunk_cannot_forge_post_root() {
    let pre = filled(16 * CHUNK_LEN);
    let ops = vec![
        Op::Resize(17 * CHUNK_LEN),
        Op::Write(16 * CHUNK_LEN, filled(16)),
    ];
    let pre_root = hash_value(&pre);
    let honest_post = hash_value(&oracle(&pre, &ops));

    // A proof of the pre value that blinds EVERYTHING (only the length was "read"): it verifies to
    // the pre root, but carries no present boundary chunk. Reconstruct and replay the grow.
    let all_blind = Blake3Proof {
        total_len: pre.len(),
        data: ProofTree::Blind(*blake3::hash(&pre).as_bytes()),
    };
    assert_eq!(verify_root(&all_blind).unwrap(), pre_root);

    let mut verify = Blake3Bytes::<Verify>::from_proof_checked(all_blind, pre_root).unwrap();
    apply(&mut verify, &ops);
    let vroot = PartialHash::from_foldable(None, &verify).to_hash();
    assert_ne!(
        vroot,
        Some(honest_post),
        "a reshape proof with no boundary chunk forged the post root"
    );
}

// ---------------------------------------------------------------------------------------------
// Property tests — the heavy campaign. Run hard with PROPTEST_CASES=100000.
// ---------------------------------------------------------------------------------------------

fn arb_len() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(0usize),
        1usize..2048,
        (1usize..=8).prop_map(|c| c * CHUNK_LEN),
        (1usize..=8).prop_map(|c| c * CHUNK_LEN + 7),
        0usize..=12 * CHUNK_LEN,
    ]
}

fn arb_op(max_len: usize) -> impl Strategy<Value = Op> {
    prop_oneof![
        (0usize..=max_len, 1usize..=2048).prop_map(|(s, l)| Op::Read(s, l)),
        (0usize..=max_len, 1usize..=2048)
            .prop_map(|(s, l)| Op::Write(s, (0..l).map(|i| (i ^ s) as u8).collect())),
        (0usize..=12 * CHUNK_LEN).prop_map(Op::Resize),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 3000, ..ProptestConfig::default() })]

    /// The core soundness+completeness property: for ANY pre value and ANY op sequence (arbitrary
    /// interleavings of reads, writes, and resizes — including shrink-below-then-regrow), the
    /// verifier reconstructs exactly the true post hash. Covers B1/B2/B3 and every multi-resize
    /// path (clean, shrink-below-then-regrow, net-length-unchanged) in one sweep.
    #[test]
    fn reshape_roundtrip_matches_oracle(
        pre_len in arb_len(),
        ops in prop::collection::vec(arb_op(12 * CHUNK_LEN), 0..=6),
    ) {
        let pre = filled(pre_len);
        let (post_hash, verify_root, _) = run(&pre, &ops);
        prop_assert_eq!(post_hash, hash_value(&oracle(&pre, &ops)));
        prop_assert_eq!(verify_root, Some(post_hash));
    }

    /// Single-bit tamper of an accepted reshape proof: reconstruction against the pre root must
    /// reject it (the guard that protects the reused CVs), i.e. no tampered proof survives to feed
    /// the reshape recompute a stale/forged CV.
    #[test]
    fn any_tamper_rejected(
        pre_len in 1usize..=8 * CHUNK_LEN,
        delta in prop_oneof![Just(16i64), Just(-16i64), Just(4096), Just(-4096)],
    ) {
        let pre = filled(pre_len);
        let new_len = (pre_len as i64 + delta).max(0) as usize;
        let ops = vec![Op::Resize(new_len)];
        let pre_root = hash_value(&pre);
        let (_, _, proof) = run(&pre, &ops);

        // Blind tamper.
        let mut t1 = proof.clone();
        if tamper_first_blind(&mut t1.data) {
            prop_assert!(Blake3Bytes::<Verify>::from_proof_checked(t1, pre_root).is_err());
        }
        // Chunk tamper.
        let mut t2 = proof.clone();
        if tamper_first_chunk(&mut t2.data) {
            prop_assert!(Blake3Bytes::<Verify>::from_proof_checked(t2, pre_root).is_err());
        }
        // Length tamper.
        let mut t3 = proof;
        t3.total_len = t3.total_len.wrapping_add(CHUNK_LEN);
        match verify_root(&t3) {
            Err(_) => {}
            Ok(root) => prop_assert_ne!(root, pre_root),
        }
    }
}

// A write covering whole chunks records no accessed range, so nothing marks those chunks as
// changed when the proof shape is chosen. They must still not be swallowed by a coarser blind:
// the verifier re-derives a blind only where it holds the whole span, and it holds exactly what
// the replay wrote, so a blind straddling written and unwritten bytes leaves it with the pre
// value and the write invisible in the root.
//
// The payload matters. `filled` repeats with period 256, so writing `filled(CHUNK_LEN)` at a
// 1024-aligned offset writes the bytes already there and passes however the proof is built.
mod written_chunks_reach_the_verifier {
    use super::*;

    fn payload(len: usize) -> Vec<u8> {
        vec![0xABu8; len]
    }

    #[test]
    fn one_aligned_chunk() {
        let pre = filled(4 * CHUNK_LEN);
        assert_ne!(
            &pre[CHUNK_LEN..2 * CHUNK_LEN],
            &payload(CHUNK_LEN)[..],
            "the payload has to differ from what it overwrites, or this proves nothing"
        );
        assert_roundtrip(&pre, &[Op::Write(CHUNK_LEN, payload(CHUNK_LEN))]);
    }

    #[test]
    fn an_aligned_pair() {
        assert_roundtrip(
            &filled(8 * CHUNK_LEN),
            &[Op::Write(2 * CHUNK_LEN, payload(2 * CHUNK_LEN))],
        );
    }

    // The whole value written needs no shape kept open: the verifier holds all of it and
    // re-derives the root itself.
    #[test]
    fn the_whole_value() {
        assert_roundtrip(&filled(4 * CHUNK_LEN), &[Op::Write(0, payload(4 * CHUNK_LEN))]);
    }

    #[test]
    fn then_a_grow() {
        assert_roundtrip(
            &filled(8 * CHUNK_LEN),
            &[
                Op::Write(CHUNK_LEN, payload(CHUNK_LEN)),
                Op::Resize(9 * CHUNK_LEN),
            ],
        );
    }

    #[test]
    fn then_a_shrink() {
        assert_roundtrip(
            &filled(8 * CHUNK_LEN),
            &[
                Op::Write(CHUNK_LEN, payload(CHUNK_LEN)),
                Op::Resize(3 * CHUNK_LEN),
            ],
        );
    }

    // A partly covered chunk keeps its pre-image instead, which already worked.
    #[test]
    fn an_unaligned_write() {
        assert_roundtrip(&filled(4 * CHUNK_LEN), &[Op::Write(CHUNK_LEN + 7, payload(100))]);
    }
}
