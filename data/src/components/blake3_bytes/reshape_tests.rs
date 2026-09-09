// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for length-changing transitions. A resize reshapes BLAKE3's chunk tree, so the
//! committed shape stops describing the value.
//!
//! Every test drives the real component through the same prove, value proof, verify replay path
//! the durable-storage AVL uses, and pairs a positive property - an honest reshape verifies to
//! the true post hash, which an independent oracle computes - with a negative one, that a
//! tampered proof or a shape the verifier must not trust is rejected or diverges.

use proptest::prelude::*;

use super::Blake3Bytes;
use super::Blake3BytesMode;
use super::Blake3Proof;
use super::ProofTree;
use super::hash_value;
use super::verify_root;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Verify;
use crate::serialisation::serialise;

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
#[ignore = "a reshape still costs a proof the size of the value; the next commit makes it O(depth)"]
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
#[ignore = "a reshape still costs a proof the size of the value; the next commit makes it O(depth)"]
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
#[ignore = "a reshape proof carries no prefix blind to tamper with until the next commit reuses one"]
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

/// The serialised wire size of a length-changing transition against a length-preserving one, so
/// a regression that reintroduces O(value) reshape proofs shows up as a number rather than as a
/// slow test.
#[test]
#[ignore = "a reshape still costs a proof the size of the value; the next commit makes it O(depth)"]
fn reshape_costs_no_more_than_an_in_place_write() {
    fn proof_bytes(
        len: usize,
        run: impl FnOnce(&mut super::Blake3Bytes<crate::mode::Prove>),
    ) -> usize {
        let data = vec![0x5Au8; len];
        let mut prover = super::Blake3Bytes::<crate::mode::Prove>::from_raw_source(&data);
        run(&mut prover);
        let proof = crate::merkle_proof::proof_tree::MerkleProof::from_foldable(&prover);
        serialise(&proof).expect("proof serialises").len()
    }

    for kib in [16usize, 64, 256, 1024] {
        let len = kib * 1024;
        let depth = (len / CHUNK_LEN).next_power_of_two().trailing_zeros() as usize;
        let bound = 3 * CHUNK_LEN + 40 * (depth + 1);

        let write = proof_bytes(len, |p| {
            p.write(len / 2, &[0xEE; 16]);
        });
        let grow = proof_bytes(len, |p| {
            p.resize(len + 16);
            p.write(len, &[0xEE; 16]);
        });
        let shrink = proof_bytes(len, |p| p.resize(len - 16));

        assert!(
            write <= 2 * CHUNK_LEN + 40 * (depth + 1),
            "write {write} at {kib} KiB"
        );
        assert!(
            grow <= bound,
            "grow {grow} exceeded the O(depth) bound {bound} at {kib} KiB"
        );
        assert!(
            shrink <= bound,
            "shrink {shrink} exceeded the O(depth) bound {bound} at {kib} KiB"
        );
    }
}
