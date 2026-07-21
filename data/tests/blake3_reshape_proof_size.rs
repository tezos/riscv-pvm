// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT
//! Measures the within-value AVL-wire proof size for length-*preserving* vs length-*changing*
//! transitions on a `Blake3Bytes` value. With the reshape optimisation
//! (`data/docs/blake3-bytes-reshape-optimisation.md`), a length change is O(depth) just like an
//! in-place access: the unchanged prefix collapses to its set-bit-of-`min(Lp,Lq)` aligned subtree
//! CVs — which are length-independent, so the verifier reuses them at the new shape — plus the one
//! reshape boundary chunk, instead of the whole value. Run with:
//!   `cargo test -p octez-riscv-data --test blake3_reshape_proof_size -- --nocapture`
//!
//! This is the size-regression guard for that optimisation (reference model
//! `data/tests/blake3_reshape_model.rs`). Before the optimisation a length change forced the whole
//! pre value present (proof ≈ value size); the `grow`/`shrink` assertions below now hold it to the
//! `O(depth)` bound, matching in-place.

use octez_riscv_data::components::blake3_bytes::Blake3Bytes;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::serialisation::serialise;

const CHUNK_LEN: usize = 1024;

/// Serialised AVL-wire proof size (bytes) for a prove-mode transition built by `run` against a
/// value of `len` bytes.
fn proof_bytes(len: usize, run: impl FnOnce(&mut Blake3Bytes<Prove>)) -> usize {
    let data = vec![0x5Au8; len];
    let mut prover = Blake3Bytes::<Prove>::from_raw_source(&data);
    run(&mut prover);
    let proof = MerkleProof::from_foldable(&prover);
    serialise(&proof).expect("proof serialises").len()
}

#[test]
fn reshape_vs_inplace_proof_size() {
    println!();
    println!(
        "{:>7}  {:>3}  {:>14}  {:>14}  {:>14}  {:>14}",
        "value", "d", "read 16B", "write 16B", "grow +16B", "shrink -16B"
    );
    for kib in [16usize, 64, 256, 1024] {
        let len = kib * 1024;
        let chunks = len / CHUNK_LEN;
        let depth = chunks.next_power_of_two().trailing_zeros() as usize;
        let mid = len / 2;

        // Length-preserving: a read / an in-place write in the middle -> O(depth) partial proof.
        let read = proof_bytes(len, |p| {
            let mut buf = [0u8; 16];
            p.read(mid, &mut buf);
        });
        let write = proof_bytes(len, |p| {
            p.write(mid, &[0xEE; 16]);
        });
        // Length-changing: grow (resize up + append) / shrink (resize down).
        let grow = proof_bytes(len, |p| {
            p.resize(len + 16);
            p.write(len, &[0xEE; 16]);
        });
        let shrink = proof_bytes(len, |p| {
            p.resize(len - 16);
        });

        let kib_str = format!("{kib}KiB");
        println!(
            "{:>7}  {:>3}  {:>11.2} KiB  {:>11.2} KiB  {:>11.2} KiB  {:>11.2} KiB",
            kib_str,
            depth,
            read as f64 / 1024.0,
            write as f64 / 1024.0,
            grow as f64 / 1024.0,
            shrink as f64 / 1024.0,
        );

        // In-place stays O(depth): a 16B access spans <= 2 chunks + depth auth-path CVs.
        assert!(write <= 2 * CHUNK_LEN + 40 * (depth + 1));
        // After the reshape optimisation, a length change is also O(depth): the unchanged prefix
        // collapses to set-bit-of-min aligned subtree CVs (reused at the new shape) plus the one
        // boundary chunk, instead of the whole value. Bound: a small constant number of present
        // chunks (the 16B access chunk + the boundary chunk, each <= CHUNK_LEN) + depth CVs. In
        // particular it must NOT scale with `len` — assert it stays far below the value size.
        let o_depth = 3 * CHUNK_LEN + 40 * (depth + 1);
        assert!(
            grow <= o_depth,
            "grow {grow} exceeded O(depth) bound {o_depth}"
        );
        assert!(
            shrink <= o_depth,
            "shrink {shrink} exceeded O(depth) bound {o_depth}"
        );
    }
}

/// Worst-case reshape proof sizes on large values, up to 256 MiB (depth 18). The worst case for a
/// reshape is a length change whose boundary chunk sits on a *different* authentication path from a
/// separate accessed region, so the two O(depth) paths share the fewest blinds. We report:
///   * `grow`   — append 16 B at the very end (boundary near the last chunk);
///   * `g+read` — the same append PLUS a 2 KiB read at offset 0 (two maximally-disjoint paths);
///   * `shrink` — truncate by 16 B.
///
/// All are O(depth): a couple of present chunks + ~2·depth blinded 32-byte CVs. This is the number
/// to quote as the worst case — it is position-independent and does NOT scale with the value size.
#[test]
#[ignore = "reporting only; run with --ignored --nocapture"]
fn worst_case_reshape_proof_sizes() {
    println!();
    println!(
        "{:>7}  {:>3}  {:>12}  {:>12}  {:>12}",
        "value", "d", "grow +16B", "g+read 2KiB", "shrink -16B"
    );
    for mib in [1usize, 16, 64, 256] {
        let len = mib * 1024 * 1024;
        let depth = (len / CHUNK_LEN).next_power_of_two().trailing_zeros() as usize;

        let grow = proof_bytes(len, |p| {
            p.resize(len + 16);
            p.write(len, &[0xEE; 16]);
        });
        let grow_read = proof_bytes(len, |p| {
            let mut buf = [0u8; 2048];
            p.read(0, &mut buf); // disjoint auth path from the boundary
            p.resize(len + 16);
            p.write(len, &[0xEE; 16]);
        });
        let shrink = proof_bytes(len, |p| {
            p.resize(len - 16);
        });

        println!(
            "{:>6}M  {:>3}  {:>9.2} KiB  {:>9.2} KiB  {:>9.2} KiB",
            mib,
            depth,
            grow as f64 / 1024.0,
            grow_read as f64 / 1024.0,
            shrink as f64 / 1024.0,
        );
    }
}
