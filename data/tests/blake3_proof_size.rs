// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT
//! Production within-value proof-size measurement + size guard for the length-committed
//! BLAKE3 scheme (`data/docs/blake3-bytes-length-committed.md` §6, §10).
//!
//! Measures the *serialised* wire size of `blake3_bytes::prove` proofs for a 2 KiB access at
//! the start / middle / end of 1/16/64 MiB values, and asserts they are `O(depth)` and
//! position-independent (the payoff of unrestricted blinding). Run with output:
//!   `cargo test -p octez-riscv-data --test blake3_proof_size -- --nocapture`

use std::ops::Range;

use octez_riscv_data::components::blake3_bytes::Blake3Proof;
use octez_riscv_data::components::blake3_bytes::ProofTree;
use octez_riscv_data::components::blake3_bytes::hash_value;
use octez_riscv_data::components::blake3_bytes::prove;
use octez_riscv_data::components::blake3_bytes::verify_root;
use octez_riscv_data::serialisation::serialise;

const CHUNK_LEN: usize = 1024;

/// Present-chunk count and blinded-CV count in a proof tree.
fn counts(t: &ProofTree) -> (usize, usize) {
    match t {
        ProofTree::Chunk(_) => (1, 0),
        ProofTree::Blind(_) => (0, 1),
        ProofTree::Node(l, r) => {
            let (a, b) = counts(l);
            let (c, d) = counts(r);
            (a + c, b + d)
        }
    }
}

fn measure(n: usize, access: Range<usize>) -> (usize, usize, usize) {
    let data = vec![0x5Au8; n];
    let proof: Blake3Proof = prove(&data, &[access]);
    // Sanity: an honest proof still verifies to the committed value hash.
    assert_eq!(
        verify_root(&proof).expect("honest proof verifies"),
        hash_value(&data)
    );
    let wire = serialise(&proof).expect("proof serialises").len();
    let (present, blinds) = counts(&proof.data);
    (present, blinds, wire)
}

#[test]
fn proof_size_is_o_depth_and_position_independent() {
    println!();
    println!(
        "{:>5}  {:>3}  {:<11}  {:>7}  {:>6}  {:>10}",
        "MiB", "d", "access", "present", "blinds", "wire"
    );
    for mib in [1usize, 16, 64] {
        let n = mib << 20;
        let chunks = n / CHUNK_LEN;
        let depth = chunks.next_power_of_two().trailing_zeros() as usize;

        let cases: [(&str, Range<usize>); 3] = [
            ("start 2KiB", 0..2048),
            ("middle 2KiB", n / 2..n / 2 + 2048),
            ("end 2KiB", n - 2048..n),
        ];
        let mut wires = Vec::new();
        for (name, access) in cases {
            let (present, blinds, wire) = measure(n, access);
            println!(
                "{mib:>5}  {depth:>3}  {name:<11}  {present:>7}  {blinds:>6}  {:>7.2} KiB",
                wire as f64 / 1024.0
            );
            wires.push(wire);
            // O(depth): a 2 KiB access spans <= 3 chunks; blinds are authentication-path
            // siblings, one per level, plus a small constant.
            assert!(present <= 3 + 1, "present {present} at {mib} MiB");
            assert!(blinds <= 2 * depth + 4, "blinds {blinds} (depth {depth})");
        }
        // Position-independent: start/middle/end wire sizes agree within a small margin.
        let (min, max) = (*wires.iter().min().unwrap(), *wires.iter().max().unwrap());
        assert!(
            max - min <= 2 * CHUNK_LEN,
            "position-dependent wire size at {mib} MiB: min {min} max {max}"
        );
    }
}
