// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT
//! Regression: the cross-span graft that broke the predecessor "no length node" scheme is
//! REJECTED by the production length-committed `verify_root`
//! (`data/docs/blake3-bytes-length-committed.md` §5 S2, §10).
//!
//! History: this test previously *demonstrated* the forgery against the shipped maximal rule
//! (it was `#[ignore]`d, and confirmed FAILING = forged). With the length commitment in place
//! it now asserts the forgery is defeated.

use blake3::hazmat::HasherExt;
use octez_riscv_data::components::blake3_bytes::Blake3Proof;
use octez_riscv_data::components::blake3_bytes::ProofTree;
use octez_riscv_data::components::blake3_bytes::hash_value;
use octez_riscv_data::components::blake3_bytes::verify_root;

const C: usize = 1024;

fn chunk_cv(bytes: &[u8], offset: u64) -> [u8; 32] {
    blake3::Hasher::new()
        .set_input_offset(offset)
        .update(bytes)
        .finalize_non_root()
}

#[test]
fn cross_span_graft_rejected_by_production() {
    // Honest value: 3 full chunks. Honest data root = merge_root(cv[0..2], cv2); the committed
    // value hash binds this with H_len(3072).
    let data = vec![0xABu8; 3 * C];
    let committed = hash_value(&data);

    // Forge: claim total_len = 4096 (4 chunks). canonical(4096) splits 2048|2048.
    // Left [0,2048): present chunks 0,1 -> reconstructs to cv[0..2] honestly.
    // Right [2048,4096): a Blind is accepted there (blinding is unrestricted now).
    // Plant cv2 (the span-1024 chunk-2 CV) -> the DATA root reconstructs to the honest one...
    let cv2 = chunk_cv(&data[2 * C..3 * C], (2 * C) as u64);
    let malicious = Blake3Proof {
        total_len: 4 * C,
        data: ProofTree::Node(
            Box::new(ProofTree::Node(
                Box::new(ProofTree::Chunk(data[0..C].to_vec())),
                Box::new(ProofTree::Chunk(data[C..2 * C].to_vec())),
            )),
            Box::new(ProofTree::Blind(cv2)),
        ),
    };

    // ...but H_len(4096) != H_len(3072), so the committed value hash differs -> no forgery.
    match verify_root(&malicious) {
        Ok(root) => assert_ne!(
            root,
            committed,
            "PRODUCTION FORGERY: 3-chunk value verifies as total_len={} against its honest hash",
            4 * C
        ),
        Err(_) => { /* rejected outright — also fine */ }
    }
}
