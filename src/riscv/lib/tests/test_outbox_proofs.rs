// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

// This ensures that Clippy does't apply rules which are allowed in tests.
#![cfg(test)]

use std::io::Write;
use std::ops::Bound;
use std::path::PathBuf;
use std::time::Instant;

use octez_riscv::machine_state::memory::M64M;
use octez_riscv::pvm::outbox::OutboxProof;
use octez_riscv::pvm::outbox::OutputInfo;
use octez_riscv::stepper::Stepper;
use octez_riscv_test_utils::*;

/// The maximum size in bytes expected for an outbox proof (message size is 4096 B)
const MAX_EXPECTED_OUTBOX_PROOF_SIZE: usize = 4770;

const ROLLUP_ADDRESS: [u8; 20] = [
    244, 228, 124, 179, 196, 58, 104, 176, 212, 142, 48, 148, 9, 44, 164, 45, 113, 58, 221, 181,
];

fn test_outbox_proofs(inputs: &TestConfig) {
    let make_stepper = make_stepper_factory::<M64M>(
        inputs,
        Some(ROLLUP_ADDRESS),
        Some(PathBuf::from("../../../assets/preimages").into_boxed_path()),
    );
    let mut stepper = make_stepper();

    let _result = stepper.step_max(Bound::Unbounded);

    let output_info = OutputInfo {
        level: stepper.level().unwrap(),
        index: 0,
    };

    eprintln!(
        "> Producing outbox proof for message at level {}, index {}...",
        output_info.level, output_info.index
    );
    let start = Instant::now();
    let proof = stepper.produce_outbox_proof(output_info).unwrap();
    let time = start.elapsed();

    let proof_serialisation: Vec<u8> = OutboxProof::serialise(&proof);
    let proof_size = proof_serialisation.len();

    eprintln!("> Outbox proof of size {proof_size} B produced in {time:?}");

    if proof_size > MAX_EXPECTED_OUTBOX_PROOF_SIZE {
        panic!(
            "Outbox proof expected to be at most {MAX_EXPECTED_OUTBOX_PROOF_SIZE} B. Please investigate: {proof:?}"
        )
    };

    assert_eq!(stepper.hash(), proof.state_hash());

    let mut mint = goldenfile::Mint::new(inputs.golden_dir);
    let mut proof_capture = mint.new_goldenfile("outbox_proof").unwrap();
    let proof_bytes = hex::encode(proof_serialisation);
    writeln!(proof_capture, "{proof_bytes}").unwrap();
}

#[test]
fn test_outbox_proofs_dummy_kernel() {
    test_outbox_proofs(&DUMMY)
}
