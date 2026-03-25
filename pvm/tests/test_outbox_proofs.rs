// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

// This ensures that Clippy does't apply rules which are allowed in tests.
#![cfg(test)]

use std::io::Write;
use std::ops::Bound;
use std::time::Instant;

use octez_riscv::machine_state::memory::M64M;
use octez_riscv::pvm::hooks::PvmHooks;
use octez_riscv::pvm::outbox::OutboxMessage;
use octez_riscv::pvm::outbox::OutboxProof;
use octez_riscv::pvm::outbox::OutboxProofError;
use octez_riscv::pvm::outbox::Output;
use octez_riscv::pvm::outbox::OutputInfo;
use octez_riscv::pvm::outbox::TEST_OUTBOX_SIZE;
use octez_riscv::state_backend::verify_backend::ProofVerificationFailure;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv::stepper::pvm::verify_outbox_proof;
use octez_riscv_data::mode::utils::NotFound;
use octez_riscv_test_utils::*;

/// The maximum size in bytes expected for an outbox proof (message size is 4096 B)
const MAX_EXPECTED_OUTBOX_PROOF_SIZE: usize = 4880;

const ROLLUP_ADDRESS: [u8; 20] = [
    244, 228, 124, 179, 196, 58, 104, 176, 212, 142, 48, 148, 9, 44, 164, 45, 113, 58, 221, 181,
];

fn test_outbox_proofs(inputs: &TestConfig) {
    let make_stepper = make_stepper_factory::<M64M>(inputs, Some(ROLLUP_ADDRESS));
    let mut stepper = make_stepper();

    let _result = stepper.step_max(Bound::Unbounded);

    // The output needs to match an outbox message written by the tested kernel during its run
    let output = Output {
        message: OutboxMessage::try_from(vec![0x1; 4096].into_boxed_slice()).unwrap(),
        info: OutputInfo {
            level: stepper.level().unwrap(),
            index: 0,
        },
    };

    eprintln!(
        "> Producing outbox proof for message at level {}, index {}...",
        output.info.level, output.info.index
    );
    let start = Instant::now();
    let proof = stepper.produce_outbox_proof(output.info).unwrap();
    let time = start.elapsed();

    let proof_serialisation: Vec<u8> = OutboxProof::serialise(&proof);
    let proof_size = proof_serialisation.len();

    eprintln!("> Outbox proof of size {proof_size} B produced in {time:?}");

    if proof_size > MAX_EXPECTED_OUTBOX_PROOF_SIZE {
        panic!(
            "Outbox proof expected to be at most {MAX_EXPECTED_OUTBOX_PROOF_SIZE} B. Please investigate: {proof:?}"
        )
    };

    let mut mint = goldenfile::Mint::new(inputs.golden_dir);
    let mut proof_capture = mint.new_goldenfile("outbox_proof").unwrap();
    let proof_bytes = hex::encode(&proof_serialisation);
    writeln!(proof_capture, "{proof_bytes}").unwrap();

    // For both `proof` and the proof deserialised from serialising `proof`,
    // check that:
    // - the outbox proof is for the expected state
    // - the outbox proof verifies, yielding the message we expected the kernel wrote
    //   at that outbox position
    eprintln!("> Verifying outbox proof ...");
    assert_eq!(stepper.hash(), proof.state_hash());
    let output_from_proof = verify_outbox_proof(&proof).unwrap();
    assert_eq!(output, output_from_proof);

    eprintln!("> Verifying outbox proof (serialised)...");
    let deserialised_proof = OutboxProof::deserialise(&proof_serialisation).unwrap();
    assert_eq!(stepper.hash(), deserialised_proof.state_hash());
    let output_from_deserialised_proof = verify_outbox_proof(&deserialised_proof).unwrap();
    assert_eq!(output, output_from_deserialised_proof);

    test_invalid_outbox_level_proofs(&mut stepper);
    test_invalid_merkle_proofs(&mut stepper);
}

fn test_invalid_outbox_level_proofs<H: PvmHooks>(stepper: &mut PvmStepper<H, M64M>) {
    let stepper_level = stepper.level().unwrap();
    // Outbox proof production fails for stale level
    assert!(matches!(
        stepper.produce_outbox_proof(OutputInfo {
            level: stepper_level
                .checked_sub(TEST_OUTBOX_SIZE as u32)
                .expect("Expected current level to be higher than the size of the outbox"),
            index: 0,
        }),
        Err(OutboxProofError::LevelNotFound { level: 2 })
    ));

    // Outbox proof production fails for future level
    assert!(matches!(
        stepper.produce_outbox_proof(OutputInfo {
            level: stepper_level + 1,
            index: 0,
        }),
        Err(OutboxProofError::LevelNotFound { level  }) if level == stepper_level + 1
    ));
}

fn test_invalid_merkle_proofs<H: PvmHooks>(stepper: &mut PvmStepper<H, M64M>) {
    let level = stepper.level().unwrap();

    // Malicious outbox proof should fail
    let mut outbox_proof1 = stepper
        .produce_outbox_proof(OutputInfo { level, index: 0 })
        .expect("Outbox proof should not be stale");

    let mut outbox_proof2 = stepper
        .produce_outbox_proof(OutputInfo {
            level: level - 1,
            index: 0,
        })
        .expect("Outbox proof should not be stale");

    assert!(verify_outbox_proof(&outbox_proof1).is_ok());
    assert!(verify_outbox_proof(&outbox_proof2).is_ok());

    std::mem::swap(&mut outbox_proof1.proof, &mut outbox_proof2.proof);

    assert!(matches!(
        verify_outbox_proof(&outbox_proof1),
        Err(ProofVerificationFailure::AbsentDataAccess(NotFound))
    ));

    assert!(matches!(
        verify_outbox_proof(&outbox_proof2),
        Err(ProofVerificationFailure::AbsentDataAccess(NotFound))
    ));
}

fn test_tampered_oversize_message(inputs: &TestConfig) {
    let make_stepper = make_stepper_factory::<M64M>(inputs, Some(ROLLUP_ADDRESS));
    let mut stepper = make_stepper();
    let _result = stepper.step_max(Bound::Unbounded);
    let level = stepper.level().unwrap();
    let outbox_proof = stepper
        .produce_outbox_proof(OutputInfo { level, index: 0 })
        .expect("Outbox proof should be valid");

    let tampered_outbox_proof = replace_outbox_message_of_proof(&outbox_proof, &[0x1; 8192]);
    let tampered_outbox_proof = OutboxProof::deserialise(tampered_outbox_proof.as_slice());
    tampered_outbox_proof.expect_err("Should fail deserialisation");
}

fn test_tampered_zero_sized_message(inputs: &TestConfig) {
    let make_stepper = make_stepper_factory::<M64M>(inputs, Some(ROLLUP_ADDRESS));
    let mut stepper = make_stepper();
    let _result = stepper.step_max(Bound::Unbounded);
    let level = stepper.level().unwrap();
    let outbox_proof = stepper
        .produce_outbox_proof(OutputInfo { level, index: 0 })
        .expect("Outbox proof should be valid");

    let tampered_outbox_proof_bytes = replace_outbox_message_of_proof(&outbox_proof, &[]);
    let tampered_outbox_proof = OutboxProof::deserialise(tampered_outbox_proof_bytes.as_slice())
        .expect("Zero sized messages are allowed");
    assert!(verify_outbox_proof(&tampered_outbox_proof).is_ok());
}

/// Returns a serialized [OutboxProof] with the outbox message
/// portion set to [message]. The original outbox proof message
/// is expected to be 4096 B
fn replace_outbox_message_of_proof(proof: &OutboxProof, message: &[u8]) -> Vec<u8> {
    let proof_bytes = OutboxProof::serialise(&proof);
    let message_size = 4096;
    let message_pos = proof_bytes
        .windows(message_size)
        .position(|w| w.iter().all(|&b| b == 0x01))
        .expect("Message content should be present in serialized proof");
    let len_pos = message_pos - 8;

    // Sanity check that the length prefix is correct
    assert_eq!(
        &proof_bytes[len_pos..len_pos + 8],
        &(message_size as u64).to_le_bytes(),
    );

    // Craft serialized proof, replacing the outbox message with [message]
    let mut tampered = vec![];
    let new_size = message.len();
    tampered.extend_from_slice(&proof_bytes[..len_pos]);
    tampered.extend_from_slice(&new_size.to_le_bytes());
    tampered.extend_from_slice(message);
    tampered.extend_from_slice(&proof_bytes[(len_pos + 8 + message_size)..]);

    tampered
}

#[test]
fn test_outbox_proofs_dummy_kernel() {
    test_outbox_proofs(&DUMMY)
}

#[test]
#[ignore = "TODO(RV-950)"]
fn test_tampered_oversize_message_dummy_kernel() {
    test_tampered_oversize_message(&DUMMY);
}

#[test]
fn test_zero_sized_message_dummy_kernel() {
    test_tampered_zero_sized_message(&DUMMY)
}
