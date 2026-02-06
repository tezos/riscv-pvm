// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

// This ensures that Clippy does't apply rules which are allowed in tests.
#![cfg(test)]

use std::fs;
use std::ops::Bound;
use std::path::PathBuf;
use std::time::Instant;

use octez_riscv::machine_state::memory::M64M;
use octez_riscv::machine_state::page_cache::PageCache;
use octez_riscv::machine_state::page_cache::PageCacheInterpreted;
use octez_riscv::pvm::hooks::NoHooks;
use octez_riscv::pvm::outbox::outbox_proof::OutboxProof;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv_data::mode::Normal;
use octez_riscv_test_utils::*;
use tezos_smart_rollup_utils::inbox::InboxBuilder;

/// The maximum size in bytes expected for an outbox proof (message size is 4096 B)
const MAX_EXPECTED_OUTBOX_PROOF_SIZE: usize = 4745;

fn test_outbox_proofs<PC: PageCache<M64M, Normal>>(inputs: &TestConfig) {
    // TODO: take program setup out
    let program = fs::read(inputs.kernel_path)
        .expect("Failed to read kernel from disk. Try running `make build`.");

    let inbox = {
        let mut inbox = InboxBuilder::new();
        inbox.load_from_file(inputs.inbox_path).unwrap();
        inbox.build()
    };

    const ROLLUP_ADDRESS: [u8; 20] = [
        244, 228, 124, 179, 196, 58, 104, 176, 212, 142, 48, 148, 9, 44, 164, 45, 113, 58, 221, 181,
    ];
    const ORIGINATION_LEVEL: u32 = 1;

    let mut stepper = PvmStepper::<_, M64M, Normal, PC>::new(
        &program,
        inbox,
        NoHooks,
        ROLLUP_ADDRESS,
        ORIGINATION_LEVEL,
        Some(PathBuf::from("../../../assets/preimages").into_boxed_path()),
    )
    .unwrap();
    // --------

    let outbox_message = vec![0x1; 4096];

    let _result = stepper.step_max(Bound::Unbounded);

    eprintln!("> Producing outbox proof ...");
    let start = Instant::now();
    let proof = stepper.produce_outbox_proof(outbox_message, 15, 0).unwrap();
    let time = start.elapsed();

    let serialisation: Vec<u8> = OutboxProof::serialise(&proof);
    let proof_size = serialisation.len();

    eprintln!("> Proof of size {proof_size} B produced in {time:?}");

    if proof_size > MAX_EXPECTED_OUTBOX_PROOF_SIZE {
        panic!(
            "Outbox proof expected to be at most {MAX_EXPECTED_OUTBOX_PROOF_SIZE} B. Please investigate: {proof:?}"
        )
    }
}

#[test]
fn test_outbox_proofs_dummy_kernel() {
    test_outbox_proofs::<PageCacheInterpreted<M64M>>(&DUMMY)
}
