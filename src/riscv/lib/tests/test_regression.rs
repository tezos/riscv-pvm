// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

// This ensures that Clippy does't apply rules which are allowed in tests.
#![cfg(test)]

use std::fs;
use std::io::Write;
use std::ops::Bound;
use std::path::PathBuf;

use octez_riscv::machine_state::memory::M64M;
use octez_riscv::machine_state::page_cache::CodePageEntry;
use octez_riscv::machine_state::page_cache::Interpreted;
use octez_riscv::machine_state::page_cache::InterpretedCompiler;
use octez_riscv::machine_state::page_cache::Jitted;
use octez_riscv::machine_state::page_cache::OutlineCompiler;
use octez_riscv::pvm::hooks::PvmHooks;
use octez_riscv::state_backend::owned_backend::Owned;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv_test_utils::*;
use tezos_smart_rollup_utils::inbox::InboxBuilder;

/// [`PvmHooks`] that direct the debug log of the PVM into a golden file
struct MintCaptureHooks {
    log_file: fs::File,
}

impl MintCaptureHooks {
    /// Create a new instance of these PVM hooks that will direct the debug log into a golden file
    /// called `log` managed by the given [`goldenfile::Mint`].
    fn new(mint: &mut goldenfile::Mint) -> Self {
        Self {
            log_file: mint.new_goldenfile("log").unwrap(),
        }
    }
}

impl PvmHooks for MintCaptureHooks {
    fn write_debug_bytes(&mut self, bytes: &[u8]) {
        self.log_file.write_all(bytes).unwrap();
    }
}

#[test]
fn regression_frozen_dummy_kernel() {
    test_regression(DUMMY, true)
}

#[test]
fn regression_dummy_kernel() {
    test_regression(DUMMY_UNCHECKED, false)
}

#[test]
fn regression_frozen_jstz() {
    test_regression(JSTZ, true)
}

#[test]
fn regression_frozen_etherlink() {
    test_regression(ETHERLINK, true)
}

fn test_regression(inputs: TestConfig, capture_volatile_properties: bool) {
    test_regression_for_block::<Interpreted<M64M, Owned>>(
        InterpretedCompiler,
        &inputs,
        capture_volatile_properties,
    );

    // This needs to run *after* the previous *interpreted* test. Otherwise, we run into trouble when
    // checking and updating the golden files.
    test_regression_for_block::<Jitted<_, _>>(
        OutlineCompiler::<M64M>::default(),
        &inputs,
        capture_volatile_properties,
    );
}

fn test_regression_for_block<CPE: CodePageEntry<M64M, Owned>>(
    compiler: CPE::Compiler,
    inputs: &TestConfig,
    capture_volatile_properties: bool,
) {
    let mut mint = goldenfile::Mint::new(inputs.golden_dir);

    let (result, initial_hash, final_hash) = {
        // We need to read the kernel in any case
        let program = fs::read(inputs.kernel_path)
            .expect("Failed to read kernel from disk. Try running `make build`.");

        let inbox = {
            let mut inbox = InboxBuilder::new();
            inbox.load_from_file(inputs.inbox_path).unwrap();
            inbox.build()
        };

        let hooks = MintCaptureHooks::new(&mut mint);

        const ROLLUP_ADDRESS: [u8; 20] = [
            244, 228, 124, 179, 196, 58, 104, 176, 212, 142, 48, 148, 9, 44, 164, 45, 113, 58, 221,
            181,
        ];
        const ORIGINATION_LEVEL: u32 = 1;

        let mut stepper = PvmStepper::<_, M64M, Owned, CPE>::new(
            &program,
            inbox,
            hooks,
            ROLLUP_ADDRESS,
            ORIGINATION_LEVEL,
            Some(PathBuf::from("../../../assets/preimages").into_boxed_path()),
            compiler,
        )
        .unwrap();

        let initial_hash = stepper.hash();

        let result = stepper.step_max(Bound::Unbounded);
        let final_hash = stepper.hash();

        (result, initial_hash, final_hash)
    };

    assert!(
        matches!(result, StepperStatus::Exited { .. }),
        "Unexpected result: {result:?}"
    );

    if capture_volatile_properties {
        let mut initial_hash_capture = mint.new_goldenfile("state_hash_initial").unwrap();
        writeln!(initial_hash_capture, "{initial_hash:?}").unwrap();

        let mut result_capture = mint.new_goldenfile("result").unwrap();
        writeln!(result_capture, "{result:#?}").unwrap();

        let mut final_hash_capture = mint.new_goldenfile("state_hash_final").unwrap();
        writeln!(final_hash_capture, "{final_hash:?}").unwrap();
    }
}
