// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Common utilities for octez-riscv integration tests

use std::fs;

use const_format::concatcp;
use octez_riscv::machine_state::memory::MemoryConfig;
use octez_riscv::machine_state::page_cache::InterpretedCompiler;
use octez_riscv::pvm::hooks::NoHooks;
use octez_riscv::stepper::pvm::PvmStepper;
use rand::Rng;
use rand::seq::SliceRandom;
use tezos_smart_rollup_utils::inbox::InboxBuilder;

const ASSETS_DIR: &str = std::env!("OCTEZ_RISCV_ASSETS_DIR");
const KERNELS_DIR: &str = std::env!("OCTEZ_RISCV_KERNELS_DIR");
const LIB_TESTS_DIR: &str = std::env!("OCTEZ_RISCV_LIB_TESTS_DIR");

/// Kernel-specific configuration for tests which run over kernels
pub struct TestConfig {
    /// Path to regressions directory
    pub golden_dir: &'static str,
    /// Path to kernel binary
    pub kernel_path: &'static str,
    /// Path to inbox
    pub inbox_path: &'static str,
}

/// Test configuration for the dummy kernel
pub const DUMMY: TestConfig = TestConfig {
    golden_dir: concatcp!(LIB_TESTS_DIR, "/expected/dummy"),
    kernel_path: concatcp!(ASSETS_DIR, "/riscv-dummy.elf"),
    inbox_path: concatcp!(ASSETS_DIR, "/dummy-kernel-inbox.json"),
};

/// Test configuration which uses the compiled version of the dummy kernel
/// instead of the checked-in kernel
pub const DUMMY_UNCHECKED: TestConfig = TestConfig {
    kernel_path: concatcp!(
        KERNELS_DIR,
        "/dummy/target/riscv64gc-unknown-linux-musl/release/riscv-dummy"
    ),
    ..DUMMY
};

/// Test configuration for the Jstz kernel
pub const JSTZ: TestConfig = TestConfig {
    golden_dir: concatcp!(LIB_TESTS_DIR, "/expected/jstz"),
    kernel_path: concatcp!(ASSETS_DIR, "/jstz"),
    inbox_path: concatcp!(ASSETS_DIR, "/jstz-regression-inbox.json"),
};

/// Test configuration for the Etherlink kernel
pub const ETHERLINK: TestConfig = TestConfig {
    golden_dir: concatcp!(LIB_TESTS_DIR, "/expected/etherlink"),
    kernel_path: concatcp!(ASSETS_DIR, "/etherlink"),
    inbox_path: concatcp!(ASSETS_DIR, "/etherlink-regression-inbox.json"),
};

/// Return a function which can produce a [`PvmStepper`] over a given [`TestConfig`].
pub fn make_stepper_factory<MC: MemoryConfig>(
    inputs: &TestConfig,
) -> impl Fn() -> PvmStepper<NoHooks, MC> {
    let program = fs::read(inputs.kernel_path).expect("Kernel path should be valid");

    let mut inbox = InboxBuilder::new();
    inbox
        .load_from_file(inputs.inbox_path)
        .expect("Inbox path should be valid");
    let inbox = inbox.build();

    let address = [0; 20];

    move || {
        let compiler = InterpretedCompiler;

        PvmStepper::<NoHooks, MC>::new(&program, inbox.clone(), NoHooks, address, 1, None, compiler)
            .expect("PvmStepper initialisation arguments should be valid")
    }
}

/// Given a minimum stepping interval `min_interval`, produce a random sequence
/// of step increments which add up to `total_steps`.
pub fn dissect_steps(mut total_steps: usize, min_interval: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let mut steps: Vec<usize> = std::iter::from_fn(|| {
        if total_steps == 0 {
            return None;
        }

        let steps = total_steps.div_euclid(2).max(min_interval + 1);
        let steps = rng.random_range(min_interval..=steps);

        total_steps = total_steps.saturating_sub(steps);

        Some(steps)
    })
    .collect();
    steps.shuffle(&mut rng);
    steps
}
