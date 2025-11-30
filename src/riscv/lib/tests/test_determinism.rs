// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::ops::Bound;

use octez_riscv::machine_state::memory::M64M;
use octez_riscv::machine_state::memory::MemoryConfig;
use octez_riscv::pvm::PvmLayout;
use octez_riscv::pvm::hooks::NoHooks;
use octez_riscv::state_backend::RefNormalAlloc;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::mode::Normal;
use octez_riscv_test_utils::*;

#[test]
fn test_jstz_determinism() {
    test_determinism(JSTZ)
}

#[test]
fn test_etherlink_determinism() {
    test_determinism(ETHERLINK)
}

fn test_determinism(inputs: TestConfig) {
    let make_stepper = make_stepper_factory(&inputs);

    let mut base_stepper = make_stepper();
    let base_result = base_stepper.step_max(Bound::Unbounded);
    assert!(matches!(base_result, StepperStatus::Exited { .. }));

    let steps = base_result.steps();
    let base_hash = base_stepper.hash();

    // If we re-do it with identical number of steps, that should work exactly the same.
    let mut stepper = make_stepper();
    let result = stepper.step_max(Bound::Included(steps));
    assert_eq!(base_result, result);
    assert_eq!(base_hash, stepper.hash());

    eprintln!("Final hash is {base_hash}");
    eprintln!("Final result is {base_result:?}");

    // Create multiple series of bisections that we will evaluate.
    let ladder = dissect_steps(steps, 0);
    run_steps_ladder::<M64M, _>(&make_stepper, &ladder, base_hash);
}

fn run_steps_ladder<MC, F>(make_stepper: F, ladder: &[usize], expected_hash: hash::Hash)
where
    MC: MemoryConfig,
    MC::State<Normal>: Foldable<HashFold>,
    for<'a> RefNormalAlloc<'a, PvmLayout<MC>>: PartialEq,
    F: Fn() -> PvmStepper<NoHooks, MC>,
{
    let expected_steps = ladder.iter().sum::<usize>();
    let mut stepper_lhs = make_stepper();
    let mut stepper_rhs = make_stepper();

    assert_eq!(
        stepper_lhs.hash(),
        stepper_rhs.hash(),
        "Stepper states have diverged before doing anything"
    );

    let mut steps_done = 0;
    for &steps in ladder {
        eprintln!("> Running {steps} steps ...");
        let result_lhs = stepper_lhs.step_max(Bound::Included(steps));
        let result_rhs = stepper_rhs.step_max(Bound::Included(steps));
        steps_done += steps;

        eprintln!(
            "> Done {:.2}%",
            (steps_done as f64 / expected_steps as f64) * 100.0
        );

        assert_eq!(result_lhs, result_rhs);
        assert_eq!(
            result_lhs.steps(),
            steps,
            "Expected {} steps to be run, but got {}",
            steps,
            result_lhs.steps()
        );
        assert_eq!(
            stepper_lhs.hash(),
            stepper_rhs.hash(),
            "Stepper states have diverged after running {steps} steps"
        );

        stepper_lhs.rebind_via_clone();
    }

    assert_eq!(stepper_lhs.hash(), expected_hash);
    assert_eq!(stepper_rhs.hash(), expected_hash);
}
