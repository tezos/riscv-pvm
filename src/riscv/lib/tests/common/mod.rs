// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fs;

use octez_riscv::machine_state::block_cache::BlockCacheConfig;
use octez_riscv::machine_state::block_cache::block::InterpretedBlockBuilder;
use octez_riscv::machine_state::memory::M64M;
use octez_riscv::pvm::hooks::NoHooks;
use octez_riscv::stepper::pvm::PvmStepper;
use rand::Rng;
use rand::seq::SliceRandom;
use tezos_smart_rollup_utils::inbox::InboxBuilder;

pub fn make_stepper_factory<BCC: BlockCacheConfig>() -> impl Fn() -> PvmStepper<NoHooks, M64M, BCC>
{
    let program = fs::read("../assets/jstz").unwrap();

    let mut inbox = InboxBuilder::new();
    inbox
        .load_from_file("../assets/regression-inbox.json")
        .unwrap();
    let inbox = inbox.build();

    let address = [0; 20];

    move || {
        let block_builder = InterpretedBlockBuilder;

        PvmStepper::<NoHooks, M64M, BCC>::new(
            &program,
            inbox.clone(),
            NoHooks,
            address,
            1,
            None,
            block_builder,
        )
        .unwrap()
    }
}

pub fn dissect_steps(mut max_steps: usize, min_step_size: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let mut steps: Vec<usize> = std::iter::from_fn(|| {
        if max_steps == 0 {
            return None;
        }

        let steps = max_steps.div_euclid(2).max(min_step_size + 1);
        let steps = rng.random_range(min_step_size..=steps);

        max_steps = max_steps.saturating_sub(steps);

        Some(steps)
    })
    .collect();
    steps.shuffle(&mut rng);
    steps
}
