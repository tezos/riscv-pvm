// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::boxed::Box;
use std::error;
use std::error::Error;
use std::fs;
use std::ops::Bound;

use octez_riscv::machine_state::block_cache::DefaultCacheConfig;
use octez_riscv::machine_state::block_cache::block;
use octez_riscv::machine_state::block_cache::block::Block;
use octez_riscv::machine_state::memory::M1G;
use octez_riscv::state_backend::owned_backend::Owned;
use octez_riscv::stepper::StepResult;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use tezos_smart_rollup::utils::console::Console;
use tezos_smart_rollup::utils::inbox::InboxBuilder;
use tezos_smart_rollup_encoding::smart_rollup::SmartRollupAddress;

use crate::cli::CommonOptions;
use crate::cli::RunOptions;

cfg_if::cfg_if! {
    if #[cfg(feature = "disable-jit")] {
        /// Inner execution strategy for blocks.
        type BlockImplInner = block::Interpreted<M1G, Owned>;
    } else if #[cfg(feature = "inline-jit")] {
        /// Inner execution strategy for blocks.
        type BlockImplInner = block::Jitted<octez_riscv::machine_state::block_cache::block::InlineCompiler<M1G>, M1G>;
    } else {
        /// Inner execution strategy for blocks.
        type BlockImplInner = block::Jitted<octez_riscv::machine_state::block_cache::block::OutlineCompiler<M1G>, M1G>;
    }
}

/// Executor of blocks
#[cfg(not(feature = "metrics"))]
pub type BlockImpl = BlockImplInner;

/// Executor of blocks
#[cfg(feature = "metrics")]
pub type BlockImpl = octez_riscv::machine_state::block_cache::metrics::BlockMetrics<BlockImplInner>;

pub fn run(opts: RunOptions) -> Result<(), Box<dyn Error>> {
    let program = fs::read(&opts.input)?;

    let stepper =
        make_pvm_stepper::<BlockImpl>(program.as_slice(), &opts.common, Default::default())?;

    let steps = run_stepper(stepper, opts.common.max_steps)?;

    if opts.print_steps {
        println!("Run consumed {steps} steps.");
    }

    #[cfg(feature = "metrics")]
    octez_riscv::dump_block_metrics!(
        file = &opts.metrics.block_metrics_file,
        exclude_supported_instructions = opts.metrics.exclude_supported_instructions
    )?;

    Ok(())
}

type PvmStepperRunner<B> = PvmStepper<Console<'static>, M1G, DefaultCacheConfig, Owned, B>;

pub(crate) fn make_pvm_stepper<B: Block<M1G, Owned>>(
    program: &[u8],
    common: &CommonOptions,
    block_builder: B::BlockBuilder,
) -> Result<PvmStepperRunner<B>, Box<dyn error::Error>> {
    let mut inbox = InboxBuilder::new();
    if let Some(inbox_file) = &common.inbox.file {
        inbox.load_from_file(inbox_file)?;
    }

    let rollup_address = SmartRollupAddress::from_b58check(common.inbox.address.as_str())?;

    let console = if common.timings {
        Console::with_timings()
    } else {
        Console::new()
    };

    let stepper = PvmStepper::<_, M1G, DefaultCacheConfig, Owned, B>::new(
        program,
        inbox.build(),
        console,
        rollup_address.into_hash().as_ref().try_into().unwrap(),
        common.inbox.origination_level,
        common.preimage.preimages_dir.clone(),
        block_builder,
    )?;

    Ok(stepper)
}

fn run_stepper(
    mut stepper: impl Stepper,
    max_steps: Option<usize>,
) -> Result<usize, Box<dyn Error>> {
    let max_steps = match max_steps {
        Some(max_steps) => Bound::Included(max_steps),
        None => Bound::Unbounded,
    };

    let result = stepper.step_max(max_steps);

    match result.to_stepper_status() {
        StepperStatus::Exited {
            success: true,
            steps,
            ..
        } => Ok(steps),
        result => Err(format!("{result:?}").into()),
    }
}
