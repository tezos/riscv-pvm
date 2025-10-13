// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod profile;
mod sample;

use std::boxed::Box;
use std::error;
use std::error::Error;
use std::fs;
use std::ops::Bound;
use std::time::Duration;

use octez_riscv::machine_state::memory;
use octez_riscv::machine_state::page_cache;
use octez_riscv::machine_state::page_cache::CodePageEntry;
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
use crate::memory_config::MemoryConfigValue;

cfg_if::cfg_if! {
    if #[cfg(feature = "disable-jit")] {
        /// Execution strategy for entrypoints.
        pub type CodePageEntryImpl<MC> = page_cache::Interpreted<MC, octez_riscv::state_backend::owned_backend::Owned>;
    } else if #[cfg(feature = "inline-jit")] {
        /// Execution strategy for entrypoints.
        pub type CodePageEntryImpl<MC> = page_cache::Jitted<page_cache::InlineCompiler<MC>, MC>;
    } else {
        /// Execution strategy for entrypoints.
        pub type CodePageEntryImpl<MC> = page_cache::Jitted<page_cache::OutlineCompiler<MC>, MC>;
    }
}

pub fn run_with_memory_config<MC: memory::MemoryConfig>(
    opts: RunOptions,
) -> Result<(), Box<dyn Error>> {
    let program = fs::read(&opts.input)?;

    let stepper = make_pvm_stepper::<MC, CodePageEntryImpl<MC>>(
        program.as_slice(),
        &opts.common,
        Default::default(),
    )?;

    // Run the profiler if a sampling interval is set
    let steps = match opts.sample_interval_us {
        None => run_stepper(stepper, opts.common.max_steps)?,
        Some(sample_interval_us) => {
            let sample_interval = Duration::from_micros(sample_interval_us);
            profile::profile_stepper(
                stepper,
                program.as_slice(),
                sample_interval,
                opts.common.max_steps,
                opts.output.as_ref(),
            )?
        }
    };

    if opts.print_steps {
        println!("Run consumed {steps} steps.");
    }

    Ok(())
}

pub fn run(opts: RunOptions) -> Result<(), Box<dyn Error>> {
    // Promote the memory configuration value to the appropriate type, then continue.
    match opts.common.memory_config {
        MemoryConfigValue::M64M => run_with_memory_config::<memory::M64M>(opts),
        MemoryConfigValue::M1G => run_with_memory_config::<memory::M1G>(opts),
        MemoryConfigValue::M4G => run_with_memory_config::<memory::M4G>(opts),
        MemoryConfigValue::M16G => run_with_memory_config::<memory::M16G>(opts),
        MemoryConfigValue::M64G => run_with_memory_config::<memory::M64G>(opts),
    }
}

type PvmStepperRunner<MC, CPE> = PvmStepper<Console<'static>, MC, Owned, CPE>;

pub(crate) fn make_pvm_stepper<MC: memory::MemoryConfig, CPE: CodePageEntry<MC, Owned>>(
    program: &[u8],
    common: &CommonOptions,
    compiler: CPE::Compiler,
) -> Result<PvmStepperRunner<MC, CPE>, Box<dyn error::Error>> {
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

    let stepper = PvmStepper::<_, MC, Owned, CPE>::new(
        program,
        inbox.build(),
        console,
        rollup_address.into_hash().as_ref().try_into()?,
        common.inbox.origination_level,
        common.preimage.preimages_dir.clone(),
        compiler,
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
