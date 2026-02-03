// SPDX-FileCopyrightText: 2024-2026 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024-2026 TriliTech <contact@trili.tech>
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
use octez_riscv::machine_state::page_cache::PageCache;
use octez_riscv::stepper::StepResult;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv_data::mode::Normal;
use tezos_smart_rollup::utils::console::Console;
use tezos_smart_rollup::utils::inbox::InboxBuilder;
use tezos_smart_rollup_encoding::smart_rollup::SmartRollupAddress;

use crate::cli::CommonOptions;
use crate::cli::RunOptions;
use crate::memory_config::MemoryConfigValue;

cfg_if::cfg_if! {
    if #[cfg(feature = "disable-jit")] {
        /// PageCache with interpreted mode selected.
        pub type PageCacheImpl<MC> = page_cache::PageCacheInterpreted<MC>;
    } else if #[cfg(feature = "inline-jit")] {
        /// PageCache with inline jit mode selected.
        pub type PageCacheImpl<MC> = page_cache::PageCacheInlineJit<MC>;
    } else {
        /// PageCache with outline jit mode selected.
        pub type PageCacheImpl<MC> = page_cache::PageCacheOutlineJit<MC>;
    }
}

pub fn run_with_memory_config<MC: memory::MemoryConfig>(
    opts: RunOptions,
) -> Result<(), Box<dyn Error>> {
    let program = fs::read(&opts.input)?;

    let stepper = make_pvm_stepper::<MC, PageCacheImpl<MC>>(program.as_slice(), &opts.common)?;

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

type PvmStepperRunner<MC, PC> = PvmStepper<Console<'static>, MC, Normal, PC>;

pub(crate) fn make_pvm_stepper<MC: memory::MemoryConfig, PC: PageCache<MC, Normal>>(
    program: &[u8],
    common: &CommonOptions,
) -> Result<PvmStepperRunner<MC, PC>, Box<dyn error::Error>> {
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

    let stepper = PvmStepper::<_, MC, Normal, PC>::new(
        program,
        inbox.build(),
        console,
        rollup_address.into_hash().as_ref().try_into()?,
        common.inbox.origination_level,
        common.preimage.preimages_dir.clone(),
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
