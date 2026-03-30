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
use std::path::PathBuf;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use octez_riscv::machine_state::memory;
use octez_riscv::machine_state::page_cache;
use octez_riscv::machine_state::page_cache::PageCache;
use octez_riscv::stepper::StepResult;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::commit::CommitId;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_durable_storage::registry::Registry;
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
    let durable_storage_dir = durable_storage_dir(&opts.common)?;

    let mut stepper =
        make_pvm_stepper::<MC, PageCacheImpl<MC>>(program.as_slice(), &opts.common, &durable_storage_dir)?;

    // Run the profiler if a sampling interval is set
    let steps = match opts.sample_interval_us {
        None => run_stepper(&mut stepper, opts.common.max_steps)?,
        Some(sample_interval_us) => {
            let sample_interval = Duration::from_micros(sample_interval_us);
            profile::profile_stepper(
                &mut stepper,
                program.as_slice(),
                sample_interval,
                opts.common.max_steps,
                opts.output.as_ref(),
            )?
        }
    };

    persist_durable_storage_head(stepper.durable_storage_mut(), &durable_storage_dir)?;

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

type RegistryDurableStorage = Registry<PersistenceLayer, Normal>;
type PvmStepperRunner<MC, PC> =
    PvmStepper<Console<'static>, MC, RegistryDurableStorage, PC, Normal>;
const DURABLE_STORAGE_HEAD_FILE: &str = "registry-head";

pub(crate) fn durable_storage_dir(common: &CommonOptions) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = &common.durable_storage_dir {
        return Ok(path.clone());
    }

    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!("riscv-sandbox-durable-storage-{unique}"));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

pub(crate) fn make_pvm_stepper<MC: memory::MemoryConfig, PC: PageCache<MC, Normal>>(
    program: &[u8],
    common: &CommonOptions,
    durable_storage_dir: &PathBuf,
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

    let durable_storage_repo = DirectoryManager::new(&durable_storage_dir)?;
    let durable_storage = load_durable_storage(durable_storage_repo, &durable_storage_dir)?;

    let stepper =
        PvmStepper::<_, MC, RegistryDurableStorage, PC, Normal>::new_with_durable_storage(
            program,
            inbox.build(),
            console,
            rollup_address.into_hash().as_ref().try_into()?,
            common.inbox.origination_level,
            common.preimage.preimages_dir.clone(),
            durable_storage,
        )?;

    Ok(stepper)
}

fn load_durable_storage(
    repo: DirectoryManager,
    durable_storage_dir: &PathBuf,
) -> Result<Registry<PersistenceLayer, Normal>, Box<dyn Error>> {
    let head_path = durable_storage_dir.join(DURABLE_STORAGE_HEAD_FILE);
    if !head_path.exists() {
        return Ok(Registry::<PersistenceLayer, Normal>::new(repo)?);
    }

    let head_hex = std::fs::read_to_string(&head_path)?;
    let head_hex = head_hex.trim();
    let head_bytes = hex::decode(head_hex)?;
    let head_array: [u8; Hash::DIGEST_SIZE] = head_bytes
        .try_into()
        .map_err(|_| format!("invalid durable storage head in {}", head_path.display()))?;
    let commit_id = CommitId::from(Hash::from(head_array));
    Ok(Registry::<PersistenceLayer, Normal>::checkout(repo, commit_id)?)
}

fn persist_durable_storage_head(
    durable_storage: &mut Registry<PersistenceLayer, Normal>,
    durable_storage_dir: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let commit_id = durable_storage.commit()?;
    let head_path = durable_storage_dir.join(DURABLE_STORAGE_HEAD_FILE);
    std::fs::write(head_path, commit_id.hex_encode())?;
    Ok(())
}

fn run_stepper(
    stepper: &mut impl Stepper,
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
