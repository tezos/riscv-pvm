// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Binary for the long-running durable storage [`Database`] test.
//!
//! See [`octez_riscv_durable_storage::long_test`] for details
//! about long tests.
//!
//! [`Database`]: octez_riscv_durable_storage::database::Database

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use octez_riscv_data::hash::Hash;
use octez_riscv_durable_storage::long_test::LongTestConfig;
use octez_riscv_durable_storage::long_test::database::replay_failure;
use octez_riscv_durable_storage::long_test::database::run_long_test;

#[derive(Debug, Parser)]
#[command(version, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run the long test.
    Test {
        /// Target number of operations sampled per epoch.
        #[arg(long, default_value_t = 1000)]
        ops_per_epoch: usize,

        /// Number of test cases per epoch.
        #[arg(long, default_value_t = 256)]
        cases_per_epoch: u32,

        /// 32-byte hex-encoded test seed (default: run with a fresh seed, printed at startup).
        #[arg(long)]
        seed: Option<String>,

        /// Time budget in minutes (default: none).
        #[arg(long)]
        max_minutes: Option<u64>,

        /// Maximum number of epochs to run (default: run until the time budget).
        #[arg(long)]
        epochs: Option<u64>,

        /// Number of most recent epoch snapshots to keep; older snapshots and
        /// commits are cleared after each successful epoch (default: keep everything).
        #[arg(long)]
        keep_epochs: Option<NonZeroUsize>,

        /// Directory for run state and failure artifacts (default: a fresh tempdir).
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Replay the failing epoch described by `<DIR>/meta.json`.
    Replay {
        /// Failure directory containing the recorded artifacts.
        dir: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Test {
            ops_per_epoch,
            cases_per_epoch,
            seed,
            max_minutes,
            epochs,
            keep_epochs,
            out_dir,
        } => {
            let seed = match seed {
                Some(seed) => {
                    let bytes = hex::decode(&seed).context("decoding hex seed")?;
                    let array: [u8; Hash::DIGEST_SIZE] =
                        bytes.as_slice().try_into().map_err(|_| {
                            anyhow::anyhow!("seed must be exactly 32 bytes ({} given)", bytes.len())
                        })?;
                    Some(Hash::from(array))
                }
                None => None,
            };

            let config = LongTestConfig {
                ops_per_epoch,
                cases_per_epoch,
                seed,
                time_budget: max_minutes.map(|m| Duration::from_secs(m * 60)),
                epochs,
                keep_epochs,
                out_dir,
            };

            run_long_test(config)
        }
        Commands::Replay { dir } => replay_failure(&dir),
    }
}
