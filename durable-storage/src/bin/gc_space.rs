// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Binary for the durable storage space accounting harness.
//!
//! Reports how much of a commit's Merkle node data is still reachable and how much is dead. See
//! [`octez_riscv_durable_storage::gc_space`] for what is measured and why.
//!
//! Point `--repo-dir` at a volume with room to spare: the base state is recorded there and reused
//! by later runs of the same shape, which matters because prepopulating dominates a large run. Each
//! run resets the repository to that base state first, so reusing a directory does not fold earlier
//! runs' commits into what is reported.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use octez_riscv_durable_storage::gc_space::SpaceConfig;

#[derive(Debug, Parser)]
#[command(version, long_about = None)]
struct Cli {
    /// Number of databases in the registry.
    #[arg(long, default_value_t = 1)]
    databases: usize,

    /// Keys written to each database before the measured commits begin.
    #[arg(long, default_value_t = 100_000)]
    keys: usize,

    /// Length of each key, in bytes.
    ///
    /// The Etherlink trace has a median path of 33 bytes and a p90 of 99, with storage slots
    /// (`/evm/world_state/eth_accounts/<address>/storage/<slot>`) around 130. Keys are stored inline
    /// in each Merkle node, so this drives node size directly.
    #[arg(long, default_value_t = 130)]
    key_size: usize,

    /// Length of an ordinary value, in bytes. The trace's median write is 32.
    #[arg(long, default_value_t = 32)]
    value_size: usize,

    /// Fraction of keys holding a large value, reproducing the trace's contract-code tail.
    #[arg(long, default_value_t = 0.0)]
    large_value_fraction: f64,

    /// Length of a large value, in bytes. The trace's largest write is 131220.
    #[arg(long, default_value_t = 131_220)]
    large_value_size: usize,

    /// Number of commits to make and measure.
    #[arg(long, default_value_t = 20)]
    commits: usize,

    /// Keys modified before each commit, spread across the registry's databases.
    #[arg(long, default_value_t = 1_000)]
    modified_keys: usize,

    /// Measure every this many commits. Raise it at large scales, where scanning a column family
    /// costs more than the commits do.
    #[arg(long, default_value_t = 1)]
    sample_every: usize,

    /// Seed for choosing which keys each commit modifies.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Where the repository lives. A fresh temporary directory when absent, which is removed on
    /// exit along with the base state.
    #[arg(long)]
    repo_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.databases == 0 {
        anyhow::bail!("--databases must be at least 1");
    }

    if cli.keys == 0 {
        anyhow::bail!("--keys must be at least 1");
    }

    if cli.sample_every == 0 {
        anyhow::bail!("--sample-every must be at least 1");
    }

    if !(0.0..=1.0).contains(&cli.large_value_fraction) {
        anyhow::bail!("--large-value-fraction must be between 0 and 1");
    }

    SpaceConfig {
        databases: cli.databases,
        keys_per_database: cli.keys,
        key_size: cli.key_size,
        value_size: cli.value_size,
        large_value_fraction: cli.large_value_fraction,
        large_value_size: cli.large_value_size,
        commits: cli.commits,
        modified_keys: cli.modified_keys,
        sample_every: cli.sample_every,
        seed: cli.seed,
        repo_dir: cli.repo_dir,
    }
    .run()
}
