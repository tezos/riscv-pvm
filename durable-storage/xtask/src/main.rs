// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! xtask binary for the `durable-storage` crate

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use octez_riscv_durable_storage::test_helpers::OperationView;
use octez_riscv_durable_storage::test_helpers::database::DatabaseOperation;
use octez_riscv_durable_storage::test_helpers::database::DatabaseOperationView;
use octez_riscv_durable_storage::test_helpers::database::make_database_operations;
use proptest::prelude::Strategy;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

const DEFAULT_OUT_DIR: &str = "durable-storage/tests/inputs";

/// Default number of regression inputs to generate
const DEFAULT_COUNT: usize = 5;

/// Number of operations to sample for each input
const OPS_RANGE: std::ops::Range<usize> = 90..100;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Regenerate the inputs used by `Database` regression tests
    GenDatabaseRegressionInputs {
        /// Number of regression inputs to generate
        #[arg(long, default_value_t = DEFAULT_COUNT)]
        count: usize,
        /// Destination directory, relative to the workspace root.
        #[arg(long, default_value = DEFAULT_OUT_DIR)]
        out_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::GenDatabaseRegressionInputs { count, out_dir } => {
            gen_database_regression_inputs(count, &out_dir)
        }
    }
}

fn gen_database_regression_inputs(count: usize, out_dir: &Path) -> Result<()> {
    let out_dir = find_repo_root()?.join(out_dir);
    if !out_dir.exists() || out_dir.metadata()?.is_file() {
        bail!("output directory {} does not exist", out_dir.display());
    }

    let mut runner = TestRunner::default();

    for i in 0..count {
        let tree = DatabaseOperationView::operations_strategy(OPS_RANGE)
            .new_tree(&mut runner)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("drawing input #{i}"))?;
        let (keys, values, ops_view) = tree.current();
        let ops: Vec<DatabaseOperation> = make_database_operations(keys, values, ops_view);

        let path = out_dir.join(format!("database_{i:02}.input"));
        let file =
            fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        serde_json::to_writer_pretty(file, &ops)
            .with_context(|| format!("serialising input #{i}"))?;
        println!("wrote {}", path.display());
    }

    Ok(())
}

// TODO TZX-145: move to a xtask helpers crate instead of duplicating in other xtasks
fn find_repo_root() -> Result<PathBuf> {
    let current_dir = env::current_dir().context("Failed to get current directory")?;

    let mut dir = current_dir.as_path();
    loop {
        if dir.join(".github").exists() {
            return Ok(dir.to_path_buf());
        }

        match dir.parent() {
            Some(parent) => dir = parent,
            None => bail!("Could not find repository root"),
        }
    }
}
