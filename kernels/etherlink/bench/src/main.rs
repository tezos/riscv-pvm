// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::error::Error;
use std::path::Path;

use clap::Parser;
use clap::Subcommand;
use results::handle_results;

mod results;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

#[derive(Debug, Parser)]
#[command(long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(about = "Extract results from inbox and log files")]
    Results {
        #[arg(long)]
        inbox_file: Box<Path>,
        #[arg(long)]
        log_file: Box<Path>,
        #[arg(long)]
        expected_transfers: usize,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Results {
            inbox_file,
            log_file,
            expected_transfers,
        } => handle_results(inbox_file, log_file, expected_transfers)?,
    }

    Ok(())
}
