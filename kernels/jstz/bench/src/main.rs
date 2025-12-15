// SPDX-FileCopyrightText: 2024-2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::error::Error;
use std::path::Path;

use clap::Parser;
use clap::Subcommand;
use generate::handle_generate;
use generate::handle_generate_script;
use results::handle_results;

mod generate;
mod results;

const DEFAULT_ROLLUP_ADDRESS: &str = "sr163Lv22CdE8QagCwf48PWDTquk6isQwv57";

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

#[derive(Debug, Parser)]
#[command(long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(about = "Generate inbox.json file")]
    Generate {
        #[arg(long, default_value = DEFAULT_ROLLUP_ADDRESS)]
        address: String,
        #[arg(long)]
        transfers: usize,
        #[arg(long, default_value = "inbox.json")]
        inbox_file: Box<Path>,
    },
    #[command(about = "Generate inbox.sh script")]
    GenerateScript {
        #[arg(long, default_value = DEFAULT_ROLLUP_ADDRESS)]
        address: String,
        #[arg(long)]
        transfers: usize,
        #[arg(long, default_value = "inbox.sh")]
        script_file: Box<Path>,
    },
    #[command(about = "Extract results from inbox.json & log file")]
    Results {
        #[arg(long)]
        inbox_file: Box<Path>,
        #[arg(long)]
        log_file: Vec<Box<Path>>,
        #[arg(long)]
        expected_transfers: usize,
        #[arg(long)]
        collapsible_results: bool,
        #[arg(long, default_value = "0")]
        exclude_warmup_transfers: usize,
        /// If set, submit the mean of the transfers to datadog as a metric.
        ///
        /// Requires the `DD_API_KEY` and `DD_SITE` environment variables to be set.
        #[arg(long, default_value = "false")]
        submit_dd_metric: bool,
    },
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Generate {
            address,
            inbox_file,
            transfers,
        } => {
            tokio::task::spawn_blocking(move || {
                handle_generate(&address, &inbox_file, transfers).map_err(|err| err.to_string())
            })
            .await??;
        }
        Commands::GenerateScript {
            address,
            script_file,
            transfers,
        } => {
            tokio::task::spawn_blocking(move || {
                handle_generate_script(&address, &script_file, transfers)
                    .map_err(|err| err.to_string())
            })
            .await??;
        }
        Commands::Results {
            inbox_file,
            log_file,
            expected_transfers,
            collapsible_results,
            exclude_warmup_transfers,
            submit_dd_metric,
        } => {
            let mean_tps = tokio::task::spawn_blocking(move || {
                handle_results(
                    inbox_file,
                    log_file,
                    expected_transfers,
                    collapsible_results,
                    exclude_warmup_transfers,
                )
                .map_err(|err| err.to_string())
            })
            .await??
            .tps();

            if submit_dd_metric {
                kernel_bench_utils::datadog::submit_kernel_tps_benchmark("jstz", "fa2", mean_tps)
                    .await?;
            }
        }
    }

    Ok(())
}
