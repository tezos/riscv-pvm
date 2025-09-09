// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::env;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use xshell::Shell;
use xshell::cmd;

const ETHERLINK_DIR: &str = "kernels/etherlink";
const SANDBOX_DIR: &str = "tools/sandbox";
const SANDBOX_BIN: &str = "riscv-sandbox";

const TX_COUNT: usize = 200;
const DEFAULT_ROLLUP_ADDRESS: &str = "sr163Lv22CdE8QagCwf48PWDTquk6isQwv57";
const DEFAULT_INBOX: &str = "assets/etherlink-erc20-inbox.json";

#[derive(Debug, Default)]
struct BuildConfig {
    static_inbox: bool,
    native: bool,
    tracing: bool,
    profiling: bool,
    data_dir: Option<PathBuf>,
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run Etherlink benchmark
    Bench {
        /// Use static inbox
        #[arg(short, long)]
        static_inbox: bool,
        /// Run natively
        #[arg(short, long)]
        native: bool,
        /// Produce kernel run trace
        #[arg(short, long)]
        tracing: bool,
        #[command(flatten)]
        common: CommonOptions,
    },
    Profile {
        /// Sampling interval in microseconds
        #[arg(short, long, default_value = "500")]
        sample_interval_us: u64,
        #[command(flatten)]
        common: CommonOptions,
    },
}

#[derive(Debug, Clone, Parser)]
pub struct CommonOptions {
    /// Data directory
    #[arg(env = "DATA_DIR")]
    data_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench {
            static_inbox,
            native,
            tracing,
            common,
        } => {
            let config = BuildConfig {
                static_inbox,
                native,
                tracing,
                profiling: false,
                data_dir: common.data_dir,
            };

            validate_config(&config)?;
            run_benchmark(config)
        }
        Commands::Profile {
            sample_interval_us,
            common,
        } => {
            let config = BuildConfig {
                profiling: true,
                data_dir: common.data_dir,
                ..BuildConfig::default()
            };

            run_profile(config, sample_interval_us)
        }
    }
}

fn validate_config(config: &BuildConfig) -> Result<()> {
    if config.native && !config.static_inbox {
        bail!("Native compilation without static inbox unsupported");
    }

    if config.tracing && config.native {
        bail!("Tracing only supported for RISC-V kernel");
    }

    Ok(())
}

fn run_benchmark(config: BuildConfig) -> Result<()> {
    let repo_root = find_repo_root()?;
    let inbox_file = repo_root.join(DEFAULT_INBOX);

    let sh = Shell::new()?;

    println!("[INFO]: Building RISC-V sandbox");
    build_sandbox(&sh, &repo_root)?;

    println!("[INFO]: Building bench tool");
    build_bench_tool(&sh, &repo_root)?;

    println!("[INFO]: Building Etherlink kernel");
    build_etherlink_kernel(&sh, &repo_root, &config, &inbox_file)?;

    let data_dir = init_data_dir(&config)?;

    run(&sh, &config, &repo_root, &inbox_file, &data_dir)?;

    if !config.tracing {
        print_bench_results(&sh, &repo_root, &inbox_file, &data_dir)?
    }

    Ok(())
}

fn run_profile(config: BuildConfig, sample_interval_us: u64) -> Result<()> {
    let repo_root = find_repo_root()?;
    let inbox_file = repo_root.join(DEFAULT_INBOX);

    let sh = Shell::new()?;

    println!("[INFO]: Building RISC-V sandbox");
    build_sandbox(&sh, &repo_root)?;

    println!("[INFO]: Building Etherlink kernel");
    build_etherlink_kernel(&sh, &repo_root, &config, &inbox_file)?;

    let data_dir = init_data_dir(&config)?;

    let sandbox_path = repo_root.join(SANDBOX_DIR).join(SANDBOX_BIN);
    let kernel_path = repo_root
        .join(ETHERLINK_DIR)
        .join("target/riscv64gc-unknown-linux-musl/profiling/etherlink");
    let output_path = data_dir.join("etherlink.profile");

    let kernel_path_str = kernel_path.to_string_lossy();
    let inbox_file_str = inbox_file.to_string_lossy();
    let output_path_str = output_path.to_string_lossy();
    let sample_interval_us_str = sample_interval_us.to_string();

    let cmd_args = vec![
        "run",
        "--input",
        kernel_path_str.as_ref(),
        "--inbox-file",
        inbox_file_str.as_ref(),
        "--address",
        DEFAULT_ROLLUP_ADDRESS,
        "--sample-interval-us",
        sample_interval_us_str.as_str(),
        "--output",
        output_path_str.as_ref(),
    ];

    println!("[INFO]: running {TX_COUNT} transfers (profiling)");
    cmd!(sh, "{sandbox_path} {cmd_args...}")
        .read()
        .context("Failed to run RISC-V sandbox")?;

    println!("[INFO]: Profile written to {output_path:?}");

    Ok(())
}

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

fn build_sandbox(sh: &Shell, repo_root: &Path) -> Result<()> {
    let sandbox_dir = repo_root.join(SANDBOX_DIR);

    cmd!(sh, "make -C {sandbox_dir} {SANDBOX_BIN}")
        .run()
        .context("Failed to build sandbox")
}

fn build_bench_tool(sh: &Shell, repo_root: &Path) -> Result<()> {
    let etherlink_dir = repo_root.join(ETHERLINK_DIR);

    cmd!(sh, "make -C {etherlink_dir} inbox-bench")
        .run()
        .context("Failed to build bench tool")
}

fn build_etherlink_kernel(
    sh: &Shell,
    repo_root: &Path,
    config: &BuildConfig,
    inbox_file: &Path,
) -> Result<()> {
    let etherlink_path = repo_root.join(ETHERLINK_DIR);

    if config.static_inbox {
        let inbox_file_str = inbox_file.to_string_lossy();
        sh.set_var("INBOX_FILE", inbox_file_str.as_ref());
    }

    let features = if config.tracing && config.static_inbox {
        Some("FEATURES=tracing,static-inbox")
    } else if config.static_inbox {
        Some("FEATURES=static-inbox")
    } else if config.tracing {
        Some("FEATURES=tracing")
    } else {
        None
    };

    let profile_opt = if config.profiling {
        Some("PROFILE=profiling")
    } else {
        None
    };

    let target = if config.native {
        "build-kernel-native"
    } else {
        "build-kernel"
    };

    cmd!(
        sh,
        "make -C {etherlink_path} {target} {features...} {profile_opt...}"
    )
    .run()
    .context("Failed to build Etherlink kernel")
}

fn get_native_target(sh: &Shell) -> Result<String> {
    let target = cmd!(sh, "rustc -vV")
        .read()
        .context("Failed to get rustc version info")?
        .lines()
        .find(|line| line.starts_with("host: "))
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_default()
        .trim()
        .to_string();

    if target.is_empty() {
        bail!("Empty native target");
    }

    Ok(target)
}

fn run(
    sh: &Shell,
    config: &BuildConfig,
    repo_root: &Path,
    inbox_file: &Path,
    data_dir: &Path,
) -> Result<()> {
    if config.tracing {
        println!("[INFO]: running {TX_COUNT} transfers (tracing)");
        run_etherlink(sh, config, repo_root, inbox_file, data_dir)?;
        println!(
            "[INFO]: Wrote trace in {}",
            data_dir.join("etherlink.trace").display()
        );
    } else if config.native {
        let native_target = get_native_target(sh)?;
        println!("[INFO]: running {TX_COUNT} transfers ({native_target})");
        run_etherlink(sh, config, repo_root, inbox_file, data_dir)?;
    } else {
        println!("[INFO]: running {TX_COUNT} transfers (riscv)");
        run_etherlink(sh, config, repo_root, inbox_file, data_dir)?;
    }

    Ok(())
}

fn run_etherlink(
    sh: &Shell,
    config: &BuildConfig,
    repo_root: &Path,
    inbox_file: &Path,
    data_dir: &Path,
) -> Result<()> {
    let (command_output, output_file) = if config.native {
        let native_target = get_native_target(sh)?;
        let etherlink_path = repo_root
            .join(ETHERLINK_DIR)
            .join("target")
            .join(native_target)
            .join("release/etherlink");

        let output = cmd!(sh, "{etherlink_path} --timings")
            .ignore_stderr()
            .read()
            .context("Failed to run native Etherlink")?;

        (output, data_dir.join("etherlink.log"))
    } else {
        let sandbox_path = repo_root.join(SANDBOX_DIR).join(SANDBOX_BIN);
        let kernel_path = repo_root
            .join(ETHERLINK_DIR)
            .join("target/riscv64gc-unknown-linux-musl/release/etherlink");

        let output_filename = if config.tracing {
            "etherlink.trace"
        } else {
            "etherlink.log"
        };

        let kernel_path_str = kernel_path.to_string_lossy();
        let inbox_file_str = inbox_file.to_string_lossy();
        let mut cmd_args = vec![
            "run",
            "--input",
            kernel_path_str.as_ref(),
            "--inbox-file",
            inbox_file_str.as_ref(),
            "--address",
            DEFAULT_ROLLUP_ADDRESS,
        ];

        if !config.tracing {
            cmd_args.push("--timings");
        }

        let output = cmd!(sh, "{sandbox_path} {cmd_args...}")
            .read()
            .context("Failed to run RISC-V sandbox")?;

        (output, data_dir.join(output_filename))
    };

    std::fs::write(&output_file, command_output).context("Failed to write output file")?;

    Ok(())
}

fn print_bench_results(
    sh: &Shell,
    repo_root: &Path,
    inbox_file: &Path,
    data_dir: &Path,
) -> Result<()> {
    let log_file = data_dir.join("etherlink.log");
    let inbox_bench_path = repo_root.join(ETHERLINK_DIR).join("inbox-bench");
    let txs = TX_COUNT.to_string();

    // Print bold formatting
    println!("\x1b[1m");

    cmd!(sh, "{inbox_bench_path} results --inbox-file {inbox_file} --log-file {log_file} --expected-transfers {txs}")
        .run()
        .context("Failed to run inbox-bench results")?;

    // Reset formatting
    println!("\x1b[0m");

    Ok(())
}

fn init_data_dir(config: &BuildConfig) -> Result<PathBuf> {
    match &config.data_dir {
        Some(dir) => Ok(dir.clone()),
        None => {
            let temp_dir = tempfile::tempdir()?;
            Ok(temp_dir.keep())
        }
    }
}
