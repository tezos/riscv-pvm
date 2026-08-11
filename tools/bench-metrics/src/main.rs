// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Collects durable-storage benchmark results and reports them.
//!
//! The figures come from the `gc_space` harness: how much of the storage is garbage, how much
//! retaining a commit costs, and how much of a commit's bytes are shared with the previous one.
//! They are read from the harness's `--json-out` samples.
//!
//! The metrics are printed, one per line by default and as a table with `--markdown`, which is what
//! a pull request comment carries.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use kernel_bench_utils::datadog::Gauge;
use serde::Deserialize;

mod markdown;

/// Prefix for every metric this tool reports.
const PREFIX: &str = "ci.riscv.durable_storage";

#[derive(Debug, Parser)]
#[command(version, long_about = None)]
struct Cli {
    /// Print the metrics as a markdown table, for a pull request comment.
    ///
    /// Without it they are printed one per line, which is what you want in a job log: the line
    /// holds the metric's full name, raw value and tags.
    #[arg(long)]
    markdown: bool,

    /// Tag recording the shape the run measured, so one metric can be split by scale.
    ///
    /// Keep the set of values small: it is meant to tell a handful of scenarios apart, not to carry
    /// a run's parameters.
    #[arg(long, default_value = "default")]
    shape: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Collect space metrics from the samples written by `gc_space --json-out`.
    Space {
        /// Path to the JSON-lines samples file.
        samples: PathBuf,
    },
}

/// The parts of a `gc_space` sample this tool reads.
#[derive(Debug, Deserialize)]
struct Sample {
    commit: usize,
    blob: Blob,
    disk: Disk,
    sharing: Option<Sharing>,
}

#[derive(Debug, Deserialize)]
struct Blob {
    stored_bytes: u64,
    live_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Disk {
    unique_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Sharing {
    carried_bytes: u64,
    new_bytes: u64,

    /// Files the earlier commit pinned that this one does not, so non-zero only after a compaction.
    dropped_files: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let gauges = match &cli.command {
        Command::Space { samples } => space_gauges(samples, &cli.shape)?,
    };

    if cli.markdown {
        println!("{}", markdown::table(&gauges));
    } else {
        for gauge in &gauges {
            println!(
                "{} = {:.3} {} [{}]",
                gauge.name,
                gauge.value,
                gauge.unit,
                gauge.tags.join(" ")
            );
        }
    }

    if !cli.markdown {
        println!("({} metric(s) collected)", gauges.len());
    }

    Ok(())
}

/// Derive the space metrics from a run's samples.
///
/// Rates are taken across the whole run rather than from the last sample, because a single commit's
/// figure swings with whether a compaction happened to land on it.
fn space_gauges(path: &Path, shape: &str) -> Result<Vec<Gauge>> {
    /// Fraction of the bytes these commits' checkpoints pinned that were already on disk.
    ///
    /// `None` when the set is empty, so a run with nothing to say submits nothing rather than a zero
    /// that would read as "shared nothing".
    fn pooled_sharing<'a>(sharing: impl Iterator<Item = &'a Sharing>) -> Option<f64> {
        let mut carried = 0u64;
        let mut new = 0u64;
        let mut seen = false;

        for one in sharing {
            carried += one.carried_bytes;
            new += one.new_bytes;
            seen = true;
        }

        if !seen {
            return None;
        }

        let total = carried + new;

        Some(if total == 0 {
            0.0
        } else {
            carried as f64 / total as f64
        })
    }

    let contents =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let samples: Vec<Sample> = contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parsing a sample"))
        .collect::<Result<_>>()?;

    let (Some(first), Some(last)) = (samples.first(), samples.last()) else {
        anyhow::bail!("{} holds no samples", path.display());
    };

    let commits = last.commit.saturating_sub(first.commit);

    anyhow::ensure!(commits > 0, "the samples cover no commits");

    let tags = vec![format!("shape:{shape}")];
    let dead_bytes = last.blob.stored_bytes.saturating_sub(last.blob.live_bytes);
    let first_dead = first
        .blob
        .stored_bytes
        .saturating_sub(first.blob.live_bytes);

    // Sharing is the assumption retention rests on, and a fall in it means retaining history has
    // become more expensive. Two figures, because one cannot carry both facts: how much sharing
    // there is to lose, and how much of it a compaction loses.
    //
    // Both sum the bytes and take one ratio rather than averaging per-commit ratios, so a commit
    // counts for what it actually pinned and the two are the same statistic — which is what makes
    // their difference the cost of the compaction commits.
    let shared: Vec<&Sharing> = samples.iter().filter_map(|s| s.sharing.as_ref()).collect();
    let overall = pooled_sharing(shared.iter().copied());
    let steady = pooled_sharing(shared.iter().copied().filter(|s| s.dropped_files == 0));

    let mut gauges = vec![
        Gauge {
            name: format!("{PREFIX}.dead_node_bytes_per_commit"),
            value: dead_bytes.saturating_sub(first_dead) as f64 / commits as f64,
            tags: tags.clone(),
            unit: "bytes".to_owned(),
        },
        Gauge {
            name: format!("{PREFIX}.dead_node_fraction"),
            value: if last.blob.stored_bytes == 0 {
                0.0
            } else {
                dead_bytes as f64 / last.blob.stored_bytes as f64 * 100.0
            },
            tags: tags.clone(),
            unit: "percent".to_owned(),
        },
        Gauge {
            name: format!("{PREFIX}.repo_bytes_per_commit"),
            value: last
                .disk
                .unique_bytes
                .saturating_sub(first.disk.unique_bytes) as f64
                / commits as f64,
            tags: tags.clone(),
            unit: "bytes".to_owned(),
        },
    ];

    if let Some(overall) = overall {
        gauges.push(Gauge {
            name: format!("{PREFIX}.checkpoint_shared_fraction"),
            value: overall * 100.0,
            tags: tags.clone(),
            unit: "percent".to_owned(),
        });
    }

    // Absent when every sampled commit followed a compaction, which would make this the same
    // series as the one above rather than something to compare it against.
    if let Some(steady) = steady {
        gauges.push(Gauge {
            name: format!("{PREFIX}.checkpoint_shared_fraction_steady"),
            value: steady * 100.0,
            tags,
            unit: "percent".to_owned(),
        });
    }

    Ok(gauges)
}
