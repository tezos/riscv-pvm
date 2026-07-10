// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Building blocks shared by both the `Database` and `Registry` long tests:
//! configuration, per-epoch seeding, failure metadata, and
//! small filesystem helpers.

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use octez_riscv_data::hash::Hash;
use proptest::test_runner::Config as ProptestConfig;
use proptest::test_runner::RngAlgorithm;
use proptest::test_runner::TestRng;
use proptest::test_runner::TestRunner;

use crate::commit::CommitId;

pub(crate) const META_FILE: &str = "meta.json";
pub(crate) const REGRESSION_FILE: &str = "regression.json";
pub(crate) const BASE_MODEL_FILE: &str = "base-model.json";

/// Configuration for a long-running test invocation.
pub struct LongTestConfig {
    /// Maximum number of epochs to run. `None` runs until the time budget.
    pub epochs: Option<u64>,
    /// Maximum number of operations sampled per epoch.
    pub ops_per_epoch: usize,
    /// Number of test cases per epoch.
    pub cases_per_epoch: u32,
    /// Optional seed.
    pub seed: Option<Hash>,
    /// Time budget. The loop stops cleanly once exceeded.
    pub time_budget: Option<Duration>,
    /// Number of most recent epoch snapshots to keep. `None` keeps everything.
    pub keep_epochs: Option<NonZeroUsize>,
    /// Directory for run state and failure artifacts. `None` uses a tempdir.
    pub out_dir: Option<PathBuf>,
}

impl LongTestConfig {
    pub fn timed_epoch_loop(&self, mut do_epoch: impl FnMut(u64) -> Result<()>) -> Result<()> {
        let mut epoch = 0u64;
        let start = Instant::now();

        loop {
            if let Some(max) = self.epochs {
                if epoch >= max {
                    break;
                }
            }
            if let Some(budget) = self.time_budget {
                if start.elapsed() >= budget {
                    eprintln!("time budget reached after {epoch} epochs")
                }
            }

            do_epoch(epoch)?;

            epoch += 1;
        }

        eprintln!("complete {epoch} epochs");

        Ok(())
    }
}

/// The committed starting state shared by every case in a test epoch.
#[derive(Clone)]
pub struct Base<M> {
    /// The commit identifying the starting state
    pub commit: CommitId,
    /// The reference model corresponding to `commit`.
    pub model: M,
}

/// Metadata persisted alongside a failure which enables replaying it.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct FailureMeta {
    /// Seed of the run.
    pub seed: Hash,
    /// Index of the failing epoch.
    pub epoch: u64,
    /// Operations sampled per epoch.
    pub ops_per_epoch: usize,
    /// Test cases per epoch.
    pub cases_per_epoch: u32,
    /// The commit identifying the failing epoch's starting state.
    pub base_commit: CommitId,
    /// Short description of the failure.
    pub reason: String,
    /// Git revision, if available from the environment.
    pub git_sha: String,
}

/// Build a deterministically seeded test runner for `epoch`.
pub(crate) fn epoch_runner(seed: Hash, epoch: u64, cases: u32) -> TestRunner {
    // XOR the epoch index into the seed so each epoch has a distinct yet
    // reproducible seed
    let mut seed: [u8; 32] = seed.into();
    let head = u64::from_le_bytes(seed[..8].try_into().expect("8 bytes")) ^ epoch;
    seed[..8].copy_from_slice(&head.to_le_bytes());

    let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &seed);
    let config = ProptestConfig {
        cases,
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    TestRunner::new_with_rng(config, rng)
}

/// Serialise `content` as pretty JSON into `<failure_dir>/<name>`.
pub(crate) fn write_failure_file(
    failure_dir: &Path,
    content: &impl serde::ser::Serialize,
    name: &str,
) -> Result<()> {
    let file = fs::File::create(failure_dir.join(name))
        .context(format!("creating failure file {name}"))?;
    serde_json::to_writer_pretty(file, content).context(format!("writing failure to {name}"))?;
    Ok(())
}

/// Deserialise `<failure_dir>/<name>` from JSON.
pub(crate) fn read_failure_file<T: serde::de::DeserializeOwned>(
    failure_dir: &Path,
    name: &str,
) -> Result<T> {
    let file =
        fs::File::open(failure_dir.join(name)).context(format!("opening failure file {name}"))?;
    serde_json::from_reader(file).context(format!("decoding {name}"))
}

/// Total size in bytes of all files under `dir`, recursively.
#[cfg(not(test))]
pub(crate) fn dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut size = 0;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            size += dir_size(&entry.path())?;
        } else {
            size += metadata.len();
        }
    }
    Ok(size)
}
