// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Long-running property-based test for [`Database`].
//!
//! Each epoch advances and commits a shared base state, then runs a `proptest`
//! test on that base. Every case applies a model-guided operation
//! sequence to the reference model and three databases (in-memory traced,
//! persistence traced, and production), cross-checking traces, root hashes, and
//! proofs.
//!
//! On failure the committed base of the failing epoch and the shrunk operation
//! sequence are written to `<out-dir>/failure/` so the failure can be replayed.
//!
//! [`Database`]: crate::database::Database

pub mod model;
pub mod run_case;
pub mod strategy;

use std::collections::VecDeque;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use octez_riscv_data::hash::Hash;
use proptest::strategy::Strategy;
use proptest::strategy::ValueTree;
use proptest::test_runner::Config as ProptestConfig;
use proptest::test_runner::RngAlgorithm;
use proptest::test_runner::TestError;
use proptest::test_runner::TestRng;
use proptest::test_runner::TestRunner;

use self::model::LongTestModel;
use self::run_case::Base;
use self::run_case::advance_base;
use self::run_case::initial_base;
use self::run_case::run_case;
use self::strategy::long_test_ops_strategy;
use crate::commit::CommitId;
use crate::persistence_layer::PersistenceLayer;
use crate::repo::DirectoryManager;
use crate::storage::PersistentKeyValueStore;
use crate::storage::in_memory::InMemoryKeyValueStore;
use crate::storage::in_memory::InMemoryRepo;
use crate::test_helpers::DatabaseOperation;

const IN_MEMORY_BASE: &str = "in-memory-base";
const PERSISTENT_BASE: &str = "persistent-base";
const META_FILE: &str = "meta.json";
const REGRESSION_FILE: &str = "regression.json";
const BASE_MODEL_FILE: &str = "base-model.json";

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

/// Metadata persisted alongside a failure which enables replaying it.
#[derive(serde::Serialize, serde::Deserialize)]
struct FailureMeta {
    /// Seed of the run.
    seed: Hash,
    /// Index of the failing epoch.
    epoch: u64,
    /// Operations sampled per epoch.
    ops_per_epoch: usize,
    /// Test cases per epoch.
    cases_per_epoch: u32,
    /// The commit identifying the failing epoch's starting state.
    base_commit: CommitId,
    /// Short description of the failure.
    reason: String,
    /// Git revision, if available from the environment.
    git_sha: String,
}

/// Run the long-running test
pub fn run_long_test(config: LongTestConfig) -> Result<()> {
    let seed = config
        .seed
        .unwrap_or_else(|| rand::random::<[u8; 32]>().into());
    let max_epochs = config.epochs;
    let ops_per_epoch = config.ops_per_epoch;
    let cases_per_epoch = config.cases_per_epoch;
    let keep_epochs = config.keep_epochs;
    let out_dir = match config.out_dir {
        Some(dir) => {
            fs::create_dir_all(&dir)
                .with_context(|| format!("creating directory {}", dir.display()))?;
            dir
        }
        None => tempfile::Builder::new()
            .prefix("database_long_test-")
            .tempdir()?
            .keep(),
    };

    let mut rerun = format!(
        "cargo run --release --features rocksdb,unstable-test-utils --bin database_long_test -- \
         test --seed {seed} --ops-per-epoch {ops_per_epoch} --cases-per-epoch {cases_per_epoch}"
    );
    if let Some(epochs) = max_epochs {
        rerun.push_str(&format!(" --epochs {epochs}"));
    }
    if let Some(keep_epochs) = keep_epochs {
        rerun.push_str(&format!(" --keep-epochs {keep_epochs}"));
    }
    if let Some(budget) = config.time_budget {
        rerun.push_str(&format!(" --max-minutes {}", budget.as_secs() / 60));
    }
    eprintln!(
        "test directory: {} | ops/epoch: {ops_per_epoch} | cases/epoch: {cases_per_epoch}\n\
         rerun with:\n{rerun}",
        out_dir.display(),
    );

    let repo_dir = out_dir.join("repo");
    fs::create_dir_all(&repo_dir)
        .with_context(|| format!("creating repo dir {}", repo_dir.display()))?;
    let persistent_repo =
        DirectoryManager::new(&repo_dir).context("creating the directory manager")?;
    let in_memory_repo = InMemoryRepo::default();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .context("building the tokio runtime")?;
    let handle = runtime.handle();

    let mut base = initial_base(handle, &in_memory_repo, &persistent_repo);
    let mut epoch = 0u64;

    // Base commits of the most recent epochs, oldest first; the newest entry is
    // the current base.
    let mut recent_commits = VecDeque::from([base.commit]);

    let start = Instant::now();
    loop {
        if let Some(max) = max_epochs {
            if epoch >= max {
                break;
            }
        }
        if let Some(budget) = config.time_budget {
            if start.elapsed() >= budget {
                eprintln!("time budget reached after {epoch} epochs");
                break;
            }
        }

        let mut runner = epoch_runner(seed, epoch, cases_per_epoch);

        // Advance and commit the base by a generated sequence (no proofs).
        let advance_ops = long_test_ops_strategy(&base.model.pools(), ops_per_epoch)
            .new_tree(&mut runner)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("drawing the epoch advance sequence")?
            .current();
        base = advance_base(
            handle,
            &in_memory_repo,
            &persistent_repo,
            &base,
            &advance_ops,
        );
        recent_commits.push_back(base.commit);

        // Run the property test on this base.
        let strategy = long_test_ops_strategy(&base.model.pools(), ops_per_epoch);
        let result = runner.run(&strategy, |ops| {
            run_case(handle, &in_memory_repo, &persistent_repo, &base, &ops);
            Ok(())
        });

        match result {
            Ok(()) => {
                // Size reporting only via the binary, not the crate test.
                #[cfg(not(test))]
                {
                    let snapshot_dir = persistent_repo.database_commit_dir(&base.commit);
                    let snapshot_size = dir_size(&snapshot_dir)
                        .context("measuring the size of the latest snapshot")?;
                    eprintln!(
                        "epoch {epoch} ok (db contains {} entries, latest snapshot: {:.2} MiB)",
                        base.model.data.len(),
                        snapshot_size as f64 / (1024.0 * 1024.0),
                    );
                }
                #[cfg(test)]
                eprintln!(
                    "epoch {epoch} ok (db contains {} entries)",
                    base.model.data.len()
                );

                // Clear snapshots older than the retention window. Only the
                // current base is needed for the next epoch and for failure replay.
                if let Some(keep_epochs) = keep_epochs {
                    while recent_commits.len() > keep_epochs.get() {
                        let old = recent_commits.pop_front().expect("non-empty");
                        // Content-addressed commits can repeat; keep the
                        // snapshot if a retained epoch still references it.
                        if recent_commits.contains(&old) {
                            continue;
                        }
                        let old_dir = persistent_repo.database_commit_dir(&old);
                        if old_dir.exists() {
                            fs::remove_dir_all(&old_dir).with_context(|| {
                                format!("removing disk snapshot {}", old_dir.display())
                            })?;
                        }
                        in_memory_repo
                            .remove_commit(&old)
                            .context("removing in-memory snapshot {old}")?;
                    }
                }
            }
            Err(TestError::Fail(reason, ops)) => {
                let meta = FailureMeta {
                    seed,
                    epoch,
                    ops_per_epoch,
                    cases_per_epoch,
                    base_commit: base.commit,
                    reason: reason.to_string(),
                    git_sha: std::env::var("GITHUB_SHA").unwrap_or_else(|_| "unknown".to_string()),
                };
                write_failure(
                    &out_dir,
                    &persistent_repo,
                    &in_memory_repo,
                    &meta,
                    &base.model,
                    &ops,
                )?;
                bail!(
                    "epoch {epoch} failed: {reason}. Artifacts written to {}",
                    out_dir.join("failure").display()
                );
            }
            Err(TestError::Abort(reason)) => {
                bail!("epoch {epoch} aborted: {reason}");
            }
        }

        epoch += 1;
    }

    eprintln!("completed {epoch} epochs");

    // Size reporting only via the binary, not the crate test.
    #[cfg(not(test))]
    {
        drop(runtime);

        let repo_size = dir_size(&repo_dir).context("measuring the size of the repo")?;
        eprintln!(
            "total repo size: {:.2} MiB",
            repo_size as f64 / (1024.0 * 1024.0)
        );
    }

    Ok(())
}

/// Total size in bytes of all files under `dir`, recursively.
#[cfg(not(test))]
fn dir_size(dir: &Path) -> std::io::Result<u64> {
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

/// Build a deterministically seeded test runner for `epoch`.
fn epoch_runner(seed: Hash, epoch: u64, cases: u32) -> TestRunner {
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

/// Write the failure artifacts: metadata, the shrunk operation sequence, the
/// reference model, and the committed base on both backends.
fn write_failure(
    out_dir: &Path,
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    meta: &FailureMeta,
    model: &LongTestModel,
    ops: &[DatabaseOperation],
) -> Result<()> {
    fn write_failure_file(
        failure_dir: &Path,
        content: &impl serde::ser::Serialize,
        name: &str,
    ) -> Result<()> {
        let file = fs::File::create(failure_dir.join(name))
            .context(format!("creating failure file {name}"))?;
        serde_json::to_writer_pretty(file, content)
            .context(format!("writing failure to {name}"))?;
        Ok(())
    }
    let failure_dir = out_dir.join("failure");
    if failure_dir.exists() {
        fs::remove_dir_all(&failure_dir)
            .with_context(|| format!("clearing {}", failure_dir.display()))?;
    }
    fs::create_dir_all(&failure_dir)
        .with_context(|| format!("creating {}", failure_dir.display()))?;

    write_failure_file(&failure_dir, meta, META_FILE)?;
    write_failure_file(&failure_dir, &ops, REGRESSION_FILE)?;
    write_failure_file(&failure_dir, model, BASE_MODEL_FILE)?;

    let in_memory_store = InMemoryKeyValueStore::checkout(in_memory_repo, &meta.base_commit)
        .context("checking out the in-memory base")?;
    in_memory_store
        .commit_to_path(&failure_dir.join(IN_MEMORY_BASE))
        .context("writing the in-memory base snapshot")?;

    let persistent_store = PersistenceLayer::checkout(persistent_repo, &meta.base_commit)
        .context("checking out the persistent base")?;
    persistent_store
        .commit_to_path(&failure_dir.join(PERSISTENT_BASE))
        .context("writing the persistent base snapshot")?;

    eprintln!(
        "failure artifacts written to {failure}\n\
         replay with:\n\
         cargo run --release \
         --features rocksdb,unstable-test-utils --bin database_long_test -- replay {failure}",
        failure = failure_dir.display(),
    );
    Ok(())
}

/// Reproduce a recorded failure by reconstructing only the failing epoch.
/// Both the persistence backend's base and the in-memory backend's base
/// are restored from disk, and the saved (shrunk) operation sequence is applied once.
pub fn replay_failure(dir: &Path) -> Result<()> {
    fn read_failure_file<T: serde::de::DeserializeOwned>(
        failure_dir: &Path,
        name: &str,
    ) -> Result<T> {
        let file = fs::File::open(failure_dir.join(name))
            .context(format!("opening failure file {name}"))?;
        serde_json::from_reader(file).context(format!("decoding {name}"))
    }

    let meta: FailureMeta = read_failure_file(dir, META_FILE)?;
    eprintln!(
        "Replaying epoch {} from {} (prior epochs are not re-run)",
        meta.epoch,
        dir.display(),
    );

    let out_dir = dir.join("replay-run");
    let repo_dir = out_dir.join("repo");
    fs::create_dir_all(&repo_dir)
        .with_context(|| format!("creating repo dir {}", repo_dir.display()))?;
    let persistent_repo =
        DirectoryManager::new(&repo_dir).context("creating the directory manager")?;
    let in_memory_repo = InMemoryRepo::default();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .context("building the tokio runtime")?;
    let handle = runtime.handle();

    // Restore the persistence backend's base from the saved commit. The saved
    // snapshot is opened read-only and re-committed into the fresh repo.
    let working_dir = persistent_repo
        .temp_database_dir()
        .context("creating a scratch directory")?;
    let persistent_store =
        PersistenceLayer::checkout_from_path(&dir.join(PERSISTENT_BASE), working_dir)
            .context("loading the persistent base snapshot")?;
    persistent_store
        .commit(&persistent_repo, &meta.base_commit)
        .context("registering the persistent base")?;

    // Restore the in-memory backend's base from its saved snapshot.
    let working_dir = persistent_repo
        .temp_database_dir()
        .context("creating a scratch directory")?;
    let in_memory_store =
        InMemoryKeyValueStore::checkout_from_path(&dir.join(IN_MEMORY_BASE), working_dir)
            .context("loading the in-memory base snapshot")?;
    in_memory_store
        .commit(&in_memory_repo, &meta.base_commit)
        .context("registering the in-memory base")?;

    // The reference model carries the expected key/value state for assertions.
    let base = Base {
        commit: meta.base_commit,
        model: read_failure_file(dir, BASE_MODEL_FILE)?,
    };

    // Apply the recorded operation sequence once and catch the resulting panic
    let ops: Vec<DatabaseOperation> = read_failure_file(dir, REGRESSION_FILE)?;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_case(handle, &in_memory_repo, &persistent_repo, &base, &ops);
    }));

    match outcome {
        Err(payload) => {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            bail!(
                "replay reproduced the failure for epoch {}: {reason}",
                meta.epoch
            );
        }
        Ok(()) => {
            eprintln!(
                "replay did NOT reproduce the failure for epoch {}",
                meta.epoch
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bytes::Bytes;
    use octez_riscv_test_utils::TestableTmpdir;
    use tokio::runtime::Runtime;

    use super::*;
    use crate::key::Key;

    // A short run of the long test
    #[test]
    fn database_long_test_restricted() {
        run_long_test(LongTestConfig {
            epochs: Some(3),
            ops_per_epoch: 200,
            cases_per_epoch: 32,
            seed: None,
            time_budget: None,
            keep_epochs: Some(NonZeroUsize::new(2).expect("non-zero")),
            out_dir: None,
        })
        .expect("the short long test run should succeed");
    }

    // Tests for the failure replay mechanism of the long test

    struct TestSetup {
        _tmp: TestableTmpdir,
        runtime: Runtime,
        out_dir: PathBuf,
        persistent_repo: DirectoryManager,
        in_memory_repo: InMemoryRepo,
        base: Base,
        key: Key,
    }

    /// Build a base committed on all backends containing a single key.
    fn build_base_with_key() -> TestSetup {
        let tmp = TestableTmpdir::new();
        let out_dir = tmp.path().to_owned();
        let repo_dir = out_dir.join("repo");
        fs::create_dir_all(&repo_dir).expect("creating the repo dir should succeed");
        let persistent_repo = DirectoryManager::new(&repo_dir)
            .expect("creating the directory manager should succeed");
        let in_memory_repo = InMemoryRepo::default();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .expect("building the tokio runtime should succeed");

        let key = Key::new(&[1, 2, 3]).expect("the key should be valid");
        let set = DatabaseOperation::Set(key.clone(), Bytes::from_static(b"value"));

        let base = initial_base(runtime.handle(), &in_memory_repo, &persistent_repo);
        let base = advance_base(
            runtime.handle(),
            &in_memory_repo,
            &persistent_repo,
            &base,
            &[set],
        );

        TestSetup {
            _tmp: tmp,
            runtime,
            out_dir,
            persistent_repo,
            in_memory_repo,
            base,
            key,
        }
    }

    /// Build a [`FailureMeta`] for `base_commit`; the non-essential fields are
    /// placeholders (replay only reads the commit and epoch index).
    fn dummy_meta(base_commit: CommitId) -> FailureMeta {
        FailureMeta {
            seed: Hash::from([0u8; 32]),
            epoch: 0,
            ops_per_epoch: 1,
            cases_per_epoch: 1,
            base_commit,
            reason: "test".to_string(),
            git_sha: "test".to_string(),
        }
    }

    // Replay of an artifact whose model disagrees with the restored database
    // reproduces the failure
    #[test]
    fn internal_test_replay_reproduces_recorded_failure() {
        let setup = build_base_with_key();
        let meta = dummy_meta(setup.base.commit);

        // An empty model disagrees with the restored base (which holds `key`):
        // checking the key's existence will mismatch and panic in `run_case`.
        let model = LongTestModel::default();
        let ops = vec![DatabaseOperation::Exists(setup.key.clone())];
        write_failure(
            &setup.out_dir,
            &setup.persistent_repo,
            &setup.in_memory_repo,
            &meta,
            &model,
            &ops,
        )
        .expect("writing the failure artifact should succeed");

        let err = replay_failure(&setup.out_dir.join("failure"))
            .expect_err("replay should reproduce the failure");
        assert!(
            err.to_string().contains("reproduced"),
            "unexpected replay error: {err}"
        );

        drop(setup.runtime);
    }

    // Replay of an artifact with the correct model must restore the base
    // faithfully on every backend. Also checks that replaying a dummy failure
    // doesn't work
    #[test]
    fn internal_test_replay_passes_for_a_consistent_base() {
        let setup = build_base_with_key();
        let meta = dummy_meta(setup.base.commit);

        // The recorded model matches the restored base, so a non-mutating
        // sequence (which also hashes, cross-checking all backends) must pass.
        let ops = vec![
            DatabaseOperation::Exists(setup.key.clone()),
            DatabaseOperation::Hash,
        ];
        write_failure(
            &setup.out_dir,
            &setup.persistent_repo,
            &setup.in_memory_repo,
            &meta,
            &setup.base.model,
            &ops,
        )
        .expect("writing the failure artifact should succeed");

        // The failure artifact must be self-contained: replay must succeed even
        // once the original repo is gone.
        fs::remove_dir_all(setup.out_dir.join("repo"))
            .expect("removing the original repo should succeed");

        replay_failure(&setup.out_dir.join("failure"))
            .expect("replay of a consistent base should not reproduce a failure");

        drop(setup.runtime);
    }
}
