// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Long-running property-based test for [`Registry`].
//!
//! The registry starts with `permanent` databases which
//! may never be moved, cleared, or copied into. Each epoch advances and commits
//! a shared base, then runs a `proptest` test on that base. Every case applies a
//! model-guided operation sequence to the reference model and two registries (an
//! in-memory-backed and a persistence-backed one), cross-checking operation
//! results and, at the end of each case, the registry root hashes.
//!
//! On failure the shrunk operation sequence, the reference model, metadata, and
//! a self-contained per-database snapshot of the base on both backends are
//! written to `<out-dir>/failure/`. Replay restores the base into fresh
//! repositories and re-applies the sequence once.
//!
//! [`Registry`]: crate::registry::Registry

mod gc;
mod model;
mod run_case;
mod strategy;

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use octez_riscv_data::mode::Normal;
use proptest::strategy::Strategy;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestError;

use self::model::RegistryLongTestModel;
use self::run_case::advance_base;
use self::run_case::initial_base;
use self::run_case::run_case;
use self::strategy::ops_strategy;
use super::harness::BASE_MODEL_FILE;
use super::harness::Base;
use super::harness::FailureMeta;
use super::harness::LongTestConfig;
use super::harness::META_FILE;
use super::harness::REGRESSION_FILE;
use super::harness::epoch_runner;
use super::harness::read_failure_file;
use super::harness::write_failure_file;
use crate::commit::CommitId;
use crate::persistence_layer::PersistenceLayer;
use crate::registry::Registry;
use crate::repo::DirectoryManager;
use crate::repo::RegistryRepo;
use crate::storage::PersistentKeyValueStore;
use crate::storage::in_memory::InMemoryKeyValueStore;
use crate::storage::in_memory::InMemoryRepo;
use crate::test_helpers::registry::RegistryOperation;

/// File holding the raw registry manifest bytes of the failing base.
const MANIFEST_FILE: &str = "manifest.bin";
/// Subdirectory holding the persistence-backed per-database base snapshots.
const PERSISTENT_DBS: &str = "persistent-databases";
/// Subdirectory holding the in-memory per-database base snapshots.
const IN_MEMORY_DBS: &str = "in-memory-databases";

/// Run the long-running registry test starting with a registry of `permanent` databases.
///
/// With `keep_stable_size`, Grow/Shrink are sampled so the registry size
/// tends to `2 * permanent` rather than monotonically growing over the run.
pub fn run_long_test(
    config: LongTestConfig,
    permanent: usize,
    keep_stable_size: bool,
) -> Result<()> {
    let seed = config
        .seed
        .unwrap_or_else(|| rand::random::<[u8; 32]>().into());
    let max_epochs = config.epochs;
    let ops_per_epoch = config.ops_per_epoch;
    let cases_per_epoch = config.cases_per_epoch;
    let keep_epochs = config.keep_epochs;
    let fail_on_warning = config.fail_on_warning;
    let out_dir = match config.out_dir.clone() {
        Some(dir) => {
            fs::create_dir_all(&dir)
                .with_context(|| format!("creating directory {}", dir.display()))?;
            dir
        }
        None => tempfile::Builder::new()
            .prefix("registry_long_test-")
            .tempdir()?
            .keep(),
    };

    let mut rerun = format!(
        "cargo run --release --features rocksdb,unstable-test-utils --bin long_test -- \
         registry test --seed {seed} --ops-per-epoch {ops_per_epoch} \
         --cases-per-epoch {cases_per_epoch} --permanent-databases {permanent}"
    );
    if let Some(epochs) = max_epochs {
        rerun.push_str(&format!(" --epochs {epochs}"));
    }
    if let Some(budget) = config.time_budget {
        rerun.push_str(&format!(" --max-minutes {}", budget.as_secs() / 60));
    }
    if let Some(keep_epochs) = keep_epochs {
        rerun.push_str(&format!(" --keep-epochs {keep_epochs}"));
    }
    if keep_stable_size {
        rerun.push_str(" --keep-stable-size");
    }
    if config.fail_on_warning {
        rerun.push_str(" --fail-on-warning");
    }
    eprintln!(
        "test directory: {} | ops/epoch: {ops_per_epoch} | cases/epoch: {cases_per_epoch} | \
         permanent: {permanent}\nrerun with:\n{rerun}",
        out_dir.display(),
    );

    let repo_dir = out_dir.join("repo");
    fs::create_dir_all(&repo_dir)
        .with_context(|| format!("creating repo dir {}", repo_dir.display()))?;
    let persistent_repo =
        DirectoryManager::new(&repo_dir).context("creating the directory manager")?;
    let in_memory_repo = InMemoryRepo::default();

    let mut base = initial_base(&in_memory_repo, &persistent_repo, permanent);
    let mut recent_commits = VecDeque::from([base.commit]);

    config.timed_epoch_loop(|epoch| {
        let mut runner = epoch_runner(seed, epoch, cases_per_epoch);

        // Advance and commit the base by a generated sequence (no proofs).
        let advance_ops = ops_strategy(
            &base.model.pools(),
            permanent,
            keep_stable_size,
            ops_per_epoch,
        )
        .new_tree(&mut runner)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("drawing the epoch advance sequence")?
        .current();
        base = advance_base(&in_memory_repo, &persistent_repo, &base, &advance_ops);
        recent_commits.push_back(base.commit);

        // Run the property test on this base.
        let strategy = ops_strategy(
            &base.model.pools(),
            permanent,
            keep_stable_size,
            ops_per_epoch,
        );
        let result = runner.run(&strategy, |ops| {
            run_case(
                &in_memory_repo,
                &persistent_repo,
                &base,
                &ops,
                fail_on_warning,
            );
            Ok(())
        });

        match result {
            Ok(()) => {
                // Size reporting only via the binary, not the crate test.
                #[cfg(not(test))]
                {
                    let repo_size = super::harness::dir_size(&repo_dir)
                        .context("measuring the size of the repo")?;
                    eprintln!(
                        "epoch {epoch} ok ({} databases, {} entries, repo: {:.2} MiB)",
                        base.model.len(),
                        base.model.total_entries(),
                        repo_size as f64 / (1024.0 * 1024.0),
                    );
                }
                #[cfg(test)]
                eprintln!(
                    "epoch {epoch} ok ({} databases, {} entries)",
                    base.model.len(),
                    base.model.total_entries()
                );

                // Garbage-collect snapshots older than the retention window.
                if let Some(keep_epochs) = keep_epochs {
                    gc::prune(
                        &persistent_repo,
                        &in_memory_repo,
                        &mut recent_commits,
                        keep_epochs,
                    )?;
                }

                Ok(())
            }
            Err(TestError::Fail(reason, ops)) => {
                let meta = FailureMeta {
                    seed,
                    epoch,
                    ops_per_epoch,
                    cases_per_epoch,
                    fail_on_warning,
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
    })
}

/// Write the failure artifacts: metadata, the shrunk operation sequence, the
/// reference model, and a self-contained snapshot of the failing base on both
/// backends.
fn write_failure(
    out_dir: &Path,
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    meta: &FailureMeta,
    model: &RegistryLongTestModel,
    ops: &[RegistryOperation],
) -> Result<()> {
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

    save_base(
        &failure_dir,
        persistent_repo,
        in_memory_repo,
        &meta.base_commit,
    )?;

    eprintln!(
        "failure artifacts written to {failure}\n\
         replay with:\n\
         cargo run --release \
         --features rocksdb,unstable-test-utils --bin long_test -- registry replay {failure}",
        failure = failure_dir.display(),
    );
    Ok(())
}

/// Save a self-contained snapshot of the base committed at `base_commit`: the
/// registry manifest, plus a per-database snapshot of every referenced database
/// commit on each backend.
fn save_base(
    failure_dir: &Path,
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    base_commit: &CommitId,
) -> Result<()> {
    let manifest = persistent_repo
        .read_registry_commit(base_commit)
        .context("reading the registry manifest")?;
    fs::write(failure_dir.join(MANIFEST_FILE), &manifest)
        .context("writing the registry manifest")?;

    let db_commits =
        Registry::<PersistenceLayer, Normal>::database_commits(persistent_repo, base_commit)
            .context("reading the base's database commits")?;

    let persistent_dir = failure_dir.join(PERSISTENT_DBS);
    let in_memory_dir = failure_dir.join(IN_MEMORY_DBS);
    fs::create_dir_all(&persistent_dir).context("creating the persistent snapshot dir")?;
    fs::create_dir_all(&in_memory_dir).context("creating the in-memory snapshot dir")?;

    for db_commit in db_commits {
        let hex = db_commit.hex_encode();
        PersistenceLayer::checkout(persistent_repo, &db_commit)
            .context("checking out a persistent database")?
            .commit_to_path(&persistent_dir.join(&hex))
            .context("snapshotting a persistent database")?;
        InMemoryKeyValueStore::checkout(in_memory_repo, &db_commit)
            .context("checking out an in-memory database")?
            .commit_to_path(&in_memory_dir.join(&hex))
            .context("snapshotting an in-memory database")?;
    }
    Ok(())
}

/// Restore the base saved by [`save_base`] into fresh repositories on both
/// backends, so a subsequent [`Registry::checkout`] of `base_commit` succeeds.
fn restore_base(
    failure_dir: &Path,
    persistent_repo: &DirectoryManager,
    in_memory_repo: &InMemoryRepo,
    base_commit: &CommitId,
) -> Result<()> {
    let manifest =
        fs::read(failure_dir.join(MANIFEST_FILE)).context("reading the registry manifest")?;
    persistent_repo
        .write_registry_commit(base_commit, &manifest)
        .context("registering the persistent manifest")?;
    in_memory_repo
        .write_registry_commit(base_commit, &manifest)
        .context("registering the in-memory manifest")?;

    // The manifest is now readable from either repo.
    let db_commits =
        Registry::<PersistenceLayer, Normal>::database_commits(persistent_repo, base_commit)
            .context("reading the base's database commits")?;

    let persistent_dir = failure_dir.join(PERSISTENT_DBS);
    let in_memory_dir = failure_dir.join(IN_MEMORY_DBS);

    for db_commit in db_commits {
        let hex = db_commit.hex_encode();

        let working = persistent_repo
            .temp_database_dir()
            .context("creating a scratch directory")?;
        PersistenceLayer::checkout_from_path(&persistent_dir.join(&hex), working)
            .context("loading a persistent database snapshot")?
            .commit(persistent_repo, &db_commit)
            .context("registering a persistent database")?;

        let working = persistent_repo
            .temp_database_dir()
            .context("creating a scratch directory")?;
        InMemoryKeyValueStore::checkout_from_path(&in_memory_dir.join(&hex), working)
            .context("loading an in-memory database snapshot")?
            .commit(in_memory_repo, &db_commit)
            .context("registering an in-memory database")?;
    }
    Ok(())
}

/// Reproduce a recorded failure by reconstructing the failing epoch's base on
/// both backends and applying the saved (shrunk) operation sequence once.
pub fn replay_failure(dir: &Path) -> Result<()> {
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

    restore_base(dir, &persistent_repo, &in_memory_repo, &meta.base_commit)?;

    let base = Base {
        commit: meta.base_commit,
        model: read_failure_file(dir, BASE_MODEL_FILE)?,
    };
    let ops: Vec<RegistryOperation> = read_failure_file(dir, REGRESSION_FILE)?;

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_case(
            &in_memory_repo,
            &persistent_repo,
            &base,
            &ops,
            meta.fail_on_warning,
        );
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
    use std::num::NonZeroUsize;
    use std::path::PathBuf;

    use bytes::Bytes;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::key::Key;
    use crate::test_helpers::database::DatabaseOperation;

    fn restricted_config() -> LongTestConfig {
        LongTestConfig {
            epochs: Some(3),
            ops_per_epoch: 200,
            cases_per_epoch: 32,
            seed: None,
            time_budget: None,
            keep_epochs: Some(NonZeroUsize::new(2).expect("non-zero")),
            out_dir: None,
            fail_on_warning: false,
        }
    }

    // A short run of the registry long test, with the default growing size.
    #[test]
    fn registry_long_test_restricted() {
        run_long_test(restricted_config(), 5, false)
            .expect("the short registry long test run should succeed");
    }

    // A short run with a stable registry size (around `2 * permanent`).
    #[test]
    fn registry_long_test_restricted_keep_stable_size() {
        run_long_test(restricted_config(), 5, true)
            .expect("the short keep-stable-size registry long test run should succeed");
    }

    const PERMANENT: usize = 2;

    struct TestSetup {
        _tmp: TestableTmpdir,
        out_dir: PathBuf,
        persistent_repo: DirectoryManager,
        in_memory_repo: InMemoryRepo,
        base: Base<RegistryLongTestModel>,
        key: Key,
    }

    /// Build a base committed on both backends whose database 0 holds a key.
    fn build_base_with_key() -> TestSetup {
        let tmp = TestableTmpdir::new();
        let out_dir = tmp.path().to_owned();
        let repo_dir = out_dir.join("repo");
        fs::create_dir_all(&repo_dir).expect("creating the repo dir should succeed");
        let persistent_repo = DirectoryManager::new(&repo_dir)
            .expect("creating the directory manager should succeed");
        let in_memory_repo = InMemoryRepo::default();

        let key = Key::new(&[1, 2, 3]).expect("the key should be valid");
        let set = RegistryOperation::Database(
            0,
            DatabaseOperation::Set(key.clone(), Bytes::from_static(b"value")),
        );

        let base = initial_base(&in_memory_repo, &persistent_repo, PERMANENT);
        let base = advance_base(&in_memory_repo, &persistent_repo, &base, &[set]);

        TestSetup {
            _tmp: tmp,
            out_dir,
            persistent_repo,
            in_memory_repo,
            base,
            key,
        }
    }

    fn dummy_meta(base_commit: CommitId) -> FailureMeta {
        FailureMeta {
            seed: Hash::from([0u8; 32]),
            epoch: 0,
            ops_per_epoch: 1,
            cases_per_epoch: 1,
            fail_on_warning: false,
            base_commit,
            reason: "test".to_string(),
            git_sha: "test".to_string(),
        }
    }

    // Replay of an artifact whose model disagrees with the restored base
    // reproduces the failure.
    #[test]
    fn internal_test_replay_reproduces_recorded_failure() {
        let setup = build_base_with_key();
        let meta = dummy_meta(setup.base.commit);

        // An empty model disagrees with the restored base (whose database 0
        // holds `key`): checking the key's existence mismatches and panics.
        let model = RegistryLongTestModel::new(PERMANENT);
        let ops = vec![RegistryOperation::Database(
            0,
            DatabaseOperation::Exists(setup.key.clone()),
        )];
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
    }

    // Replay of an artifact with the correct model faithfully restores the base
    // on both backends, so a non-mutating sequence passes.
    #[test]
    fn internal_test_replay_passes_for_a_consistent_base() {
        let setup = build_base_with_key();
        let meta = dummy_meta(setup.base.commit);

        let ops = vec![
            RegistryOperation::Database(0, DatabaseOperation::Exists(setup.key.clone())),
            RegistryOperation::Database(0, DatabaseOperation::Hash),
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
    }
}
