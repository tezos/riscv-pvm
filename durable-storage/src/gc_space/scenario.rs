// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! The shape of the scenario, and the state to measure it on.
//!
//! [`SpaceConfig`] is the shape. The rest builds a registry of that shape — a recorded base
//! state is reused when one matches and prepopulated when none does — and then modifies it once
//! per commit. Keys and values are derived from their indices, so no run has to hold a key list.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;
use serde::Serialize;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

use super::measure::measure;
use super::prune::prune_unreachable;
use super::report::record;
use super::report::report;
use super::report::report_header;
use super::report::report_prune;
use super::report::summarise;
use crate::commit::CommitId;
use crate::database::Database;
use crate::key::Key;
use crate::persistence_layer::PersistenceLayer;
use crate::registry::Registry;
use crate::repo::DirectoryManager;

/// The registry type this harness measures.
pub(super) type Reg = Registry<PersistenceLayer, Normal>;

/// Name of the file recording a reusable base state in the repository directory.
const BASE_STATE_FILE: &str = "gc_space_base.json";

/// How often to report progress while prepopulating.
const PREPOPULATE_PROGRESS_INTERVAL: usize = 250_000;

/// Shape of the scenario to measure.
#[derive(Debug, Clone)]
pub struct SpaceConfig {
    /// Number of databases in the registry.
    pub databases: usize,

    /// Keys written to each database before the measured commits begin.
    pub keys_per_database: usize,

    /// Length of each key, in bytes.
    pub key_size: usize,

    /// Length of an ordinary value, in bytes.
    ///
    /// The Etherlink trace in `benches/erc20_core/store_accesses.json` has a median write of 32
    /// bytes — balances and storage slots — so that is the size that characterises the workload.
    pub value_size: usize,

    /// Fraction of keys holding a large value instead, between 0 and 1.
    ///
    /// The same trace has a heavy tail: contract code and similar reach 131,220 bytes. A fraction
    /// of keys is given [`SpaceConfig::large_value_size`] to reproduce that, chosen deterministically
    /// per key so a key keeps its class across commits, the way a code slot stays code.
    pub large_value_fraction: f64,

    /// Length of a large value, in bytes.
    ///
    /// Values beyond `MAX_FILE_CHUNK_SIZE` cannot be written in one call, so these are built in
    /// chunks through successive offset writes — which is what a kernel storing contract code does.
    pub large_value_size: usize,

    /// Number of commits to make and measure.
    pub commits: usize,

    /// Keys modified before each commit, spread across the registry's databases.
    pub modified_keys: usize,

    /// Measure every this many commits. Scanning a column family is linear in its size, so at
    /// large scales it is worth sampling less often than every commit.
    pub sample_every: usize,

    /// Seed for choosing which keys each commit modifies.
    pub seed: u64,

    /// Where the repository lives. A fresh temporary directory when absent, in which case nothing
    /// survives the run.
    pub repo_dir: Option<PathBuf>,

    /// Write one JSON object per sample to this path.
    pub json_out: Option<PathBuf>,

    /// After the last commit, delete every commit not reachable from it and measure again.
    ///
    /// This is what collecting at directory granularity would reclaim, so running with it splits
    /// the storage into three parts: what a directory-level collection frees, what remains and is
    /// still needed, and what remains but is dead. The last part is the dead node data, which sits
    /// in files the surviving commit still references and which no directory deletion can reach.
    pub simulate_dir_gc: bool,
}

impl SpaceConfig {
    /// The parts of the shape that determine the base state, used to decide whether a recorded
    /// base state can be reused.
    fn base_shape(&self) -> BaseShape {
        BaseShape {
            databases: self.databases,
            keys_per_database: self.keys_per_database,
            key_size: self.key_size,
            value_size: self.value_size,
            large_value_per_mille: (self.large_value_fraction * 1000.0) as u32,
            large_value_size: self.large_value_size,
        }
    }

    /// Whether the key at `key_index` holds a large value.
    ///
    /// Decided from the index alone, so it is stable across commits and needs no bookkeeping.
    fn is_large_key(&self, db_index: usize, key_index: usize) -> bool {
        if self.large_value_fraction <= 0.0 {
            return false;
        }

        let per_mille = (self.large_value_fraction * 1000.0) as u64;

        mix(db_index as u64 ^ mix(key_index as u64)) % 1000 < per_mille
    }

    /// The size of the value held at `key_index`.
    fn value_size_of(&self, db_index: usize, key_index: usize) -> usize {
        if self.is_large_key(db_index, key_index) {
            self.large_value_size
        } else {
            self.value_size
        }
    }

    /// Run the scenario and report.
    pub fn run(self) -> Result<()> {
        // Everything the run prints goes to one handle, taken once here and lent to whatever
        // reports. Locking it once also keeps the table's rows from interleaving.
        let mut out = io::stderr().lock();
        // Kept alive for the whole run: dropping it removes the repository.
        let temp_dir = match self.repo_dir {
            Some(_) => None,
            None => Some(tempfile::tempdir().context("creating a temporary repository directory")?),
        };

        let repo_path = match (&self.repo_dir, &temp_dir) {
            (Some(path), _) => path.clone(),
            (None, Some(temp)) => temp.path().to_path_buf(),
            (None, None) => unreachable!("one of the two is always set"),
        };

        fs::create_dir_all(&repo_path).with_context(|| {
            format!("creating the repository directory {}", repo_path.display())
        })?;

        let repo = DirectoryManager::new(&repo_path).context("opening the repository")?;

        writeln!(
            out,
            "repository: {}\nshape: {} database(s) x {} keys x {} B values, {} commits x {} modified keys",
            repo_path.display(),
            self.databases,
            self.keys_per_database,
            self.value_size,
            self.commits,
            self.modified_keys,
        )?;

        let (mut registry, base_commit) = base_registry(&mut out, &repo, &repo_path, &self)?;

        // Every repository-wide figure below counts the whole directory, so anything an earlier
        // run left in it would be measured as part of this one.
        reset_to_base(&mut out, &repo, &repo_path, &base_commit)?;

        let mut samples = Vec::new();
        let mut json = match &self.json_out {
            Some(path) => Some(BufWriter::new(fs::File::create(path).with_context(
                || format!("creating the JSON output file {}", path.display()),
            )?)),
            None => None,
        };

        // The base state is sample zero, so the first measured commit has something to be a delta of.
        let (base, mut files) = measure(&repo, &repo_path, &base_commit, 0, Duration::ZERO, None)
            .context("measuring the base state")?;
        report_header(&mut out)?;
        report(&mut out, &base)?;
        record(&mut json, &base)?;
        samples.push(base);

        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut last_commit = base_commit;

        for commit_index in 1..=self.commits {
            modify(&mut registry, &self, &mut rng, commit_index)
                .with_context(|| format!("applying modifications for commit {commit_index}"))?;

            let started = Instant::now();
            let commit = registry
                .commit()
                .with_context(|| format!("committing at commit {commit_index}"))?;
            let commit_time = started.elapsed();
            last_commit = commit;

            if commit_index % self.sample_every != 0 && commit_index != self.commits {
                continue;
            }

            let (sample, current) = measure(
                &repo,
                &repo_path,
                &commit,
                commit_index,
                commit_time,
                Some(&files),
            )
            .with_context(|| format!("measuring commit {commit_index}"))?;
            files = current;
            report(&mut out, &sample)?;
            record(&mut json, &sample)?;
            samples.push(sample);
        }

        if let Some(json) = &mut json {
            json.flush().context("flushing the JSON output")?;
        }

        summarise(&mut out, &samples)?;

        if self.simulate_dir_gc {
            // The base state is retained alongside the last commit, not because a collection would
            // keep it, but because a persistent `--repo-dir` records it for later runs to check out.
            // Pruning it would leave `gc_space_base.json` naming a commit whose directory is gone, and
            // the next run of the same shape would fail instead of reusing the base.
            let retained = [last_commit, base_commit];
            let outcome = prune_unreachable(&repo, &repo_path, &retained)
                .context("simulating a directory-level collection")?;
            report_prune(&mut out, &outcome, samples.last())?;
        }

        Ok(())
    }
}

/// Identifies a prepopulated base state, so a run can tell whether the one on disk is the one it
/// wants. The commit sequence is deliberately not part of this: it starts from the base state and
/// never alters it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct BaseShape {
    databases: usize,
    keys_per_database: usize,
    key_size: usize,
    value_size: usize,
    large_value_per_mille: u32,
    large_value_size: usize,
}

/// A recorded base state that later runs can check out instead of prepopulating again.
#[derive(Debug, Serialize, Deserialize)]
struct BaseState {
    shape: BaseShape,

    /// Hex-encoded registry commit of the prepopulated state.
    commit: String,
}

/// Obtain the prepopulated registry to run the commit sequence against, reusing a recorded base
/// state when the shape matches and building one otherwise.
///
/// Either way what comes back is a *checkout*, never the registry that did the prepopulating. A
/// database that has just been written to holds a different LSM shape from a checkout of its own
/// commit, so returning the one that built the base would make the first run against a directory
/// report different occupancy, pinning and sharing figures from every run after it.
fn base_registry(
    out: &mut impl Write,
    repo: &DirectoryManager,
    repo_path: &Path,
    config: &SpaceConfig,
) -> Result<(Reg, CommitId)> {
    if let Some(commit) = recorded_base(out, repo_path, config)? {
        writeln!(
            out,
            "reusing the recorded base state {}",
            commit.hex_encode()
        )?;

        let registry =
            Reg::checkout(repo.clone(), commit).context("checking out the base state")?;

        return Ok((registry, commit));
    }

    let mut registry = Reg::new(repo.clone());

    for size in 1..=config.databases {
        registry
            .resize_tick(size)
            .with_context(|| format!("growing the registry to {size} database(s)"))?;
    }

    prepopulate(out, &mut registry, config)?;

    writeln!(out, "committing the base state...")?;
    let started = Instant::now();
    let commit = registry.commit().context("committing the base state")?;
    writeln!(
        out,
        "base state committed as {} in {:.1}s",
        commit.hex_encode(),
        started.elapsed().as_secs_f64()
    )?;

    record_base(repo_path, config, &commit)?;

    // Measure a checkout of the base state even on the run that builds it. Dropping the registry
    // first releases the working database it prepopulated into, so the checkout below starts from
    // the committed files alone.
    drop(registry);

    let registry = Reg::checkout(repo.clone(), commit).context("checking out the base state")?;

    Ok((registry, commit))
}

/// Delete every commit the base state does not reach, so a reused repository measures what a
/// fresh one would.
///
/// A persistent `--repo-dir` is how a run avoids prepopulating again, but it also keeps whatever
/// earlier runs committed, and occupancy, the commit directory count, what the history pins and
/// what a simulated collection frees are all repository-wide. Clearing them up front is what makes
/// successive runs comparable. It matters for a mismatched recorded base too: that base is ignored
/// and rebuilt, but its commits stay on disk until this removes them.
fn reset_to_base(
    out: &mut impl Write,
    repo: &DirectoryManager,
    repo_path: &Path,
    base: &CommitId,
) -> Result<()> {
    let reachable =
        Reg::database_commits(repo, base).context("reading the base state's registry manifest")?;

    let mut removed = 0;

    let commits_dir = repo_path.join("databases").join("commits");
    let entries =
        fs::read_dir(&commits_dir).with_context(|| format!("reading {}", commits_dir.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("reading an entry of {}", commits_dir.display()))?;

        // A commit is a directory, and `remove_dir_all` would fail on anything else.
        if !entry
            .file_type()
            .with_context(|| format!("reading the type of {}", entry.path().display()))?
            .is_dir()
        {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();

        if reachable.iter().any(|commit| commit.hex_encode() == name) {
            continue;
        }

        fs::remove_dir_all(entry.path())
            .with_context(|| format!("removing {}", entry.path().display()))?;
        removed += 1;
    }

    let registries_dir = repo_path.join("registries").join("commits");
    let entries = fs::read_dir(&registries_dir)
        .with_context(|| format!("reading {}", registries_dir.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("reading an entry of {}", registries_dir.display()))?;

        if entry.file_name().to_string_lossy() == base.hex_encode() {
            continue;
        }

        fs::remove_file(entry.path())
            .with_context(|| format!("removing {}", entry.path().display()))?;
        removed += 1;
    }

    if removed > 0 {
        writeln!(
            out,
            "removed {removed} commit(s) left by earlier runs, resetting to the base state"
        )?;
    }

    Ok(())
}

/// Read a recorded base state, if one is present and was built for this shape.
fn recorded_base(
    out: &mut impl Write,
    repo_path: &Path,
    config: &SpaceConfig,
) -> Result<Option<CommitId>> {
    let path = repo_path.join(BASE_STATE_FILE);

    if !path.exists() {
        return Ok(None);
    }

    let bytes = fs::read(&path)
        .with_context(|| format!("reading the base state file {}", path.display()))?;
    let base: BaseState = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing the base state file {}", path.display()))?;

    if base.shape != config.base_shape() {
        writeln!(
            out,
            "ignoring the recorded base state: it was built for a different shape ({:?})",
            base.shape
        )?;
        return Ok(None);
    }

    let bytes = hex::decode(&base.commit).context("decoding the recorded base commit")?;
    let hash: [u8; Hash::DIGEST_SIZE] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("recorded base commit is not {} bytes", Hash::DIGEST_SIZE))?;

    Ok(Some(CommitId::from(Hash::from(hash))))
}

/// Record a base state so later runs of the same shape can skip prepopulating.
fn record_base(repo_path: &Path, config: &SpaceConfig, commit: &CommitId) -> Result<()> {
    let path = repo_path.join(BASE_STATE_FILE);
    let base = BaseState {
        shape: config.base_shape(),
        commit: commit.hex_encode(),
    };

    let bytes = serde_json::to_vec_pretty(&base).context("encoding the base state")?;
    fs::write(&path, bytes)
        .with_context(|| format!("writing the base state file {}", path.display()))?;

    Ok(())
}

/// Fill every database with `keys_per_database` keys.
fn prepopulate(out: &mut impl Write, registry: &mut Reg, config: &SpaceConfig) -> Result<()> {
    let total = config.databases * config.keys_per_database;
    let started = Instant::now();
    let mut written = 0;

    for db_index in 0..config.databases {
        let database = registry
            .database_mut(db_index)
            .with_context(|| format!("getting database {db_index}"))?;

        for key_index in 0..config.keys_per_database {
            store_value(database, config, db_index, key_index, 0)
                .with_context(|| format!("writing key {key_index} in database {db_index}"))?;

            written += 1;

            if written % PREPOPULATE_PROGRESS_INTERVAL == 0 {
                writeln!(
                    out,
                    "prepopulating: {written}/{total} keys ({:.1}s elapsed)",
                    started.elapsed().as_secs_f64()
                )?;
            }
        }
    }

    writeln!(
        out,
        "prepopulated {total} keys in {:.1}s",
        started.elapsed().as_secs_f64()
    )?;

    Ok(())
}

/// Overwrite `modified_keys` existing keys, chosen uniformly across the registry.
///
/// The value depends on the commit index, so each modification really changes the value and
/// therefore the node hashes along its path to the root.
fn modify(
    registry: &mut Reg,
    config: &SpaceConfig,
    rng: &mut StdRng,
    generation: usize,
) -> Result<()> {
    // Group the picks by database so each database is borrowed once rather than per key.
    let mut picks: HashMap<usize, Vec<usize>> = HashMap::new();

    for _ in 0..config.modified_keys {
        let db_index = rng.random_range(0..config.databases);
        let key_index = rng.random_range(0..config.keys_per_database);
        picks.entry(db_index).or_default().push(key_index);
    }

    for (db_index, key_indices) in picks {
        let database = registry
            .database_mut(db_index)
            .with_context(|| format!("getting database {db_index}"))?;

        for key_index in key_indices {
            store_value(database, config, db_index, key_index, generation)
                .with_context(|| format!("writing key {key_index} in database {db_index}"))?;
        }
    }

    Ok(())
}

/// Write the value for a key, in chunks when it is too large for a single call.
///
/// `set` is capped at [`MAX_FILE_CHUNK_SIZE`], so anything above it is built by successive offset
/// writes, which is how a kernel stores contract code. The offset path also exercises the merge
/// operator, unlike `set`.
fn store_value(
    database: &mut Database<PersistenceLayer, Normal>,
    config: &SpaceConfig,
    db_index: usize,
    key_index: usize,
    generation: usize,
) -> Result<()> {
    let key = derive_key(db_index, key_index, config.key_size);
    let size = config.value_size_of(db_index, key_index);

    if size <= MAX_FILE_CHUNK_SIZE {
        database.set(key, derive_value(db_index, key_index, generation, size))?;

        return Ok(());
    }

    let value = derive_value(db_index, key_index, generation, size);

    for (chunk, offset) in value
        .chunks(MAX_FILE_CHUNK_SIZE)
        .zip((0..).step_by(MAX_FILE_CHUNK_SIZE))
    {
        database.write(key.clone(), offset, value.slice_ref(chunk))?;
    }

    Ok(())
}

/// Derive the key at `key_index` in database `db_index`.
///
/// Derived rather than stored so that a run at ten million keys needs no key list, and mixed so
/// that keys are spread through the tree the way real ones would be rather than arriving in order.
fn derive_key(db_index: usize, key_index: usize, key_size: usize) -> Key {
    let mut bytes = vec![0; key_size];
    let mut state = mix(db_index as u64) ^ mix(key_index as u64 + 1);

    for chunk in bytes.chunks_mut(8) {
        state = mix(state);
        let source = state.to_le_bytes();
        let len = chunk.len();
        chunk.copy_from_slice(&source[..len]);
    }

    Key::new(&bytes).expect("a derived key should be within the size limit")
}

/// Derive the value for a key at a given generation.
fn derive_value(db_index: usize, key_index: usize, generation: usize, value_size: usize) -> Bytes {
    let mut bytes = vec![0; value_size];
    let mut state =
        mix(db_index as u64) ^ mix(key_index as u64 + 1) ^ mix(generation as u64 + 0x9E37_79B9);

    for chunk in bytes.chunks_mut(8) {
        state = mix(state);
        let source = state.to_le_bytes();
        let len = chunk.len();
        chunk.copy_from_slice(&source[..len]);
    }

    Bytes::from(bytes)
}

/// SplitMix64, to turn indices into well-spread bytes without carrying an RNG around.
const fn mix(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;

    /// A run small enough for the test suite, still covering every measurement the harness makes:
    /// more than one database, a large-value tail, and enough commits for a sample to be a delta of
    /// the one before it.
    fn restricted_config(repo_dir: &Path, simulate_dir_gc: bool) -> SpaceConfig {
        SpaceConfig {
            databases: 2,
            keys_per_database: 100,
            key_size: 40,
            value_size: 32,
            large_value_fraction: 0.1,
            // Above MAX_FILE_CHUNK_SIZE, so the offset-write path in `store_value` is
            // exercised alongside the single-call one.
            large_value_size: 5000,
            commits: 2,
            modified_keys: 10,
            sample_every: 1,
            seed: 0,
            repo_dir: Some(repo_dir.to_path_buf()),
            json_out: None,
            simulate_dir_gc,
        }
    }

    /// A short run of the space harness, prepopulating its own base state.
    #[test]
    fn space_harness_restricted() {
        let tmp = TestableTmpdir::new();

        restricted_config(tmp.path(), false)
            .run()
            .expect("the short space harness run should succeed");
    }

    /// A recorded base state built for a different shape has to be ignored and rebuilt, and the
    /// commits it left behind cleared. Also samples every other commit rather than all of them.
    #[test]
    fn space_harness_restricted_rebuilds_mismatched_base() {
        let tmp = TestableTmpdir::new();

        restricted_config(tmp.path(), false)
            .run()
            .expect("the run that records the base state should succeed");

        let mut config = restricted_config(tmp.path(), false);
        // Part of the base shape, so the recorded state no longer matches.
        config.keys_per_database += 1;
        config.commits = 3;
        config.sample_every = 2;

        config
            .run()
            .expect("the run that rebuilds a mismatched base state should succeed");
    }

    /// The same shape run twice over one directory, the second time collecting. Covers the paths a
    /// single run cannot reach: reading back a recorded base state, resetting the repository to it,
    /// and pruning the commits the retained roots no longer hold.
    #[test]
    fn space_harness_restricted_reuses_base_and_collects() {
        let tmp = TestableTmpdir::new();

        restricted_config(tmp.path(), false)
            .run()
            .expect("the run that records the base state should succeed");

        // The second run also writes the samples out, so the JSON path is covered.
        let json_out = tmp.path().join("samples.json");
        let mut config = restricted_config(tmp.path(), true);
        config.json_out = Some(json_out.clone());

        config
            .run()
            .expect("the run that reuses the base state and collects should succeed");

        let written =
            std::fs::read_to_string(&json_out).expect("the samples file should be written");

        assert!(
            written.lines().count() > 1,
            "every sample should be one line of JSON, got {written:?}"
        );
    }
}
