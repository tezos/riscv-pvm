// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Durable storage lifecycle of the Etherlink kernel run by the EVM node, for blocks of ERC-20
//! transfers: storage accesses, state-root hashing and commits, with optional checkouts and garbage
//! collection.
//!
//! The durable storage is modelled as a [`Registry`] of the kernel's five keyspaces (see
//! [`BASE_DB`] and the following constants). Blocks of ERC-20 transfers are replayed from the
//! recorded store-access trace, with each transfer between a random sender and recipient drawn from
//! the prepopulated accounts, so the number of dirtied keys grows with the block size. The Tezos
//! keyspaces are prepopulated but never touched afterwards.
//!
//! As the kernel does with the Tezos runtime enabled, each block copies the four world-state and
//! accounts keyspaces into `/tmp` mirrors (further databases of the registry) after its setup,
//! runs against the mirrors, and moves them back at its end.
//!
//! A number of warm-up blocks are run, then the measured ones, all on one chain. Each block is
//! timed in parts: storage access, split into its critical part (replaying its accesses, and
//! copying and moving the mirrors) and the catch-up (waiting for the accounts keyspaces' Merkle
//! workers to apply the block's writes, if they are hashed), and the end of the block: hashing the
//! two accounts keyspaces (as the kernel does for its state roots), if enabled, and committing the
//! registry. Checkouts and garbage collections, with their compactions, are timed separately. At
//! the end of the run, each part, the end of the block, the whole block, the checkouts and the
//! collections are summarised over the measured blocks, followed by the throughput in transactions
//! per second over the measured blocks' durations.
//!
//! After each block, a summary is printed: the block's duration, split into its parts, how much the
//! committed state on disk grew, and the sizes of the committed state and of the working databases.
//! Each checkout and collection is printed on its own line with its duration; a collection also
//! with what it removed and how much disk it freed.
//!
//! The benchmark runs for hours with the default sizes, so it only runs when opted into with
//! `RUN_ETHERLINK_BENCH=1`; otherwise, as when `cargo bench` runs every benchmark of the crate, it
//! exits at once.
//!
//! Configuration, through environment variables:
//!
//! - `BLOCK_COUNT`: measured blocks.
//! - `WARM_UP_BLOCKS`: blocks run before the measured ones, left out of the summary.
//! - `TRANSACTIONS_PER_BLOCK`: ERC-20 transfers per block.
//! - `CHECKOUT_INTERVAL_BLOCKS`: check the registry out afresh from the last commit every this
//!   many blocks, instead of continuing from the in-memory state. `0` never checks out.
//! - `HASH_BEFORE_COMMIT`: whether to hash the accounts keyspaces before committing.
//! - `GC_INTERVAL_BLOCKS`: collect garbage every this many blocks. `0` never collects.
//! - `GC_KEEP_COMMITS`: how many of the most recent commits a collection retains.
//! - `ETH_ACCOUNT_COUNT`, `EVM_WORLD_STATE_KEYS_COUNT`, `TEZ_ACCOUNTS_KEYS_COUNT`,
//!   `TEZ_WORLD_STATE_KEYS_COUNT`: prepopulation sizes.
//! - `ETHERLINK_BENCH_DIR`: where the repositories live. Nothing else in it is touched. The run's
//!   repository is wiped at the start and at the end of the run.
//!
//! The prepopulated repository is kept in `ETHERLINK_BENCH_DIR` beside the run's, which is built
//! from it by hard links. If one built with the same prepopulation sizes is already there, it is
//! reused rather than built again. It is removed at the end of the run unless the benchmark is
//! passed `--keep-prepopulated-context`, for example:
//!
//! ```text
//! RUN_ETHERLINK_BENCH=1 cargo bench --features rocksdb --bench etherlink_lifecycle -- \
//!     --keep-prepopulated-context
//! ```
//!
//! A kept repository is reused on its prepopulation sizes alone. Should the derivation of the
//! accounts change in the meantime, the replay fails on the first account it does not find, and
//! the kept repository must be deleted. The trace itself reads keys it has deleted, so a missing
//! fixed key is tolerated, as the kernel does.
//!
//! Hard-linked files share the operating system's page cache, so a run reusing the prepopulated
//! repository reads what earlier runs left cached. Storage access and hashing, which load nodes
//! lazily from disk, get markedly faster from one run to the next. To compare configurations,
//! discard a first run and alternate between them, or drop the page cache before each run (`sudo
//! purge` on macOS, `echo 3 | sudo tee /proc/sys/vm/drop_caches` on Linux).
//!
//! [`Registry`]: octez_riscv_durable_storage::registry::Registry

mod erc20_trace;
mod random;
mod registry_common;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::hint::black_box;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use octez_riscv_durable_storage::collect::Suspend;
use octez_riscv_durable_storage::collect::collect_all;
use octez_riscv_durable_storage::commit::CommitId;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_durable_storage::gc_space::disk_usage;
use octez_riscv_durable_storage::key::Key;
use rand::prelude::*;
use rand::rngs::SmallRng;
use serde::Deserialize;
use serde::Serialize;

use crate::erc20_trace::Operation;
use crate::erc20_trace::SerialisedOperation;
use crate::erc20_trace::apply_operation;
use crate::erc20_trace::database_write_all;
use crate::erc20_trace::load_store_accesses;
use crate::random::generate_keys;
use crate::random::generate_random_bytes;
use crate::random::generate_random_bytes_in_range;
use crate::registry_common::Reg;
use crate::registry_common::env_usize;
use crate::registry_common::populate_registry;

/// `/base`
const BASE_DB: usize = 0;
/// `/evm/eth_accounts`
const ETH_ACCOUNTS_DB: usize = 1;
/// `/evm/world_state`
const EVM_WORLD_STATE_DB: usize = 2;
/// `/tez/tez_accounts`
const TEZ_ACCOUNTS_DB: usize = 3;
/// `/tez/world_state`
const TEZ_WORLD_STATE_DB: usize = 4;
const DATABASE_COUNT: usize = 5;
const DATABASE_NAMES: [&str; DATABASE_COUNT] = [
    "/base",
    "/evm/eth_accounts",
    "/evm/world_state",
    "/tez/tez_accounts",
    "/tez/world_state",
];
/// The databases the kernel copies into `/tmp` mirrors for each block, so that it can revert it.
const SAFE_ROOTS: [usize; 4] = [
    ETH_ACCOUNTS_DB,
    EVM_WORLD_STATE_DB,
    TEZ_ACCOUNTS_DB,
    TEZ_WORLD_STATE_DB,
];
/// The prepopulated databases, followed by the `/tmp` mirrors of [`SAFE_ROOTS`].
const REGISTRY_SIZE: usize = DATABASE_COUNT + SAFE_ROOTS.len();

/// The index of the `/tmp` mirror of the safe root `database`.
fn mirror(database: usize) -> usize {
    let position = SAFE_ROOTS
        .iter()
        .position(|&root| root == database)
        .expect("Only safe roots have mirrors");
    DATABASE_COUNT + position
}

// Default values that can be overridden by passing environment variables.
const BLOCK_COUNT: usize = 50;
const WARM_UP_BLOCKS: usize = 3;
const TRANSACTIONS_PER_BLOCK: usize = 1_000;
const CHECKOUT_INTERVAL_BLOCKS: usize = 0;
const HASH_BEFORE_COMMIT: bool = true;
const GC_INTERVAL_BLOCKS: usize = 0;
const GC_KEEP_COMMITS: usize = 1;
const ETH_ACCOUNT_COUNT: usize = 5_000_000;
const EVM_WORLD_STATE_KEYS_COUNT: usize = 100_000;
const TEZ_ACCOUNTS_KEYS_COUNT: usize = 1_000_000;
const TEZ_WORLD_STATE_KEYS_COUNT: usize = 100_000;

/// The sender of the recorded transfer, replaced by a random account.
const TRACE_SENDER: &str = "f0affc80a5f69f4a9a3ee01a640873b6ba53e539";
/// The ERC-20 contract of the recorded transfer.
const TRACE_ERC20: &str = "12f142944da31ab85458787aaecaf5e34128619d";
/// The balance slots of the recorded transfer, replaced by those of random accounts.
const TRACE_SENDER_SLOT: &str = "53a660bc3650c8e2ac16d52c0384149d93b4a8bb3796a1b3c6a241218b21b272";
const TRACE_RECIPIENT_SLOT: &str =
    "08e00ad51b7ea136b939b8b13bad52c822c0f203c98e9e0f355bcdfe4b989af2";

/// Passed to the benchmark to keep the prepopulated repository after the run.
const KEEP_PREPOPULATED_FLAG: &str = "--keep-prepopulated-context";
/// Set to opt into running the benchmark.
const RUN_FLAG: &str = "RUN_ETHERLINK_BENCH";

/// Directories and files within `ETHERLINK_BENCH_DIR`.
const PREPOPULATED_DIR: &str = "prepopulated";
const PREPOPULATED_MANIFEST: &str = "prepopulated.json";
const RUN_DIR: &str = "run";

const ACCOUNT_INFO_SIZE: usize = 77;
const BALANCE_SIZE: usize = 32;

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).as_deref() {
        Ok("1" | "true") => true,
        Ok("0" | "false") => false,
        Ok(_) => panic!("{name} must be true or false"),
        Err(_) => default,
    }
}

/// The prepopulation sizes, which decide whether a kept repository can be reused.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Shape {
    eth_account_count: usize,
    evm_world_state_keys_count: usize,
    tez_accounts_keys_count: usize,
    tez_world_state_keys_count: usize,
}

struct Config {
    block_count: usize,
    warm_up_blocks: usize,
    transactions_per_block: usize,
    checkout_interval_blocks: usize,
    hash_before_commit: bool,
    gc_interval_blocks: usize,
    gc_keep_commits: usize,
    shape: Shape,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Ethereum accounts, {} EVM world state keys, {} Tezos accounts keys, {} Tezos \
             world state keys",
            self.eth_account_count,
            self.evm_world_state_keys_count,
            self.tez_accounts_keys_count,
            self.tez_world_state_keys_count,
        )
    }
}

impl Config {
    fn print(&self) {
        let every = |interval: usize| match interval {
            0 => "never".to_owned(),
            interval => format!("every {interval} blocks"),
        };
        let gc = match self.gc_interval_blocks {
            0 => "off".to_owned(),
            interval => format!(
                "every {interval} blocks, keeping {} commits",
                self.gc_keep_commits
            ),
        };

        eprintln!("Configuration:");
        eprintln!(
            "  blocks                   {} (after {} warm-up)",
            self.block_count, self.warm_up_blocks
        );
        eprintln!("  transactions per block   {}", self.transactions_per_block);
        eprintln!(
            "  checkout                 {}",
            every(self.checkout_interval_blocks)
        );
        eprintln!("  hash before commit       {}", self.hash_before_commit);
        eprintln!("  GC                       {gc}");
    }

    fn from_env() -> Self {
        let config = Self {
            block_count: env_usize("BLOCK_COUNT", BLOCK_COUNT),
            warm_up_blocks: env_usize("WARM_UP_BLOCKS", WARM_UP_BLOCKS),
            transactions_per_block: env_usize("TRANSACTIONS_PER_BLOCK", TRANSACTIONS_PER_BLOCK),
            checkout_interval_blocks: env_usize(
                "CHECKOUT_INTERVAL_BLOCKS",
                CHECKOUT_INTERVAL_BLOCKS,
            ),
            hash_before_commit: env_bool("HASH_BEFORE_COMMIT", HASH_BEFORE_COMMIT),
            gc_interval_blocks: env_usize("GC_INTERVAL_BLOCKS", GC_INTERVAL_BLOCKS),
            gc_keep_commits: env_usize("GC_KEEP_COMMITS", GC_KEEP_COMMITS),
            shape: Shape {
                eth_account_count: env_usize("ETH_ACCOUNT_COUNT", ETH_ACCOUNT_COUNT),
                evm_world_state_keys_count: env_usize(
                    "EVM_WORLD_STATE_KEYS_COUNT",
                    EVM_WORLD_STATE_KEYS_COUNT,
                ),
                tez_accounts_keys_count: env_usize(
                    "TEZ_ACCOUNTS_KEYS_COUNT",
                    TEZ_ACCOUNTS_KEYS_COUNT,
                ),
                tez_world_state_keys_count: env_usize(
                    "TEZ_WORLD_STATE_KEYS_COUNT",
                    TEZ_WORLD_STATE_KEYS_COUNT,
                ),
            },
        };
        assert!(
            config.shape.eth_account_count >= 2,
            "Transfers need at least two accounts"
        );
        assert!(
            config.gc_keep_commits >= 1,
            "Collection must keep at least the latest commit"
        );
        config
    }
}

/// A prepopulated Ethereum account.
///
/// Derived from the account's index, so any account can be picked without keeping millions of
/// keys in memory.
struct Account {
    info: Key,
    balance: Key,
}

impl Account {
    fn new(index: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(index as u64);
        let mut address = [0u8; 20];
        let mut slot = [0u8; 32];
        rng.fill(&mut address);
        rng.fill(&mut slot);

        Account {
            info: key(&format!("/{}/info", hex::encode(address))),
            balance: key(&format!("/{TRACE_ERC20}/storage/{}", hex::encode(slot))),
        }
    }
}

fn print_stats(name: &str, samples: impl Iterator<Item = Duration>) {
    let mut samples: Vec<Duration> = samples.collect();
    samples.sort();

    let Some(&max) = samples.last() else {
        return;
    };
    let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
    let percentile = |p: f64| samples[((samples.len() - 1) as f64 * p).round() as usize];

    eprintln!(
        "  {name:<36} mean {:>10.2} ms, median {:>10.2} ms, p95 {:>10.2} ms, min {:>10.2} ms, \
         max {:>10.2} ms",
        millis(mean),
        millis(percentile(0.5)),
        millis(percentile(0.95)),
        millis(samples[0]),
        millis(max),
    );
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn mib(bytes: i64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn key(path: &str) -> Key {
    Key::new(path.as_bytes()).expect("The path should be a valid key")
}

/// Where the key of a replayed step comes from.
enum KeyTemplate {
    Fixed(Key),
    SenderInfo,
    SenderBalance,
    RecipientBalance,
    /// `{prefix}{block number}{suffix}`
    PerBlock {
        prefix: &'static str,
        suffix: String,
    },
}

#[derive(Clone, Copy)]
enum Access {
    Delete,
    Exists,
    Read(usize),
    ValueLength,
    Write(usize),
}

impl Access {
    fn operation(self, key: Key) -> Operation {
        match self {
            Access::Delete => Operation::Delete { key },
            Access::Exists => Operation::Exists { key },
            Access::Read(size) => Operation::Read { key, size },
            Access::ValueLength => Operation::ValueLength { key },
            Access::Write(size) => Operation::Write { key, size },
        }
    }
}

/// A trace operation mapped onto a database of the registry.
struct Step {
    database: usize,
    /// Whether the access goes to the `/tmp` mirror of `database`.
    mirrored: bool,
    key: KeyTemplate,
    access: Access,
}

impl Step {
    /// Map a recorded operation from the trace's single-tree layout onto the registry.
    ///
    /// Returns `None` for operations with no counterpart: the copy into the `/tmp` mirror (which
    /// the block makes itself), the trace's own hash (replaced by `HASH_BEFORE_COMMIT`), and
    /// the code reference count writes (which the kernel only makes on deployment).
    fn from_trace(operation: SerialisedOperation) -> Option<Self> {
        let (key, access) = match Operation::from(operation) {
            Operation::Clone | Operation::Hash => return None,
            Operation::Delete { key } => (key, Access::Delete),
            Operation::Exists { key } => (key, Access::Exists),
            Operation::Read { key, size } => (key, Access::Read(size)),
            Operation::ValueLength { key } => (key, Access::ValueLength),
            Operation::Write { key, size } => (key, Access::Write(size)),
        };
        let path = std::str::from_utf8(key.as_ref()).expect("Trace paths should be UTF-8");
        let in_tmp = path.starts_with("/tmp");

        let world_state = path
            .strip_prefix("/tmp")
            .unwrap_or(path)
            .strip_prefix("/evm/world_state");

        let (database, key) = if let Some(rest) = world_state {
            if let Some(account) = rest.strip_prefix("/eth_accounts") {
                let template = if account == format!("/{TRACE_SENDER}/info") {
                    KeyTemplate::SenderInfo
                } else if account.ends_with(TRACE_SENDER_SLOT) {
                    KeyTemplate::SenderBalance
                } else if account.ends_with(TRACE_RECIPIENT_SLOT) {
                    KeyTemplate::RecipientBalance
                } else {
                    KeyTemplate::Fixed(self::key(account))
                };
                (ETH_ACCOUNTS_DB, template)
            } else if rest.starts_with("/eth_codes") {
                if rest.ends_with("/ref_count") && matches!(access, Access::Write(_)) {
                    return None;
                }
                (ETH_ACCOUNTS_DB, KeyTemplate::Fixed(self::key(rest)))
            } else if rest == "/indexes/blocks/0" {
                let template = KeyTemplate::PerBlock {
                    prefix: "/blocks/indexes/",
                    suffix: String::new(),
                };
                (EVM_WORLD_STATE_DB, template)
            } else {
                (EVM_WORLD_STATE_DB, KeyTemplate::Fixed(self::key(rest)))
            }
        } else if let Some(rest) = path.strip_prefix("/evm") {
            let template = match rest.strip_prefix("/blueprints/0") {
                Some(suffix) => KeyTemplate::PerBlock {
                    prefix: "/blueprints/",
                    suffix: suffix.to_owned(),
                },
                None => KeyTemplate::Fixed(self::key(rest)),
            };
            (BASE_DB, template)
        } else {
            (BASE_DB, KeyTemplate::Fixed(key))
        };

        Some(Step {
            database,
            mirrored: in_tmp && SAFE_ROOTS.contains(&database),
            key,
            access,
        })
    }
}

fn steps(operations: Vec<SerialisedOperation>) -> Vec<Step> {
    operations
        .into_iter()
        .filter_map(Step::from_trace)
        .collect()
}

/// The mapped trace, and the data needed to replay it.
struct Workload {
    setup: Vec<Step>,
    transaction: Vec<Step>,
    block_creation: Vec<Step>,
    read_max_size: usize,
    random_data: Vec<u8>,
}

impl Workload {
    fn new(rng: &mut impl Rng) -> Self {
        let store_accesses = load_store_accesses();
        let setup = steps(store_accesses.setup);
        let transaction = steps(store_accesses.transaction);
        let block_creation = steps(store_accesses.block_creation);

        let (read_max_size, write_max_size) = setup
            .iter()
            .chain(&transaction)
            .chain(&block_creation)
            .fold((0, 0), |(read, write), step| match step.access {
                Access::Read(size) => (read.max(size), write),
                Access::Write(size) => (read, write.max(size)),
                _ => (read, write),
            });

        Workload {
            setup,
            transaction,
            block_creation,
            read_max_size,
            random_data: generate_random_bytes(rng, write_max_size),
        }
    }

    /// The fixed keys the trace reads, per database, with the largest size read from each.
    ///
    /// These are written up front so that the reads find values. Per-transaction keys are covered
    /// by the prepopulated accounts, and per-block keys are written before they are read.
    fn read_keys(&self) -> Vec<HashMap<Key, usize>> {
        let mut reads = vec![HashMap::new(); DATABASE_COUNT];
        for step in self
            .setup
            .iter()
            .chain(&self.transaction)
            .chain(&self.block_creation)
        {
            if let (KeyTemplate::Fixed(key), Access::Read(size)) = (&step.key, step.access) {
                let entry = reads[step.database].entry(key.clone()).or_insert(0);
                *entry = (*entry).max(size);
            }
        }
        reads
    }
}

/// Build the five-database registry the benchmark starts from.
fn populate(repo: &DirectoryManager, config: &Config, workload: &Workload) -> Manifest {
    let reads = workload.read_keys();
    let mut rng = rand::rng();

    let (_, key_counts, commit_id) = populate_registry(repo, DATABASE_COUNT, |index, database| {
        let filler_keys_count = match index {
            EVM_WORLD_STATE_DB => config.shape.evm_world_state_keys_count,
            TEZ_ACCOUNTS_DB => config.shape.tez_accounts_keys_count,
            TEZ_WORLD_STATE_DB => config.shape.tez_world_state_keys_count,
            _ => 0,
        };
        for key in generate_keys(&mut rng, filler_keys_count) {
            let value = Bytes::from(generate_random_bytes_in_range(&mut rng, 1..32));
            database.set(key, value).ok();
        }

        let mut account_keys_count = 0;
        if index == ETH_ACCOUNTS_DB {
            account_keys_count = 2 * config.shape.eth_account_count;
            for account in (0..config.shape.eth_account_count).map(Account::new) {
                let info = Bytes::from(generate_random_bytes(&mut rng, ACCOUNT_INFO_SIZE));
                let balance = Bytes::from(generate_random_bytes(&mut rng, BALANCE_SIZE));
                database
                    .set(account.info, info)
                    .expect("Writing an account should succeed");
                database
                    .set(account.balance, balance)
                    .expect("Writing a balance should succeed");
            }
        }

        for (key, &size) in &reads[index] {
            let value = generate_random_bytes(&mut rng, size);
            database_write_all(database, key.clone(), &value)
                .expect("Writing a read value should succeed");
        }

        filler_keys_count + account_keys_count + reads[index].len()
    });

    Manifest {
        shape: config.shape,
        commit_id,
        key_counts,
    }
}

/// Describes the prepopulated repository kept in `ETHERLINK_BENCH_DIR`.
#[derive(Serialize, Deserialize)]
struct Manifest {
    shape: Shape,
    commit_id: CommitId,
    key_counts: Vec<usize>,
}

/// Return the prepopulated repository's manifest, building the repository first unless one with
/// the configured shape is already in `bench_dir`.
fn prepopulated(bench_dir: &Path, config: &Config, workload: &Workload) -> Manifest {
    let manifest_path = bench_dir.join(PREPOPULATED_MANIFEST);
    let repo_dir = bench_dir.join(PREPOPULATED_DIR);

    let existing = std::fs::File::open(&manifest_path)
        .ok()
        .and_then(|file| serde_json::from_reader::<_, Manifest>(file).ok());
    let shape = config.shape;
    let reason = match existing {
        Some(manifest) if manifest.shape == shape && repo_dir.exists() => {
            eprintln!(
                "Reusing the prepopulated repository at {}.",
                repo_dir.display()
            );
            eprintln!(
                "Warning: earlier runs may have left it in the page cache, which speeds up storage \
                 access and hashing. Discard a first run, or drop the page cache, before \
                 comparing runs."
            );
            return manifest;
        }
        Some(manifest) if manifest.shape != shape => Some(format!(
            "it was built with {}, but {shape} were requested",
            manifest.shape
        )),
        Some(_) => Some("its manifest names a repository that is missing".to_owned()),
        None if repo_dir.exists() || manifest_path.exists() => Some(
            "it has no readable manifest, so it may be left from an interrupted run".to_owned(),
        ),
        None => None,
    };

    if let Some(reason) = reason {
        eprintln!(
            "Deleting the prepopulated repository at {}: {reason}.",
            repo_dir.display()
        );
    }

    // The manifest goes first, so that an interruption leaves no manifest naming a partial repository.
    remove_file(&manifest_path);
    remove_dir(&repo_dir);

    let repo = DirectoryManager::new(&repo_dir).expect("Failed to create directory manager");
    let manifest = populate(&repo, config, workload);

    let file = std::fs::File::create(&manifest_path).expect("Creating the manifest should succeed");
    serde_json::to_writer_pretty(file, &manifest).expect("Writing the manifest should succeed");

    manifest
}

/// Build the run's repository at `run_dir` from the prepopulated one at `prepopulated_dir`.
///
/// The Merkle store is checkpointed, and the published database commits - which are never modified
/// in place - are hard-linked, so this costs next to nothing and leaves the prepopulated repository
/// as it was.
fn build_run_repo(prepopulated_dir: &Path, run_dir: &Path) {
    let prepopulated = DirectoryManager::new(prepopulated_dir)
        .expect("Failed to open the prepopulated repository");

    std::fs::create_dir_all(run_dir).expect("Creating the run directory should succeed");
    prepopulated
        .merkle_store()
        .checkpoint(&DirectoryManager::merkle_dir_in(run_dir))
        .expect("Checkpointing the Merkle store should succeed");

    for dir in [
        Path::new("databases").join("commits"),
        PathBuf::from("registries"),
    ] {
        link_tree(&prepopulated_dir.join(&dir), &run_dir.join(&dir));
    }
}

/// Copy the tree at `from` to `to`, hard-linking RocksDB's immutable table and blob files and
/// copying the rest, which RocksDB may rewrite.
fn link_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("Creating a directory should succeed");
    for entry in std::fs::read_dir(from).expect("Reading a directory should succeed") {
        let entry = entry.expect("Reading a directory entry should succeed");
        let (source, target) = (entry.path(), to.join(entry.file_name()));

        let is_dir = entry
            .file_type()
            .expect("Reading a file type should succeed")
            .is_dir();
        if is_dir {
            link_tree(&source, &target);
        } else if matches!(
            source.extension().and_then(|extension| extension.to_str()),
            Some("sst" | "blob")
        ) {
            std::fs::hard_link(&source, &target).expect("Hard-linking a file should succeed");
        } else {
            std::fs::copy(&source, &target).expect("Copying a file should succeed");
        }
    }
}

/// The keys of the transfer being replayed.
struct Transfer {
    sender: Account,
    recipient: Account,
}

/// The chain of blocks being run.
struct Chain<'a> {
    config: &'a Config,
    workload: &'a Workload,
    read_buffer: Vec<u8>,
    repo: DirectoryManager,
    repo_dir: &'a Path,
    registry: Reg,
    /// Most recent last, trimmed to `GC_KEEP_COMMITS` whenever garbage is collected.
    recent_commits: VecDeque<CommitId>,
    block_number: usize,
    blocks_since_checkout: usize,
    rng: ThreadRng,
    /// The timed parts of the measured blocks.
    times: Vec<BlockTimes>,
    /// The checkouts before the measured blocks.
    checkout_times: Vec<Duration>,
    /// The collections after the measured blocks.
    gc_times: Vec<GcTimes>,
}

/// The timed parts of a block.
struct BlockTimes {
    /// The critical part of storage access: replaying the block, copying the safe roots into their
    /// mirrors and promoting them back.
    storage_access: Duration,
    /// The catch-up part of storage access: waiting for the accounts keyspaces' Merkle trees to
    /// apply the block's writes, before hashing.
    merkle_catch_up: Duration,
    /// The accounts keyspaces, if they are hashed before committing.
    hashing: Duration,
    commit: Duration,
}

/// The timed parts of a garbage collection.
struct GcTimes {
    collect: Duration,
    compact: Duration,
}

impl BlockTimes {
    /// The hashing and the commit.
    fn end(&self) -> Duration {
        self.hashing + self.commit
    }

    fn block(&self) -> Duration {
        self.storage_access + self.merkle_catch_up + self.end()
    }
}

impl Chain<'_> {
    /// Run one block, recording its timed parts if it is `measured`.
    fn run_block(&mut self, measured: bool) {
        // Warm-up and measured blocks are each numbered from 1.
        let label = if measured {
            format!(
                "block {}",
                self.block_number + 1 - self.config.warm_up_blocks
            )
        } else {
            format!("warm-up block {}", self.block_number + 1)
        };

        let interval = self.config.checkout_interval_blocks;
        if interval > 0 && self.blocks_since_checkout >= interval {
            let start = Instant::now();
            let registry = Reg::checkout(self.repo.clone(), self.last_commit())
                .expect("Checking out the last commit should succeed");
            let checkout_time = start.elapsed();
            self.registry = registry;
            self.blocks_since_checkout = 0;

            eprintln!("checkout before {label}: {:.2} ms", millis(checkout_time));
            if measured {
                self.checkout_times.push(checkout_time);
            }
        }

        let workload = self.workload;
        let start = Instant::now();
        self.replay(&workload.setup, None);
        // The kernel copies the safe roots at the end of the setup, where the trace has its copy.
        for root in SAFE_ROOTS {
            self.registry
                .copy_database(root, mirror(root))
                .expect("Copying a safe root should succeed");
        }
        for _ in 0..self.config.transactions_per_block {
            let account_count = self.config.shape.eth_account_count;
            let sender = self.rng.random_range(0..account_count);
            let recipient = (sender + self.rng.random_range(1..account_count)) % account_count;
            let transfer = Transfer {
                sender: Account::new(sender),
                recipient: Account::new(recipient),
            };
            self.replay(&workload.transaction, Some(&transfer));
        }
        self.replay(&workload.block_creation, None);
        for root in SAFE_ROOTS {
            self.registry
                .move_database(mirror(root), root)
                .expect("Promoting a safe root should succeed");
        }
        let storage_access = start.elapsed();

        let size_before = self.committed_size();

        let start = Instant::now();
        if self.config.hash_before_commit {
            for index in [ETH_ACCOUNTS_DB, TEZ_ACCOUNTS_DB] {
                self.registry
                    .database(index)
                    .expect("The database index should be valid")
                    .wait_for_merkle()
                    .expect("Waiting for the Merkle tree should succeed");
            }
        }
        let merkle_catch_up = start.elapsed();

        let start = Instant::now();
        if self.config.hash_before_commit {
            for index in [ETH_ACCOUNTS_DB, TEZ_ACCOUNTS_DB] {
                black_box(
                    self.registry
                        .database(index)
                        .expect("The database index should be valid")
                        .hash()
                        .expect("Hashing should succeed"),
                );
            }
        }
        let hashing = start.elapsed();

        let start = Instant::now();
        let commit_id = self.registry.commit().expect("Committing should succeed");
        let commit = start.elapsed();
        let times = BlockTimes {
            storage_access,
            merkle_catch_up,
            hashing,
            commit,
        };

        let size_after = self.committed_size();
        let total = self.disk_usage(self.repo_dir);
        self.recent_commits.push_back(commit_id);
        self.block_number += 1;
        self.blocks_since_checkout += 1;

        let summary = format!(
            "{label}: {:.2} ms (storage access - critical {:.2} ms, storage access - catch-up {:.2} ms, \
             hashing {:.2} ms, commit {:.2} ms), commit {:+.2} MiB, committed {:.2} MiB, \
             working {:.2} MiB",
            millis(times.block()),
            millis(times.storage_access),
            millis(times.merkle_catch_up),
            millis(times.hashing),
            millis(times.commit),
            mib(size_after as i64 - size_before as i64),
            mib(size_after as i64),
            mib(total as i64 - size_after as i64),
        );

        eprintln!("{summary}");
        if measured {
            self.times.push(times);
        }

        let interval = self.config.gc_interval_blocks;
        if interval > 0 && self.block_number.is_multiple_of(interval) {
            let (description, gc_times) = self.collect_garbage();
            eprintln!("GC after {label}: {description}");
            if measured {
                self.gc_times.push(gc_times);
            }
        }
    }

    /// Print statistics of the timed parts of every measured block run so far.
    fn print_summary(&self) {
        eprintln!("Timed parts of {} measured blocks:", self.times.len());

        print_stats(
            "storage access - critical",
            self.times.iter().map(|times| times.storage_access),
        );
        print_stats(
            "storage access - catch-up",
            self.times.iter().map(|times| times.merkle_catch_up),
        );
        print_stats("hashing", self.times.iter().map(|times| times.hashing));
        print_stats("commit", self.times.iter().map(|times| times.commit));
        print_stats("end of block", self.times.iter().map(BlockTimes::end));
        print_stats("block", self.times.iter().map(BlockTimes::block));
        print_stats(
            &format!("checkout ({} checkouts)", self.checkout_times.len()),
            self.checkout_times.iter().copied(),
        );
        let rounds = self.gc_times.len();
        print_stats(
            &format!("GC collection ({rounds} rounds)"),
            self.gc_times.iter().map(|times| times.collect),
        );
        print_stats(
            &format!("GC compaction ({rounds} rounds)"),
            self.gc_times.iter().map(|times| times.compact),
        );

        let blocks: Duration = self.times.iter().map(BlockTimes::block).sum();
        if !blocks.is_zero() {
            let transactions = self.times.len() * self.config.transactions_per_block;
            eprintln!(
                "  {:<36} {:>13.2}",
                "transactions per second",
                transactions as f64 / blocks.as_secs_f64()
            );
        }
    }

    fn last_commit(&self) -> CommitId {
        *self
            .recent_commits
            .back()
            .expect("There is always at least one commit")
    }

    /// Bytes `dir` occupies on disk, counting hard-linked files once.
    fn disk_usage(&self, dir: &Path) -> u64 {
        disk_usage(dir)
            .expect("Measuring the repository should succeed")
            .unique_bytes
    }

    /// Bytes the committed state occupies on disk: the database commits, the registry manifests
    /// and the Merkle store.
    ///
    /// The working databases are left out. Checking out replaces them, and the old ones are
    /// deleted once their Merkle workers stop (see [`remove_dir`]), which can happen while the
    /// commit is being measured.
    fn committed_size(&self) -> u64 {
        [
            self.repo_dir.join("databases").join("commits"),
            self.repo_dir.join("registries"),
            self.repo.merkle_store_dir(),
            self.repo.merkle_slots_dir(),
        ]
        .iter()
        .map(|dir| self.disk_usage(dir))
        .sum()
    }

    /// Drop the commits older than the `GC_KEEP_COMMITS` most recent, describe what that took,
    /// removed and freed, and return how long it took.
    ///
    /// The store is compacted straight away, so that the freed space is measured rather than left
    /// to RocksDB's background compaction.
    fn collect_garbage(&mut self) -> (String, GcTimes) {
        while self.recent_commits.len() > self.config.gc_keep_commits {
            self.recent_commits.pop_front();
        }
        let oldest = self
            .recent_commits
            .front()
            .expect("There is always at least one commit");

        let size_before = self.committed_size();

        let start = Instant::now();
        let round =
            collect_all(&self.repo, oldest, &Suspend::new()).expect("Collecting should succeed");
        let collect_time = start.elapsed();

        let start = Instant::now();
        self.repo.merkle_store().compact();
        let compact_time = start.elapsed();

        let freed = size_before as i64 - self.committed_size() as i64;
        let description = format!(
            "removed {} registry commits, {} database commits, {} of {} examined nodes \
             ({:.2} MiB) in {:.2} s, compacted in {:.2} s, freed {:.2} MiB",
            round.collected.registry_commits,
            round.collected.database_commits,
            round.swept.nodes,
            round.swept.examined,
            mib(round.swept.bytes as i64),
            collect_time.as_secs_f64(),
            compact_time.as_secs_f64(),
            mib(freed),
        );
        let times = GcTimes {
            collect: collect_time,
            compact: compact_time,
        };
        (description, times)
    }

    fn replay(&mut self, steps: &[Step], transfer: Option<&Transfer>) {
        for step in steps {
            let transfer = || transfer.expect("Only transactions use transfer keys");
            let key = match &step.key {
                KeyTemplate::Fixed(key) => key.clone(),
                KeyTemplate::SenderInfo => transfer().sender.info.clone(),
                KeyTemplate::SenderBalance => transfer().sender.balance.clone(),
                KeyTemplate::RecipientBalance => transfer().recipient.balance.clone(),
                KeyTemplate::PerBlock { prefix, suffix } => {
                    key(&format!("{prefix}{}{suffix}", self.block_number))
                }
            };

            let index = if step.mirrored {
                mirror(step.database)
            } else {
                step.database
            };
            let database = self
                .registry
                .database_mut(index)
                .expect("The database index should be valid");
            apply_operation(
                database,
                step.access.operation(key),
                &mut self.read_buffer,
                &self.workload.random_data,
            );
        }
    }
}

/// Fail if the prepopulated accounts are not where the transfers look for them.
///
/// A kept repository is reused on its shape alone, so this catches a change to [`Account::new`]
/// (such as a `rand` bump changing `SmallRng`'s output) that would otherwise make every transfer
/// access accounts that are not there, and silently measure a different workload.
fn check_accounts(registry: &Reg, config: &Config, bench_dir: &Path) {
    let database = registry
        .database(ETH_ACCOUNTS_DB)
        .expect("The database index should be valid");
    for index in [0, config.shape.eth_account_count - 1] {
        let account = Account::new(index);
        let found = database
            .exists(&account.info)
            .expect("The existence check should succeed")
            && database
                .exists(&account.balance)
                .expect("The existence check should succeed");
        assert!(
            found,
            "Account {index} is not in {}. The prepopulated repository is probably stale: delete \
             it from {} and run again.",
            DATABASE_NAMES[ETH_ACCOUNTS_DB],
            bench_dir.display(),
        );
    }
}

/// The directory holding the benchmark's repositories.
///
/// When dropped, the run's repository is removed, and so is the prepopulated one unless it is
/// being kept. The directory itself, and anything else in it, is left alone.
struct BenchDir {
    path: PathBuf,
    keep_prepopulated: bool,
}

impl BenchDir {
    /// Create the directory, first removing any run repository an interrupted run left behind.
    fn create(path: PathBuf, keep_prepopulated: bool) -> Self {
        std::fs::create_dir_all(&path).expect("Creating the benchmark directory should succeed");
        let bench_dir = BenchDir {
            path,
            keep_prepopulated,
        };
        remove_dir(&bench_dir.run());

        let kept = if keep_prepopulated { "kept" } else { "removed" };
        eprintln!(
            "Using {} for the repositories; the prepopulated one is {kept} at the end of the run.",
            bench_dir.path.display()
        );
        bench_dir
    }

    fn run(&self) -> PathBuf {
        self.path.join(RUN_DIR)
    }
}

impl Drop for BenchDir {
    fn drop(&mut self) {
        remove_dir(&self.run());
        if !self.keep_prepopulated {
            remove_file(&self.path.join(PREPOPULATED_MANIFEST));
            remove_dir(&self.path.join(PREPOPULATED_DIR));
        }
    }
}

fn remove_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => panic!("Failed to remove {}: {e}", path.display()),
    }
}

/// Remove the directory at `path`, if there is one.
///
/// A dropped database deletes its working directory once its Merkle worker stops, which can be
/// after the database is dropped. Removing a repository can therefore race with such deletions,
/// which abort the removal part-way; it is retried until the directory is gone, for a while.
fn remove_dir(path: &Path) {
    const ATTEMPTS: usize = 100;
    const PAUSE: Duration = Duration::from_millis(100);

    for attempt in 1.. {
        let error = match std::fs::remove_dir_all(path) {
            Ok(()) => return,
            Err(e) => e,
        };
        if !path.exists() {
            return;
        }
        if attempt == ATTEMPTS {
            eprintln!("Failed to remove {}: {error}", path.display());
            return;
        }
        std::thread::sleep(PAUSE);
    }
}

fn main() {
    if !env_bool(RUN_FLAG, false) {
        eprintln!("Skipping the Etherlink lifecycle benchmark; set {RUN_FLAG}=1 to run it.");
        return;
    }

    let mut keep_prepopulated = false;
    // `cargo bench` passes `--bench`.
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            KEEP_PREPOPULATED_FLAG => keep_prepopulated = true,
            "--bench" => {}
            _ => {
                eprintln!(
                    "Unknown argument {arg}; the only one accepted is {KEEP_PREPOPULATED_FLAG}."
                );
                std::process::exit(2);
            }
        }
    }

    let config = Config::from_env();

    // Declared first so it is dropped last, once the registry has let go of the repository.
    let bench_dir = BenchDir::create(
        std::env::var_os("ETHERLINK_BENCH_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("etherlink-lifecycle-bench")),
        keep_prepopulated,
    );

    let workload = Workload::new(&mut rand::rng());
    let manifest = prepopulated(&bench_dir.path, &config, &workload);
    for (name, count) in DATABASE_NAMES.iter().zip(&manifest.key_counts) {
        eprintln!("Populated {name} with {count} keys.");
    }

    let run_dir = bench_dir.run();
    build_run_repo(&bench_dir.path.join(PREPOPULATED_DIR), &run_dir);
    let repo = DirectoryManager::new(&run_dir).expect("Failed to create directory manager");
    let mut registry = Reg::checkout(repo.clone(), manifest.commit_id)
        .expect("Checking out the prepopulated commit should succeed");
    for size in DATABASE_COUNT + 1..=REGISTRY_SIZE {
        registry
            .resize_tick(size)
            .expect("Adding the mirrors should succeed");
    }
    check_accounts(&registry, &config, &bench_dir.path);

    let mut chain = Chain {
        config: &config,
        workload: &workload,
        read_buffer: vec![0u8; workload.read_max_size],
        repo,
        repo_dir: &run_dir,
        registry,
        recent_commits: VecDeque::from([manifest.commit_id]),
        block_number: 0,
        blocks_since_checkout: 0,
        rng: rand::rng(),
        times: Vec::new(),
        checkout_times: Vec::new(),
        gc_times: Vec::new(),
    };

    config.print();

    for measured in [false, true] {
        let count = if measured {
            config.block_count
        } else {
            config.warm_up_blocks
        };
        for _ in 0..count {
            chain.run_block(measured);
        }
    }

    chain.print_summary();
}
