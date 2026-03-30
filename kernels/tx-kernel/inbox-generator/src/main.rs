// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use clap::Parser;
use clap::Subcommand;
use libsecp256k1::Message;
use libsecp256k1::PublicKey;
use libsecp256k1::SecretKey;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::commit::CommitId;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_durable_storage::registry::Registry;
use octez_riscv_durable_storage::repo::DirectoryManager;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use sha3::Digest;
use sha3::Keccak256;

const BLOCK_BLUEPRINT_MAGIC: [u8; 4] = *b"TXB1";
const BLOCK_CHUNK_MAGIC: [u8; 4] = *b"TXC1";
const CONTEXT_NAME: &str = "/tx-kernel/context";
const ACCOUNT_KEY_PREFIX: &[u8] = b"/acct/";
const KEYSPACE_INDEX_PREFIX: &[u8] = b"/keyspaces/";
const DEFAULT_INITIAL_BALANCE: u64 = 1_000_000;
const DURABLE_STORAGE_HEAD_FILE: &str = "registry-head";
const META_BOOTSTRAPPED_KEY: &[u8] = b"/meta/bootstrapped";
const META_HEAD_KEY: &[u8] = b"/meta/head";
const PREPARE_CONTEXT_PROGRESS_INTERVAL: usize = 1_000;
const ROOT_TX_KERNEL_DIR_COMPONENTS: usize = 3;
const BLOCK_CHUNK_HEADER_SIZE: usize = 4 + 8 + 2 + 2;
const MAX_INPUT_MESSAGE_SIZE: usize = 4096;
const EXTERNAL_FRAME_SIZE: usize = 21;

#[derive(Parser)]
#[command(long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Generate a tx-kernel inbox.json file")]
    Generate {
        #[arg(long)]
        transactions: usize,
        #[arg(long, default_value_t = 100)]
        block_frequency: usize,
        #[arg(long, default_value_t = 1024)]
        accounts: usize,
        #[arg(long, default_value = "tx-kernel-inbox.json")]
        inbox_file: PathBuf,
    },
    #[command(about = "Prepare a durable-storage context with prepopulated accounts")]
    PrepareContext {
        #[arg(long)]
        durable_storage_dir: PathBuf,
        #[arg(long)]
        accounts: usize,
        #[arg(long, default_value_t = DEFAULT_INITIAL_BALANCE)]
        initial_balance: u64,
        #[arg(long, default_value_t = false)]
        rebuild: bool,
    },
    #[command(about = "Run the tx-kernel benchmark in the sandbox and report TPS")]
    Benchmark {
        #[arg(long)]
        transactions: usize,
        #[arg(long, default_value_t = 100)]
        block_frequency: usize,
        #[arg(long)]
        durable_storage_dir: PathBuf,
        #[arg(long, default_value_t = 1024)]
        accounts: usize,
        #[arg(long, default_value_t = DEFAULT_INITIAL_BALANCE)]
        initial_balance: u64,
        #[arg(long, default_value_t = false)]
        rebuild_context: bool,
        #[arg(long)]
        inbox_file: Option<PathBuf>,
        #[arg(long)]
        kernel: Option<PathBuf>,
        #[arg(long)]
        sandbox: Option<PathBuf>,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
enum InboxMessageFile {
    External {
        #[serde(with = "hex::serde")]
        external: Vec<u8>,
    },
}

#[derive(Serialize)]
struct InboxFile(Vec<Vec<InboxMessageFile>>);

#[derive(Deserialize)]
struct LogLine {
    elapsed: Elapsed,
    message: String,
}

#[derive(Deserialize)]
struct Elapsed {
    secs: u64,
    nanos: u32,
}

impl Elapsed {
    fn as_duration(&self) -> Duration {
        Duration::new(self.secs, self.nanos)
    }
}

#[derive(Clone, Copy)]
struct Account {
    secret_key: SecretKey,
    public_key: [u8; 65],
    address: [u8; 20],
}

#[derive(Clone, Copy)]
struct SignedTransaction {
    from: [u8; 20],
    to: [u8; 20],
    amount: u64,
    nonce: u64,
    public_key: [u8; 65],
    signature: [u8; 64],
}

struct BenchmarkOutcome {
    wall_duration: Duration,
    block_count: usize,
    tx_count: usize,
    applied_count: usize,
    block_window: Option<Duration>,
    tx_window: Option<Duration>,
    sampled_signature_window: Option<Duration>,
    sampled_state_window: Option<Duration>,
    sampled_tx_count: usize,
    finalization_window: Option<Duration>,
}

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn keccak256(bytes: &[u8]) -> [u8; 32] {
    Keccak256::digest(bytes).into()
}

fn sequencer_secret_key(index: usize) -> SecretKey {
    let mut bytes = [0u8; 32];
    bytes[24..].copy_from_slice(&(index as u64 + 1).to_be_bytes());
    SecretKey::parse(&bytes).expect("deterministic secret key should be valid")
}

fn make_account(index: usize) -> Account {
    let secret_key = sequencer_secret_key(index);
    let public_key = PublicKey::from_secret_key(&secret_key).serialize();
    let address = address_from_public_key(&public_key);
    Account {
        secret_key,
        public_key,
        address,
    }
}

fn address_from_public_key(public_key: &[u8; 65]) -> [u8; 20] {
    let hash = keccak256(&public_key[1..]);
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..]);
    address
}

fn sign_hash(secret_key: &SecretKey, hash: &[u8; 32]) -> [u8; 64] {
    let message = Message::parse(hash);
    let (signature, _) = libsecp256k1::sign(&message, secret_key);
    signature.serialize()
}

fn encode_unsigned_transaction(transaction: &SignedTransaction) -> [u8; 56] {
    let mut bytes = [0u8; 56];
    bytes[..20].copy_from_slice(&transaction.from);
    bytes[20..40].copy_from_slice(&transaction.to);
    bytes[40..48].copy_from_slice(&transaction.amount.to_le_bytes());
    bytes[48..56].copy_from_slice(&transaction.nonce.to_le_bytes());
    bytes
}

fn encode_transaction(transaction: &SignedTransaction, out: &mut Vec<u8>) {
    out.extend_from_slice(&transaction.from);
    out.extend_from_slice(&transaction.to);
    out.extend_from_slice(&transaction.amount.to_le_bytes());
    out.extend_from_slice(&transaction.nonce.to_le_bytes());
    out.extend_from_slice(&transaction.public_key);
    out.extend_from_slice(&transaction.signature);
}

fn transaction_hash(transaction: &SignedTransaction) -> [u8; 32] {
    keccak256(&encode_unsigned_transaction(transaction))
}

fn block_hash(number: u64, transactions: &[SignedTransaction]) -> [u8; 32] {
    let mut header = [0u8; 14];
    header[..4].copy_from_slice(&BLOCK_BLUEPRINT_MAGIC);
    header[4..12].copy_from_slice(&number.to_le_bytes());
    header[12..14].copy_from_slice(&(transactions.len() as u16).to_le_bytes());

    let mut acc = keccak256(&header);
    for transaction in transactions {
        let tx_hash = transaction_hash(transaction);
        let mut bytes = [0u8; 64];
        bytes[..32].copy_from_slice(&acc);
        bytes[32..].copy_from_slice(&tx_hash);
        acc = keccak256(&bytes);
    }

    acc
}

fn build_transaction(from: Account, to: Account, amount: u64, nonce: u64) -> SignedTransaction {
    let mut unsigned = SignedTransaction {
        from: from.address,
        to: to.address,
        amount,
        nonce,
        public_key: from.public_key,
        signature: [0u8; 64],
    };
    let tx_hash = keccak256(&encode_unsigned_transaction(&unsigned));
    unsigned.signature = sign_hash(&from.secret_key, &tx_hash);
    unsigned
}

fn build_block(number: u64, transactions: &[SignedTransaction], sequencer: &Account) -> Vec<u8> {
    let mut block = Vec::with_capacity(4 + 8 + 2 + transactions.len() * 185 + 64);
    block.extend_from_slice(&BLOCK_BLUEPRINT_MAGIC);
    block.extend_from_slice(&number.to_le_bytes());
    block.extend_from_slice(&(transactions.len() as u16).to_le_bytes());
    for transaction in transactions {
        encode_transaction(transaction, &mut block);
    }
    let block_hash = block_hash(number, transactions);
    let signature = sign_hash(&sequencer.secret_key, &block_hash);
    block.extend_from_slice(&signature);
    block
}

fn chunk_block(block_number: u64, block: &[u8]) -> Vec<Vec<u8>> {
    let max_chunk_payload = MAX_INPUT_MESSAGE_SIZE
        .checked_sub(EXTERNAL_FRAME_SIZE)
        .and_then(|size| size.checked_sub(BLOCK_CHUNK_HEADER_SIZE))
        .expect("chunk header must fit within max input size");
    let chunk_count = block.len().div_ceil(max_chunk_payload);
    assert!(
        u16::try_from(chunk_count).is_ok(),
        "block {} requires too many chunks: {}",
        block_number,
        chunk_count
    );

    let mut chunks = Vec::with_capacity(chunk_count);
    for (chunk_index, payload) in block.chunks(max_chunk_payload).enumerate() {
        let mut chunk = Vec::with_capacity(BLOCK_CHUNK_HEADER_SIZE + payload.len());
        chunk.extend_from_slice(&BLOCK_CHUNK_MAGIC);
        chunk.extend_from_slice(&block_number.to_le_bytes());
        chunk.extend_from_slice(&(chunk_index as u16).to_le_bytes());
        chunk.extend_from_slice(&(chunk_count as u16).to_le_bytes());
        chunk.extend_from_slice(payload);
        chunks.push(chunk);
    }
    chunks
}

fn build_inbox(transactions: usize, block_frequency: usize, account_count: usize) -> InboxFile {
    build_inbox_with_state(
        transactions,
        block_frequency,
        account_count,
        1,
        vec![0; account_count.max(1)],
    )
}

fn build_inbox_with_state(
    transactions: usize,
    block_frequency: usize,
    account_count: usize,
    first_block_number: u64,
    mut nonces: Vec<u64>,
) -> InboxFile {
    let account_count = account_count.max(1);
    let block_frequency = block_frequency.max(1);
    let accounts: Vec<_> = (0..account_count).map(make_account).collect();
    let sequencer = accounts[0];
    nonces.resize(account_count, 0);
    let mut levels = Vec::new();
    let mut messages = Vec::new();

    let mut block_number = first_block_number;
    let mut start = 0usize;
    while start < transactions {
        let end = (start + block_frequency).min(transactions);
        let mut block_transactions = Vec::with_capacity(end - start);
        for tx_index in start..end {
            let sender_idx = tx_index % account_count;
            let recipient_idx = (sender_idx + 1) % account_count;
            let nonce = nonces[sender_idx];
            nonces[sender_idx] += 1;
            block_transactions.push(build_transaction(
                accounts[sender_idx],
                accounts[recipient_idx],
                1,
                nonce,
            ));
        }

        let block = build_block(block_number, &block_transactions, &sequencer);
        for chunk in chunk_block(block_number, &block) {
            messages.push(InboxMessageFile::External { external: chunk });
        }
        block_number += 1;
        start = end;
    }

    levels.push(messages);
    InboxFile(levels)
}

fn account_key_bytes(address: &[u8; 20]) -> [u8; 26] {
    let mut key = [0u8; 26];
    key[..ACCOUNT_KEY_PREFIX.len()].copy_from_slice(ACCOUNT_KEY_PREFIX);
    key[ACCOUNT_KEY_PREFIX.len()..].copy_from_slice(address);
    key
}

fn context_mapping_key() -> Vec<u8> {
    let mut key = Vec::with_capacity(KEYSPACE_INDEX_PREFIX.len() + CONTEXT_NAME.len());
    key.extend_from_slice(KEYSPACE_INDEX_PREFIX);
    key.extend_from_slice(CONTEXT_NAME.as_bytes());
    key
}

fn account_state_bytes(balance: u64, nonce: u64) -> [u8; 16] {
    let mut state = [0u8; 16];
    state[..8].copy_from_slice(&balance.to_le_bytes());
    state[8..].copy_from_slice(&nonce.to_le_bytes());
    state
}

fn ensure_registry_size(
    registry: &mut Registry<PersistenceLayer, Normal>,
    size: usize,
) -> Result<()> {
    while registry.len() < size {
        registry.resize_tick(registry.len() + 1)?;
    }
    Ok(())
}

fn prepare_context(durable_storage_dir: &Path, accounts: usize, initial_balance: u64, rebuild: bool) -> Result<()> {
    if rebuild && durable_storage_dir.exists() {
        fs::remove_dir_all(durable_storage_dir)?;
    }
    fs::create_dir_all(durable_storage_dir)?;

    let repo = DirectoryManager::new(durable_storage_dir)?;
    let mut registry = Registry::<PersistenceLayer, Normal>::new(repo)?;
    ensure_registry_size(&mut registry, 2)?;

    let mapping_key = Key::new(&context_mapping_key())?;
    registry
        .database_mut(0)?
        .set(mapping_key, Bytes::copy_from_slice(&1u64.to_le_bytes()))?;

    let context = registry.database_mut(1)?;
    context.set(
        Key::new(META_BOOTSTRAPPED_KEY)?,
        Bytes::copy_from_slice(&[1u8]),
    )?;
    context.set(
        Key::new(META_HEAD_KEY)?,
        Bytes::copy_from_slice(&0u64.to_le_bytes()),
    )?;

    let total_accounts = accounts.max(1);
    for index in 0..total_accounts {
        let account = make_account(index);
        let balance = if index == 0 {
            initial_balance.max(DEFAULT_INITIAL_BALANCE)
        } else {
            initial_balance
        };
        let key = Key::new(&account_key_bytes(&account.address))?;
        context.set(key, Bytes::copy_from_slice(&account_state_bytes(balance, 0)))?;

        let populated = index + 1;
        if populated == total_accounts
            || populated % PREPARE_CONTEXT_PROGRESS_INTERVAL == 0
        {
            let remaining = total_accounts - populated;
            let percent = (populated as f64 / total_accounts as f64) * 100.0;
            println!(
                "prepare-context progress: populated {} accounts, remaining {}, {:.2}%",
                populated, remaining, percent
            );
        }
    }

    let commit = registry.commit()?;
    write_registry_head(durable_storage_dir, commit)?;

    Ok(())
}

fn write_registry_head(dir: &Path, commit: CommitId) -> Result<()> {
    let head_path = dir.join(DURABLE_STORAGE_HEAD_FILE);
    fs::write(head_path, commit.hex_encode())?;
    Ok(())
}

fn read_registry_head(dir: &Path) -> Result<Option<CommitId>> {
    let head_path = dir.join(DURABLE_STORAGE_HEAD_FILE);
    if !head_path.exists() {
        return Ok(None);
    }

    let hex = fs::read_to_string(head_path)?;
    let bytes = hex::decode(hex.trim())?;
    let digest: [u8; 32] = bytes
        .try_into()
        .map_err(|_| "registry-head must contain a 32-byte hex digest")?;
    Ok(Some(CommitId::from(octez_riscv_data::hash::Hash::from(
        digest,
    ))))
}

fn read_database_value(
    database: &octez_riscv_durable_storage::database::Database<PersistenceLayer, Normal>,
    key: &Key,
) -> Result<Option<Vec<u8>>> {
    if !database.exists(key)? {
        return Ok(None);
    }

    let length = database.value_length(key)?;
    let value = database.read_bytes(key, 0, length)?;
    Ok(Some(value.as_ref().to_vec()))
}

fn decode_account_nonce(bytes: &[u8]) -> Result<u64> {
    let raw: [u8; 16] = bytes
        .try_into()
        .map_err(|_| "stored account state must be 16 bytes")?;
    Ok(u64::from_le_bytes(raw[8..].try_into().expect("slice has fixed length")))
}

fn read_existing_context_state(
    durable_storage_dir: &Path,
    account_count: usize,
) -> Result<Option<(u64, Vec<u64>)>> {
    let Some(commit_id) = read_registry_head(durable_storage_dir)? else {
        return Ok(None);
    };

    let repo = DirectoryManager::new(durable_storage_dir)?;
    let registry = Registry::<PersistenceLayer, Normal>::checkout(repo, commit_id)?;
    let context = registry.database(1)?;

    let head_key = Key::new(META_HEAD_KEY)?;
    let current_head = match read_database_value(context, &head_key)? {
        Some(bytes) => {
            let raw: [u8; 8] = bytes
                .try_into()
                .map_err(|_| "stored block head must be 8 bytes")?;
            u64::from_le_bytes(raw)
        }
        None => 0,
    };

    let mut nonces = Vec::with_capacity(account_count.max(1));
    for index in 0..account_count.max(1) {
        let account = make_account(index);
        let key = Key::new(&account_key_bytes(&account.address))?;
        let nonce = match read_database_value(context, &key)? {
            Some(bytes) => decode_account_nonce(&bytes)?,
            None => 0,
        };
        nonces.push(nonce);
    }

    Ok(Some((current_head + 1, nonces)))
}

fn default_repo_root() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..ROOT_TX_KERNEL_DIR_COMPONENTS {
        root.pop();
    }
    root
}

fn default_kernel_path() -> PathBuf {
    default_repo_root().join("kernels/tx-kernel/target/riscv64gc-unknown-linux-musl/release/riscv-tx-kernel")
}

fn default_sandbox_path() -> PathBuf {
    default_repo_root().join("target/debug/riscv-sandbox")
}

fn parse_benchmark_logs(stdout: &str) -> Result<BenchmarkOutcome> {
    let block_regex = Regex::new(
        r"^applied block (?P<block>\d+) with (?P<txs>\d+) txs \((?P<applied>\d+) applied\), state root \[(?P<root>.+)\]$",
    )?;
    let first_tx_regex =
        Regex::new(r"^first processed tx block=(?P<block>\d+) tx_index=(?P<tx>\d+) total_processed=0$")?;
    let last_tx_regex = Regex::new(
        r"^last processed tx block=(?P<block>\d+) tx_index=(?P<tx>\d+) total_processed=(?P<total>\d+)$",
    )?;
    let tx_sample_start_regex = Regex::new(
        r"^tx sample start block=(?P<block>\d+) tx_index=(?P<tx>\d+) total_processed=(?P<total>\d+)$",
    )?;
    let tx_sample_sig_regex = Regex::new(
        r"^tx sample signature verified block=(?P<block>\d+) tx_index=(?P<tx>\d+) total_processed=(?P<total>\d+)$",
    )?;
    let tx_sample_end_regex = Regex::new(
        r"^tx sample complete block=(?P<block>\d+) tx_index=(?P<tx>\d+) total_processed=(?P<total>\d+)$",
    )?;
    let finalization_start_regex = Regex::new(
        r"^block finalization start block=(?P<block>\d+) total_processed=(?P<total>\d+)$",
    )?;
    let mut blocks = 0usize;
    let mut txs = 0usize;
    let mut applied = 0usize;
    let mut first_block = None;
    let mut last_block = None;
    let mut first_tx = None;
    let mut last_tx = None;
    let mut current_sample_start = None;
    let mut current_sample_sig = None;
    let mut sampled_signature_window = Duration::ZERO;
    let mut sampled_state_window = Duration::ZERO;
    let mut sampled_tx_count = 0usize;
    let mut finalization_start = None;
    let mut finalization_window = Duration::ZERO;

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let log: LogLine = serde_json::from_str(line)?;
        if let Some(captures) = block_regex.captures(&log.message) {
            let elapsed = log.elapsed.as_duration();
            first_block.get_or_insert(elapsed);
            last_block = Some(elapsed);
            blocks += 1;
            txs += captures["txs"].parse::<usize>()?;
            applied += captures["applied"].parse::<usize>()?;
            if let Some(start) = finalization_start.take() {
                if let Some(duration) = elapsed.checked_sub(start) {
                    finalization_window += duration;
                }
            }
        } else if first_tx_regex.is_match(&log.message) {
            first_tx.get_or_insert(log.elapsed.as_duration());
        } else if last_tx_regex.is_match(&log.message) {
            last_tx = Some(log.elapsed.as_duration());
        } else if tx_sample_start_regex.is_match(&log.message) {
            current_sample_start = Some(log.elapsed.as_duration());
            current_sample_sig = None;
        } else if tx_sample_sig_regex.is_match(&log.message) {
            if let Some(start) = current_sample_start {
                let elapsed = log.elapsed.as_duration();
                if let Some(duration) = elapsed.checked_sub(start) {
                    sampled_signature_window += duration;
                }
                current_sample_sig = Some(elapsed);
            }
        } else if tx_sample_end_regex.is_match(&log.message) {
            if let Some(sig) = current_sample_sig {
                let elapsed = log.elapsed.as_duration();
                if let Some(duration) = elapsed.checked_sub(sig) {
                    sampled_state_window += duration;
                }
                sampled_tx_count += 1;
            }
            current_sample_start = None;
            current_sample_sig = None;
        } else if finalization_start_regex.is_match(&log.message) {
            finalization_start = Some(log.elapsed.as_duration());
        }
    }

    let block_window = first_block
        .zip(last_block)
        .and_then(|(start, end)| end.checked_sub(start));
    let tx_window = first_tx.zip(last_tx).and_then(|(start, end)| end.checked_sub(start));
    Ok(BenchmarkOutcome {
        wall_duration: Duration::ZERO,
        block_count: blocks,
        tx_count: txs,
        applied_count: applied,
        block_window,
        tx_window,
        sampled_signature_window: (sampled_tx_count > 0).then_some(sampled_signature_window),
        sampled_state_window: (sampled_tx_count > 0).then_some(sampled_state_window),
        sampled_tx_count,
        finalization_window: (blocks > 0).then_some(finalization_window),
    })
}

fn write_inbox(path: &Path, inbox: &InboxFile) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(inbox)?)?;
    Ok(())
}

fn run_benchmark(
    transactions: usize,
    block_frequency: usize,
    accounts: usize,
    durable_storage_dir: PathBuf,
    initial_balance: u64,
    rebuild_context: bool,
    inbox_file: Option<PathBuf>,
    kernel: Option<PathBuf>,
    sandbox: Option<PathBuf>,
) -> Result<()> {
    if rebuild_context || !durable_storage_dir.exists() {
        prepare_context(&durable_storage_dir, accounts, initial_balance, rebuild_context)?;
    }

    let inbox_path =
        inbox_file.unwrap_or_else(|| std::env::temp_dir().join("tx-kernel-benchmark-inbox.json"));
    let inbox = match read_existing_context_state(&durable_storage_dir, accounts)? {
        Some((first_block_number, nonces)) => build_inbox_with_state(
            transactions,
            block_frequency,
            accounts,
            first_block_number,
            nonces,
        ),
        None => build_inbox(transactions, block_frequency, accounts),
    };
    write_inbox(&inbox_path, &inbox)?;

    let sandbox = sandbox.unwrap_or_else(default_sandbox_path);
    let kernel = kernel.unwrap_or_else(default_kernel_path);

    let start = Instant::now();
    let output = Command::new(&sandbox)
        .arg("run")
        .arg("--input")
        .arg(&kernel)
        .arg("--inbox-file")
        .arg(&inbox_path)
        .arg("--durable-storage-dir")
        .arg(&durable_storage_dir)
        .arg("--timings")
        .output()?;
    let wall_duration = start.elapsed();

    if !output.status.success() {
        return Err(format!(
            "sandbox run failed with status {}:\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let stdout = String::from_utf8(output.stdout)?;
    let mut outcome = parse_benchmark_logs(&stdout)?;
    outcome.wall_duration = wall_duration;

    println!("{stdout}");
    let processing_window = outcome.tx_window.or(outcome.block_window);
    let processing_tps = processing_window.and_then(|window| {
        if window.is_zero() {
            None
        } else {
            Some(outcome.tx_count as f64 / window.as_secs_f64())
        }
    });
    println!(
        "Processed {} blocks and {} transactions in {:.3}s ({:.2} TPS from processing timestamps)",
        outcome.block_count,
        outcome.tx_count,
        processing_window.unwrap_or(wall_duration).as_secs_f64(),
        processing_tps.unwrap_or(0.0)
    );
    println!(
        "Applied {} transactions, rejected {}",
        outcome.applied_count,
        outcome.tx_count.saturating_sub(outcome.applied_count)
    );
    println!(
        "Wall-clock runtime: {:.3}s ({:.2} TPS including startup/teardown)",
        wall_duration.as_secs_f64(),
        outcome.tx_count as f64 / wall_duration.as_secs_f64()
    );
    if let Some(window) = outcome.tx_window {
        println!(
            "Transaction processing window: {:.3}s ({:.2} TPS)",
            window.as_secs_f64(),
            if window.is_zero() {
                0.0
            } else {
                outcome.tx_count as f64 / window.as_secs_f64()
            }
        );
    }
    if outcome.sampled_tx_count > 0 {
        if let Some(window) = outcome.sampled_signature_window {
            println!(
                "Sampled signature verification: {:.3} ms/tx over {} samples",
                (window.as_secs_f64() * 1_000.0) / outcome.sampled_tx_count as f64,
                outcome.sampled_tx_count
            );
        }
        if let Some(window) = outcome.sampled_state_window {
            println!(
                "Sampled durable-state work: {:.3} ms/tx over {} samples",
                (window.as_secs_f64() * 1_000.0) / outcome.sampled_tx_count as f64,
                outcome.sampled_tx_count
            );
        }
    }
    if let Some(window) = outcome.finalization_window {
        println!(
            "Block finalization window: {:.3}s ({:.3} ms/block)",
            window.as_secs_f64(),
            if outcome.block_count == 0 {
                0.0
            } else {
                (window.as_secs_f64() * 1_000.0) / outcome.block_count as f64
            }
        );
    }
    if let Some(window) = outcome.block_window {
        println!(
            "Block processing window: {:.3}s ({:.2} TPS)",
            window.as_secs_f64(),
            if window.is_zero() {
                0.0
            } else {
                outcome.tx_count as f64 / window.as_secs_f64()
            }
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Generate {
            transactions,
            block_frequency,
            accounts,
            inbox_file,
        } => {
            let inbox = build_inbox(transactions, block_frequency, accounts);
            write_inbox(&inbox_file, &inbox)?;
            println!("Wrote {}", inbox_file.display());
        }
        Commands::PrepareContext {
            durable_storage_dir,
            accounts,
            initial_balance,
            rebuild,
        } => {
            prepare_context(&durable_storage_dir, accounts, initial_balance, rebuild)?;
            println!(
                "Prepared durable-storage context with {} accounts in {}",
                accounts.max(1),
                durable_storage_dir.display()
            );
        }
        Commands::Benchmark {
            transactions,
            block_frequency,
            durable_storage_dir,
            accounts,
            initial_balance,
            rebuild_context,
            inbox_file,
            kernel,
            sandbox,
        } => {
            run_benchmark(
                transactions,
                block_frequency,
                accounts,
                durable_storage_dir,
                initial_balance,
                rebuild_context,
                inbox_file,
                kernel,
                sandbox,
            )?;
        }
    }

    Ok(())
}
