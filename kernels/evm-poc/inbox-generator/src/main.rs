// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cell::RefCell;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
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
use riscv_evm_poc::AsyncKeccak;
use riscv_evm_poc::AsyncSecp256k1;
use riscv_evm_poc::AsyncSecp256k1Recover;
use riscv_evm_poc::CONTEXT_NAME;
use riscv_evm_poc::ChainKernel;
use riscv_evm_poc::ContextLoader;
use riscv_evm_poc::ContextStore;
use riscv_evm_poc::Crypto;
use riscv_evm_poc::DEFAULT_EVM_BASE_FEE;
use riscv_evm_poc::DEFAULT_EVM_BLOCK_GAS_LIMIT;
use riscv_evm_poc::DEFAULT_EVM_CHAIN_ID;
use riscv_evm_poc::DEFAULT_EVM_SPEC_ID;
use riscv_evm_poc::DEFAULT_EVM_TIMESTAMP;
use riscv_evm_poc::EVM_META_BASE_FEE_KEY;
use riscv_evm_poc::EVM_META_BLOCK_GAS_LIMIT_KEY;
use riscv_evm_poc::EVM_META_BOOTSTRAPPED_KEY;
use riscv_evm_poc::EVM_META_CHAIN_ID_KEY;
use riscv_evm_poc::EVM_META_HEAD_KEY;
use riscv_evm_poc::EVM_META_SPEC_ID_KEY;
use riscv_evm_poc::EVM_META_TIMESTAMP_KEY;
use riscv_evm_poc::Eip1559Transaction;
use riscv_evm_poc::Logger;
use riscv_evm_poc::META_HEAD_KEY as LEGACY_META_HEAD_KEY;
use riscv_evm_poc::account_nonce_key;
use riscv_evm_poc::build_ethereum_block_blueprint;
use riscv_evm_poc::ethereum_block_hash_header;
use riscv_evm_poc::u64_to_be_u256;
use serde::Deserialize;
use serde::Serialize;
use sha3::Digest;
use sha3::Keccak256;

const BLOCK_CHUNK_MAGIC: [u8; 4] = *b"TXC1";
const KEYSPACE_INDEX_PREFIX: &[u8] = b"/keyspaces/";
const KEYSPACE_INDEX_DATABASE: usize = 0;
const DEFAULT_INITIAL_BALANCE: u64 = 10_000_000_000_000_000_000;
const DURABLE_STORAGE_HEAD_FILE: &str = "registry-head";
const LEGACY_META_BOOTSTRAPPED_KEY: &[u8] = b"/meta/bootstrapped";
const PREPARE_CONTEXT_PROGRESS_INTERVAL: usize = 1_000;
const ROOT_EVM_POC_DIR_COMPONENTS: usize = 3;
const BLOCK_CHUNK_HEADER_SIZE: usize = 4 + 8 + 2 + 2;
const MAX_INPUT_MESSAGE_SIZE: usize = 4096;
const EXTERNAL_FRAME_SIZE: usize = 21;
// ERC-20 benchmark contract init code (22 bytes).
//
// A 12-byte constructor CODECOPYs the 10-byte runtime into memory and RETURNs it.
// Runtime: `600160005260206000f3` — a stub that always returns 1 (32 bytes).
// This exercises the contract-call path without real ERC-20 storage logic.
const ERC20_INIT_BYTECODE_HEX: &str = "600a600c600039600a6000f3600160005260206000f3";
const ERC20_MINT_SELECTOR: [u8; 4] = [0x42, 0x96, 0x6c, 0x68];
const ERC20_TRANSFER_SELECTOR: [u8; 4] = [0xa9, 0x05, 0x9c, 0xbb];
const ERC20_INITIAL_MINT: u64 = 1_000_000_000;
const ERC20_DEPLOY_GAS_LIMIT: u64 = 300_000;
const ERC20_TRANSFER_GAS_LIMIT: u64 = 120_000;
const ERC20_MINT_GAS_LIMIT: u64 = 200_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchmarkScenario {
    EthTransfer,
    Erc20,
}

#[derive(Parser)]
#[command(long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Generate a evm-poc inbox.json file")]
    Generate {
        #[arg(long)]
        transactions: usize,
        #[arg(long, default_value_t = 100)]
        block_frequency: usize,
        #[arg(long, default_value_t = 1024)]
        accounts: usize,
        #[arg(long, value_enum, default_value_t = BenchmarkScenario::EthTransfer)]
        scenario: BenchmarkScenario,
        #[arg(long, default_value = "evm-poc-inbox.json")]
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
        #[arg(long, value_enum, default_value_t = BenchmarkScenario::EthTransfer)]
        scenario: BenchmarkScenario,
        #[arg(long, default_value_t = false)]
        rebuild: bool,
    },
    #[command(about = "Run the evm-poc benchmark in the sandbox and report TPS")]
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
        #[arg(long, value_enum, default_value_t = BenchmarkScenario::EthTransfer)]
        scenario: BenchmarkScenario,
        #[arg(long, default_value_t = false)]
        rebuild_context: bool,
        #[arg(long)]
        inbox_file: Option<PathBuf>,
        #[arg(long)]
        kernel: Option<PathBuf>,
        #[arg(long)]
        sandbox: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        native: bool,
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
    address: [u8; 20],
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

#[derive(Serialize)]
struct NativeLogLine {
    elapsed: ElapsedLog,
    message: String,
}

#[derive(Serialize)]
struct ElapsedLog {
    secs: u64,
    nanos: u32,
}

#[derive(Clone)]
struct NativeKeySpace {
    index: usize,
    registry: Rc<RefCell<Registry<PersistenceLayer, Normal>>>,
}

#[derive(Default)]
struct NativeKeySpaceLoader {
    registry: Option<Rc<RefCell<Registry<PersistenceLayer, Normal>>>>,
}

struct NativeCrypto;

std::thread_local! {
    static KECCAK_QUEUE: std::cell::RefCell<std::collections::VecDeque<[u8; 32]>> =
        std::cell::RefCell::new(std::collections::VecDeque::new());
    static SECP_RECOVER_QUEUE: std::cell::RefCell<std::collections::VecDeque<Option<[u8; 65]>>> =
        std::cell::RefCell::new(std::collections::VecDeque::new());
}

struct NativeLogger {
    start: Instant,
    lines: String,
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
    let address = address_from_public_key(&PublicKey::from_secret_key(&secret_key).serialize());
    Account {
        secret_key,
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

fn build_eip1559_transaction(
    from: Account,
    to: Option<[u8; 20]>,
    value: u64,
    nonce: u64,
    chain_id: u64,
    gas_limit: u64,
    data: Vec<u8>,
) -> Vec<u8> {
    let tx = Eip1559Transaction {
        chain_id,
        nonce,
        max_priority_fee_per_gas: 1_000_000_000,
        max_fee_per_gas: 1_000_000_000,
        gas_limit,
        to,
        value: u64_to_be_u256(value),
        data,
        access_list: vec![],
        signature_y_parity: 0,
        signature_r: [0u8; 32],
        signature_s: [0u8; 32],
    };

    let sighash: [u8; 32] = keccak256(&tx.signing_payload());
    let message = Message::parse(&sighash);
    let (signature, recovery_id) = libsecp256k1::sign(&message, &from.secret_key);
    let compact = signature.serialize();

    let mut signed = tx;
    signed.signature_y_parity = Into::<u8>::into(recovery_id);
    signed.signature_r.copy_from_slice(&compact[..32]);
    signed.signature_s.copy_from_slice(&compact[32..]);
    signed.encode()
}

fn build_ethereum_block(number: u64, transactions: &[Vec<u8>], sequencer: &Account) -> Vec<u8> {
    // Chained block hash: h = keccak(header), then for each tx: h = keccak(h || keccak(tx_bytes))
    // This keeps every individual keccak input small so it fits within the PVM keccak size limit.
    let header = ethereum_block_hash_header(number, transactions.len());
    let mut h = keccak256(&header);
    for tx in transactions {
        let tx_hash = keccak256(tx);
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&h);
        buf[32..].copy_from_slice(&tx_hash);
        h = keccak256(&buf);
    }
    let signature = sign_hash(&sequencer.secret_key, &h);
    build_ethereum_block_blueprint(number, transactions, &signature)
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

fn build_inbox(
    scenario: BenchmarkScenario,
    transactions: usize,
    block_frequency: usize,
    account_count: usize,
) -> InboxFile {
    build_inbox_with_state(
        scenario,
        transactions,
        block_frequency,
        account_count,
        1,
        vec![0; account_count.max(1)],
    )
}

fn build_eth_transfer_transactions(
    accounts: &[Account],
    transactions: usize,
    nonces: &mut [u64],
) -> Vec<Vec<u8>> {
    let account_count = accounts.len();
    let mut block_transactions = Vec::with_capacity(transactions);
    for tx_index in 0..transactions {
        let sender_idx = tx_index % account_count;
        let recipient_idx = (sender_idx + 1) % account_count;
        let nonce = nonces[sender_idx];
        nonces[sender_idx] += 1;
        block_transactions.push(build_eip1559_transaction(
            accounts[sender_idx],
            Some(accounts[recipient_idx].address),
            1,
            nonce,
            DEFAULT_EVM_CHAIN_ID,
            21_000,
            vec![],
        ));
    }
    block_transactions
}

fn build_inbox_with_state(
    scenario: BenchmarkScenario,
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
    let all_transactions = match scenario {
        BenchmarkScenario::EthTransfer => {
            build_eth_transfer_transactions(&accounts, transactions, &mut nonces)
        }
        BenchmarkScenario::Erc20 => build_erc20_transactions(&accounts, transactions, &mut nonces),
    };
    let mut levels = Vec::new();
    let mut messages = Vec::new();

    let mut block_number = first_block_number;
    let mut start = 0usize;
    while start < all_transactions.len() {
        let end = (start + block_frequency).min(all_transactions.len());
        let block = build_ethereum_block(block_number, &all_transactions[start..end], &sequencer);
        for chunk in chunk_block(block_number, &block) {
            messages.push(InboxMessageFile::External { external: chunk });
        }
        block_number += 1;
        start = end;
    }

    levels.push(messages);
    InboxFile(levels)
}

fn erc20_init_bytecode() -> Vec<u8> {
    hex::decode(ERC20_INIT_BYTECODE_HEX).expect("embedded ERC-20 init code must be valid hex")
}

/// Compute the address where CREATE will deploy a contract.
///
/// Ethereum CREATE address: `keccak256(RLP([sender, nonce]))[12..]`
fn compute_create_address(deployer: &[u8; 20], nonce: u64) -> [u8; 20] {
    let mut stream = rlp::RlpStream::new_list(2);
    stream.append(&deployer.as_slice());
    stream.append(&nonce);
    let hash = keccak256(&stream.out());
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..]);
    address
}

fn erc20_mint_call_data(amount: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(4 + 32);
    data.extend_from_slice(&ERC20_MINT_SELECTOR);
    data.extend_from_slice(&u64_to_be_u256(amount));
    data
}

fn erc20_transfer_call_data(recipient: [u8; 20], amount: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(4 + 32 + 32);
    data.extend_from_slice(&ERC20_TRANSFER_SELECTOR);
    data.extend_from_slice(&[0u8; 12]);
    data.extend_from_slice(&recipient);
    data.extend_from_slice(&u64_to_be_u256(amount));
    data
}

fn build_erc20_transactions(
    accounts: &[Account],
    transfers: usize,
    nonces: &mut [u64],
) -> Vec<Vec<u8>> {
    let deployer = accounts[0];
    let account_count = accounts.len();
    let deploy_nonce = nonces[0];
    let contract_address = compute_create_address(&deployer.address, deploy_nonce);

    let mut transactions = Vec::with_capacity(transfers + 2);

    // 1. Deploy the ERC-20 contract via CREATE (to = None).
    transactions.push(build_eip1559_transaction(
        deployer,
        None,
        0,
        deploy_nonce,
        DEFAULT_EVM_CHAIN_ID,
        ERC20_DEPLOY_GAS_LIMIT,
        erc20_init_bytecode(),
    ));
    nonces[0] += 1;

    // 2. Mint initial supply into the deployer's balance.
    transactions.push(build_eip1559_transaction(
        deployer,
        Some(contract_address),
        0,
        nonces[0],
        DEFAULT_EVM_CHAIN_ID,
        ERC20_MINT_GAS_LIMIT,
        erc20_mint_call_data(ERC20_INITIAL_MINT),
    ));
    nonces[0] += 1;

    // 3. Transfer txs — spread gas payment across all available senders.
    for transfer_index in 0..transfers {
        let sender_index = if account_count == 1 {
            0
        } else {
            transfer_index % account_count
        };
        let sender = accounts[sender_index];
        let recipient = if account_count == 1 {
            sender.address
        } else {
            accounts[(sender_index + 1) % account_count].address
        };
        transactions.push(build_eip1559_transaction(
            sender,
            Some(contract_address),
            0,
            nonces[sender_index],
            DEFAULT_EVM_CHAIN_ID,
            ERC20_TRANSFER_GAS_LIMIT,
            erc20_transfer_call_data(recipient, 1),
        ));
        nonces[sender_index] += 1;
    }

    transactions
}

fn legacy_account_key_bytes(address: &[u8; 20]) -> [u8; 26] {
    let mut key = [0u8; 26];
    key[..b"/acct/".len()].copy_from_slice(b"/acct/");
    key[b"/acct/".len()..].copy_from_slice(address);
    key
}

fn account_balance_key_bytes(address: &[u8; 20]) -> Vec<u8> {
    let mut key = Vec::with_capacity(b"/evm/accounts/".len() + address.len() + b"/balance".len());
    key.extend_from_slice(b"/evm/accounts/");
    key.extend_from_slice(address);
    key.extend_from_slice(b"/balance");
    key
}

fn account_code_key_bytes(address: &[u8; 20]) -> Vec<u8> {
    let mut key = Vec::with_capacity(b"/evm/accounts/".len() + address.len() + b"/code".len());
    key.extend_from_slice(b"/evm/accounts/");
    key.extend_from_slice(address);
    key.extend_from_slice(b"/code");
    key
}

fn account_code_hash_key_bytes(address: &[u8; 20]) -> Vec<u8> {
    let mut key = Vec::with_capacity(b"/evm/accounts/".len() + address.len() + b"/code_hash".len());
    key.extend_from_slice(b"/evm/accounts/");
    key.extend_from_slice(address);
    key.extend_from_slice(b"/code_hash");
    key
}

fn context_mapping_key() -> Vec<u8> {
    let mut key = Vec::with_capacity(KEYSPACE_INDEX_PREFIX.len() + CONTEXT_NAME.len());
    key.extend_from_slice(KEYSPACE_INDEX_PREFIX);
    key.extend_from_slice(CONTEXT_NAME.as_bytes());
    key
}

fn keyspace_index_key(name: &str) -> std::result::Result<Vec<u8>, String> {
    let mut key = Vec::with_capacity(KEYSPACE_INDEX_PREFIX.len() + name.len());
    key.extend_from_slice(KEYSPACE_INDEX_PREFIX);
    key.extend_from_slice(name.as_bytes());
    if key.len() > 256 {
        return Err("keyspace name exceeds durable key size limit".to_string());
    }
    Ok(key)
}

fn account_state_bytes(balance: u64, nonce: u64) -> [u8; 16] {
    let mut state = [0u8; 16];
    state[..8].copy_from_slice(&balance.to_le_bytes());
    state[8..].copy_from_slice(&nonce.to_le_bytes());
    state
}

fn evm_balance_bytes(balance: u64) -> [u8; 32] {
    u64_to_be_u256(balance)
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

fn prepare_context(
    durable_storage_dir: &Path,
    accounts: usize,
    initial_balance: u64,
    scenario: BenchmarkScenario,
    rebuild: bool,
) -> Result<()> {
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
        Key::new(LEGACY_META_BOOTSTRAPPED_KEY)?,
        Bytes::copy_from_slice(&[1u8]),
    )?;
    context.set(
        Key::new(LEGACY_META_HEAD_KEY)?,
        Bytes::copy_from_slice(&0u64.to_le_bytes()),
    )?;
    context.set(
        Key::new(EVM_META_BOOTSTRAPPED_KEY)?,
        Bytes::copy_from_slice(&[1u8]),
    )?;
    context.set(
        Key::new(EVM_META_HEAD_KEY)?,
        Bytes::copy_from_slice(&0u64.to_le_bytes()),
    )?;
    context.set(
        Key::new(EVM_META_CHAIN_ID_KEY)?,
        Bytes::copy_from_slice(&DEFAULT_EVM_CHAIN_ID.to_le_bytes()),
    )?;
    context.set(
        Key::new(EVM_META_BASE_FEE_KEY)?,
        Bytes::copy_from_slice(&DEFAULT_EVM_BASE_FEE.to_le_bytes()),
    )?;
    context.set(
        Key::new(EVM_META_BLOCK_GAS_LIMIT_KEY)?,
        Bytes::copy_from_slice(&DEFAULT_EVM_BLOCK_GAS_LIMIT.to_le_bytes()),
    )?;
    context.set(
        Key::new(EVM_META_TIMESTAMP_KEY)?,
        Bytes::copy_from_slice(&DEFAULT_EVM_TIMESTAMP.to_le_bytes()),
    )?;
    context.set(
        Key::new(EVM_META_SPEC_ID_KEY)?,
        Bytes::copy_from_slice(DEFAULT_EVM_SPEC_ID),
    )?;

    let total_accounts = accounts.max(1);
    for index in 0..total_accounts {
        let account = make_account(index);
        let balance = if index == 0 {
            initial_balance.max(DEFAULT_INITIAL_BALANCE)
        } else {
            initial_balance
        };
        let legacy_key = Key::new(&legacy_account_key_bytes(&account.address))?;
        context.set(
            legacy_key,
            Bytes::copy_from_slice(&account_state_bytes(balance, 0)),
        )?;

        let nonce_key = Key::new(&account_nonce_key(&account.address))?;
        context.set(nonce_key, Bytes::copy_from_slice(&0u64.to_le_bytes()))?;
        let balance_key = Key::new(&account_balance_key_bytes(&account.address))?;
        context.set(
            balance_key,
            Bytes::copy_from_slice(&evm_balance_bytes(balance)),
        )?;
        let code_hash_key = Key::new(&account_code_hash_key_bytes(&account.address))?;
        context.set(code_hash_key, Bytes::copy_from_slice(&[0u8; 32]))?;
        let code_key = Key::new(&account_code_key_bytes(&account.address))?;
        context.set(code_key, Bytes::copy_from_slice(&[]))?;

        let populated = index + 1;
        if populated == total_accounts || populated % PREPARE_CONTEXT_PROGRESS_INTERVAL == 0 {
            let remaining = total_accounts - populated;
            let percent = (populated as f64 / total_accounts as f64) * 100.0;
            println!(
                "prepare-context progress: populated {} accounts, remaining {}, {:.2}%",
                populated, remaining, percent
            );
        }
    }

    // ERC-20 scenario: the contract is deployed via a CREATE transaction in the inbox.
    // No contract bootstrapping is needed here.
    let _ = scenario;

    let commit = registry.commit()?;
    write_registry_head(durable_storage_dir, commit)?;

    Ok(())
}

fn write_registry_head(dir: &Path, commit: CommitId) -> Result<()> {
    let head_path = dir.join(DURABLE_STORAGE_HEAD_FILE);
    fs::write(head_path, commit.hex_encode())?;
    Ok(())
}

impl NativeKeySpaceLoader {
    fn open(durable_storage_dir: &Path) -> Result<Self> {
        let repo = DirectoryManager::new(durable_storage_dir)?;
        let registry = match read_registry_head(durable_storage_dir)? {
            Some(commit_id) => Registry::<PersistenceLayer, Normal>::checkout(repo, commit_id)?,
            None => Registry::<PersistenceLayer, Normal>::new(repo)?,
        };
        Ok(Self {
            registry: Some(Rc::new(RefCell::new(registry))),
        })
    }

    fn persist(&self, durable_storage_dir: &Path) -> Result<()> {
        let Some(registry) = &self.registry else {
            return Ok(());
        };
        let commit = registry.borrow_mut().commit()?;
        write_registry_head(durable_storage_dir, commit)
    }

    fn registry(&self) -> Rc<RefCell<Registry<PersistenceLayer, Normal>>> {
        self.registry
            .as_ref()
            .expect("native keyspace loader must be opened before use")
            .clone()
    }
}

impl ContextStore for NativeKeySpace {
    fn get(&self, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, String> {
        let key = Key::new(key).map_err(|error| error.to_string())?;
        let registry = self.registry.borrow();
        let database = registry
            .database(self.index)
            .map_err(|error| error.to_string())?;
        read_database_value(database, &key).map_err(|error| error.to_string())
    }

    fn set(&mut self, key: &[u8], value: &[u8]) -> std::result::Result<(), String> {
        let key = Key::new(key).map_err(|error| error.to_string())?;
        self.registry
            .borrow_mut()
            .database_mut(self.index)
            .map_err(|error| error.to_string())?
            .set(key, Bytes::copy_from_slice(value))
            .map_err(|error| error.to_string())
    }

    fn contains(&self, key: &[u8]) -> std::result::Result<bool, String> {
        let key = Key::new(key).map_err(|error| error.to_string())?;
        self.registry
            .borrow()
            .database(self.index)
            .map_err(|error| error.to_string())?
            .exists(&key)
            .map_err(|error| error.to_string())
    }

    fn hash(&self) -> std::result::Result<Vec<u8>, String> {
        self.registry
            .borrow()
            .database(self.index)
            .map_err(|error| error.to_string())?
            .hash()
            .map(|hash| hash.as_ref().to_vec())
            .map_err(|error| error.to_string())
    }
}

impl ContextLoader for NativeKeySpaceLoader {
    type Context = NativeKeySpace;

    fn load_or_create(&mut self, name: &str) -> std::result::Result<Self::Context, String> {
        let mapping_key = keyspace_index_key(name)?;
        let mapping_key = Key::new(&mapping_key).map_err(|error| error.to_string())?;
        let registry = self.registry();
        let mut registry_mut = registry.borrow_mut();
        let mut registry_len = registry_mut.len();
        if registry_len == 0 {
            registry_mut
                .resize_tick(1)
                .map_err(|error| error.to_string())?;
            registry_len = 1;
        }

        let index = if registry_mut
            .database(KEYSPACE_INDEX_DATABASE)
            .map_err(|error| error.to_string())?
            .exists(&mapping_key)
            .map_err(|error| error.to_string())?
        {
            let bytes = read_database_value(
                registry_mut
                    .database(KEYSPACE_INDEX_DATABASE)
                    .map_err(|error| error.to_string())?,
                &mapping_key,
            )
            .map_err(|error| error.to_string())?
            .expect("native mapping must exist after exists() check");
            decode_database_index(&bytes)
        } else {
            let new_index = registry_len;
            registry_mut
                .resize_tick(new_index + 1)
                .map_err(|error| error.to_string())?;
            registry_mut
                .database_mut(KEYSPACE_INDEX_DATABASE)
                .map_err(|error| error.to_string())?
                .set(
                    mapping_key,
                    Bytes::copy_from_slice(&(new_index as u64).to_le_bytes()),
                )
                .map_err(|error| error.to_string())?;
            new_index
        };
        drop(registry_mut);

        Ok(NativeKeySpace { index, registry })
    }
}

impl Crypto for NativeCrypto {
    fn keccak256(&self, bytes: &[u8]) -> std::result::Result<[u8; 32], String> {
        Ok(keccak256(bytes))
    }

    fn verify_signature(
        &self,
        public_key: &[u8; 65],
        signature: &[u8; 64],
        message_hash: &[u8; 32],
    ) -> bool {
        let message = Message::parse(message_hash);
        let signature = libsecp256k1::Signature::parse_standard(signature)
            .expect("benchmark signatures must be canonical");
        let public_key = PublicKey::parse(public_key).expect("benchmark public keys must be valid");
        libsecp256k1::verify(&message, &signature, &public_key)
    }
}

impl AsyncKeccak for NativeCrypto {
    fn enqueue(&self, bytes: &[u8]) -> std::result::Result<(), String> {
        KECCAK_QUEUE.with(|q| q.borrow_mut().push_back(keccak256(bytes)));
        Ok(())
    }

    fn dequeue(&self) -> std::result::Result<[u8; 32], String> {
        KECCAK_QUEUE
            .with(|q| q.borrow_mut().pop_front())
            .ok_or_else(|| "keccak queue is empty".to_string())
    }
}

std::thread_local! {
    static SECP_QUEUE: std::cell::RefCell<std::collections::VecDeque<bool>> =
        std::cell::RefCell::new(std::collections::VecDeque::new());
}

impl AsyncSecp256k1 for NativeCrypto {
    fn secp256k1_enqueue(
        &self,
        public_key: &[u8; 65],
        signature: &[u8; 64],
        message_hash: &[u8; 32],
    ) -> std::result::Result<(), String> {
        let message = libsecp256k1::Message::parse(message_hash);
        let sig = libsecp256k1::Signature::parse_standard(signature)
            .expect("benchmark signatures must be canonical");
        let pk = PublicKey::parse(public_key).expect("benchmark public keys must be valid");
        let valid = libsecp256k1::verify(&message, &sig, &pk);
        SECP_QUEUE.with(|q| q.borrow_mut().push_back(valid));
        Ok(())
    }

    fn secp256k1_dequeue(&self) -> std::result::Result<bool, String> {
        SECP_QUEUE
            .with(|q| q.borrow_mut().pop_front())
            .ok_or_else(|| "secp256k1 queue is empty".to_string())
    }
}

impl AsyncSecp256k1Recover for NativeCrypto {
    fn secp256k1_recover_enqueue(
        &self,
        signature: &[u8; 64],
        recovery_id: u8,
        message_hash: &[u8; 32],
    ) -> std::result::Result<(), String> {
        let message = libsecp256k1::Message::parse(message_hash);
        let sig = libsecp256k1::Signature::parse_standard(signature)
            .expect("benchmark signatures must be canonical");
        let recovery_id = libsecp256k1::RecoveryId::parse(recovery_id)
            .map_err(|_| "invalid secp256k1 recovery id".to_string())?;
        let recovered = libsecp256k1::recover(&message, &sig, &recovery_id)
            .ok()
            .map(|public_key| public_key.serialize());
        SECP_RECOVER_QUEUE.with(|q| q.borrow_mut().push_back(recovered));
        Ok(())
    }

    fn secp256k1_recover_dequeue(&self) -> std::result::Result<Option<[u8; 65]>, String> {
        SECP_RECOVER_QUEUE
            .with(|q| q.borrow_mut().pop_front())
            .ok_or_else(|| "secp256k1 recover queue is empty".to_string())
    }
}

impl NativeLogger {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            lines: String::new(),
        }
    }

    fn finish(self) -> String {
        self.lines
    }
}

impl Logger for NativeLogger {
    fn log(&mut self, message: &str) {
        let elapsed = self.start.elapsed();
        let line = NativeLogLine {
            elapsed: ElapsedLog {
                secs: elapsed.as_secs(),
                nanos: elapsed.subsec_nanos(),
            },
            message: message.trim_end_matches('\n').to_string(),
        };
        self.lines
            .push_str(&serde_json::to_string(&line).expect("native log line should serialize"));
        self.lines.push('\n');
    }
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
    let raw: [u8; 8] = bytes
        .try_into()
        .map_err(|_| "stored account nonce must be 8 bytes")?;
    Ok(u64::from_le_bytes(raw))
}

fn decode_database_index(bytes: &[u8]) -> usize {
    let raw: [u8; 8] = bytes
        .try_into()
        .expect("persisted keyspace index must be encoded as u64");
    u64::from_le_bytes(raw) as usize
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

    let head_key = Key::new(EVM_META_HEAD_KEY)?;
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
        let key = Key::new(&account_nonce_key(&account.address))?;
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
    for _ in 0..ROOT_EVM_POC_DIR_COMPONENTS {
        root.pop();
    }
    root
}

fn default_kernel_path() -> PathBuf {
    default_repo_root()
        .join("kernels/evm-poc/target/riscv64gc-unknown-linux-musl/release/riscv-evm-poc")
}

fn default_sandbox_path() -> PathBuf {
    default_repo_root().join("target/release/riscv-sandbox")
}

fn parse_benchmark_logs(stdout: &str) -> Result<BenchmarkOutcome> {
    let block_regex = Regex::new(
        r"^applied block (?P<block>\d+) with (?P<txs>\d+) txs \((?P<applied>\d+) applied\), state root \[(?P<root>.+)\]$",
    )?;
    let first_tx_regex = Regex::new(
        r"^first processed tx block=(?P<block>\d+) tx_index=(?P<tx>\d+) total_processed=0$",
    )?;
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
    let tx_window = first_tx
        .zip(last_tx)
        .and_then(|(start, end)| end.checked_sub(start));
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

fn inbox_payloads(inbox: &InboxFile) -> Vec<Vec<u8>> {
    inbox
        .0
        .iter()
        .flat_map(|level| level.iter())
        .map(|message| match message {
            InboxMessageFile::External { external } => external.clone(),
        })
        .collect()
}

fn run_native_benchmark(
    inbox: &InboxFile,
    durable_storage_dir: &Path,
) -> Result<(String, Duration)> {
    let payloads = inbox_payloads(inbox);
    let mut loader = NativeKeySpaceLoader::open(durable_storage_dir)?;
    let crypto = NativeCrypto;
    let mut logger = NativeLogger::new();
    let mut kernel = ChainKernel::new(&mut loader, &crypto)?;

    let start = Instant::now();
    for payload in &payloads {
        kernel.handle_external_payload(&mut logger, &crypto, payload);
    }
    let wall_duration = start.elapsed();

    loader.persist(durable_storage_dir)?;
    Ok((logger.finish(), wall_duration))
}

fn run_benchmark(
    transactions: usize,
    block_frequency: usize,
    accounts: usize,
    durable_storage_dir: PathBuf,
    initial_balance: u64,
    scenario: BenchmarkScenario,
    rebuild_context: bool,
    inbox_file: Option<PathBuf>,
    kernel: Option<PathBuf>,
    sandbox: Option<PathBuf>,
    native: bool,
) -> Result<()> {
    if rebuild_context || !durable_storage_dir.exists() {
        prepare_context(
            &durable_storage_dir,
            accounts,
            initial_balance,
            scenario,
            rebuild_context,
        )?;
    }

    let inbox_path =
        inbox_file.unwrap_or_else(|| std::env::temp_dir().join("evm-poc-benchmark-inbox.json"));
    let inbox = match read_existing_context_state(&durable_storage_dir, accounts)? {
        Some((first_block_number, nonces)) => build_inbox_with_state(
            scenario,
            transactions,
            block_frequency,
            accounts,
            first_block_number,
            nonces,
        ),
        None => build_inbox(scenario, transactions, block_frequency, accounts),
    };
    write_inbox(&inbox_path, &inbox)?;

    let (stdout, wall_duration) = if native {
        run_native_benchmark(&inbox, &durable_storage_dir)?
    } else {
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

        (String::from_utf8(output.stdout)?, wall_duration)
    };
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
            scenario,
            inbox_file,
        } => {
            let inbox = build_inbox(scenario, transactions, block_frequency, accounts);
            write_inbox(&inbox_file, &inbox)?;
            println!("Wrote {}", inbox_file.display());
        }
        Commands::PrepareContext {
            durable_storage_dir,
            accounts,
            initial_balance,
            scenario,
            rebuild,
        } => {
            prepare_context(
                &durable_storage_dir,
                accounts,
                initial_balance,
                scenario,
                rebuild,
            )?;
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
            scenario,
            rebuild_context,
            inbox_file,
            kernel,
            sandbox,
            native,
        } => {
            run_benchmark(
                transactions,
                block_frequency,
                accounts,
                durable_storage_dir,
                initial_balance,
                scenario,
                rebuild_context,
                inbox_file,
                kernel,
                sandbox,
                native,
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a pre-built inbox against a prepared durable-storage dir and return the outcome.
    fn run_native_with_inbox(durable_dir: &Path, inbox: &InboxFile) -> Result<BenchmarkOutcome> {
        let (stdout, _) = run_native_benchmark(inbox, durable_dir)?;
        parse_benchmark_logs(&stdout)
    }

    /// Create a fresh (empty) temporary directory for a test.
    fn fresh_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("evm-poc-test-{name}"));
        if dir.exists() {
            fs::remove_dir_all(&dir).expect("failed to remove stale test directory");
        }
        dir
    }

    /// Wrap transactions in a single block and return an inbox ready for replay.
    fn make_inbox(block_number: u64, transactions: &[Vec<u8>]) -> InboxFile {
        let sequencer = make_account(0);
        let block = build_ethereum_block(block_number, transactions, &sequencer);
        let msgs = chunk_block(block_number, &block)
            .into_iter()
            .map(|c| InboxMessageFile::External { external: c })
            .collect();
        InboxFile(vec![msgs])
    }

    #[test]
    fn eth_transfer_applies() {
        let dir = fresh_test_dir("eth-transfer");
        prepare_context(
            &dir,
            2,
            DEFAULT_INITIAL_BALANCE,
            BenchmarkScenario::EthTransfer,
            false,
        )
        .expect("prepare_context failed");

        let accounts: Vec<_> = (0..2).map(make_account).collect();
        let mut nonces = vec![0u64; 2];
        let txs = build_eth_transfer_transactions(&accounts, 1, &mut nonces);
        let inbox = make_inbox(1, &txs);

        let outcome = run_native_with_inbox(&dir, &inbox).expect("run failed");
        assert_eq!(outcome.tx_count, 1);
        assert_eq!(outcome.applied_count, 1, "ETH transfer should be applied");
    }

    #[test]
    fn contract_creation_applies() {
        let dir = fresh_test_dir("contract-creation");
        prepare_context(
            &dir,
            1,
            DEFAULT_INITIAL_BALANCE,
            BenchmarkScenario::EthTransfer,
            false,
        )
        .expect("prepare_context failed");

        let deployer = make_account(0);
        let create_tx = build_eip1559_transaction(
            deployer,
            None, // CREATE
            0,
            0, // nonce 0
            DEFAULT_EVM_CHAIN_ID,
            ERC20_DEPLOY_GAS_LIMIT,
            erc20_init_bytecode(),
        );
        let inbox = make_inbox(1, &[create_tx]);

        let outcome = run_native_with_inbox(&dir, &inbox).expect("run failed");
        assert_eq!(outcome.tx_count, 1);
        assert_eq!(outcome.applied_count, 1, "contract creation should succeed");
    }

    #[test]
    fn erc20_deploy_and_call_applies() {
        let dir = fresh_test_dir("erc20-deploy-and-call");
        prepare_context(
            &dir,
            2,
            DEFAULT_INITIAL_BALANCE,
            BenchmarkScenario::EthTransfer,
            false,
        )
        .expect("prepare_context failed");

        let accounts: Vec<_> = (0..2).map(make_account).collect();
        let mut nonces = vec![0u64; 2];
        // deploy + mint + 1 transfer = 3 transactions
        let txs = build_erc20_transactions(&accounts, 1, &mut nonces);
        assert_eq!(txs.len(), 3);
        let inbox = make_inbox(1, &txs);

        let outcome = run_native_with_inbox(&dir, &inbox).expect("run failed");
        assert_eq!(outcome.tx_count, 3, "deploy + mint + 1 transfer");
        assert_eq!(
            outcome.applied_count, 3,
            "all three transactions should succeed"
        );
    }
}
