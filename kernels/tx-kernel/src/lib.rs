// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod ethereum;
mod revm_backend;
mod world_state;

pub use ethereum::*;
pub use revm_backend::*;
pub use world_state::*;

pub const BLOCK_BLUEPRINT_MAGIC: [u8; 4] = *b"TXB1";
pub const BLOCK_CHUNK_MAGIC: [u8; 4] = *b"TXC1";
pub const ACCOUNT_KEY_PREFIX: &[u8] = b"/acct/";
pub const META_BOOTSTRAPPED_KEY: &[u8] = b"/meta/bootstrapped";
pub const META_HEAD_KEY: &[u8] = b"/meta/head";
pub const CONTEXT_NAME: &str = "/tx-kernel/context";
pub const BOOTSTRAP_BALANCE: u64 = 1_000_000;
pub const TX_TIMING_SAMPLE_INTERVAL: usize = 100;
pub const SEQUENCER_PUBLIC_KEY: [u8; 65] = [
    0x04, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
    0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17,
    0x98, 0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65, 0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08,
    0xa8, 0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19, 0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4,
    0xb8,
];

pub trait ContextStore: Clone {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String>;
    fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), String>;
    fn contains(&self, key: &[u8]) -> Result<bool, String>;
    fn hash(&self) -> Result<Vec<u8>, String>;
}

pub trait ContextLoader {
    type Context: ContextStore;

    fn load_or_create(&mut self, name: &str) -> Result<Self::Context, String>;
}

pub trait Crypto {
    fn keccak256(&self, bytes: &[u8]) -> Result<[u8; 32], String>;
    fn verify_signature(
        &self,
        public_key: &[u8; 65],
        signature: &[u8; 64],
        message_hash: &[u8; 32],
    ) -> bool;
}

/// Asynchronous keccak-256 interface backed by the PVM parallel crypto queue.
///
/// In Normal mode the hash computation starts in a background thread immediately,
/// so that `dequeue` can return with minimal or no blocking.
pub trait AsyncKeccak {
    /// Submit a keccak-256 hash request.  Returns immediately.
    fn enqueue(&self, bytes: &[u8]) -> Result<(), String>;

    /// Retrieve the oldest pending keccak-256 result.
    fn dequeue(&self) -> Result<[u8; 32], String>;
}

/// Asynchronous secp256k1 signature-verification interface.
///
/// In Normal mode verification runs in a background thread started by `enqueue`,
/// so that `dequeue` returns with minimal or no blocking.
pub trait AsyncSecp256k1 {
    /// Submit a secp256k1 verify request.  Returns immediately.
    fn secp256k1_enqueue(
        &self,
        public_key: &[u8; 65],
        signature: &[u8; 64],
        message_hash: &[u8; 32],
    ) -> Result<(), String>;

    /// Retrieve the oldest pending secp256k1 result.
    ///
    /// Returns `true` if the signature was valid, `false` if invalid.
    fn secp256k1_dequeue(&self) -> Result<bool, String>;
}

/// Asynchronous secp256k1 public-key recovery interface.
///
/// This is needed for Ethereum-style transactions, where the sender is derived
/// from the signature recovery result rather than provided explicitly.
pub trait AsyncSecp256k1Recover {
    /// Submit a secp256k1 recovery request. Returns immediately.
    fn secp256k1_recover_enqueue(
        &self,
        signature: &[u8; 64],
        recovery_id: u8,
        message_hash: &[u8; 32],
    ) -> Result<(), String>;

    /// Retrieve the oldest pending secp256k1 recovery result.
    ///
    /// Returns `Ok(Some(public_key))` on successful recovery, `Ok(None)` if the
    /// signature/recovery id pair is invalid, and `Err(...)` if the queue is
    /// empty or the host call failed.
    fn secp256k1_recover_dequeue(&self) -> Result<Option<[u8; 65]>, String>;
}

pub trait Logger {
    fn log(&mut self, message: &str);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccountState {
    pub balance: u64,
    pub nonce: u64,
}

impl AccountState {
    pub fn encode(self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.balance.to_le_bytes());
        bytes[8..].copy_from_slice(&self.nonce.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Self {
        let raw: [u8; 16] = bytes.try_into().expect("account state must be 16 bytes");
        let balance = u64::from_le_bytes(raw[..8].try_into().unwrap());
        let nonce = u64::from_le_bytes(raw[8..].try_into().unwrap());
        Self { balance, nonce }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SignedTransaction {
    pub from: [u8; 20],
    pub to: [u8; 20],
    pub amount: u64,
    pub nonce: u64,
    pub public_key: [u8; 65],
    pub signature: [u8; 64],
}

#[derive(Clone, Debug)]
pub struct BlockBlueprint {
    pub number: u64,
    pub transactions: Vec<SignedTransaction>,
    pub signature: [u8; 64],
}

#[derive(Clone, Copy, Debug)]
enum TxStatus {
    Applied = 1,
    Rejected = 0,
}

#[derive(Clone, Copy, Debug)]
struct TxReceipt {
    status: TxStatus,
}

struct Reader<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) offset: usize,
}

pub struct BlockChunkAccumulator {
    block_number: Option<u64>,
    expected_chunks: u16,
    next_chunk_index: u16,
    payload: Vec<u8>,
}

pub struct ChainKernel<C: ContextStore> {
    context: C,
    processed_transactions: usize,
    block_chunks: BlockChunkAccumulator,
}

impl BlockChunkAccumulator {
    pub fn new() -> Self {
        Self {
            block_number: None,
            expected_chunks: 0,
            next_chunk_index: 0,
            payload: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.block_number = None;
        self.expected_chunks = 0;
        self.next_chunk_index = 0;
        self.payload.clear();
    }

    fn push_chunk<'a>(&'a mut self, payload: &[u8]) -> Result<Option<(u64, &'a [u8])>, String> {
        let mut reader = Reader::new(payload);
        let magic = reader.take::<4>().map_err(|error| error.to_string())?;
        if magic != BLOCK_CHUNK_MAGIC {
            return Err("invalid block chunk magic".to_string());
        }

        let block_number = reader.read_u64().map_err(|error| error.to_string())?;
        let chunk_index = reader.read_u16().map_err(|error| error.to_string())?;
        let chunk_count = reader.read_u16().map_err(|error| error.to_string())?;
        let chunk_payload = &payload[reader.offset()..];

        if chunk_count == 0 {
            self.reset();
            return Err("block chunk count must be non-zero".to_string());
        }

        if chunk_index == 0 {
            self.reset();
            self.block_number = Some(block_number);
            self.expected_chunks = chunk_count;
        }

        if self.block_number != Some(block_number) {
            self.reset();
            return Err("received chunk for unexpected block number".to_string());
        }
        if self.expected_chunks != chunk_count {
            self.reset();
            return Err("received chunk with inconsistent chunk count".to_string());
        }
        if self.next_chunk_index != chunk_index {
            self.reset();
            return Err("received block chunks out of order".to_string());
        }

        self.payload.extend_from_slice(chunk_payload);
        self.next_chunk_index += 1;

        if self.next_chunk_index == self.expected_chunks {
            let completed_block = self.block_number.expect("block number must be present");
            let payload = &self.payload[..];
            Ok(Some((completed_block, payload)))
        } else {
            Ok(None)
        }
    }
}

impl Default for BlockChunkAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn take<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        let end = self.offset.checked_add(N).ok_or("offset overflow")?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or("unexpected end of input")?;
        self.offset = end;
        slice.try_into().map_err(|_| "invalid fixed-size field")
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16, &'static str> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    pub(crate) fn is_eof(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

impl<C: ContextStore> ChainKernel<C> {
    pub fn new<L: ContextLoader<Context = C>>(
        loader: &mut L,
        crypto: &impl Crypto,
    ) -> Result<Self, String> {
        let mut context = loader.load_or_create(CONTEXT_NAME)?;
        bootstrap_context(&mut context, crypto)?;
        Ok(Self {
            context,
            processed_transactions: 0,
            block_chunks: BlockChunkAccumulator::new(),
        })
    }

    pub fn handle_external_payload(
        &mut self,
        logger: &mut impl Logger,
        crypto: &(impl Crypto + AsyncKeccak + AsyncSecp256k1 + AsyncSecp256k1Recover),
        payload: &[u8],
    ) {
        if payload.starts_with(&BLOCK_CHUNK_MAGIC) {
            match self.block_chunks.push_chunk(payload) {
                Ok(Some((_block_number, block_payload))) => {
                    let complete_payload = block_payload.to_vec();
                    self.block_chunks.reset();
                    self.handle_block_payload(logger, crypto, &complete_payload);
                }
                Ok(None) => {}
                Err(error) => {
                    self.block_chunks.reset();
                    logger.log(&format!("rejected block chunk: {error}\n"));
                }
            }
        } else {
            self.block_chunks.reset();
            self.handle_block_payload(logger, crypto, payload);
        }
    }

    fn handle_block_payload(
        &mut self,
        logger: &mut impl Logger,
        crypto: &(impl Crypto + AsyncKeccak + AsyncSecp256k1 + AsyncSecp256k1Recover),
        payload: &[u8],
    ) {
        let result = if payload.starts_with(&ETH_BLOCK_BLUEPRINT_MAGIC) {
            apply_ethereum_block_blueprint(
                logger,
                crypto,
                &mut self.context,
                payload,
                &mut self.processed_transactions,
            )
        } else {
            apply_block_blueprint(
                logger,
                crypto,
                &mut self.context,
                payload,
                &mut self.processed_transactions,
            )
        };

        if let Err(error) = result {
            logger.log(&format!("rejected block blueprint: {error}\n"));
        }
    }
}

fn address_from_public_key(
    crypto: &impl Crypto,
    public_key: &[u8; 65],
) -> Result<[u8; 20], String> {
    if public_key[0] != 0x04 {
        return Err("only uncompressed secp256k1 public keys are supported".to_string());
    }

    let hash = crypto.keccak256(&public_key[1..])?;
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..]);
    Ok(address)
}

fn account_key_bytes(address: &[u8; 20]) -> [u8; 26] {
    let mut key = [0u8; 26];
    key[..ACCOUNT_KEY_PREFIX.len()].copy_from_slice(ACCOUNT_KEY_PREFIX);
    key[ACCOUNT_KEY_PREFIX.len()..].copy_from_slice(address);
    key
}

fn read_account(context: &impl ContextStore, address: &[u8; 20]) -> Result<AccountState, String> {
    match context.get(&account_key_bytes(address))? {
        Some(bytes) => Ok(AccountState::decode(&bytes)),
        None => Ok(AccountState {
            balance: 0,
            nonce: 0,
        }),
    }
}

fn write_account(
    context: &mut impl ContextStore,
    address: &[u8; 20],
    state: AccountState,
) -> Result<(), String> {
    context.set(&account_key_bytes(address), &state.encode())
}

fn read_block_head(context: &impl ContextStore) -> Result<u64, String> {
    match context.get(META_HEAD_KEY)? {
        Some(bytes) => Ok(u64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| "invalid stored block head".to_string())?,
        )),
        None => Ok(0),
    }
}

fn write_block_head(context: &mut impl ContextStore, head: u64) -> Result<(), String> {
    context.set(META_HEAD_KEY, &head.to_le_bytes())
}

fn bootstrap_context(context: &mut impl ContextStore, crypto: &impl Crypto) -> Result<(), String> {
    if context.contains(META_BOOTSTRAPPED_KEY)? {
        return Ok(());
    }

    let funded_address = address_from_public_key(crypto, &SEQUENCER_PUBLIC_KEY)?;
    write_account(
        context,
        &funded_address,
        AccountState {
            balance: BOOTSTRAP_BALANCE,
            nonce: 0,
        },
    )?;
    write_block_head(context, 0)?;

    let mut world_state = EvmWorldState::new(context);
    world_state.write_account(
        &funded_address,
        &EvmAccountState::with_balance_u64(BOOTSTRAP_BALANCE),
    )?;
    world_state.write_meta_u64(EVM_META_HEAD_KEY, 0)?;
    world_state.write_meta_u64(EVM_META_CHAIN_ID_KEY, DEFAULT_EVM_CHAIN_ID)?;
    world_state.write_meta_u64(EVM_META_BASE_FEE_KEY, DEFAULT_EVM_BASE_FEE)?;
    world_state.write_meta_u64(EVM_META_BLOCK_GAS_LIMIT_KEY, DEFAULT_EVM_BLOCK_GAS_LIMIT)?;
    world_state.write_meta_u64(EVM_META_TIMESTAMP_KEY, DEFAULT_EVM_TIMESTAMP)?;
    world_state.write_meta_bytes(EVM_META_SPEC_ID_KEY, DEFAULT_EVM_SPEC_ID)?;
    world_state.write_meta_bytes(EVM_META_BOOTSTRAPPED_KEY, &[1u8])?;

    context.set(META_BOOTSTRAPPED_KEY, &[1u8])?;
    Ok(())
}

fn encode_unsigned_transaction(transaction: &SignedTransaction) -> [u8; 56] {
    let mut bytes = [0u8; 56];
    bytes[..20].copy_from_slice(&transaction.from);
    bytes[20..40].copy_from_slice(&transaction.to);
    bytes[40..48].copy_from_slice(&transaction.amount.to_le_bytes());
    bytes[48..56].copy_from_slice(&transaction.nonce.to_le_bytes());
    bytes
}

fn transaction_hash(
    crypto: &impl Crypto,
    transaction: &SignedTransaction,
) -> Result<[u8; 32], String> {
    crypto.keccak256(&encode_unsigned_transaction(transaction))
}

fn block_hash(
    crypto: &impl Crypto,
    block_number: u64,
    transactions: &[SignedTransaction],
) -> Result<[u8; 32], String> {
    let mut header = [0u8; 14];
    header[..4].copy_from_slice(&BLOCK_BLUEPRINT_MAGIC);
    header[4..12].copy_from_slice(&block_number.to_le_bytes());
    header[12..14].copy_from_slice(&(transactions.len() as u16).to_le_bytes());

    let mut acc = crypto.keccak256(&header)?;
    for transaction in transactions {
        let tx_hash = transaction_hash(crypto, transaction)?;
        let mut bytes = [0u8; 64];
        bytes[..32].copy_from_slice(&acc);
        bytes[32..].copy_from_slice(&tx_hash);
        acc = crypto.keccak256(&bytes)?;
    }

    Ok(acc)
}

fn validate_transaction_signature(
    crypto: &impl Crypto,
    transaction: &SignedTransaction,
) -> Result<(), String> {
    let sender = address_from_public_key(crypto, &transaction.public_key)?;
    validate_transaction_signature_with_sender(crypto, transaction, &sender)
}

/// Like [`validate_transaction_signature`] but accepts a pre-computed sender address,
/// avoiding the redundant keccak call in the parallel-crypto two-pass path.
fn validate_transaction_signature_with_sender(
    crypto: &impl Crypto,
    transaction: &SignedTransaction,
    sender: &[u8; 20],
) -> Result<(), String> {
    if sender != &transaction.from {
        return Err("transaction sender does not match public key".to_string());
    }

    let hash = transaction_hash(crypto, transaction)?;
    if !crypto.verify_signature(&transaction.public_key, &transaction.signature, &hash) {
        return Err("invalid transaction signature".to_string());
    }

    Ok(())
}

fn apply_valid_transaction(
    context: &mut impl ContextStore,
    transaction: &SignedTransaction,
) -> Result<TxReceipt, String> {
    let mut sender = read_account(context, &transaction.from)?;
    let mut recipient = read_account(context, &transaction.to)?;

    let accepted = sender.nonce == transaction.nonce
        && sender.balance >= transaction.amount
        && recipient.balance.checked_add(transaction.amount).is_some();

    if accepted {
        sender.balance -= transaction.amount;
        sender.nonce += 1;
        recipient.balance += transaction.amount;
        write_account(context, &transaction.from, sender)?;
        write_account(context, &transaction.to, recipient)?;
    }

    Ok(TxReceipt {
        status: if accepted {
            TxStatus::Applied
        } else {
            TxStatus::Rejected
        },
    })
}

pub fn parse_block_blueprint(bytes: &[u8]) -> Result<BlockBlueprint, &'static str> {
    let mut reader = Reader::new(bytes);
    let magic = reader.take::<4>()?;
    if magic != BLOCK_BLUEPRINT_MAGIC {
        return Err("unexpected block blueprint magic");
    }

    let number = reader.read_u64()?;
    let tx_count = reader.read_u16()? as usize;

    let mut transactions = Vec::with_capacity(tx_count);
    for _ in 0..tx_count {
        transactions.push(SignedTransaction {
            from: reader.take()?,
            to: reader.take()?,
            amount: reader.read_u64()?,
            nonce: reader.read_u64()?,
            public_key: reader.take()?,
            signature: reader.take()?,
        });
    }

    let signature = reader.take::<64>()?;
    if !reader.is_eof() {
        return Err("unexpected trailing bytes in block blueprint");
    }

    Ok(BlockBlueprint {
        number,
        transactions,
        signature,
    })
}

fn verify_block_signature(crypto: &impl Crypto, block: &BlockBlueprint) -> Result<(), String> {
    let digest = block_hash(crypto, block.number, &block.transactions)?;
    if crypto.verify_signature(&SEQUENCER_PUBLIC_KEY, &block.signature, &digest) {
        Ok(())
    } else {
        Err("invalid sequencer signature".to_string())
    }
}

fn apply_block_blueprint(
    logger: &mut impl Logger,
    crypto: &(impl Crypto + AsyncKeccak + AsyncSecp256k1),
    context: &mut impl ContextStore,
    payload: &[u8],
    processed_transactions: &mut usize,
) -> Result<(), String> {
    let block = parse_block_blueprint(payload).map_err(str::to_string)?;
    let n = block.transactions.len();

    if *processed_transactions == 0 && !block.transactions.is_empty() {
        logger.log(&format!(
            "first processed tx block={} tx_index=0 total_processed=0\n",
            block.number
        ));
    }

    // ── Phase 1: enqueue all keccak work upfront ──────────────────────────────
    //
    // Enqueue 2N keccak requests before any synchronous work so the background
    // thread has maximum lead time:
    //   – N hashes to derive sender addresses from public keys
    //   – N hashes of the unsigned transaction data (for signature verification)
    //
    for transaction in &block.transactions {
        // Address derivation: keccak(public_key[1..])
        crypto.enqueue(&transaction.public_key[1..])?;
    }
    for transaction in &block.transactions {
        // Transaction hash: keccak(encode_unsigned_transaction(tx))
        crypto.enqueue(&encode_unsigned_transaction(transaction))?;
    }

    // Verify the block signature synchronously while the background thread
    // processes the enqueued keccak requests.
    verify_block_signature(crypto, &block)?;

    let current_head = read_block_head(context)?;
    if block.number != current_head + 1 {
        return Err(format!(
            "unexpected block number {}, expected {}",
            block.number,
            current_head + 1
        ));
    }

    // ── Phase 2: dequeue keccak results; enqueue secp256k1 verifications ──────
    //
    // Dequeue N address hashes and N tx hashes.  For each transaction with a
    // valid sender address, enqueue a secp256k1 verification request so the
    // background thread can start those while phase 3 applies earlier results.
    //
    // `sender_ok[i]` is true iff address derivation succeeded AND sender matches.
    let mut sender_ok: Vec<bool> = Vec::with_capacity(n);

    // First collect all N address hashes.
    let mut addr_hashes: Vec<[u8; 32]> = Vec::with_capacity(n);
    for _ in 0..n {
        addr_hashes.push(crypto.dequeue()?);
    }
    // Then collect all N tx hashes and enqueue secp verifications.
    let mut tx_hashes: Vec<[u8; 32]> = Vec::with_capacity(n);
    for _ in 0..n {
        tx_hashes.push(crypto.dequeue()?);
    }

    for (transaction, (addr_hash, tx_hash)) in block
        .transactions
        .iter()
        .zip(addr_hashes.iter().zip(tx_hashes.iter()))
    {
        let valid_key = transaction.public_key[0] == 0x04;
        let sender_addr = if valid_key {
            let mut addr = [0u8; 20];
            addr.copy_from_slice(&addr_hash[12..]);
            addr
        } else {
            [0u8; 20]
        };
        let addr_matches = valid_key && sender_addr == transaction.from;
        sender_ok.push(addr_matches);

        if addr_matches {
            crypto.secp256k1_enqueue(&transaction.public_key, &transaction.signature, tx_hash)?;
        }
    }

    // ── Phase 3: dequeue secp256k1 results and apply transactions ─────────────
    let mut receipts = Vec::with_capacity(n);

    // secp dequeue index — only advance for transactions that had a valid sender.
    for (tx_index, (transaction, &addr_ok)) in
        block.transactions.iter().zip(sender_ok.iter()).enumerate()
    {
        // let sample_tx = tx_index % TX_TIMING_SAMPLE_INTERVAL == 0;
        // if sample_tx {
        //     logger.log(&format!(
        //         "tx sample start block={} tx_index={} total_processed={}\n",
        //         block.number, tx_index, *processed_transactions
        //     ));
        // }

        let signature_ok = addr_ok && crypto.secp256k1_dequeue()?;

        // if sample_tx {
        //     logger.log(&format!(
        //         "tx sample signature verified block={} tx_index={} total_processed={}\n",
        //         block.number, tx_index, *processed_transactions
        //     ));
        // }

        let receipt = if signature_ok {
            apply_valid_transaction(context, transaction)?
        } else {
            TxReceipt {
                status: TxStatus::Rejected,
            }
        };
        receipts.push(receipt);
        *processed_transactions += 1;

        // if sample_tx {
        //     logger.log(&format!(
        //         "tx sample complete block={} tx_index={} total_processed={}\n",
        //         block.number, tx_index, *processed_transactions
        //     ));
        // }
        if tx_index + 1 == n {
            logger.log(&format!(
                "last processed tx block={} tx_index={} total_processed={}\n",
                block.number, tx_index, *processed_transactions
            ));
        }
    }

    logger.log(&format!(
        "block finalization start block={} total_processed={}\n",
        block.number, *processed_transactions
    ));
    write_block_head(context, block.number)?;
    let state_root = context.hash()?;
    let applied = receipts
        .iter()
        .filter(|receipt| matches!(receipt.status, TxStatus::Applied))
        .count();
    logger.log(&format!(
        "applied block {} with {} txs ({} applied), state root {:02x?}\n",
        block.number,
        receipts.len(),
        applied,
        state_root
    ));

    Ok(())
}

fn verify_ethereum_block_signature(
    crypto: &impl Crypto,
    block: &EthereumBlockBlueprint,
) -> Result<(), String> {
    let digest = crypto.keccak256(&ethereum_block_preimage(block.number, &block.transactions))?;
    if crypto.verify_signature(&SEQUENCER_PUBLIC_KEY, &block.signature, &digest) {
        Ok(())
    } else {
        Err("invalid sequencer signature".to_string())
    }
}

fn apply_ethereum_block_blueprint(
    logger: &mut impl Logger,
    crypto: &(impl Crypto + AsyncKeccak + AsyncSecp256k1Recover),
    context: &mut impl ContextStore,
    payload: &[u8],
    processed_transactions: &mut usize,
) -> Result<(), String> {
    let block = parse_ethereum_block_blueprint(payload).map_err(str::to_string)?;
    let n = block.transactions.len();

    if *processed_transactions == 0 && !block.transactions.is_empty() {
        logger.log(&format!(
            "first processed tx block={} tx_index=0 total_processed=0\n",
            block.number
        ));
    }

    let mut transactions = Vec::with_capacity(n);
    for transaction_bytes in &block.transactions {
        transactions.push(EthereumTransaction::parse(transaction_bytes)?);
    }

    for transaction in &transactions {
        crypto.enqueue(&transaction.signing_payload())?;
    }

    verify_ethereum_block_signature(crypto, &block)?;

    let current_head = read_block_head(context)?;
    if block.number != current_head + 1 {
        return Err(format!(
            "unexpected block number {}, expected {}",
            block.number,
            current_head + 1
        ));
    }

    let mut sighashes = Vec::with_capacity(n);
    for _ in 0..n {
        sighashes.push(crypto.dequeue()?);
    }

    for (transaction, sighash) in transactions.iter().zip(sighashes.iter()) {
        match transaction {
            EthereumTransaction::Eip1559(tx) => {
                crypto.secp256k1_recover_enqueue(
                    &tx.signature(),
                    tx.signature_y_parity,
                    sighash,
                )?;
            }
        }
    }

    let mut recovered_public_keys = Vec::with_capacity(n);
    for _ in 0..n {
        recovered_public_keys.push(crypto.secp256k1_recover_dequeue()?);
    }

    let mut recovery_ok = Vec::with_capacity(n);
    for public_key in &recovered_public_keys {
        match public_key {
            Some(public_key) if public_key[0] == 0x04 => {
                crypto.enqueue(&public_key[1..])?;
                recovery_ok.push(true);
            }
            _ => recovery_ok.push(false),
        }
    }

    let expected_hashes = recovery_ok.iter().filter(|ok| **ok).count();
    let mut derived_addresses = Vec::with_capacity(expected_hashes);
    for _ in 0..expected_hashes {
        derived_addresses.push(crypto.dequeue()?);
    }

    let mut derived_index = 0usize;
    let mut valid_transactions = 0usize;
    for tx_index in 0..n {
        let valid = if recovery_ok[tx_index] {
            let hash = derived_addresses[derived_index];
            derived_index += 1;
            let mut address = [0u8; 20];
            address.copy_from_slice(&hash[12..]);
            logger.log(&format!(
                "ethereum tx prevalidated block={} tx_index={} sender={:02x?}\n",
                block.number, tx_index, address
            ));
            true
        } else {
            false
        };

        if valid {
            valid_transactions += 1;
        }
        *processed_transactions += 1;
        if tx_index + 1 == n {
            logger.log(&format!(
                "last processed tx block={} tx_index={} total_processed={}\n",
                block.number, tx_index, *processed_transactions
            ));
        }
    }

    logger.log(&format!(
        "block finalization start block={} total_processed={}\n",
        block.number, *processed_transactions
    ));
    write_block_head(context, block.number)?;
    let state_root = context.hash()?;
    logger.log(&format!(
        "applied block {} with {} txs ({} applied), state root {:02x?}\n",
        block.number,
        block.transactions.len(),
        valid_transactions,
        state_root
    ));

    Ok(())
}
