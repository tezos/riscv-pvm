// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod keyspace;
mod sbi_crypto;
mod sbi_durable;

use std::str::FromStr;

use keyspace::DurableKeySpace;
use keyspace::DurableKeySpaceLoader;
use keyspace::Key;
use keyspace::KeySpace;
use keyspace::KeySpaceLoader;
use keyspace::Name;
use tezos_smart_rollup::entrypoint;
use tezos_smart_rollup::inbox::InboxMessage;
use tezos_smart_rollup::inbox::InternalInboxMessage;
use tezos_smart_rollup::michelson::MichelsonUnit;
use tezos_smart_rollup::prelude::*;

const BLOCK_BLUEPRINT_MAGIC: [u8; 4] = *b"TXB1";
const BLOCK_CHUNK_MAGIC: [u8; 4] = *b"TXC1";
const ACCOUNT_KEY_PREFIX: &[u8] = b"/acct/";
const META_BOOTSTRAPPED_KEY: &Key = Key::from_static(b"/meta/bootstrapped");
const META_HEAD_KEY: &Key = Key::from_static(b"/meta/head");
const CONTEXT_NAME: &str = "/tx-kernel/context";
const BOOTSTRAP_BALANCE: u64 = 1_000_000;
const TX_TIMING_SAMPLE_INTERVAL: usize = 100;
const SEQUENCER_PUBLIC_KEY: [u8; 65] = [
    0x04, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
    0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17,
    0x98, 0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65, 0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08,
    0xa8, 0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19, 0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4,
    0xb8,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AccountState {
    balance: u64,
    nonce: u64,
}

impl AccountState {
    fn encode(self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.balance.to_le_bytes());
        bytes[8..].copy_from_slice(&self.nonce.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> Self {
        let raw: [u8; 16] = bytes.try_into().expect("account state must be 16 bytes");
        let balance = u64::from_le_bytes(raw[..8].try_into().unwrap());
        let nonce = u64::from_le_bytes(raw[8..].try_into().unwrap());
        Self { balance, nonce }
    }
}

#[derive(Clone, Copy, Debug)]
struct SignedTransaction {
    from: [u8; 20],
    to: [u8; 20],
    amount: u64,
    nonce: u64,
    public_key: [u8; 65],
    signature: [u8; 64],
}

#[derive(Clone, Debug)]
struct BlockBlueprint {
    number: u64,
    transactions: Vec<SignedTransaction>,
    signature: [u8; 64],
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
    bytes: &'a [u8],
    offset: usize,
}

struct BlockChunkAccumulator {
    block_number: Option<u64>,
    expected_chunks: u16,
    next_chunk_index: u16,
    payload: Vec<u8>,
}

impl BlockChunkAccumulator {
    fn new() -> Self {
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

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        let end = self.offset.checked_add(N).ok_or("offset overflow")?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or("unexpected end of input")?;
        self.offset = end;
        slice.try_into().map_err(|_| "invalid fixed-size field")
    }

    fn read_u16(&mut self) -> Result<u16, &'static str> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    fn read_u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn is_eof(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn keccak256(bytes: &[u8]) -> Result<[u8; 32], &'static str> {
    unsafe { sbi_crypto::keccak256_hash(bytes) }.map_err(|_| "keccak host call failed")
}

fn verify_signature(public_key: &[u8; 65], signature: &[u8; 64], message_hash: &[u8; 32]) -> bool {
    unsafe { sbi_crypto::secp256k1_verify(public_key, signature, message_hash) }
}

fn address_from_public_key(public_key: &[u8; 65]) -> Result<[u8; 20], &'static str> {
    if public_key[0] != 0x04 {
        return Err("only uncompressed secp256k1 public keys are supported");
    }

    let hash = keccak256(&public_key[1..])?;
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

fn read_account(context: &DurableKeySpace, address: &[u8; 20]) -> Result<AccountState, String> {
    let key_bytes = account_key_bytes(address);
    let key = Key::from_bytes(&key_bytes).expect("account key must be valid");
    match context.get(key).map_err(|error| error.to_string())? {
        Some(bytes) => Ok(AccountState::decode(&bytes)),
        None => Ok(AccountState {
            balance: 0,
            nonce: 0,
        }),
    }
}

fn write_account(
    context: &mut DurableKeySpace,
    address: &[u8; 20],
    state: AccountState,
) -> Result<(), String> {
    let key_bytes = account_key_bytes(address);
    let key = Key::from_bytes(&key_bytes).expect("account key must be valid");
    context
        .set(key, state.encode())
        .map_err(|error| error.to_string())
}

fn read_block_head(context: &DurableKeySpace) -> Result<u64, String> {
    match context
        .get(META_HEAD_KEY)
        .map_err(|error| error.to_string())?
    {
        Some(bytes) => Ok(u64::from_le_bytes(
            bytes.try_into().map_err(|_| "invalid stored block head")?,
        )),
        None => Ok(0),
    }
}

fn write_block_head(context: &mut DurableKeySpace, head: u64) -> Result<(), String> {
    context
        .set(META_HEAD_KEY, head.to_le_bytes())
        .map_err(|error| error.to_string())
}

fn bootstrap_context(context: &mut DurableKeySpace) -> Result<(), String> {
    if context
        .contains(META_BOOTSTRAPPED_KEY)
        .map_err(|error| error.to_string())?
    {
        return Ok(());
    }

    let funded_address = address_from_public_key(&SEQUENCER_PUBLIC_KEY)?.to_vec();
    let funded_address: [u8; 20] = funded_address.try_into().expect("address has fixed length");
    write_account(
        context,
        &funded_address,
        AccountState {
            balance: BOOTSTRAP_BALANCE,
            nonce: 0,
        },
    )?;
    write_block_head(context, 0)?;
    context
        .set(META_BOOTSTRAPPED_KEY, [1u8])
        .map_err(|error| error.to_string())?;
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

fn transaction_hash(transaction: &SignedTransaction) -> Result<[u8; 32], &'static str> {
    keccak256(&encode_unsigned_transaction(transaction))
}

fn block_hash(block_number: u64, transactions: &[SignedTransaction]) -> Result<[u8; 32], &'static str> {
    let mut header = [0u8; 14];
    header[..4].copy_from_slice(&BLOCK_BLUEPRINT_MAGIC);
    header[4..12].copy_from_slice(&block_number.to_le_bytes());
    header[12..14].copy_from_slice(&(transactions.len() as u16).to_le_bytes());

    let mut acc = keccak256(&header)?;
    for transaction in transactions {
        let tx_hash = transaction_hash(transaction)?;
        let mut bytes = [0u8; 64];
        bytes[..32].copy_from_slice(&acc);
        bytes[32..].copy_from_slice(&tx_hash);
        acc = keccak256(&bytes)?;
    }

    Ok(acc)
}

fn validate_transaction_signature(transaction: &SignedTransaction) -> Result<(), &'static str> {
    let sender = address_from_public_key(&transaction.public_key)?;
    if sender != transaction.from {
        return Err("transaction sender does not match public key");
    }

    let hash = transaction_hash(transaction)?;
    if !verify_signature(&transaction.public_key, &transaction.signature, &hash) {
        return Err("invalid transaction signature");
    }

    Ok(())
}

fn apply_valid_transaction(
    context: &mut DurableKeySpace,
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

fn parse_block_blueprint(bytes: &[u8]) -> Result<BlockBlueprint, &'static str> {
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

fn verify_block_signature(block: &BlockBlueprint) -> Result<(), &'static str> {
    let block_hash = block_hash(block.number, &block.transactions)?;
    if verify_signature(&SEQUENCER_PUBLIC_KEY, &block.signature, &block_hash) {
        Ok(())
    } else {
        Err("invalid sequencer signature")
    }
}

fn apply_block_blueprint(
    host: &mut impl Runtime,
    context: &mut DurableKeySpace,
    payload: &[u8],
    processed_transactions: &mut usize,
) -> Result<(), String> {
    let block = parse_block_blueprint(payload).map_err(|error| error.to_string())?;
    verify_block_signature(&block).map_err(|error| error.to_string())?;

    let current_head = read_block_head(context)?;
    if block.number != current_head + 1 {
        return Err(format!(
            "unexpected block number {}, expected {}",
            block.number,
            current_head + 1
        ));
    }

    let mut receipts = Vec::with_capacity(block.transactions.len());
    if *processed_transactions == 0 && !block.transactions.is_empty() {
        debug_msg!(
            host,
            "first processed tx block={} tx_index=0 total_processed=0\n",
            block.number
        );
    }

    for (tx_index, transaction) in block.transactions.iter().enumerate() {
        let sample_tx = tx_index % TX_TIMING_SAMPLE_INTERVAL == 0;
        if sample_tx {
            debug_msg!(
                host,
                "tx sample start block={} tx_index={} total_processed={}\n",
                block.number,
                tx_index,
                *processed_transactions
            );
        }

        let signature_ok = validate_transaction_signature(transaction).is_ok();
        if sample_tx {
            debug_msg!(
                host,
                "tx sample signature verified block={} tx_index={} total_processed={}\n",
                block.number,
                tx_index,
                *processed_transactions
            );
        }

        let receipt = if signature_ok {
            apply_valid_transaction(context, transaction)?
        } else {
            TxReceipt {
                status: TxStatus::Rejected,
            }
        };
        receipts.push(receipt);
        *processed_transactions += 1;
        if sample_tx {
            debug_msg!(
                host,
                "tx sample complete block={} tx_index={} total_processed={}\n",
                block.number,
                tx_index,
                *processed_transactions
            );
        }
        if tx_index + 1 == block.transactions.len() {
            debug_msg!(
                host,
                "last processed tx block={} tx_index={} total_processed={}\n",
                block.number,
                tx_index,
                *processed_transactions
            );
        }
    }

    debug_msg!(
        host,
        "block finalization start block={} total_processed={}\n",
        block.number,
        *processed_transactions
    );
    write_block_head(context, block.number)?;
    let state_root = context.hash().map_err(|error| error.to_string())?;
    let applied = receipts
        .iter()
        .filter(|receipt| matches!(receipt.status, TxStatus::Applied))
        .count();
    debug_msg!(
        host,
        "applied block {} with {} txs ({} applied), state root {:02x?}\n",
        block.number,
        receipts.len(),
        applied,
        state_root
    );

    Ok(())
}

fn handle_external_message(
    host: &mut impl Runtime,
    context: &mut DurableKeySpace,
    payload: &[u8],
    processed_transactions: &mut usize,
) {
    if let Err(error) = apply_block_blueprint(host, context, payload, processed_transactions) {
        debug_msg!(host, "rejected block blueprint: {error}\n");
    }
}

fn handle_external_payload(
    host: &mut impl Runtime,
    context: &mut DurableKeySpace,
    accumulator: &mut BlockChunkAccumulator,
    payload: &[u8],
    processed_transactions: &mut usize,
) {
    if payload.starts_with(&BLOCK_CHUNK_MAGIC) {
        match accumulator.push_chunk(payload) {
            Ok(Some((_block_number, block_payload))) => {
                let complete_payload = block_payload.to_vec();
                accumulator.reset();
                handle_external_message(host, context, &complete_payload, processed_transactions);
            }
            Ok(None) => {}
            Err(error) => {
                accumulator.reset();
                debug_msg!(host, "rejected block chunk: {error}\n");
            }
        }
    } else {
        accumulator.reset();
        handle_external_message(host, context, payload, processed_transactions);
    }
}

#[entrypoint::main]
pub fn entry(host: &mut impl Runtime) {
    let mut loader = DurableKeySpaceLoader::default();
    let mut context = loader
        .load_or_create(Name::from_str(CONTEXT_NAME).unwrap())
        .expect("Could not create tx-kernel durable context")
        .clone();

    bootstrap_context(&mut context).expect("Could not bootstrap blockchain context");
    let mut processed_transactions = 0usize;
    let mut block_chunks = BlockChunkAccumulator::new();

    while let Some(input) = host.read_input().expect("Could not read inbox message") {
        let (_, message) =
            InboxMessage::<MichelsonUnit>::parse(input.as_ref()).expect("Invalid inbox message");

        match message {
            InboxMessage::Internal(InternalInboxMessage::InfoPerLevel(_)) => {}
            InboxMessage::Internal(InternalInboxMessage::StartOfLevel) => {}
            InboxMessage::Internal(other) => {
                debug_msg!(host, "ignored internal inbox message: {other:?}\n");
            }
            InboxMessage::External(payload) => handle_external_payload(
                host,
                &mut context,
                &mut block_chunks,
                payload,
                &mut processed_transactions,
            ),
        }
    }
}
