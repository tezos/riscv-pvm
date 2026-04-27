// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use rlp::DecoderError;
use rlp::Rlp;
use rlp::RlpStream;

use crate::Reader;

pub const ETH_BLOCK_BLUEPRINT_MAGIC: [u8; 4] = *b"TXE1";
pub const ETH_TX_TYPE_EIP1559: u8 = 0x02;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessListItem {
    pub address: [u8; 20],
    pub storage_keys: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eip1559Transaction {
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee_per_gas: u64,
    pub max_fee_per_gas: u64,
    pub gas_limit: u64,
    pub to: Option<[u8; 20]>,
    pub value: [u8; 32],
    pub data: Vec<u8>,
    pub access_list: Vec<AccessListItem>,
    pub signature_y_parity: u8,
    pub signature_r: [u8; 32],
    pub signature_s: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EthereumTransaction {
    Eip1559(Eip1559Transaction),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EthereumBlockBlueprint {
    pub number: u64,
    pub transactions: Vec<Vec<u8>>,
    pub signature: [u8; 64],
}

impl Eip1559Transaction {
    pub fn signature(&self) -> [u8; 64] {
        let mut signature = [0u8; 64];
        signature[..32].copy_from_slice(&self.signature_r);
        signature[32..].copy_from_slice(&self.signature_s);
        signature
    }

    pub fn signing_payload(&self) -> Vec<u8> {
        let mut stream = RlpStream::new_list(9);
        append_u64(&mut stream, self.chain_id);
        append_u64(&mut stream, self.nonce);
        append_u64(&mut stream, self.max_priority_fee_per_gas);
        append_u64(&mut stream, self.max_fee_per_gas);
        append_u64(&mut stream, self.gas_limit);
        append_destination(&mut stream, self.to.as_ref());
        append_u256(&mut stream, &self.value);
        stream.append(&self.data.as_slice());
        append_access_list(&mut stream, &self.access_list);

        let payload = stream.out();
        let mut encoded = Vec::with_capacity(1 + payload.len());
        encoded.push(ETH_TX_TYPE_EIP1559);
        encoded.extend_from_slice(&payload);
        encoded
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut stream = RlpStream::new_list(12);
        append_u64(&mut stream, self.chain_id);
        append_u64(&mut stream, self.nonce);
        append_u64(&mut stream, self.max_priority_fee_per_gas);
        append_u64(&mut stream, self.max_fee_per_gas);
        append_u64(&mut stream, self.gas_limit);
        append_destination(&mut stream, self.to.as_ref());
        append_u256(&mut stream, &self.value);
        stream.append(&self.data.as_slice());
        append_access_list(&mut stream, &self.access_list);
        append_u64(&mut stream, self.signature_y_parity as u64);
        append_u256(&mut stream, &self.signature_r);
        append_u256(&mut stream, &self.signature_s);

        let payload = stream.out();
        let mut encoded = Vec::with_capacity(1 + payload.len());
        encoded.push(ETH_TX_TYPE_EIP1559);
        encoded.extend_from_slice(&payload);
        encoded
    }
}

impl EthereumTransaction {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let Some(tx_type) = bytes.first().copied() else {
            return Err("empty Ethereum transaction".to_string());
        };

        match tx_type {
            ETH_TX_TYPE_EIP1559 => {
                let rlp = Rlp::new(&bytes[1..]);
                Ok(Self::Eip1559(parse_eip1559_transaction(&rlp)?))
            }
            _ => Err(format!(
                "unsupported Ethereum transaction type {tx_type:#04x}"
            )),
        }
    }

    pub fn signing_payload(&self) -> Vec<u8> {
        match self {
            Self::Eip1559(tx) => tx.signing_payload(),
        }
    }
}

pub fn ethereum_block_preimage(number: u64, transactions: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&ETH_BLOCK_BLUEPRINT_MAGIC);
    out.extend_from_slice(&number.to_le_bytes());
    out.extend_from_slice(&(transactions.len() as u16).to_le_bytes());
    for transaction in transactions {
        out.extend_from_slice(&(transaction.len() as u32).to_le_bytes());
        out.extend_from_slice(transaction);
    }
    out
}

/// Returns the compact 14-byte block header used as the seed for the chained block hash.
///
/// The chained block hash avoids large keccak inputs (capped at 4096 bytes in the PVM):
///   `h = keccak(header)`, then for each tx: `h = keccak(h || keccak(tx_bytes))`
/// Each individual keccak input is ≤ 64 bytes regardless of block size.
pub fn ethereum_block_hash_header(number: u64, tx_count: usize) -> [u8; 14] {
    let mut header = [0u8; 14];
    header[..4].copy_from_slice(&ETH_BLOCK_BLUEPRINT_MAGIC);
    header[4..12].copy_from_slice(&number.to_le_bytes());
    header[12..14].copy_from_slice(&(tx_count as u16).to_le_bytes());
    header
}

pub fn build_ethereum_block_blueprint(
    number: u64,
    transactions: &[Vec<u8>],
    signature: &[u8; 64],
) -> Vec<u8> {
    let mut out = ethereum_block_preimage(number, transactions);
    out.extend_from_slice(signature);
    out
}

pub fn parse_ethereum_block_blueprint(
    bytes: &[u8],
) -> Result<EthereumBlockBlueprint, &'static str> {
    let mut reader = Reader::new(bytes);
    let magic = reader.take::<4>()?;
    if magic != ETH_BLOCK_BLUEPRINT_MAGIC {
        return Err("unexpected Ethereum block blueprint magic");
    }

    let number = reader.read_u64()?;
    let tx_count = reader.read_u16()? as usize;
    let mut transactions = Vec::with_capacity(tx_count);
    for _ in 0..tx_count {
        let len = reader.read_u32()? as usize;
        let start = reader.offset();
        let end = start
            .checked_add(len)
            .ok_or("transaction length overflow")?;
        let tx = reader
            .bytes
            .get(start..end)
            .ok_or("unexpected end of input")?
            .to_vec();
        reader.offset = end;
        transactions.push(tx);
    }

    let signature = reader.take::<64>()?;
    if !reader.is_eof() {
        return Err("unexpected trailing bytes in Ethereum block blueprint");
    }

    Ok(EthereumBlockBlueprint {
        number,
        transactions,
        signature,
    })
}

fn parse_eip1559_transaction(rlp: &Rlp<'_>) -> Result<Eip1559Transaction, String> {
    if !rlp.is_list() {
        return Err("EIP-1559 transaction payload must be an RLP list".to_string());
    }
    if rlp.item_count().map_err(decoder_error_to_string)? != 12 {
        return Err("EIP-1559 transaction payload must have 12 fields".to_string());
    }

    let chain_id = rlp.val_at::<u64>(0).map_err(decoder_error_to_string)?;
    let nonce = rlp.val_at::<u64>(1).map_err(decoder_error_to_string)?;
    let max_priority_fee_per_gas = rlp.val_at::<u64>(2).map_err(decoder_error_to_string)?;
    let max_fee_per_gas = rlp.val_at::<u64>(3).map_err(decoder_error_to_string)?;
    let gas_limit = rlp.val_at::<u64>(4).map_err(decoder_error_to_string)?;
    let to = parse_destination(&rlp.at(5).map_err(decoder_error_to_string)?)?;
    let value = rlp_u256_at(rlp, 6)?;
    let data = rlp.val_at::<Vec<u8>>(7).map_err(decoder_error_to_string)?;
    let access_list = parse_access_list(&rlp.at(8).map_err(decoder_error_to_string)?)?;
    let signature_y_parity = rlp.val_at::<u8>(9).map_err(decoder_error_to_string)?;
    if signature_y_parity > 1 {
        return Err("invalid EIP-1559 signature y parity".to_string());
    }
    let signature_r = rlp_u256_at(rlp, 10)?;
    let signature_s = rlp_u256_at(rlp, 11)?;

    Ok(Eip1559Transaction {
        chain_id,
        nonce,
        max_priority_fee_per_gas,
        max_fee_per_gas,
        gas_limit,
        to,
        value,
        data,
        access_list,
        signature_y_parity,
        signature_r,
        signature_s,
    })
}

fn parse_destination(rlp: &Rlp<'_>) -> Result<Option<[u8; 20]>, String> {
    if rlp.is_empty() {
        return Ok(None);
    }
    let bytes = rlp.data().map_err(decoder_error_to_string)?;
    let address: [u8; 20] = bytes
        .try_into()
        .map_err(|_| "EIP-1559 destination must be 20 bytes".to_string())?;
    Ok(Some(address))
}

fn parse_access_list(rlp: &Rlp<'_>) -> Result<Vec<AccessListItem>, String> {
    if !rlp.is_list() {
        return Err("EIP-1559 access list must be an RLP list".to_string());
    }

    let count = rlp.item_count().map_err(decoder_error_to_string)?;
    let mut items = Vec::with_capacity(count);
    for index in 0..count {
        let item = rlp.at(index).map_err(decoder_error_to_string)?;
        if item.item_count().map_err(decoder_error_to_string)? != 2 {
            return Err("access list item must contain address and storage key list".to_string());
        }
        let address_bytes = item
            .at(0)
            .map_err(decoder_error_to_string)?
            .data()
            .map_err(decoder_error_to_string)?;
        let address: [u8; 20] = address_bytes
            .try_into()
            .map_err(|_| "access list address must be 20 bytes".to_string())?;

        let storage_list = item.at(1).map_err(decoder_error_to_string)?;
        if !storage_list.is_list() {
            return Err("access list storage keys must be an RLP list".to_string());
        }
        let storage_count = storage_list.item_count().map_err(decoder_error_to_string)?;
        let mut storage_keys = Vec::with_capacity(storage_count);
        for storage_index in 0..storage_count {
            let storage_bytes = storage_list
                .at(storage_index)
                .map_err(decoder_error_to_string)?
                .data()
                .map_err(decoder_error_to_string)?;
            let storage_key: [u8; 32] = storage_bytes
                .try_into()
                .map_err(|_| "access list storage key must be 32 bytes".to_string())?;
            storage_keys.push(storage_key);
        }

        items.push(AccessListItem {
            address,
            storage_keys,
        });
    }

    Ok(items)
}

fn rlp_u256_at(rlp: &Rlp<'_>, index: usize) -> Result<[u8; 32], String> {
    let bytes = rlp
        .at(index)
        .map_err(decoder_error_to_string)?
        .data()
        .map_err(decoder_error_to_string)?;
    if bytes.len() > 32 {
        return Err("RLP integer exceeds 256 bits".to_string());
    }
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(bytes);
    Ok(out)
}

fn append_u64(stream: &mut RlpStream, value: u64) {
    stream.append(&value);
}

fn append_u256(stream: &mut RlpStream, value: &[u8; 32]) {
    let first_non_zero = value
        .iter()
        .position(|byte| *byte != 0)
        .unwrap_or(value.len());
    if first_non_zero == value.len() {
        stream.append_empty_data();
    } else {
        let trimmed = &value[first_non_zero..];
        stream.append(&trimmed);
    }
}

fn append_destination(stream: &mut RlpStream, destination: Option<&[u8; 20]>) {
    match destination {
        Some(address) => {
            stream.append(&address.as_slice());
        }
        None => {
            stream.append_empty_data();
        }
    }
}

fn append_access_list(stream: &mut RlpStream, access_list: &[AccessListItem]) {
    stream.begin_list(access_list.len());
    for item in access_list {
        stream.begin_list(2);
        stream.append(&item.address.as_slice());
        stream.begin_list(item.storage_keys.len());
        for storage_key in &item.storage_keys {
            stream.append(&storage_key.as_slice());
        }
    }
}

fn decoder_error_to_string(error: DecoderError) -> String {
    error.to_string()
}

impl<'a> Reader<'a> {
    fn read_u32(&mut self) -> Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.take()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_eip1559_transaction() -> Eip1559Transaction {
        Eip1559Transaction {
            chain_id: 1337,
            nonce: 7,
            max_priority_fee_per_gas: 1_000_000_000,
            max_fee_per_gas: 2_000_000_000,
            gas_limit: 21_000,
            to: Some([0x11; 20]),
            value: {
                let mut value = [0u8; 32];
                value[31] = 5;
                value
            },
            data: vec![0xaa, 0xbb, 0xcc],
            access_list: vec![AccessListItem {
                address: [0x22; 20],
                storage_keys: vec![[0x33; 32]],
            }],
            signature_y_parity: 1,
            signature_r: [0x44; 32],
            signature_s: [0x55; 32],
        }
    }

    #[test]
    fn parses_and_reencodes_eip1559_transaction() {
        let tx = sample_eip1559_transaction();
        let encoded = tx.encode();
        let parsed = EthereumTransaction::parse(&encoded).unwrap();
        assert_eq!(parsed, EthereumTransaction::Eip1559(tx));
    }

    #[test]
    fn eip1559_signing_payload_excludes_signature_fields() {
        let tx = sample_eip1559_transaction();
        let signing_payload = tx.signing_payload();
        let signed_payload = tx.encode();
        assert_ne!(signing_payload, signed_payload);
        assert_eq!(signing_payload[0], ETH_TX_TYPE_EIP1559);
    }

    #[test]
    fn parses_ethereum_block_blueprint() {
        let tx = sample_eip1559_transaction().encode();
        let block = build_ethereum_block_blueprint(3, &[tx.clone()], &[0x99; 64]);
        let parsed = parse_ethereum_block_blueprint(&block).unwrap();
        assert_eq!(parsed.number, 3);
        assert_eq!(parsed.transactions, vec![tx]);
        assert_eq!(parsed.signature, [0x99; 64]);
    }

    #[test]
    fn parses_contract_creation_transaction() {
        let mut tx = sample_eip1559_transaction();
        tx.to = None;
        tx.data = vec![0x60, 0x00, 0x60, 0x00];
        let encoded = tx.encode();
        let parsed = EthereumTransaction::parse(&encoded).unwrap();
        assert_eq!(parsed, EthereumTransaction::Eip1559(tx));
    }
}
