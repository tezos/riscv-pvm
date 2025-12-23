// SPDX-FileCopyrightText: 2024 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Serialisation and deserialisation

pub mod elem;

use std::io::Read;
use std::io::Write;

use bincode::BorrowDecode;
use bincode::Decode;
use bincode::Encode;
use bincode::config::Config;
use bincode::config::standard;
use bincode::error::DecodeError;
use bincode::error::EncodeError;

/// Constructs the default options for bincode serialisation and deserialisation.
#[inline]
pub const fn bincode_default_config() -> impl Config {
    standard()
        .with_limit::<{ 1024 * 1024 * 1024 }>()
        .with_little_endian()
        .with_fixed_int_encoding()
}

/// Deserialise a slice of bytes into a value of type `T`.
pub fn deserialise<T: Decode<()>>(data: &[u8]) -> Result<T, DecodeError> {
    let (value, _) = bincode::decode_from_slice(data, bincode_default_config())?;
    Ok(value)
}

/// Borrows the bytes of a slice and cast it into a type `T`
pub fn borrow_decode<'a, T: BorrowDecode<'a, ()>>(slice: &'a [u8]) -> Result<T, DecodeError> {
    let (value, _) = bincode::borrow_decode_from_slice(slice, bincode_default_config())?;
    Ok(value)
}

/// Deserialise a value of type `T` from a byte source.
pub fn deserialise_from<T: Decode<()>, R: Read>(source: &mut R) -> Result<T, DecodeError> {
    bincode::decode_from_std_read(source, bincode_default_config())
}

/// Serialize `T` into a vector of bytes.
pub fn serialise<T: Encode>(value: T) -> Result<Vec<u8>, EncodeError> {
    bincode::encode_to_vec(value, bincode_default_config())
}

/// Serialize `T` into a sink.
pub fn serialise_into<T: Encode, W: Write>(value: T, sink: &mut W) -> Result<usize, EncodeError> {
    bincode::encode_into_std_write(value, sink, bincode_default_config())
}
