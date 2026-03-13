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

/// Deserialise a slice of bytes, returning an error if there were leftover bytes.
pub fn deserialise_checked<T: Decode<()>>(data: &[u8]) -> Result<T, DecodeError> {
    let (value, length) = bincode::decode_from_slice(data, bincode_default_config())?;

    if length != data.len() {
        return Err(DecodeError::OtherString(
            format!("Slice was length {}, expected {}", data.len(), length).to_string(),
        ));
    }

    Ok(value)
}

/// Deserialises a slice into `T`, which may contain data borrowed from the slice.
pub fn deserialise_borrowed<'a, T: BorrowDecode<'a, ()>>(
    slice: &'a [u8],
) -> Result<(T, usize), DecodeError> {
    bincode::borrow_decode_from_slice(slice, bincode_default_config())
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

#[cfg(test)]
mod tests {
    use bincode::error::DecodeError;

    use super::deserialise_checked;

    #[test]
    fn deserialise_checked_errors() {
        let r = deserialise_checked::<u32>(&[2, 1, 0, 0]);
        assert!(matches!(r, Ok(258)));

        let r = deserialise_checked::<u32>(&[2, 1, 0, 0, 3]);
        assert!(matches!(
            r,
            Err(DecodeError::OtherString(string))
            if string == "Slice was length 5, expected 4"));

        let r = deserialise_checked::<u32>(&[2, 1]);
        assert!(matches!(
            r,
            Err(DecodeError::UnexpectedEnd { additional: 2 }),
        ));
    }
}
