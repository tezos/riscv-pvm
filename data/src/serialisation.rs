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
use rkyv::api::high::HighSerializer;
use rkyv::api::high::HighValidator;
use rkyv::bytecheck::CheckBytes;
use rkyv::de::Pool;
use rkyv::rancor;
use rkyv::rancor::Strategy;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;

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

/// Alignment of the [`AlignedVec`] used by the rkyv helpers.
///
/// This matches the default alignment of the [`AlignedVec`] that rkyv's high-level `to_bytes`
/// produces, so the read side ([`rkyv_deserialise`]) and the write side ([`rkyv_serialise`]) agree.
const RKYV_ALIGN: usize = 16;

/// Serialise `value` into an rkyv archive.
///
/// The bytes are returned in an [`AlignedVec`] so that, when read back, the buffer already meets
/// rkyv's alignment requirement (see [`rkyv_deserialise`]).
pub fn rkyv_serialise<T>(value: &T) -> Result<AlignedVec, rancor::Error>
where
    T: for<'a> rkyv::Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    rkyv::to_bytes::<rancor::Error>(value)
}

/// Deserialise an rkyv archive into an owned `T`, validating the archive first.
///
/// An rkyv archive must be aligned to the archived type's alignment before it can be accessed, but a
/// slice coming from RocksDB or an arbitrary buffer is only byte-aligned. We therefore copy the
/// bytes into an [`AlignedVec`] before validated access.
pub fn rkyv_deserialise<T>(bytes: &[u8]) -> Result<T, rancor::Error>
where
    T: rkyv::Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, rancor::Error>>
        + rkyv::Deserialize<T, Strategy<Pool, rancor::Error>>,
{
    // Monomorphisation-time guard: our `AlignedVec` is aligned to `RKYV_ALIGN`, so the archived root
    // must not require a stronger alignment. This is the alignment of the *root* archived type; it is
    // a lower bound (out-of-line data could in principle need more), but rkyv's high-level `to_bytes`
    // is itself built around `AlignedVec<16>`, so anything that would exceed this could not have been
    // serialised by `rkyv_serialise` in the first place. If it were ever under-aligned at runtime,
    // `from_bytes` would still return an error rather than misbehave.
    const {
        assert!(
            align_of::<T::Archived>() <= RKYV_ALIGN,
            "archived alignment exceeds the AlignedVec alignment used by the rkyv helpers"
        )
    };

    let mut aligned = AlignedVec::<RKYV_ALIGN>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    rkyv::from_bytes::<T, rancor::Error>(&aligned)
}

#[cfg(test)]
mod tests {
    use bincode::error::DecodeError;

    use super::deserialise_checked;
    use super::rkyv_deserialise;
    use super::rkyv_serialise;

    #[derive(Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    struct RkyvRoundTrip {
        a: u32,
        b: Vec<u8>,
        c: Option<u64>,
    }

    #[test]
    fn rkyv_serialise_deserialise_round_trip() {
        let value = RkyvRoundTrip {
            a: 0xDEAD_BEEF,
            b: vec![1, 2, 3, 4, 5],
            c: Some(42),
        };

        let bytes = rkyv_serialise(&value).expect("rkyv serialisation should succeed");

        // Deserialise from a deliberately byte-aligned-only copy to exercise the alignment handling.
        let unaligned = bytes.as_slice().to_vec();
        let decoded: RkyvRoundTrip =
            rkyv_deserialise(&unaligned).expect("rkyv deserialisation should succeed");

        assert_eq!(
            value, decoded,
            "rkyv encode then decode must be the identity"
        );
    }

    #[test]
    fn rkyv_deserialise_rejects_corrupt_bytes() {
        let res: Result<RkyvRoundTrip, _> = rkyv_deserialise(&[0xFF; 3]);
        assert!(res.is_err(), "validated access must reject malformed bytes");
    }

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
