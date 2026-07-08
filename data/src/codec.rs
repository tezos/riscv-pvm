// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Leaf serialisation codecs.
//!
//! Hashing and merkle proofs reduce a state structure to a tree of *leaves*, where each leaf is the
//! serialisation of some value. Historically that serialisation was always bincode. To let different
//! consumers pick different leaf encodings (e.g. durable-storage on rkyv while the PVM stays on
//! bincode) without duplicating the fold/proof machinery, the leaf boundary is abstracted over a
//! [`LeafCodec`].
//!
//! A [`crate::foldable::Fold`] (and the proof [`crate::merkle_proof::Deserialiser`]) carries an
//! associated [`LeafCodec`], and leaf values are (de)serialised through [`LeafEncode`]/[`LeafDecode`]
//! parameterised by that codec. The [`Bincode`] codec reproduces the historical byte format exactly;
//! [`Rkyv`] is the new format (its leaf impls are added in a later step).

use bincode::Decode;
use bincode::Encode;
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

use crate::serialisation::deserialise;
use crate::serialisation::rkyv_deserialise;
use crate::serialisation::rkyv_serialise;
use crate::serialisation::serialise;

/// A leaf serialisation codec: the strategy used to turn a leaf value into bytes (and back) when
/// folding/hashing or building/verifying proofs.
pub trait LeafCodec {}

/// The bincode codec — the historical leaf format.
///
/// Its [`LeafEncode`]/[`LeafDecode`] impls delegate to the bincode helpers, so output is
/// byte-identical to the pre-codec code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bincode;

impl LeafCodec for Bincode {}

/// The rkyv codec.
///
/// This marker exists so folds and proof deserialisers can be parameterised by it now; the leaf
/// impls (and proof leaf framing) are added in a subsequent step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rkyv;

impl LeafCodec for Rkyv {}

/// Error raised while encoding a leaf value under some [`LeafCodec`].
#[derive(Debug, thiserror::Error)]
pub enum LeafEncodeError {
    #[error("bincode encode error: {0}")]
    Bincode(#[from] EncodeError),

    #[error("rkyv serialisation error: {0}")]
    Rkyv(#[from] rkyv::rancor::Error),
}

/// Error raised while decoding a leaf value under some [`LeafCodec`].
#[derive(Debug, thiserror::Error)]
pub enum LeafDecodeError {
    #[error("bincode decode error: {0}")]
    Bincode(#[from] DecodeError),

    #[error("rkyv deserialisation error: {0}")]
    Rkyv(#[from] rkyv::rancor::Error),

    #[error("rkyv leaf framing error: {0}")]
    Framing(&'static str),
}

/// A value that can be encoded as a hash/proof leaf under codec `C`.
pub trait LeafEncode<C: LeafCodec> {
    /// Encode `self` into the leaf byte representation for codec `C`.
    fn leaf_encode(&self) -> Result<Vec<u8>, LeafEncodeError>;
}

/// A value that can be reconstructed from a hash/proof leaf's bytes under codec `C`.
pub trait LeafDecode<C: LeafCodec>: Sized {
    /// Decode the leaf byte representation for codec `C` from an exact byte slice.
    fn leaf_decode(bytes: &[u8]) -> Result<Self, LeafDecodeError>;

    /// Decode a leaf from the front of `bytes`, returning the value and the number of bytes
    /// consumed.
    ///
    /// This is the streaming counterpart used by the byte-stream proof deserialiser, which lays
    /// leaves back-to-back and needs to know where each one ends. For [`Bincode`] the archive is
    /// self-delimiting so the length falls out of decoding; other codecs may need explicit framing.
    fn leaf_decode_stream(bytes: &[u8]) -> Result<(Self, usize), LeafDecodeError>;
}

// --- Bincode codec: exactly the historical behaviour. ---

impl<T: Encode> LeafEncode<Bincode> for T {
    fn leaf_encode(&self) -> Result<Vec<u8>, LeafEncodeError> {
        Ok(serialise(self)?)
    }
}

impl<T: Decode<()>> LeafDecode<Bincode> for T {
    fn leaf_decode(bytes: &[u8]) -> Result<Self, LeafDecodeError> {
        Ok(deserialise(bytes)?)
    }

    fn leaf_decode_stream(bytes: &[u8]) -> Result<(Self, usize), LeafDecodeError> {
        // Decode from the front of the slice exactly as the stream deserialiser historically did
        // (streaming read over `&[u8]`), computing the consumed length from how far the cursor
        // advanced. This preserves the precise bincode error kind (e.g. `Io` on truncation).
        let mut cursor: &[u8] = bytes;
        let value = crate::serialisation::deserialise_from(&mut cursor)?;
        let consumed = bytes.len() - cursor.len();
        Ok((value, consumed))
    }
}

// --- Rkyv codec. ---
//
// An rkyv archive is not self-delimiting from the front, so each leaf is length-framed: an 8-byte
// little-endian archive length followed by the archive bytes. The prefix is part of the leaf's byte
// representation, so it is included in the leaf hash and stored verbatim in a proof — this keeps the
// merkle-proof machinery codec-agnostic (it always shuttles opaque leaf bytes) while remaining
// self-delimiting for the stream deserialiser. Decoding copies the archive into an `AlignedVec`
// before validated access (see [`rkyv_deserialise`]), since a leaf slice sits at an arbitrary offset.

/// Number of bytes in the leaf length prefix used by the rkyv codec.
const RKYV_LEN_PREFIX: usize = size_of::<u64>();

impl<T> LeafEncode<Rkyv> for T
where
    T: for<'a> rkyv::Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    fn leaf_encode(&self) -> Result<Vec<u8>, LeafEncodeError> {
        let archive = rkyv_serialise(self)?;

        let mut out = Vec::with_capacity(RKYV_LEN_PREFIX + archive.len());
        out.extend_from_slice(&(archive.len() as u64).to_le_bytes());
        out.extend_from_slice(&archive);
        Ok(out)
    }
}

impl<T> LeafDecode<Rkyv> for T
where
    T: rkyv::Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, rancor::Error>>
        + rkyv::Deserialize<T, Strategy<Pool, rancor::Error>>,
{
    fn leaf_decode(bytes: &[u8]) -> Result<Self, LeafDecodeError> {
        let (value, _consumed) = Self::leaf_decode_stream(bytes)?;
        Ok(value)
    }

    fn leaf_decode_stream(bytes: &[u8]) -> Result<(Self, usize), LeafDecodeError> {
        let len_bytes = bytes
            .get(..RKYV_LEN_PREFIX)
            .ok_or(LeafDecodeError::Framing("leaf shorter than length prefix"))?;
        // The slice length matches the prefix width, so the conversion cannot fail.
        let len = u64::from_le_bytes(
            len_bytes
                .try_into()
                .map_err(|_| LeafDecodeError::Framing("invalid length prefix"))?,
        ) as usize;

        let end = RKYV_LEN_PREFIX
            .checked_add(len)
            .ok_or(LeafDecodeError::Framing("leaf length overflow"))?;
        let archive = bytes
            .get(RKYV_LEN_PREFIX..end)
            .ok_or(LeafDecodeError::Framing("leaf shorter than framed length"))?;

        let value = rkyv_deserialise(archive)?;
        Ok((value, end))
    }
}

#[cfg(test)]
mod tests {
    use super::Bincode;
    use super::LeafDecode;
    use super::LeafDecodeError;
    use super::LeafEncode;
    use super::Rkyv;
    use crate::serialisation::serialise;

    #[test]
    fn bincode_codec_matches_plain_bincode() {
        let value: u64 = 0x0123_4567_89AB_CDEF;

        // Encoding through the Bincode codec is byte-identical to calling `serialise` directly.
        let via_codec = LeafEncode::<Bincode>::leaf_encode(&value).expect("encode ok");
        let via_bincode = serialise(value).expect("serialise ok");
        assert_eq!(via_codec, via_bincode);

        // And it round-trips.
        let decoded: u64 = LeafDecode::<Bincode>::leaf_decode(&via_codec).expect("decode ok");
        assert_eq!(decoded, value);
    }

    #[test]
    fn rkyv_codec_round_trip() {
        let value: u64 = 0x0123_4567_89AB_CDEF;

        let encoded = LeafEncode::<Rkyv>::leaf_encode(&value).expect("rkyv encode ok");

        // Whole-slice decode.
        let decoded: u64 = LeafDecode::<Rkyv>::leaf_decode(&encoded).expect("rkyv decode ok");
        assert_eq!(decoded, value);

        // Stream decode reports the exact number of bytes consumed (prefix + archive), so
        // back-to-back leaves can be parsed in sequence.
        let (decoded, consumed): (u64, usize) =
            LeafDecode::<Rkyv>::leaf_decode_stream(&encoded).expect("rkyv stream decode ok");
        assert_eq!(decoded, value);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn rkyv_stream_decode_stops_at_leaf_boundary() {
        // Two framed leaves back-to-back: the stream decoder must consume exactly the first.
        let mut buf = LeafEncode::<Rkyv>::leaf_encode(&1u32).expect("encode ok");
        let first_len = buf.len();
        buf.extend_from_slice(&LeafEncode::<Rkyv>::leaf_encode(&2u32).expect("encode ok"));

        let (a, consumed): (u32, usize) =
            LeafDecode::<Rkyv>::leaf_decode_stream(&buf).expect("decode ok");
        assert_eq!(a, 1);
        assert_eq!(consumed, first_len);

        let (b, _): (u32, usize) =
            LeafDecode::<Rkyv>::leaf_decode_stream(&buf[consumed..]).expect("decode ok");
        assert_eq!(b, 2);
    }

    #[test]
    fn rkyv_decode_rejects_truncated_frame() {
        let encoded = LeafEncode::<Rkyv>::leaf_encode(&12345u64).expect("encode ok");

        // Chop a byte off the framed archive: the length prefix now over-runs the buffer.
        let truncated = &encoded[..encoded.len() - 1];
        let res: Result<u64, _> = LeafDecode::<Rkyv>::leaf_decode(truncated);
        assert!(matches!(res, Err(LeafDecodeError::Framing(_))));

        // Fewer bytes than the length prefix itself.
        let res: Result<u64, _> = LeafDecode::<Rkyv>::leaf_decode(&[0u8; 3]);
        assert!(matches!(res, Err(LeafDecodeError::Framing(_))));
    }
}
