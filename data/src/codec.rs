// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Leaf serialisation codecs.
//!
//! Hashing and merkle proofs reduce a state structure to a tree of *leaves*, where each leaf is the
//! serialisation of some value. Historically that serialisation was always bincode. To let different
//! consumers pick different leaf encodings (e.g. durable-storage on an alternative while the PVM stays on
//! bincode) without duplicating the fold/proof machinery, the leaf boundary is abstracted over a
//! [`LeafCodec`].
//!
//! A [`Fold`] (and the [proof deserialiser]) carries an associated [`LeafCodec`],
//! and leaf values are (de)serialised through [`LeafEncode`]/[`LeafDecode`] parameterised by that codec.
//! The [`Bincode`] codec reproduces the historical byte format exactly.
//!
//! [`Fold`]: crate::foldable::Fold
//! [proof deserialiser]: crate::merkle_proof::Deserialiser

use bincode::Decode;
use bincode::Encode;
use bincode::error::DecodeError;
use bincode::error::EncodeError;

use crate::serialisation::deserialise;
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

/// Error raised while encoding a leaf value under some [`LeafCodec`].
#[derive(Debug, thiserror::Error)]
pub enum LeafEncodeError {
    #[error("bincode encode error: {0}")]
    Bincode(#[from] EncodeError),
}

/// Error raised while decoding a leaf value under some [`LeafCodec`].
#[derive(Debug, thiserror::Error)]
pub enum LeafDecodeError {
    #[error("bincode decode error: {0}")]
    Bincode(#[from] DecodeError),
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

#[cfg(test)]
mod tests {
    use super::Bincode;
    use super::LeafDecode;
    use super::LeafEncode;
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
}
