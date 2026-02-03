// SPDX-FileCopyrightText: 2025-2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

use bincode::Encode;

/// Get the bitmask formed of `n` ones.
pub const fn ones(n: u64) -> u64 {
    // This function should not panic
    let sh_amt = 64_u64.saturating_sub(n);
    match n {
        0 => 0,
        _ => !0 >> sh_amt,
    }
}

/// Tag of a node
pub const TAG_NODE: u8 = 0b00;
/// Tag of a blind leaf
pub const TAG_BLIND: u8 = 0b10;
/// Tag of a read leaf
pub const TAG_READ: u8 = 0b11;

/// Number of bits used to represent a tag
const TAG_BITS: u32 = 2;
/// Number of tags that can fit in a single byte
const TAGS_PER_BYTE: usize = u8::BITS as usize / TAG_BITS as usize;
/// Bitmask for tags
const TAG_MASK: u8 = ones(TAG_BITS as u64) as u8;

/// Return the offset of the `index`-th tag in a byte.
const fn tag_offset(index: usize) -> usize {
    debug_assert!(index < TAGS_PER_BYTE);
    u8::BITS as usize - (index + 1) * TAG_BITS as usize
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeafTag {
    Blind,
    Read,
}

impl Encode for LeafTag {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        match self {
            LeafTag::Blind => TAG_BLIND,
            LeafTag::Read => TAG_READ,
        }
        .encode(encoder)
    }
}

/// The tag is invalid.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Invalid tag")]
pub struct InvalidTagError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tag {
    Node,
    Leaf(LeafTag),
}

impl Tag {
    /// Obtain the parsed tags from the most significant bits to the lower ones.
    pub fn ordered_tags_from_u8(byte: u8) -> [Result<Tag, InvalidTagError>; TAGS_PER_BYTE] {
        core::array::from_fn(tag_offset)
            .map(|offset| (byte >> offset) & TAG_MASK)
            .map(Tag::try_from)
    }
}

impl bincode::Encode for Tag {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        match self {
            Tag::Node => TAG_NODE.encode(encoder),
            Tag::Leaf(leaf_tag) => leaf_tag.encode(encoder),
        }
    }
}

impl<'de, C> bincode::BorrowDecode<'de, C> for Tag {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let byte = u8::borrow_decode(decoder)?;
        Tag::try_from(byte)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))
    }
}

impl<C> bincode::Decode<C> for Tag {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let byte = u8::decode(decoder)?;
        Tag::try_from(byte)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))
    }
}

impl From<Tag> for u8 {
    fn from(value: Tag) -> Self {
        match value {
            Tag::Node => TAG_NODE,
            Tag::Leaf(leaf_tag) => match leaf_tag {
                LeafTag::Blind => TAG_BLIND,
                LeafTag::Read => TAG_READ,
            },
        }
    }
}

impl From<LeafTag> for Tag {
    fn from(value: LeafTag) -> Self {
        Tag::Leaf(value)
    }
}

impl TryFrom<u8> for Tag {
    type Error = InvalidTagError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            TAG_NODE => Ok(Self::Node),
            TAG_BLIND => Ok(Self::Leaf(LeafTag::Blind)),
            TAG_READ => Ok(Self::Leaf(LeafTag::Read)),
            _ => Err(InvalidTagError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialisation::bincode_default_config;

    fn tag_encode_cycle_checker(tag: &Tag) {
        let mut encoded = bincode::encode_to_vec(tag, bincode_default_config())
            .expect("Failed to encode the tag");
        let (decoded, _): (Tag, usize) =
            bincode::decode_from_slice(encoded.as_mut_slice(), bincode_default_config())
                .expect("Failed to decode the tag");
        assert_eq!(*tag, decoded);
    }

    #[test]
    fn can_encode_and_decode_tags() {
        let tags = [
            Tag::Node,
            Tag::Leaf(LeafTag::Blind),
            Tag::Leaf(LeafTag::Read),
        ];
        for tag in tags.iter() {
            tag_encode_cycle_checker(tag);
        }
    }
}
