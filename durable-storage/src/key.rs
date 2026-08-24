// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The 'key' part of the durable storage 'key-value' store.

use bincode::BorrowDecode;
use bincode::Decode;
use bincode::Encode;
use bincode::de::BorrowDecoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;

use crate::errors::InvalidArgumentError;

/// Maximum size of a key in bytes
pub const KEY_MAX_SIZE: usize = u8::MAX as usize;

/// A unique key used to store, retrieve and mutate data in durable storage.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Key(Vec<u8>);

impl Key {
    /// Create a new key from a byte slice, ensuring it is valid.
    pub fn new(bytes: &[u8]) -> Result<Self, InvalidArgumentError> {
        Self::check_bytes_validity(bytes)?;

        Ok(Key(bytes.to_vec()))
    }

    /// Check whether a given byte-slice is a valid key.
    fn check_bytes_validity(bytes: &[u8]) -> Result<(), InvalidArgumentError> {
        if bytes.len() > KEY_MAX_SIZE {
            return Err(InvalidArgumentError::KeyTooLong);
        }

        Ok(())
    }
}

impl AsRef<[u8]> for Key {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// Manual implementation of Encode to allow the smaller prefix size to be used.
impl Encode for Key {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        let bytes = self.0.as_slice();
        let len = u8::try_from(bytes.len()).expect("KEY_MAX_SIZE is precisely u8::MAX");

        Encode::encode(&len, encoder)?;
        encoder.writer().write(bytes)?;

        Ok(())
    }
}

// Manual implementation of Decode to allow the smaller prefix size to be used.
impl<Context> Decode<Context> for Key {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let len = u8::decode(decoder)? as usize;
        let mut key = vec![0; len];

        decoder.reader().read(key.as_mut_slice())?;

        Ok(Key(key))
    }
}

// Manual implementation of BorrowDecode to ensure that we do not decode a key that would
// be longer than `KEY_MAX_SIZE`.
impl<'de, Context> BorrowDecode<'de, Context> for Key {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Key::decode(decoder)
    }
}

#[cfg(test_utils)]
impl serde::Serialize for Key {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(&self.0))
    }
}

#[cfg(test_utils)]
impl<'de> serde::Deserialize<'de> for Key {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let s = String::deserialize(d)?;
        let bytes = hex::decode(&s).map_err(D::Error::custom)?;
        Key::new(&bytes).map_err(D::Error::custom)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use proptest::collection::vec;
    use proptest::prelude::Arbitrary;
    use proptest::prelude::BoxedStrategy;
    use proptest::prelude::Strategy;
    use proptest::prelude::any;
    use proptest::proptest;

    use super::KEY_MAX_SIZE;
    use super::Key;

    /// Generate Keys of any length up to [`KEY_MAX_SIZE`].
    impl Arbitrary for Key {
        type Parameters = ();

        type Strategy = BoxedStrategy<Key>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            vec(any::<u8>(), 0..=KEY_MAX_SIZE)
                .prop_map(|bytes| {
                    Key::new(bytes.as_slice())
                        .expect("All byte sequences of length <= KEY_MAX_SIZE are valid keys")
                })
                .boxed()
        }
    }

    proptest! {
        #[test]
        fn key_encode_decode_ok(key_len in 0..=KEY_MAX_SIZE) {
            use octez_riscv_data::serialisation as binary;

            let bytes = vec![0; key_len];
            let key = Key::new(&bytes).expect("Key is valid");

            let serialised = binary::serialise(&key)
                .expect("serialisation of a key should succeed");

            let key_decoded = binary::deserialise(&serialised)
                .expect("Decoding of a valid encoded key should succeed");

            assert_eq!(key, key_decoded, "encode then decode must produce the same key");

            // TODO: RV-838: introduce and use a direct helper from `octez_riscv_data::deserialisation`
            //               for borrow-decode
            let (key_decoded_slice, _): (Key, _) = bincode::borrow_decode_from_slice(
                &serialised,
                binary::bincode_default_config(),
            )
                .expect("Borrow decoding of a valid encoded key should succeed");

            assert_eq!(key, key_decoded_slice, "encode then borrow decode must produce the same key");
        }
    }
}
