// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The 'key' part of the durable storage 'key-value' store.

use bincode::BorrowDecode;
use bincode::Decode;
use bincode::Encode;
use bincode::de::BorrowDecoder;

/// Maximum size of a key in bytes
pub const KEY_MAX_SIZE: usize = 256;

/// Errors related to construction of keys.
#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    /// The maximum size of a key is [`KEY_MAX_SIZE`].
    #[error("The provided key is too long")]
    KeyTooLong,
}

/// A unique key used to store, retrieve and mutate data in durable storage.
#[derive(Clone, Debug, Default, Encode, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Key(Vec<u8>);

impl Key {
    pub fn new(bytes: &[u8]) -> Result<Self, KeyError> {
        Self::check_bytes_validity(bytes)?;

        Ok(Key(bytes.to_vec()))
    }

    /// Check whether a given byte-slice is a valid key.
    fn check_bytes_validity(bytes: &[u8]) -> Result<(), KeyError> {
        if bytes.len() > KEY_MAX_SIZE {
            return Err(KeyError::KeyTooLong);
        }

        Ok(())
    }
}

impl AsRef<[u8]> for Key {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// Manual implementation of Decode to ensure that we do not decode a key that is formed of
// invalid bytes.
impl<Context> Decode<Context> for Key {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let bytes = Vec::decode(decoder)?;

        Self::check_bytes_validity(&bytes)
            .map_err(|err| bincode::error::DecodeError::OtherString(err.to_string()))?;

        Ok(Key(bytes))
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

#[cfg(test)]
mod tests {
    use proptest::proptest;

    use super::KEY_MAX_SIZE;
    use super::Key;

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

    #[test]
    fn key_decode_protects_key_too_large() {
        // NB the public api does not allow for such an invalid key.
        let key = Key(vec![0; KEY_MAX_SIZE + 1]);

        let bytes = octez_riscv_data::serialisation::serialise(&key)
            .expect("serialisation of a key should succeed");

        let res: Result<Key, _> = octez_riscv_data::serialisation::deserialise(&bytes);
        assert!(
            res.is_err(),
            "Decoding of a key that's too large should fail"
        );

        // TODO: RV-838: introduce and use a direct helper from `octez_riscv_data::deserialisation`
        //               for borrow-decode
        let res: Result<(Key, usize), _> = bincode::borrow_decode_from_slice(
            &bytes,
            octez_riscv_data::serialisation::bincode_default_config(),
        );
        assert!(
            res.is_err(),
            "Decoding of a key that's too large should fail"
        );
    }
}
