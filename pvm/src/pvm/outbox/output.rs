// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Types for outbox messages and their inclusion information which mirror Tezos
//! protocol types

use std::ops::Deref;
use std::ops::DerefMut;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::error::DecodeError;
use tezos_smart_rollup_constants::core::MAX_OUTPUT_SIZE;
use tezos_smart_rollup_constants::riscv::SbiError;
use thiserror::Error;

/// The outbox level and the index within that level for an outbox message
#[derive(Debug, PartialEq, Eq, Copy, Clone, Encode, Decode)]
pub struct OutputInfo {
    pub level: u32,
    pub index: u32,
}

/// A raw outbox message and its outbox information
#[derive(Debug, PartialEq, Eq)]
pub struct Output {
    pub message: OutboxMessage,
    pub info: OutputInfo,
}

/// Errors which can be raised when creating an outbox message
#[derive(Error, Debug, PartialEq, Eq)]
pub enum OutboxMessageError {
    #[error(
        "The size of the outbox message is {size} B, which is larger than the maximum message size ({MAX_OUTPUT_SIZE})."
    )]
    MessageTooLarge { size: usize },
}

impl From<OutboxMessageError> for SbiError {
    fn from(err: OutboxMessageError) -> Self {
        match err {
            OutboxMessageError::MessageTooLarge { .. } => Self::OutputTooLarge,
        }
    }
}

/// An outbox message is a boxed byte slice, restricted to at most [`MAX_OUTPUT_SIZE`]
/// in length
#[derive(Clone, Debug, PartialEq, Eq, Encode)]
#[repr(transparent)]
pub struct OutboxMessage(Box<[u8]>);

impl Default for OutboxMessage {
    fn default() -> Self {
        Self(Box::new([]))
    }
}

impl OutboxMessage {
    /// Constructs a zeroed, boxed outbox message buffer of size `size`
    ///
    /// Fails if `size` exceeds [`MAX_OUTPUT_SIZE`]
    pub(crate) fn new(size: usize) -> Result<Self, OutboxMessageError> {
        if size > MAX_OUTPUT_SIZE {
            return Err(OutboxMessageError::MessageTooLarge { size });
        }
        let boxed_slice = vec![0u8; size].into_boxed_slice();
        Ok(OutboxMessage(boxed_slice))
    }
}

impl TryFrom<Box<[u8]>> for OutboxMessage {
    type Error = OutboxMessageError;

    fn try_from(value: Box<[u8]>) -> Result<Self, Self::Error> {
        if value.len() > MAX_OUTPUT_SIZE {
            return Err(OutboxMessageError::MessageTooLarge { size: value.len() });
        }
        Ok(OutboxMessage(value))
    }
}

impl<Context> Decode<Context> for OutboxMessage {
    /// Start by decoding the length and fail without trying to decode the rest
    /// of the message if it is larger than `MAX_OUTPUT_SIZE`.
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let length = u64::decode(decoder)?;

        if length > MAX_OUTPUT_SIZE as u64 {
            return Err(DecodeError::OtherString(
                OutboxMessageError::MessageTooLarge {
                    size: length as usize,
                }
                .to_string(),
            ));
        }

        let mut data = vec![0u8; length as usize];
        decoder.reader().read(&mut data)?;

        Ok(OutboxMessage(data.into_boxed_slice()))
    }
}

impl From<OutboxMessage> for Box<[u8]> {
    fn from(value: OutboxMessage) -> Self {
        value.0
    }
}

impl AsRef<Box<[u8]>> for OutboxMessage {
    fn as_ref(&self) -> &Box<[u8]> {
        &self.0
    }
}

impl Deref for OutboxMessage {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OutboxMessage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_data::serialisation::serialise;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_oversized_outbox_message_is_rejected() {
        // Can't initialise an oversized outbox message
        proptest!(|(size in MAX_OUTPUT_SIZE + 1..)| {
            let res = OutboxMessage::new(size).unwrap_err();
            assert!(matches!(res.into(), SbiError::OutputTooLarge));
        });

        let size = MAX_OUTPUT_SIZE + 1;
        let message = vec![1u8; size].into_boxed_slice();

        // Can't construct an oversized outbox message from a slice
        assert_eq!(
            OutboxMessage::try_from(message.clone()),
            Err(OutboxMessageError::MessageTooLarge { size })
        );

        // Can't deserialise an oversized outbox message
        let serialised = serialise(OutboxMessage(message)).unwrap();
        let err = deserialise::<OutboxMessage>(&serialised)
            .expect_err("Decoding an oversized outbox message should fail");

        let DecodeError::OtherString(message) = err else {
            panic!("Expected a MessageTooLarge decode error, got {err:?}");
        };
        assert_eq!(
            message,
            OutboxMessageError::MessageTooLarge { size }.to_string()
        );
    }
}
