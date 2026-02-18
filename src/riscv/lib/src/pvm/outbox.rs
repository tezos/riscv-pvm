// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! A state component which can hold outbox messages produced by the kernel
//!
//! The outbox is configured with a certain size. Currently the only supported
//! size is [`TEST_OUTBOX_SIZE`]. The outbox holds messages produced
//! during the last [`TEST_OUTBOX_SIZE`] levels.
//!
//! The outbox supports:
//! - Adding a new message at the current level, if the level is not already full
//! - Retrieving all messages at a given level
//! - Producing an outbox proof for a given message
//! - Verifying an outbox proof

use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::serialise;
use perfect_derive::perfect_derive;
use tezos_smart_rollup_constants::core::MAX_OUTPUT_SIZE;
use tezos_smart_rollup_constants::riscv::SbiError;
use thiserror::Error;

use super::Pvm;
use super::durable_storage::DurableStorage;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::PageCache;

/// Small outbox size for testing
///
/// Currently, this is the length of the fixed-size array which holds all the outbox levels.
pub const TEST_OUTBOX_SIZE: usize = 16;

/// The maximum number of messages an outbox level can hold
///
/// Currently, this is the length of the fixed-size array which holds a level.
const MAX_LEVEL_SIZE: usize = 100;

/// The arity used to Merkleise the levels of the outbox
const OUTBOX_MERKLE_ARITY: usize = 2;

/// The arity used to Merkleise arrays in each level
const LEVEL_MERKLE_ARITY: usize = 2;

/// The outbox level and the index within that level for an outbox message
#[derive(Debug, PartialEq, Eq, Copy, Clone, Encode)]
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

/// Errors which can be raised when producing or verifying an outbox proof
#[derive(Error, Debug, PartialEq, Eq)]
pub enum OutboxProofError {
    #[error("The outbox does not contain the level {level}")]
    LevelNotFound { level: u32 },

    #[error("The outbox for level {} does not contain a message at index {}", info.level, info.index)]
    MessageNotFound { info: OutputInfo },

    #[error(transparent)]
    MessageError(#[from] OutboxMessageError),
}

/// Errors which can be raised when writing a message to the outbox
#[derive(Error, Debug)]
pub(crate) enum OutboxWriteError {
    #[error("Outbox is full")]
    FullOutbox,

    #[error(transparent)]
    MessageError(#[from] OutboxMessageError),
}

impl From<OutboxWriteError> for SbiError {
    fn from(err: OutboxWriteError) -> Self {
        match err {
            OutboxWriteError::FullOutbox => Self::FullOutbox,
            OutboxWriteError::MessageError(e) => e.into(),
        }
    }
}

/// An outbox proof, containing a partial Merkle tree of a PVM state which ties
/// an outbox message with the PVM state in which the outbox includes it
#[derive(Debug, Encode)]
pub struct OutboxProof {
    proof: MerkleProof,
    info: OutputInfo,
}

impl OutboxProof {
    /// Create a new outbox proof from the given Merkle proof and output information
    pub(crate) fn new(proof: MerkleProof, info: OutputInfo) -> Self {
        Self { proof, info }
    }

    /// Get the state hash of the outbox proof
    pub fn state_hash(&self) -> Hash {
        self.proof.root_hash()
    }

    /// Serialise the outbox proof
    pub fn serialise(&self) -> Vec<u8> {
        serialise(self).expect("Serialisation of an outbox proof should not fail")
    }
}

/// Outbox state
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct Outbox<M: Mode> {
    levels: Box<[OutboxLevel<M>]>,
}

impl<M: AtomMode> Outbox<M> {
    /// Write `message` to the outbox at the current level
    ///
    /// Returns `OutboxWriteError::FullOutbox` if the outbox is full.
    ///
    /// # Panics
    ///
    /// Panics if `current_level` is lt the last recorded level in the modded outbox
    /// level slot
    pub(crate) fn write_message(
        &mut self,
        message: OutboxMessage,
        current_level: u32,
    ) -> Result<(), OutboxWriteError> {
        let level_index = self.level_index(current_level);
        self.levels[level_index].write_message(message, current_level)
    }

    /// Get the internal index in the outbox corresponding to the given level
    fn level_index(&self, level: u32) -> usize {
        level as usize % self.levels.len()
    }

    /// Read the message associated with the given level and index from outbox
    fn read_message(&self, info: OutputInfo) -> Result<Output, OutboxProofError> {
        let level_index = self.level_index(info.level);
        let message = self.levels[level_index].read_message(info)?;
        Ok(Output {
            info,
            message: message.try_into()?,
        })
    }

    /// Get the number of levels stored in the outbox
    fn len(&self) -> usize {
        self.levels.len()
    }
}

impl Outbox<Normal> {
    /// Read outbox messages at the given level
    ///
    /// Warning: The caller must ensure that `level` is within the outbox
    /// validity window
    #[cfg_attr(not(test), expect(dead_code, reason = "outbox not in use"))]
    pub(crate) fn read_level(&self, level: u32) -> Box<[Box<[u8]>]> {
        let level_index = self.level_index(level);
        self.levels[level_index].read_level(level)
    }
}

impl<'normal> Provable<'normal> for Outbox<Normal> {
    type Prover = Outbox<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        let levels = self
            .levels
            .iter()
            .map(OutboxLevel::start_proof)
            .collect::<Box<[_]>>();

        Outbox { levels }
    }
}

impl<M: AtomMode> Default for Outbox<M> {
    fn default() -> Self {
        let mut levels = Vec::with_capacity(TEST_OUTBOX_SIZE);
        levels.resize_with(TEST_OUTBOX_SIZE, OutboxLevel::<M>::default);
        let levels = levels.into_boxed_slice();
        Self { levels }
    }
}

impl<M: CloneAtomMode> CloneState for Outbox<M> {
    fn clone_state(&self) -> Self {
        Self {
            levels: self.levels.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for Outbox<M>
where
    M: Mode,
    F: Fold,
    OutboxLevel<M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let level_generator = |idx| &self.levels[idx];
        IndexableSeqAsTree::new(TEST_OUTBOX_SIZE, OUTBOX_MERKLE_ARITY, &level_generator)
            .fold(builder)
    }
}

impl FromProof for Outbox<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let result =
            merkle_proof::Many::<_, OUTBOX_MERKLE_ARITY, TEST_OUTBOX_SIZE>::from_proof(proof)?;
        Ok(result.map(|arr| Outbox {
            levels: arr.into_boxed_array(),
        }))
    }
}

impl<M: EncodeAtomMode> Encode for Outbox<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.levels.encode(encoder)
    }
}

impl<C> Decode<C> for Outbox<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let levels = Decode::decode(decoder)?;
        Ok(Self { levels })
    }
}

impl<MC: MemoryConfig, PC: PageCache<MC, M>, DS: DurableStorage<M>, M: Mode> Pvm<MC, PC, DS, M> {
    /// Get the outbox message at the given level and index. This is the state transition
    /// captured in outbox proofs.
    pub fn get_outbox_message(&self, info: OutputInfo) -> Result<Output, OutboxProofError>
    where
        M: AtomMode,
    {
        // This check reads the current level which ensures it is also included in the
        // proof when running in `Prove` mode.
        self.check_level_in_outbox(info.level)?;

        self.outbox.read_message(info)
    }

    fn check_level_in_outbox(&self, level: u32) -> Result<(), OutboxProofError>
    where
        M: AtomMode,
    {
        // An uninitialised outbox contains no levels
        if !self.level_is_set.read() {
            return Err(OutboxProofError::LevelNotFound { level });
        }
        let current_level = self.level.read();

        // A future level is not in the outbox
        if level > current_level {
            return Err(OutboxProofError::LevelNotFound { level });
        }

        // Levels older than the size of the outbox are not in the outbox
        let oldest_outbox_level = current_level.saturating_sub(self.outbox.len() as u32 - 1);
        if level < oldest_outbox_level {
            return Err(OutboxProofError::LevelNotFound { level });
        }

        Ok(())
    }
}

#[perfect_derive(Clone, PartialEq, Eq)]
struct OutboxLevel<M: Mode> {
    messages: Box<[Atom<Box<[u8]>, M>]>,
    /// Next available message slot
    next_index: Atom<u32, M>,
    /// The level associated with this OutboxLevel
    level: Atom<u32, M>,
}

impl<M: AtomMode> OutboxLevel<M> {
    fn write_message(
        &mut self,
        message: OutboxMessage,
        current_level: u32,
    ) -> Result<(), OutboxWriteError> {
        let last_written_level = self.level.read();
        assert!(
            current_level >= last_written_level,
            "current_level {current_level} must be gte to any level already stored in the outbox. Found {last_written_level}"
        );

        if current_level > last_written_level {
            self.next_index.write(0);
            self.level.write(current_level);
        }

        let next_index = self.next_index.read() as usize;
        if next_index >= MAX_LEVEL_SIZE {
            return Err(OutboxWriteError::FullOutbox);
        }

        self.messages[next_index].write(message.0);
        self.next_index.write(next_index as u32 + 1);

        Ok(())
    }

    fn read_message(&self, info: OutputInfo) -> Result<Box<[u8]>, OutboxProofError> {
        if self.level.read() != info.level || info.index >= self.next_index.read() {
            return Err(OutboxProofError::MessageNotFound { info });
        }
        Ok(self.messages[info.index as usize].clone())
    }
}

impl OutboxLevel<Normal> {
    fn read_level(&self, level: u32) -> Box<[Box<[u8]>]> {
        let last_written_level = self.level.read();
        debug_assert!(
            level >= last_written_level,
            "level {level} must be gte to the last written level for this outbox level slot. Found {last_written_level}"
        );

        let next_index = self.next_index.read() as usize;
        if level != last_written_level || next_index == 0 {
            // The outbox is empty for `level`
            return Box::new([]);
        }

        self.messages[..next_index]
            .iter()
            .map(|msg| Box::from(msg.as_ref()))
            .collect::<Box<[_]>>()
    }
}

impl<'normal> Provable<'normal> for OutboxLevel<Normal> {
    type Prover = OutboxLevel<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        let messages = self
            .messages
            .iter()
            .map(|m| m.start_proof())
            .collect::<Box<[_]>>();

        OutboxLevel {
            messages,
            next_index: self.next_index.start_proof(),
            level: self.level.start_proof(),
        }
    }
}

impl<M: AtomMode> Default for OutboxLevel<M> {
    fn default() -> Self {
        let mut messages = Vec::with_capacity(MAX_LEVEL_SIZE);
        messages.resize_with(MAX_LEVEL_SIZE, || Atom::<Box<[u8]>, M>::new(Box::new([])));

        Self {
            messages: messages.into_boxed_slice(),
            next_index: Atom::default(),
            level: Atom::default(),
        }
    }
}

impl<M: CloneAtomMode> CloneState for OutboxLevel<M> {
    fn clone_state(&self) -> Self {
        Self {
            messages: self.messages.clone_state(),
            next_index: self.next_index.clone_state(),
            level: self.level.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for OutboxLevel<M>
where
    M: Mode,
    F: Fold,
    Atom<Box<[u8]>, M>: Foldable<F>,
    Atom<u32, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let message_generator = |idx| self.messages.index(idx);
        let mut builder = builder.into_node_fold();
        builder.add(&IndexableSeqAsTree::new(
            self.messages.len(),
            LEVEL_MERKLE_ARITY,
            &message_generator,
        ));
        builder.add(&self.next_index);
        builder.add(&self.level);
        builder.done()
    }
}

impl FromProof for OutboxLevel<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;
        let (proof, messages) = proof.next_branch_with(|p| {
            let result =
                merkle_proof::Many::<_, LEVEL_MERKLE_ARITY, MAX_LEVEL_SIZE>::from_proof(p)?;
            Ok(result.map(|arr| arr.into_boxed_array()))
        })?;

        let (proof, next_index) = proof.next_branch()?;
        let (proof, level) = proof.next_branch()?;

        proof.done(OutboxLevel {
            messages,
            next_index,
            level,
        })
    }
}

impl<M: EncodeAtomMode> Encode for OutboxLevel<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.messages.encode(encoder)?;
        self.next_index.encode(encoder)?;
        self.level.encode(encoder)?;
        Ok(())
    }
}

impl<C> Decode<C> for OutboxLevel<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let messages = Decode::decode(decoder)?;
        let next_index = Decode::decode(decoder)?;
        let level = Decode::decode(decoder)?;
        Ok(Self {
            messages,
            next_index,
            level,
        })
    }
}

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

/// An Outbox Message is a boxed byte slice, restricted to at most [`MAX_OUTPUT_SIZE`]
/// in length
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct OutboxMessage(Box<[u8]>);

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

impl From<OutboxMessage> for Box<[u8]> {
    fn from(message: OutboxMessage) -> Self {
        message.0
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
    use std::ops::Bound::*;
    use std::ops::RangeBounds;
    use std::ops::RangeInclusive;

    use itertools::Itertools;
    use proptest::prelude::*;

    use super::*;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::page_cache::EmptyPageCache;
    use crate::pvm::durable_storage::DurableStorageDummy;

    fn safe_size_range(size_range: impl RangeBounds<usize>) -> RangeInclusive<usize> {
        let start_bound = match size_range.start_bound() {
            Included(s) => *s,
            Excluded(s) => *s + 1,
            Unbounded => 0,
        };

        match size_range.end_bound() {
            Included(end) => start_bound..=MAX_OUTPUT_SIZE.min(*end),
            Excluded(end) => start_bound..=MAX_OUTPUT_SIZE.min(end.saturating_sub(1)),
            Unbounded => start_bound..=MAX_OUTPUT_SIZE,
        }
    }

    fn message_strategy(
        size_range: impl RangeBounds<usize>,
    ) -> impl Strategy<Value = OutboxMessage> {
        let safe_range = safe_size_range(size_range);
        proptest::collection::vec(any::<u8>(), safe_range)
            .prop_map(|data| OutboxMessage(data.into_boxed_slice()))
    }

    fn messages_strategy(
        size_range: impl RangeBounds<usize>,
        len: usize,
    ) -> impl Strategy<Value = Vec<OutboxMessage>> {
        proptest::collection::vec(message_strategy(size_range), len)
    }

    #[test]
    fn test_outbox_message_too_large() {
        let size = MAX_OUTPUT_SIZE + 1;
        assert_eq!(
            OutboxMessage::try_from(vec![1u8; size].into_boxed_slice()),
            Err(OutboxMessageError::MessageTooLarge { size })
        );

        proptest!(|(size in MAX_OUTPUT_SIZE + 1..)| {
            let res = OutboxMessage::new(size).unwrap_err();
            assert!(matches!(res.into(), SbiError::OutputTooLarge));
        })
    }

    #[test]
    fn write_messages_with_varying_sizes() {
        proptest!(|(
            messages in messages_strategy(0.., MAX_LEVEL_SIZE),
            level in 0u32..TEST_OUTBOX_SIZE as u32
        )| {
            let mut outbox = Outbox::<Normal>::default();
            let idx = level as usize % TEST_OUTBOX_SIZE;
            for (i, message) in messages.into_iter().enumerate() {
                prop_assert!(outbox.write_message(message.clone(), level).is_ok());

                prop_assert_eq!(&*outbox.levels[idx].messages[i], &message.0);
                prop_assert_eq!(*outbox.levels[idx].next_index, (i + 1) as u32);
                prop_assert_eq!(*outbox.levels[idx].level, level);
            }
        });
    }

    #[test]
    fn write_message_level_wrap() {
        proptest!(|(level in 0u32..TEST_OUTBOX_SIZE as u32)| {
            let mut outbox = Outbox::<Normal>::default();
            let first_message = OutboxMessage(Box::new([1u8; 32]));
            let second_message = OutboxMessage(Box::new([2u8; 32]));
            let idx = level as usize;

            // Write two messages at `level`
            assert!(outbox.write_message(first_message, level).is_ok());
            assert!(outbox.write_message(second_message, level).is_ok());

            assert_eq!(*outbox.levels[idx].next_index, 2);
            assert_eq!(*outbox.levels[idx].level, level);

            // Now write at `level` + TEST_OUTBOX_SIZE which should wrap to `level`
            let overflow_level = level + TEST_OUTBOX_SIZE as u32;
            let third_message = OutboxMessage(Box::new([3u8; 32]));
            assert!(outbox.write_message(third_message.clone(), overflow_level).is_ok());

            assert_eq!(&*outbox.levels[idx].messages[0], &third_message.0);
            assert_eq!(*outbox.levels[idx].next_index, 1);
            assert_eq!(*outbox.levels[idx].level, overflow_level);
        });
    }

    #[test]
    fn write_to_full_outbox_fails() {
        proptest!(|(messages in messages_strategy(0..=64, MAX_LEVEL_SIZE + 10))| {
            let mut outbox = Outbox::<Normal>::default();
             for (i, message) in messages.iter().enumerate().take(MAX_LEVEL_SIZE) {
                assert!(outbox.write_message(message.clone(), 0).is_ok());
                assert_eq!(*outbox.levels[0].next_index, (i + 1) as u32);
                assert_eq!(&*outbox.levels[0].messages[i], &message.0);
            }

            for message in &messages[MAX_LEVEL_SIZE..] {
                // Verify that outbox is full
                assert_eq!(*outbox.levels[0].next_index, MAX_LEVEL_SIZE as u32);

                prop_assert!(matches!(outbox.write_message(message.clone(), 0), Err(OutboxWriteError::FullOutbox)));
                prop_assert_eq!(*outbox.levels[0].next_index, MAX_LEVEL_SIZE as u32);
            }
        });
    }

    #[test]
    fn read_level_after_write() {
        proptest!(|(
            messages in proptest::collection::vec(message_strategy(1..), 1..MAX_LEVEL_SIZE),
            level in 0u32..1000
        )| {
            let mut outbox = Outbox::<Normal>::default();
            for msg in &messages {
                prop_assert!(outbox.write_message(msg.clone(), level).is_ok());
            }

            let read_messages = outbox.read_level(level);

            prop_assert_eq!(read_messages.len(), messages.len());
            for (i, msg) in messages.iter().enumerate() {
                prop_assert_eq!(read_messages[i].as_ref(), msg.as_ref() as &[u8]);
            }
        });
    }

    #[test]
    fn read_overwritten_slot_returns_new_level_data() {
        proptest!(|(
            messages1 in proptest::collection::vec(messages_strategy(0..=32, 50), TEST_OUTBOX_SIZE),
            messages2 in proptest::collection::vec(messages_strategy(0..=16, 10), TEST_OUTBOX_SIZE)
        )|{
            let mut outbox = Outbox::<Normal>::default();
            for (level, msgs) in messages1.iter().enumerate() {
                for msg in msgs {
                    prop_assert!(outbox.write_message(msg.clone(), level as u32).is_ok());
                }
            }

            for (offset, msgs) in messages2.iter().enumerate() {
                let wrap_level = TEST_OUTBOX_SIZE + offset;
                for msg in msgs {
                    prop_assert!(outbox.write_message(msg.clone(), wrap_level as u32).is_ok());
                }
                let read_messages = outbox.read_level(wrap_level as u32);
                let expected_messages: Box<[Box<[u8]>]> = Box::from(messages2[offset].clone().into_iter().map(|m|m.0).collect_vec());
                prop_assert_eq!(read_messages, expected_messages);
            }
        });
    }

    #[test]
    fn read_fresh_outbox_is_empty() {
        proptest!(|(level in 0u32..TEST_OUTBOX_SIZE as u32)| {
            let outbox = Outbox::<Normal>::default();
            let result = outbox.read_level(level);
            prop_assert_eq!(result.len(), 0)
        });
    }

    #[test]
    fn test_read_message() {
        proptest!(|(
            messages in messages_strategy(0.., 5),
            write_level in 0u32..1000
        )| {
            let mut outbox = Outbox::<Normal>::default();

            // Write messages at write_level
            for message in &messages {
                prop_assert!(outbox.write_message(message.clone(), write_level).is_ok());
            }

            // Read messages back
            for (i, message) in messages.iter().enumerate() {
                let info = OutputInfo {
                    level: write_level,
                    index: i as u32,
                };
                let output = outbox.read_message(info).unwrap();
                prop_assert_eq!(&*output.message, &*message.0);
                prop_assert_eq!(output.info, info);
            }
        });
    }

    #[test]
    fn test_read_message_with_invalid_index_fails() {
        proptest!(|(
            messages in messages_strategy(0.., 5),
            write_level in 0u32..1000,
            invalid_offset in 0usize..10
        )| {
            let mut outbox = Outbox::<Normal>::default();

            // Write N messages at write_level
            for message in &messages {
                prop_assert!(outbox.write_message(message.clone(), write_level).is_ok());
            }

            // Try to read with index >= N
            let invalid_index = messages.len() + invalid_offset;
            let info = OutputInfo {
                level: write_level,
                index: invalid_index as u32,
            };
            let output = outbox.read_message(info);
            prop_assert_eq!(output, Err(OutboxProofError::MessageNotFound { info }));
        });
    }

    #[test]
    fn test_read_message_after_level_wraparound() {
        proptest!(|(
            messages in messages_strategy(0.., 15),
            write_level in 0u32..1000
        )| {
            let mut outbox = Outbox::<Normal>::default();

            // Write messages at write_level
            for message in &messages {
                prop_assert!(outbox.write_message(message.clone(), write_level).is_ok());
            }

            // Try to read at the wrapped level without writing to it first
            // The wrapped level maps to the same outbox slot but differs from the stored level
            let wrapped_level = write_level + TEST_OUTBOX_SIZE as u32;
            for i in 0..messages.len() {
                let info = OutputInfo {
                    level: wrapped_level,
                    index: i as u32,
                };
                let output = outbox.read_message(info);
                prop_assert_eq!(output, Err(OutboxProofError::MessageNotFound { info }));
            }
        });
    }

    #[test]
    fn test_read_message_from_empty_level_fails() {
        proptest!(|(
            level in 0u32..1000,
            index in 0u32..MAX_LEVEL_SIZE as u32
        )| {
            let outbox = Outbox::<Normal>::default();

            let info = OutputInfo { level, index };
            let output = outbox.read_message(info);
            prop_assert_eq!(output, Err(OutboxProofError::MessageNotFound { info }));
        });
    }

    #[test]
    fn test_get_outbox_message_from_future_level_fails() {
        proptest!(|(
            messages in messages_strategy(0.., 5),
            write_level in 0u32..1000
        )| {
            type MC = M1M;
            type PC = EmptyPageCache;
            type DS = DurableStorageDummy;

            let mut pvm = Pvm::<MC, PC, DS, Normal>::default();

            // Getting a message from an uninitialised outbox fails
            let info = OutputInfo { level: 0, index: 0 };
            let output = pvm.get_outbox_message(info);
            prop_assert_eq!(output, Err(OutboxProofError::LevelNotFound { level: info.level }));

            pvm.level_is_set.write(true);
            pvm.level.write(write_level);

            // Write messages at write_level
            for message in &messages {
                prop_assert!(pvm.outbox.write_message(message.clone(), write_level).is_ok());
            }

            // Getting a message at a future level fails
            let info = OutputInfo {
                level: write_level + 1,
                index: 0,
            };
            let output = pvm.get_outbox_message(info);
            prop_assert_eq!(output, Err(OutboxProofError::LevelNotFound { level: info.level }));
        })
    }

    #[test]
    fn test_get_outbox_message_from_valid_level() {
        proptest!(|(
            messages in messages_strategy(0.., 5),
            write_level in 0u32..1000
        )| {
            type MC = M1M;
            type PC = EmptyPageCache;
            type DS = DurableStorageDummy;

            let mut pvm = Pvm::<MC, PC, DS, Normal>::default();

            pvm.level_is_set.write(true);
            pvm.level.write(write_level);

            // Write messages at write_level
            for message in &messages {
                prop_assert!(pvm.outbox.write_message(message.clone(), write_level).is_ok());
            }

            // Read messages back at write_level
            for (i, message) in messages.iter().enumerate() {
                let info = OutputInfo {
                    level: write_level,
                    index: i as u32,
                };
                let output = pvm.get_outbox_message(info).unwrap();
                prop_assert_eq!(&output.message, message);
                prop_assert_eq!(output.info, info);
            }

            // Also verify we can read with current_level up to write_level + TEST_OUTBOX_SIZE - 1
            let future_level = write_level + (TEST_OUTBOX_SIZE as u32) - 1;
            pvm.level.write(future_level);

            for (i, message) in messages.iter().enumerate() {
                let info = OutputInfo {
                    level: write_level,
                    index: i as u32,
                };
                let output = pvm.get_outbox_message(info).unwrap();
                prop_assert_eq!(&output.message, message);
            }
        });
    }

    #[test]
    fn test_get_outbox_message_from_old_level_fails() {
        proptest!(|(
            first_messages in messages_strategy(0.., 15),
            second_messages in messages_strategy(0.., 5),
            write_level in TEST_OUTBOX_SIZE as u32..1000
        )| {
            type MC = M1M;
            type PC = EmptyPageCache;
            type DS = DurableStorageDummy;

            let mut pvm = Pvm::<MC, PC, DS, Normal>::default();

            let m = first_messages.len();
            let n = second_messages.len();

            // Write M messages at write_level
            for message in &first_messages {
                prop_assert!(pvm.outbox.write_message(message.clone(), write_level).is_ok());
            }

            // Write N messages at write_level + TEST_OUTBOX_SIZE (where N < M)
            let wrapped_level = write_level + TEST_OUTBOX_SIZE as u32;
            for message in &second_messages {
                prop_assert!(pvm.outbox.write_message(message.clone(), wrapped_level).is_ok());
            }

            // Set up PVM level at the wrapped level
            pvm.level_is_set.write(true);
            pvm.level.write(wrapped_level);

            // Reading at old level should fail
            for i in 0..m {
                let info = OutputInfo {
                    level: write_level,
                    index: i as u32,
                };
                let output = pvm.get_outbox_message(info);
                prop_assert_eq!(output, Err(OutboxProofError::LevelNotFound { level: info.level }))
            }

            // Reading at wrapped level for indices 0..N should work
            for (i, message) in second_messages.iter().enumerate() {
                let info = OutputInfo {
                    level: wrapped_level,
                    index: i as u32,
                };
                let output = pvm.get_outbox_message(info).unwrap();
                prop_assert_eq!(&output.message, message);
            }

            // Reading at wrapped level for indices N..M should fail
            for i in n..m {
                let info = OutputInfo {
                    level: wrapped_level,
                    index: i as u32,
                };
                let output = pvm.get_outbox_message(info);
                prop_assert_eq!(output, Err(OutboxProofError::MessageNotFound { info }));
            }
        });
    }
}
