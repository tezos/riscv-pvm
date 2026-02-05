// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
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
use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;
use tezos_smart_rollup_constants::core::MAX_OUTPUT_SIZE;
use thiserror::Error;

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

/// Outbox state
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct Outbox<M: Mode> {
    levels: Box<[OutboxLevel<M>]>,
}

impl<M: AtomMode> Outbox<M> {
    /// Reset the outbox
    pub(crate) fn reset(&mut self) {
        for level in self.levels.iter_mut() {
            *level = OutboxLevel::default()
        }
    }
}

impl Outbox<Normal> {
    /// Return a proof-generating version of this outbox
    pub fn start_proof(&self) -> Outbox<Prove<'_>> {
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

#[perfect_derive(Clone, PartialEq, Eq)]
struct OutboxLevel<M: Mode> {
    messages: Box<[Atom<Vec<u8>, M>]>,
    next_index: Atom<u32, M>,
}

impl OutboxLevel<Normal> {
    /// Return a proof-generating version of this outbox level
    fn start_proof(&self) -> OutboxLevel<Prove<'_>> {
        let messages = self
            .messages
            .iter()
            .map(|m| m.start_proof())
            .collect::<Box<[_]>>();

        OutboxLevel {
            messages,
            next_index: self.next_index.start_proof(),
        }
    }
}

impl<M: AtomMode> Default for OutboxLevel<M> {
    fn default() -> Self {
        let mut messages = Vec::with_capacity(MAX_LEVEL_SIZE);
        messages.resize_with(MAX_LEVEL_SIZE, || Atom::<Vec<u8>, M>::new(Vec::new()));

        Self {
            messages: messages.into_boxed_slice(),
            next_index: Atom::default(),
        }
    }
}

impl<M: CloneAtomMode> CloneState for OutboxLevel<M> {
    fn clone_state(&self) -> Self {
        Self {
            messages: self.messages.clone_state(),
            next_index: self.next_index.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for OutboxLevel<M>
where
    M: Mode,
    F: Fold,
    Atom<Vec<u8>, M>: Foldable<F>,
    Atom<u32, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let message_generator = |idx| self.messages.index(idx);
        let mut builder = builder.into_node_fold();
        builder.add(&IndexableSeqAsTree::new(
            MAX_LEVEL_SIZE,
            LEVEL_MERKLE_ARITY,
            &message_generator,
        ));
        builder.add(&self.next_index);
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

        proof.done(OutboxLevel {
            messages,
            next_index,
        })
    }
}

impl<M: EncodeAtomMode> Encode for OutboxLevel<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.messages.encode(encoder)?;
        self.next_index.encode(encoder)?;
        Ok(())
    }
}

impl<C> Decode<C> for OutboxLevel<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let messages = Decode::decode(decoder)?;
        let next_index = Decode::decode(decoder)?;
        Ok(Self {
            messages,
            next_index,
        })
    }
}

#[derive(Error, Debug)]
pub(crate) enum OutboxError {
    #[error("Outbox message exceeds allowable size of {MAX_OUTPUT_SIZE}. Found: {size}")]
    OutboxMessageTooLarge { size: usize },
}

#[derive(Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct OutboxMessage([u8]);

impl OutboxMessage {
    /// Constructs a zeroed, boxed outbox message buffer of size `size`
    ///
    /// Fails if `size` exceeds [`MAX_OUTPUT_SIZE`]
    #[cfg_attr(not(test), expect(dead_code, reason = "outbox not in use"))]
    pub(crate) fn new(size: usize) -> Result<Box<Self>, OutboxError> {
        if size > MAX_OUTPUT_SIZE {
            return Err(OutboxError::OutboxMessageTooLarge { size });
        }
        let boxed_slice = vec![0u8; size].into_boxed_slice();
        let raw = Box::into_raw(boxed_slice) as *mut OutboxMessage;

        // Safety: Re-wrapping raw pointer back into a Box of equivalent
        // type (guaranteed by #[repr(transparent)])
        Ok(unsafe { Box::from_raw(raw) })
    }

    #[cfg(test)]
    fn from_boxed_slice(boxed_slice: Box<[u8]>) -> Result<Box<Self>, OutboxError> {
        if boxed_slice.len() > MAX_OUTPUT_SIZE {
            return Err(OutboxError::OutboxMessageTooLarge {
                size: boxed_slice.len(),
            });
        }
        let raw = Box::into_raw(boxed_slice) as *mut OutboxMessage;
        
        // Safety: Re-wrapping raw pointer back into a Box of equivalent
        // type (guaranteed by #[repr(transparent)])
        Ok(unsafe { Box::from_raw(raw) })
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

impl From<Box<OutboxMessage>> for Box<[u8]> {
    fn from(value: Box<OutboxMessage>) -> Self {
        // SAFETY: OutboxMessage's in memory layout is equivalent to
        // [u8] as guaranteed by #[repr(transparent)]
        unsafe { std::mem::transmute(value) }
    }
}
