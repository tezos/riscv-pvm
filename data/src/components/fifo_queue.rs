// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a FIFO queue
//!
//! See [`FifoQueue`] for more details.

use std::convert::Infallible;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use perfect_derive::perfect_derive;

use crate::clone::CloneState;
use crate::components::atom::Atom;
use crate::components::atom::AtomMode;
use crate::components::atom::CloneAtomMode;
use crate::components::atom::EncodeAtomMode;
use crate::components::vector::CloneVectorMode;
use crate::components::vector::EncodeVectorMode;
use crate::components::vector::Vector;
use crate::components::vector::VectorMode;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::foldable::Unfoldable;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::DeserialiserNode;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::SuspendedResult;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;

/// FIFO queue state component
///
/// The [`FifoQueue`] component holds an ordered sequence of values with FIFO semantics.
/// Internally, it is backed by a [`Vector`] (which grows monotonically) and an [`Atom`]
/// tracking the head index. Dequeued entries are logically removed by advancing the head.
///
/// This component participates fully in the proof/verification machinery.
#[perfect_derive(Debug)]
pub struct FifoQueue<T, M: Mode> {
    /// Index of the front of the queue within `items`
    head: Atom<u64, M>,
    /// All items ever enqueued (items before `head` are logically consumed)
    items: Vector<T, M>,
}

impl<T, M: AtomMode + VectorMode> FifoQueue<T, M> {
    /// Append `item` to the back of the queue.
    pub fn enqueue(&mut self, item: T) {
        let mut slot = Some(item);
        self.items
            .try_resize_with::<Infallible>(self.items.len() + 1, || {
                Ok(slot.take().expect("resize closure called exactly once"))
            })
            .unwrap();
    }

    /// Advance the head by one, logically consuming the front entry.
    ///
    /// Returns `true` if an entry was consumed, `false` if the queue was empty.
    pub fn advance(&mut self) -> bool {
        let head = self.head.read() as usize;
        if head >= self.items.len() {
            return false;
        }
        self.head.write(head as u64 + 1);
        true
    }

    /// Return a reference to the front entry without consuming it.
    ///
    /// Returns `None` if the queue is empty.
    pub fn front(&self) -> Option<&T> {
        let head = self.head.read() as usize;
        if head >= self.items.len() {
            None
        } else {
            Some(&self.items[head])
        }
    }

    /// Return the number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.items.len().saturating_sub(self.head.read() as usize)
    }

    /// Return `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone, M: AtomMode + VectorMode> FifoQueue<T, M> {
    /// Remove and return the front entry.
    ///
    /// Returns `None` if the queue is empty.
    pub fn dequeue(&mut self) -> Option<T> {
        let head = self.head.read() as usize;
        if head >= self.items.len() {
            return None;
        }
        let item = self.items[head].clone();
        self.head.write(head as u64 + 1);
        Some(item)
    }
}

impl<T, M: AtomMode + VectorMode> Default for FifoQueue<T, M> {
    fn default() -> Self {
        Self {
            head: Atom::default(),
            items: Vector::default(),
        }
    }
}

impl<T: PartialEq, M: AtomMode + VectorMode> PartialEq for FifoQueue<T, M> {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.items == other.items
    }
}

impl<T: Eq, M: AtomMode + VectorMode> Eq for FifoQueue<T, M> {}

impl<T: Clone, M: CloneAtomMode + CloneVectorMode> Clone for FifoQueue<T, M> {
    fn clone(&self) -> Self {
        Self {
            head: self.head.clone(),
            items: self.items.clone(),
        }
    }
}

impl<T: CloneState, M: CloneAtomMode + CloneVectorMode> CloneState for FifoQueue<T, M> {
    fn clone_state(&self) -> Self {
        Self {
            head: self.head.clone_state(),
            items: self.items.clone_state(),
        }
    }
}

impl<T, M, F> Foldable<F> for FifoQueue<T, M>
where
    M: Mode,
    F: Fold,
    Atom<u64, M>: Foldable<F>,
    Vector<T, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut node = builder.into_node_fold();
        node.add(&self.head);
        node.add(&self.items);
        node.done()
    }
}

impl<T: Unfoldable> Unfoldable for FifoQueue<T, Normal> {
    fn unfold<U: Unfold>(src: U) -> Result<Self, UnfoldError> {
        let mut src = src.into_node()?;
        let head = src.next_branch()?;
        let items = src.next_branch()?;
        src.done(Self { head, items })
    }
}

impl<T: FromProof> FromProof for FifoQueue<T, Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;
        let (proof, head) = proof.next_branch()?;
        let (proof, items) = proof.next_branch()?;
        proof.done(Self { head, items })
    }
}

impl<'normal, T: Provable<'normal>> Provable<'normal> for FifoQueue<T, Normal> {
    type Prover = FifoQueue<T::Prover, Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        FifoQueue {
            head: self.head.start_proof(),
            items: self.items.start_proof(),
        }
    }
}

impl<T: Encode, M: EncodeAtomMode + EncodeVectorMode> Encode for FifoQueue<T, M> {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.head.encode(encoder)?;
        self.items.encode(encoder)?;
        Ok(())
    }
}

impl<C, T: Decode<C>> Decode<C> for FifoQueue<T, Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            head: Decode::decode(decoder)?,
            items: Decode::decode(decoder)?,
        })
    }
}
