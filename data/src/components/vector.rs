// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a vector
//!
//! See [`Vector`] for more details.

use std::cell::Cell;
use std::cell::RefCell;
use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::Index;
use std::ops::IndexMut;

use bincode::Encode;
use perfect_derive::perfect_derive;
use range_collections::RangeSet2;

use crate::clone::CloneState;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::seq_tree::DepthAdjustedSeqAsTree;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::Partial;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::SuspendedResult;
use crate::merkle_proof::proof_tree::ForceMinimumPresence;
use crate::merkle_proof::proof_tree::MerkleProofFold;
use crate::merkle_proof::proof_tree::MinimumPresence;
use crate::merkle_proof::sequence_as_tree_from_proof;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::not_found;
use crate::partial_vec::PartialVec;
use crate::serialisation::serialise;

/// Vector state component
///
/// The [`Vector`] component holds a sequence of values of type `T`.
#[perfect_derive(Debug)]
pub struct Vector<T, M: Mode> {
    vector: M::Select<VectorTemplate<T>>,
}

impl<T, M: VectorMode> Vector<T, M> {
    /// Create a new vector from the given values.
    pub fn new(values: Vec<T>) -> Self {
        M::new(values)
    }

    /// Get the number of items the vector holds.
    pub fn len(&self) -> usize {
        M::len(self)
    }

    /// Is the vector empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resize the vector to the new length, filling new items with the given value.
    pub fn resize_with(&mut self, new_len: usize, new_value: impl FnMut() -> T) {
        M::resize_with(self, new_len, new_value)
    }
}

impl<T, M: VectorMode> Index<usize> for Vector<T, M> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        M::index(self, idx)
    }
}

impl<T, M: VectorMode> IndexMut<usize> for Vector<T, M> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        M::index_mut(self, idx)
    }
}

impl<T, M: VectorMode> Default for Vector<T, M> {
    fn default() -> Self {
        M::new(Vec::new())
    }
}

impl<T: Clone, M: CloneVectorMode> CloneState for Vector<T, M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl<T: Foldable<HashFold>> Foldable<HashFold> for Vector<T, Normal> {
    fn fold(&self, builder: HashFold) -> Hash {
        let mut node = builder.into_node_fold();

        let length = self.vector.len();
        let length_node =
            Hash::hash_encodable(length as u64).expect("Hashing length should not fail");
        node.add(&length_node);

        let get_item = |idx: usize| &self.vector[idx];
        let seq_as_tree = IndexableSeqAsTree::new(length, NODE_ARITY, &get_item);
        node.add(&seq_as_tree);

        node.done()
    }
}

impl<T: Foldable<HashFold>> Foldable<HashFold> for Vector<T, Prove<'_>> {
    fn fold(&self, builder: HashFold) -> Hash {
        let mut node = builder.into_node_fold();

        let length = self.vector.unrecorded_len();
        let length_node =
            Hash::hash_encodable(length as u64).expect("Hashing length should not fail");
        node.add(&length_node);

        let get_item = |idx: usize| self.vector.unrecorded_index(idx);
        let seq_as_tree = IndexableSeqAsTree::new(length, NODE_ARITY, &get_item);
        node.add(&seq_as_tree);

        node.done()
    }
}

impl<T: Foldable<MerkleProofFold>> Foldable<MerkleProofFold> for Vector<T, Prove<'_>> {
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        // Reminder: Merkle trees generated in Prove mode capture the state at beginning of proof
        // generation. This means we need to use `previous` state for the length and data.

        let mut node = builder.into_node_fold();

        let length = self.vector.previous.len();
        let length_data = serialise(length as u64).expect("Serialising length should not fail");
        let is_length_needed = self.vector.need_length_in_proof();
        let length_constraint = if is_length_needed {
            MinimumPresence::Present
        } else {
            MinimumPresence::MayOmit
        };
        let length_node = MerkleProofFold::new_leaf(length_constraint, length_data);
        node.add(&length_node);

        let get_item = |idx: usize| ForceMinimumPresence {
            min_constraint: if self.vector.has_accessed(idx) {
                MinimumPresence::MayBlind
            } else {
                MinimumPresence::MayOmit
            },
            inner: &self.vector.previous[idx],
        };
        let seq_as_tree = IndexableSeqAsTree::new(length, NODE_ARITY, &get_item);
        node.add(&seq_as_tree);

        node.done()
    }
}

impl<T: Foldable<PartialHashFold>> Foldable<PartialHashFold> for Vector<T, Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        if self.vector.is_completely_absent() {
            return builder.previous();
        }

        let Some(original_length) = self.vector.original_length.clone().to_present() else {
            return PartialHash::InvalidProof;
        };

        let Some(length) = self.vector.length.clone().to_present() else {
            return PartialHash::InvalidProof;
        };

        let mut node = builder.into_node_fold();

        let length_hash = Hash::hash_encodable(length as u64).expect("Hashing should not fail");
        let length_node = PartialHash::Present(length_hash);
        node.add(&length_node);

        let get_item = |index| {
            self.vector
                .items
                .get(index)
                .map(Partial::Present)
                .unwrap_or(Partial::Absent)
        };

        node.add(&DepthAdjustedSeqAsTree {
            inner: IndexableSeqAsTree::new(length, NODE_ARITY, &get_item),
            // `IndexableSeqAsTree` has a special-case layout where a single-element sequence is a
            // bare leaf and all other lengths are wrapped in at least one node. We encode that in
            // the adjusted depth by adding one level for all non-singleton lengths.
            original_depth: original_length
                .saturating_sub(1)
                .checked_ilog(NODE_ARITY)
                .unwrap_or(0)
                .saturating_add(u32::from(original_length != 1)),
            current_depth: length
                .saturating_sub(1)
                .checked_ilog(NODE_ARITY)
                .unwrap_or(0)
                .saturating_add(u32::from(length != 1)),
        });

        node.done()
    }
}

impl<'normal, T: Provable<'normal>> Provable<'normal> for Vector<T, Normal> {
    type Prover = Vector<T::Prover, Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        let previous = self
            .vector
            .iter()
            .map(Provable::start_proof)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Vector {
            vector: ProveImpl {
                active_previous: previous.len(),
                previous,
                accessed_indices: Default::default(),
                appended: Vec::new(),
                read_length: Cell::new(false),
            },
        }
    }
}

impl<T: FromProof> FromProof for Vector<T, Verify> {
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let with_length = |length: Partial<u64>| {
            let length = length.map_present(|len: u64| len as usize);
            let state = Vector {
                vector: VerifyImpl {
                    original_length: length.clone(),
                    length: length.clone(),
                    items: PartialVec::empty(),
                },
            };
            (state, length)
        };

        let with_item = |state: &mut Vector<T, Verify>, idx, proof| {
            let result: Proof::Suspended<T> = T::from_proof(proof)?;
            let result = result.map(|item: T| {
                state.vector.items.define(idx, vec![item]);
            });
            Ok(result)
        };

        sequence_as_tree_from_proof(proof, NODE_ARITY, with_length, with_item)
    }
}

impl<T: Clone, M: CloneVectorMode> Clone for Vector<T, M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

impl<T: Encode, M: EncodeVectorMode> Encode for Vector<T, M> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        M::encode(self, encoder)
    }
}

/// Modal template for the [`Vector`] component
///
/// This is used to select the appropriate implementation for the mode.
struct VectorTemplate<T: ?Sized>(PhantomData<T>, Infallible);

impl<T> Modal for VectorTemplate<T> {
    type Normal = Vec<T>;

    type Prove<'normal> = ProveImpl<T>;

    type Verify = VerifyImpl<T>;
}

/// Mode types that implement this trait support common operations on the [`Vector`] component.
///
/// See [`Vector`] for a more convenient interface.
pub trait VectorMode: Mode {
    /// See [`Vector::new`].
    fn new<T>(values: Vec<T>) -> Vector<T, Self>;

    /// See [`Vector::index`].
    fn index<T>(this: &Vector<T, Self>, idx: usize) -> &T;

    /// See [`Vector::index_mut`].
    fn index_mut<T>(this: &mut Vector<T, Self>, idx: usize) -> &mut T;

    /// See [`Vector::len`].
    fn len<T>(this: &Vector<T, Self>) -> usize;

    /// See [`Vector::resize_with`].
    fn resize_with<T>(this: &mut Vector<T, Self>, new_len: usize, value: impl FnMut() -> T);
}

impl VectorMode for Normal {
    fn new<T>(vector: Vec<T>) -> Vector<T, Self> {
        Vector { vector }
    }

    fn index<T>(this: &Vector<T, Self>, idx: usize) -> &T {
        &this.vector[idx]
    }

    fn index_mut<T>(this: &mut Vector<T, Self>, idx: usize) -> &mut T {
        &mut this.vector[idx]
    }

    fn len<T>(this: &Vector<T, Self>) -> usize {
        this.vector.len()
    }

    fn resize_with<T>(this: &mut Vector<T, Self>, new_len: usize, value: impl FnMut() -> T) {
        this.vector.resize_with(new_len, value);
    }
}

impl VectorMode for Prove<'_> {
    fn new<T>(values: Vec<T>) -> Vector<T, Self> {
        Vector {
            vector: ProveImpl {
                active_previous: values.len(),
                previous: values.into_boxed_slice(),
                accessed_indices: Default::default(),
                appended: Vec::new(),
                read_length: Cell::new(false),
            },
        }
    }

    fn index<T>(this: &Vector<T, Self>, idx: usize) -> &T {
        this.vector.record_access(idx);

        if idx < this.vector.active_previous {
            &this.vector.previous[idx]
        } else {
            &this.vector.appended[idx - this.vector.active_previous]
        }
    }

    fn index_mut<T>(this: &mut Vector<T, Self>, idx: usize) -> &mut T {
        this.vector.record_access(idx);

        if idx < this.vector.active_previous {
            &mut this.vector.previous[idx]
        } else {
            &mut this.vector.appended[idx - this.vector.active_previous]
        }
    }

    fn len<T>(this: &Vector<T, Self>) -> usize {
        this.vector.read_length.set(true);
        this.vector.unrecorded_len()
    }

    fn resize_with<T>(this: &mut Vector<T, Self>, new_len: usize, mut value: impl FnMut() -> T) {
        let current_len = this.len();

        this.vector.record_resize_boundary(new_len);

        // Increasing the size simply requires us to add items to the end, i.e. add them to `extra`.
        if new_len > current_len {
            let growth = new_len - current_len;
            this.vector.appended.reserve(growth);
            this.vector.appended.extend((0..growth).map(|_| value()));
            return;
        }

        // When not shrinking below the number of active previous items, we can truncate `extra`.
        if new_len >= this.vector.active_previous {
            this.vector
                .appended
                .truncate(new_len - this.vector.active_previous);
            return;
        }

        // When shrinking below the number of active previous items, we need to clear `extra` as
        // those are now invalid, and shrink `active_previous` to reflect the new size.
        this.vector.active_previous = new_len;
        this.vector.appended.clear();
    }
}

impl VectorMode for Verify {
    fn new<T>(values: Vec<T>) -> Vector<T, Self> {
        Vector {
            vector: VerifyImpl {
                original_length: Partial::Present(values.len()),
                length: Partial::Present(values.len()),
                items: PartialVec::from(values),
            },
        }
    }

    fn index<T>(this: &Vector<T, Self>, idx: usize) -> &T {
        match this.vector.items.get(idx) {
            Some(item) => item,
            None => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            }
        }
    }

    fn index_mut<T>(this: &mut Vector<T, Self>, idx: usize) -> &mut T {
        match this.vector.items.get_mut(idx) {
            Some(item) => item,
            None => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            }
        }
    }

    fn len<T>(this: &Vector<T, Self>) -> usize {
        match this.vector.length {
            Partial::Present(len) => len,
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            }
        }
    }

    fn resize_with<T>(this: &mut Vector<T, Self>, new_len: usize, mut value: impl FnMut() -> T) {
        let current_len = this.len();

        // If there is anything to grow, we need to ensure the items are defined for the range that
        // we're growing.
        if new_len > current_len {
            let growth = new_len - current_len;
            let values = Vec::from_iter((0..growth).map(|_| value()));
            this.vector.items.define(current_len, values);
        }

        // When shrinking, the excessive items need to be truncated.
        if new_len < current_len {
            this.vector.items.truncate(new_len);
        }

        this.vector.length = Partial::Present(new_len);
    }
}

/// Mode types that implement this trait support cloning of the [`Vector`] component.
pub trait CloneVectorMode: Mode {
    /// Clones the given [`Vector`] component.
    ///
    /// This clones the entire component, not just the internal value. Consider this when cloning
    /// components in [`crate::mode::Prove`] mode.
    fn clone<T: Clone>(this: &Vector<T, Self>) -> Vector<T, Self>;
}

impl CloneVectorMode for Normal {
    fn clone<T: Clone>(this: &Vector<T, Self>) -> Vector<T, Self> {
        Vector {
            vector: this.vector.clone(),
        }
    }
}

impl CloneVectorMode for Prove<'_> {
    fn clone<T: Clone>(this: &Vector<T, Self>) -> Vector<T, Self> {
        Vector {
            vector: this.vector.clone(),
        }
    }
}

impl CloneVectorMode for Verify {
    fn clone<T: Clone>(this: &Vector<T, Self>) -> Vector<T, Self> {
        Vector {
            vector: this.vector.clone(),
        }
    }
}

/// Mode types that implement this trait support encoding of the [`Vector`] component.
pub trait EncodeVectorMode: Mode {
    /// Encodes the [`Vector`] component as a vector of items.
    fn encode<T: Encode, E: bincode::enc::Encoder>(
        vector: &Vector<T, Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError>;
}

impl EncodeVectorMode for Normal {
    fn encode<T: Encode, E: bincode::enc::Encoder>(
        vector: &Vector<T, Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        vector.vector.encode(encoder)
    }
}

impl EncodeVectorMode for Prove<'_> {
    fn encode<T: Encode, E: bincode::enc::Encoder>(
        vector: &Vector<T, Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let len = vector.vector.unrecorded_len();
        len.encode(encoder)?;

        for idx in 0..len {
            vector.vector.unrecorded_index(idx).encode(encoder)?;
        }

        Ok(())
    }
}

/// [`crate::mode::Prove`] mode implementation for the [`Vector`] component
#[perfect_derive(Clone)]
struct ProveImpl<T> {
    /// Items at the time of starting the proof generation
    ///
    /// This collection must not be resized, because it is important to use as-is in the Merkle tree
    /// generation.
    previous: Box<[T]>,

    /// Number of active items from the `previous` field
    active_previous: usize,

    /// Indices that were accessed during the proof generation
    accessed_indices: RefCell<RangeSet2<usize>>,

    /// New items that were added after the proof generation started, or after a resize
    ///
    /// The first extra item is located at index `active_previous`.
    appended: Vec<T>,

    /// Was the length requested?
    ///
    /// This field is used to determine whether the node in the Merkle tree that holds the length
    /// needs to be present or not. This cell may contain `false` but the length node might still be
    /// present in the Merkle tree - there are other means to influence the length node presence.
    read_length: Cell<bool>,
}

impl<T> ProveImpl<T> {
    /// Returns a reference to the item at index `idx` without recording it as accessed.
    fn unrecorded_index(&self, idx: usize) -> &T {
        if idx < self.active_previous {
            &self.previous[idx]
        } else {
            &self.appended[idx - self.active_previous]
        }
    }

    /// Returns the length of the vector without recording any indices as accessed.
    fn unrecorded_len(&self) -> usize {
        self.appended.len().saturating_add(self.active_previous)
    }

    /// Records that the item at index `idx` has been accessed.
    fn record_access(&self, idx: usize) {
        let range = idx..idx.checked_add(1).expect("Index must not be max");
        let range = RangeSet2::from(range);
        self.accessed_indices.borrow_mut().union_with(&range);
    }

    /// Records that the vector has been resized to `new_len`, updating the accessed indices.
    fn record_resize_boundary(&self, new_len: usize) {
        let prev_len = self.unrecorded_len();
        let boundary_pos = new_len.min(prev_len);

        if boundary_pos == 0 || prev_len == new_len {
            return;
        }

        self.record_access(boundary_pos - 1)
    }

    /// Returns `true` if the item at index `idx` has been accessed.
    fn has_accessed(&self, idx: usize) -> bool {
        self.accessed_indices.borrow().contains(&idx)
    }

    /// Returns `true` if the length of the vector needs to be included in the proof.
    fn need_length_in_proof(&self) -> bool {
        self.read_length.get() || !self.accessed_indices.borrow().is_empty()
    }
}

/// [`crate::mode::Verify`] mode implementation for the [`Vector`] component
#[perfect_derive(Clone)]
struct VerifyImpl<T> {
    /// Original length of the vector
    original_length: Partial<usize>,

    /// Current length of the vector
    length: Partial<usize>,

    /// Items in the vector
    items: PartialVec<T>,
}

impl<T> VerifyImpl<T> {
    fn is_completely_absent(&self) -> bool {
        if let Partial::Present(_) = self.length {
            return false;
        }

        self.items.is_all_undefined()
    }
}

/// Arity of internal nodes in the Merkle tree
const NODE_ARITY: usize = 4;
