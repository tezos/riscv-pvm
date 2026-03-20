// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a vector
//!
//! See [`Vector`] for more details.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::Index;
use std::ops::IndexMut;

use bincode::Encode;
use perfect_derive::perfect_derive;

use crate::clone::CloneState;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;

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

    type Prove<'normal> = ProveImpl;

    type Verify = VerifyImpl;
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

/// [`crate::mode::Prove`] mode implementation for the [`Vector`] component
struct ProveImpl {}

/// [`crate::mode::Verify`] mode implementation for the [`Vector`] component
struct VerifyImpl {}

/// Arity of internal nodes in the Merkle tree
const NODE_ARITY: usize = 4;
