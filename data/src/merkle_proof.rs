// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Merkle proofs

pub mod proof_tree;
pub mod tag;
pub mod transform;

use std::error;

use bincode::Decode;

use crate::hash::Hash;

/// Possible outcomes when parsing a node or a leaf from a Merkle proof
/// where the leaf is assumed to have type `T`.
#[derive(Clone)]
pub enum Partial<T> {
    /// The leaf or node is absent from the proof.
    Absent,

    /// A blinded subtree and its [`struct@Hash`] is provided.
    Blinded(Hash),

    /// Data successfully parsed and its type is `T`.
    Present(T),
}

impl<T> Partial<T> {
    /// Map the present result of a [`Partial<T>`] into [`Partial<R>`].
    pub fn map_present<R>(self, f: impl FnOnce(T) -> R) -> Partial<R> {
        match self {
            Partial::Absent => Partial::Absent,
            Partial::Blinded(hash) => Partial::Blinded(hash),
            Partial::Present(data) => Partial::Present(f(data)),
        }
    }

    /// Same as [`Partial::map_present`] but can fail.
    pub fn map_present_fallible<R, E>(
        self,
        f: impl FnOnce(T) -> Result<R, E>,
    ) -> Result<Partial<R>, E> {
        match self {
            Partial::Absent => Ok(Partial::Absent),
            Partial::Blinded(hash) => Ok(Partial::Blinded(hash)),
            Partial::Present(data) => Ok(Partial::Present(f(data)?)),
        }
    }

    /// Convert a [`Partial<T>`] into an [`Option<T>`], discarding blinded and absent cases.
    pub fn to_present(self) -> Option<T> {
        match self {
            Partial::Present(data) => Some(data),
            Partial::Absent | Partial::Blinded(_) => None,
        }
    }
}

/// Error type that can occur during proof deserialisation
pub trait DeserialiserError: error::Error {
    /// Create a custom deserialiser error from any error type.
    fn custom<E: error::Error + 'static>(error: E) -> Self;
}

/// [`Deserialiser::Suspended<T>`] wrapped in a [`Result`].
///
/// Clippy yells at us for making the type signatures in [`Deserialiser`] too complex, so we
/// provide some aliases to simplify them on the surface level.
pub type SuspendedResult<D, T> =
    Result<<D as Deserialiser>::Suspended<T>, <D as Deserialiser>::Error>;

/// The main trait used for deserialising a proof.
///
/// Having an object of this trait is equivalent to having a proof and being able to deserialise it.
///
/// A proof can be interpreted in 3 cases:
/// 1. [`Deserialiser::into_leaf_raw`] The proof is a leaf and raw bytes are obtained.
/// 2. [`Deserialiser::into_leaf<T>`] The proof is a leaf and the type `T` is parsed.
/// 3. [`Deserialiser::into_node`] The proof is a node in the tree.
pub trait Deserialiser {
    /// Error type
    type Error: DeserialiserError;

    /// After deserialising a proof, a [`Suspended<R>`] computation is obtained.
    type Suspended<R>: Suspended<Output = R, Parent = Self>;

    /// In case the proof is a node, [`Deserialiser::DeserialiserNode`] is the deserialiser for the branch case.
    type DeserialiserNode: DeserialiserNode<Parent = Self>;

    /// It is expected for the proof to be a leaf. Obtain the raw bytes from that leaf.
    fn into_leaf_raw<const LEN: usize>(self) -> SuspendedResult<Self, Partial<Box<[u8; LEN]>>>;

    /// It is expected for the proof to be a leaf. Parse the raw bytes of that leaf into a type `T`.
    fn into_leaf<T: Decode<()>>(self) -> SuspendedResult<Self, Partial<T>>;

    /// It is expected for the proof to be a node. Obtain the deserialiser for the branch case.
    fn into_node(self) -> Result<Self::DeserialiserNode, Self::Error>;
}

/// The trait used for deserialising a proof's node.
/// Having an object of this trait is equivalent to knowing the current proof is a node.
/// Deserialisers for each of its branches are expected to be provided to continue the deserialisation.
pub trait DeserialiserNode: Sized {
    type Parent: Deserialiser;

    /// Get the presence information for the node that is being parsed.
    fn presence(&self) -> Partial<()>;

    /// The next branch of the current node is deserialised using the provided deserialiser
    /// `branch_deserialiser`.
    fn next_branch<T>(
        self,
        branch_deserialiser: impl FnOnce(Self::Parent) -> SuspendedResult<Self::Parent, T>,
    ) -> Result<(Self, T), <Self::Parent as Deserialiser>::Error>;

    /// Signal the end of deserialisation of the node's branches.
    /// Call this method after all calls to [`DeserialiserNode::next_branch`] have been made.
    fn done<T>(self, value: T) -> SuspendedResult<Self::Parent, T>;
}

/// The trait represents a computation function obtained after deserialising a proof.
pub trait Suspended {
    /// End result of the computation.
    type Output;

    type Parent: Deserialiser;

    /// Helper to map the current result into a new type.
    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T,
    ) -> <Self::Parent as Deserialiser>::Suspended<T>;
}
