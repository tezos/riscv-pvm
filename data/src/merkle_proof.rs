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
    fn next_branch_with<T>(
        self,
        branch_deserialiser: impl FnOnce(Self::Parent) -> SuspendedResult<Self::Parent, T>,
    ) -> Result<(Self, T), <Self::Parent as Deserialiser>::Error>;

    /// The next branch of the current node is deserialised using the [`FromProof`] implementation
    /// of type `T`.
    fn next_branch<T: FromProof>(self) -> Result<(Self, T), <Self::Parent as Deserialiser>::Error> {
        self.next_branch_with(|deser| T::from_proof(deser))
    }

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

/// Trait for types that can be constructed from a Merkle proof
pub trait FromProof: Sized {
    /// Parse the given proof to construct an instance of `Self`.
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self>;
}

impl<A: FromProof, B: FromProof> FromProof for (A, B) {
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let proof = proof.into_node()?;

        let (proof, a) = proof.next_branch()?;
        let (proof, b) = proof.next_branch()?;

        proof.done((a, b))
    }
}

impl<Item: FromProof, const LEN: usize> FromProof for [Item; LEN] {
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let proof = proof.into_node()?;

        let mut items: [Option<Item>; LEN] = std::array::from_fn(|_| None);
        let proof = (0..LEN).try_fold(proof, |proof, index| {
            let (proof, item) = proof.next_branch()?;
            items[index] = Some(item);
            Ok(proof)
        })?;

        let items = items.map(|opt| opt.expect("All items should have been filled"));
        proof.done(items)
    }
}

/// Many items in a Merkle tree with given arity and length
pub struct Many<T, const ARITY: usize, const LEN: usize>(Box<[T; LEN]>);

impl<T, const ARITY: usize, const LEN: usize> Many<T, ARITY, LEN> {
    /// Turn this into the underlying boxed array.
    pub fn into_boxed_array(self) -> Box<[T; LEN]> {
        self.0
    }
}

impl<Item: FromProof, const ARITY: usize, const LEN: usize> FromProof for Many<Item, ARITY, LEN> {
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let mut leaves = Vec::with_capacity(LEN);

        let result = descend_tree(proof, ARITY, 0, LEN, &mut |_idx, proof| {
            let result = Item::from_proof(proof)?;
            let result = result.map(|leaf| {
                leaves.push(leaf);
            });
            Ok(result)
        })?;

        let Ok(boxed_array) = leaves.into_boxed_slice().try_into() else {
            panic!("Unexpected number of leaves collected")
        };

        let result = result.map(|_| Many(boxed_array));
        Ok(result)
    }
}

/// Descend a Merkle proof tree, calling `for_leaf` on each leaf encountered. The tree is described
/// to have a number of leaves equal to the given `leaves`. The node arity is given by `arity`.
/// Leaves are encountered in depth-first order left-to-right.
pub fn descend_tree<Proof, LeafHandler>(
    proof: Proof,
    arity: usize,
    start_leaf: usize,
    leaves: usize,
    for_leaf: &mut LeafHandler,
) -> SuspendedResult<Proof, ()>
where
    LeafHandler: FnMut(usize, Proof) -> SuspendedResult<Proof, ()>,
    Proof: Deserialiser,
{
    if leaves == 1 {
        return for_leaf(start_leaf, proof);
    }

    let ctx = proof.into_node()?;

    let mut child_start = start_leaf;
    let ctx = node_child_length(arity, leaves).try_fold(ctx, |ctx, child_leaves| {
        let (ctx, ()) = ctx.next_branch_with(|proof| {
            descend_tree(proof, arity, child_start, child_leaves, for_leaf)
        })?;

        child_start += child_leaves;

        Ok(ctx)
    })?;

    ctx.done(())
}

/// Compute the lengths covered by each child of a node.
fn node_child_length(arity: usize, length: usize) -> impl Iterator<Item = usize> {
    let child_max_length = length.div_ceil(arity);

    (0..arity).map(move |idx| {
        let start = idx * child_max_length;
        let end = ((idx + 1) * child_max_length).min(length);
        end.saturating_sub(start)
    })
}
