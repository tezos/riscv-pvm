// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Foldable data structures

use std::error;

use bincode::Decode;

pub mod seq_tree;

/// Implementing types which define a state structure that can be folded
///
/// The `F` parameter is used to describe the folding capabilities. Implementing types will commonly
/// impose constraints on `F` to ensure it has the necessary methods to perform the folding.
///
/// For example, if the the type implements a tree node-like structure, `F` may be required to
/// implement the [`NodeFold`] trait that provides methods for combining child nodes into a parent
/// node. This way, the implementing type can tell `F` about its internal structure.
///
/// Implementing types can also pick concrete `F` types to provide specific folding behaviours.
/// Such concrete types could be [`crate::hash::Hash`] for leaves in a state data structure, or
/// [`crate::merkle_tree::MerkleTree`] for things in [`crate::mode::Prove`] mode.
pub trait Foldable<F: Fold> {
    /// Fold the state data structure.
    fn fold(&self, builder: F) -> F::Folded;
}

impl<T: Foldable<F>, F: Fold> Foldable<F> for &T {
    fn fold(&self, builder: F) -> F::Folded {
        T::fold(self, builder)
    }
}

impl<A: Foldable<F>, B: Foldable<F>, F: Fold> Foldable<F> for (A, B) {
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.0);
        builder.add(&self.1);
        builder.done()
    }
}

impl<A: Foldable<F>, B: Foldable<F>, C: Foldable<F>, F: Fold> Foldable<F> for (A, B, C) {
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.0);
        builder.add(&self.1);
        builder.add(&self.2);
        builder.done()
    }
}

impl<T: Foldable<F>, const LEN: usize, F: Fold> Foldable<F> for [T; LEN] {
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();

        for item in self.iter() {
            builder.add(item);
        }

        builder.done()
    }
}

/// Implementing types describe a folding scheme
pub trait Fold {
    /// Result of the folding operations
    type Folded;

    /// Corresponding node-folding type
    ///
    /// This type provides additional methods for folding nodes.
    type NodeFold: NodeFold<Parent = Self>;

    /// Start folding a node.
    fn into_node_fold(self) -> Self::NodeFold;
}

/// Type with node-folding capabilities
pub trait NodeFold {
    /// Parent fold type
    type Parent: Fold;

    /// Add a child branch to the node folder.
    fn add<F: Foldable<Self::Parent>>(&mut self, child: &F);

    /// Finalise the node folding and produce the node.
    fn done(self) -> <Self::Parent as Fold>::Folded;
}

/// Implementing types define a state structure which can be unfolded.
///
/// Unfolding is the inverse operation to folding and can be thought of as a kind of
/// deserialisation process. The source type `U` may be a merkle tree or some kind of storage
/// backend: it doesn't have a type that tells you anything about the structure stored. For this
/// reason, unfolding is always fallible, because `Self` expects a certain shape of tree and
/// each leaf needs to contain a valid serialisation of the expected type.
pub trait Unfoldable: Sized {
    /// Unfold the data into a state type.
    fn unfold<U: Unfold>(source: U) -> Result<Self, U::Error>;
}

/// Error trait to allow generic unfold implementations to use their own error types.
pub trait UnfoldError {
    fn custom<E: error::Error>(error: E) -> Self;
}

/// Implementing types describe 'source' data structures than can be deserialised or extracted in a
/// tree-like manner.
pub trait Unfold {
    /// Corresponding node-unfolding type. When we are unfolding a subtree we expect to be able to
    /// extract child nodes one after another, the `NodeUnfold` trait gives methods to do this.
    type NodeUnfold: NodeUnfold<Parent = Self>;

    /// Unfolding is always fallible and the type of errors may depend on the source.
    type Error: UnfoldError;

    /// We expect the source to contain a node.
    fn into_node(self) -> Result<Self::NodeUnfold, Self::Error>;

    /// We expect the source to contain a leaf, which we attempt to deserialise into type `T`.
    fn into_leaf<T: Decode<()>>(self) -> Result<T, Self::Error>;

    /// We expect the source to contain a leaf, which we extract as raw bytes.
    fn into_leaf_raw<const LEN: usize>(self) -> Result<Box<[u8; LEN]>, Self::Error>;
}

/// When we are unfolding a node, we need a separate type with methods to extract a series of child
/// nodes or leaves.
pub trait NodeUnfold {
    /// The corresponding parent `Unfold` type
    type Parent: Unfold;

    /// Handle the next child node using a specified `unfolder` function.
    fn next_branch_with<T>(
        &mut self,
        unfolder: impl FnOnce(Self::Parent) -> Result<T, <Self::Parent as Unfold>::Error>,
    ) -> Result<T, <Self::Parent as Unfold>::Error>;

    /// Default handler for next child which uses the `Unfoldable` implementation.
    fn next_branch<T: Unfoldable>(&mut self) -> Result<T, <Self::Parent as Unfold>::Error> {
        self.next_branch_with(T::unfold)
    }

    /// Complete the deserialisation of this node.
    fn done<T>(self, value: T) -> Result<T, <Self::Parent as Unfold>::Error>;
}

pub(crate) mod tests;
