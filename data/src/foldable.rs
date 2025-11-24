// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Foldable data structures

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

impl<T: Foldable<F>, const LEN: usize, F: Fold> Foldable<F> for [T; LEN] {
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();

        for item in self.iter() {
            builder.add(item);
        }

        builder.done()
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
