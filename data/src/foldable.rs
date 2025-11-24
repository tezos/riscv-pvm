// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Foldable data structures

/// Implementing types define a structure that can be folded
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
pub trait Foldable<F> {
    /// Fold the data structure.
    fn fold(&self) -> F;
}

impl<T: Foldable<F>, F> Foldable<F> for &T {
    fn fold(&self) -> F {
        T::fold(self)
    }
}

impl<T: Foldable<F>, const LEN: usize, F: NodeFold> Foldable<F> for [T; LEN] {
    fn fold(&self) -> F {
        F::fold_children(self.iter().map(T::fold))
    }
}

impl<A: Foldable<F>, B: Foldable<F>, F: NodeFold> Foldable<F> for (A, B) {
    fn fold(&self) -> F {
        F::fold_children([self.0.fold(), self.1.fold()])
    }
}

impl<A: Foldable<F>, B: Foldable<F>, C: Foldable<F>, F: NodeFold> Foldable<F> for (A, B, C) {
    fn fold(&self) -> F {
        F::fold_children([self.0.fold(), self.1.fold(), self.2.fold()])
    }
}

/// Type with node-folding capabilities
pub trait NodeFold {
    /// Create a node from its children.
    fn fold_children(children: impl IntoIterator<Item = Self>) -> Self;
}
