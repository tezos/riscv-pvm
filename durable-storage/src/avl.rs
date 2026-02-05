// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of a Merkleisable AVL tree.

pub(crate) mod node;

pub(crate) mod resolver;
pub(crate) mod tree;

// Re-exports for tests and benchmarks
#[cfg(test)]
pub(crate) use node::Node;
#[cfg(test)]
pub(crate) use node::hash;
#[cfg(test)]
pub use tree::Tree;
