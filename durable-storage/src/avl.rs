// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of a Merkleisable AVL tree.

pub(crate) mod node;
pub(crate) mod resolver;

cfg_if::cfg_if! {
    if #[cfg(feature = "bench")] {
        pub mod tree;
    } else {
        pub(crate) mod tree;
    }
}

// Re-exports for tests and benchmarks
#[cfg(test)]
pub(crate) use node::Node;
#[cfg(test)]
pub(crate) use node::hash;
#[cfg(any(test, feature = "bench"))]
pub use tree::Tree;
