// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

/// Generic tree structure used to model the [`crate::merkle_proof::proof_tree::MerkleProof`],
/// as well as the full & partial shapes of a [`crate::merkle_tree::MerkleTree`].
#[derive(Clone, Debug, PartialEq)]
pub enum Tree<A> {
    Node(Vec<Self>),
    Leaf(A),
}
