// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

/// Generic tree structure used to model the [`crate::merkle_proof::proof_tree::MerkleProof`],
/// as well as the full & partial shapes of a [`crate::merkle_tree::MerkleTree`].
#[derive(Clone, Debug, PartialEq)]
pub enum Tree<LeafData, NodeData> {
    Node { data: NodeData, children: Vec<Self> },
    Leaf { data: LeafData },
}

/// Used in [`crate::merkle_proof::transform::impl_modify_map_collect`]
impl<D, N> From<D> for Tree<D, N> {
    fn from(value: D) -> Self {
        Tree::Leaf { data: value }
    }
}
