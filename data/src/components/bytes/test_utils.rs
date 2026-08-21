// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Utilities for testing `Bytes` component, which need to be accessible from the benchmarking code
//! as well as within test modules.

#![cfg(any(test, feature = "unstable-test-utils"))]

use proptest::collection::vec;
use proptest::prelude::Just;
use proptest::prelude::Strategy;
use proptest::prelude::any;
use proptest::prop_oneof;

use crate::components::bytes::Bytes;
use crate::components::bytes::BytesMode;
use crate::components::bytes::NODE_ARITY;
use crate::components::bytes::PAGE_SIZE;
use crate::hash::Hash;

/// Operations to be issued against an immutable Bytes state component
#[derive(Debug, Clone)]
pub enum BytesOp {
    Read { offset: usize, size: usize },
    Len,
}

impl BytesOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub fn any(length: usize) -> impl Strategy<Value = Self> + Clone {
        prop_oneof![
            (0..length, 0usize..50).prop_map(|(offset, size)| Self::Read { offset, size }),
            Just(Self::Len),
        ]
    }

    /// Run an operation against an immutable Bytes state component.
    pub fn run<M: BytesMode>(&self, bytes: &Bytes<M>) -> BytesOpResult {
        match self {
            Self::Read { offset, size } => {
                let mut data = vec![0u8; *size];
                let read = bytes.read(*offset, &mut data);
                BytesOpResult::Read { read, data }
            }

            Self::Len => BytesOpResult::Len { len: bytes.len() },
        }
    }
}

/// Operations to be issued against a mutable Bytes state component
#[derive(Debug, Clone)]
pub enum BytesMutOp {
    Write { offset: usize, data: Vec<u8> },
    Set { data: Vec<u8> },
    Resize { new_size: usize },
    Immutable { op: BytesOp },
}

impl BytesMutOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub fn any(length: usize) -> impl Strategy<Value = Self> + Clone {
        // Sets of both a handful of bytes and of a size straddling a page boundary, so that the
        // pages the set covers are sometimes partial and sometimes whole.
        let set_data = prop_oneof![
            vec(any::<u8>(), 0..50),
            vec(any::<u8>(), PAGE_SIZE - 2..2 * PAGE_SIZE + 2),
        ];

        prop_oneof![
            (0..length, vec(any::<u8>(), 0..50))
                .prop_map(|(offset, data)| Self::Write { offset, data }),
            set_data.prop_map(|data| Self::Set { data }),
            (0..length).prop_map(|new_size| Self::Resize { new_size }),
            BytesOp::any(length).prop_map(|op| Self::Immutable { op }),
        ]
    }

    /// Run the operation against the Bytes state component.
    pub fn run<M: BytesMode>(&self, bytes: &mut Bytes<M>) -> BytesOpResult {
        match self {
            Self::Write { offset, data } => {
                let wrote = bytes.write(*offset, data);
                BytesOpResult::Wrote { wrote }
            }

            Self::Set { data } => {
                bytes.set(data);
                BytesOpResult::Void
            }

            Self::Resize { new_size } => {
                bytes.resize(*new_size);
                BytesOpResult::Void
            }

            Self::Immutable { op } => op.run(bytes),
        }
    }
}

/// Results of operations issued against the Bytes state component
#[derive(Debug, PartialEq, Eq)]
pub enum BytesOpResult {
    Read { read: usize, data: Vec<u8> },
    Wrote { wrote: usize },
    Len { len: usize },
    Void,
}

/// 64 MiB: the maximum size of a `Bytes` component in the durable storage
pub const NDS_BYTES_LENGTH: usize = 1024 * 1024 * 64;

/// The depth of the (data part of the) Merkle tree for a maximal durable storage leaf.
pub const NDS_BYTES_TREE_DEPTH: usize = (NDS_BYTES_LENGTH / PAGE_SIZE).ilog(NODE_ARITY) as usize;

/// A layer in the proof tree with two non-blinded nodes in two separate branches. Each branch
/// contributes one less than `NODE_ARITY` hashes and `NODE_ARITY` worth of tags.
pub const BIFURCATED_LAYER: usize = ((NODE_ARITY - 1) * Hash::DIGEST_SIZE + NODE_ARITY) * 2;

/// A layer in the proof tree with three non-blinded nodes -
/// two in the same branch, one in the other.
///
/// - both branches contribute `NODE_ARITY` worth of tags
/// - one branch contributes two less than `NODE_ARITY` hashes
/// - the other branch contributes one less than `NODE_ARITY` hashes
pub const BIFURCATED_ASYMMETRIC_LAYER: usize =
    NODE_ARITY * 2 + (NODE_ARITY - 2) * Hash::DIGEST_SIZE + (NODE_ARITY - 1) * Hash::DIGEST_SIZE;

/// A layer in the proof tree with two non-blinded nodes both in the same branch. This has two less
/// than `NODE_ARITY` hashes and `NODE_ARITY` worth of tags.
pub const MONOFURCATED_LAYER: usize = (NODE_ARITY - 2) * Hash::DIGEST_SIZE + NODE_ARITY;

/// The calculated theoretical maximum length of a proof for a single operation on a leaf in the
/// durable storage. It is caused by a read that crosses the worst possible boundaries of three
/// pages. A write across the same boundaries is shorter - see [`MAX_WRITE_PROOF_LENGTH`].
///
/// Page size is 1024 bytes, plus 8 for a `u64` representing the length of the page; there are at
/// most three such pages in any proof, because the maximum read/write is twice the size of a page.
///
/// There is a single asymmetric bifurcated layer of the tree: it has one branch with 0 blinded nodes,
/// and one branch with 1. e.g. on the left the tags would be (node, node) and on the right
/// (node, blind).
///
/// There are then 14 identical bifurcated layers of the tree, each with 2 blinded nodes:
/// (blind, node) on one side, (node, blind) on the other.
///
/// There is then one more monofurcated layer with only zero blinded nodes and two tags
/// (node, node).
///
/// Finally there are 8 bytes for the overall length of the `Bytes` component (serialised as a
/// `u64`) and an extra two tags for the layer containing 'length' and 'data'. There is then one
/// extra tag for the root node of the entire tree.
pub const MAX_PROOF_LENGTH: usize = (PAGE_SIZE + 8) * 3
    + BIFURCATED_ASYMMETRIC_LAYER
    + BIFURCATED_LAYER * (NDS_BYTES_TREE_DEPTH - 2)
    + MONOFURCATED_LAYER
    + 8
    + 2
    + 1;

/// The maximum length of a proof for a single write, which is shorter than [`MAX_PROOF_LENGTH`].
///
/// A write of twice the page size spanning three pages covers the middle one in full, and a page the
/// write covers in full is blinded instead of carried: the verifier recomputes its hash from the
/// written data. So that page contributes a hash rather than its content plus the `u64` page length.
/// The tree layers are unchanged - a blinded leaf sits in the same position as the leaf it replaces.
pub const MAX_WRITE_PROOF_LENGTH: usize = MAX_PROOF_LENGTH - (PAGE_SIZE + 8) + Hash::DIGEST_SIZE;

/// There is one 'area' at which the worst possible boundary between two pages occurs---at the
/// halway point in the `Bytes` component.
///
/// Essentially in the worst case, two pages will be present on one path down from the top of the tree,
/// and one more page on aother path from the top of the tree. The pages are contiguous by 'idx'.
///
/// ```custom,{class=language-markdown}
///                   [root_hash]
///                      __|__
///                     /     \
///              ______/       \______
///     ________/  /               \  \________
///    /          /                 \          \
/// [blind]      /                   \      [blind]
///             /\                   /\
///         ___/  \                 /  \___
///        /       \               /       \
/// [blind/data] [data]         [data] [data/blind]
/// ```
pub const MAX_PROOF_OFFSETS: [usize; 2] = [
    (NDS_BYTES_LENGTH / 4) * 2 - 1,
    (NDS_BYTES_LENGTH / 4) * 2 - PAGE_SIZE - 1,
];

#[cfg(test)]
mod tests {
    use std::io::Write;

    use goldenfile::Mint;

    use super::*;

    /// Sanity check that the constant values calculated above have not changed.
    #[test]
    fn const_values() {
        let mut mint = Mint::new("tests/goldenfiles");
        let mut file = mint.new_goldenfile("bytes_test_utils_constants").unwrap();

        writeln!(file, "NDS_BYTES_LENGTH = {NDS_BYTES_LENGTH}").unwrap();
        writeln!(file, "NDS_BYTES_TREE_DEPTH = {NDS_BYTES_TREE_DEPTH}").unwrap();
        writeln!(file, "MONOFURCATED_LAYER = {MONOFURCATED_LAYER}").unwrap();
        writeln!(file, "BIFURCATED_LAYER = {BIFURCATED_LAYER}").unwrap();
        writeln!(
            file,
            "BIFURCATED_ASYMMETRIC_LAYER = {BIFURCATED_ASYMMETRIC_LAYER}"
        )
        .unwrap();
        writeln!(file, "MAX_PROOF_LENGTH = {MAX_PROOF_LENGTH}").unwrap();
        writeln!(file, "MAX_WRITE_PROOF_LENGTH = {MAX_WRITE_PROOF_LENGTH}").unwrap();
    }
}
