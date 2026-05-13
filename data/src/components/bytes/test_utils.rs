// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
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
    Resize { new_size: usize },
    Immutable { op: BytesOp },
}

impl BytesMutOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub fn any(length: usize) -> impl Strategy<Value = Self> + Clone {
        prop_oneof![
            (0..length, vec(any::<u8>(), 0..50))
                .prop_map(|(offset, data)| Self::Write { offset, data }),
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

/// The depth of the (data part of the) Merkle tree for a maximal durable storage leaf. This should
/// be 7.
pub const NDS_BYTES_TREE_DEPTH: usize = (NDS_BYTES_LENGTH / PAGE_SIZE).ilog(NODE_ARITY) as usize;

/// A layer in the proof tree with two non-blinded nodes in two separate branches. Each branch
/// contributes one less than `NODE_ARITY` hashes and `NODE_ARITY` worth of tags.
pub const BIFURCATED_LAYER: usize = ((NODE_ARITY - 1) * Hash::DIGEST_SIZE + NODE_ARITY) * 2;

/// A layer in the proof tree with two non-blinded nodes both in the same branch. This has two less
/// than `NODE_ARITY` hashes and `NODE_ARITY` worth of tags.
pub const MONOFURCATED_LAYER: usize = (NODE_ARITY - 2) * Hash::DIGEST_SIZE + NODE_ARITY;

/// The calculated theoretical maximum length of a proof for a single operation on a leaf in the
/// durable storage. It is caused by a read or write that crosses the worst possible boundary of
/// two pages.
///
/// Page size is 4096 bytes, plus 8 for a `u64` representing the length of the page; there are at
/// most two such pages in any proof, because the maximum read/write is smaller than one page.
///
/// There are then 6 identical bifurcated layers of the tree, each with 6 blinded nodes (6 * 32 =
/// 192) and 8 bytes of tags (blind, blind, blind, node) on one side, (node, blind, blind, blind)
/// on the other.
///
/// There is then one more monofurcated layer with only two blinded nodes (64 bytes) and four tags
/// (node, node, blind, blind).
///
/// Finally there are 8 bytes for the overall length of the `Bytes` component (serialised as a
/// `u64`) and an extra two tags for the layer containing 'length' and 'data'. There is then one
/// extra tag for the root node of the entire tree.
///
/// This all adds up to give 9487 bytes.
pub const MAX_PROOF_LENGTH: usize = (PAGE_SIZE + 8) * 2
    + BIFURCATED_LAYER * (NDS_BYTES_TREE_DEPTH - 1)
    + MONOFURCATED_LAYER
    + 8
    + 2
    + 1;

/// There are three offsets at which the worst possible boundary between two pages occurs---at the
/// 1st, 2nd and 3rd quartile in the `Bytes` component.
pub const MAX_PROOF_OFFSETS: [usize; 3] = [
    NDS_BYTES_LENGTH / 4 - 1,
    (NDS_BYTES_LENGTH / 4) * 2 - 1,
    (NDS_BYTES_LENGTH / 4) * 3 - 1,
];

/// At each of the `MAX_PROOF_OFFSETS` a read or write of only two bytes is all that is needed to
/// get a maximally long proof. This returns a vector of all six such operations.
pub fn max_proof_ops() -> Vec<BytesMutOp> {
    let mut v = vec![];

    for offset in MAX_PROOF_OFFSETS {
        v.push(BytesMutOp::Immutable {
            op: BytesOp::Read { offset, size: 2 },
        });
        v.push(BytesMutOp::Write {
            offset,
            data: vec![0, 0],
        });
    }

    v
}

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
        writeln!(file, "MAX_PROOF_LENGTH = {MAX_PROOF_LENGTH}").unwrap();
    }
}
