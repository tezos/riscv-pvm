// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Merkle trees used for proof generation by the PVM

use octez_riscv_data::compressed_merkle_tree::MERKLE_LEAF_SIZE;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashError;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
use octez_riscv_data::merkle_tree::MerkleTree;
use octez_riscv_data::tree::Tree;

use super::DynAccess;

// TODO RV-322: Choose optimal Merkleisation parameters for main memory.
/// Arity of the Merkle tree used for Merkleising [`DynArrays`].
///
/// [`DynArrays`]: [`crate::state_backend::layout::DynArray`]
pub const MERKLE_ARITY: usize = 4;

pub trait AccessInfoAggregatable {
    /// Aggregate the access information of the Merkle tree described by
    /// the layout of the given data, without constructing the tree.
    ///
    /// Used in implementations of `to_merkle_tree` in which certain leaves can
    /// combine data corresponding to multiple layout elements.
    fn aggregate_access_info(&self) -> bool;
}

impl AccessInfoAggregatable for () {
    fn aggregate_access_info(&self) -> bool {
        false
    }
}

/// Helper function which allows iterating over chunks of a dynamic array
/// and writing them to a writer. The last chunk may be smaller than the
/// Merkle leaf size. The implementations of [`Foldable`] and
/// [`ProofLayout`] both use it, ensuring consistency between the two.
///
/// [`Foldable`]: octez_riscv_data::foldable::Foldable
/// [`ProofLayout`]: crate::state_backend::proof_layout::ProofLayout
pub(crate) fn chunks_to_writer<T: std::io::Write, F: Fn(usize) -> [u8; MERKLE_LEAF_SIZE.get()]>(
    writer: &mut T,
    len: usize,
    read: F,
) -> Result<(), std::io::Error> {
    let merkle_leaf_size = MERKLE_LEAF_SIZE.get();
    assert!(len >= merkle_leaf_size);

    let mut address = 0;

    while address + merkle_leaf_size <= len {
        writer.write_all(read(address).as_slice())?;
        address += merkle_leaf_size;
    }

    // When the last chunk is smaller than `MERKLE_LEAF_SIZE`,
    // read the last `MERKLE_LEAF_SIZE` bytes and pass a subslice containing
    // only the bytes not previously read to the writer.
    if address != len {
        address += merkle_leaf_size;
        let buffer = read(len.saturating_sub(merkle_leaf_size));
        writer.write_all(&buffer[address.saturating_sub(len)..])?;
    };

    Ok(())
}

/// Writer which splits data in fixed-sized chunks and produces a [`MerkleTree`]
/// with a given arity in which each leaf represents a chunk.
pub struct MerkleWriter {
    leaf_size: usize,
    arity: usize,
    read_log: DynAccess,
    write_log: DynAccess,
    buffer: Vec<u8>,
    leaves: Vec<MerkleTree>,
}

impl MerkleWriter {
    /// Initialise a new writer with a leaf size and arity for the Merkle tree,
    /// the access logs of the underlying data, and the expected number of leaves.
    ///
    /// # Panics
    /// Panics if `arity < 2`.
    pub fn new(
        leaf_size: std::num::NonZeroUsize,
        arity: usize,
        read_log: DynAccess,
        write_log: DynAccess,
        expected_leaves: usize,
    ) -> Self {
        assert!(arity >= 2, "Arity must be at least 2");

        let leaf_size = leaf_size.get();
        Self {
            leaf_size,
            arity,
            read_log,
            write_log,
            buffer: Vec::with_capacity(leaf_size),
            leaves: Vec::with_capacity(expected_leaves),
        }
    }

    /// Commit the leaf corresponding to the contents of the buffer before
    /// clearing it.
    fn flush_buffer(&mut self) {
        let pos = self.leaves.len() * self.leaf_size;
        let range = pos..pos + self.leaf_size;

        // Determine whether addresses in the range of the current buffer
        // have been accessed.
        let read = self.read_log.includes_range(range.clone());
        let write = self.write_log.includes_range(range);
        let access_info = read || write;

        self.leaves.push(MerkleTree::make_merkle_leaf(
            self.buffer.clone(),
            access_info,
        ));
        self.buffer.clear();
    }

    /// Finalise the writer by generating the Merkle tree with the configured
    /// arity from the stored leaves. The last node in every level might have
    /// a smaller arity.
    pub fn finalise(mut self) -> Result<MerkleTree, HashError> {
        if !self.buffer.is_empty() {
            self.flush_buffer();
        }

        build_custom_merkle_tree(self.arity, self.leaves)
    }
}

impl std::io::Write for MerkleWriter {
    fn write(&mut self, mut buf: &[u8]) -> std::io::Result<usize> {
        let consumed = buf.len();

        while !buf.is_empty() {
            let rem_buffer_len = self.leaf_size - self.buffer.len();
            let new_buf_len = std::cmp::min(rem_buffer_len, buf.len());

            let new_buf = &buf[..new_buf_len];
            buf = &buf[new_buf_len..];
            self.buffer.extend_from_slice(new_buf);

            // If the buffer has been completely filled, flush it.
            if rem_buffer_len == new_buf_len {
                self.flush_buffer();
            }
        }
        Ok(consumed)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Build a Merkle tree whose leaves are the elements of `nodes` and in which
/// each node has the given `arity`.
pub(crate) fn build_custom_merkle_tree(
    arity: usize,
    mut nodes: Vec<MerkleTree>,
) -> Result<MerkleTree, HashError> {
    if nodes.is_empty() {
        return Err(HashError::NonEmptyBufferExpected);
    }

    let mut next_level = Vec::with_capacity(nodes.len().div_ceil(arity));

    while nodes.len() > 1 {
        for chunk in nodes.chunks(arity) {
            next_level.push(MerkleTree::make_merkle_node(chunk.to_vec()))
        }

        std::mem::swap(&mut nodes, &mut next_level);
        next_level.truncate(0);
    }

    Ok(nodes.pop().unwrap_or_else(|| {
        unreachable!(
            "After the loop, `nodes` could only have 0 or 1 elements. It had \
             more than 1 element at the beginning of the last iteration of the \
             loop and exactly one element was pushed to it because `nodes.chunks` \
             could not have resulted in 0 chunks for a non-empty vector."
        )
    }))
}
