// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Hashing

use std::borrow::Borrow;
use std::num::NonZeroUsize;

use bincode::Decode;
use bincode::Encode;
use bincode::error::EncodeError;
use thiserror::Error;

use crate::serialisation as binary;

#[derive(Error, Debug)]
pub enum HashError {
    #[error("Encoding error: {0}")]
    Encode(#[from] EncodeError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("The input buffer was expected to be non-empty")]
    NonEmptyBufferExpected,
}

/// Size of digest produced by the underlying hash function
pub const DIGEST_SIZE: usize = 32;

/// A value of type [struct@Hash] indicates that the enclosed array is a digest
/// produced by a preset hash function, currently BLAKE2b. It can be obtained
/// by either hashing data directly or after hashing by converting from
/// a suitably sized byte slice or vector.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Encode,
    Decode,
    Hash,
    PartialOrd,
    Ord,
    derive_more::From,
    derive_more::Debug,
)]
#[debug("{}", self)]
pub struct Hash {
    digest: [u8; DIGEST_SIZE],
}

impl Hash {
    /// Hash a slice of bytes
    pub fn blake3_hash_bytes(bytes: &[u8]) -> Self {
        let digest = blake3::hash(bytes).into();
        Hash { digest }
    }

    /// Get the hash of a value that can be serialised by hashing its serialisation
    pub fn blake3_hash<T: Encode>(data: T) -> Result<Self, EncodeError> {
        let mut hasher = blake3::Hasher::new();
        binary::serialise_into(&data, &mut hasher)?;

        let digest = hasher.finalize().into();
        Ok(Hash { digest })
    }

    /// Combine multiple [`struct@Hash`] values into a single one.
    ///
    /// The hashes are combined by concatenating them, then hashing the result.
    /// Pre-image resistance is not compromised because the concatenation is not
    /// ambiguous, with hashes having a fixed size ([`DIGEST_SIZE`]).
    pub fn combine<H: Borrow<Hash>, HS: IntoIterator<Item = H>>(hashes: HS) -> Hash {
        let mut hasher = blake3::Hasher::new();

        for hash in hashes {
            let hash: &Hash = hash.borrow();
            hasher.update(hash.as_ref());
        }

        let digest = hasher.finalize().into();
        Hash { digest }
    }

    /// Like [`Self::combine`], but the iterator can yield errors.
    pub fn try_combine<H: Borrow<Hash>, E, HS: IntoIterator<Item = Result<H, E>>>(
        hashes: HS,
    ) -> Result<Hash, E> {
        let mut hasher = blake3::Hasher::new();

        for hash in hashes {
            let hash = hash?;
            hasher.update(hash.borrow().as_ref());
        }

        let digest = hasher.finalize().into();
        Ok(Hash { digest })
    }
}

impl std::fmt::Display for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(self.digest).fmt(f)
    }
}

impl From<Hash> for [u8; DIGEST_SIZE] {
    fn from(value: Hash) -> Self {
        value.digest
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.digest
    }
}

/// Writer which hashes fixed-sized chunks of data and produces the digests.
pub struct HashWriter {
    size: usize,
    buffer: Vec<u8>,
    hashes: Vec<Hash>,
}

impl HashWriter {
    /// Initialise a new writer with the given `size`.
    pub fn new(size: NonZeroUsize) -> Self {
        let size = size.get();
        Self {
            size,
            hashes: Vec::new(),
            buffer: Vec::with_capacity(size),
        }
    }

    /// Finalise the writer by hashing any remaining data and returning the vector
    /// of hashes.
    pub fn finalise(mut self) -> Vec<Hash> {
        if !self.buffer.is_empty() {
            self.flush_buffer();
        }

        self.hashes
    }

    /// Hash the contents of the buffer.
    fn flush_buffer(&mut self) {
        let hash = Hash::blake3_hash_bytes(&self.buffer);
        self.hashes.push(hash);
        self.buffer.clear();
    }
}

impl std::io::Write for HashWriter {
    fn write(&mut self, mut buf: &[u8]) -> std::io::Result<usize> {
        let consumed = buf.len();

        while !buf.is_empty() {
            let rem_buffer_len = self.size - self.buffer.len();
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

/// Compute the Merkle hash of a vector of leaf hashes by building a Merkle tree
/// with the given `arity`. The last node in every level might have
/// a smaller arity.
///
/// # Panics
/// Panics if `arity < 2`.
pub fn build_custom_merkle_hash(arity: usize, mut nodes: Vec<Hash>) -> Result<Hash, HashError> {
    assert!(arity >= 2, "Arity must be at least 2");

    if nodes.is_empty() {
        return Err(HashError::NonEmptyBufferExpected);
    }

    let mut next_level = Vec::with_capacity(nodes.len().div_ceil(arity));

    while nodes.len() > 1 {
        // Group the nodes into chunks of size `arity` and hash each chunk.
        for chunk in nodes.chunks(arity) {
            next_level.push(Hash::combine(chunk))
        }

        std::mem::swap(&mut nodes, &mut next_level);
        next_level.truncate(0);
    }

    Ok(nodes[0])
}
