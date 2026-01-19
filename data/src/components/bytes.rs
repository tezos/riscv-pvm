// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a byte array
//!
//! See [`Bytes`] for more details.

use std::borrow::Borrow;
use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Range;

use bincode::Encode;
use perfect_derive::perfect_derive;
use range_collections::RangeSet2;

use crate::clone::CloneState;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::hash::Hasher;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::Partial;
use crate::merkle_tree::MerkleTree;
use crate::merkle_tree::MerkleTreeFold;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::Source;
use crate::mode::utils::not_found;
use crate::partial_vec::PartialVec;
use crate::serialisation::serialise;

/// Byte array state component
#[perfect_derive(Debug)]
pub struct Bytes<M: Mode> {
    bytes: M::Select<BytesTemplate>,
}

impl<M: BytesMode> Bytes<M> {
    /// Create an empty byte array.
    pub fn new() -> Self {
        M::new()
    }

    /// Read from the byte array.
    ///
    /// `start` is the starting index to read from. All data will be read into `buffer`. The
    /// maximum number of bytes read is the length of `buffer`.
    ///
    /// Returns the number of bytes actually read. This could be less than the length of `buffer`
    /// if the read goes or starts out of bounds.
    pub fn read(&self, start: usize, buffer: &mut [u8]) -> usize {
        M::read(self, start, buffer)
    }

    /// Write to the byte array.
    ///
    /// `start` is the starting index to write to. Data will be written from `buffer`. The maximum
    /// number of bytes written is the length of `buffer`.
    ///
    /// Returns the number of bytes actually written. This could be less than the length of `buffer`
    /// if the write goes or starts out of bounds.
    pub fn write(&mut self, start: usize, buffer: &[u8]) -> usize {
        M::write(self, start, buffer)
    }

    /// Append data to the end of the byte array.
    ///
    /// Returns the number of bytes actually appended. This could be less than the length of
    /// `buffer` if the byte array length would overflow [`usize`].
    pub fn append(&mut self, buffer: &[u8]) -> usize {
        let current_len = self.len();
        let new_len = buffer.len().saturating_add(current_len);
        self.resize(new_len);

        let bytes_to_write = buffer.len().min(new_len - current_len);
        self.write(current_len, &buffer[..bytes_to_write])
    }

    /// Overwrite the entire contents of the byte array with the given data.
    pub fn set(&mut self, buffer: &[u8]) {
        self.resize(buffer.len());
        self.write(0, buffer);
    }

    /// Is the length of the byte array zero?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of bytes the byte array holds.
    pub fn len(&self) -> usize {
        M::len(self)
    }

    /// Change the length of the byte array.
    ///
    /// If the new length is greater than the current length, the new bytes will be zeroed. If the
    /// new length is less than the current length, the byte array will be truncated.
    pub fn resize(&mut self, new_len: usize) {
        M::resize(self, new_len);
    }

    /// Fold [`Self`] generically.
    fn fold_generic<Item: Foldable<Build>, Build: Fold>(
        &self,
        builder: Build,
        length: usize,
        length_node: impl Foldable<Build>,
        get_item: impl Fn(Range<usize>) -> Item,
    ) -> Build::Folded {
        let generator = |idx: usize| {
            let address = PAGE_SIZE
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of PAGE_SIZE bytes before");
            let address_end = address
                .checked_add(PAGE_SIZE)
                .expect("Address overflow")
                .min(length);
            let address_range = address..address_end;

            get_item(address_range)
        };

        let pages = length.div_ceil(PAGE_SIZE);

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(pages, NODE_ARITY, &generator));
        builder.done()
    }

    /// Folding hasher function that works for both [`Normal`] and [`Prove`] modes
    fn fold_hash<D: Borrow<[u8]>, F: Fn(Range<usize>) -> D>(
        &self,
        builder: HashFold,
        length: usize,
        get_data: F,
    ) -> Hash {
        let length_node =
            Hash::hash_encodable(length as u64).expect("Hashing length should not fail");

        let get_item = |range| {
            let data = get_data(range);
            Hash::hash_bytes(data.borrow())
        };

        self.fold_generic(builder, length, length_node, get_item)
    }
}

impl<M: BytesMode> Default for Bytes<M> {
    fn default() -> Self {
        M::new()
    }
}

impl<M: CloneBytesMode> CloneState for Bytes<M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl Foldable<HashFold> for Bytes<Normal> {
    fn fold(&self, builder: HashFold) -> Hash {
        self.fold_hash(builder, self.bytes.len(), |addr_range| {
            &self.bytes[addr_range]
        })
    }
}

impl Foldable<HashFold> for Bytes<Prove<'_>> {
    fn fold(&self, builder: HashFold) -> Hash {
        self.fold_hash(builder, self.bytes.unrecorded_len(), |addr_range| {
            let previous_len = self.bytes.previous.len();
            let previous_range =
                addr_range.start.min(previous_len)..addr_range.end.min(previous_len);

            let mut data = self.bytes.previous.bytes[previous_range].to_vec();
            data.resize(addr_range.len(), 0);

            for (index, chunk) in self.bytes.writes.defined_range(addr_range) {
                data[index..][..chunk.len()].copy_from_slice(chunk);
            }

            data
        })
    }
}

impl Foldable<MerkleTreeFold> for Bytes<Prove<'_>> {
    fn fold(&self, builder: MerkleTreeFold) -> MerkleTree {
        // Reminder: Merkle trees generated in Prove mode capture the state at beginning of proof
        // generation. This means we need to use `previous` state for the length and data.

        let length = self.bytes.previous.len();
        let length_data = serialise(length as u64).expect("Serialising length should not fail");
        let is_length_needed = self.bytes.need_length_in_proof();
        let length_node = MerkleTree::make_merkle_leaf(length_data, is_length_needed);

        let get_item = |range: Range<usize>| {
            let accessed = self.bytes.was_accessed(range.clone());
            let data = self.bytes.previous.bytes[range].to_vec();
            MerkleTree::make_merkle_leaf(data, accessed)
        };

        self.fold_generic(builder, length, length_node, get_item)
    }
}

impl Foldable<PartialHashFold<'_>> for Bytes<Verify> {
    fn fold(&self, builder: PartialHashFold<'_>) -> PartialHash {
        if self.bytes.is_completely_absent() {
            return builder.previous();
        }

        // The length must be present if the byte array is not completely absent. Otherwise we can't
        // properly construct the partial Merkle tree and therefore obtain the final hash.
        let Some(length) = self.bytes.length.clone().to_present() else {
            return PartialHash::InvalidProof;
        };
        let length_hash =
            Hash::hash_encodable(length as u64).expect("Hashing length should not fail");
        let length_node = PartialHash::Present(length_hash);

        let get_item = |range: Range<usize>| {
            match self.bytes.data.continuous_defined_range(range.clone()) {
                None => {
                    if self.bytes.data.is_any_defined(range) {
                        // This means there are undefined and defined ranges in this page.
                        // That's not allowed as pages must be either fully present or fully
                        // absent.
                        return PartialHash::InvalidProof;
                    }

                    PartialHash::Previous
                }

                Some(chunks) => {
                    let mut hasher = Hasher::default();

                    for chunk in chunks {
                        hasher.update(chunk);
                    }

                    PartialHash::Present(hasher.to_hash())
                }
            }
        };

        self.fold_generic(builder, length, length_node, get_item)
    }
}

impl<M: CloneBytesMode> Clone for Bytes<M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

impl<M: EncodeBytesMode> Encode for Bytes<M> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        M::encode(self, encoder)
    }
}

impl<T: AsRef<[u8]>, M: BytesMode> PartialEq<T> for Bytes<M> {
    fn eq(&self, other: &T) -> bool {
        let other = other.as_ref();
        let len = self.len();

        if len != other.len() {
            return false;
        }

        let mut chunk_lhs = vec![0u8; 4096];
        for start in (0..len).step_by(chunk_lhs.len()) {
            let read = self.read(start, &mut chunk_lhs);

            if chunk_lhs[..read] != other[start..][..read] {
                return false;
            }
        }

        true
    }
}

impl<M: BytesMode, N: BytesMode> PartialEq<Bytes<N>> for Bytes<M> {
    fn eq(&self, other: &Bytes<N>) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        let mut chunk_lhs = vec![0u8; 4096];
        let mut chunk_rhs = chunk_lhs.clone();

        for offset in (0..len).step_by(chunk_lhs.len()) {
            let read = self.read(offset, &mut chunk_lhs);

            if read != other.read(offset, &mut chunk_rhs) {
                return false;
            }

            if chunk_lhs[..read] != chunk_rhs[..read] {
                return false;
            }
        }

        true
    }
}

impl<M: BytesMode> Eq for Bytes<M> {}

impl From<bytes::Bytes> for Bytes<Normal> {
    fn from(bytes: bytes::Bytes) -> Self {
        let bytes = bytes::BytesMut::from(bytes);
        Bytes { bytes }
    }
}

impl From<&[u8]> for Bytes<Normal> {
    fn from(slice: &[u8]) -> Self {
        let bytes = bytes::BytesMut::from(slice);
        Bytes { bytes }
    }
}

/// Modal template for the [`Bytes`] component
///
/// This is used to select the appropriate implementation for the mode.
enum BytesTemplate {}

impl Modal for BytesTemplate {
    type Normal = bytes::BytesMut;

    type Prove<'normal> = ProveImpl<'normal>;

    type Verify = VerifyImpl;
}

/// Mode types that implement this trait support common operations on the [`Bytes`] component.
///
/// See [`Bytes`] for a more convenient interface.
pub trait BytesMode: Mode {
    /// See [`Bytes::new`].
    fn new() -> Bytes<Self>;

    /// See [`Bytes::read`].
    fn read(this: &Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize;

    /// See [`Bytes::write`].
    fn write(this: &mut Bytes<Self>, start: usize, buffer: &[u8]) -> usize;

    /// See [`Bytes::len`].
    fn len(this: &Bytes<Self>) -> usize;

    /// See [`Bytes::resize`].
    fn resize(this: &mut Bytes<Self>, new_len: usize);
}

impl BytesMode for Normal {
    fn new() -> Bytes<Self> {
        Bytes {
            bytes: bytes::BytesMut::new(),
        }
    }

    fn read(this: &Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        // If the read starts out of bounds, there is nothing to read.
        if start >= this.bytes.len() {
            return 0;
        }

        let range = clamp_range(this.bytes.len(), start, buffer.len());
        let len = range.len();

        buffer[..len].copy_from_slice(&this.bytes[range]);

        len
    }

    fn write(this: &mut Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        // We can't write if we start out of bounds.
        if start >= this.bytes.len() {
            return 0;
        }

        let range = clamp_range(this.bytes.len(), start, buffer.len());
        let len = range.len();

        this.bytes[range].copy_from_slice(&buffer[..len]);

        len
    }

    fn len(this: &Bytes<Self>) -> usize {
        this.bytes.len()
    }

    fn resize(this: &mut Bytes<Self>, new_len: usize) {
        this.bytes.resize(new_len, 0x0);
    }
}

/// Mode types that implement this trait support cloning of the [`Bytes`] component
pub trait CloneBytesMode: Mode {
    /// Clones the given [`Bytes`] component.
    ///
    /// This clones the entire component, not just the internal value. Consider this when cloning
    /// components in [`crate::mode::Prove`] mode.
    fn clone(this: &Bytes<Self>) -> Bytes<Self>;
}

impl CloneBytesMode for Normal {
    fn clone(this: &Bytes<Self>) -> Bytes<Self> {
        Bytes {
            bytes: this.bytes.clone(),
        }
    }
}

impl CloneBytesMode for Prove<'_> {
    fn clone(this: &Bytes<Self>) -> Bytes<Self> {
        Bytes {
            bytes: this.bytes.clone(),
        }
    }
}

impl CloneBytesMode for Verify {
    fn clone(this: &Bytes<Self>) -> Bytes<Self> {
        Bytes {
            bytes: this.bytes.clone(),
        }
    }
}

/// Mode types that implement this trait support encoding of the [`Bytes`] component
pub trait EncodeBytesMode: Mode {
    /// Encodes the [`Bytes`] component as a byte vector.
    fn encode<E: bincode::enc::Encoder>(
        bytes: &Bytes<Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError>;
}

impl EncodeBytesMode for Normal {
    fn encode<E: bincode::enc::Encoder>(
        bytes: &Bytes<Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bytes.bytes.encode(encoder)
    }
}

impl BytesMode for Prove<'_> {
    fn new() -> Bytes<Self> {
        Bytes {
            bytes: ProveImpl {
                previous: Source::default(),
                length: 0,
                did_access_length: Cell::new(false),
                reads: RefCell::new(RangeSet2::empty()),
                writes: PartialVec::default(),
            },
        }
    }

    fn read(this: &Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        let len = this.len();

        // If the read starts out of bounds, there is nothing to read.
        if start >= len {
            return 0;
        }

        let range = clamp_range(len, start, buffer.len());
        let len = range.len();

        // The read range needs to be recorded, as it is used in the Merkle tree compression.
        this.bytes
            .reads
            .borrow_mut()
            .union_with(&RangeSet2::from(range.clone()));

        let buffer = &mut buffer[..len];
        let from_previous = this.bytes.previous.read(range.start, buffer);

        // Bytes beyond the previous length are implicitly zero, if they are not present in the
        // `writes` vector.
        buffer[from_previous..].fill(0);

        for (offset, bytes) in this.bytes.writes.defined_range(range.clone()) {
            buffer[offset..][..bytes.len()].copy_from_slice(bytes);
        }

        len
    }

    fn write(this: &mut Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        let len = this.len();

        // We can't write if we start out of bounds.
        if start >= len {
            return 0;
        }

        let range = clamp_range(len, start, buffer.len());
        let len = range.len();

        let data = buffer[..len].to_vec();
        this.bytes.writes.define(range.start, data);

        len
    }

    fn len(this: &Bytes<Self>) -> usize {
        this.bytes.did_access_length.set(true);
        this.bytes.length
    }

    fn resize(this: &mut Bytes<Self>, new_len: usize) {
        if new_len < this.len() {
            // We need to clear out the previously written bytes that are now out of bounds.
            // Otherwise, resizing up again would restore them.
            this.bytes.writes.truncate(new_len);
        }

        this.bytes.length = new_len;

        // NOTE: We do not extend the `writes` vector, as the new bytes are implicitly zero.
    }
}

impl BytesMode for Verify {
    fn new() -> Bytes<Self> {
        Bytes {
            bytes: VerifyImpl {
                length: Partial::Present(0),
                data: PartialVec::empty(),
            },
        }
    }

    fn read(this: &Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        let len = this.len();

        // If the read starts out of bounds, there is nothing to read.
        if start >= len {
            return 0;
        }

        let range = clamp_range(len, start, buffer.len());
        let len = range.len();

        let buffer = &mut buffer[..len];
        let Some(data) = this.bytes.data.continuous_defined_range(range) else {
            // SAFETY: We're in verify mode where `not_found` may be used.
            unsafe { not_found() }
        };

        let mut offset = 0;
        for chunk in data {
            buffer[offset..][..chunk.len()].copy_from_slice(chunk);
            offset += chunk.len();
        }

        len
    }

    fn write(this: &mut Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        let len = this.len();

        // We can't write if we start out of bounds.
        if start >= len {
            return 0;
        }

        let range = clamp_range(len, start, buffer.len());
        let len = range.len();

        this.bytes.data.define(start, buffer[..len].to_vec());

        len
    }

    fn len(this: &Bytes<Self>) -> usize {
        match this.bytes.length {
            Partial::Present(len) => len,
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: We're in verify mode where `not_found` may be used.
                unsafe { not_found() }
            }
        }
    }

    fn resize(this: &mut Bytes<Self>, new_len: usize) {
        let prev_len = this.len();

        if new_len > prev_len {
            // Define the grown part of the byte array as zeros.
            // This is different to Prove mode where we just implicitly treat undefined bytes as
            // zeros. We can't do this in Verify mode because we don't have a previous length to
            // compare against. So we can't differentiate between undefined bytes that should have
            // been part of the proof and undefined bytes as a result of resizing.
            let zero_data = vec![0u8; new_len - prev_len];
            this.bytes.data.define(prev_len, zero_data);
        }

        if new_len < prev_len {
            // If we shrink the byte array, we need to remove any data beyond the new length to
            // ensure it is not accessible later on.
            this.bytes.data.truncate(new_len);
        }

        this.bytes.length = Partial::Present(new_len);
    }
}

/// [`crate::mode::Prove`] mode implementation for the [`Bytes`] component
#[perfect_derive(Clone)]
struct ProveImpl<'normal> {
    previous: Source<'normal, Bytes<Normal>>,
    length: usize,
    did_access_length: Cell<bool>,
    reads: RefCell<RangeSet2<usize>>,
    writes: PartialVec<u8>,
}

impl<'normal> ProveImpl<'normal> {
    /// Get the length of the bytes without recording access.
    fn unrecorded_len(&self) -> usize {
        self.length
    }

    /// Check if the length needs to be included in the proof.
    fn need_length_in_proof(&self) -> bool {
        self.did_access_length.get()
            || !self.reads.borrow().is_empty()
            || !self.writes.is_all_undefined()
    }

    /// Check if any byte in the given address range was accessed (read or written).
    fn was_accessed(&self, addr_range: Range<usize>) -> bool {
        let query_range = RangeSet2::from(addr_range.clone());
        if self.reads.borrow().intersects(&query_range) {
            return true;
        }

        self.writes.is_any_defined(addr_range)
    }
}

/// [`crate::mode::Verify`] mode implementation for the [`Bytes`] component
#[perfect_derive(Clone)]
struct VerifyImpl {
    length: Partial<usize>,
    data: PartialVec<u8>,
}

impl VerifyImpl {
    /// Check if the entire byte array is completely absent.
    ///
    /// This means length and underlying pages are all absent.
    fn is_completely_absent(&self) -> bool {
        if let Partial::Present(_) = self.length {
            return false;
        }

        self.data.is_all_undefined()
    }
}

/// Construct a range `start..start+len` then clamp it so it doesn't extend beyond `total_len`.
fn clamp_range(total_len: usize, start: usize, len: usize) -> Range<usize> {
    let end = start.saturating_add(len).min(total_len);
    start..end
}

/// Arity of internal nodes in the Merkle tree that holds the pages
const NODE_ARITY: usize = 4;

/// Size of a page in bytes
const PAGE_SIZE: usize = 4096;
