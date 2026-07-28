// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a byte array
//!
//! See [`Bytes`] for more details.

pub mod test_utils;

use std::borrow::Borrow;
use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Index;
use std::ops::Range;

use bincode::BorrowDecode;
use bincode::Decode;
use bincode::Encode;
use bincode::de::BorrowDecoder;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::write::Writer;
use bincode::error::DecodeError;
use perfect_derive::perfect_derive;
use range_collections::RangeSet2;

use crate::clone::CloneState;
use crate::codec::LeafCodec;
use crate::codec::LeafDecode;
use crate::codec::LeafEncode;
use crate::foldable::EncodeLeaf;
use crate::foldable::Fold;
use crate::foldable::FoldLeaf;
use crate::foldable::Foldable;
use crate::foldable::FoldableClosure;
use crate::foldable::NodeFold;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::foldable::Unfoldable;
use crate::foldable::seq_tree;
use crate::foldable::seq_tree::DepthAdjustedSeqAsTree;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::foldable::seq_tree::tree_depth;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::Partial;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::proof_tree::MerkleProofFold;
use crate::merkle_proof::proof_tree::MinimumPresence;
use crate::merkle_proof::sequence_as_tree_from_proof;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Provable;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::Source;
use crate::mode::utils::not_found;
use crate::partial_vec::PartialVec;
use crate::partial_vec::RangeEntry;

/// Byte array state component
#[perfect_derive(Debug)]
pub struct Bytes<M: Mode> {
    bytes: M::Select<BytesTemplate>,
}

impl<M: BytesMode> Bytes<M> {
    /// Create a zero-initialised byte array of the given length.
    pub fn new(len: usize) -> Self {
        M::new(len)
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

    /// Folding function that works for both [`Normal`] and [`Prove`] modes, and for any fold that
    /// implements `FoldLeaf`.
    fn fold_with_fold_leaf<D: Borrow<[u8]>, F: Fn(Range<usize>) -> D, Build: FoldLeaf>(
        &self,
        builder: Build,
        length: usize,
        get_data: F,
    ) -> <Build as Fold>::Folded
    where
        u64: LeafEncode<Build::Codec>,
        for<'x> ChunkedPage<'x>: LeafEncode<Build::Codec>,
    {
        let length_node = EncodeLeaf::new(length as u64, "Serialising length should not fail.");

        let get_item = move |range| {
            let data = get_data(range);
            FoldableClosure::new(move |builder: Build| {
                let page = ChunkedPage {
                    chunks: &[data.borrow()],
                };
                builder
                    .fold_leaf(&page)
                    .expect("Serialising page should not fail.")
            })
        };

        self.fold_generic(builder, length, length_node, get_item)
    }
}

impl<'a> Bytes<Prove<'a>> {
    /// Construct the state component in [`Prove`] mode given the source data representing the state
    /// at the beginning of the proof recording.
    pub fn from_raw_source(source: &'a [u8]) -> Self {
        Bytes {
            bytes: ProveImpl {
                previous: Source::borrowed(source),
                length: source.len(),
                did_access_length: Cell::new(false),
                reads: RefCell::new(RangeSet2::empty()),
                writes: PartialVec::default(),
            },
        }
    }
}

impl Bytes<Verify> {
    /// Construct a [`Bytes<Verify>`] which is absent apart from its length.
    pub(crate) fn absent(len: usize) -> Self {
        Bytes {
            bytes: VerifyImpl {
                original_length: Partial::Present(len),
                length: Partial::Present(len),
                data: PartialVec::empty(),
            },
        }
    }

    /// Returns the given range as a contiguous byte slice.
    ///
    /// Panics (via [`not_found`]) if the range extends beyond the length, or includes any
    /// undefined data, indicating the Merkle proof did not include the data required for this
    /// access.
    pub fn partial_slice(&self, range: Range<usize>) -> &[u8] {
        if range.is_empty() {
            return &[];
        }
        if range.start >= self.len() {
            // SAFETY: called only in `Verify` mode
            unsafe { not_found() }
        }
        match self.bytes.data.contiguous_range(range) {
            Some(slice) => slice,
            None => {
                // SAFETY: called only in `Verify` mode
                unsafe { not_found() }
            }
        }
    }

    /// Zero-initialise all undefined bytes in the given range.
    pub(crate) fn zero_init_range(&mut self, range: Range<usize>) {
        let mut offset = range.start;
        let gaps = self
            .bytes
            .data
            .range(range)
            .filter_map(|entry| {
                let current_offset = offset;
                offset += entry.width();

                match entry {
                    RangeEntry::Undefined { length } => Some((current_offset, length)),
                    RangeEntry::Defined { .. } => None,
                }
            })
            .collect::<Vec<_>>(); // Need to collect to avoid lifetime issues

        for (gap_addr, gap_zeros) in gaps {
            self.bytes.data.define(gap_addr, vec![0u8; gap_zeros]);
        }
    }
}

impl<'normal> Provable<'normal> for Bytes<Normal> {
    type Prover = Bytes<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        Bytes::from_raw_source(&self.bytes)
    }
}

impl Bytes<Normal> {
    /// Convert this [`Bytes`] to [`Prove`] mode.
    ///
    /// Implementation matches [`Bytes::from_raw_source`], but takes ownership of the data instead
    /// of borrowing it, with the `previous` set to [`Source::Owned`].
    pub fn into_proof(self) -> Bytes<Prove<'static>> {
        Bytes {
            bytes: ProveImpl {
                length: self.len(),
                previous: Source::owned(self),
                did_access_length: Cell::new(false),
                reads: RefCell::new(RangeSet2::empty()),
                writes: PartialVec::default(),
            },
        }
    }
}

impl Borrow<[u8]> for Bytes<Normal> {
    fn borrow(&self) -> &[u8] {
        &self.bytes
    }
}

impl Index<Range<usize>> for Bytes<Normal> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &[u8] {
        &self.bytes[range]
    }
}

impl<M: BytesMode> Default for Bytes<M> {
    fn default() -> Self {
        M::new(0)
    }
}

impl<M: CloneBytesMode> CloneState for Bytes<M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl<F: FoldLeaf> Foldable<F> for Bytes<Normal>
where
    u64: LeafEncode<F::Codec>,
    for<'x> ChunkedPage<'x>: LeafEncode<F::Codec>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.fold_with_fold_leaf(builder, self.bytes.len(), |addr_range| {
            &self.bytes[addr_range]
        })
    }
}

impl<F: FoldLeaf> Foldable<F> for Bytes<Prove<'_>>
where
    u64: LeafEncode<F::Codec>,
    for<'x> ChunkedPage<'x>: LeafEncode<F::Codec>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.fold_with_fold_leaf(builder, self.bytes.unrecorded_len(), |addr_range| {
            let previous_len = self.bytes.previous.len();
            let previous_range =
                addr_range.start.min(previous_len)..addr_range.end.min(previous_len);

            let mut data = self.bytes.previous[previous_range].to_vec();
            data.resize(addr_range.len(), 0);

            for (index, chunk) in self.bytes.writes.defined_range(addr_range) {
                data[index..][..chunk.len()].copy_from_slice(chunk);
            }

            data
        })
    }
}

impl<C: LeafCodec> Foldable<MerkleProofFold<C>> for Bytes<Prove<'_>>
where
    u64: LeafEncode<C>,
    for<'x> ChunkedPage<'x>: LeafEncode<C>,
{
    fn fold(&self, builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        // Reminder: Merkle trees generated in Prove mode capture the state at beginning of proof
        // generation. This means we need to use `previous` state for the length and data.

        let length = self.bytes.previous.len();
        let length_data = LeafEncode::<C>::leaf_encode(&(length as u64))
            .expect("Serialising length should not fail");
        let is_length_needed = self.bytes.need_length_in_proof();
        let length_constraint = if is_length_needed {
            MinimumPresence::Present
        } else {
            MinimumPresence::MayOmit
        };
        let length_node = MerkleProofFold::new_leaf(length_constraint, length_data);

        let get_item = |range: Range<usize>| {
            let accessed = self.bytes.was_accessed(range.clone());
            let constraint = if accessed {
                MinimumPresence::Present
            } else {
                MinimumPresence::MayOmit
            };

            let data = &self.bytes.previous[range];
            let page = ChunkedPage { chunks: &[data] };

            // We need to serialise the data to be able to recover it later, given that it is
            // variably sized.
            let leaf_data =
                LeafEncode::<C>::leaf_encode(&page).expect("Serialising leaf data should not fail");

            MerkleProofFold::new_leaf(constraint, leaf_data)
        };

        self.fold_generic(builder, length, length_node, get_item)
    }
}

impl<C: LeafCodec> Foldable<PartialHashFold<C>> for Bytes<Verify>
where
    u64: LeafEncode<C>,
    for<'x> ChunkedPage<'x>: LeafEncode<C>,
{
    fn fold(&self, builder: PartialHashFold<C>) -> PartialHash {
        if self.bytes.is_completely_absent() {
            return builder.previous();
        }

        let Some(original_length) = self.bytes.original_length.clone().to_present() else {
            return PartialHash::InvalidProof;
        };

        // The length must be present if the byte array is not completely absent. Otherwise we can't
        // properly construct the partial Merkle tree and therefore obtain the final hash.
        let Some(length) = self.bytes.length.clone().to_present() else {
            return PartialHash::InvalidProof;
        };

        let length_hash = Hash::hash_bytes(
            &LeafEncode::<C>::leaf_encode(&(length as u64))
                .expect("Hashing length should not fail"),
        );
        let length_node = PartialHash::Present(length_hash);

        let generator = |idx: usize| {
            let address = PAGE_SIZE
                    .checked_mul(idx)
                    .expect("This should not overflow as we split the length into chunks of PAGE_SIZE bytes before");
            let address_end = address
                .checked_add(PAGE_SIZE)
                .expect("Address overflow")
                .min(length);
            let range = address..address_end;

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
                    let page = ChunkedPage {
                        chunks: chunks.as_slice(),
                    };
                    let hash = Hash::hash_bytes(
                        &LeafEncode::<C>::leaf_encode(&page)
                            .expect("Hashing encoded bytes should not fail"),
                    );
                    PartialHash::Present(hash)
                }
            }
        };

        let original_pages = original_length.div_ceil(PAGE_SIZE);
        let pages = length.div_ceil(PAGE_SIZE);

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&DepthAdjustedSeqAsTree {
            inner: IndexableSeqAsTree::new(pages, NODE_ARITY, &generator),
            original_depth: tree_depth(original_pages, NODE_ARITY),
            current_depth: tree_depth(pages, NODE_ARITY),
        });
        builder.done()
    }
}

impl Unfoldable for Bytes<Normal> {
    fn unfold<U: Unfold>(source: U) -> Result<Self, UnfoldError> {
        let mut source = source.into_node()?;

        let length: u64 = source.next_branch_with(|source| source.into_leaf())?;
        let length_in_pages = (length as usize).div_ceil(PAGE_SIZE);

        let state = source.next_branch_with(|source| {
            let mut state = bytes::BytesMut::new();

            let mut for_leaf = |_idx, source: U| {
                let page = source.into_leaf::<Page>()?;

                state.extend_from_slice(&page.data[..]);

                Ok(())
            };

            seq_tree::descend_tree(source, NODE_ARITY, length_in_pages, &mut for_leaf)?;

            Ok(state)
        })?;

        Ok(Bytes { bytes: state })
    }
}

impl<C: LeafCodec> FromProof<C> for Bytes<Verify>
where
    u64: LeafDecode<C>,
    Page: LeafDecode<C>,
{
    fn from_proof<Proof: Deserialiser<Codec = C>>(
        proof: Proof,
    ) -> Result<<Proof as Deserialiser>::Suspended<Self>, <Proof as Deserialiser>::Error> {
        sequence_as_tree_from_proof::<u64, Self, _>(
            proof,
            NODE_ARITY,
            |length| {
                let length_in_bytes = length.map_present(|len| len as usize);
                let state = Bytes {
                    bytes: VerifyImpl {
                        original_length: length_in_bytes.clone(),
                        length: length_in_bytes.clone(),
                        data: PartialVec::empty(),
                    },
                };

                let length_in_pages = length_in_bytes.map_present(|len| len.div_ceil(PAGE_SIZE));
                (state, length_in_pages)
            },
            |state, idx, proof| {
                // The index is the page number, but we need the starting address of that page.
                let address = PAGE_SIZE
                    .checked_mul(idx)
                    .expect("This should not overflow");

                let result = proof.into_leaf::<Page>()?;

                let result = result.map(|page| {
                    if let Partial::Present(page) = page {
                        state.bytes.data.define(address, page.data);
                    }
                });

                Ok(result)
            },
        )
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

/// Decode into owned [`Bytes<Normal>`].
impl<Context> Decode<Context> for Bytes<Normal> {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let raw = Vec::decode(decoder)?;
        let bytes = bytes::Bytes::from(raw);
        Ok(Self::from(bytes))
    }
}

/// Decode from borrowed input by delegating to owned decode.
///
/// [`Bytes<Normal>`] owns its backing storage, so borrowed decode does not need
/// a separate representation.
impl<'de, Context> BorrowDecode<'de, Context> for Bytes<Normal> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        Self::decode(decoder)
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
    fn new(len: usize) -> Bytes<Self>;

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
    fn new(len: usize) -> Bytes<Self> {
        Bytes {
            bytes: bytes::BytesMut::zeroed(len),
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

impl EncodeBytesMode for Prove<'_> {
    fn encode<E: bincode::enc::Encoder>(
        this: &Bytes<Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        if this.bytes.writes.is_all_undefined() {
            // If no writes were recorded, we can serialise the underlying byte array as is.
            return this.bytes.previous.encode(encoder);
        }

        // This variable keeps the index of the next item from the region that should be written.
        let mut write_index = 0;

        let full_range = 0..this.bytes.unrecorded_len();
        for (addr, data) in this.bytes.writes.defined_range(full_range) {
            // There are items before the current index that have not been written yet.
            if write_index < addr {
                let data = &this.bytes.previous[write_index..addr];
                encoder.writer().write(data)?;
            }

            encoder.writer().write(data)?;

            // Make sure we expect to write the next item after the current range entry in the
            // `PartialVec`.
            write_index = addr.saturating_add(data.len());
        }

        // Write the remaining items from the region that were not written yet.
        let data = &this.bytes.previous[write_index..];
        encoder.writer().write(data)?;

        Ok(())
    }
}

impl BytesMode for Prove<'_> {
    fn new(len: usize) -> Bytes<Self> {
        Bytes {
            bytes: ProveImpl {
                previous: Source::owned(Bytes::new(len)),
                length: len,
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

        // We need to figure out which bytes are available in the previous block.
        let previous_len = this.bytes.previous.len();
        let from_previous_start = range.start.min(previous_len);
        let from_previous_end = range.end.min(previous_len);
        let from_previous_len = from_previous_end - from_previous_start;

        // The bytes available in the previous block will be a non-strict prefix of the range.
        buffer[..from_previous_len]
            .copy_from_slice(&this.bytes.previous[from_previous_start..from_previous_end]);

        // Bytes beyond the previous length are implicitly zero, if they are not present in the
        // `writes` vector.
        buffer[from_previous_len..].fill(0);

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
        let prev_len = this.len();

        // Resizing can change the hash of a partially-filled boundary page even when no explicit
        // reads/writes happened. Record a single-byte read from the old in-bounds side of the
        // boundary so the proof includes previous page data needed by Verify mode.
        this.bytes
            .record_resize_boundary_dependency(prev_len, new_len);

        if new_len < prev_len {
            // We need to clear out the previously written bytes that are now out of bounds.
            // Otherwise, resizing up again would restore them.
            this.bytes.writes.truncate(new_len);
        }

        this.bytes.length = new_len;

        // NOTE: We do not extend the `writes` vector, as the new bytes are implicitly zero.
    }
}

impl BytesMode for Verify {
    fn new(len: usize) -> Bytes<Self> {
        Bytes {
            bytes: VerifyImpl {
                original_length: Partial::Present(len),
                length: Partial::Present(len),
                data: PartialVec::from(vec![0u8; len]),
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
#[perfect_derive(Clone, Debug)]
struct ProveImpl<'normal> {
    previous: Source<'normal, Bytes<Normal>, [u8]>,
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

    /// Record synthetic read dependency needed for prove-mode resize transitions.
    ///
    /// Resizing can change the contents of leaves in the underlying Merkle tree. This is because
    /// the modified pages are padded with zeros, or truncated. Without this method, the proof tree
    /// is not guaranteed to contain the data that is truncated or extended with zeros. This leads
    /// to problems when trying to compute the state hash after the verification step. Using this
    /// method ensures the leaf will be included in the proof and we will be able to re-hash it
    /// thanks to all necessary data being present.
    fn record_resize_boundary_dependency(&self, prev_len: usize, new_len: usize) {
        let boundary_pos = new_len.min(prev_len);
        if boundary_pos == 0 || prev_len == new_len {
            return;
        }

        let boundary_range = RangeSet2::from((boundary_pos - 1)..boundary_pos);
        self.reads.borrow_mut().union_with(&boundary_range);
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
#[perfect_derive(Clone, Debug)]
struct VerifyImpl {
    original_length: Partial<usize>,
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

/// Byte array capped at [`PAGE_SIZE`] bytes
///
/// This type serves primarily as a validator when decoding. We don't want to decode data for a page
/// that exceeds the maximum page size, as that could lead to memory attacks. So we decode the
/// length first, check it, and then decode the data.
///
/// The encoding dual is [`ChunkedPage`].
pub struct Page {
    data: Vec<u8>,
}

impl Page {
    /// Return the page data as an array. Panics if the page is not a full one of length
    /// `PAGE_SIZE`.
    pub fn full_page(self) -> [u8; PAGE_SIZE] {
        <[u8; PAGE_SIZE]>::try_from(self.data).expect("Should be a page of length `PAGE_SIZE`")
    }
}

impl<C> Decode<C> for Page {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let length = u64::decode(decoder)?;

        if length > PAGE_SIZE as u64 {
            return Err(DecodeError::OtherString(format!(
                "Page length {length} exceeds maximum page size {PAGE_SIZE}"
            )));
        }

        let mut data = vec![0u8; length as usize];
        decoder.reader().read(&mut data)?;

        Ok(Page { data })
    }
}

/// Chunked page capped at [`PAGE_SIZE`] bytes
///
/// This type is useful for hashing and serialising pages from byte slices, or even slices of byte
/// slices.
///
/// The decoding dual is [`Page`].
pub(super) struct ChunkedPage<'a> {
    pub(super) chunks: &'a [&'a [u8]],
}

impl Encode for ChunkedPage<'_> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let length: u64 = self.chunks.iter().map(|chunk| chunk.len() as u64).sum();

        if length > PAGE_SIZE as u64 {
            return Err(bincode::error::EncodeError::OtherString(format!(
                "Total chunk length {length} exceeds maximum page size {PAGE_SIZE}"
            )));
        }

        length.encode(encoder)?;

        let writer = encoder.writer();
        for chunk in self.chunks {
            writer.write(chunk)?;
        }

        Ok(())
    }
}

/// Construct a range `start..start+len` then clamp it so it doesn't extend beyond `total_len`.
fn clamp_range(total_len: usize, start: usize, len: usize) -> Range<usize> {
    let end = start.saturating_add(len).min(total_len);
    start..end
}

/// Arity of internal nodes in the Merkle tree that holds the pages
pub const NODE_ARITY: usize = 2;

/// Size of a page in bytes
///
/// Kept to half that of the maximum read-size: ensuring that at most
/// three pages are pulled in on each read.
pub const PAGE_SIZE: usize = 1024;

#[cfg(test)]
pub(crate) mod tests;
