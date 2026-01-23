// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a byte array-like structure
//!
//! See [`DataSpace`] for more details.

use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Range;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
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
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::DeserialiserError;
use crate::merkle_proof::DeserialiserNode;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::Partial;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::SuspendedResult;
use crate::merkle_proof::descend_tree;
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
use crate::partial_vec::RangeEntry;
use crate::serialisation::elem::Elem;
use crate::serialisation::serialise;

/// Byte array-like state component which allows reading and writing values of various types
#[repr(transparent)]
pub struct DataSpace<M: Mode> {
    /// Mode-specific representation
    ///
    /// See [`DataSpaceTemplate`].
    data_space: M::Select<DataSpaceTemplate>,
}

impl<M: DataSpaceMode> DataSpace<M> {
    /// Create a new data space of the given length in bytes.
    pub fn new(len: usize) -> Self {
        M::new(len)
    }

    /// Check if there is nothing in the data space.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the length of the data space in bytes.
    pub fn len(&self) -> usize {
        M::len(self)
    }

    /// Read a value of type `E` at the given address.
    ///
    /// # Safety
    ///
    /// The caller must ensure the read is within bounds of the data space.
    pub unsafe fn read<E: Elem>(&self, addr: usize) -> E {
        unsafe { M::read(self, addr) }
    }

    /// Read multiple values of type `E` starting at the given address.
    ///
    /// The output will be stored in the provided slice. The slice thereby determines how many
    /// values will be read.
    ///
    /// The values will be read contiguously from the data space. That means they are adjacent in
    /// the data space without any gaps. Alignment is not respected.
    ///
    /// # Panics
    ///
    /// Panics if the read is not within bounds of the data space.
    pub fn read_all<E: Elem>(&self, addr: usize, values: &mut [E]) {
        let total_len = values
            .len()
            .checked_mul(E::STORED_SIZE.get())
            .expect("Total length should not overflow");
        assert!(self.len().saturating_sub(addr) >= total_len);

        for (idx, value) in values.iter_mut().enumerate() {
            let address = E::STORED_SIZE
                .get()
                .checked_mul(idx)
                .expect("Address overflow")
                .checked_add(addr)
                .expect("Address overflow");

            // SAFETY: We ensured the total length is within bounds and the address is valid.
            *value = unsafe { self.read::<E>(address) };
        }
    }

    /// Write a value of type `E` at the given address.
    ///
    /// # Safety
    ///
    /// The caller must ensure the write is within bounds of the data space.
    pub unsafe fn write<E: Elem>(&mut self, addr: usize, value: E) {
        unsafe { M::write(self, addr, value) }
    }

    /// Write multiple values of type `E` starting at the given address.
    ///
    /// The values will be written contiguously to the data space. That means they are adjacent in
    /// the data space without any gaps. Alignment is not respected.
    ///
    /// # Panics
    ///
    /// Panics if the write is not within bounds of the data space.
    pub fn write_all<E: Elem + Copy>(&mut self, addr: usize, values: &[E]) {
        let total_len = values
            .len()
            .checked_mul(E::STORED_SIZE.get())
            .expect("Total length should not overflow");
        assert!(self.len().saturating_sub(addr) >= total_len);

        for (idx, value) in values.iter().copied().enumerate() {
            let address = E::STORED_SIZE
                .get()
                .checked_mul(idx)
                .expect("Address overflow")
                .checked_add(addr)
                .expect("Address overflow");

            // SAFETY: We ensured the total length is within bounds and the address is valid.
            unsafe { self.write::<E>(address, value) };
        }
    }
}

impl DataSpace<Normal> {
    /// Start proof generation for this data space.
    pub fn start_proof(&self) -> DataSpace<Prove<'_>> {
        DataSpace {
            data_space: ProveImpl {
                source: Source::from(self),
                did_access_length: Cell::new(false),
                reads: Default::default(),
                writes: Default::default(),
            },
        }
    }

    /// Fill the data space with a given value.
    ///
    /// This method is specialised to the `Normal` mode to make it more efficient.
    pub fn fill(&mut self, value: u8) {
        self.data_space.fill(value);
    }
}

impl DataSpace<Verify> {
    /// Construct a [`DataSpace`] which is absent apart from its length.
    pub fn absent(size: usize) -> Self {
        DataSpace {
            data_space: VerifyImpl {
                length: Partial::Present(size),
                data: PartialVec::empty(),
            },
        }
    }

    /// Write bytes to the data space, creating new present pages if needed.
    ///
    /// This is rarely what you want. It is primarily useful in tests.
    ///
    /// This is very similar to using [`DataSpace::write_all`] with `u8`, but with the important
    /// distinction that this method will implicitly zero-initialise absent or blinded pages. So if
    /// you write a single byte to an absent page, the written byte will be reflected in the page's
    /// contents, but the remainder of the page will be zero-bytes.
    pub fn populate_pages_with_bytes(&mut self, addr: usize, bytes: &[u8]) {
        self.data_space.data.define(addr, bytes.to_vec());

        let end = addr.saturating_add(bytes.len());

        let start_page = addr & PAGE_MASK;
        let mut end_page = end & PAGE_MASK;

        // Unless end coincides with a page boundary, we need to bump the end page by 1 to make it
        // the exclusive range end.
        if end > end_page {
            end_page = end_page.saturating_add(PAGE_SIZE);
        }

        let mut offset = start_page;
        let gaps = self
            .data_space
            .data
            .range(start_page..end_page)
            .filter_map(|entry| {
                let current_offset = offset;
                offset += entry.width();

                match entry {
                    RangeEntry::Undefined { length } => Some((current_offset, length)),
                    RangeEntry::Defined { .. } => None,
                }
            })
            .collect::<Vec<_>>(); // Need to collect to avoid lifetime issues

        // Fill in the gaps with zero-bytes. After all this method promises to zero-initialise
        // absent or blinded pages.
        for (gap_addr, gap_zeros) in gaps {
            self.data_space.data.define(gap_addr, vec![0u8; gap_zeros]);
        }
    }
}

impl<M: CloneDataSpaceMode> Clone for DataSpace<M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

impl<M: CloneDataSpaceMode> CloneState for DataSpace<M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl<M: EncodeDataSpaceMode> Encode for DataSpace<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        M::encode(self, encoder)
    }
}

impl<C> Decode<C> for DataSpace<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: u64 = Decode::decode(decoder)?;
        let len = len as usize;

        let mut this: Self = DataSpace::new(len);
        decoder.reader().read(&mut this.data_space)?;

        Ok(this)
    }
}

impl Foldable<HashFold> for DataSpace<Normal> {
    fn fold(&self, builder: HashFold) -> Hash {
        let length = self.data_space.len();
        let length_node =
            Hash::hash_encodable(length as u64).expect("Hashing length should not fail");

        let generator = |idx: usize| {
            let address = PAGE_SIZE
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of PAGE_SIZE bytes before");
            let address_end = address.checked_add(PAGE_SIZE).expect("Address overflow");

            let data = &self.data_space[address..address_end];
            Hash::hash_bytes(data)
        };

        let pages = length.div_ceil(PAGE_SIZE);

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(pages, NODE_ARITY, &generator));
        builder.done()
    }
}

impl Foldable<HashFold> for DataSpace<Prove<'_>> {
    fn fold(&self, builder: HashFold) -> Hash {
        let length = self.data_space.source.data_space.len();
        let length_node =
            Hash::hash_encodable(length as u64).expect("Hashing length should not fail");

        let generator = |idx: usize| {
            let address = PAGE_SIZE
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of PAGE_SIZE bytes before");
            let address_end = address.checked_add(PAGE_SIZE).expect("Address overflow");
            let address_range = address..address_end;

            let mut data = self.data_space.source.data_space[address_range.clone()].to_vec();
            for (index, bytes) in self.data_space.writes.defined_range(address_range.clone()) {
                let data_from = &mut data[index..];
                let len = bytes.len().min(data_from.len());
                data_from[..len].copy_from_slice(&bytes[..len]);
            }

            Hash::hash_bytes(&data)
        };

        let pages = length.div_ceil(PAGE_SIZE);

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(pages, NODE_ARITY, &generator));
        builder.done()
    }
}

impl Foldable<MerkleTreeFold> for DataSpace<Prove<'_>> {
    fn fold(&self, builder: MerkleTreeFold) -> MerkleTree {
        let length = self.data_space.unrecorded_len();
        let length_data = serialise(length as u64).expect("Serialising length should not fail");
        let length_needed = self.data_space.need_length_in_proof();
        let length_node = MerkleTree::make_merkle_leaf(length_data, length_needed);

        let page_tree_generator = |idx| {
            let address = PAGE_SIZE
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of PAGE_SIZE bytes before");
            let address_end = address.checked_add(PAGE_SIZE).expect("Address overflow");
            let address_range = address..address_end;

            let accessed = self.data_space.was_accessed(address_range.clone());
            let data = self.data_space.source.data_space[address_range].to_vec();
            MerkleTree::make_merkle_leaf(data, accessed)
        };

        let pages = length.div_ceil(PAGE_SIZE);

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(
            pages,
            NODE_ARITY,
            &page_tree_generator,
        ));
        builder.done()
    }
}

impl Foldable<PartialHashFold<'_>> for DataSpace<Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        if self.data_space.is_completely_absent() {
            return builder.previous();
        }

        // The length must be present if the space is not completely absent. Otherwise we can't
        // properly construct the partial Merkle tree and therefore obtain the final hash.
        let Some(len) = self.data_space.length.clone().to_present() else {
            return PartialHash::InvalidProof;
        };
        let length_hash = Hash::hash_encodable(len as u64).expect("Hashing length should not fail");

        let page_hash_generator = |idx| {
            let address = PAGE_SIZE
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of PAGE_SIZE bytes before");

            let range = address..address + PAGE_SIZE;
            let page = self.data_space.data.continuous_defined_range(range.clone());

            match page {
                None => {
                    if self.data_space.data.is_any_defined(range) {
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

        let mut builder = builder.into_node_fold();
        builder.add(&PartialHash::Present(length_hash));
        builder.add(&IndexableSeqAsTree::new(
            len.div_ceil(PAGE_SIZE),
            NODE_ARITY,
            &page_hash_generator,
        ));
        builder.done()
    }
}

impl FromProof for DataSpace<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, length) = proof.next_branch_with(|proof| proof.into_leaf::<u64>())?;
        let length = length.map_present(|len| len as usize);

        let (proof, data) = proof.next_branch_with(|proof| {
            // When the length node is present, we can properly parse all pages.
            // But when the length node is not present, we cannot parse any pages. This needs to be
            // validated. In other words, the node for the pages must be blinded or absent.
            let Partial::Present(len) = length else {
                // XXX: We can't pick whether this is a node or leaf given we don't know the
                // length. However, absent or blinded leaves are encoded the same way as nodes.
                // In the case where the node is present (which is an error in here), we would
                // trigger an unexpected leaf error instead of the more appropriate error below.
                let proof = proof.into_node()?;

                // When the node for the pages is present, that's a problem. There may be pages and
                // we don't know how to extract them because we don't know how many there are.
                if let Partial::Present(_) = proof.presence() {
                    return Err(DeserialiserError::custom(
                        BadProofError::LengthAbsentButPagesPresent,
                    ));
                }

                return proof.done(PartialVec::empty());
            };

            let mut partial_data = PartialVec::empty();

            let mut for_leaf = |idx, proof: D| {
                // The index is the page number, but the page ID is the starting address.
                let address = PAGE_SIZE
                    .checked_mul(idx)
                    .expect("This should not overflow");

                let result = proof.into_leaf_raw::<PAGE_SIZE>()?;
                let result = result.map(|data| {
                    if let Partial::Present(data) = data {
                        let data = Vec::from(data as Box<[u8]>);
                        partial_data.define(address, data);
                    }
                });

                Ok(result)
            };

            let num_leaves = len.div_ceil(PAGE_SIZE);
            let result = descend_tree(proof, NODE_ARITY, 0, num_leaves, &mut for_leaf)?;

            Ok(result.map(|()| partial_data))
        })?;

        proof.done(DataSpace {
            data_space: VerifyImpl { length, data },
        })
    }
}

impl<M: DataSpaceMode> PartialEq for DataSpace<M> {
    fn eq(&self, other: &Self) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        for idx in 0..len {
            // SAFETY: We know that `idx < len` from the loop condition. Therefore, the reads are
            // always within the maximum bounds.
            unsafe {
                if self.read::<u8>(idx) != other.read::<u8>(idx) {
                    return false;
                }
            }
        }

        true
    }
}

/// Modal template for the [`DataSpace`] component
///
/// This type helps us pick the representation of [`DataSpace`] for each mode by implementing [`Modal`].
enum DataSpaceTemplate {}

impl Modal for DataSpaceTemplate {
    type Normal = memmap2::MmapMut;

    type Prove<'normal> = ProveImpl<'normal>;

    type Verify = VerifyImpl;
}

/// Mode types that implement this trait support common operations on [`DataSpace`] components
///
/// See [`DataSpace`] for a more convenient interface to this trait.
pub trait DataSpaceMode: Mode {
    /// See [`DataSpace::new`].
    fn new(len: usize) -> DataSpace<Self>;

    /// See [`DataSpace::len`].
    fn len(this: &DataSpace<Self>) -> usize;

    /// See [`DataSpace::read`].
    #[expect(
        clippy::missing_safety_doc,
        reason = "Safety requirements are documented in `DataSpace::read`"
    )]
    unsafe fn read<E: Elem>(this: &DataSpace<Self>, addr: usize) -> E;

    /// See [`DataSpace::write`].
    #[expect(
        clippy::missing_safety_doc,
        reason = "Safety requirements are documented in `DataSpace::write`"
    )]
    unsafe fn write<E: Elem>(this: &mut DataSpace<Self>, addr: usize, value: E);
}

impl DataSpaceMode for Normal {
    fn new(len: usize) -> DataSpace<Self> {
        let data_space = memmap2::MmapMut::map_anon(len).expect("Failed to allocate address space");

        assert_eq!(
            data_space.as_ptr().align_offset(PAGE_SIZE),
            0,
            "Address space must be page-aligned"
        );

        DataSpace { data_space }
    }

    fn len(this: &DataSpace<Self>) -> usize {
        this.data_space.len()
    }

    unsafe fn read<E: Elem>(this: &DataSpace<Self>, addr: usize) -> E {
        debug_assert!(addr + E::STORED_SIZE.get() <= this.data_space.len());
        unsafe { E::read_unaligned(this.data_space.as_ptr().add(addr)) }
    }

    unsafe fn write<E: Elem>(this: &mut DataSpace<Self>, addr: usize, value: E) {
        debug_assert!(addr + E::STORED_SIZE.get() <= this.data_space.len());
        unsafe { value.write_unaligned(this.data_space.as_mut_ptr().add(addr)) }
    }
}

impl DataSpaceMode for Prove<'_> {
    fn new(len: usize) -> DataSpace<Self> {
        DataSpace {
            data_space: ProveImpl {
                source: Source::from(DataSpace::new(len)),
                did_access_length: Cell::new(false),
                reads: Default::default(),
                writes: Default::default(),
            },
        }
    }

    fn len(this: &DataSpace<Self>) -> usize {
        this.data_space.did_access_length.set(true);
        this.data_space.unrecorded_len()
    }

    unsafe fn read<E: Elem>(this: &DataSpace<Self>, addr: usize) -> E {
        let addr_range = addr_range::<E>(addr);

        let read_range = RangeSet2::from(addr_range.clone());
        this.data_space.reads.borrow_mut().union_with(&read_range);

        let mut data = this.data_space.source.data_space[addr_range.clone()].to_vec();

        // We need to overlay previously written bytes on to the source data. Without this step,
        // writes would not be reflected in subsequent reads.
        let prev_bytewise_writes = this.data_space.writes.defined_range(addr_range.clone());
        for (index, bytes) in prev_bytewise_writes {
            let data_from = &mut data[index..];
            let len = bytes.len().min(data_from.len());
            data_from[..len].copy_from_slice(&bytes[..len]);
        }

        unsafe { E::read_unaligned(data.as_ptr()) }
    }

    unsafe fn write<E: Elem>(this: &mut DataSpace<Self>, addr: usize, value: E) {
        let mut data = vec![0u8; E::STORED_SIZE.get()];
        this.read_all(addr, &mut data);

        // SAFETY: The vector has been allocated with sufficient space.
        unsafe {
            value.write_unaligned(data.as_mut_ptr());
        }

        this.data_space.writes.define(addr, data);
    }
}

impl DataSpaceMode for Verify {
    fn new(len: usize) -> DataSpace<Self> {
        DataSpace {
            data_space: VerifyImpl {
                length: Partial::Present(len),
                data: PartialVec::from(vec![0u8; len]),
            },
        }
    }

    fn len(this: &DataSpace<Self>) -> usize {
        match this.data_space.length {
            Partial::Present(len) => len,
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            }
        }
    }

    unsafe fn read<E: Elem>(this: &DataSpace<Self>, addr: usize) -> E {
        let mut data = vec![0; E::STORED_SIZE.get()];

        let Some(data_chunks) = this
            .data_space
            .data
            .continuous_defined_range(addr_range::<E>(addr))
        else {
            // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
            unsafe { not_found() }
        };

        let mut offset = 0;
        for page_chunk in data_chunks {
            data[offset..][..page_chunk.len()].copy_from_slice(page_chunk);
            offset += page_chunk.len();
        }

        unsafe { E::read_unaligned(data.as_ptr()) }
    }

    unsafe fn write<E: Elem>(this: &mut DataSpace<Self>, addr: usize, value: E) {
        let mut data = vec![0u8; E::STORED_SIZE.get()];
        this.read_all(addr, &mut data);

        // SAFETY: The vector has been allocated with sufficient space.
        unsafe {
            value.write_unaligned(data.as_mut_ptr());
        }

        this.data_space.data.define(addr, data);
    }
}

/// Mode types that implement this trait support cloning of [`DataSpace`] components
pub trait CloneDataSpaceMode: Mode {
    /// Clones the given [`DataSpace`] component.
    ///
    /// This clones the entire component, not just the internal value. Consider this when cloning
    /// components in [`Prove`] mode.
    fn clone(this: &DataSpace<Self>) -> DataSpace<Self>;
}

impl CloneDataSpaceMode for Normal {
    fn clone(this: &DataSpace<Self>) -> DataSpace<Self> {
        let mut new: DataSpace<Self> = DataSpace::new(this.data_space.len());
        new.data_space.copy_from_slice(&this.data_space);
        new
    }
}

impl CloneDataSpaceMode for Prove<'_> {
    fn clone(this: &DataSpace<Self>) -> DataSpace<Self> {
        DataSpace {
            data_space: this.data_space.clone(),
        }
    }
}

impl CloneDataSpaceMode for Verify {
    fn clone(this: &DataSpace<Self>) -> DataSpace<Self> {
        DataSpace {
            data_space: this.data_space.clone(),
        }
    }
}

/// Mode types that implement this trait support encoding of [`DataSpace`] components
pub trait EncodeDataSpaceMode: Mode {
    /// Encode the [`DataSpace`] component as a byte vector (e.g. `Vec<u8>`).
    fn encode<E: Encoder>(this: &DataSpace<Self>, encoder: &mut E) -> Result<(), EncodeError>;
}

impl EncodeDataSpaceMode for Normal {
    fn encode<E: Encoder>(this: &DataSpace<Self>, encoder: &mut E) -> Result<(), EncodeError> {
        let len = this.data_space.len() as u64;
        len.encode(encoder)?;

        encoder.writer().write(&this.data_space)
    }
}

impl EncodeDataSpaceMode for Prove<'_> {
    fn encode<E: Encoder>(this: &DataSpace<Self>, encoder: &mut E) -> Result<(), EncodeError> {
        if this.data_space.writes.is_all_undefined() {
            // If no writes were recorded, we can serialise the underlying data space as is.
            return Normal::encode(&this.data_space.source, encoder);
        }

        // This variable keeps the index of the next item from the region that should be written.
        let mut write_index = 0;

        let full_range = 0..this.data_space.unrecorded_len();
        for (addr, data) in this.data_space.writes.defined_range(full_range) {
            // There are items before the current index that have not been written yet.
            if write_index < addr {
                let data = &this.data_space.source.data_space[write_index..addr];
                encoder.writer().write(data)?;
            }

            encoder.writer().write(data)?;

            // Make sure we expect to write the next item after the current range entry in the
            // `PartialVec`.
            write_index = addr.saturating_add(data.len());
        }

        // Write the remaining items from the region that were not written yet.
        let data = &this.data_space.source.data_space[write_index..];
        encoder.writer().write(data)?;

        Ok(())
    }
}

/// [`Prove`] mode implementation for the [`DataSpace`] component
#[derive(Clone)]
struct ProveImpl<'normal> {
    /// State at the start of proof generation
    source: Source<'normal, DataSpace<Normal>>,

    /// Was the length of the data space accessed?
    did_access_length: Cell<bool>,

    /// Addresses that were read
    reads: RefCell<RangeSet2<usize>>,

    /// Addresses that were written, mapped to their latest byte value
    writes: PartialVec<u8>,
}

impl<'normal> ProveImpl<'normal> {
    /// Get the length of the data space without recording access.
    fn unrecorded_len(&self) -> usize {
        self.source.len()
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

/// [`Verify`] mode implementation for the [`DataSpace`] component
#[derive(Clone)]
struct VerifyImpl {
    /// Length of the data space
    length: Partial<usize>,

    /// Available data in this space
    data: PartialVec<u8>,
}

impl VerifyImpl {
    /// Check if the entire data space is completely absent.
    ///
    /// This means length and underlying pages are all absent.
    fn is_completely_absent(&self) -> bool {
        if let Partial::Present(_) = self.length {
            return false;
        }

        self.data.is_all_undefined()
    }
}

/// Generate an address range for a value of type `E` at the given address.
fn addr_range<E: Elem>(addr: usize) -> Range<usize> {
    // We don't specify the type of addition (e.g. wrapping or saturating) to let the compiler
    // pick the fastest option. This will commonly be wrapping addition on modern hardware. We
    // assert this is safe because the safety requirement for this function includes that the
    // read is within bounds of the address space. In other words, we assume the addition won't
    // overflow and hence the type of addition ought not matter.
    let addr_end = addr + E::STORED_SIZE.get();
    addr..addr_end
}

/// Arity of internal nodes in the Merkle tree that holds the pages
const NODE_ARITY: usize = 4;

/// Size of a page in bytes
const PAGE_SIZE: usize = 4096;

/// Bit mask to extract the offset within a page
const OFFSET_MASK: usize = PAGE_SIZE - 1;

/// Bit mask to extract the starting address of a page
const PAGE_MASK: usize = !OFFSET_MASK;

/// Errors indicating a bad proof for [`DataSpace<Verify>`]
#[derive(Debug, thiserror::Error)]
enum BadProofError {
    #[error("Length node is absent but some page nodes are present")]
    LengthAbsentButPagesPresent,
}

#[cfg(test)]
mod tests;
