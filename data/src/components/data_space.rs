// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a byte array-like structure
//!
//! See [`DataSpace`] for more details.

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::DecodeError;
use bincode::error::EncodeError;

use super::bytes::Bytes;
use crate::clone::CloneState;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::seq_tree::IndexableSeqAsTree;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::SuspendedResult;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::not_found;
use crate::serialisation::elem::Elem;

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
            data_space: Bytes::from_raw_source(&self.data_space),
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
        let data_space = Bytes::absent(size);
        DataSpace { data_space }
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
        let end = addr.saturating_add(bytes.len());

        let start_page = addr & PAGE_MASK;
        let mut end_page = end & PAGE_MASK;

        // Unless end coincides with a page boundary, we need to bump the end page by 1 to make it
        // the exclusive range end.
        if end > end_page {
            end_page = end_page.saturating_add(PAGE_SIZE);
        }

        self.data_space.zero_init_range(start_page..end_page);
        self.data_space.write(addr, bytes);
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

impl<'a, F: Fold> Foldable<F> for DataSpace<Prove<'a>>
where
    Bytes<Prove<'a>>: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        self.data_space.fold(builder)
    }
}

impl<F: Fold> Foldable<F> for DataSpace<Verify>
where
    Bytes<Verify>: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        self.data_space.fold(builder)
    }
}

impl FromProof for DataSpace<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let bytes = Bytes::<Verify>::from_proof(proof)?;
        let bytes = bytes.map(|bytes| DataSpace { data_space: bytes });
        Ok(bytes)
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

    type Prove<'normal> = Bytes<Prove<'normal>>;

    type Verify = Bytes<Verify>;
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
            data_space: Bytes::new(len),
        }
    }

    fn len(this: &DataSpace<Self>) -> usize {
        this.data_space.len()
    }

    unsafe fn read<E: Elem>(this: &DataSpace<Self>, addr: usize) -> E {
        let mut data = vec![0; E::STORED_SIZE.get()];

        let read_bytes = this.data_space.read(addr, &mut data);
        assert_eq!(read_bytes, data.len());

        unsafe { E::read_unaligned(data.as_ptr()) }
    }

    unsafe fn write<E: Elem>(this: &mut DataSpace<Self>, addr: usize, value: E) {
        let mut data = vec![0u8; E::STORED_SIZE.get()];
        this.read_all(addr, &mut data);

        // SAFETY: The vector has been allocated with sufficient space.
        unsafe {
            value.write_unaligned(data.as_mut_ptr());
        }

        let written = this.data_space.write(addr, &data);
        assert_eq!(written, data.len());
    }
}

impl DataSpaceMode for Verify {
    fn new(len: usize) -> DataSpace<Self> {
        DataSpace {
            data_space: Bytes::new(len),
        }
    }

    fn len(this: &DataSpace<Self>) -> usize {
        this.data_space.len()
    }

    unsafe fn read<E: Elem>(this: &DataSpace<Self>, addr: usize) -> E {
        let mut data = vec![0; E::STORED_SIZE.get()];
        let read_bytes = this.data_space.read(addr, &mut data);

        if read_bytes < data.len() {
            // SAFETY: We're in Verify mode, so calling `not_found` is safe.
            unsafe { not_found() }
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

        this.data_space.write(addr, &data);
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
        this.data_space.encode(encoder)
    }
}

/// Arity of internal nodes in the Merkle tree that holds the pages
const NODE_ARITY: usize = super::bytes::NODE_ARITY;

/// Size of a page in bytes
const PAGE_SIZE: usize = super::bytes::PAGE_SIZE;

/// Bit mask to extract the offset within a page
const OFFSET_MASK: usize = PAGE_SIZE - 1;

/// Bit mask to extract the starting address of a page
const PAGE_MASK: usize = !OFFSET_MASK;

#[cfg(test)]
mod tests;
