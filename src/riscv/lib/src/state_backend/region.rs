// SPDX-FileCopyrightText: 2023,2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;
use std::ops::Deref;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_tree::MerkleTree;
use octez_riscv_data::merkle_tree::MerkleTreeFold;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::serialise;
use perfect_derive::perfect_derive;

use super::ManagerAlloc;
use super::ManagerBase;
use super::ManagerClone;
use super::ManagerRead;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::default::ConstDefault;
use crate::machine_state::memory::MemoryConfig;
use crate::state::NewState;
use crate::state_backend::Elem;
use crate::state_backend::ProofError;
use crate::state_backend::RegionProj;
use crate::state_backend::normal_backend::region_elem_offset;
use crate::state_backend::proof_backend;
use crate::state_backend::proof_backend::merkle::MERKLE_ARITY;
use crate::state_backend::proof_backend::merkle::MERKLE_LEAF_SIZE;
use crate::state_backend::verify_backend;
use crate::state_context::projection::ApplyCons;
use crate::state_context::projection::CellCons;
use crate::state_context::projection::CellsCons;
use crate::state_context::projection::Projection;
use crate::state_context::projection::ProjectionOffset;

/// Single element of type `E`
#[perfect_derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Cell<E: 'static, M: ManagerBase> {
    region: Cells<E, 1, M>,
}

impl<E: 'static, M: ManagerBase> Cell<E, M> {
    /// Allocate a new cell with the given value.
    pub fn new_with(value: E) -> Self
    where
        M: ManagerAlloc,
    {
        let region = M::allocate_region([value]);
        Self {
            region: Cells::bind(region),
        }
    }

    /// Bind this state to the single element region.
    pub const fn bind(region: M::Region<E, 1>) -> Self {
        Self {
            region: Cells::bind(region),
        }
    }

    /// Obtain the underlying region.
    pub fn into_region(self) -> M::Region<E, 1> {
        self.region.into_region()
    }

    /// Read the value managed by the cell.
    #[inline(always)]
    pub fn read(&self) -> E
    where
        E: Copy,
        M: ManagerRead,
    {
        self.region.read(0)
    }

    /// Write the value managed by the cell.
    #[inline(always)]
    pub fn write(&mut self, value: E)
    where
        M: ManagerWrite,
    {
        self.region.write(0, value)
    }
}

impl<E: 'static> Cell<E, Normal> {
    /// Return a proof-generating version of this Cell.
    pub fn start_proof(&self) -> Cell<E, Prove<'_>> {
        Cell {
            region: self.region.start_proof(),
        }
    }
}

impl<E: ConstDefault, M: ManagerBase> NewState<M> for Cell<E, M> {
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        Self::new_with(E::DEFAULT)
    }
}

impl<E: 'static, M: ManagerBase> From<Cells<E, 1, M>> for Cell<E, M> {
    fn from(region: Cells<E, 1, M>) -> Self {
        Self { region }
    }
}

impl<T: Encode, M: ManagerSerialise> Encode for Cell<T, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.region.encode(encoder)
    }
}

impl<C, E: Decode<C>> Decode<C> for Cell<E, Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let region = Decode::decode(decoder)?;
        Ok(Self { region })
    }
}

impl<E, M: ManagerRead> AsRef<E> for Cell<E, M> {
    #[inline]
    fn as_ref(&self) -> &E {
        self.region.read_ref(0)
    }
}

impl<E, M: ManagerRead> Deref for Cell<E, M> {
    type Target = E;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<E: Clone, M: ManagerClone> CloneState for Cell<E, M> {
    fn clone_state(&self) -> Self {
        self.clone()
    }
}

impl<T, M, F> Foldable<F> for Cell<T, M>
where
    M: ManagerBase,
    F: Fold,
    Cells<T, 1, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.region.fold(builder)
    }
}

impl<E: Decode<()>> FromProof for Cell<E, Verify> {
    fn from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self> {
        let result = Cells::from_proof(proof)?;
        let result = result.map(|region| Cell { region });
        Ok(result)
    }
}

/// Projection from [`Cell`] to its value type `E`
pub struct CellProj<E>(PhantomData<E>);

impl<E: 'static> Projection for CellProj<E> {
    type Subject = CellCons<E>;

    type Target = E;

    type Parameter = ();

    #[inline]
    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        _param: Self::Parameter,
    ) -> &'a Self::Target {
        state.as_ref()
    }

    #[inline]
    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        _param: Self::Parameter,
    ) -> Self::Target
    where
        Self::Target: Copy,
    {
        state.read()
    }

    #[inline]
    fn project_write<'a, MC: MemoryConfig, M: ManagerWrite + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
        _param: Self::Parameter,
        value: Self::Target,
    ) {
        state.write(value);
    }

    fn normal_pointer_offset<MC: MemoryConfig>(_param: Self::Parameter) -> ProjectionOffset {
        let field_offset = std::mem::offset_of!(Cell<E, Normal>, region.region);

        RegionProj::<E, 1>::normal_pointer_offset::<MC>((0,)) + field_offset
    }
}

/// Multiple elements of type `E`
#[repr(transparent)]
pub struct Cells<E: 'static, const LEN: usize, M: ManagerBase> {
    region: M::Region<E, LEN>,
}

impl<E: 'static, const LEN: usize, M: ManagerBase> Cells<E, LEN, M> {
    /// Allocate new cells with the given values.
    pub fn new_with(values: [E; LEN]) -> Self
    where
        M: ManagerAlloc,
    {
        let region = M::allocate_region(values);
        Self { region }
    }

    /// Bind this state to the given region.
    pub const fn bind(region: M::Region<E, LEN>) -> Self {
        Self { region }
    }

    /// Obtain a reference to the underlying region.
    pub fn region_ref(&self) -> &M::Region<E, LEN> {
        &self.region
    }

    /// Obtain the underlying region.
    pub fn into_region(self) -> M::Region<E, LEN> {
        self.region
    }

    /// Read an element in the region.
    #[inline]
    pub fn read(&self, index: usize) -> E
    where
        E: Copy,
        M: ManagerRead,
    {
        M::region_read(&self.region, index)
    }

    /// Obtain a reference to an element in the region.
    pub fn read_ref(&self, index: usize) -> &E
    where
        M: ManagerRead,
    {
        M::region_ref(&self.region, index)
    }

    /// Read all elements in the region.
    #[inline]
    pub fn read_all(&self) -> Vec<E>
    where
        E: Copy,
        M: ManagerRead,
    {
        M::region_read_all(&self.region)
    }

    /// Update an element in the region.
    #[inline]
    pub fn write(&mut self, index: usize, value: E)
    where
        M: ManagerWrite,
    {
        M::region_write(&mut self.region, index, value)
    }

    /// Update all elements in the region.
    #[inline]
    pub fn write_all(&mut self, value: &[E])
    where
        E: Copy,
        M: ManagerWrite,
    {
        M::region_write_all(&mut self.region, value)
    }
}

impl<E: 'static, const LEN: usize> Cells<E, LEN, Normal> {
    /// Return a proof-generating version of these Cells.
    pub fn start_proof(&self) -> Cells<E, LEN, Prove<'_>> {
        Cells {
            region: proof_backend::ProofRegion::bind(&self.region),
        }
    }
}

impl<E: 'static, const LEN: usize> Cells<E, LEN, Normal> {
    /// Obtain the byte offset from a pointer to `Cells<E, LEN, M>` to the memory of the elem at
    /// `index`.
    pub(crate) const fn region_elem_offset(index: usize) -> usize {
        std::mem::offset_of!(Self, region) + region_elem_offset::<E, LEN>(index)
    }
}

impl<E: ConstDefault + 'static, const LEN: usize, M: ManagerBase> NewState<M> for Cells<E, LEN, M> {
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        Self::new_with([E::DEFAULT; LEN])
    }
}

impl<T: Encode, const LEN: usize, M: ManagerSerialise> Encode for Cells<T, LEN, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        M::serialise_region(&self.region, encoder)
    }
}

impl<C, T: Decode<C>, const LEN: usize> Decode<C> for Cells<T, LEN, Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            region: Decode::decode(decoder)?,
        })
    }
}

impl<A: PartialEq<B>, B, const LEN: usize, M: ManagerRead, N: ManagerRead>
    PartialEq<Cells<B, LEN, N>> for Cells<A, LEN, M>
{
    fn eq(&self, other: &Cells<B, LEN, N>) -> bool {
        (0..LEN).all(|i| self.read_ref(i).eq(other.read_ref(i)))
    }
}

impl<T: Eq, const LEN: usize, M: ManagerRead> Eq for Cells<T, LEN, M> {}

impl<E: Clone, const LEN: usize, M: ManagerClone> Clone for Cells<E, LEN, M> {
    fn clone(&self) -> Self {
        Self {
            region: M::clone_region(&self.region),
        }
    }
}

impl<E: Clone, const LEN: usize, M: ManagerClone> CloneState for Cells<E, LEN, M> {
    fn clone_state(&self) -> Self {
        self.clone()
    }
}

impl<T: Encode, const LEN: usize, M: ManagerSerialise> Foldable<HashFold> for Cells<T, LEN, M> {
    fn fold(&self, _builder: HashFold) -> Hash {
        Hash::blake3_hash(self).expect("Hashing should not fail")
    }
}

impl<T: Encode, const LEN: usize> Foldable<MerkleTreeFold> for Cells<T, LEN, Prove<'_>> {
    fn fold(&self, _builder: MerkleTreeFold) -> MerkleTree {
        // RV-282: Break down into multiple leaves if the size of the `Cells`
        // is too large for a proof.

        // The Merkle leaf must hold the serialisation of the initial state.
        // Directly serialising the `Prove` state would produce the serialisation
        // of the final state. Therefore, access the inner region which contains the initial state.
        let data = self.region.inner_region_ref();
        let leaf_data = serialise(data).expect("Serialising cells should not fail");

        let was_accessed = self.region.get_access_info();
        MerkleTree::make_merkle_leaf(leaf_data, was_accessed)
    }
}

impl<E: Decode<()>, const LEN: usize> FromProof for Cells<E, LEN, Verify> {
    fn from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self> {
        let result = proof.into_leaf::<[E; LEN]>()?;
        let result = result.map(|region| {
            let region = match region {
                Partial::Absent | Partial::Blinded(_) => verify_backend::Region::Absent,
                Partial::Present(values) => {
                    let values: Box<[Option<E>; LEN]> = Box::new(values.map(Some));
                    verify_backend::Region::Partial(values)
                }
            };
            super::Cells::bind(region)
        });
        Ok(result)
    }
}

impl<T: Encode, const LEN: usize> Foldable<PartialHashFold<'_>> for Cells<T, LEN, Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        let verify_backend::Region::Partial(values) = &self.region else {
            return builder.previous();
        };

        let values = values
            .iter()
            .filter_map(|item| item.as_ref())
            .collect::<Vec<_>>();

        if values.is_empty() {
            // Nothing has changed.
            return builder.previous();
        }

        if values.len() != LEN {
            // Some elements are missing, so we cannot compute the full hash.
            return PartialHash::InvalidProof;
        }

        let Ok(values) = <[&T; LEN]>::try_from(values) else {
            unreachable!("We checked the length before")
        };

        let hash =
            Hash::blake3_hash(values).expect("Hashing element in partial region should not fail");
        PartialHash::Present(hash)
    }
}

/// Projection from [`Cells`] to its element type `E`
pub struct CellsProj<E, const LEN: usize>(PhantomData<E>);

impl<E: 'static, const LEN: usize> Projection for CellsProj<E, LEN> {
    type Subject = CellsCons<E, LEN>;

    type Target = E;

    type Parameter = <RegionProj<E, LEN> as Projection>::Parameter;

    #[inline]
    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> &'a Self::Target {
        RegionProj::<E, LEN>::project_ref::<MC, M>(&state.region, param)
    }

    #[inline]
    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> Self::Target
    where
        Self::Target: Copy,
    {
        RegionProj::<E, LEN>::project_read::<MC, M>(&state.region, param)
    }

    #[inline]
    fn project_write<'a, MC: MemoryConfig, M: ManagerWrite + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
        value: Self::Target,
    ) {
        RegionProj::<E, LEN>::project_write::<MC, M>(&mut state.region, param, value);
    }

    fn normal_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> ProjectionOffset {
        let field_offset = std::mem::offset_of!(Cells<E, LEN, Normal>, region);

        RegionProj::<E, LEN>::normal_pointer_offset::<MC>(param) + field_offset
    }
}

/// Multiple elements of an unspecified type
pub struct DynCells<M: ManagerBase> {
    region: M::DynRegion,
}

impl<M: ManagerBase> DynCells<M> {
    /// Allocate a new dynamic region with the given length in bytes.
    pub fn new(len: usize) -> Self
    where
        M: ManagerAlloc,
    {
        let region = M::allocate_dyn_region(len);
        Self { region }
    }

    /// Bind this state to the given dynamic region.
    pub fn bind(region: M::DynRegion) -> Self {
        Self { region }
    }

    /// Obtain a reference to the underlying dynamic region.
    pub fn region_ref(&self) -> &M::DynRegion {
        &self.region
    }

    /// Obtain the underlying dynamic region.
    pub fn into_region(self) -> M::DynRegion {
        self.region
    }

    /// Is the dynamic region empty?
    pub fn is_empty(&self) -> bool
    where
        M: ManagerRead,
    {
        self.len() == 0
    }

    /// Retrieve the number of bytes in the dynamic region.
    #[inline]
    pub fn len(&self) -> usize
    where
        M: ManagerRead,
    {
        M::dyn_region_len(&self.region)
    }

    /// Read an element in the region. `address` is in bytes.
    ///
    /// # Safety
    ///
    /// See [`ManagerRead::dyn_region_read`] for safety requirements.
    #[inline]
    pub unsafe fn read<E: Elem>(&self, address: usize) -> E
    where
        M: ManagerRead,
    {
        unsafe { M::dyn_region_read(&self.region, address) }
    }

    /// Read elements from the region. `address` is in bytes.
    #[inline]
    pub fn read_all<E: Elem>(&self, address: usize, values: &mut [E])
    where
        M: ManagerRead,
    {
        M::dyn_region_read_all(&self.region, address, values)
    }

    /// Update an element in the region. `address` is in bytes.
    ///
    /// # Safety
    ///
    /// See [`ManagerWrite::dyn_region_write`] for safety requirements.
    #[inline]
    pub unsafe fn write<E: Elem>(&mut self, address: usize, value: E)
    where
        M: ManagerWrite,
    {
        unsafe { M::dyn_region_write(&mut self.region, address, value) }
    }

    /// Update multiple elements in the region. `address` is in bytes.
    #[inline]
    pub fn write_all<E: Elem + Copy>(&mut self, address: usize, values: &[E])
    where
        M: ManagerWrite + ManagerRead,
    {
        M::dyn_region_write_all(&mut self.region, address, values)
    }
}

impl DynCells<Normal> {
    /// Return a proof-generating version of these DynCells.
    pub fn start_proof(&self) -> DynCells<Prove<'_>> {
        DynCells {
            region: proof_backend::ProofDynRegion::bind(&self.region),
        }
    }
}

impl<M: ManagerSerialise> Encode for DynCells<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        M::serialise_dyn_region(&self.region, encoder)
    }
}

impl<C> Decode<C> for DynCells<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = u64::decode(decoder)? as usize;

        let mut region = Normal::allocate_dyn_region(len);
        decoder.reader().read(&mut region)?;

        Ok(DynCells { region })
    }
}

impl<M: ManagerRead, N: ManagerRead> PartialEq<DynCells<N>> for DynCells<M> {
    fn eq(&self, other: &DynCells<N>) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        for i in 0..len {
            // SAFETY: We know that `i < len` from the loop condition. Therefore, the reads are
            // always within the maximum bounds.
            unsafe {
                if self.read::<u8>(i) != other.read::<u8>(i) {
                    return false;
                }
            }
        }

        true
    }
}

impl<M: ManagerRead> Eq for DynCells<M> {}

impl<M: ManagerClone> Clone for DynCells<M> {
    fn clone(&self) -> Self {
        Self {
            region: M::clone_dyn_region(&self.region),
        }
    }
}

impl<M: ManagerClone> CloneState for DynCells<M> {
    fn clone_state(&self) -> Self {
        self.clone()
    }
}

impl<M: ManagerSerialise> Foldable<HashFold> for DynCells<M> {
    fn fold(&self, builder: HashFold) -> Hash {
        let length = self.len();
        let length_node = Hash::blake3_hash(length as u64).expect("Hashing length should not fail");

        let generator = |idx: usize| {
            let address = MERKLE_LEAF_SIZE
                .get()
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of MERKLE_LEAF_SIZE bytes before");

            // SAFETY: The chunk writer will only request data within the bounds that we specified.
            // Given we provided the correct length, this is safe.
            let data = unsafe { self.read::<[u8; MERKLE_LEAF_SIZE.get()]>(address) };
            Hash::blake3_hash_bytes(&data)
        };

        let pages = length.div_ceil(MERKLE_LEAF_SIZE.get());

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(pages, MERKLE_ARITY, &generator));
        builder.done()
    }
}

impl FromProof for DynCells<Verify> {
    fn from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, length) = proof.next_branch_with(|proof| proof.into_leaf::<u64>())?;
        let length = length.to_present().map(|len| len as usize);

        let (proof, pages) = proof.next_branch_with(|proof| {
            // When the length node is present, we can properly parse all pages.
            // But when the length node is not present, we cannot parse any pages. This needs to be
            // validated. In other words, the node for the pages must be blinded or absent.
            let Some(len) = length else {
                // XXX: We can't pick whether this is a node or leaf given we don't know the
                // length. However, absent or blinded leaves are encoded the same way as nodes.
                // In the case where the node is present (which is an error in here), we would
                // trigger an unexpected leaf error instead of the more appropriate error below.
                let proof = proof.into_node()?;

                // When the node for the pages is present, that's a problem. There may be pages and
                // we don't know how to extract them because we don't know how many there are.
                if let Partial::Present(_) = proof.presence() {
                    return Err(merkle_proof::DeserialiserError::custom(
                        ProofError::AbsentProof,
                    ));
                }

                return proof.done(Vec::new());
            };

            let mut pages = Vec::new();
            let mut for_leaf = |idx, proof: D| {
                // The index is the page number, but the page ID is the starting address.
                let Some(address) = MERKLE_LEAF_SIZE.get().checked_mul(idx) else {
                    return Err(merkle_proof::DeserialiserError::custom(
                        verify_backend::PageIdOverflow,
                    ));
                };
                let page_id = verify_backend::PageId::from_address(address);

                let result = proof.into_leaf_raw()?;
                let result = result.map(|data| {
                    if let Partial::Present(data) = data {
                        pages.push((page_id, data));
                    }
                });

                Ok(result)
            };

            let num_leaves = len.div_ceil(MERKLE_LEAF_SIZE.get());
            let result =
                merkle_proof::descend_tree(proof, MERKLE_ARITY, 0, num_leaves, &mut for_leaf)?;

            Ok(result.map(|()| pages))
        })?;

        // After the parsing, convert all pages into cells.
        let region = verify_backend::DynRegion::from_pages(length, pages);
        let pages = super::DynCells::bind(region);

        proof.done(pages)
    }
}

impl Foldable<MerkleTreeFold> for DynCells<Prove<'_>> {
    fn fold(&self, builder: MerkleTreeFold) -> MerkleTree {
        let length = self.region.unrecorded_len();
        let length_data = serialise(length as u64).expect("Serialising length should not fail");
        let length_needed = self.region.need_length_in_proof();
        let length_node = MerkleTree::make_merkle_leaf(length_data, length_needed);

        let region = self.region_ref();
        let reads = region.get_read();
        let writes = region.get_write();

        let page_tree_generator = |idx| {
            let address = MERKLE_LEAF_SIZE
                .get()
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of MERKLE_LEAF_SIZE bytes before");
            let range = address..(address + MERKLE_LEAF_SIZE.get());
            let accessed = reads.includes_range(range.clone()) || writes.includes_range(range);
            let data: [u8; MERKLE_LEAF_SIZE.get()] =
                unsafe { region.inner_dyn_region_read(address) };
            MerkleTree::make_merkle_leaf(data.to_vec(), accessed)
        };

        let pages = length.div_ceil(MERKLE_LEAF_SIZE.get());

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(
            pages,
            MERKLE_ARITY,
            &page_tree_generator,
        ));
        builder.done()
    }
}

impl Foldable<PartialHashFold<'_>> for DynCells<Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        if self.region.is_completely_absent() {
            return PartialHash::Previous;
        }

        // The length must be present if the region is not completely absent. Otherwise we can't
        // properly construct the partial Merkle tree and therefore obtain the final hash.
        let Some(len) = self.region.len_opt() else {
            return PartialHash::InvalidProof;
        };
        let length_hash = Hash::blake3_hash(len as u64).expect("Hashing length should not fail");

        let page_hash_generator = |idx| {
            let address = MERKLE_LEAF_SIZE
                .get()
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of MERKLE_LEAF_SIZE bytes before");
            let page_id = verify_backend::PageId::from_address(address);

            let page = self.region.get_partial_page(page_id);
            match page {
                verify_backend::PartialState::Incomplete => PartialHash::InvalidProof,
                verify_backend::PartialState::Absent => PartialHash::Previous,
                verify_backend::PartialState::Complete(data) => {
                    let hash = Hash::blake3_hash(data).expect("Hashing page should not fail");
                    PartialHash::Present(hash)
                }
            }
        };

        let mut builder = builder.into_node_fold();
        builder.add(&PartialHash::Present(length_hash));
        builder.add(&IndexableSeqAsTree::new(
            len.div_ceil(MERKLE_LEAF_SIZE.get()),
            MERKLE_ARITY,
            &page_hash_generator,
        ));
        builder.done()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::num::NonZeroUsize;

    use bincode::Encode;
    use bincode::enc::Encoder;
    use bincode::error::EncodeError;
    use octez_riscv_data::foldable::Fold;
    use octez_riscv_data::foldable::Foldable;
    use octez_riscv_data::foldable::NodeFold;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::utils::catch_not_found;

    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::state::NewState;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;
    use crate::state_backend::DynCells;
    use crate::state_backend::Elem;
    use crate::state_backend::ManagerBase;
    use crate::state_backend::ProofPart;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned;

    /// Dummy type that helps us implement custom normalisation via [`Elem`]
    #[repr(C, packed)]
    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
    struct Flipper {
        a: u8,
        b: u8,
    }

    impl ConstDefault for Flipper {
        const DEFAULT: Self = Self { a: 0, b: 0 };
    }

    impl Encode for Flipper {
        fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
            self.b.encode(encoder)?;
            self.a.encode(encoder)?;
            Ok(())
        }
    }

    impl Elem for Flipper {
        const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(2).unwrap();

        unsafe fn read_unaligned(source: *const u8) -> Self {
            unsafe {
                Self {
                    a: source.add(1).read(),
                    b: source.read(),
                }
            }
        }

        unsafe fn write_unaligned(self, dest: *mut u8) {
            unsafe {
                dest.add(1).write(self.a);
                dest.write(self.b);
            }
        }
    }

    backend_test!(test_region_overlap, F, {
        const LEN: usize = 64;

        let mut array1: Cells<u64, LEN, F> = Cells::new();
        let mut array2: Cells<u64, LEN, F> = Cells::new();

        // Allocate two consecutive arrays
        // let mut array1 = manager.allocate_region(array1_place);
        let mut array1_mirror = [0; LEN];

        for (i, item) in array1_mirror.iter_mut().enumerate() {
            // Ensure the array is zero-initialised.
            assert_eq!(array1.read(i), 0);

            // Then write something random in it.
            let value = rand::random();
            array1.write(i, value);
            assert_eq!(array1.read(i), value);

            // Retain the value for later.
            *item = value;
        }

        let array1_vec = array1.read_all();
        assert_eq!(array1_vec, array1_mirror);

        for i in 0..LEN {
            // Check the array is zero-initialised and that the first array
            // did not mess with the second array.
            assert_eq!(array2.read(i), 0);

            // Write a random value to it.
            let value = rand::random();
            array2.write(i, value);
            assert_eq!(array2.read(i), value);
        }

        for (i, item) in array1_mirror.into_iter().enumerate() {
            // Ensure that writing to the second array didn't mess with the
            // first array.
            assert_eq!(item, array1.read(i));
        }
    });

    backend_test!(test_cell_overlap, F, {
        let mut cell1: Cell<[u64; 4], F> = Cell::new();
        let mut cell2: Cell<[u64; 4], F> = Cell::new();

        // Cell should be zero-initialised.
        assert_eq!(cell1.read(), [0; 4]);
        assert_eq!(cell2.read(), [0; 4]);

        // Write something to cell 1 and check.
        let cell1_value: [u64; 4] = rand::random();
        cell1.write(cell1_value);
        assert_eq!(cell1.read(), cell1_value);

        // Second cell should still be zero-initialised
        assert_eq!(cell2.read(), [0; 4]);

        // Write something to cell 2 and check.
        let cell2_value: [u64; 4] = rand::random();
        cell2.write(cell2_value);
        assert_eq!(cell2.read(), cell2_value);

        // Cell 1 should not have its value changed.
        assert_eq!(cell1.read(), cell1_value);
    });

    backend_test!(
        #[should_panic]
        test_dynregion_oob_2,
        F,
        {
            const LEN: usize = 4096;

            let mut state = DynCells::<F>::new(LEN);

            // This should panic because we are trying to write an element at the address which
            // corresponds to the end of the buffer.
            unsafe {
                state.write(LEN * Flipper::STORED_SIZE.get(), Flipper { a: 1, b: 2 });
            }
        }
    );

    backend_test!(test_dynregion_stored_format, F, {
        // Writing to one item of the region must convert to stored format.
        let mut region = DynCells::<F>::new(4096);

        unsafe {
            region.write(0, Flipper { a: 13, b: 37 });
            assert_eq!(region.read::<Flipper>(0), Flipper { a: 13, b: 37 });
        }

        let buffer = unsafe { region.read::<[u8; 2]>(0) };
        assert_eq!(buffer, [37, 13]);

        // Writing to the entire region must convert properly to stored format.
        region.write_all::<Flipper>(0, &[
            Flipper { a: 11, b: 22 },
            Flipper { a: 13, b: 24 },
            Flipper { a: 15, b: 26 },
            Flipper { a: 17, b: 28 },
        ]);

        let mut buff = [Flipper::default(); 4];
        region.read_all::<Flipper>(0, &mut buff);
        assert_eq!(buff, [
            Flipper { a: 11, b: 22 },
            Flipper { a: 13, b: 24 },
            Flipper { a: 15, b: 26 },
            Flipper { a: 17, b: 28 },
        ]);

        let buffer = unsafe { region.read::<[u8; 8]>(0) };
        assert_eq!(buffer, [22, 11, 24, 13, 26, 15, 28, 17]);
    });

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MyFoo(u64);

    impl ConstDefault for MyFoo {
        const DEFAULT: Self = MyFoo(42);
    }

    // Test that the Atom layout initialises the underlying Cell correctly.
    backend_test!(test_cell_init, F, {
        assert_eq!(Cell::<MyFoo, F>::new().read(), MyFoo::DEFAULT);
    });

    // Test that the Array layout initialises the underlying Cells correctly.
    backend_test!(test_cells_init, F, {
        assert_eq!(
            Cells::<MyFoo, 1337, F>::new().read_all(),
            [MyFoo::DEFAULT; 1337]
        );
    });

    #[test]
    fn test_struct_example() {
        struct Foo<M: ManagerBase> {
            bar: Cell<u64, M>,
            qux: Cells<u8, 64, M>,
        }

        impl<F: Fold, M: ManagerBase> Foldable<F> for Foo<M>
        where
            Cell<u64, M>: Foldable<F>,
            Cells<u8, 64, M>: Foldable<F>,
        {
            fn fold(&self, builder: F) -> <F as Fold>::Folded {
                let mut builder = builder.into_node_fold();
                builder.add(&self.bar);
                builder.add(&self.qux);
                builder.done()
            }
        }

        fn inner(bar: u64, qux: [u8; 64]) {
            let mut foo = Foo::<Normal> {
                bar: Cell::new(),
                qux: Cells::new(),
            };

            foo.bar.write(bar);
            foo.qux.write_all(&qux);

            // Obtain the state hash
            let hash = Hash::from_foldable(&foo);

            // Obtain the Merkle tree via the `Prove` mode
            let mut proof_foo = Foo {
                bar: foo.bar.start_proof(),
                qux: foo.qux.start_proof(),
            };

            let tree = MerkleTree::from_foldable(&proof_foo);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Modify the values so they appear in the proof
            proof_foo.bar.write(bar.wrapping_add(1));
            proof_foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));

            // Obtain the Merkle tree, again, to make sure the root hash has not changed
            let tree = MerkleTree::from_foldable(&proof_foo);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Produce a proof
            let proof = merkle_tree_to_merkle_proof(tree);
            let proof_hash = proof.root_hash();
            assert_eq!(hash, proof_hash);

            // Apply the same modification on the `Normal` state in order to obtain
            // the final state hash
            foo.bar.write(bar.wrapping_add(1));
            foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));
            let final_hash = Hash::from_foldable(&foo);

            // Verify the proof and check the final hash
            catch_not_found(|| {
                let mut verify_foo = {
                    let (bar, qux) = deserialise_owned::deserialise(ProofPart::Present(&proof))
                        .unwrap()
                        .0;
                    Foo { bar, qux }
                };

                assert_eq!(bar, verify_foo.bar.read());
                assert_eq!(qux, verify_foo.qux.read_all().as_slice());

                // Apply the same modification to the state in `Verify` mode and check
                // that the final hash is correct
                verify_foo.bar.write(bar.wrapping_add(1));
                verify_foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));

                let verify_hash = PartialHash::from_foldable(Some(&proof), &verify_foo)
                    .to_hash()
                    .unwrap();
                assert_eq!(verify_hash, final_hash)
            })
            .unwrap();
        }

        proptest::proptest!(|(bar: u64, qux: [u8; 64])| {
            inner(bar, qux);
        });
    }
}
