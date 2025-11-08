// SPDX-FileCopyrightText: 2023,2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;
use std::ops::Deref;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use perfect_derive::perfect_derive;

use super::FnManager;
use super::ManagerAlloc;
use super::ManagerBase;
use super::ManagerClone;
use super::ManagerDeserialise;
use super::ManagerRead;
use super::ManagerReadWrite;
use super::ManagerSerialise;
use super::ManagerWrite;
use super::Ref;
use super::owned_backend::Owned;
use super::proof_backend::ProofGen;
use super::proof_backend::merkle::AccessInfoAggregatable;
use crate::default::ConstDefault;
use crate::machine_state::memory::MemoryConfig;
use crate::state::NewState;
use crate::state_backend::Elem;
use crate::state_backend::RegionProj;
use crate::state_context::projection::ApplyCons;
use crate::state_context::projection::CellCons;
use crate::state_context::projection::CellsCons;
use crate::state_context::projection::Projection;
use crate::state_context::projection::ProjectionOffset;

/// Single element of type `E`
#[perfect_derive(Clone)]
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

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(&'a self) -> Cell<E, F::Output> {
        Cell {
            region: self.region.struct_ref::<F>(),
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

    /// Replace the value managed by the cell, returning the old value.
    #[inline(always)]
    pub fn replace(&mut self, value: E) -> E
    where
        E: Copy,
        M: ManagerReadWrite,
    {
        self.region.replace(0, value)
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

impl<E: Decode<()>, M: ManagerDeserialise> Decode<()> for Cell<E, M> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let region = Decode::decode(decoder)?;
        Ok(Self { region })
    }
}

impl<A: PartialEq<B>, B, M: ManagerRead, N: ManagerRead> PartialEq<Cell<B, N>> for Cell<A, M> {
    fn eq(&self, other: &Cell<B, N>) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl<E: Eq, M: ManagerRead> Eq for Cell<E, M> {}

impl<E: Encode, M: ManagerSerialise> AccessInfoAggregatable for Cell<E, Ref<'_, ProofGen<M>>> {
    fn aggregate_access_info(&self) -> bool {
        self.region.region.get_access_info()
    }
}

impl<E, M: ManagerRead> AsRef<E> for Cell<E, M> {
    #[inline]
    fn as_ref(&self) -> &E {
        M::region_ref(&self.region.region, 0)
    }
}

impl<E, M: ManagerRead> Deref for Cell<E, M> {
    type Target = E;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
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

    fn owned_pointer_offset<MC: MemoryConfig>(_param: Self::Parameter) -> ProjectionOffset {
        let field_offset = std::mem::offset_of!(Cell<E, Owned>, region.region);

        RegionProj::<E, 1>::owned_pointer_offset::<MC>((0,)) + field_offset
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

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(&'a self) -> Cells<E, LEN, F::Output> {
        Cells {
            region: F::map_region(&self.region),
        }
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

    /// Update the element in the region and return the previous value.
    #[inline]
    pub fn replace(&mut self, index: usize, value: E) -> E
    where
        E: Copy,
        M: ManagerReadWrite,
    {
        M::region_replace(&mut self.region, index, value)
    }
}

impl<E: 'static, const LEN: usize> Cells<E, LEN, Owned> {
    /// Obtain the byte offset from a pointer to `Cells<E, LEN, M>` to the memory of the elem at
    /// `index`.
    pub(crate) const fn region_elem_offset(index: usize) -> usize {
        std::mem::offset_of!(Self, region) + Owned::region_elem_offset::<E, LEN>(index)
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

impl<E: Decode<()>, const LEN: usize, M: ManagerDeserialise> Decode<()> for Cells<E, LEN, M> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let region = M::deserialise_region(decoder)?;
        Ok(Self { region })
    }
}

impl<A: PartialEq<B> + Copy, B: Copy, const LEN: usize, M: ManagerRead, N: ManagerRead>
    PartialEq<Cells<B, LEN, N>> for Cells<A, LEN, M>
{
    fn eq(&self, other: &Cells<B, LEN, N>) -> bool {
        (0..LEN).all(|i| self.read(i) == other.read(i))
    }
}

impl<E: Encode, const LEN: usize, M: ManagerSerialise> AccessInfoAggregatable
    for Cells<E, LEN, Ref<'_, ProofGen<M>>>
{
    fn aggregate_access_info(&self) -> bool {
        self.region.get_access_info()
    }
}

impl<E: Clone, const LEN: usize, M: ManagerClone> Clone for Cells<E, LEN, M> {
    fn clone(&self) -> Self {
        Self {
            region: M::clone_region(&self.region),
        }
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

    fn owned_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> ProjectionOffset {
        let field_offset = std::mem::offset_of!(Cells<E, LEN, Owned>, region);

        RegionProj::<E, LEN>::owned_pointer_offset::<MC>(param) + field_offset
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

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(&'a self) -> DynCells<F::Output> {
        DynCells {
            region: F::map_dyn_region(&self.region),
        }
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

impl<M: ManagerSerialise> Encode for DynCells<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        M::serialise_dyn_region(&self.region, encoder)
    }
}

impl<M: ManagerDeserialise> Decode<()> for DynCells<M> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let region = M::deserialise_dyn_region(decoder)?;
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

#[cfg(test)]
pub(crate) mod tests {
    use std::num::NonZeroUsize;

    use bincode::Encode;
    use bincode::enc::Encoder;
    use bincode::error::EncodeError;

    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::state::NewState;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;
    use crate::state_backend::DynCells;
    use crate::state_backend::Elem;

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
}
