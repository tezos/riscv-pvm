// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::array;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::Deref;

use bincode::Encode;
use bincode::de::Decode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::DecodeError;
use bincode::error::EncodeError;

use super::Elem;
use super::EnrichedValue;
use super::EnrichedValueLinked;
use super::ManagerAlloc;
use super::ManagerBase;
use super::ManagerClone;
use super::ManagerDeserialise;
use super::ManagerRead;
use super::ManagerReadWrite;
use super::ManagerSerialise;
use super::ManagerWrite;
use super::StaticCopy;
use crate::machine_state::memory::PAGE_SIZE;

/// Manager that allows state binders to own the state storage
#[derive(Clone, Copy, Debug)]
pub struct Owned;

impl Owned {
    /// Get the byte offset from a pointer to `Owned::Region` to the start of the element at `index`.
    pub(crate) const fn region_elem_offset<E: 'static, const LEN: usize>(index: usize) -> usize {
        assert!(index < LEN, "Out of bounds access for region");

        index * std::mem::size_of::<E>()
    }
}

impl ManagerBase for Owned {
    type Region<E: 'static, const LEN: usize> = [E; LEN];

    type DynRegion<const LEN: usize> = memmap2::MmapMut;

    type EnrichedCell<V: EnrichedValue> = (V::E, V::D);

    type ManagerRoot = Self;

    fn enrich_cell<V: EnrichedValueLinked>(cell: Self::Region<V::E, 1>) -> Self::EnrichedCell<V> {
        let [value] = cell;
        let derived = V::derive(&value);
        (value, derived)
    }

    fn as_devalued_cell<V: EnrichedValue>(cell: &Self::EnrichedCell<V>) -> &Self::Region<V::E, 1> {
        array::from_ref(&cell.0)
    }
}

impl ManagerAlloc for Owned {
    fn allocate_region<E: 'static, const LEN: usize>(value: [E; LEN]) -> Self::Region<E, LEN> {
        value
    }

    fn allocate_dyn_region<const LEN: usize>() -> Self::DynRegion<LEN> {
        let region = memmap2::MmapMut::map_anon(LEN).expect("Failed to allocate dynamic region");

        assert_eq!(
            region.as_ptr().align_offset(PAGE_SIZE.get() as usize),
            0,
            "The dynamic region must be page-aligned"
        );

        region
    }
}

impl ManagerRead for Owned {
    fn region_read<E: StaticCopy, const LEN: usize>(
        region: &Self::Region<E, LEN>,
        index: usize,
    ) -> E {
        region[index]
    }

    fn region_ref<E: 'static, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> &E {
        &region[index]
    }

    fn region_read_all<E: StaticCopy, const LEN: usize>(region: &Self::Region<E, LEN>) -> Vec<E> {
        region.to_vec()
    }

    fn dyn_region_read<E: Elem, const LEN: usize>(
        region: &Self::DynRegion<LEN>,
        address: usize,
    ) -> E {
        assert!(address + E::STORED_SIZE_USIZE.get() <= LEN);

        // SAFETY: The assertion above ensures that the address can be read for at least
        // `E::STORED_SIZE` bytes.
        unsafe { E::read_unaligned(region.as_ptr().add(address)) }
    }

    fn dyn_region_read_all<E: Elem, const LEN: usize>(
        region: &Self::DynRegion<LEN>,
        address: usize,
        values: &mut [E],
    ) {
        for (i, value) in values.iter_mut().enumerate() {
            *value = Self::dyn_region_read::<E, LEN>(
                region,
                E::STORED_SIZE_USIZE
                    .get()
                    .wrapping_mul(i)
                    .wrapping_add(address),
            );
        }
    }

    fn enriched_cell_read_stored<V>(cell: &Self::EnrichedCell<V>) -> V::E
    where
        V: EnrichedValue,
        V::E: Copy,
    {
        cell.0
    }

    fn enriched_cell_read_derived<V>(cell: &Self::EnrichedCell<V>) -> V::D
    where
        V: EnrichedValue,
        V::D: Copy,
    {
        cell.1
    }

    fn enriched_cell_ref_stored<V>(cell: &Self::EnrichedCell<V>) -> &V::E
    where
        V: EnrichedValue,
    {
        &cell.0
    }
}

impl ManagerWrite for Owned {
    fn region_write<E: 'static, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    ) {
        region[index] = value;
    }

    fn region_write_all<E: StaticCopy, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        value: &[E],
    ) {
        region.copy_from_slice(value)
    }

    fn dyn_region_write<E: Elem, const LEN: usize>(
        region: &mut Self::DynRegion<LEN>,
        address: usize,
        value: E,
    ) {
        assert!(address + E::STORED_SIZE_USIZE.get() <= LEN);

        // SAFETY: The assertion above ensures that the address can be written for at least
        // `E::STORED_SIZE` bytes.
        unsafe { value.write_unaligned(region.as_mut_ptr().add(address)) }
    }

    fn dyn_region_write_all<E: Elem + Copy, const LEN: usize>(
        region: &mut Self::DynRegion<LEN>,
        address: usize,
        values: &[E],
    ) {
        for (i, value) in values.iter().enumerate() {
            Self::dyn_region_write::<E, LEN>(
                region,
                E::STORED_SIZE_USIZE
                    .get()
                    .wrapping_mul(i)
                    .wrapping_add(address),
                *value,
            );
        }
    }

    fn enriched_cell_write<V>(cell: &mut Self::EnrichedCell<V>, value: V::E)
    where
        V: EnrichedValueLinked,
    {
        let derived = V::derive(&value);

        cell.0 = value;
        cell.1 = derived;
    }
}

impl ManagerReadWrite for Owned {
    fn region_replace<E: StaticCopy, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    ) -> E {
        mem::replace(&mut region[index], value)
    }
}

impl ManagerSerialise for Owned {
    fn serialise_region<T: Encode + 'static, const LEN: usize, E: Encoder>(
        region: &Self::Region<T, LEN>,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        for elem in region.iter() {
            elem.encode(&mut encoder)?;
        }

        Ok(())
    }

    fn serialise_dyn_region<const LEN: usize, E: Encoder>(
        region: &Self::DynRegion<LEN>,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        encoder.writer().write(region)
    }
}

impl ManagerDeserialise for Owned {
    fn deserialise_region<T: Decode<()> + 'static, const LEN: usize, D: Decoder<Context = ()>>(
        mut decoder: D,
    ) -> Result<Self::Region<T, LEN>, DecodeError> {
        let mut items = array::from_fn(|_| MaybeUninit::<T>::uninit());

        for item in items.iter_mut() {
            item.write(T::decode(&mut decoder)?);
        }

        // SAFETY: We have iterated through all items and initialised them.
        let values = items.map(|value| unsafe { value.assume_init() });
        Ok(values)
    }

    fn deserialise_dyn_region<'de, const LEN: usize, D: Decoder>(
        mut decoder: D,
    ) -> Result<Self::DynRegion<LEN>, DecodeError> {
        let mut target = Owned::allocate_dyn_region::<LEN>();
        decoder.reader().read(&mut target)?;
        Ok(target)
    }
}

impl ManagerClone for Owned {
    fn clone_region<E: Clone + 'static, const LEN: usize>(
        region: &Self::Region<E, LEN>,
    ) -> Self::Region<E, LEN> {
        region.clone()
    }

    fn clone_dyn_region<const LEN: usize>(region: &Self::DynRegion<LEN>) -> Self::DynRegion<LEN> {
        let mut new_region = Owned::allocate_dyn_region::<LEN>();
        new_region.copy_from_slice(region.deref());
        new_region
    }

    fn clone_enriched_cell<V: EnrichedValue>(cell: &Self::EnrichedCell<V>) -> Self::EnrichedCell<V>
    where
        V::E: Clone,
        V::D: Clone,
    {
        cell.clone()
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;
    use crate::state_backend::DynCells;
    use crate::state_backend::EnrichedCell;
    use crate::state_backend::FnManagerIdent;
    use crate::state_backend::Ref;
    use crate::state_backend::proof_backend::ProofDynRegion;
    use crate::state_backend::proof_backend::ProofGen;
    use crate::state_backend::proof_backend::ProofRegion;
    use crate::storage::binary;

    /// Ensure [`Cell`] can be serialised and deserialised in a consistent way.
    #[test]
    fn cell_serialise() {
        proptest::proptest!(|(value: u64)|{
            let region = [value; 1];
            let cell: Cell<u64, Owned> = Cell::bind(region);
            let bytes = binary::serialise(&cell).unwrap();

            let cell_after: Cell<u64, Owned> = binary::deserialise(&bytes).unwrap();
            assert_eq!(cell.read(), cell_after.read());

            let bytes_after = binary::serialise(&cell_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `ProofGen` backend.
            let proof_cell: Cell<u64, ProofGen<Ref<'_, Owned>>> =
                Cell::bind(ProofRegion::bind(&region));
            let proof_bytes = binary::serialise(&proof_cell).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure [`Cells`] can be serialised and deserialised in a consistent way.
    #[test]
    fn cells_serialise() {
        proptest::proptest!(|(a: u64, b: u64, c: u64)|{
            let cell: Cells<u64, 3, Owned> = Cells::bind([a, b, c]);
            let bytes = binary::serialise(&cell).unwrap();

            let cell_after: Cells<u64, 3, Owned> = binary::deserialise(&bytes).unwrap();

            assert_eq!(cell.read_all(), cell_after.read_all());

            for i in 0..3 {
                assert_eq!(cell.read(i), cell_after.read(i));
            }

            let bytes_after = binary::serialise(&cell_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `ProofGen` backend.
            let proof_cells: Cells<u64, 3, ProofGen<Ref<'_, Owned>>> =
                Cells::bind(ProofRegion::bind(cell.region_ref()));
            let proof_bytes = binary::serialise(&proof_cells).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure [`DynCells`] can be serialised and deserialised in a consistent way.
    #[test]
    fn dyn_cells_serialise() {
        proptest::proptest!(|(address in (0usize..120), value: u64)|{
            let mapping = Owned::allocate_dyn_region::<128>();
            let mut cells: DynCells<128, Owned> = DynCells::bind(mapping);
            cells.write(address, value);
            let bytes = binary::serialise(&cells).unwrap();

            let cells_after: DynCells<128, Owned> = binary::deserialise(&bytes).unwrap();
            for i in 0..128 {
                assert_eq!(cells.read::<u8>(i), cells_after.read::<u8>(i));
            }

            let bytes_after = binary::serialise(&cells_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `ProofGen` backend.
            let proof_cells: DynCells<128, ProofGen<Ref<'_, Owned>>> =
                DynCells::bind(ProofDynRegion::bind(cells.region_ref()));
            let proof_bytes = binary::serialise(&proof_cells).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure [`EnrichedCell`] can be serialised and deserialised in a consistent way.
    #[test]
    fn enriched_cell_serialise() {
        pub struct Enriching;

        impl EnrichedValue for Enriching {
            type E = u64;
            type D = T;
        }

        #[derive(Clone, Copy)]
        pub struct T(u64);

        impl<'a> From<&'a u64> for T {
            fn from(value: &'a u64) -> Self {
                T(value.wrapping_add(1))
            }
        }

        proptest::proptest!(|(value: u64)| {
            let cell = Cell::bind([0u64]);
            let mut cell: EnrichedCell<Enriching, Owned> = EnrichedCell::bind(cell);
            cell.write(value);

            let read_value = cell.read_ref_stored();

            assert_eq!(value, *read_value);
            let bytes = binary::serialise(&cell).unwrap();

            let cell_after: EnrichedCell<Enriching, Owned> = binary::deserialise(&bytes).unwrap();

            assert_eq!(*cell.read_ref_stored(), *cell_after.read_ref_stored());

            let derived = cell.read_derived();
            let derived_after = cell_after.read_derived();

            assert_eq!(T::from(read_value).0, derived.0);
            assert_eq!(derived.0, derived_after.0);

            // Serialisation is consistent with that of the `ProofGen` backend.
            let proof_cell: EnrichedCell<Enriching, Ref<'_, Owned>> = EnrichedCell::bind(cell.struct_ref::<FnManagerIdent>());
            let proof_bytes = binary::serialise(&proof_cell).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure [`EnrichedCell`] is serialized identically to [`Cell`].
    #[test]
    fn enriched_cell_serialise_match_cell() {
        pub struct Enriching;
        pub struct Fun;

        impl EnrichedValue for Enriching {
            type E = u64;
            type D = Fun;
        }

        impl<'a> From<&'a u64> for Fun {
            fn from(_value: &'a u64) -> Self {
                Self
            }
        }

        proptest::proptest!(|(value: u64)| {
            let cell = Cell::bind([0u64]);
            let mut ecell: EnrichedCell<Enriching, Owned> = EnrichedCell::bind(cell);
            let mut cell: Cell<u64, Owned> = Cell::bind([0; 1]);
            ecell.write(value);
            cell.write(value);

            assert_eq!(value, ecell.read_stored());
            assert_eq!(value, cell.read());

            let ebytes = binary::serialise(&ecell).unwrap();
            let cbytes = binary::serialise(&cell).unwrap();

            assert_eq!(ebytes, cbytes, "Serializing EnrichedCell and Cell should match");
        });
    }

    /// Ensure that [`Cell`] serialises in a way that represents the underlying element
    /// directly instead of wrapping it into an array (as it is an array under the hood).
    #[test]
    fn cell_direct_serialise() {
        let cell: Cell<u64, Owned> = Cell::bind([42]);
        let binary_value = binary::serialise(cell).unwrap();
        let expected_binary_value = binary::serialise(42u64).unwrap();
        assert_eq!(binary_value, expected_binary_value);
    }

    /// Check that regions are properly initialised.
    #[test]
    fn region_init() {
        proptest::proptest!(|(init_value: [u64; 17])| {
            let region = Owned::allocate_region(init_value);
            proptest::prop_assert_eq!(region, init_value);
        });
    }
}
