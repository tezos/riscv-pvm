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

    type DynRegion = memmap2::MmapMut;

    type ManagerRoot = Self;
}

impl ManagerAlloc for Owned {
    fn allocate_region<E: 'static, const LEN: usize>(value: [E; LEN]) -> Self::Region<E, LEN> {
        value
    }

    fn allocate_dyn_region(len: usize) -> Self::DynRegion {
        let region = memmap2::MmapMut::map_anon(len).expect("Failed to allocate dynamic region");

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

    fn dyn_region_len(region: &Self::DynRegion) -> usize {
        region.len()
    }

    unsafe fn dyn_region_read<E: Elem>(region: &Self::DynRegion, address: usize) -> E {
        debug_assert!(address + E::STORED_SIZE.get() <= region.len());

        unsafe { E::read_unaligned(region.as_ptr().add(address)) }
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

    unsafe fn dyn_region_write<E: Elem>(region: &mut Self::DynRegion, address: usize, value: E) {
        debug_assert!(address + E::STORED_SIZE.get() <= region.len());

        unsafe { value.write_unaligned(region.as_mut_ptr().add(address)) }
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

    fn serialise_dyn_region<E: Encoder>(
        region: &Self::DynRegion,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        let len = region.len() as u64;
        len.encode(&mut encoder)?;

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

    fn deserialise_dyn_region<'de, D: Decoder>(
        mut decoder: D,
    ) -> Result<Self::DynRegion, DecodeError> {
        let len = u64::decode(&mut decoder)? as usize;

        let mut target = Owned::allocate_dyn_region(len);
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

    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion {
        let len = region.len();
        let mut new_region = Owned::allocate_dyn_region(len);
        new_region.copy_from_slice(region.deref());
        new_region
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;
    use crate::state_backend::DynCells;
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
        proptest::proptest!(|(address in (0usize..120), value: u64)| {
            let mapping = Owned::allocate_dyn_region(128);
            let mut cells: DynCells<Owned> = DynCells::bind(mapping);

            unsafe {
                cells.write(address, value);
            }

            let bytes = binary::serialise(&cells).unwrap();

            let cells_after: DynCells<Owned> = binary::deserialise(&bytes).unwrap();
            for i in 0..128 {
                unsafe {
                    assert_eq!(cells.read::<u8>(i), cells_after.read::<u8>(i));
                }
            }

            let bytes_after = binary::serialise(&cells_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `ProofGen` backend.
            let proof_cells: DynCells<ProofGen<Ref<'_, Owned>>> =
                DynCells::bind(ProofDynRegion::bind(cells.region_ref()));
            let proof_bytes = binary::serialise(&proof_cells).unwrap();
            assert_eq!(bytes, proof_bytes);
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
