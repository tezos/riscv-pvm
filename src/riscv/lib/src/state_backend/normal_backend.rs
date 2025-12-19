// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::ops::Deref;

use bincode::Encode;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::EncodeError;
use octez_riscv_data::mode::Normal;

use super::Elem;
use super::ManagerAlloc;
use super::ManagerBase;
use super::ManagerClone;
use super::ManagerRead;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::machine_state::memory::PAGE_SIZE;

impl ManagerBase for Normal {
    type DynRegion = memmap2::MmapMut;

    type ManagerRoot = Self;
}

impl ManagerAlloc for Normal {
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

impl ManagerRead for Normal {
    fn dyn_region_len(region: &Self::DynRegion) -> usize {
        region.len()
    }

    unsafe fn dyn_region_read<E: Elem>(region: &Self::DynRegion, address: usize) -> E {
        debug_assert!(address + E::STORED_SIZE.get() <= region.len());

        unsafe { E::read_unaligned(region.as_ptr().add(address)) }
    }
}

impl ManagerWrite for Normal {
    unsafe fn dyn_region_write<E: Elem>(region: &mut Self::DynRegion, address: usize, value: E) {
        debug_assert!(address + E::STORED_SIZE.get() <= region.len());

        unsafe { value.write_unaligned(region.as_mut_ptr().add(address)) }
    }
}

impl ManagerSerialise for Normal {
    fn serialise_dyn_region<E: Encoder>(
        region: &Self::DynRegion,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        let len = region.len() as u64;
        len.encode(&mut encoder)?;

        encoder.writer().write(region)
    }
}

impl ManagerClone for Normal {
    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion {
        let len = region.len();
        let mut new_region = Normal::allocate_dyn_region(len);
        new_region.copy_from_slice(region.deref());
        new_region
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use octez_riscv_data::components::atom::Atom;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_data::serialisation::serialise;

    use super::*;
    use crate::state_backend::DynCells;
    use crate::state_backend::proof_backend::ProofDynRegion;

    /// Ensure [`Atom`] can be serialised and deserialised in a consistent way.
    #[test]
    fn atom_serialise() {
        proptest::proptest!(|(value: u64)|{
            let cell: Atom<u64, Normal> = Atom::new(value);
            let bytes = serialise(&cell).unwrap();

            let cell_after: Atom<u64, Normal> = deserialise(&bytes).unwrap();
            assert_eq!(cell.read(), cell_after.read());

            let bytes_after = serialise(&cell_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `Prove` mode.
            let proof_cell: Atom<u64, Prove> = cell.start_proof();
            let proof_bytes = serialise(&proof_cell).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure [`Atom`] can be serialised and deserialised in a consistent way.
    #[test]
    fn atoms_serialise() {
        proptest::proptest!(|(a: u64, b: u64, c: u64)|{
            let atoms: Atom<_, Normal> = Atom::new([a, b, c]);
            let bytes = serialise(&atoms).unwrap();

            let atoms_after: Atom<[u64; 3], Normal> = deserialise(&bytes).unwrap();

            assert_eq!(atoms.read(), atoms_after.read());

            for i in 0..3 {
                assert_eq!(atoms[i], atoms_after[i]);
            }

            let bytes_after = serialise(&atoms_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `Prove` mode.
            let proof_atoms: Atom<_, Prove> = atoms.start_proof();
            let proof_bytes = serialise(&proof_atoms).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure [`DynCells`] can be serialised and deserialised in a consistent way.
    #[test]
    fn dyn_cells_serialise() {
        proptest::proptest!(|(address in (0usize..120), value: u64)| {
            let mapping = Normal::allocate_dyn_region(128);
            let mut cells: DynCells<Normal> = DynCells::bind(mapping);

            unsafe {
                cells.write(address, value);
            }

            let bytes = serialise(&cells).unwrap();

            let cells_after: DynCells<Normal> = deserialise(&bytes).unwrap();
            for i in 0..128 {
                unsafe {
                    assert_eq!(cells.read::<u8>(i), cells_after.read::<u8>(i));
                }
            }

            let bytes_after = serialise(&cells_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `Prove` mode.
            let proof_cells: DynCells<Prove> =
                DynCells::bind(ProofDynRegion::bind(cells.region_ref()));
            let proof_bytes = serialise(&proof_cells).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }

    /// Ensure that [`Atom`] serialises in a way that represents the underlying element
    /// directly instead of wrapping it into an array (as it is an array under the hood).
    #[test]
    fn atom_direct_serialise() {
        let cell: Atom<u64, Normal> = Atom::new(42);
        let binary_value = serialise(cell).unwrap();
        let expected_binary_value = serialise(42u64).unwrap();
        assert_eq!(binary_value, expected_binary_value);
    }
}
