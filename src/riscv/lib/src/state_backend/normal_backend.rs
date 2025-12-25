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
use octez_riscv_data::serialisation::elem::Elem;

use super::ManagerAlloc;
use super::ManagerBase;
use super::ManagerClone;
use super::ManagerRead;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::machine_state::memory::PAGE_SIZE;

impl ManagerBase for Normal {
    type DynRegion = memmap2::MmapMut;
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
    use octez_riscv_data::components::data_space::DataSpace;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::serialisation::deserialise;
    use octez_riscv_data::serialisation::serialise;

    use super::*;

    /// Ensure [`DataSpace`] can be serialised and deserialised in a consistent way.
    #[test]
    fn data_space_serialise() {
        proptest::proptest!(|(address in (0usize..120), value: u64)| {
            let mut space: DataSpace<Normal> = DataSpace::new(128);

            unsafe {
                space.write(address, value);
            }

            let bytes = serialise(&space).unwrap();

            let space_after: DataSpace<Normal> = deserialise(&bytes).unwrap();
            for i in 0..128 {
                unsafe {
                    assert_eq!(space.read::<u8>(i), space_after.read::<u8>(i));
                }
            }

            let bytes_after = serialise(&space_after).unwrap();
            assert_eq!(bytes, bytes_after);

            // Serialisation is consistent with that of the `Prove` mode.
            let proof_space: DataSpace<Prove> = space.start_proof();
            let proof_bytes = serialise(&proof_space).unwrap();
            assert_eq!(bytes, proof_bytes);
        });
    }
}
