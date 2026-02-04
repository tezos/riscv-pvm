// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::ops::RangeInclusive;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::data_space::CloneDataSpaceMode;
use octez_riscv_data::components::data_space::DataSpace;
use octez_riscv_data::components::data_space::DataSpaceMode;
use octez_riscv_data::components::data_space::EncodeDataSpaceMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use super::Address;
use super::address_to_page_index;
use crate::state_backend::NarrowlySized;

/// Tracks access permissions for each page
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct PagePermissions<const PAGES: usize, M: Mode> {
    pages: DataSpace<M>,
}

impl<const PAGES: usize, M: Mode> PagePermissions<PAGES, M> {
    /// Check if the memory at `address..address+length` can be accessed.
    ///
    /// # Safety
    ///
    /// The address and length must be valid for an address space consisting of a number of `PAGES`.
    /// This function is not defined for address and length combinations which are out of bounds.
    #[inline]
    pub unsafe fn can_access(&self, pages: RangeInclusive<u64>) -> bool
    where
        M: DataSpaceMode,
    {
        for page in pages {
            if unsafe { self.pages.read::<u8>(page as usize) == 0 } {
                return false;
            }
        }

        true
    }

    /// Same as [`Self::can_access`], but slightly faster. Requires additional invariants to be upheld.
    /// The generic parameter `E` is used to specify the type of the element being accessed. It
    /// also determines the length of the access.
    ///
    /// # Safety
    ///
    /// The length must be non-zero and less than the page size. Otherwise, same as
    /// [`Self::can_access`].
    #[inline]
    pub unsafe fn can_access_narrow<E>(&self, address: Address) -> bool
    where
        E: NarrowlySized,
        M: DataSpaceMode,
    {
        let start_page = address_to_page_index(address);
        if unsafe { self.pages.read::<u8>(start_page) == 0 } {
            return false;
        }

        let end_address = address
            .wrapping_add(E::NARROW_SIZE.get() as Address)
            .wrapping_sub(1);

        let end_page = address_to_page_index(end_address);
        unsafe { self.pages.read::<u8>(end_page) != 0 }
    }

    /// Change the access permissions for the given range.
    pub fn modify_access(&mut self, pages: RangeInclusive<u64>, accessible: bool)
    where
        M: DataSpaceMode,
    {
        pages.filter(|&page| page < PAGES as u64).for_each(|page| {
            // SAFETY: TODO
            unsafe {
                self.pages.write(page as usize, accessible as u8);
            }
        })
    }

    /// Reset access permissions on all pages.
    pub fn reset(&mut self)
    where
        M: DataSpaceMode,
    {
        for page in 0..PAGES {
            // SAFETY: TODO
            unsafe {
                self.pages.write(page, 0u8);
            }
        }
    }
}

impl<'normal, const PAGES: usize> Provable<'normal> for PagePermissions<PAGES, Normal> {
    type Prover = PagePermissions<PAGES, Prove<'normal>>;

    fn start_proof(&self) -> PagePermissions<PAGES, Prove<'_>> {
        PagePermissions {
            pages: self.pages.start_proof(),
        }
    }
}

impl<const PAGES: usize, M: DataSpaceMode> Default for PagePermissions<PAGES, M> {
    fn default() -> Self {
        PagePermissions {
            pages: DataSpace::new(PAGES),
        }
    }
}

impl<const PAGES: usize, M: CloneDataSpaceMode> CloneState for PagePermissions<PAGES, M> {
    fn clone_state(&self) -> Self {
        Self {
            pages: self.pages.clone_state(),
        }
    }
}

impl<C, const PAGES: usize> Decode<C> for PagePermissions<PAGES, Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pages = Decode::decode(decoder)?;
        Ok(Self { pages })
    }
}

impl<const PAGES: usize, M: EncodeDataSpaceMode> Encode for PagePermissions<PAGES, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.pages.encode(encoder)
    }
}

impl<const PAGES: usize, M, F> Foldable<F> for PagePermissions<PAGES, M>
where
    M: Mode,
    F: Fold,
    DataSpace<M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.pages.fold(builder)
    }
}
impl<const PAGES: usize> FromProof for PagePermissions<PAGES, Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let result = DataSpace::from_proof(proof)?;
        let result = result.map(|pages| Self { pages });
        Ok(result)
    }
}
