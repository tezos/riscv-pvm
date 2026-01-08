// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::ops::Index;
use std::ops::RangeInclusive;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use super::Address;
use super::address_to_page_index;
use crate::array_utils::boxed_from_fn;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;
use crate::state_backend::NarrowlySized;
use crate::state_backend::proof_backend::merkle::MERKLE_ARITY;

/// Tracks access permissions for each page
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct PagePermissions<const PAGES: usize, M: Mode> {
    pages: Box<[Atom<bool, M>; PAGES]>,
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
        M: ManagerRead,
    {
        for page in pages {
            if unsafe { !self.pages.get_unchecked(page as usize).read() } {
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
        M: ManagerRead,
    {
        let start_page = address_to_page_index(address);
        if unsafe { !self.pages.get_unchecked(start_page).read() } {
            return false;
        }

        let end_address = address
            .wrapping_add(E::NARROW_SIZE.get() as Address)
            .wrapping_sub(1);

        let end_page = address_to_page_index(end_address);
        unsafe { self.pages.get_unchecked(end_page).read() }
    }

    /// Change the access permissions for the given range.
    pub fn modify_access(&mut self, pages: RangeInclusive<u64>, accessible: bool)
    where
        M: ManagerWrite,
    {
        pages.filter(|&page| page < PAGES as u64).for_each(|page| {
            self.pages[page as usize].write(accessible);
        })
    }

    /// Reset access permissions on all pages.
    pub fn reset(&mut self)
    where
        M: ManagerWrite,
    {
        self.pages.iter_mut().for_each(|page| page.write(false));
    }
}

impl<const PAGES: usize> PagePermissions<PAGES, Normal> {
    /// Return a proof-generating version of this PagePermissions.
    pub fn start_proof(&self) -> PagePermissions<PAGES, Prove<'_>> {
        let Ok(pages) = self
            .pages
            .iter()
            .map(Atom::start_proof)
            .collect::<Vec<_>>()
            .try_into()
        else {
            unreachable!("Collecting into an array of the same length should always succeed")
        };

        PagePermissions { pages }
    }
}

impl<const PAGES: usize, M: ManagerAlloc> Default for PagePermissions<PAGES, M> {
    fn default() -> Self {
        PagePermissions {
            pages: boxed_from_fn(Atom::default),
        }
    }
}

impl<const PAGES: usize, M: ManagerClone> CloneState for PagePermissions<PAGES, M> {
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

impl<const PAGES: usize, M: EncodeAtomMode> Encode for PagePermissions<PAGES, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.pages.encode(encoder)
    }
}

impl<const PAGES: usize, M, F> Foldable<F> for PagePermissions<PAGES, M>
where
    M: Mode,
    F: Fold,
    Atom<bool, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let page_generator = |idx| self.pages.index(idx);
        IndexableSeqAsTree::new(PAGES, MERKLE_ARITY, &page_generator).fold(builder)
    }
}
impl<const PAGES: usize> FromProof for PagePermissions<PAGES, Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let result = merkle_proof::Many::<_, MERKLE_ARITY, PAGES>::from_proof(proof)?;
        let result = result.map(|pages| Self {
            pages: pages.into_boxed_array(),
        });
        Ok(result)
    }
}
