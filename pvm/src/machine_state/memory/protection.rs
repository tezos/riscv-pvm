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
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::components::vector::CloneVectorMode;
use octez_riscv_data::components::vector::EncodeVectorMode;
use octez_riscv_data::components::vector::Vector;
use octez_riscv_data::components::vector::VectorMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::Unfold;
use octez_riscv_data::foldable::UnfoldError;
use octez_riscv_data::foldable::Unfoldable;
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
pub struct PagePermissions<M: Mode> {
    pages: Vector<Atom<bool, M>, M>,
}

impl<M: Mode> PagePermissions<M> {
    /// Create a new [`PagePermissions`] with `pages` as the number of pages.
    pub fn new(pages: usize) -> Self
    where
        M: AtomMode + VectorMode,
    {
        let values = (0..pages).map(|_| Atom::new(false)).collect();
        let pages = Vector::new(values);
        Self { pages }
    }

    /// Check if the memory at `address..address+length` can be accessed.
    ///
    /// # Safety
    ///
    /// The address and length must be valid for an address space consisting of a number of `pages`.
    /// This function is not defined for address and length combinations which are out of bounds.
    #[inline]
    pub unsafe fn can_access(&self, pages: RangeInclusive<u64>) -> bool
    where
        M: AtomMode + VectorMode,
    {
        for page in pages {
            if unsafe { !self.pages.index_unchecked(page as usize).read() } {
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
        M: AtomMode + VectorMode,
    {
        let start_page = address_to_page_index(address);
        if unsafe { !self.pages.index_unchecked(start_page).read() } {
            return false;
        }

        let end_address = address
            .wrapping_add(E::NARROW_SIZE.get() as Address)
            .wrapping_sub(1);

        let end_page = address_to_page_index(end_address);
        unsafe { self.pages.index_unchecked(end_page).read() }
    }

    /// Change the access permissions for the given range.
    pub fn modify_access(&mut self, pages: RangeInclusive<u64>, accessible: bool)
    where
        M: AtomMode + VectorMode,
    {
        let len_pages = self.pages.len() as u64;

        for page in pages {
            if page >= len_pages {
                break;
            }

            self.pages[page as usize].write(accessible);
        }
    }

    /// Reset access permissions on all pages.
    pub fn reset(&mut self)
    where
        M: AtomMode + VectorMode,
    {
        for page in 0..self.pages.len() {
            self.pages[page].write(false);
        }
    }
}

impl<'normal> Provable<'normal> for PagePermissions<Normal> {
    type Prover = PagePermissions<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        let pages = self.pages.start_proof();
        PagePermissions { pages }
    }
}

impl<M: CloneAtomMode + CloneVectorMode> CloneState for PagePermissions<M> {
    fn clone_state(&self) -> Self {
        Self {
            pages: self.pages.clone_state(),
        }
    }
}

impl<C> Decode<C> for PagePermissions<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pages = Decode::decode(decoder)?;
        Ok(Self { pages })
    }
}

impl<M: EncodeAtomMode + EncodeVectorMode> Encode for PagePermissions<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.pages.encode(encoder)
    }
}

impl<M, F> Foldable<F> for PagePermissions<M>
where
    M: Mode,
    F: Fold,
    Vector<Atom<bool, M>, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.pages.fold(builder)
    }
}

impl FromProof for PagePermissions<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        Vector::from_proof(proof).map(|result| result.map(|pages| Self { pages }))
    }
}

impl Unfoldable for PagePermissions<Normal> {
    fn unfold<U: Unfold>(src: U) -> Result<Self, UnfoldError> {
        let pages = Vector::<Atom<bool, Normal>, Normal>::unfold(src)?;
        Ok(Self { pages })
    }
}
