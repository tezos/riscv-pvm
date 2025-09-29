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
use perfect_derive::perfect_derive;

use super::Address;
use super::address_to_page_index;
use crate::array_utils::boxed_from_fn;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerDeserialise;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerSerialise;
use crate::state_backend::ManagerWrite;
use crate::state_backend::Many;
use crate::state_backend::NarrowlySized;
use crate::state_backend::Ref;

/// State layout for page permissions
pub type PagePermissionsLayout<const PAGES: usize> = Many<Atom<bool>, PAGES>;

/// Tracks access permissions for each page
#[perfect_derive(Clone)]
pub struct PagePermissions<const PAGES: usize, M: ManagerBase> {
    pages: Box<[Cell<bool, M>; PAGES]>,
}

impl<const PAGES: usize, M: ManagerBase> PagePermissions<PAGES, M> {
    /// Bind the given allocated space as a page protections state value.
    pub fn bind(space: AllocatedOf<PagePermissionsLayout<PAGES>, M>) -> Self {
        Self {
            pages: space.try_into().unwrap_or_else(|_| {
                unreachable!("Converting a vector into an array of the same length always succeeds")
            }),
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<PagePermissionsLayout<PAGES>, F::Output> {
        self.pages
            .iter()
            .map(|item| item.struct_ref::<F>())
            .collect()
    }

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

impl<const PAGES: usize, M: ManagerBase> NewState<M> for PagePermissions<PAGES, M> {
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        PagePermissions {
            pages: boxed_from_fn(|| Cell::new()),
        }
    }
}

impl<const PAGES: usize, M: ManagerDeserialise> Decode<()> for PagePermissions<PAGES, M> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pages: AllocatedOf<PagePermissionsLayout<PAGES>, M> = Decode::decode(decoder)?;
        Ok(Self::bind(pages))
    }
}

impl<const PAGES: usize, M: ManagerSerialise> Encode for PagePermissions<PAGES, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.pages.encode(encoder)
    }
}
