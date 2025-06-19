// SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use crate::jit::builder::arithmetic::Alignment;
use crate::machine_state::memory::PAGE_SIZE;

/// Types that have a non-zero size
pub trait NonZeroSized: Sized {
    /// Size of the type
    const NON_ZERO_SIZE: NonZeroUsize = {
        let size = NonZeroUsize::new(std::mem::size_of::<Self>());
        if let Some(size) = size {
            size
        } else {
            panic!("Type has zero size");
        }
    };
}

impl<T: Sized> NonZeroSized for T {}

/// Types that are less than one page wide
pub trait NarrowlySized: NonZeroSized {
    /// Size of the type
    const NARROW_SIZE: NonZeroUsize = {
        if Self::NON_ZERO_SIZE.get() >= PAGE_SIZE.get() as usize {
            panic!("Type is too wide");
        }

        Self::NON_ZERO_SIZE
    };
}

impl<T: NonZeroSized> NarrowlySized for T {}

/// Types that can be copied and contain no non-static references
pub trait StaticCopy: Copy + 'static {}

impl<T: Copy + 'static> StaticCopy for T {}

/// Elements that may be stored using a Backend - i.e. implementors of [super::ManagerBase]
pub trait Elem: StaticCopy {
    const KNOWN_ALIGNMENT: Alignment = Alignment::One;

    /// Copy from `source` and convert to stored representation.
    fn store(&mut self, source: &Self);

    /// Convert to stored representation in place.
    fn to_stored_in_place(&mut self);

    /// Convert from stored representation in place.
    // The naming of this function trips Clippy.
    #[expect(
        clippy::wrong_self_convention,
        reason = "It's an appropriate name when ignoring Rust's from/to naming convention"
    )]
    fn from_stored_in_place(&mut self);

    /// Read a value from its stored representation.
    fn from_stored(source: &Self) -> Self;
}

/// `Elem`s which are known to have 8-byte alignment
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct AlignedElem<E: Elem>(pub E);

impl<E: Elem> Elem for AlignedElem<E> {
    const KNOWN_ALIGNMENT: Alignment = Alignment::Eight;

    #[inline(always)]
    fn store(&mut self, source: &Self) {
        self.0.store(&source.0)
    }

    #[inline(always)]
    fn to_stored_in_place(&mut self) {
        self.0.to_stored_in_place();
    }

    #[inline(always)]
    fn from_stored_in_place(&mut self) {
        self.0.from_stored_in_place();
    }

    #[inline(always)]
    fn from_stored(source: &Self) -> Self {
        AlignedElem(E::from_stored(&source.0))
    }
}

macro_rules! impl_elem_prim {
    ( $x:ty ) => {
        impl Elem for $x {
            const KNOWN_ALIGNMENT: Alignment = Alignment::One;

            #[inline(always)]
            fn store(&mut self, source: &Self) {
                *self = source.to_le();
            }

            #[inline(always)]
            fn to_stored_in_place(&mut self) {
                *self = self.to_le();
            }

            #[inline(always)]
            fn from_stored_in_place(&mut self) {
                *self = Self::from_le(*self);
            }

            #[inline(always)]
            fn from_stored(source: &Self) -> Self {
                Self::from_le(*source)
            }
        }
    };
}

impl_elem_prim!(u8);
impl_elem_prim!(i8);
impl_elem_prim!(u16);
impl_elem_prim!(i16);
impl_elem_prim!(u32);
impl_elem_prim!(i32);
impl_elem_prim!(u64);
impl_elem_prim!(i64);
impl_elem_prim!(u128);
impl_elem_prim!(i128);

impl<E: Elem, const LEN: usize> Elem for [E; LEN] {
    #[inline(always)]
    fn store(&mut self, source: &Self) {
        self.copy_from_slice(source);

        // NOTE: This loop may be eliminated if [to_stored_in_place] is a no-op.
        for elem in self {
            elem.to_stored_in_place();
        }
    }

    #[inline(always)]
    fn to_stored_in_place(&mut self) {
        // NOTE: This loop may be eliminated if [to_stored_in_place] is a no-op.
        for elem in self {
            elem.to_stored_in_place();
        }
    }

    #[inline(always)]
    fn from_stored_in_place(&mut self) {
        // NOTE: This loop may be eliminated if [from_stored_in_place] is a no-op.
        for elem in self {
            elem.from_stored_in_place();
        }
    }

    #[inline(always)]
    fn from_stored(source: &Self) -> Self {
        let mut new = *source;

        // NOTE: This loop may be eliminated if [from_stored_in_place] is a no-op.
        for elem in new.iter_mut() {
            elem.from_stored_in_place();
        }

        new
    }
}
