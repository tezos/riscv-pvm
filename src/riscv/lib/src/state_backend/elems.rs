// SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use crate::machine_state::memory::PAGE_SIZE;

/// Types that are less than one page wide
pub trait NarrowlySized: Elem {
    /// Size of the type
    const NARROW_SIZE: NonZeroUsize = {
        if Self::STORED_SIZE.get() >= PAGE_SIZE.get() as usize {
            panic!("Type is too wide");
        }

        Self::STORED_SIZE
    };
}

impl<T: Elem> NarrowlySized for T {}

/// Types that can be copied and contain no non-static references
pub trait StaticCopy: Copy + 'static {}

impl<T: Copy + 'static> StaticCopy for T {}

/// Values that can be stored in dynamic regions
pub trait Elem {
    /// Size of the stored representation in bytes
    const STORED_SIZE: NonZeroUsize;

    /// Read a value from its stored representation.
    ///
    /// # Safety
    ///
    /// You must ensure that the source is valid for reads of `Self::STORED_SIZE` bytes.
    unsafe fn read_unaligned(source: *const u8) -> Self;

    /// Write a value as its stored representation.
    ///
    /// # Safety
    ///
    /// You must ensure that the destination is valid for writes of `Self::STORED_SIZE` bytes.
    unsafe fn write_unaligned(self, dest: *mut u8);
}

/// Capture the stored representation of an element from a dynamic region.
pub fn elem_bytes<E: Elem>(value: E) -> Box<[u8]> {
    let mut value_bytes = vec![0u8; E::STORED_SIZE.get()];

    // SAFETY: The vector has been allocated with sufficient space.
    unsafe {
        value.write_unaligned(value_bytes.as_mut_ptr());
    }

    value_bytes.into_boxed_slice()
}

macro_rules! impl_dyn_value_prim {
    ( $x:ty ) => {
        impl Elem for $x {
            const STORED_SIZE: NonZeroUsize =
                NonZeroUsize::new(std::mem::size_of::<$x>()).expect("Type has zero size");

            #[inline]
            unsafe fn read_unaligned(source: *const u8) -> Self {
                Self::from_le(unsafe { source.cast::<Self>().read_unaligned() })
            }

            #[inline]
            unsafe fn write_unaligned(self, dest: *mut u8) {
                unsafe { dest.cast::<Self>().write_unaligned(self.to_le()) }
            }
        }
    };
}

impl_dyn_value_prim!(u8);
impl_dyn_value_prim!(i8);
impl_dyn_value_prim!(u16);
impl_dyn_value_prim!(i16);
impl_dyn_value_prim!(u32);
impl_dyn_value_prim!(i32);
impl_dyn_value_prim!(u64);
impl_dyn_value_prim!(i64);
impl_dyn_value_prim!(u128);
impl_dyn_value_prim!(i128);

impl<E: Elem, const LEN: usize> Elem for [E; LEN] {
    const STORED_SIZE: NonZeroUsize = {
        let len = NonZeroUsize::new(LEN).expect("Array length must be non-zero");
        E::STORED_SIZE
            .checked_mul(len)
            .expect("Array size must not overflow")
    };

    unsafe fn read_unaligned(source: *const u8) -> Self {
        std::array::from_fn(|i| {
            let offset = E::STORED_SIZE.get().wrapping_mul(i);
            unsafe { E::read_unaligned(source.add(offset)) }
        })
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        for (i, elem) in self.into_iter().enumerate() {
            let offset = E::STORED_SIZE.get().wrapping_mul(i);
            unsafe { elem.write_unaligned(dest.add(offset)) };
        }
    }
}
