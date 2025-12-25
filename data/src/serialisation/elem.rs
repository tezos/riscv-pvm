// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Fast serialisation of primitive types
//!
//! This module provides the [`Elem`] trait for zero-overhead serialisation and deserialisation
//! of primitive types and fixed-size arrays. The design prioritises performance and simplicity
//! over flexibility and error handling.
//!
//! # Goals
//!
//! - **Memory-speed serialisation**: Operations are designed to be as fast as memory bandwidth allows
//! - **Infallible operations**: Serialisation and deserialisation cannot fail (given valid pointers)
//! - **Little-endian format**: All multi-byte values are stored in little-endian byte order for platform-independent compatibility
//! - **Unaligned access**: Support reading/writing from/to unaligned memory addresses
//!
//! # Safety
//!
//! The [`Elem`] trait uses unsafe methods because it operates directly on raw pointers for
//! maximum performance. Users of this trait must ensure:
//!
//! - Source pointers are valid for reads of [`Elem::STORED_SIZE`] bytes
//! - Destination pointers are valid for writes of [`Elem::STORED_SIZE`] bytes
//! - The memory regions do not overlap in unsafe ways

use std::num::NonZeroUsize;

/// Types that implement this trait can be loaded from or stored in a byte vector
///
/// This is a special kind of (de-)serialisation. Its goal is to be extremely fast, i.e. serialise
/// at memory speed. Additionally, it must not fail.
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

macro_rules! impl_elem_prim {
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
