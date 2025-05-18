// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Value types that the ICB can deal with, e.g. in memory load/store operations

use super::LoadStoreWidth;
use crate::jit::state_access::stack::Stackable;
use crate::machine_state::registers::XValue;
use crate::state_backend::Elem;

/// Types which can be loaded and stored using the [`super::ICB`]
pub trait StoreLoadInt: Stackable + Elem {
    /// The width of the value in memory
    const WIDTH: LoadStoreWidth;

    /// Whether the value is signed or unsigned
    const SIGNED: bool;

    /// Convert the value to an [`XValue`]. This will sign-extend or zero-extend the value.
    fn to_xvalue(self) -> XValue;

    /// Convert an [`XValue`] to the value type. This truncates the value to the width of the type.
    fn from_xvalue(xvalue: XValue) -> Self;
}

macro_rules! impl_store_load_int {
    ($width:literal, $variant:expr) => {
        paste::paste! {
            impl StoreLoadInt for [<u $width>] {
                const WIDTH: LoadStoreWidth = $variant;

                const SIGNED: bool = false;

                #[inline(always)]
                fn to_xvalue(self) -> XValue {
                    self as XValue
                }

                #[inline(always)]
                fn from_xvalue(xvalue: XValue) -> Self {
                    xvalue as Self
                }
            }

            impl StoreLoadInt for [<i $width>] {
                const WIDTH: LoadStoreWidth = $variant;

                const SIGNED: bool = true;

                #[inline(always)]
                fn to_xvalue(self) -> XValue {
                    self as XValue
                }

                #[inline(always)]
                fn from_xvalue(xvalue: XValue) -> Self {
                    xvalue as Self
                }
            }
        }
    };
}

impl_store_load_int!(8, LoadStoreWidth::Byte);
impl_store_load_int!(16, LoadStoreWidth::Half);
impl_store_load_int!(32, LoadStoreWidth::Word);
impl_store_load_int!(64, LoadStoreWidth::Double);
