// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Value types that the ICB can deal with, e.g. in memory load/store operations

use cranelift::codegen::ir;
use cranelift::codegen::ir::types::I64;

use super::LoadStoreWidth;
use crate::instruction_context::ICB;
use crate::jit;
use crate::jit::state_access::JitStateAccess;
use crate::jit::state_access::stack::Stackable;
use crate::machine_state::memory::MemoryConfig;
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

/// PhiValue allows the conversion of values to and from cranelift primitive
/// [`ir::Value`] when in the context of [`JIT`] compilation. It represents a chosen
/// correct value from multiple control flows possible in `ICB::branch_merge`.
///
/// These methods have no relevance in the context of interpreted mode.
///
/// [`JIT`]: crate::jit::JIT
pub(crate) trait PhiValue {
    /// Represents the generic representation of the value in the ICB.
    type IcbValue<I: ICB + ?Sized>: Sized;

    /// In JIT, convert the value to an iterator of [`ir::Value`]s.
    fn to_ir_vals<MC: MemoryConfig, JSA: JitStateAccess>(
        icb_repr: Self::IcbValue<jit::builder::Builder<'_, MC, JSA>>,
    ) -> impl IntoIterator<Item = ir::Value>;

    /// Convert [`ir::Value`]s back to itself.
    fn from_ir_vals<'a, MC: MemoryConfig, JSA: JitStateAccess>(
        params: &[ir::Value],
    ) -> Self::IcbValue<jit::builder::Builder<'a, MC, JSA>>;

    /// The cranelift primitive types of the IR values representing this value in JIT.
    const IR_TYPES: &'static [ir::Type];
}

impl PhiValue for () {
    type IcbValue<I: ICB + ?Sized> = ();

    fn to_ir_vals<MC: MemoryConfig, JSA: JitStateAccess>(
        _: Self::IcbValue<jit::builder::Builder<'_, MC, JSA>>,
    ) -> impl IntoIterator<Item = ir::Value> {
        []
    }

    fn from_ir_vals<'a, MC: MemoryConfig, JSA: JitStateAccess>(
        _: &[ir::Value],
    ) -> Self::IcbValue<jit::builder::Builder<'a, MC, JSA>> {
    }

    const IR_TYPES: &'static [ir::Type] = &[];
}

impl PhiValue for XValue {
    type IcbValue<I: ICB + ?Sized> = I::XValue;

    fn to_ir_vals<MC: MemoryConfig, JSA: JitStateAccess>(
        icb_repr: Self::IcbValue<jit::builder::Builder<'_, MC, JSA>>,
    ) -> impl IntoIterator<Item = ir::Value> {
        [icb_repr.0]
    }

    fn from_ir_vals<'a, MC: MemoryConfig, JSA: JitStateAccess>(
        params: &[ir::Value],
    ) -> Self::IcbValue<jit::builder::Builder<'a, MC, JSA>> {
        jit::builder::X64(params[0])
    }

    const IR_TYPES: &'static [ir::Type] = &[I64];
}
