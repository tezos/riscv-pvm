// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Value types that the ICB can deal with, e.g. in memory load/store operations

use cranelift::codegen::ir;
use cranelift::codegen::ir::InstBuilder;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::types::I32;
use cranelift::prelude::types::I64;

use super::LoadStoreWidth;
use crate::instruction_context::ICB;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;
use crate::jit::state_access::stack::Stackable;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::XValue;
use crate::machine_state::registers::XValue32;
use crate::state_backend::Elem;

/// Types which can be loaded and stored using the [`super::ICB`]
pub trait StoreLoadInt: Typed + Stackable + Elem + 'static {
    /// The width of the value in memory
    const WIDTH: LoadStoreWidth;

    /// Whether the value is signed or unsigned
    const SIGNED: bool;

    /// Convert the value to an [`XValue`]. This will sign-extend or zero-extend the value.
    fn to_xvalue(self) -> XValue;

    /// Convert an [`XValue`] to the value type. This truncates the value to the width of the type.
    fn from_xvalue(xvalue: XValue) -> Self;

    /// Convert an IR value from the representation of `Self` to its `XValue` representation.
    fn to_xvalue_ir(builder: &mut FunctionBuilder, value: Value<Self>) -> Value<XValue>;
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

                fn to_xvalue_ir(builder: &mut FunctionBuilder, value: Value<Self>) -> Value<XValue> {
                    if $width == 64 {
                        // SAFETY: Both are 64-bit integers.
                        unsafe { value.cast() }
                    } else {
                        let value = value.to_value();
                        let value = builder.ins().uextend(I64, value);

                        // SAFETY: We extended it to the right width.
                        unsafe { Value::<XValue>::from_raw(value) }
                    }
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

                fn to_xvalue_ir(builder: &mut FunctionBuilder, value: Value<Self>) -> Value<XValue> {
                    if $width == 64 {
                        // SAFETY: Both are 64-bit integers.
                        unsafe { value.cast() }
                    } else {
                        let value = value.to_value();
                        let value = builder.ins().sextend(I64, value);

                        // SAFETY: We extended it to the right width.
                        unsafe { Value::<XValue>::from_raw(value) }
                    }
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
    fn to_ir_vals<MC: MemoryConfig>(
        icb_repr: Self::IcbValue<InstructionBuilder<'_, '_, MC>>,
    ) -> impl IntoIterator<Item = ir::Value>;

    /// Convert [`ir::Value`]s back to itself.
    fn from_ir_vals<'a, 'b, MC: MemoryConfig>(
        params: &[ir::Value],
        builder: &mut InstructionBuilder<'a, 'b, MC>,
    ) -> Self::IcbValue<InstructionBuilder<'a, 'b, MC>>;

    /// The cranelift primitive types of the IR values representing this value in JIT.
    const IR_TYPES: &'static [ir::Type];
}

impl PhiValue for () {
    type IcbValue<I: ICB + ?Sized> = ();

    fn to_ir_vals<MC: MemoryConfig>(
        _: Self::IcbValue<InstructionBuilder<'_, '_, MC>>,
    ) -> impl IntoIterator<Item = ir::Value> {
        []
    }

    fn from_ir_vals<'a, 'b, MC: MemoryConfig>(
        _: &[ir::Value],
        _: &mut InstructionBuilder<'a, 'b, MC>,
    ) -> Self::IcbValue<InstructionBuilder<'a, 'b, MC>> {
    }

    const IR_TYPES: &'static [ir::Type] = &[];
}

impl PhiValue for XValue {
    type IcbValue<I: ICB + ?Sized> = I::XValue;

    fn to_ir_vals<MC: MemoryConfig>(
        icb_repr: Self::IcbValue<InstructionBuilder<'_, '_, MC>>,
    ) -> impl IntoIterator<Item = ir::Value> {
        [icb_repr.to_value()]
    }

    fn from_ir_vals<'a, 'b, MC: MemoryConfig>(
        params: &[ir::Value],
        _: &mut InstructionBuilder<'a, 'b, MC>,
    ) -> Self::IcbValue<InstructionBuilder<'a, 'b, MC>> {
        // SAFETY: We know the value is an `XValue` as per [`Self::to_ir_vals`].
        unsafe { Value::<XValue>::from_raw(params[0]) }
    }

    const IR_TYPES: &'static [ir::Type] = &[I64];
}

impl PhiValue for XValue32 {
    type IcbValue<I: ICB + ?Sized> = I::XValue32;

    fn to_ir_vals<MC: MemoryConfig>(
        icb_repr: Self::IcbValue<InstructionBuilder<'_, '_, MC>>,
    ) -> impl IntoIterator<Item = ir::Value> {
        [icb_repr.to_value()]
    }

    fn from_ir_vals<'a, 'b, MC: MemoryConfig>(
        params: &[ir::Value],
        _: &mut InstructionBuilder<'a, 'b, MC>,
    ) -> Self::IcbValue<InstructionBuilder<'a, 'b, MC>> {
        // SAFETY: We know the value is an `XValue32` as per [`Self::to_ir_vals`].
        unsafe { Value::<XValue32>::from_raw(params[0]) }
    }

    const IR_TYPES: &'static [ir::Type] = &[I32];
}

impl<E> PhiValue for Result<(), E> {
    type IcbValue<I: ICB + ?Sized> = I::IResult<()>;

    /// For handling an `IResult` output from a branch merge, we are only catering
    /// for the `Ok` case. `Err` is handled within the block, whilst we are continuing
    /// building IR for the `Ok` case.
    fn to_ir_vals<MC: MemoryConfig>(
        _: Self::IcbValue<InstructionBuilder<'_, '_, MC>>,
    ) -> impl IntoIterator<Item = ir::Value> {
        []
    }

    fn from_ir_vals<'a, 'b, MC: MemoryConfig>(
        _: &[ir::Value],
        icb: &mut InstructionBuilder<'a, 'b, MC>,
    ) -> Self::IcbValue<InstructionBuilder<'a, 'b, MC>> {
        icb.ok(())
    }

    const IR_TYPES: &'static [ir::Type] = &[];
}
