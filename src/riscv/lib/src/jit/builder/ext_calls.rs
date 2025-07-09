// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Framework for calling external functions from Cranelift IR

use cranelift::codegen::ir;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::Signature;
use cranelift::prelude::isa::TargetFrontendConfig;

use crate::jit::builder::typed::Type;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;

/// Trait for types that can be returned from external function calls
pub trait ReturnTyped {
    /// IR types of the function return values
    const RETURN_TYPES: &[Type];

    /// IR [`self::Value`] representation of the return values
    type Value;

    /// Construct the IR [`self::Value`] from the return values.
    fn from_return_values(values: &[ir::Value]) -> Self::Value;
}

impl<T: Typed> ReturnTyped for T {
    const RETURN_TYPES: &[Type] = &[T::TYPE];

    type Value = Value<T>;

    fn from_return_values(values: &[ir::Value]) -> Value<T> {
        // SAFETY: We declare the type of the return value using `Self::RETURN_TYPES`. The type of
        // the `ir::Value` aligns with this.
        unsafe { Value::from_raw(values[0]) }
    }
}

impl ReturnTyped for () {
    const RETURN_TYPES: &[Type] = &[];

    type Value = ();

    fn from_return_values(_values: &[ir::Value]) {}
}

/// Trait for tuples that represent function arguments
trait ArgumentsTyped: tuples::Tuple {
    /// Function argument types
    const ARGS_TYPES: &[Type];
}

impl ArgumentsTyped for () {
    const ARGS_TYPES: &[Type] = &[];
}

impl<A0: Typed> ArgumentsTyped for (A0,) {
    const ARGS_TYPES: &[Type] = &[A0::TYPE];
}

impl<A0: Typed, A1: Typed> ArgumentsTyped for (A0, A1) {
    const ARGS_TYPES: &[Type] = &[A0::TYPE, A1::TYPE];
}

impl<A0: Typed, A1: Typed, A2: Typed> ArgumentsTyped for (A0, A1, A2) {
    const ARGS_TYPES: &[Type] = &[A0::TYPE, A1::TYPE, A2::TYPE];
}

impl<A0: Typed, A1: Typed, A2: Typed, A3: Typed> ArgumentsTyped for (A0, A1, A2, A3) {
    const ARGS_TYPES: &[Type] = &[A0::TYPE, A1::TYPE, A2::TYPE, A3::TYPE];
}

impl<A0: Typed, A1: Typed, A2: Typed, A3: Typed, A4: Typed> ArgumentsTyped
    for (A0, A1, A2, A3, A4)
{
    const ARGS_TYPES: &[Type] = &[A0::TYPE, A1::TYPE, A2::TYPE, A3::TYPE, A4::TYPE];
}

/// Construct a Cranelift function signature for the given argument and return types.
fn make_signature<A: ArgumentsTyped, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
) -> Signature {
    let params = A::ARGS_TYPES
        .iter()
        .map(|typ| ir::AbiParam::new(typ.to_type(target_config)))
        .collect();

    let returns = R::RETURN_TYPES
        .iter()
        .map(|typ| ir::AbiParam::new(typ.to_type(target_config)))
        .collect();

    Signature {
        params,
        returns,
        call_conv: target_config.default_call_conv,
    }
}

/// This is a helper function to call an external function with the given arguments. It builds the
/// base for dispatching external function calls and is configured by the real `callN` variants.
fn call_raw<A: ArgumentsTyped, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    callee: usize,
    args: &[ir::Value],
) -> R::Value {
    let sig = make_signature::<A, R>(target_config);
    let sig_ref = builder.import_signature(sig);

    let callee = builder
        .ins()
        .iconst(target_config.pointer_type(), callee as i64);

    let inst = builder.ins().call_indirect(sig_ref, callee, args);
    let ret_values = builder.inst_results(inst);

    R::from_return_values(ret_values)
}

/// Call an external function with 1 argument.
pub fn call1<A0: Typed, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    callee: extern "C" fn(A0) -> R,
    arg0: Value<A0>,
) -> R::Value {
    call_raw::<(A0,), R>(target_config, builder, callee as usize, &[arg0.to_value()])
}

/// Call an external function with 2 arguments.
pub fn call2<A0: Typed, A1: Typed, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    callee: extern "C" fn(A0, A1) -> R,
    arg0: Value<A0>,
    arg1: Value<A1>,
) -> R::Value {
    call_raw::<(A0, A1), R>(target_config, builder, callee as usize, &[
        arg0.to_value(),
        arg1.to_value(),
    ])
}

pub fn call3<A0: Typed, A1: Typed, A2: Typed, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    callee: extern "C" fn(A0, A1, A2) -> R,
    arg0: Value<A0>,
    arg1: Value<A1>,
    arg2: Value<A2>,
) -> R::Value {
    call_raw::<(A0, A1, A2), R>(target_config, builder, callee as usize, &[
        arg0.to_value(),
        arg1.to_value(),
        arg2.to_value(),
    ])
}

/// Call an external function with 4 arguments.
pub fn call4<A0: Typed, A1: Typed, A2: Typed, A3: Typed, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    callee: extern "C" fn(A0, A1, A2, A3) -> R,
    arg0: Value<A0>,
    arg1: Value<A1>,
    arg2: Value<A2>,
    arg3: Value<A3>,
) -> R::Value {
    call_raw::<(A0, A1, A2, A3), R>(target_config, builder, callee as usize, &[
        arg0.to_value(),
        arg1.to_value(),
        arg2.to_value(),
        arg3.to_value(),
    ])
}
