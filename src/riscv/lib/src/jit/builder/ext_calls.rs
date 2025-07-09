// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use cranelift::codegen::ir;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::Signature;
use cranelift::prelude::isa::TargetFrontendConfig;

use crate::jit::builder::typed::Type;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;

pub trait ReturnTyped {
    const RETURN_TYPES: &[Type];

    type Value;

    fn from_return_values(values: &[ir::Value]) -> Self::Value;
}

impl<T: Typed> ReturnTyped for T {
    const RETURN_TYPES: &[Type] = &[T::TYPE];

    type Value = Value<T>;

    fn from_return_values(values: &[ir::Value]) -> Value<T> {
        unsafe { Value::from_raw(values[0]) }
    }
}

impl ReturnTyped for () {
    const RETURN_TYPES: &[Type] = &[];

    type Value = ();

    fn from_return_values(_values: &[ir::Value]) {}
}

trait ArgumentsTyped {
    const PARAM_TYPES: &[Type];
}

impl ArgumentsTyped for () {
    const PARAM_TYPES: &[Type] = &[];
}

impl<A0: Typed> ArgumentsTyped for (A0,) {
    const PARAM_TYPES: &[Type] = &[A0::TYPE];
}

impl<A0: Typed, A1: Typed> ArgumentsTyped for (A0, A1) {
    const PARAM_TYPES: &[Type] = &[A0::TYPE, A1::TYPE];
}

impl<A0: Typed, A1: Typed, A2: Typed> ArgumentsTyped for (A0, A1, A2) {
    const PARAM_TYPES: &[Type] = &[A0::TYPE, A1::TYPE, A2::TYPE];
}

impl<A0: Typed, A1: Typed, A2: Typed, A3: Typed> ArgumentsTyped for (A0, A1, A2, A3) {
    const PARAM_TYPES: &[Type] = &[A0::TYPE, A1::TYPE, A2::TYPE, A3::TYPE];
}

impl<A0: Typed, A1: Typed, A2: Typed, A3: Typed, A4: Typed> ArgumentsTyped
    for (A0, A1, A2, A3, A4)
{
    const PARAM_TYPES: &[Type] = &[A0::TYPE, A1::TYPE, A2::TYPE, A3::TYPE, A4::TYPE];
}

fn make_signature<A: ArgumentsTyped, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
) -> Signature {
    let params = A::PARAM_TYPES
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

pub fn call1<A0: Typed, R: ReturnTyped>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    callee: extern "C" fn(A0) -> R,
    arg0: Value<A0>,
) -> R::Value {
    call_raw::<(A0,), R>(target_config, builder, callee as usize, &[arg0.to_value()])
}

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
