// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Builder for turning [instructions] into functions.
//!
//! [instructions]: crate::machine_state::instruction::Instruction

pub(super) mod arithmetic;
pub(super) mod comparable;
pub(super) mod errno;
pub(crate) mod ext_calls;
pub(crate) mod instruction;
pub(crate) mod sequence;
pub(crate) mod typed;

use cranelift::codegen::ir::condcodes::IntCC;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::MemFlags;
use cranelift::prelude::types::I64;

use crate::instruction_context::Predicate;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::XValue;
use crate::state_backend::owned_backend::Owned;
use crate::state_context::projection::MachineCoreProjection;

impl From<Predicate> for IntCC {
    fn from(value: Predicate) -> Self {
        match value {
            Predicate::Equal => IntCC::Equal,
            Predicate::NotEqual => IntCC::NotEqual,
            Predicate::LessThanSigned => IntCC::SignedLessThan,
            Predicate::LessThanUnsigned => IntCC::UnsignedLessThan,
            Predicate::LessThanOrEqualSigned => IntCC::SignedLessThanOrEqual,
            Predicate::GreaterThanSigned => IntCC::SignedGreaterThan,
            Predicate::GreaterThanOrEqualSigned => IntCC::SignedGreaterThanOrEqual,
            Predicate::GreaterThanOrEqualUnsigned => IntCC::UnsignedGreaterThanOrEqual,
        }
    }
}

/// Reusable implementation of [`crate::state_context::StateContext::read_proj`] for
/// the sequencer and instruction builder
fn read_proj<MC, P>(
    builder: &mut FunctionBuilder,
    core_param: Pointer<MachineCoreState<MC, Owned>>,
    param: P::Parameter,
) -> Value<XValue>
where
    MC: MemoryConfig,
    P: MachineCoreProjection<Target = u64>,
{
    let offset = P::owned_pointer_offset::<MC>(param);

    // The JIT-compiled function requires that the `core` parameter is a valid pointer to the
    // `MachineCoreState`. Additionally, the offset produced by `P::owned_pointer_offset` must
    // result in a valid pointer when applied to `core`. We trust that both properties are upheld,
    // hence we use `MemFlags::trusted()`.
    let val = builder
        .ins()
        .load(I64, MemFlags::trusted(), core_param.to_value(), offset);

    // SAFETY: If the projection is correct, then it should resolve to a value of type `XValue`.
    unsafe { Value::<XValue>::from_raw(val) }
}

/// Reusable implementation of [`crate::state_context::StateContext::write_proj`] for
/// the sequencer and instruction builder
fn write_proj<MC, P>(
    builder: &mut FunctionBuilder,
    core_param: Pointer<MachineCoreState<MC, Owned>>,
    param: P::Parameter,
    value: Value<XValue>,
) where
    MC: MemoryConfig,
    P: MachineCoreProjection<Target = u64>,
{
    let offset = P::owned_pointer_offset::<MC>(param);

    // The JIT-compiled function requires that the `core` parameter is a valid pointer to the
    // `MachineCoreState`. Additionally, the offset produced by `P::owned_pointer_offset` must
    // result in a valid pointer when applied to `core`. We trust that both properties are upheld,
    // hence we use `MemFlags::trusted()`.
    builder.ins().store(
        MemFlags::trusted(),
        value.to_value(),
        core_param.to_value(),
        offset,
    );
}
