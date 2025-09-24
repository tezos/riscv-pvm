// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Builder for turning [instructions] into functions.
//!
//! [instructions]: crate::machine_state::instruction::Instruction

pub(super) mod arithmetic;
pub(super) mod comparable;
mod control_flow_graph;
pub(super) mod errno;
pub(crate) mod ext_calls;
mod graph_walker;
mod instr_map;
pub(crate) mod instruction;
mod outcome_map;
pub(crate) mod sequence;
pub(crate) mod typed;

use cranelift::codegen::ir::condcodes::IntCC;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::MemFlags;
use cranelift::prelude::isa::TargetFrontendConfig;

use crate::instruction_context::Predicate;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
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
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    base: Pointer<MachineCoreState<MC, Owned>>,
    param: P::Parameter,
) -> Value<P::Target>
where
    MC: MemoryConfig,
    P: MachineCoreProjection,
    P::Target: typed::Typed,
{
    let (base, offset) = P::owned_pointer_offset::<MC>(param).build_base_and_offset(
        target_config,
        builder,
        base.to_value(),
    );

    // The `offset` when added to the final `base` pointer must result in a valid pointer to the
    // target value. We trust that both properties are upheld, hence we use `MemFlags::trusted()`.
    let val = builder.ins().load(
        <P::Target as typed::Typed>::TYPE.to_type(target_config),
        MemFlags::trusted(),
        base,
        offset,
    );

    // SAFETY: If the projection is correct, then it should resolve to a value of type `XValue`.
    unsafe { Value::<P::Target>::from_raw(val) }
}

/// Reusable implementation of [`crate::state_context::StateContext::write_proj`] for
/// the sequencer and instruction builder
fn write_proj<MC, P>(
    target_config: &TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    base: Pointer<MachineCoreState<MC, Owned>>,
    param: P::Parameter,
    value: Value<P::Target>,
) where
    MC: MemoryConfig,
    P: MachineCoreProjection,
{
    let (base, offset) = P::owned_pointer_offset::<MC>(param).build_base_and_offset(
        target_config,
        builder,
        base.to_value(),
    );

    // The `offset` when added to the final `base` pointer must result in a valid pointer to the
    // target value. We trust that both properties are upheld, hence we use `MemFlags::trusted()`.
    builder
        .ins()
        .store(MemFlags::trusted(), value.to_value(), base, offset);
}
