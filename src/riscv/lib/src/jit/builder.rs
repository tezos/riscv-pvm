// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Builder for turning [instructions] into functions.
//!
//! [instructions]: crate::machine_state::instruction::Instruction

pub(super) mod arithmetic;
pub(super) mod comparable;
pub(super) mod errno;
pub(crate) mod instruction;
pub(crate) mod sequence;

use cranelift::codegen::ir::Value;
use cranelift::codegen::ir::condcodes::IntCC;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::MemFlags;
use cranelift::prelude::types::I64;

use crate::instruction_context::Predicate;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::owned_backend::Owned;
use crate::state_context::projection::MachineCoreProjection;
use crate::state_context::projection::RegionCons;

/// A newtype for wrapping [`Value`], representing a 64-bit value in the JIT context.
#[derive(Copy, Clone, Debug)]
pub struct X64(pub Value);

/// A newtype for wrapping [`Value`], representing a 32-bit value in the JIT context.
#[derive(Copy, Clone, Debug)]
pub struct X32(pub Value);

/// A newtype for wrapping [`Value`], representing a 64-bit floating-point value in the JIT context.
#[derive(Copy, Clone, Debug)]
pub struct F64(pub Value);

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

/// Reusable implementation of [`crate::state_context::StateContext::read_machine_region`] for
/// the sequencer and instruction builder
fn read_machine_region<MC, L, const LEN: usize>(
    builder: &mut FunctionBuilder,
    core_param: Value,
    index: usize,
) -> X64
where
    MC: MemoryConfig,
    L: MachineCoreProjection<Target = RegionCons<u64, LEN>>,
{
    assert!(index < LEN);

    let region_offset = L::pointer_offset::<MC, Owned>();
    let offset = 8 * index + region_offset;

    let val = builder
        .ins()
        .load(I64, MemFlags::trusted(), core_param, offset as i32);
    X64(val)
}

/// Reusable implementation of [`crate::state_context::StateContext::write_machine_region`] for
/// the sequencer and instruction builder
fn write_machine_region<MC, L, const LEN: usize>(
    builder: &mut FunctionBuilder,
    core_param: Value,
    index: usize,
    value: X64,
) where
    MC: MemoryConfig,
    L: MachineCoreProjection<Target = RegionCons<u64, LEN>>,
{
    assert!(index < LEN);

    let region_offset = L::pointer_offset::<MC, Owned>();
    let offset = 8 * index + region_offset;

    builder
        .ins()
        .store(MemFlags::trusted(), value.0, core_param, offset as i32);
}
