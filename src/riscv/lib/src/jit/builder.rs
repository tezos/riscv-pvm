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

use crate::instruction_context::Predicate;

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
