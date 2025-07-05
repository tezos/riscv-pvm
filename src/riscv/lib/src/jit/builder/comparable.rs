// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of comparison operations in JIT mode.

use cranelift::codegen::ir::InstBuilder;
use cranelift::prelude::Value;

use crate::instruction_context::Predicate;
use crate::instruction_context::comparable::Comparable;
use crate::jit::builder::X32;
use crate::jit::builder::X64;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::machine_state::memory::MemoryConfig;

impl<MC: MemoryConfig> Comparable<InstructionBuilder<'_, '_, MC>> for X64 {
    type Result = Value;

    // icmp returns 1 if the condition holds, 0 if it does not.
    //
    // This matches the required semantics of bool - namely that it coerces to XValue with
    // - true => 1
    // - false => 0
    //
    // See
    // <https://docs.rs/cranelift-codegen/0.117.2/cranelift_codegen/ir/trait.InstBuilder.html#method.icmp>
    fn compare(
        self,
        other: Self,
        comparison: Predicate,
        builder: &mut InstructionBuilder<'_, '_, MC>,
    ) -> Value {
        builder.ins().icmp(comparison, self.0, other.0)
    }
}

impl<MC: MemoryConfig> Comparable<InstructionBuilder<'_, '_, MC>> for X32 {
    type Result = Value;

    // icmp returns 1 if the condition holds, 0 if it does not.
    //
    // This matches the required semantics of bool - namely that it coerces to XValue with
    // - true => 1
    // - false => 0
    //
    // See
    // <https://docs.rs/cranelift-codegen/0.117.2/cranelift_codegen/ir/trait.InstBuilder.html#method.icmp>
    fn compare(
        self,
        other: Self,
        comparison: Predicate,
        builder: &mut InstructionBuilder<'_, '_, MC>,
    ) -> Value {
        builder.ins().icmp(comparison, self.0, other.0)
    }
}
