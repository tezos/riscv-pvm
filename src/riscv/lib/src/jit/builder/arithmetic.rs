// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of arithmetic instructions in JIT mode.

use cranelift::codegen::ir::InstBuilder;

use crate::instruction_context::Shift;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::jit::builder::typed::Value;
use crate::machine_state::memory::MemoryConfig;

impl<T, MC: MemoryConfig> Arithmetic<InstructionBuilder<'_, '_, MC>> for Value<T> {
    fn add(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `iadd` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().iadd(lhs, rhs), other) }
    }

    fn sub(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `isub` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().isub(lhs, rhs), other) }
    }

    fn and(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `band` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().band(lhs, rhs), other) }
    }

    fn or(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `bor` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().bor(lhs, rhs), other) }
    }

    fn xor(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `bxor` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().bxor(lhs, rhs), other) }
    }

    fn mul(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `imul` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().imul(lhs, rhs), other) }
    }

    fn div_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `udiv` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().udiv(lhs, rhs), other) }
    }

    fn div_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `sdiv` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().sdiv(lhs, rhs), other) }
    }

    fn negate(self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `ineg` operation preserves the value type.
        unsafe { self.lift_unary(|value| builder.ins().ineg(value)) }
    }

    fn shift(
        self,
        shift: Shift,
        amount: Self,
        builder: &mut InstructionBuilder<'_, '_, MC>,
    ) -> Self {
        match shift {
            Shift::Left =>
            // SAFETY: `ishl` operation preserves the value type.
            unsafe { self.lift_binary(|lhs, rhs| builder.ins().ishl(lhs, rhs), amount) },

            Shift::RightUnsigned =>
            // SAFETY: `ushr` operation preserves the value type.
            unsafe { self.lift_binary(|lhs, rhs| builder.ins().ushr(lhs, rhs), amount) },

            Shift::RightSigned =>
            // SAFETY: `sshr` operation preserves the value type.
            unsafe { self.lift_binary(|lhs, rhs| builder.ins().sshr(lhs, rhs), amount) },
        }
    }

    fn modulus_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `urem` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().urem(lhs, rhs), other) }
    }

    fn modulus_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `srem` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().srem(lhs, rhs), other) }
    }

    fn min_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `smin` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().smin(lhs, rhs), other) }
    }

    fn min_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `umin` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().umin(lhs, rhs), other) }
    }

    fn max_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `smax` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().smax(lhs, rhs), other) }
    }

    fn max_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        // SAFETY: `umax` operation preserves the value type.
        unsafe { self.lift_binary(|lhs, rhs| builder.ins().umax(lhs, rhs), other) }
    }
}
