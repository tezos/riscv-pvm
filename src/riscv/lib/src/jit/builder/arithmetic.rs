// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of arithmetic instructions in JIT mode.

use cranelift::codegen::ir::InstBuilder;

use super::X32;
use super::X64;
use crate::instruction_context::Shift;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::machine_state::memory::MemoryConfig;

impl<MC: MemoryConfig> Arithmetic<InstructionBuilder<'_, '_, MC>> for X64 {
    fn add(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().iadd(self.0, other.0);
        X64(res)
    }

    fn sub(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().isub(self.0, other.0);
        X64(res)
    }

    fn and(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().band(self.0, other.0);
        X64(res)
    }

    fn or(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().bor(self.0, other.0);
        X64(res)
    }

    fn xor(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().bxor(self.0, other.0);
        X64(res)
    }

    fn mul(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().imul(self.0, other.0);
        X64(res)
    }

    fn div_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().udiv(self.0, other.0);
        X64(res)
    }

    fn div_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().sdiv(self.0, other.0);
        X64(res)
    }

    fn negate(self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().ineg(self.0))
    }

    fn shift(
        self,
        shift: Shift,
        amount: Self,
        builder: &mut InstructionBuilder<'_, '_, MC>,
    ) -> Self {
        match shift {
            Shift::Left => X64(builder.ins().ishl(self.0, amount.0)),
            Shift::RightUnsigned => X64(builder.ins().ushr(self.0, amount.0)),
            Shift::RightSigned => X64(builder.ins().sshr(self.0, amount.0)),
        }
    }

    fn modulus_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().urem(self.0, other.0))
    }

    fn modulus_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().srem(self.0, other.0))
    }

    fn min_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().smin(self.0, other.0))
    }

    fn min_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().umin(self.0, other.0))
    }

    fn max_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().smax(self.0, other.0))
    }

    fn max_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X64(builder.ins().umax(self.0, other.0))
    }
}

impl<MC: MemoryConfig> Arithmetic<InstructionBuilder<'_, '_, MC>> for X32 {
    fn add(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().iadd(self.0, other.0);
        X32(res)
    }

    fn sub(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().isub(self.0, other.0);
        X32(res)
    }

    fn and(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().band(self.0, other.0);
        X32(res)
    }

    fn or(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().bor(self.0, other.0);
        X32(res)
    }

    fn xor(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().bxor(self.0, other.0);
        X32(res)
    }

    fn mul(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().imul(self.0, other.0);
        X32(res)
    }

    fn div_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().udiv(self.0, other.0);
        X32(res)
    }

    fn div_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        let res = builder.ins().sdiv(self.0, other.0);
        X32(res)
    }

    fn negate(self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().ineg(self.0))
    }

    fn shift(
        self,
        shift: Shift,
        amount: Self,
        builder: &mut InstructionBuilder<'_, '_, MC>,
    ) -> Self {
        match shift {
            Shift::Left => X32(builder.ins().ishl(self.0, amount.0)),
            Shift::RightUnsigned => X32(builder.ins().ushr(self.0, amount.0)),
            Shift::RightSigned => X32(builder.ins().sshr(self.0, amount.0)),
        }
    }

    fn modulus_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().urem(self.0, other.0))
    }

    fn modulus_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().srem(self.0, other.0))
    }

    fn min_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().smin(self.0, other.0))
    }

    fn min_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().umin(self.0, other.0))
    }

    fn max_signed(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().smax(self.0, other.0))
    }

    fn max_unsigned(self, other: Self, builder: &mut InstructionBuilder<'_, '_, MC>) -> Self {
        X32(builder.ins().umax(self.0, other.0))
    }
}
