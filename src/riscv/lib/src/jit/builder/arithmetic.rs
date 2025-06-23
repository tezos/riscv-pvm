// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of arithmetic instructions in JIT mode.

use std::ops::Add;
use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::BitXor;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Shl;
use std::ops::Shr;
use std::ops::Sub;

use cranelift::codegen::ir::InstBuilder;
use cranelift::codegen::ir::Value;

use super::Builder;
use super::X32;
use super::X64;
use crate::instruction_context::Shift;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::jit::state_access::JitStateAccess;
use crate::machine_state::memory::MemoryConfig;

/// Known alignments of addresses which have values that aren't known at runtime. The arithmetic is
/// counter-intuitively non-associative - see the tests for examples.
#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Alignment {
    #[default]
    One,
    Two,
    Four,
    Eight,
}

impl From<i64> for Alignment {
    fn from(address: i64) -> Self {
        match address as isize % align_of::<i64>() as isize {
            -6 => Alignment::Two,
            -4 => Alignment::Four,
            -2 => Alignment::Two,
            0 => Alignment::Eight,
            2 => Alignment::Two,
            4 => Alignment::Four,
            6 => Alignment::Two,
            _ => Alignment::One,
        }
    }
}

impl From<u64> for Alignment {
    fn from(address: u64) -> Self {
        match address as usize % align_of::<u64>() {
            0 => Alignment::Eight,
            2 => Alignment::Two,
            4 => Alignment::Four,
            6 => Alignment::Two,
            _ => Alignment::One,
        }
    }
}

impl Add for Alignment {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match self {
            Alignment::One => Alignment::One,
            Alignment::Two => match rhs {
                Alignment::One => Alignment::One,
                _ => Alignment::Two,
            },
            Alignment::Four => match rhs {
                Alignment::One => Alignment::One,
                Alignment::Two => Alignment::Two,
                _ => Alignment::Four,
            },
            Alignment::Eight => match rhs {
                Alignment::One => Alignment::One,
                Alignment::Two => Alignment::Two,
                Alignment::Four => Alignment::Four,
                Alignment::Eight => Alignment::Eight,
            },
        }
    }
}

impl BitAnd for Alignment {
    type Output = Self;

    fn bitand(self, _rhs: Self) -> Self {
        Alignment::Eight
    }
}

impl BitOr for Alignment {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        std::cmp::min(self, rhs)
    }
}

impl BitXor for Alignment {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        std::cmp::min(self, rhs)
    }
}

impl Div for Alignment {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        std::cmp::min(self, rhs)
    }
}

impl Mul for Alignment {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        std::cmp::max(self, rhs)
    }
}

impl Neg for Alignment {
    type Output = Self;

    fn neg(self) -> Self {
        self
    }
}

impl Rem for Alignment {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        std::cmp::min(self, rhs)
    }
}

impl Shl<Value> for Alignment {
    type Output = Self;

    // The SSA values are opaque, so assume the worst
    fn shl(self, _rhs: Value) -> Self {
        Alignment::One
    }
}

impl Shr<Value> for Alignment {
    type Output = Self;

    // The SSA values are opaque, so assume the worst
    fn shr(self, _rhs: Value) -> Self {
        Alignment::One
    }
}

impl Sub for Alignment {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
}

impl<MC: MemoryConfig, JSA: JitStateAccess> Arithmetic<Builder<'_, MC, JSA>> for X64 {
    fn add(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().iadd(self.0, other.0),
        )
    }

    fn sub(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().isub(self.0, other.0),
        )
    }

    fn and(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().band(self.0, other.0),
        )
    }

    fn or(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().bor(self.0, other.0),
        )
    }

    fn xor(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().bxor(self.0, other.0),
        )
    }

    fn mul(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().imul(self.0, other.0),
        )
    }

    fn div_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().udiv(self.0, other.0),
        )
    }

    fn div_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().sdiv(self.0, other.0),
        )
    }

    fn negate(self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(icb.builder.ins().ineg(self.0))
    }

    fn shift(self, shift: Shift, amount: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        match shift {
            Shift::Left => X64(
                icb.builder.ins().ishl(self.0, amount.0),
            ),
            Shift::RightUnsigned => X64(
                icb.builder.ins().ushr(self.0, amount.0),
            ),
            Shift::RightSigned => X64(
                icb.builder.ins().sshr(self.0, amount.0),
            ),
        }
    }

    fn modulus_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().urem(self.0, other.0),
        )
    }

    fn modulus_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().srem(self.0, other.0),
        )
    }

    fn min_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().smin(self.0, other.0),
        )
    }

    fn min_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().umin(self.0, other.0),
        )
    }

    fn max_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().smax(self.0, other.0),
        )
    }

    fn max_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X64(
            icb.builder.ins().umax(self.0, other.0),
        )
    }
}

impl<MC: MemoryConfig, JSA: JitStateAccess> Arithmetic<Builder<'_, MC, JSA>> for X32 {
    fn add(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().iadd(self.0, other.0);
        X32(res)
    }

    fn sub(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().isub(self.0, other.0);
        X32(res)
    }

    fn and(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().band(self.0, other.0);
        X32(res)
    }

    fn or(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().bor(self.0, other.0);
        X32(res)
    }

    fn xor(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().bxor(self.0, other.0);
        X32(res)
    }

    fn mul(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().imul(self.0, other.0);
        X32(res)
    }

    fn div_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().udiv(self.0, other.0);
        X32(res)
    }

    fn div_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        let res = icb.builder.ins().sdiv(self.0, other.0);
        X32(res)
    }

    fn negate(self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().ineg(self.0))
    }

    fn shift(self, shift: Shift, amount: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        match shift {
            Shift::Left => X32(icb.builder.ins().ishl(self.0, amount.0)),
            Shift::RightUnsigned => X32(icb.builder.ins().ushr(self.0, amount.0)),
            Shift::RightSigned => X32(icb.builder.ins().sshr(self.0, amount.0)),
        }
    }

    fn modulus_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().urem(self.0, other.0))
    }

    fn modulus_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().srem(self.0, other.0))
    }

    fn min_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().smin(self.0, other.0))
    }

    fn min_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().umin(self.0, other.0))
    }

    fn max_signed(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().smax(self.0, other.0))
    }

    fn max_unsigned(self, other: Self, icb: &mut Builder<'_, MC, JSA>) -> Self {
        X32(icb.builder.ins().umax(self.0, other.0))
    }
}

#[cfg(test)]
mod tests {
    use cranelift::codegen::ir::Value;

    use crate::jit::builder::arithmetic::Alignment;

    #[test]
    fn test_alignment_add() {
        let zero: Alignment = 0u64.into();
        let one = 1u64.into();

        assert_eq!(zero, Alignment::Eight);
        assert_eq!(one, Alignment::One);
        assert_eq!(zero + one, Alignment::One);
        assert_eq!(one + one, Alignment::One);

        let two = 2u64.into();
        assert!(one + one != two);

        let three: Alignment = 3u64.into();
        assert_eq!(three, Alignment::One);
        assert!(Alignment::One + one != Alignment::Two);
        assert!(three + one != Alignment::Two);
        assert!(three + one != Alignment::Four);
        assert_eq!(three + one, Alignment::One);
        let four: Alignment = (3u64 + 1u64).into();
        assert_eq!(four, Alignment::Four);
    }

    #[test]
    fn test_alignment_bitand() {
        const ZERO: u64 = 0b0;
        const ONE: u64 = 0b1;
        const TWO: u64 = 0b10;
        const FOUR: u64 = 0b100;

        let zero: Alignment = ZERO.into();
        let one: Alignment = ONE.into();
        let two: Alignment = TWO.into();
        let four: Alignment = FOUR.into();

        assert_eq!(ZERO & ONE, 0b0);
        assert_eq!(zero & one, Alignment::Eight);

        assert_eq!(ONE & TWO, 0b00);
        assert_eq!(one & two, Alignment::Eight);

        assert_eq!(TWO & FOUR, 0b000);
        assert_eq!(two & four, Alignment::Eight);
    }

    #[test]
    fn test_alignment_bitor() {
        const ZERO: u64 = 0b0;
        const ONE: u64 = 0b1;
        const TWO: u64 = 0b10;
        const FOUR: u64 = 0b100;

        let zero: Alignment = ZERO.into();
        let one: Alignment = ONE.into();
        let two: Alignment = TWO.into();
        let four: Alignment = FOUR.into();

        assert_eq!(ZERO | ONE, 0b1);
        assert_eq!(zero | one, Alignment::One);

        assert_eq!(ONE | TWO, 0b11);
        assert_eq!(one | two, Alignment::One);

        assert_eq!(TWO | FOUR, 0b110);
        assert_eq!(two | four, Alignment::Two);
    }

    #[test]
    fn test_alignment_bitxor() {
        const ZERO: u64 = 0b0;
        const ONE: u64 = 0b1;
        const TWO: u64 = 0b10;
        const FOUR: u64 = 0b100;

        let zero: Alignment = ZERO.into();
        let one: Alignment = ONE.into();
        let two: Alignment = TWO.into();
        let four: Alignment = FOUR.into();

        assert_eq!(ZERO ^ ONE, ZERO | ONE);
        assert_eq!(zero ^ one, Alignment::One);

        assert_eq!(ONE ^ TWO, ONE | TWO);
        assert_eq!(one ^ two, Alignment::One);

        assert_eq!(TWO ^ FOUR, TWO | FOUR);
        assert_eq!(two ^ four, Alignment::Two);
    }

    #[test]
    fn test_alignment_div() {
        const ONE: u64 = 0b1;
        const TWO: u64 = 0b10;
        const THREE: u64 = 0b11;
        const FIVE: u64 = 0b101;

        let one: Alignment = ONE.into();
        let two: Alignment = TWO.into();
        let three: Alignment = THREE.into();
        let five: Alignment = FIVE.into();

        assert_eq!(THREE / TWO, ONE);
        assert_eq!(three / two, Alignment::One);

        assert_eq!(five, Alignment::One);
        assert_eq!(one / two, Alignment::One);
        assert_eq!(five / two, Alignment::One);

        assert_eq!(one / three, Alignment::One);
        assert_eq!(five / three, Alignment::One);
    }

    #[test]
    fn test_alignment_mul() {
        const TWO: u64 = 0b10;
        const THREE: u64 = 0b11;
        const FIVE: u64 = 0b101;
        const SIX: u64 = 0b110;

        let two: Alignment = TWO.into();
        let three: Alignment = THREE.into();
        let five: Alignment = FIVE.into();
        let six: Alignment = SIX.into();

        assert_eq!(THREE * TWO, SIX);
        assert_eq!(three * two, Alignment::Two);

        assert_eq!((FIVE * THREE) & 0b1, 0b1);
        assert_eq!(five * three, Alignment::One);

        assert_eq!((FIVE * SIX) & 0b1, 0b0);
        assert_eq!(five * six, Alignment::Two);
    }

    #[test]
    fn test_alignment_rem() {
        let two: Alignment = 2u64.into();
        let three: Alignment = 3u64.into();
        let five: Alignment = 5u64.into();
        let six: Alignment = 6u64.into();
        let ten: Alignment = 10u64.into();

        assert_eq!(two, Alignment::Two);
        assert_eq!(six, Alignment::Two);
        assert_eq!(ten, Alignment::Two);
        // True alignment is Eight, but we can only guess at the worst case...
        assert_eq!(6 % 2, 0);
        assert_eq!(six % two, Alignment::Two);
        // ... because the same alignments can have a true output of four
        assert_eq!(10 % 6, 4);
        assert_eq!(ten % six, Alignment::Two);

        // This even applies for what should be a noop
        assert_eq!(two % three, Alignment::One);

        assert_eq!(ten % five, Alignment::One);
        assert_eq!(10 % 3, 1);
        assert_eq!(ten % three, Alignment::One);
    }

    #[test]
    fn test_alignment_shl() {
        let one: Alignment = 1u64.into();

        assert_eq!(one << Value::from_u32(1), Alignment::One);
        assert_eq!(one << Value::from_u32(2), Alignment::One);
        assert_eq!(one << Value::from_u32(3), Alignment::One);
    }

    #[test]
    fn test_alignment_shr() {
        let zero: Alignment = 0u64.into();

        assert_eq!(zero >> Value::from_u32(1), Alignment::One);
        assert_eq!(zero >> Value::from_u32(2), Alignment::One);
        assert_eq!(zero >> Value::from_u32(3), Alignment::One);
    }
}
