// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Comparison operations required for a given instruction builder context, including
//! implementations for interpreted mode.

use super::Predicate;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::XValue;
use crate::machine_state::registers::XValue32;
use crate::state_backend::ManagerReadWrite;

/// Trait for comparison operations on **XValues** used in the instruction builder context `I`.
pub trait Comparable<I: ?Sized> {
    /// Result of the comparison operation
    type Result;

    /// Compare two values, given the operation to compare them with.
    fn compare(self, other: Self, predicate: Predicate, icb: &mut I) -> Self::Result;
}

impl<MC: MemoryConfig, M: ManagerReadWrite> Comparable<MachineCoreState<MC, M>> for XValue {
    type Result = bool;

    #[inline(always)]
    fn compare(self, other: Self, predicate: Predicate, _: &mut MachineCoreState<MC, M>) -> bool {
        match predicate {
            Predicate::Equal => self == other,
            Predicate::NotEqual => self != other,
            Predicate::LessThanSigned => (self as i64) < (other as i64),
            Predicate::LessThanUnsigned => self < other,
            Predicate::LessThanOrEqualSigned => (self as i64) <= (other as i64),
            Predicate::GreaterThanSigned => (self as i64) > (other as i64),
            Predicate::GreaterThanOrEqualSigned => (self as i64) >= (other as i64),
            Predicate::GreaterThanOrEqualUnsigned => self >= other,
        }
    }
}

impl<MC: MemoryConfig, M: ManagerReadWrite> Comparable<MachineCoreState<MC, M>> for XValue32 {
    type Result = bool;

    #[inline(always)]
    fn compare(self, other: Self, predicate: Predicate, _: &mut MachineCoreState<MC, M>) -> bool {
        match predicate {
            Predicate::Equal => self == other,
            Predicate::NotEqual => self != other,
            Predicate::LessThanSigned => (self as i32) < (other as i32),
            Predicate::LessThanUnsigned => self < other,
            Predicate::LessThanOrEqualSigned => (self as i32) <= (other as i32),
            Predicate::GreaterThanSigned => (self as i32) > (other as i32),
            Predicate::GreaterThanOrEqualSigned => (self as i32) >= (other as i32),
            Predicate::GreaterThanOrEqualUnsigned => self >= other,
        }
    }
}
