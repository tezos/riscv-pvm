// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of comparison operations in JIT mode.

use cranelift::codegen::ir::InstBuilder;

use crate::instruction_context::Predicate;
use crate::instruction_context::comparable::Comparable;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::jit::builder::typed::Value;
use crate::machine_state::memory::MemoryConfig;

mod seal {
    /// Seal trait to prevent comparisons on non-integer types
    pub trait ComparableInteger {}

    impl ComparableInteger for u8 {}

    impl ComparableInteger for i8 {}

    impl ComparableInteger for u16 {}

    impl ComparableInteger for i16 {}

    impl ComparableInteger for u32 {}

    impl ComparableInteger for i32 {}

    impl ComparableInteger for u64 {}

    impl ComparableInteger for i64 {}

    impl ComparableInteger for u128 {}

    impl ComparableInteger for i128 {}

    impl<T> ComparableInteger for std::ptr::NonNull<T> {}
}

impl<T: seal::ComparableInteger, MC: MemoryConfig> Comparable<InstructionBuilder<'_, '_, MC>>
    for Value<T>
{
    type Result = Value<bool>;

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
    ) -> Self::Result {
        let raw = builder
            .ins()
            .icmp(comparison, self.to_value(), other.to_value());

        // SAFETY: Integer comparison operations return a `0` or `1` value as `I8`. We enforce that
        // `icmp` only compares integers using the `ComparableInteger` constraint on `T`.
        unsafe { Value::<bool>::from_raw(raw) }
    }
}
