// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Loading and storing of temporary values to/from the stack.
//!
//! Certain information is occassionally required to be on the stack, when executing
//! a JIT-compiled block.
//!
//! See for example [`handle_exception`]:
//!
//! ```ignore
//! use crate::machine_state::MachineCoreState;
//! use crate::machine_state::memory::Address;
//! use crate::machine_state::memory::MemoryConfig;
//! use crate::traps::EnvironException;
//! use crate::traps::Exception;
//!
//! extern "C" fn handle_exception<MC: MemoryConfig>(
//!     core: &mut MachineCoreState<MC, Owned>,
//!     current_pc: &mut Address,
//!     exception: &Option<Exception>,
//!     result: &mut Result<(), EnvironException>,
//! ) -> bool;
//! ```
//!
//! The boolean return indicates whether the function has failed. If it has, then an exception will
//! have been written to the place in memory specified by `exception`. `current_pc` is also an out
//! pointer - if the exception is successfully handled, an updated pc is returned.
//!
//! The most natural place to point to in memory, is the function stack. Certain values can safely
//! live in the registers, and so can be loaded from/stored to the stack.
//!
//! Larger/more complex types cannot be moved between registers and the stack safely. Exceptions
//! are a good example of this. For now, we choose to pass pointers from values created in rust
//! code, for safety.
//!
//! [`handle_exception`]: super::handle_exception

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use cranelift::codegen::ir::InstBuilder;
use cranelift::codegen::ir::StackSlot;
use cranelift::codegen::ir::StackSlotData;
use cranelift::codegen::ir::StackSlotKind;
use cranelift::codegen::ir::Type;
use cranelift::codegen::ir::types::I64;
use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::types::I8;
use cranelift::prelude::types::I16;
use cranelift::prelude::types::I32;

use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::machine_state::registers::FValue;
use crate::traps::Exception;

/// Any value of type `T: StackAddressable` may be placed on the stack, and
/// a pointer to it obtained.
pub(crate) trait StackAddressable {
    /// The underlying 'runtime' type of the value - e.g. a u64/u32 etc.
    ///
    /// Separating these allows us to explicitly use different stack slot kinds, for different
    /// 'semantic' values (e.g. Address vs XValue), despite the underlying type being shared.
    ///
    /// This offers an additional level of type-safety.
    type Underlying: Sized;

    /// The size, in bytes, of the underlying value.
    const SIZE: u32 = std::mem::size_of::<Self::Underlying>() as u32;

    /// Require `1 << ALIGN_SHIFT == align(Underlying)`, since cranelift
    /// consumes the _shift_ rather than the raw align.
    const ALIGN_SHIFT: u8 = {
        let align = std::mem::align_of::<Self::Underlying>();

        if !align.is_power_of_two() {
            panic!("Cranelift requires values stored on the stack to be align to a power of two");
        }

        align.trailing_zeros() as u8
    };

    /// Declaration of how a stack slot should be placed - including size and alignment.
    ///
    /// This is consumed by Cranelift's builder when declaring a stack slot.
    const STACK_SLOT_DATA: StackSlotData = StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        size: Self::SIZE,
        align_shift: Self::ALIGN_SHIFT,
    };
}

impl StackAddressable for Exception {
    type Underlying = Exception;
}

impl<T: StackAddressable> StackAddressable for MaybeUninit<T> {
    type Underlying = MaybeUninit<T::Underlying>;
}

/// Any value of type `T: Stackable` may be loaded to/from the stack, by JIT-compiled code.
pub(crate) trait Stackable: StackAddressable {
    /// The type's representation in Cranelift IR.
    const IR_TYPE: Type;
}

macro_rules! impl_stackable_int {
    ($bits:literal) => {
        paste::paste! {
            impl StackAddressable for [<u $bits>] {
                type Underlying = [<u $bits>];
            }

            impl Stackable for [<u $bits>] {
                const IR_TYPE: Type = [<I $bits>];
            }

            impl StackAddressable for [<i $bits>] {
                type Underlying = [<i $bits>];
            }

            impl Stackable for [<i $bits>] {
                const IR_TYPE: Type = [<I $bits>];
            }
        }
    };
}

impl_stackable_int!(8);
impl_stackable_int!(16);
impl_stackable_int!(32);
impl_stackable_int!(64);

impl StackAddressable for FValue {
    type Underlying = FValue;
}

impl Stackable for FValue {
    const IR_TYPE: Type = I64;
}

/// Dedicated space on the stack to store a value of the underlying type.
pub(super) struct Slot<T> {
    slot: StackSlot,
    ptr_type: Type,
    _pd: PhantomData<T>,
}

impl<T: StackAddressable> Slot<MaybeUninit<T>> {
    /// Create a new slot on the stack, to hold a value of the underlying type of T.
    pub(super) fn new(ptr_type: Type, builder: &mut FunctionBuilder) -> Self {
        let slot = builder.create_sized_stack_slot(T::STACK_SLOT_DATA);

        Self {
            slot,
            ptr_type,
            _pd: PhantomData,
        }
    }

    /// Treat the slot as initialised.
    ///
    /// # Safety
    ///
    /// You must ensure the slot is initialised before using it.
    pub(super) unsafe fn assume_init(self) -> Slot<T> {
        Slot {
            slot: self.slot,
            ptr_type: self.ptr_type,
            _pd: PhantomData,
        }
    }
}

impl<T: Stackable> Slot<MaybeUninit<T>> {
    /// Emit IR to initialise the stack slot.
    pub(super) fn init(self, builder: &mut FunctionBuilder, value: Value<T>) -> Slot<T> {
        builder.ins().stack_store(value.to_value(), self.slot, 0);
        Slot {
            slot: self.slot,
            ptr_type: self.ptr_type,
            _pd: PhantomData,
        }
    }
}

impl<T: StackAddressable> Slot<T> {
    /// Get a pointer to the memory of the slot.
    pub(super) fn ptr(&self, builder: &mut FunctionBuilder) -> Pointer<T> {
        let raw_value = builder.ins().stack_addr(self.ptr_type, self.slot, 0);

        // SAFETY: If `T` is the type of the slot, then the pointer to that slot is `NonNull<T>`.
        unsafe { Value::<NonNull<T>>::from_raw(raw_value) }
    }
}

impl<T: Stackable> Slot<T> {
    /// Emit IR to load a value from the current stack slot.
    pub(super) fn load(&self, builder: &mut FunctionBuilder) -> Value<T> {
        let raw_value = builder.ins().stack_load(T::IR_TYPE, self.slot, 0);

        // SAFETY: `T` is the slot value type.
        unsafe { Value::<T>::from_raw(raw_value) }
    }
}

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot,
            ptr_type: self.ptr_type,
            _pd: PhantomData,
        }
    }
}
