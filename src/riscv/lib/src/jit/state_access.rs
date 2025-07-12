// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! External state access function registry
//!
//! This module provides state access methods using external functions (i.e. not implement using
//! Cranelift IR).

pub(crate) mod stack;

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use cranelift::codegen::ir;
use cranelift::codegen::ir::InstBuilder;
use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::isa::TargetFrontendConfig;

use super::builder::errno::ErrnoImpl;
use crate::instruction_context::ICB;
use crate::instruction_context::StoreLoadInt;
use crate::interpreter::float::RoundRDN;
use crate::interpreter::float::RoundRMM;
use crate::interpreter::float::RoundRNE;
use crate::interpreter::float::RoundRTZ;
use crate::interpreter::float::RoundRUP;
use crate::interpreter::float::RoundingMode;
use crate::interpreter::float::StaticRoundingMode;
use crate::jit::builder::ext_calls;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::BadMemoryAccess;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FRegister;
use crate::machine_state::registers::XValue;
use crate::state_backend::Elem;
use crate::state_backend::owned_backend::Owned;
use crate::traps::EnvironException;
use crate::traps::Exception;

/// Read the value of the given [`FRegister`].
extern "C" fn fregister_read<MC: MemoryConfig>(
    core: &MachineCoreState<MC, Owned>,
    reg: FRegister,
) -> f64 {
    let fval = core.hart.fregisters.read(reg);
    fval.bits()
}

/// Write the given value to the given [`FRegister`].
extern "C" fn fregister_write<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    reg: FRegister,
    val: f64,
) {
    let fval = val.to_bits();
    core.hart.fregisters.write(reg, fval.into())
}

/// Handle an [`Exception`].
///
/// If the exception is succesfully handled, the
/// `current_pc` is updated to the new value, and returns true. The `current_pc`
/// remains initialised to its previous value otherwise.
///
/// If the exception needs to be treated by the execution environment,
/// `result` is updated with the `EnvironException` and `false` is
/// returned.
///
/// # Panics
///
/// Panics if the exception does not have `Some(_)` value.
///
/// See [`MachineCoreState::address_on_exception`].
extern "C" fn handle_exception<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    current_pc: &mut Address,
    exception: &Exception,
    result: &mut Result<(), EnvironException>,
) -> bool {
    let res = core.address_on_exception(*exception, *current_pc);

    match res {
        Err(e) => {
            *result = Err(e);
            false
        }
        Ok(address) => {
            *current_pc = address;
            true
        }
    }
}

/// Raise an [`Exception::IllegalInstruction`].
///
/// Writes the instruction to the given exception memory, after which it would be safe to
/// assume it is initialised.
extern "C" fn raise_illegal_instruction_exception(exception_out: &mut MaybeUninit<Exception>) {
    exception_out.write(Exception::IllegalInstruction);
}

/// Raise an [`Exception::StoreAMOAccessFault`].
///
/// Writes the instruction to the given exception memory, after which it would be safe to
/// assume it is initialised.
extern "C" fn raise_store_amo_access_fault_exception(exception_out: &mut MaybeUninit<Exception>) {
    exception_out.write(Exception::StoreAMOAccessFault);
}

/// Raise the appropriate environment-call exception given the current machine mode.
///
/// Writes the exception to the given exception memory, after which it would be safe to
/// assume it is initialised.
extern "C" fn ecall(exception_out: &mut MaybeUninit<Exception>) {
    exception_out.write(Exception::EnvCall);
}

/// Store the lowest `width` bytes of the given value to memory, at the physical address.
///
/// If the store is successful, `false` is returned to indicate no exception handling is necessary.
///
/// If the store fails (due to out of bouds etc) then an exception will be written
/// to `exception_out` and `true` returned to indicate exception handling will be necessary.
extern "C" fn memory_store<E: Elem, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    address: u64,
    value: E,
    exception_out: &mut MaybeUninit<Exception>,
) -> bool {
    match core.main_memory.write(address, value) {
        Ok(()) => false,
        Err(BadMemoryAccess) => {
            exception_out.write(Exception::StoreAMOAccessFault);
            true
        }
    }
}

/// Load `width` bytes from memory, at the physical address, into lowest `width` bytes of an
/// `XValue`, with (un)signed extension.
///
/// If the load is successful, `false` is returned to indicate no exception handling is
/// necessary.
///
/// If the load fails (due to out of bouds etc) then an exception will be written
/// to `exception_out` and `true` returned to indicate exception handling will be necessary.
extern "C" fn memory_load<E: Elem, MC: MemoryConfig>(
    core: &MachineCoreState<MC, Owned>,
    address: u64,
    xval_out: &mut MaybeUninit<E>,
    exception_out: &mut MaybeUninit<Exception>,
) -> bool {
    match core.main_memory.read::<E>(address) {
        Ok(value) => {
            xval_out.write(value);
            false
        }
        Err(BadMemoryAccess) => {
            exception_out.write(Exception::LoadAccessFault);
            true
        }
    }
}

extern "C" fn f64_from_x64_unsigned_dynamic<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    exception_out: &mut MaybeUninit<Exception>,
    xval: XValue,
    f64_out: &mut MaybeUninit<f64>,
) -> bool {
    match MachineCoreState::f64_from_x64_unsigned_dynamic(core, xval) {
        Ok(fval) => {
            let f64val = fval.bits();
            f64_out.write(f64val);
            false
        }
        Err(e) => {
            exception_out.write(e);
            true
        }
    }
}

/// Convert an unsigned 64-bit `XValue` to a 64-bit `FValue` using the given static rounding mode.
extern "C" fn f64_from_x64_unsigned_static<RM: StaticRoundingMode, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    xval: XValue,
) -> f64 {
    let fval = MachineCoreState::f64_from_x64_unsigned_static(core, xval, RM::ROUND);
    fval.bits()
}

/// References to locally imported state access methods, used to directly call these accessor
/// methods in the JIT-compilation context.
pub struct JsaCalls<MC: MemoryConfig> {
    /// Target configuration which provides useful information about the target ISA, such as
    /// pointer type and width
    target_config: TargetFrontendConfig,
    f64_from_x64_unsigned_dynamic: Option<FuncRef>,
    /// Reusable stack slot for the exception pointer
    exception_ptr_slot: Option<stack::Slot<MaybeUninit<Exception>>>,

    /// Reusable stack slot for the PC value
    pc_slot: Option<stack::Slot<MaybeUninit<Address>>>,

    /// Reusable stack slot for an FValue.
    f64_ptr_slot: Option<stack::Slot<MaybeUninit<f64>>>,

    _pd: PhantomData<MC>,
}

impl<MC: MemoryConfig> JsaCalls<MC> {
    /// Get the stack slot for the exception pointer.
    fn exception_ptr_slot(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> stack::Slot<MaybeUninit<Exception>> {
        self.exception_ptr_slot
            .get_or_insert_with(|| stack::Slot::new(self.target_config.pointer_type(), builder))
            .clone()
    }

    /// Get the stack slot for the PC value.
    fn pc_slot(&mut self, builder: &mut FunctionBuilder) -> stack::Slot<MaybeUninit<Address>> {
        self.pc_slot
            .get_or_insert_with(|| stack::Slot::new(self.target_config.pointer_type(), builder))
            .clone()
    }

    /// Get the stack slot for an FValue.
    fn f64_ptr_slot(&mut self, builder: &mut FunctionBuilder) -> stack::Slot<MaybeUninit<f64>> {
        self.f64_ptr_slot
            .get_or_insert_with(|| stack::Slot::new(self.target_config.pointer_type(), builder))
            .clone()
    }

    /// Wrapper to simplify calling state access functions from within the function under construction.
    pub(super) fn func_calls(target_config: TargetFrontendConfig) -> Self {
        Self {
            target_config,
            f64_from_x64_unsigned_dynamic: None,
            exception_ptr_slot: None,
            pc_slot: None,
            f64_ptr_slot: None,
            _pd: PhantomData,
        }
    }

    /// Emit the required IR to call `handle_exception`.
    ///
    /// # Panics
    ///
    /// The call to `handle_exception` will panic (at runtime) if no exception
    /// has occurred so-far in the JIT-compiled function, if the error-handling
    /// code is triggerred.
    pub(super) fn handle_exception(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        exception_ptr: Pointer<Exception>,
        result_ptr: Pointer<Result<(), EnvironException>>,
        current_pc: Value<Address>,
    ) -> ExceptionHandledOutcome {
        let pc_slot = self.pc_slot(builder).init(builder, current_pc);
        let pc_ptr = pc_slot.ptr(builder);

        // SAFETY: Arguments get cast into references with valid lifetimes.
        // - `core_ptr` is a JIT function argument
        // - `pc_ptr` points to a stack slot which is valid for the duration of the JIT function
        // - `exception_ptr` points to a stack slot as well (allocated by the caller)
        // - `result_ptr` is a JIT function argument
        let handled = ext_calls::call4(
            &self.target_config,
            builder,
            self::handle_exception,
            unsafe { core_ptr.as_mut() },
            unsafe { pc_ptr.as_mut() },
            unsafe { exception_ptr.as_ref() },
            unsafe { result_ptr.as_mut() },
        );

        let new_pc = pc_slot.load(builder);
        ExceptionHandledOutcome { handled, new_pc }
    }

    /// Emit the required IR to call `raise_illegal_exception`.
    ///
    /// This returns an initialised pointer to the exception.
    pub(super) fn raise_illegal_instruction_exception(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> Pointer<Exception> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        // SAFETY: The exception pointer reference is scoped to the JIT function. Hence it is safe
        // to pass it to the external function which is called within the JIT function scope.
        ext_calls::call1(
            &self.target_config,
            builder,
            self::raise_illegal_instruction_exception,
            unsafe { exception_ptr.as_mut() },
        );

        // SAFETY: The `raise_illegal_instruction_exception` function writes to the exception slot
        // unconditionally.
        unsafe { exception_slot.assume_init().ptr(builder) }
    }

    /// Emit the required IR to call `raise_store_amo_access_fault_exception`.
    ///
    /// This returns an initialised pointer to the exception.
    pub(super) fn raise_store_amo_access_fault_exception(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> Pointer<Exception> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        // SAFETY: The exception reference is guaranteed to be valid for the duration of the call as
        // it is scoped to the JIT function.
        ext_calls::call1(
            &self.target_config,
            builder,
            self::raise_store_amo_access_fault_exception,
            unsafe { exception_ptr.as_mut() },
        );

        // SAFETY: The `raise_store_amo_access_fault_exception` function writes to the exception
        // slot unconditionally.
        unsafe { exception_slot.assume_init().ptr(builder) }
    }

    /// Emit the required IR to call `ecall`.
    ///
    /// This returns an initialised pointer to the appropriate environment
    /// call exception for the current machine mode.
    pub(super) fn ecall(&mut self, builder: &mut FunctionBuilder) -> Pointer<Exception> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        // SAFETY: The exception reference is guaranteed to be valid for the duration of the call as
        // it points to a stack slot which is valid for the duration of the JIT function.
        ext_calls::call1(&self.target_config, builder, self::ecall, unsafe {
            exception_ptr.as_mut()
        });

        // SAFETY: The `ecall` function writes to the exception slot unconditionally.
        unsafe { exception_slot.assume_init().ptr(builder) }
    }

    /// Emit the required IR to call `memory_store`.
    ///
    /// Returns `errno` - on success, no additional values are returned.
    pub(super) fn memory_store<V: StoreLoadInt>(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        phys_address: Value<Address>,
        value: Value<XValue>,
    ) -> ErrnoImpl<(), impl FnOnce(&mut FunctionBuilder) + 'static> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let value = V::from_xvalue_ir(builder, value);

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument
        // - `exception_ptr` points to a stack slot within the JIT function
        let is_exception = ext_calls::call4(
            &self.target_config,
            builder,
            self::memory_store,
            unsafe { core_ptr.as_mut() },
            phys_address,
            value,
            unsafe { exception_ptr.as_mut() },
        );

        ErrnoImpl::new(is_exception, exception_ptr, |_| {})
    }

    /// Emit the required IR to call `memory_load`.
    ///
    /// Returns `errno` - on success, the loaded value is returned.
    pub(super) fn memory_load<V: StoreLoadInt>(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        phys_address: Value<Address>,
    ) -> ErrnoImpl<Value<XValue>, impl FnOnce(&mut FunctionBuilder) -> Value<XValue> + 'static>
    {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let xval_slot =
            stack::Slot::<MaybeUninit<V>>::new(self.target_config.pointer_type(), builder);
        let xval_ptr = xval_slot.ptr(builder);

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument
        // - `xval_ptr` points to a stack slot which is valid for the duration of the JIT function
        // - `exception_ptr` points to a stack slot within the JIT function as well
        let is_exception = ext_calls::call4(
            &self.target_config,
            builder,
            self::memory_load,
            unsafe { core_ptr.as_ref() },
            phys_address,
            unsafe { xval_ptr.as_mut() },
            unsafe { exception_ptr.as_mut() },
        );

        ErrnoImpl::new(is_exception, exception_ptr, move |builder| {
            // SAFETY: The slot is guaranteed to be initialised at this point as this closure
            // generates IR for the success case when the external function will have written to
            // the stack slot.
            let xval = unsafe { xval_slot.assume_init().load(builder) };

            V::to_xvalue_ir(builder, xval)
        })
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_dynamic`.
    ///
    /// Returns `errno` - on success, the new F64 value is returned.
    pub(super) fn f64_from_x64_unsigned_dynamic(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        xval: Value<XValue>,
    ) -> ErrnoImpl<Value<f64>, impl FnOnce(&mut FunctionBuilder) -> Value<f64> + 'static> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let f64_slot = self.f64_ptr_slot(builder);
        let f64_ptr = f64_slot.ptr(builder);

        let is_exception = ext_calls::call4(
            &self.target_config,
            builder,
            self::f64_from_x64_unsigned_dynamic,
            unsafe { core_ptr.as_mut() },
            unsafe { exception_ptr.as_mut() },
            xval,
            unsafe { f64_ptr.as_mut() },
        );

        ErrnoImpl::new(is_exception, exception_ptr, move |builder| {
            // SAFETY: This closure runs after the success case of the call, where the `f64_slot`
            // is guaranteed to have been initialised with an f64 value.
            unsafe { f64_slot.assume_init().load(builder) }
        })
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_static`.
    /// The converted value is returned as `F64`.
    pub(super) fn f64_from_x64_unsigned_static(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        xval: Value<XValue>,
        rm: RoundingMode,
    ) -> Value<f64> {
        let callee = match rm {
            RoundingMode::RNE => self::f64_from_x64_unsigned_static::<RoundRNE, _>,
            RoundingMode::RTZ => self::f64_from_x64_unsigned_static::<RoundRTZ, _>,
            RoundingMode::RDN => self::f64_from_x64_unsigned_static::<RoundRDN, _>,
            RoundingMode::RUP => self::f64_from_x64_unsigned_static::<RoundRUP, _>,
            RoundingMode::RMM => self::f64_from_x64_unsigned_static::<RoundRMM, _>,
        };

        ext_calls::call2(
            &self.target_config,
            builder,
            callee,
            unsafe { core_ptr.as_mut() },
            xval,
        )
    }

    /// Emit the required IR to read the value from the given fregister.
    pub(super) fn ir_freg_read(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        reg: FRegister,
    ) -> Value<f64> {
        let reg_value = unsafe {
            Value::<FRegister>::from_literal(&self.target_config, builder, reg as u8 as i64)
        };

        ext_calls::call2(
            &self.target_config,
            builder,
            self::fregister_read,
            unsafe { core_ptr.as_ref() },
            reg_value,
        )
    }

    /// Emit the required IR to write the value to the given fregister.
    pub(super) fn ir_freg_write(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        reg: FRegister,
        value: Value<f64>,
    ) {
        let reg_value = unsafe {
            Value::<FRegister>::from_literal(&self.target_config, builder, reg as u8 as i64)
        };

        ext_calls::call3(
            &self.target_config,
            builder,
            self::fregister_write,
            unsafe { core_ptr.as_mut() },
            reg_value,
            value,
        );
    }
}

/// Outcome of handling an exception.
pub(super) struct ExceptionHandledOutcome {
    /// Whether the exception was succesfully handled.
    ///
    /// - If true, the exception was handled and the step is completed.
    /// - If false, the exception must be instead handled by the environment.
    ///   The step is not complete.
    pub handled: Value<bool>,

    /// The new value of the instruction pc, after exception handling.
    pub new_pc: Value<Address>,
}
