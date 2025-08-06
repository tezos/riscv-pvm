// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! External state access function registry
//!
//! This module provides state access methods using external functions (i.e. not implemented using
//! Cranelift IR).

pub(crate) mod stack;

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::IntCC;
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift::prelude::types::I64;

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
use crate::jit::builder::typed;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::machine_state::MachineCoreState;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::BadMemoryAccess;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::XValue;
use crate::state_backend::Elem;
use crate::state_backend::owned_backend::Owned;
use crate::traps::EnvironException;
use crate::traps::Exception;

/// Exception codes used for efficient exception handling in JIT-compiled code
///
/// This enum represents different types of exceptions that can occur during
/// instruction execution, encoded as i64 values for easy transmission between
/// JIT-compiled code and runtime handlers.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum ExceptionCode {
    NoException = 0,
    InstructionAccessFault = Exception::InstructionAccessFault as i64,
    IllegalInstruction = Exception::IllegalInstruction as i64,
    Breakpoint = Exception::Breakpoint as i64,
    LoadAccessFault = Exception::LoadAccessFault as i64,
    StoreAMOAccessFault = Exception::StoreAMOAccessFault as i64,
    EnvCall = Exception::EnvCall as i64,
    InstructionPageFault = Exception::InstructionPageFault as i64,
    LoadPageFault = Exception::LoadPageFault as i64,
    StoreAMOPageFault = Exception::StoreAMOPageFault as i64,
}

impl ExceptionCode {
    /// Construct the corresponding exception code.
    pub fn from_exception(exception: Exception) -> Self {
        match exception {
            Exception::InstructionAccessFault => Self::InstructionAccessFault,
            Exception::IllegalInstruction => Self::IllegalInstruction,
            Exception::Breakpoint => Self::Breakpoint,
            Exception::LoadAccessFault => Self::LoadAccessFault,
            Exception::StoreAMOAccessFault => Self::StoreAMOAccessFault,
            Exception::EnvCall => Self::EnvCall,
            Exception::InstructionPageFault => Self::InstructionPageFault,
            Exception::LoadPageFault => Self::LoadPageFault,
            Exception::StoreAMOPageFault => Self::StoreAMOPageFault,
        }
    }

    /// Convert the exception code back to an [`Exception`].
    ///
    /// # Panics
    ///
    /// This function panics if called on `NoException`, as it is not a valid exception.
    pub fn to_exception(self) -> Exception {
        match self {
            Self::NoException => unreachable!(),
            Self::InstructionAccessFault => Exception::InstructionAccessFault,
            Self::IllegalInstruction => Exception::IllegalInstruction,
            Self::Breakpoint => Exception::Breakpoint,
            Self::LoadAccessFault => Exception::LoadAccessFault,
            Self::StoreAMOAccessFault => Exception::StoreAMOAccessFault,
            Self::EnvCall => Exception::EnvCall,
            Self::InstructionPageFault => Exception::InstructionPageFault,
            Self::LoadPageFault => Exception::LoadPageFault,
            Self::StoreAMOPageFault => Exception::StoreAMOPageFault,
        }
    }

    /// Construct an IR value representing the exception.
    pub fn build_exception_code(
        builder: &mut FunctionBuilder,
        exception: Exception,
    ) -> Value<ExceptionCode> {
        let exception_code = Self::from_exception(exception);
        let raw = builder.ins().iconst(I64, exception_code as i64);

        // SAFETY: The raw value is constructed from a valid discriminant. ExceptionCode is
        // represented as a `i64`, so we can safely convert it to a `Value<ExceptionCode>`.
        unsafe { Value::<ExceptionCode>::from_raw(raw) }
    }
}

impl Value<ExceptionCode> {
    /// Does the exception code represent an exception?
    pub fn is_exception(self, builder: &mut FunctionBuilder) -> Value<bool> {
        let raw = builder.ins().icmp_imm(
            IntCC::NotEqual,
            self.to_value(),
            ExceptionCode::NoException as i64,
        );

        // SAFETY: `icmp_imm` returns a boolean.
        unsafe { Value::<bool>::from_raw(raw) }
    }
}

impl typed::Typed for ExceptionCode {
    const TYPE: typed::Type = typed::Type::Basic(I64);
}

/// Handle an [`ExceptionCode`].
///
/// If the exception is succesfully handled, the
/// `current_pc` is updated to the new value, and returns true. The `current_pc`
/// remains initialised to its previous value otherwise.
///
/// If the exception needs to be treated by the execution environment,
/// `result` is updated with the `EnvironException` and `false` is
/// returned.
///
/// See [`MachineCoreState::address_on_exception`].
extern "C" fn handle_exception<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    current_pc: &mut Address,
    exception: ExceptionCode,
    result: &mut Result<(), EnvironException>,
) -> bool {
    let exception = exception.to_exception();
    let res = core.address_on_exception(exception, *current_pc);

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

/// Store the lowest `width` bytes of the given value to memory, at the physical address.
///
/// Returns [`ExceptionCode::NoException`] if the store is successful.
///
/// If the store fails (due to out of bounds etc) then the appropriate exception code
/// is returned to indicate the type of failure that occurred.
extern "C" fn memory_store<E: Elem, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    address: u64,
    value: E,
) -> ExceptionCode {
    match core.main_memory.write(address, value) {
        Ok(()) => ExceptionCode::NoException,
        Err(BadMemoryAccess) => ExceptionCode::from_exception(Exception::StoreAMOAccessFault),
    }
}

/// Load `width` bytes from memory, at the physical address, into lowest `width` bytes of an
/// `XValue`, with (un)signed extension.
///
/// If the load is successful, the value is written to `xval_out` and
/// [`ExceptionCode::NoException`] is returned.
///
/// If the load fails (due to out of bounds etc) then the appropriate exception code
/// is returned to indicate the type of failure that occurred.
extern "C" fn memory_load<E: Elem, MC: MemoryConfig>(
    core: &MachineCoreState<MC, Owned>,
    address: u64,
    xval_out: &mut MaybeUninit<E>,
) -> ExceptionCode {
    match core.main_memory.read::<E>(address) {
        Ok(value) => {
            xval_out.write(value);
            ExceptionCode::NoException
        }

        Err(BadMemoryAccess) => ExceptionCode::from_exception(Exception::LoadAccessFault),
    }
}

extern "C" fn f64_from_x64_unsigned_dynamic<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    xval: XValue,
) -> FValue {
    MachineCoreState::f64_from_x64_unsigned_dynamic(core, xval)
}

/// Convert an unsigned 64-bit `XValue` to a 64-bit `FValue` using the given static rounding mode.
extern "C" fn f64_from_x64_unsigned_static<RM: StaticRoundingMode, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    xval: XValue,
) -> FValue {
    MachineCoreState::f64_from_x64_unsigned_static(core, xval, RM::ROUND)
}

/// Write to a Control and Status register.
extern "C" fn csr_write<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    csr: CSRegister,
    value: XValue,
) {
    core.hart.csregisters.write(csr, value);
}

/// Read from a Control and Status register.
extern "C" fn csr_read<MC: MemoryConfig>(
    core: &MachineCoreState<MC, Owned>,
    csr: CSRegister,
) -> XValue {
    core.hart.csregisters.read(csr)
}

/// External function call registry for state accesses
pub struct JsaCalls<MC: MemoryConfig> {
    /// Target configuration which provides useful information about the target ISA, such as
    /// pointer type and width
    target_config: TargetFrontendConfig,

    /// Reusable stack slot for the PC value
    pc_slot: Option<stack::Slot<MaybeUninit<Address>>>,

    _pd: PhantomData<MC>,
}

impl<MC: MemoryConfig> JsaCalls<MC> {
    /// Get the stack slot for the PC value.
    fn pc_slot(&mut self, builder: &mut FunctionBuilder) -> stack::Slot<MaybeUninit<Address>> {
        self.pc_slot
            .get_or_insert_with(|| stack::Slot::new(self.target_config.pointer_type(), builder))
            .clone()
    }

    /// Construct a new `JsaCalls` instance with the given target configuration.
    pub(super) fn new(target_config: TargetFrontendConfig) -> Self {
        Self {
            target_config,
            pc_slot: None,
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
        exception: Value<ExceptionCode>,
        result_ptr: Pointer<Result<(), EnvironException>>,
        current_pc: Value<Address>,
    ) -> ExceptionHandledOutcome {
        let pc_slot = self.pc_slot(builder).init(builder, current_pc);
        let pc_ptr = pc_slot.ptr(builder);

        // SAFETY: Arguments get cast into references with valid lifetimes.
        // - `core_ptr` is a JIT function argument
        // - `pc_ptr` points to a stack slot which is valid for the duration of the JIT function
        // - `result_ptr` is a JIT function argument
        let handled = ext_calls::call4(
            &self.target_config,
            builder,
            self::handle_exception,
            unsafe { core_ptr.as_mut() },
            unsafe { pc_ptr.as_mut() },
            exception,
            unsafe { result_ptr.as_mut() },
        );

        let new_pc = pc_slot.load(builder);
        ExceptionHandledOutcome { handled, new_pc }
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
        let value = V::from_xvalue_ir(builder, value);

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument
        let exception = ext_calls::call3(
            &self.target_config,
            builder,
            self::memory_store,
            unsafe { core_ptr.as_mut() },
            phys_address,
            value,
        );

        ErrnoImpl::new(exception, |_| {})
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
        let xval_slot =
            stack::Slot::<MaybeUninit<V>>::new(self.target_config.pointer_type(), builder);
        let xval_ptr = xval_slot.ptr(builder);

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument
        // - `xval_ptr` points to a stack slot which is valid for the duration of the JIT function
        let exception = ext_calls::call3(
            &self.target_config,
            builder,
            self::memory_load,
            unsafe { core_ptr.as_ref() },
            phys_address,
            unsafe { xval_ptr.as_mut() },
        );

        ErrnoImpl::new(exception, move |builder| {
            // SAFETY: The slot is guaranteed to be initialised at this point as this closure
            // generates IR for the success case when the external function will have written to
            // the stack slot.
            let xval = unsafe { xval_slot.assume_init().load(builder) };

            V::to_xvalue_ir(builder, xval)
        })
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_dynamic`.
    ///
    /// Returns the converted FValue.
    pub(super) fn f64_from_x64_unsigned_dynamic(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        xval: Value<XValue>,
    ) -> Value<FValue> {
        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument, therefore valid for the entire function
        ext_calls::call2(
            &self.target_config,
            builder,
            self::f64_from_x64_unsigned_dynamic,
            unsafe { core_ptr.as_mut() },
            xval,
        )
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_static`.
    /// The converted value is returned as `FValue`.
    pub(super) fn f64_from_x64_unsigned_static(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        xval: Value<XValue>,
        rm: RoundingMode,
    ) -> Value<FValue> {
        let callee = match rm {
            RoundingMode::RNE => self::f64_from_x64_unsigned_static::<RoundRNE, _>,
            RoundingMode::RTZ => self::f64_from_x64_unsigned_static::<RoundRTZ, _>,
            RoundingMode::RDN => self::f64_from_x64_unsigned_static::<RoundRDN, _>,
            RoundingMode::RUP => self::f64_from_x64_unsigned_static::<RoundRUP, _>,
            RoundingMode::RMM => self::f64_from_x64_unsigned_static::<RoundRMM, _>,
        };

        // SAFETY: The machine core state pointer is a JIT function argument, and therefore its
        // pointee will outlive this call.
        ext_calls::call2(
            &self.target_config,
            builder,
            callee,
            unsafe { core_ptr.as_mut() },
            xval,
        )
    }

    /// Write to a Control and Status register.
    pub(super) fn csr_write(
        &self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        csr: CSRegister,
        value: Value<XValue>,
    ) {
        // SAFETY: We construct the typed value from the CSRegister enum discriminant, ensuring
        // the correct value representation for the CSR parameter.
        let reg_value = unsafe {
            Value::<CSRegister>::from_discriminant(&self.target_config, builder, csr as u64 as i64)
        };

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument which means the reference through it
        //   will be valid for the duration of the call
        ext_calls::call3(
            &self.target_config,
            builder,
            self::csr_write,
            unsafe { core_ptr.as_mut() },
            reg_value,
            value,
        );
    }

    /// Read from a Control and Status register.
    pub(super) fn csr_read(
        &self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        csr: CSRegister,
    ) -> Value<XValue> {
        // SAFETY: We construct the typed value from the CSRegister enum discriminant, ensuring
        // the correct value representation for the CSR parameter.
        let reg_value = unsafe {
            Value::<CSRegister>::from_discriminant(&self.target_config, builder, csr as u64 as i64)
        };

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument which means the reference through it
        //   will be valid for the duration of the call
        ext_calls::call2(
            &self.target_config,
            builder,
            self::csr_read,
            unsafe { core_ptr.as_ref() },
            reg_value,
        )
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
