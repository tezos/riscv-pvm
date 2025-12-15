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
use octez_riscv_data::mode::Normal;
use octez_riscv_data::serialisation::elem::Elem;

use super::builder::errno::ErrnoImpl;
use crate::exceptions::Exception;
use crate::instruction_context::ICB;
use crate::instruction_context::StoreLoadFloat;
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
use crate::machine_state::page_cache::jitted::JittedPage;
use crate::machine_state::page_cache::run_code_page_interpreted;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::XValue;

/// Exception codes used for efficient exception handling in JIT-compiled code
///
/// This enum represents different types of exceptions that can occur during
/// instruction execution, encoded as i64 values for easy transmission between
/// JIT-compiled code and runtime handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum ExceptionCode {
    NoException = 0,
    InstructionAccessFault = Exception::InstructionAccessFault as i64,
    IllegalInstruction = Exception::IllegalInstruction as i64,
    Breakpoint = Exception::Breakpoint as i64,
    LoadAccessFault = Exception::LoadAccessFault as i64,
    StoreAMOAccessFault = Exception::StoreAMOAccessFault as i64,
    EnvCall = Exception::EnvCall as i64,
    FenceI = Exception::FenceI as i64,
    ForceFetchRun = Exception::ForceFetchRun as i64,
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
            Exception::FenceI => Self::FenceI,
            Exception::ForceFetchRun => Self::ForceFetchRun,
        }
    }

    /// Convert the exception code back to an [`Exception`] if it represents an exception.
    /// Otherwise it will return `None` for [`ExceptionCode::NoException`].
    pub fn to_exception(self) -> Option<Exception> {
        match self {
            Self::NoException => None,
            Self::InstructionAccessFault => Some(Exception::InstructionAccessFault),
            Self::IllegalInstruction => Some(Exception::IllegalInstruction),
            Self::Breakpoint => Some(Exception::Breakpoint),
            Self::LoadAccessFault => Some(Exception::LoadAccessFault),
            Self::StoreAMOAccessFault => Some(Exception::StoreAMOAccessFault),
            Self::EnvCall => Some(Exception::EnvCall),
            Self::FenceI => Some(Exception::FenceI),
            Self::ForceFetchRun => Some(Exception::ForceFetchRun),
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

/// Store the lowest `E::STORED_SIZE` bytes of the given value to memory, at the physical address.
///
/// Returns [`ExceptionCode::NoException`] if the store is successful.
///
/// If the store fails (due to out of bounds etc) then the appropriate exception code
/// is returned to indicate the type of failure that occurred.
extern "C" fn memory_store<E: Elem, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Normal>,
    address: u64,
    value: E,
) -> ExceptionCode {
    match core.main_memory.write(address, value) {
        Ok(()) => ExceptionCode::NoException,
        Err(BadMemoryAccess) => ExceptionCode::from_exception(Exception::StoreAMOAccessFault),
    }
}

/// Load `E::STORED_SIZE` bytes from memory, at the physical address, into lowest `E::STORED_SIZE`
/// bytes of an `XValue`, with (un)signed extension.
///
/// If the load is successful, the value is written to `xval_out` and
/// [`ExceptionCode::NoException`] is returned.
///
/// If the load fails (due to out of bounds etc) then the appropriate exception code
/// is returned to indicate the type of failure that occurred.
extern "C" fn memory_load<E: Elem, MC: MemoryConfig>(
    core: &MachineCoreState<MC, Normal>,
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
    core: &mut MachineCoreState<MC, Normal>,
    xval: XValue,
) -> FValue {
    MachineCoreState::f64_from_x64_unsigned_dynamic(core, xval)
}

/// Convert an unsigned 64-bit `XValue` to a 64-bit `FValue` using the given static rounding mode.
extern "C" fn f64_from_x64_unsigned_static<RM: StaticRoundingMode, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Normal>,
    xval: XValue,
) -> FValue {
    MachineCoreState::f64_from_x64_unsigned_static(core, xval, RM::ROUND)
}

/// Write to a Control and Status register.
extern "C" fn csr_write<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Normal>,
    csr: CSRegister,
    value: XValue,
) {
    core.hart.csregisters.write(csr, value);
}

/// Read from a Control and Status register.
extern "C" fn csr_read<MC: MemoryConfig>(
    core: &MachineCoreState<MC, Normal>,
    csr: CSRegister,
) -> XValue {
    core.hart.csregisters.read(csr)
}

/// Wrapper around [`run_code_page_interpreted`] for fallback to the interpreter
/// from the JIT context. Returns the steps remaining after interpreter execution.
extern "C" fn run_interpreter_fallback<D, MC: MemoryConfig>(
    page: &JittedPage<D, MC>,
    core: &mut MachineCoreState<MC, Normal>,
    max_steps: usize,
    result: &mut ExceptionCode,
) -> usize {
    let instr_pc = core.hart.pc.read();

    let interpreted_result = run_code_page_interpreted(&page.entries, core, instr_pc, max_steps);

    *result = interpreted_result
        .error
        .map(ExceptionCode::from_exception)
        .unwrap_or(ExceptionCode::NoException);

    max_steps - interpreted_result.steps
}

/// External function call registry for state accesses
pub struct JsaCalls<MC: MemoryConfig> {
    /// Target configuration which provides useful information about the target ISA, such as
    /// pointer type and width
    target_config: TargetFrontendConfig,

    _pd: PhantomData<MC>,
}

impl<MC: MemoryConfig> JsaCalls<MC> {
    /// Construct a new `JsaCalls` instance with the given target configuration.
    pub(super) fn new(target_config: TargetFrontendConfig) -> Self {
        Self {
            target_config,
            _pd: PhantomData,
        }
    }

    /// Emit the required IR to call `memory_store`.
    ///
    /// On success, the returned `ErrnoImpl` code is set to `NoException`.
    /// No additional values are returned.
    pub(super) fn memory_store<V: StoreLoadInt>(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
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
    /// On success, the returned `ErrnoImpl` code is set to `NoException`.
    /// The loaded value is returned as the `ErrnoImpl` value.
    pub(super) fn memory_load<V: StoreLoadInt>(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
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

    /// Emit the required IR to call `memory_store` for floating-point values.
    ///
    /// On success, the returned `ErrnoImpl` code is set to `NoException`.
    /// No additional values are returned.
    pub(super) fn memory_store_float<V: StoreLoadFloat>(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
        phys_address: Value<Address>,
        value: Value<FValue>,
    ) -> ErrnoImpl<(), impl FnOnce(&mut FunctionBuilder) + 'static> {
        let value = V::from_fvalue_ir(builder, value);

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

    /// Emit the required IR to call `memory_load` for floating-point values.
    ///
    /// On success, the returned `ErrnoImpl` code is set to `NoException`.
    /// The loaded value is returned as `FValue`.
    pub(super) fn memory_load_float<V: StoreLoadFloat>(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
        phys_address: Value<Address>,
    ) -> ErrnoImpl<Value<FValue>, impl FnOnce(&mut FunctionBuilder) -> Value<FValue> + 'static>
    {
        let fval_slot =
            stack::Slot::<MaybeUninit<V>>::new(self.target_config.pointer_type(), builder);
        let fval_ptr = fval_slot.ptr(builder);

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument
        // - `fval_ptr` points to a stack slot which is valid for the duration of the JIT function
        let exception = ext_calls::call3(
            &self.target_config,
            builder,
            self::memory_load,
            unsafe { core_ptr.as_ref() },
            phys_address,
            unsafe { fval_ptr.as_mut() },
        );

        ErrnoImpl::new(exception, move |builder| {
            // SAFETY: The slot is guaranteed to be initialised at this point as this closure
            // generates IR for the success case when the external function will have written to
            // the stack slot.
            let fval = unsafe { fval_slot.assume_init().load(builder) };

            V::to_fvalue_ir(builder, fval)
        })
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_dynamic`.
    ///
    /// Returns the converted FValue.
    pub(super) fn f64_from_x64_unsigned_dynamic(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
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
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
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
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
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
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
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

    /// Fallback to interpreter execution when JIT-compiled code cannot proceed.
    pub(super) fn run_interpreter_fallback<D>(
        &self,
        builder: &mut FunctionBuilder,
        page: Pointer<JittedPage<D, MC>>,
        core_ptr: Pointer<MachineCoreState<MC, Normal>>,
        max_steps: Value<usize>,
        result_ptr: Pointer<ExceptionCode>,
    ) -> Value<usize> {
        ext_calls::call4(
            &self.target_config,
            builder,
            self::run_interpreter_fallback::<D, MC>,
            unsafe { page.as_ref() },
            unsafe { core_ptr.as_mut() },
            max_steps,
            unsafe { result_ptr.as_mut() },
        )
    }
}
#[cfg(test)]
mod state_access_test_utils {
    use std::ffi::CStr;
    use std::ffi::c_char;
    use std::ptr::NonNull;

    use cranelift::prelude::FunctionBuilder;
    use octez_riscv_data::serialisation::elem::Elem;

    use crate::jit::builder::ext_calls;
    use crate::jit::builder::typed::Pointer;
    use crate::jit::builder::typed::Typed;
    use crate::jit::builder::typed::Value;
    use crate::jit::state_access::JsaCalls;
    use crate::machine_state::memory::Address;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::page_cache::address_to_halfword_index;
    use crate::machine_state::page_cache::jitted::JittedPage;

    extern "C" fn record_jit_call<D, MC: MemoryConfig>(
        page: &JittedPage<D, MC>,
        entrypoint: Address,
    ) {
        let entry = address_to_halfword_index(entrypoint);
        page.entries[entry].dispatch.jit_counters.record_jit_call();
    }

    extern "C" fn record_budget_check_pass<D, MC: MemoryConfig>(
        page: &JittedPage<D, MC>,
        entrypoint: Address,
    ) {
        let entry = address_to_halfword_index(entrypoint);
        page.entries[entry]
            .dispatch
            .jit_counters
            .record_budget_check_pass();
    }

    extern "C" fn record_fallback_to_interpreter<D, MC: MemoryConfig>(
        page: &JittedPage<D, MC>,
        entrypoint: Address,
    ) {
        let entry = address_to_halfword_index(entrypoint);
        page.entries[entry]
            .dispatch
            .jit_counters
            .record_fallback_to_interpreter();
    }

    extern "C" fn debug_print<E: Elem + std::fmt::Debug>(message: NonNull<c_char>, value: E) {
        // SAFETY: `message` is a non-null pointer to a NUL-terminated C string with static
        // storage duration, provided by the JIT helper.
        let message = unsafe { CStr::from_ptr(message.as_ptr()) };
        let s = message.to_string_lossy();
        eprintln!("[DEBUG]: {s} {value:?}");
    }

    impl<MC: MemoryConfig> JsaCalls<MC> {
        /// Record a JIT function call (test-only).
        pub(crate) fn record_jit_call<D>(
            &self,
            builder: &mut FunctionBuilder,
            page: Pointer<JittedPage<D, MC>>,
            entrypoint: Value<Address>,
        ) {
            ext_calls::call2(
                &self.target_config,
                builder,
                self::record_jit_call::<D, MC>,
                unsafe { page.as_ref() },
                entrypoint,
            );
        }

        /// Record that a budget check passed (test-only).
        pub(crate) fn record_budget_check_pass<D>(
            &self,
            builder: &mut FunctionBuilder,
            page: Pointer<JittedPage<D, MC>>,
            entrypoint: Value<Address>,
        ) {
            ext_calls::call2(
                &self.target_config,
                builder,
                self::record_budget_check_pass::<D, MC>,
                unsafe { page.as_ref() },
                entrypoint,
            );
        }

        /// Record a fallback to interpreted execution (test-only).
        pub(crate) fn record_fallback_to_interpreter<D>(
            &self,
            builder: &mut FunctionBuilder,
            page: Pointer<JittedPage<D, MC>>,
            entrypoint: Value<Address>,
        ) {
            ext_calls::call2(
                &self.target_config,
                builder,
                self::record_fallback_to_interpreter::<D, MC>,
                unsafe { page.as_ref() },
                entrypoint,
            );
        }

        /// Print a debug message in a JIT function with an additional possible value (test-only).
        ///
        /// Note: C Strings can be created using `c"Hello World!"` in Rust.
        pub(crate) fn debug<E: Elem + std::fmt::Debug + Typed>(
            &self,
            builder: &mut FunctionBuilder,
            msg: &'static CStr,
            value: Value<E>,
        ) {
            let msg = NonNull::new(msg.as_ptr().cast_mut())
                .expect("Debug message CStr pointer should be non-null");
            // SAFETY: The pointer is constructed from a reference to a valid CStr with a static lifetime.
            let ptr: Pointer<c_char> = unsafe {
                Pointer::from_discriminant(&self.target_config, builder, msg.addr().get() as i64)
            };
            ext_calls::call2(&self.target_config, builder, self::debug_print, ptr, value);
        }
    }
}
