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
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift::prelude::types::I64;
use cranelift_jit::JITModule;

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
use crate::jit::builder::data::define_function_data;
use crate::jit::builder::ext_calls;
use crate::jit::builder::typed::FunctionPointer;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Type;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::instruction::Args;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::instruction::RunInstr;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::BadMemoryAccess;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FRegister;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::XValue;
use crate::state_backend::Elem;
use crate::state_backend::owned_backend::Owned;
use crate::traps::EnvironException;
use crate::traps::Exception;

/// Read the value of the given [`FRegister`].
extern "C" fn fregister_read<MC: MemoryConfig>(
    core: &MachineCoreState<MC, Owned>,
    reg: FRegister,
) -> FValue {
    core.hart.fregisters.read(reg)
}

/// Write the given value to the given [`FRegister`].
extern "C" fn fregister_write<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    reg: FRegister,
    val: FValue,
) {
    core.hart.fregisters.write(reg, val)
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
extern "C" fn raise_store_amo_access_fault_exception(
    exception_out: &mut MaybeUninit<Exception>,
    address: u64,
) {
    exception_out.write(Exception::StoreAMOAccessFault(address));
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
            exception_out.write(Exception::StoreAMOAccessFault(address));
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
            exception_out.write(Exception::LoadAccessFault(address));
            true
        }
    }
}

extern "C" fn f64_from_x64_unsigned_dynamic<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    exception_out: &mut MaybeUninit<Exception>,
    xval: XValue,
    fvalue_out: &mut MaybeUninit<FValue>,
) -> bool {
    match MachineCoreState::f64_from_x64_unsigned_dynamic(core, xval) {
        Ok(fval) => {
            fvalue_out.write(fval);
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
) -> FValue {
    MachineCoreState::f64_from_x64_unsigned_static(core, xval, RM::ROUND)
}

/// Result of running an instruction
#[repr(i64)]
pub enum RunInstructionResult {
    /// Run the next instruction
    Next = 0,

    /// Jump to an unknown branch
    UnknownBranch = 1,

    /// Running the instruction resulted in an environment exception
    EnvironException = -1,
}

impl Typed for RunInstructionResult {
    const TYPE: Type = Type::Basic(I64);
}

/// Run the given instruction using the provided runner function pointer.
extern "C" fn run_instruction<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    runner: FunctionPointer<RunInstr<MC, Owned>>,
    args: &Args,
    current_pc: &mut Address,
    result: &mut Result<(), EnvironException>,
) -> RunInstructionResult {
    // SAFETY: The `runner` is a valid function pointer.
    let runner = unsafe { runner.to_inner() };

    // We need to write the current program counter to the state because the JIT function
    // only does it at its exit. If we wouldn't do this instructions which operate on the
    // program counter would use the one from the beggining of the JIT function.
    core.hart.pc.write(*current_pc);

    match (runner)(args, core) {
        Ok(ProgramCounterUpdate::Set(abs)) => {
            *current_pc = abs;
            RunInstructionResult::UnknownBranch
        }

        Ok(ProgramCounterUpdate::Relative(offset)) => {
            *current_pc = current_pc.wrapping_add_signed(offset);
            RunInstructionResult::UnknownBranch
        }

        Ok(ProgramCounterUpdate::Next(_width)) => RunInstructionResult::Next,

        Err(exception) => match core.address_on_exception(exception, *current_pc) {
            Ok(abs) => {
                *current_pc = abs;
                RunInstructionResult::UnknownBranch
            }

            Err(env_exception) => {
                *result = Err(env_exception);
                RunInstructionResult::EnvironException
            }
        },
    }
}

/// External function call registry for state accesses
pub struct JsaCalls<MC: MemoryConfig> {
    /// Target configuration which provides useful information about the target ISA, such as
    /// pointer type and width
    target_config: TargetFrontendConfig,

    /// Reusable stack slot for the exception pointer
    exception_ptr_slot: Option<stack::Slot<MaybeUninit<Exception>>>,

    /// Reusable stack slot for the PC value
    pc_slot: Option<stack::Slot<MaybeUninit<Address>>>,

    /// Reusable stack slot for an FValue.
    fvalue_ptr_slot: Option<stack::Slot<MaybeUninit<FValue>>>,

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
    fn fvalue_ptr_slot(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> stack::Slot<MaybeUninit<FValue>> {
        self.fvalue_ptr_slot
            .get_or_insert_with(|| stack::Slot::new(self.target_config.pointer_type(), builder))
            .clone()
    }

    /// Construct a new `JsaCalls` instance with the given target configuration.
    pub(super) fn new(target_config: TargetFrontendConfig) -> Self {
        Self {
            target_config,
            exception_ptr_slot: None,
            pc_slot: None,
            fvalue_ptr_slot: None,
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
        address: Value<Address>,
    ) -> Pointer<Exception> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        // SAFETY: The exception reference is guaranteed to be valid for the duration of the call as
        // it is scoped to the JIT function.
        ext_calls::call2(
            &self.target_config,
            builder,
            self::raise_store_amo_access_fault_exception,
            unsafe { exception_ptr.as_mut() },
            address,
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
    /// Returns `errno` - on success, the new FValue is returned.
    pub(super) fn f64_from_x64_unsigned_dynamic(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        xval: Value<XValue>,
    ) -> ErrnoImpl<Value<FValue>, impl FnOnce(&mut FunctionBuilder) -> Value<FValue> + 'static>
    {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let fvalue_slot = self.fvalue_ptr_slot(builder);
        let fvalue_ptr = fvalue_slot.ptr(builder);

        // SAFETY: The reference argument lifetimes are valid for the duration of the call:
        // - `core_ptr` is a JIT function argument, therefore valid for the entire function
        // - `exception_ptr` points to a stack slot which is valid for the duration of the function
        // - `fvalue_ptr` also points to a stack slot
        let is_exception = ext_calls::call4(
            &self.target_config,
            builder,
            self::f64_from_x64_unsigned_dynamic,
            unsafe { core_ptr.as_mut() },
            unsafe { exception_ptr.as_mut() },
            xval,
            unsafe { fvalue_ptr.as_mut() },
        );

        ErrnoImpl::new(is_exception, exception_ptr, move |builder| {
            // SAFETY: This closure runs after the success case of the call, where the fvalue_slot
            // is guaranteed to have been initialised with an fvalue.
            unsafe { fvalue_slot.assume_init().load(builder) }
        })
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

    /// Emit the required IR to read the value from the given fregister.
    pub(super) fn ir_freg_read(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        reg: FRegister,
    ) -> Value<FValue> {
        // SAFETY: We construct the typed value from the `FRegister`, thereby ensuring the use of
        // the right discriminant.
        let reg_value = unsafe {
            Value::<FRegister>::from_discriminant(&self.target_config, builder, reg as u8 as i64)
        };

        // SAFETY: The `core_ptr` is a JIT function argument which means the reference through it
        // will be valid for the duration of the call.
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
        value: Value<FValue>,
    ) {
        // SAFETY: We construct the typed value from the `FRegister`, thereby ensuring the use of
        // the right discriminant.
        let reg_value = unsafe {
            Value::<FRegister>::from_discriminant(&self.target_config, builder, reg as u8 as i64)
        };

        // SAFETY: The `core_ptr` is a JIT function argument which means the reference through it
        // will be valid for the duration of the call.
        ext_calls::call3(
            &self.target_config,
            builder,
            self::fregister_write,
            unsafe { core_ptr.as_mut() },
            reg_value,
            value,
        );
    }

    /// Run the given instruction, returning the new program counter and the result of the
    /// instruction.
    pub(super) fn run_instruction(
        &mut self,
        module: &mut JITModule,
        builder: &mut FunctionBuilder,
        core_param: Pointer<MachineCoreState<MC, Owned>>,
        result_param: Pointer<Result<(), EnvironException>>,
        current_pc: Value<Address>,
        instr: &Instruction,
    ) -> (Value<Address>, Value<RunInstructionResult>) {
        let args_ptr = define_function_data(module, builder, &instr.args).unwrap();

        let runner_value = {
            let runner = instr.opcode.to_run::<MC, Owned>();
            let raw_value = builder.ins().iconst(I64, runner as usize as i64);
            unsafe { Value::<FunctionPointer<RunInstr<MC, Owned>>>::from_raw(raw_value) }
        };

        let pc_slot = self.pc_slot(builder);
        let pc_slot = pc_slot.init(builder, current_pc);
        let pc_slot_ptr = pc_slot.ptr(builder);

        let result = ext_calls::call5(
            &self.target_config,
            builder,
            run_instruction,
            unsafe { core_param.as_mut() },
            runner_value,
            unsafe { args_ptr.as_ref() },
            unsafe { pc_slot_ptr.as_mut() },
            unsafe { result_param.as_mut() },
        );

        let pc_value = pc_slot.load(builder);

        (pc_value, result)
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
