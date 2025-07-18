// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! JIT-compiled blocks must be able to interact with the
//! RISC-V [`MachineCoreState`] passed to them.
//!
//! In Cranelift, this works in two stages.
//!
//! First, the `extern "C"` function pointers must be
//! registered as external symbols in the [jit builder] & the corresponding signatures declared
//! in the [jit module]. This allows generated code to link with these functions.
//!
//! The second step occurs _during jit compilation_ itself. The linked functions must be re-declared
//! within the [function builder] itself. This then allows for a [direct function call] to be issued,
//! which will indeed perform the function call at runtime.
//!
//! [jit builder]: JITBuilder
//! [jit module]: cranelift_jit::JITModule
//! [function builder]: cranelift::frontend::FunctionBuilderContext
//! [direct function call]: cranelift::codegen::ir::InstBuilder::call

mod abi;
pub(crate) mod stack;

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use abi::AbiCall;
use cranelift::codegen::ir::FuncRef;
use cranelift::codegen::ir::InstBuilder;
use cranelift::codegen::ir::Type;
use cranelift::codegen::ir::types::I8;
use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_module::FuncId;
use cranelift_module::Module;
use cranelift_module::ModuleResult;

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
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::XValue;
use crate::state_backend::Elem;
use crate::state_backend::owned_backend::Owned;
use crate::traps::EnvironException;
use crate::traps::Exception;

macro_rules! register_jsa_functions {
    ($($name:ident => ($field:path, $fn:path)),* $(,)?) => {
        /// Register state access symbols in the builder.
        pub(super) fn register_jsa_symbols<MC: MemoryConfig>(
            builder: &mut JITBuilder,
        ) {
            $(builder.symbol(stringify!($field), $field as *const u8);)*
        }

        /// Identifications of globally imported state access methods.
        pub(crate) struct JsaImports<MC: MemoryConfig> {
            $(
                pub $name: FuncId,
            )*
            _pd: PhantomData<MC>,
        }

        impl<MC: MemoryConfig> JsaImports<MC> {
            /// Register external functions within the JIT Module.
            pub(super) fn declare_in_module(module: &mut JITModule) -> ModuleResult<Self> {
                let ptr_type = module.target_config().pointer_type();
                let call_conv = module.target_config().default_call_conv;

                $(
                    let abi = $fn($field);
                    let $name = abi.declare_function(module, stringify!($field), ptr_type, call_conv)?;
                )*

                Ok(Self {
                    $(
                        $name,
                    )*
                    _pd: PhantomData,
                })
            }
        }
    };
}

register_jsa_functions!(
    freg_read => (fregister_read::<MC>, AbiCall<2>::args),
    freg_write => (fregister_write::<MC>, AbiCall<3>::args),
    f64_from_x64_unsigned_dynamic => (
        f64_from_x64_unsigned_dynamic::<MC>,
        AbiCall<4>::args
    ),
    f64_from_x64_unsigned_static_rne => (
        f64_from_x64_unsigned_static::<RoundRNE, MC>,
        AbiCall<2>::args
    ),
    f64_from_x64_unsigned_static_rtz => (
        f64_from_x64_unsigned_static::<RoundRTZ, MC>,
        AbiCall<2>::args
    ),
    f64_from_x64_unsigned_static_rup => (
        f64_from_x64_unsigned_static::<RoundRUP, MC>,
        AbiCall<2>::args
    ),
    f64_from_x64_unsigned_static_rdn => (
        f64_from_x64_unsigned_static::<RoundRDN, MC>,
        AbiCall<2>::args
    ),
    f64_from_x64_unsigned_static_rmm => (
        f64_from_x64_unsigned_static::<RoundRMM, MC>,
        AbiCall<2>::args
    ),
);

/// Read the value of the given [`FRegister`].
extern "C" fn fregister_read<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
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

/// References to locally imported state access methods, used to directly call these accessor
/// methods in the JIT-compilation context.
pub struct JsaCalls<'a, MC: MemoryConfig> {
    /// Target configuration which provides useful information about the target ISA, such as
    /// pointer type and width
    target_config: TargetFrontendConfig,

    module: &'a mut JITModule,
    imports: &'a JsaImports<MC>,
    ptr_type: Type,
    freg_read: Option<FuncRef>,
    freg_write: Option<FuncRef>,
    f64_from_x64_unsigned_dynamic: Option<FuncRef>,
    f64_from_x64_unsigned_static: Option<FuncRef>,

    /// Reusable stack slot for the exception pointer
    exception_ptr_slot: Option<stack::Slot<MaybeUninit<Exception>>>,

    /// Reusable stack slot for the PC value
    pc_slot: Option<stack::Slot<MaybeUninit<Address>>>,

    /// Reusable stack slot for an FValue.
    fvalue_ptr_slot: Option<stack::Slot<MaybeUninit<FValue>>>,

    _pd: PhantomData<MC>,
}

impl<'a, MC: MemoryConfig> JsaCalls<'a, MC> {
    /// Get the stack slot for the exception pointer.
    fn exception_ptr_slot(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> stack::Slot<MaybeUninit<Exception>> {
        self.exception_ptr_slot
            .get_or_insert_with(|| stack::Slot::new(self.ptr_type, builder))
            .clone()
    }

    /// Get the stack slot for the PC value.
    fn pc_slot(&mut self, builder: &mut FunctionBuilder) -> stack::Slot<MaybeUninit<Address>> {
        self.pc_slot
            .get_or_insert_with(|| stack::Slot::new(self.ptr_type, builder))
            .clone()
    }

    /// Get the stack slot for an FValue.
    fn fvalue_ptr_slot(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> stack::Slot<MaybeUninit<FValue>> {
        self.fvalue_ptr_slot
            .get_or_insert_with(|| stack::Slot::new(self.ptr_type, builder))
            .clone()
    }

    /// Wrapper to simplify calling state access functions from within the function under construction.
    pub(super) fn func_calls(
        module: &'a mut JITModule,
        imports: &'a JsaImports<MC>,
        ptr_type: Type,
    ) -> Self {
        Self {
            target_config: module.target_config(),
            module,
            imports,
            ptr_type,
            freg_read: None,
            freg_write: None,
            f64_from_x64_unsigned_dynamic: None,
            f64_from_x64_unsigned_static: None,
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

        let xval_slot = stack::Slot::<MaybeUninit<V>>::new(self.ptr_type, builder);
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

        let new_f64_from_x64_unsigned_dynamic =
            self.f64_from_x64_unsigned_dynamic.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.f64_from_x64_unsigned_dynamic, builder.func)
            });

        let call = builder.ins().call(*new_f64_from_x64_unsigned_dynamic, &[
            core_ptr.to_value(),
            exception_ptr.to_value(),
            xval.to_value(),
            fvalue_ptr.to_value(),
        ]);

        // SAFETY: [`self::f64_from_x64_unsigned_dynamic`] returns a `bool`.
        let is_exception = unsafe { Value::<bool>::from_raw(builder.inst_results(call)[0]) };

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
        let new_f64_from_x64_unsigned_static =
            self.f64_from_x64_unsigned_static.get_or_insert_with(|| {
                let rm_func_id = match rm {
                    RoundingMode::RNE => self.imports.f64_from_x64_unsigned_static_rne,
                    RoundingMode::RTZ => self.imports.f64_from_x64_unsigned_static_rtz,
                    RoundingMode::RUP => self.imports.f64_from_x64_unsigned_static_rup,
                    RoundingMode::RDN => self.imports.f64_from_x64_unsigned_static_rdn,
                    RoundingMode::RMM => self.imports.f64_from_x64_unsigned_static_rmm,
                };
                self.module.declare_func_in_func(rm_func_id, builder.func)
            });

        let call = builder.ins().call(*new_f64_from_x64_unsigned_static, &[
            core_ptr.to_value(),
            xval.to_value(),
        ]);

        // SAFETY: [`self::f64_from_x64_unsigned_static`] returns a `FValue`.
        unsafe { Value::<FValue>::from_raw(builder.inst_results(call)[0]) }
    }

    /// Emit the required IR to read the value from the given fregister.
    pub(super) fn ir_freg_read(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        reg: FRegister,
    ) -> Value<FValue> {
        let freg_read = self.freg_read.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.freg_read, builder.func)
        });
        let reg = builder.ins().iconst(I8, reg as i64);
        let call = builder.ins().call(*freg_read, &[core_ptr.to_value(), reg]);

        // SAFETY: [`self::fregister_read`] returns a `FValue`.
        unsafe { Value::<FValue>::from_raw(builder.inst_results(call)[0]) }
    }

    /// Emit the required IR to write the value to the given fregister.
    pub(super) fn ir_freg_write(
        &mut self,
        builder: &mut FunctionBuilder,
        core_ptr: Pointer<MachineCoreState<MC, Owned>>,
        reg: FRegister,
        value: Value<FValue>,
    ) {
        let freg_write = self.freg_write.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.freg_write, builder.func)
        });
        let reg = builder.ins().iconst(I8, reg as i64);
        builder
            .ins()
            .call(*freg_write, &[core_ptr.to_value(), reg, value.to_value()]);
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
