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
use cranelift::codegen::ir;
use cranelift::codegen::ir::FuncRef;
use cranelift::codegen::ir::InstBuilder;
use cranelift::codegen::ir::Type;
use cranelift::codegen::ir::Value;
use cranelift::codegen::ir::types::I8;
use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::MemFlags;
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_module::FuncId;
use cranelift_module::Module;
use cranelift_module::ModuleResult;

use super::builder::F64;
use super::builder::X64;
use super::builder::errno::ErrnoImpl;
use crate::instruction_context::ICB;
use crate::instruction_context::LoadStoreWidth;
use crate::instruction_context::StoreLoadInt;
use crate::interpreter::float::RoundRDN;
use crate::interpreter::float::RoundRMM;
use crate::interpreter::float::RoundRNE;
use crate::interpreter::float::RoundRTZ;
use crate::interpreter::float::RoundRUP;
use crate::interpreter::float::RoundingMode;
use crate::interpreter::float::StaticRoundingMode;
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
    pc_write => (pc_write::<MC>, AbiCall<2>::args),
    freg_read => (fregister_read::<MC>, AbiCall<2>::args),
    freg_write => (fregister_write::<MC>, AbiCall<3>::args),
    handle_exception => (handle_exception::<MC>, AbiCall<4>::args),
    raise_illegal_instruction_exception => (raise_illegal_instruction_exception, AbiCall<1>::args),
    raise_store_amo_access_fault_exception => (raise_store_amo_access_fault_exception, AbiCall<2>::args),
    ecall_from_mode => (ecall::<MC>, AbiCall<2>::args),
    memory_store_u8 => (memory_store::<u8, MC>, AbiCall<4>::args),
    memory_store_u16 => (memory_store::<u16, MC>, AbiCall<4>::args),
    memory_store_u32 => (memory_store::<u32, MC>, AbiCall<4>::args),
    memory_store_u64 => (memory_store::<u64, MC>, AbiCall<4>::args),
    memory_load_i8 => (memory_load::<i8, MC>, AbiCall<4>::args),
    memory_load_u8 => (memory_load::<u8, MC>, AbiCall<4>::args),
    memory_load_i16 => (memory_load::<i16, MC>, AbiCall<4>::args),
    memory_load_u16 => (memory_load::<u16, MC>, AbiCall<4>::args),
    memory_load_i32 => (memory_load::<i32, MC>, AbiCall<4>::args),
    memory_load_u32 => (memory_load::<u32, MC>, AbiCall<4>::args),
    memory_load_i64 => (memory_load::<i64, MC>, AbiCall<4>::args),
    memory_load_u64 => (memory_load::<u64, MC>, AbiCall<4>::args),
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
    reservation_set_write => (reservation_set_write::<MC>, AbiCall<2>::args),
    reservation_set_read => (reservation_set_read::<MC>, AbiCall<1>::args),
);

/// Update the instruction pc in the state.
extern "C" fn pc_write<MC: MemoryConfig>(core: &mut MachineCoreState<MC, Owned>, pc: u64) {
    core.hart.pc.write(pc)
}

/// Read the value of the given [`FRegister`].
extern "C" fn fregister_read<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
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
extern "C" fn ecall<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    exception_out: &mut MaybeUninit<Exception>,
) {
    exception_out.write(core.hart.run_ecall());
}

/// Store the lowest `width` bytes of the given value to memory, at the physical address.
///
/// If the store is successful, `false` is returned to indicate no exception handling is necessary.
///
/// If the store fails (due to out of bouds etc) then an exception will be written
/// to `exception_out` and `true` returned to indicate exception handling will be necessary.
///
/// # Panics
///
/// Panics if the `width` passed is not a supported [`LoadStoreWidth`].
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
///
/// # Panics
///
/// Panics if the `width` passed is not a supported [`LoadStoreWidth`].
extern "C" fn memory_load<E: Elem, MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
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

/// Set the reservation set to the given starting address.
extern "C" fn reservation_set_write<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
    address: u64,
) {
    MachineCoreState::reservation_set_write(core, address);
}

/// Read the reservation set starting address.
extern "C" fn reservation_set_read<MC: MemoryConfig>(
    core: &mut MachineCoreState<MC, Owned>,
) -> u64 {
    MachineCoreState::reservation_set_read(core)
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
pub struct JsaCalls<'a, MC: MemoryConfig> {
    module: &'a mut JITModule,
    imports: &'a JsaImports<MC>,
    ptr_type: Type,
    pc_write: Option<FuncRef>,
    freg_read: Option<FuncRef>,
    freg_write: Option<FuncRef>,
    handle_exception: Option<FuncRef>,
    raise_illegal_instruction_exception: Option<FuncRef>,
    raise_store_amo_access_fault_exception: Option<FuncRef>,
    ecall_from_mode: Option<FuncRef>,
    memory_store_u8: Option<FuncRef>,
    memory_store_u16: Option<FuncRef>,
    memory_store_u32: Option<FuncRef>,
    memory_store_u64: Option<FuncRef>,
    memory_load_i8: Option<FuncRef>,
    memory_load_u8: Option<FuncRef>,
    memory_load_i16: Option<FuncRef>,
    memory_load_u16: Option<FuncRef>,
    memory_load_i32: Option<FuncRef>,
    memory_load_u32: Option<FuncRef>,
    memory_load_i64: Option<FuncRef>,
    memory_load_u64: Option<FuncRef>,
    reservation_set_write: Option<FuncRef>,
    reservation_set_read: Option<FuncRef>,
    f64_from_x64_unsigned_dynamic: Option<FuncRef>,
    f64_from_x64_unsigned_static: Option<FuncRef>,

    /// Reusable stack slot for the exception pointer
    exception_ptr_slot: Option<stack::Slot<Exception>>,

    /// Reusable stack slot for the PC value
    pc_slot: Option<stack::Slot<stack::Address>>,

    // Reusable stack slot for an FValue.
    f64_ptr_slot: Option<stack::Slot<f64>>,
    _pd: PhantomData<MC>,
}

impl<'a, MC: MemoryConfig> JsaCalls<'a, MC> {
    /// Get the stack slot for the exception pointer.
    fn exception_ptr_slot(&mut self, builder: &mut FunctionBuilder<'_>) -> stack::Slot<Exception> {
        self.exception_ptr_slot
            .get_or_insert_with(|| stack::Slot::<Exception>::new(self.ptr_type, builder))
            .clone()
    }

    /// Get the stack slot for the PC value.
    fn pc_slot(&mut self, builder: &mut FunctionBuilder<'_>) -> stack::Slot<stack::Address> {
        self.pc_slot
            .get_or_insert_with(|| stack::Slot::<stack::Address>::new(self.ptr_type, builder))
            .clone()
    }

    /// Get the stack slot for an FValue.
    fn f64_ptr_slot(&mut self, builder: &mut FunctionBuilder<'_>) -> stack::Slot<f64> {
        self.f64_ptr_slot
            .get_or_insert_with(|| stack::Slot::<f64>::new(self.ptr_type, builder))
            .clone()
    }

    /// Wrapper to simplify calling state access functions from within the function under construction.
    pub(super) fn func_calls(
        module: &'a mut JITModule,
        imports: &'a JsaImports<MC>,
        ptr_type: Type,
    ) -> Self {
        Self {
            module,
            imports,
            ptr_type,
            pc_write: None,
            freg_read: None,
            freg_write: None,
            handle_exception: None,
            raise_illegal_instruction_exception: None,
            raise_store_amo_access_fault_exception: None,
            ecall_from_mode: None,
            memory_store_u8: None,
            memory_store_u16: None,
            memory_store_u32: None,
            memory_store_u64: None,
            memory_load_i8: None,
            memory_load_u8: None,
            memory_load_i16: None,
            memory_load_u16: None,
            memory_load_i32: None,
            memory_load_u32: None,
            memory_load_i64: None,
            memory_load_u64: None,
            reservation_set_write: None,
            reservation_set_read: None,
            f64_from_x64_unsigned_dynamic: None,
            f64_from_x64_unsigned_static: None,
            exception_ptr_slot: None,
            pc_slot: None,
            f64_ptr_slot: None,
            _pd: PhantomData,
        }
    }

    /// Emit the required IR to set the pc to the given value.
    pub(super) fn pc_write(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        pc_val: X64,
    ) {
        let pc_write = self.pc_write.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.pc_write, builder.func)
        });
        builder.ins().call(*pc_write, &[core_ptr, pc_val.0]);
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
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        exception_ptr: Value,
        result_ptr: Value,
        current_pc: X64,
    ) -> ExceptionHandledOutcome {
        let pc_slot = self.pc_slot(builder);
        pc_slot.store(builder, current_pc.0);
        let pc_ptr = pc_slot.ptr(builder);

        let handle_exception = self.handle_exception.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.handle_exception, builder.func)
        });

        let call = builder.ins().call(*handle_exception, &[
            core_ptr,
            pc_ptr,
            exception_ptr,
            result_ptr,
        ]);

        let handled = builder.inst_results(call)[0];
        // SAFETY: the pc is initialised prior to the call, and is guaranteed to
        // remain initialised regardless of the result of external call.
        let new_pc = unsafe { pc_slot.load(builder) };

        ExceptionHandledOutcome {
            handled,
            new_pc: X64(new_pc),
        }
    }

    /// Emit the required IR to call `raise_illegal_exception`.
    ///
    /// This returns an initialised pointer to the exception.
    pub(super) fn raise_illegal_instruction_exception(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
    ) -> Value {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let raise_illegal = self
            .raise_illegal_instruction_exception
            .get_or_insert_with(|| {
                self.module.declare_func_in_func(
                    self.imports.raise_illegal_instruction_exception,
                    builder.func,
                )
            });

        builder.ins().call(*raise_illegal, &[exception_ptr]);

        exception_ptr
    }

    /// Emit the required IR to call `raise_store_amo_access_fault_exception`.
    ///
    /// This returns an initialised pointer to the exception.
    pub(super) fn raise_store_amo_access_fault_exception(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        address: Value,
    ) -> Value {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let raise_store_amo_access_fault = self
            .raise_store_amo_access_fault_exception
            .get_or_insert_with(|| {
                self.module.declare_func_in_func(
                    self.imports.raise_store_amo_access_fault_exception,
                    builder.func,
                )
            });

        builder
            .ins()
            .call(*raise_store_amo_access_fault, &[exception_ptr, address]);

        exception_ptr
    }

    /// Emit the required IR to call `ecall`.
    ///
    /// This returns an initialised pointer to the appropriate environment
    /// call exception for the current machine mode.
    pub(super) fn ecall(&mut self, builder: &mut FunctionBuilder<'_>, core_ptr: Value) -> Value {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let ecall_from_mode = self.ecall_from_mode.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.ecall_from_mode, builder.func)
        });

        builder
            .ins()
            .call(*ecall_from_mode, &[core_ptr, exception_ptr]);

        exception_ptr
    }

    /// Emit the required IR to call `memory_store`.
    ///
    /// Returns `errno` - on success, no additional values are returned.
    pub(super) fn memory_store<V: StoreLoadInt>(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        phys_address: X64,
        value: X64,
    ) -> ErrnoImpl<(), impl FnOnce(&mut FunctionBuilder<'_>) + 'static> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let memory_store = match V::WIDTH {
            LoadStoreWidth::Byte => self.memory_store_u8.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_store_u8, builder.func)
            }),
            LoadStoreWidth::Half => self.memory_store_u16.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_store_u16, builder.func)
            }),
            LoadStoreWidth::Word => self.memory_store_u32.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_store_u32, builder.func)
            }),
            LoadStoreWidth::Double => self.memory_store_u64.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_store_u64, builder.func)
            }),
        };

        let value = match V::WIDTH {
            LoadStoreWidth::Byte | LoadStoreWidth::Half | LoadStoreWidth::Word => {
                builder.ins().ireduce(V::IR_TYPE, value.0)
            }
            LoadStoreWidth::Double => value.0,
        };

        let call = builder.ins().call(*memory_store, &[
            core_ptr,
            phys_address.0,
            value,
            exception_ptr,
        ]);

        let errno = builder.inst_results(call)[0];

        ErrnoImpl::new(errno, exception_ptr, |_| {})
    }

    /// Emit the required IR to call `memory_load`.
    ///
    /// Returns `errno` - on success, the loaded value is returned.
    pub(super) fn memory_load<V: StoreLoadInt>(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        phys_address: X64,
    ) -> ErrnoImpl<X64, impl FnOnce(&mut FunctionBuilder<'_>) -> X64 + 'static> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let xval_ptr = stack::Slot::<V>::new(self.ptr_type, builder).ptr(builder);

        let memory_load = match (V::WIDTH, V::SIGNED) {
            (LoadStoreWidth::Byte, true) => self.memory_load_i8.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_i8, builder.func)
            }),
            (LoadStoreWidth::Byte, false) => self.memory_load_u8.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_u8, builder.func)
            }),
            (LoadStoreWidth::Half, true) => self.memory_load_i16.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_i16, builder.func)
            }),
            (LoadStoreWidth::Half, false) => self.memory_load_u16.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_u16, builder.func)
            }),
            (LoadStoreWidth::Word, true) => self.memory_load_i32.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_i32, builder.func)
            }),
            (LoadStoreWidth::Word, false) => self.memory_load_u32.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_u32, builder.func)
            }),
            (LoadStoreWidth::Double, true) => self.memory_load_i64.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_i64, builder.func)
            }),
            (LoadStoreWidth::Double, false) => self.memory_load_u64.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.memory_load_u64, builder.func)
            }),
        };

        let call = builder.ins().call(*memory_load, &[
            core_ptr,
            phys_address.0,
            xval_ptr,
            exception_ptr,
        ]);

        let errno = builder.inst_results(call)[0];

        ErrnoImpl::new(errno, exception_ptr, move |builder| {
            let xval = builder
                .ins()
                .load(V::IR_TYPE, MemFlags::trusted(), xval_ptr, 0);

            let xval = if V::IR_TYPE == ir::types::I64 {
                xval
            } else if V::SIGNED {
                builder.ins().sextend(ir::types::I64, xval)
            } else {
                builder.ins().uextend(ir::types::I64, xval)
            };

            X64(xval)
        })
    }

    /// Emit the required IR to call `set_reservation_set`.
    ///
    /// Sets the reservation set to the given address.
    pub(super) fn reservation_set_write(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        address: X64,
    ) {
        let reservation_set_write = self.reservation_set_write.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.reservation_set_write, builder.func)
        });

        builder
            .ins()
            .call(*reservation_set_write, &[core_ptr, address.0]);
    }

    /// Emit the required IR to call `reservation_set_read`.
    ///
    /// Reads the reservation set starting address.
    pub(super) fn reservation_set_read(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
    ) -> X64 {
        let reservation_set_read = self.reservation_set_read.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.reservation_set_read, builder.func)
        });

        let call = builder.ins().call(*reservation_set_read, &[core_ptr]);
        X64(builder.inst_results(call)[0])
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_dynamic`.
    ///
    /// Returns `errno` - on success, the new F64 value is returned.
    pub(super) fn f64_from_x64_unsigned_dynamic(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        xval: X64,
    ) -> ErrnoImpl<F64, impl FnOnce(&mut FunctionBuilder<'_>) -> F64 + 'static> {
        let exception_slot = self.exception_ptr_slot(builder);
        let exception_ptr = exception_slot.ptr(builder);

        let f64_slot = self.f64_ptr_slot(builder);
        let f64_ptr = f64_slot.ptr(builder);

        let new_f64_from_x64_unsigned_dynamic =
            self.f64_from_x64_unsigned_dynamic.get_or_insert_with(|| {
                self.module
                    .declare_func_in_func(self.imports.f64_from_x64_unsigned_dynamic, builder.func)
            });

        let call = builder.ins().call(*new_f64_from_x64_unsigned_dynamic, &[
            core_ptr,
            exception_ptr,
            xval.0,
            f64_ptr,
        ]);

        let errno = builder.inst_results(call)[0];
        ErrnoImpl::new(errno, exception_ptr, move |builder| {
            // SAFETY: This closure runs after the success case of the call, where the f64_slot
            // is guaranteed to have been initialised with an f64 value.
            let fval = unsafe { f64_slot.load(builder) };
            F64(fval)
        })
    }

    /// Emit the required IR to call `f64_from_x64_unsigned_static`.
    /// The converted value is returned as `F64`.
    pub(super) fn f64_from_x64_unsigned_static(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        xval: X64,
        rm: RoundingMode,
    ) -> F64 {
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

        let call = builder
            .ins()
            .call(*new_f64_from_x64_unsigned_static, &[core_ptr, xval.0]);

        let fval = builder.inst_results(call)[0];
        F64(fval)
    }

    /// Emit the required IR to read the value from the given fregister.
    pub(super) fn ir_freg_read(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        reg: FRegister,
    ) -> F64 {
        let freg_read = self.freg_read.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.freg_read, builder.func)
        });
        let reg = builder.ins().iconst(I8, reg as i64);
        let call = builder.ins().call(*freg_read, &[core_ptr, reg]);
        F64(builder.inst_results(call)[0])
    }

    /// Emit the required IR to write the value to the given fregister.
    pub(super) fn ir_freg_write(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        core_ptr: Value,
        reg: FRegister,
        value: F64,
    ) {
        let freg_write = self.freg_write.get_or_insert_with(|| {
            self.module
                .declare_func_in_func(self.imports.freg_write, builder.func)
        });
        let reg = builder.ins().iconst(I8, reg as i64);
        builder.ins().call(*freg_write, &[core_ptr, reg, value.0]);
    }
}

/// Outcome of handling an exception.
pub(super) struct ExceptionHandledOutcome {
    /// Whether the exception was succesfully handled.
    ///
    /// - If true, the exception was handled and the step is completed.
    /// - If false, the exception must be instead handled by the environment.
    ///   The step is not complete.
    pub handled: Value,
    /// The new value of the instruction pc, after exception handling.
    pub new_pc: X64,
}
