// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Builder for individual instructions
//!
//! This module provides the infrastructure for compiling individual RISC-V instructions
//! into Cranelift IR within a JIT compilation context. It handles instruction-level
//! control flow, exception handling, and integration with the broader [sequence builder].
//!
//! The lifecycle of an instruction build begins when the [sequence builder] creates an
//! [instruction builder] through [`build_next_instruction`]. During
//! IR generation, the instruction implementation uses [`ICB`] methods to produce Cranelift
//! IR that represents the instruction's behavior, while the builder automatically tracks
//! all possible execution outcomes. Once the instruction logic is complete,
//! [`InstructionBuilder::finish`] converts the builder into a [`LoweredInstruction`] with the
//! [outcomes] properly connected at their source, allowing the [sequence builder] to integrate
//! them into the overall sequence control flow. Any exceptions raised during instruction execution
//! are linked to a common exception block and handled separately.
//!
//! [sequence builder]: super::sequence::SequenceBuilder
//! [`build_next_instruction`]: super::sequence::SequenceBuilder::build_next_instruction
//! [instruction builder]: InstructionBuilder
//! [outcomes]: InstructionOutcomes

use cranelift::codegen::ir::BlockArg;
use cranelift::prelude::Block;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift::prelude::types::I32;
use cranelift::prelude::types::I64;
use cranelift::prelude::types::I128;
use octez_riscv_data::mode::Normal;

use crate::exceptions::Exception;
use crate::instruction_context::ICB;
use crate::instruction_context::MulHighType;
use crate::instruction_context::StoreLoadFloat;
use crate::instruction_context::StoreLoadInt;
use crate::instruction_context::value::PhiValue;
use crate::interpreter::float::RoundingMode;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;
use crate::jit::state_access::ExceptionCode;
use crate::jit::state_access::JsaCalls;
use crate::machine_state::MachineCoreState;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FRegister;
use crate::machine_state::registers::FRegisterProj;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::NonZeroXRegister;
use crate::machine_state::registers::XRegisterProj;
use crate::machine_state::registers::XValue;
use crate::machine_state::registers::XValue32;
use crate::machine_state::reservation_set::ReservationSetProj;
use crate::parser::instruction::InstrWidth;
use crate::state_context::StateContext;
use crate::state_context::projection::MachineCoreProjection;

/// Probability of taking an outcome of a particular instruction.
///
/// As detailed in RISC-V Control Transfer Instructions specification (2.5),
/// backward-branches should be treated as likely taken, while forward-branches
/// should be treated as likely not-taken. Also, exception handlers should be treated
/// as likely not-taken (except for a few instructions, such as `ECall`, which are guaranteed to
/// result in an exception).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutcomeProbability {
    High,
    Low,
}

/// The resulting control flow information after executing an instruction.
#[derive(Debug)]
pub enum InstructionOutcomes {
    /// Continue execution
    Next {
        /// The block that the instruction will jump to in order to continue execution with the
        /// next instruction
        hook: Block,
    },
    /// Jump to a relative offset.
    Relative {
        /// Instruction destination relative to the instruction's program counter
        offset: i64,
        /// The block that the instruction will jump to in case of this outcome
        hook: Block,
    },
    /// Jump to an absolute address.
    Absolute {
        /// Address of the outcome's destination
        destination: Value<Address>,
        /// The block that the instruction will jump to in case of this outcome
        hook: Block,
    },
    /// Either fall-through to the next instruction or branch to a relative offset.
    Branch {
        /// The block that the instruction will jump to in case of fall-through
        fallthrough_hook: Block,
        /// Branch destination relative to the instruction's program counter
        branch_offset: i64,
        /// The block that the instruction will jump to in case of branch taken
        branch_hook: Block,
    },
    /// Instruction is guaranteed to raise an exception.
    GuaranteedException,
}

/// Lowered RISC-V instruction
#[derive(Debug)]
pub struct LoweredInstruction {
    /// Location of the instruction
    pub(super) program_counter: Address,

    /// Block that runs the instruction
    pub(super) run_block: Block,

    /// Execution outcomes of the instruction
    outcomes: InstructionOutcomes,

    /// Exception block, if any
    exception_block: Option<Block>,

    /// Width of the instruction
    width: InstrWidth,
}

impl LoweredInstruction {
    /// Return the address of the instruction following this one.
    pub fn next_instruction_address(&self) -> Address {
        self.program_counter.wrapping_add(self.width as u64)
    }

    /// Access the outcomes of the instruction.
    pub fn outcomes(&self) -> &InstructionOutcomes {
        &self.outcomes
    }

    /// Access the exception block, if any.
    pub fn exception_block(&self) -> Option<Block> {
        self.exception_block
    }
}

/// Result of an instruction execution
#[derive(Debug)]
pub enum InstructionResult<T> {
    /// The instruction will exit the sequence no matter what
    NoNext,

    /// Instruction can continue with another instruction
    HasNext(T),
}

/// Builder for a single RISC-V instruction
pub struct InstructionBuilder<'seq, 'jit, MC: MemoryConfig> {
    /// Target configuration for the JIT module
    target_config: TargetFrontendConfig,

    /// IR builder
    builder: &'seq mut FunctionBuilder<'jit>,

    /// External function call manager
    ext_calls: &'seq mut JsaCalls<MC>,

    /// Block that starts the instruction
    entry_block: Block,

    /// Program counter for the instruction being built
    instruction_pc: Address,

    /// Parameter pointing to the `MachineCoreState`
    core_param: Pointer<MachineCoreState<MC, Normal>>,

    /// Parameter pointing to the sequence result
    result_param: Pointer<ExceptionCode>,

    /// Width of the instruction being built
    width: InstrWidth,

    /// Block that all exception paths jump to
    exception_block: Option<Block>,
}

impl<'seq, 'jit, MC: MemoryConfig> InstructionBuilder<'seq, 'jit, MC> {
    /// Create a new instruction builder.
    #[expect(
        clippy::too_many_arguments,
        reason = "All parameters are required to define a new instruction builder."
    )]
    pub(super) fn new(
        target_config: TargetFrontendConfig,
        builder: &'seq mut FunctionBuilder<'jit>,
        ext_calls: &'seq mut JsaCalls<MC>,
        entry_block: Block,
        instruction_pc: Address,
        core_param: Pointer<MachineCoreState<MC, Normal>>,
        result_param: Pointer<ExceptionCode>,
        width: InstrWidth,
    ) -> Self {
        Self {
            target_config,
            builder,
            ext_calls,
            entry_block,
            instruction_pc,
            core_param,
            result_param,
            exception_block: None,
            width,
        }
    }

    /// Obtain an instruction inserter.
    pub(super) fn ins(&mut self) -> impl InstBuilder {
        self.builder.ins()
    }

    /// Handle an exception raised by the instruction.
    fn handle_exception<Any>(&mut self, exception: Value<ExceptionCode>) -> InstructionResult<Any> {
        self.result_param.write(self.builder, exception);

        let exception_block = self
            .exception_block
            .get_or_insert_with(|| self.builder.create_block());
        self.builder.ins().jump(*exception_block, []);

        InstructionResult::NoNext
    }

    /// Finalise the instruction building and produce an instruction.
    pub fn finish(self, result: InstructionResult<InstructionOutcomes>) -> LoweredInstruction {
        let outcomes = match result {
            InstructionResult::NoNext => {
                // When the instruction being built exits regardless, that means the instruction
                // is guaranteed to raise an exception.
                InstructionOutcomes::GuaranteedException
            }

            InstructionResult::HasNext(outcomes) => outcomes,
        };

        LoweredInstruction {
            program_counter: self.instruction_pc,
            run_block: self.entry_block,
            outcomes,
            width: self.width,
            exception_block: self.exception_block,
        }
    }
}

impl<MC: MemoryConfig> ICB for InstructionBuilder<'_, '_, MC> {
    type XValue = Value<XValue>;

    type XValue32 = Value<XValue32>;

    type FValue = Value<FValue>;

    type Bool = Value<bool>;

    type IResult<T> = InstructionResult<T>;

    type Outcome = InstructionOutcomes;

    fn xvalue_of_imm(&mut self, imm: i64) -> Self::XValue {
        let raw = self.ins().iconst(I64, imm);

        // SAFETY: The value returned by `iconst` is of type `I64` which matches the representation
        // of `XValue`.
        unsafe { Value::<XValue>::from_raw(raw) }
    }

    fn xvalue32_of_imm(&mut self, imm: i32) -> Self::XValue32 {
        let raw = self.ins().iconst(I32, imm as i64);

        // SAFETY: The value returned by `iconst` is of type `I32` which matches the representation
        // of `XValue32`.
        unsafe { Value::<XValue32>::from_raw(raw) }
    }

    fn xvalue_from_bool(&mut self, value: Self::Bool) -> Self::XValue {
        // Unsigned extension works as boolean can never be negative (only 0 or 1)
        let raw = self.ins().uextend(I64, value.to_value());

        // SAFETY: The value returned by `uextend` is of type `I64` which matches the representation
        // of `XValue`.
        unsafe { Value::<XValue>::from_raw(raw) }
    }

    fn bool_and(&mut self, lhs: Self::Bool, rhs: Self::Bool) -> Self::Bool {
        // SAFETY: `band` preserves the type of the values, so the result is also a `bool`.
        unsafe { lhs.lift_binary(|lhs, rhs| self.ins().band(lhs, rhs), rhs) }
    }

    fn narrow(&mut self, value: Self::XValue) -> Self::XValue32 {
        let raw = self.ins().ireduce(I32, value.to_value());

        // SAFETY: The value returned by `ireduce` is of type `I32` which matches the representation
        // of `XValue32`.
        unsafe { Value::<XValue32>::from_raw(raw) }
    }

    fn extend_signed(&mut self, value: Self::XValue32) -> Self::XValue {
        let raw = self.ins().sextend(I64, value.to_value());

        // SAFETY: The value returned by `sextend` is of type `I64` which matches the representation
        // of `XValue`.
        unsafe { Value::<XValue>::from_raw(raw) }
    }

    fn extend_unsigned(&mut self, value: Self::XValue32) -> Self::XValue {
        let raw = self.ins().uextend(I64, value.to_value());

        // SAFETY: The value returned by `uextend` is of type `I64` which matches the representation
        // of `XValue`.
        unsafe { Value::<XValue>::from_raw(raw) }
    }

    fn mul_high(
        &mut self,
        lhs: Self::XValue,
        rhs: Self::XValue,
        mul_high_type: MulHighType,
    ) -> Self::XValue {
        let mul_high_impl = |lhs, rhs| {
            let (lhs, rhs) = match mul_high_type {
                MulHighType::Signed => {
                    (self.ins().sextend(I128, lhs), self.ins().sextend(I128, rhs))
                }
                MulHighType::Unsigned => {
                    (self.ins().uextend(I128, lhs), self.ins().uextend(I128, rhs))
                }
                MulHighType::SignedUnsigned => {
                    (self.ins().sextend(I128, lhs), self.ins().uextend(I128, rhs))
                }
            };

            let result = self.ins().imul(lhs, rhs);
            let (_low, high) = self.ins().isplit(result);
            high
        };

        // SAFETY: `mul_high_impl` takes two `I64` values and produces an `I64` value. This means
        // the value types are preserved.
        unsafe { lhs.lift_binary(mul_high_impl, rhs) }
    }

    fn next(&mut self, _instr_width: InstrWidth) -> Self::Outcome {
        let hook = self.builder.create_block();
        self.ins().jump(hook, []);
        InstructionOutcomes::Next { hook }
    }

    fn jump_relative(&mut self, offset: i64) -> Self::Outcome {
        let hook = self.builder.create_block();
        self.ins().jump(hook, []);
        InstructionOutcomes::Relative { offset, hook }
    }

    fn jump_absolute(&mut self, destination: Self::XValue) -> Self::Outcome {
        let hook = self.builder.create_block();
        self.ins().jump(hook, []);
        InstructionOutcomes::Absolute { destination, hook }
    }

    fn branch(
        &mut self,
        condition: Self::Bool,
        branch_offset: i64,
        _instr_width: InstrWidth,
    ) -> Self::Outcome {
        let fallthrough_hook = self.builder.create_block();
        let branch_hook = self.builder.create_block();

        self.ins()
            .brif(condition.to_value(), branch_hook, [], fallthrough_hook, []);

        self.builder.seal_block(branch_hook);
        self.builder.seal_block(fallthrough_hook);

        InstructionOutcomes::Branch {
            fallthrough_hook,
            branch_offset,
            branch_hook,
        }
    }

    fn if_then<OnTrue>(&mut self, cond: Self::Bool, true_branch: OnTrue) -> Self::IResult<()>
    where
        OnTrue: FnOnce(&mut Self) -> Self::IResult<()>,
    {
        let true_block = self.builder.create_block();
        let false_block = self.builder.create_block();

        self.ins()
            .brif(cond.to_value(), true_block, [], false_block, []);
        self.builder.seal_block(true_block);
        self.builder.seal_block(false_block);

        // Code for true
        {
            self.builder.switch_to_block(true_block);
            (true_branch)(self);
        }

        // Code for false
        self.builder.switch_to_block(false_block);
        InstructionResult::HasNext(())
    }

    fn if_then_else<Phi: PhiValue, OnTrue, OnFalse>(
        &mut self,
        cond: Self::Bool,
        true_branch: OnTrue,
        false_branch: OnFalse,
    ) -> Phi::IcbValue<Self>
    where
        OnTrue: FnOnce(&mut Self) -> Phi::IcbValue<Self>,
        OnFalse: FnOnce(&mut Self) -> Phi::IcbValue<Self>,
    {
        let true_block = self.builder.create_block();
        let false_block = self.builder.create_block();
        let continue_block = self.builder.create_block();

        // Add a parameter to the continue-block for each parameter returned by the closures.
        Phi::IR_TYPES.iter().for_each(|v| {
            self.builder.append_block_param(continue_block, *v);
        });

        self.ins()
            .brif(cond.to_value(), true_block, [], false_block, []);

        self.builder.seal_block(true_block);
        self.builder.seal_block(false_block);

        // Code for true
        {
            self.builder.switch_to_block(true_block);

            let res_val = Phi::to_ir_vals(true_branch(self))
                .into_iter()
                .map(BlockArg::Value)
                .collect::<Vec<_>>();
            self.ins().jump(continue_block, res_val.as_slice());
        }

        // Code for false
        {
            self.builder.switch_to_block(false_block);

            let res_val = Phi::to_ir_vals(false_branch(self))
                .into_iter()
                .map(BlockArg::Value)
                .collect::<Vec<_>>();
            self.ins().jump(continue_block, res_val.as_slice());
        }

        // Code for after each branch
        {
            self.builder.switch_to_block(continue_block);

            // We need to make a copy of the parameter values to decouple the lifetimes
            let params = self.builder.block_params(continue_block).to_vec();
            Phi::from_ir_vals(params.as_slice(), self)
        }
    }

    fn main_memory_store<V: StoreLoadInt>(
        &mut self,
        phys_address: Self::XValue,
        value: Self::XValue,
    ) -> Self::IResult<()> {
        let errno =
            self.ext_calls
                .memory_store::<V>(self.builder, self.core_param, phys_address, value);

        let exception_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        let is_exception = errno.code.is_exception(self.builder);
        self.ins().brif(
            is_exception.to_value(),
            exception_block,
            [],
            success_block,
            [],
        );

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the store failed
        {
            self.builder.switch_to_block(exception_block);
            self.handle_exception::<()>(errno.code);
        }

        // Code for when the store succeeded
        {
            self.builder.switch_to_block(success_block);
            (errno.on_ok)(self.builder);
        }

        InstructionResult::HasNext(())
    }

    fn main_memory_load<V: StoreLoadInt>(
        &mut self,
        phys_address: Self::XValue,
    ) -> Self::IResult<Self::XValue> {
        let errno = self
            .ext_calls
            .memory_load::<V>(self.builder, self.core_param, phys_address);

        let exception_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        let is_exception = errno.code.is_exception(self.builder);
        self.ins().brif(
            is_exception.to_value(),
            exception_block,
            [],
            success_block,
            [],
        );

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the load failed
        {
            self.builder.switch_to_block(exception_block);

            self.handle_exception::<()>(errno.code);
        }

        // Code for when the load succeeded
        {
            self.builder.switch_to_block(success_block);

            let return_value = (errno.on_ok)(self.builder);
            InstructionResult::HasNext(return_value)
        }
    }

    fn main_memory_store_float<V: StoreLoadFloat>(
        &mut self,
        phys_address: Self::XValue,
        value: Self::FValue,
    ) -> Self::IResult<()> {
        let errno = self.ext_calls.memory_store_float::<V>(
            self.builder,
            self.core_param,
            phys_address,
            value,
        );

        let exception_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        let is_exception = errno.code.is_exception(self.builder);
        self.ins().brif(
            is_exception.to_value(),
            exception_block,
            [],
            success_block,
            [],
        );

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the store failed
        {
            self.builder.switch_to_block(exception_block);
            self.handle_exception::<()>(errno.code);
        }

        // Code for when the store succeeded
        {
            self.builder.switch_to_block(success_block);
            (errno.on_ok)(self.builder);
        }

        InstructionResult::HasNext(())
    }

    fn main_memory_load_float<V: StoreLoadFloat>(
        &mut self,
        phys_address: Self::XValue,
    ) -> Self::IResult<Self::FValue> {
        let errno =
            self.ext_calls
                .memory_load_float::<V>(self.builder, self.core_param, phys_address);

        let exception_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        let is_exception = errno.code.is_exception(self.builder);
        self.ins().brif(
            is_exception.to_value(),
            exception_block,
            [],
            success_block,
            [],
        );

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the load failed
        {
            self.builder.switch_to_block(exception_block);
            self.handle_exception::<()>(errno.code);
        }

        // Code for when the load succeeded
        {
            self.builder.switch_to_block(success_block);
            let return_value = (errno.on_ok)(self.builder);
            InstructionResult::HasNext(return_value)
        }
    }

    fn ok<Value>(&mut self, val: Value) -> Self::IResult<Value> {
        InstructionResult::HasNext(val)
    }

    fn raise_exception<In>(&mut self, exception: Exception) -> Self::IResult<In> {
        let code = ExceptionCode::build_exception_code(self.builder, exception);
        self.handle_exception(code)
    }

    fn map<Value, Next, F>(res: Self::IResult<Value>, f: F) -> Self::IResult<Next>
    where
        F: FnOnce(Value) -> Next,
    {
        match res {
            InstructionResult::NoNext => InstructionResult::NoNext,
            InstructionResult::HasNext(val) => InstructionResult::HasNext(f(val)),
        }
    }

    fn and_then<Value, Next, F>(res: Self::IResult<Value>, f: F) -> Self::IResult<Next>
    where
        F: FnOnce(Value) -> Self::IResult<Next>,
    {
        match res {
            InstructionResult::NoNext => InstructionResult::NoNext,
            InstructionResult::HasNext(val) => f(val),
        }
    }

    fn f64_from_x64_unsigned_dynamic(&mut self, xval: Self::XValue) -> Self::FValue {
        self.ext_calls
            .f64_from_x64_unsigned_dynamic(self.builder, self.core_param, xval)
    }

    fn f64_from_x64_unsigned_static(
        &mut self,
        xval: Self::XValue,
        rm: RoundingMode,
    ) -> Self::FValue {
        self.ext_calls
            .f64_from_x64_unsigned_static(self.builder, self.core_param, xval, rm)
    }

    fn csr_read(&mut self, reg: CSRegister) -> Self::XValue {
        self.ext_calls.csr_read(self.builder, self.core_param, reg)
    }

    fn csr_write(&mut self, reg: CSRegister, value: Self::XValue) {
        self.ext_calls
            .csr_write(self.builder, self.core_param, reg, value);
    }
}

impl<MC: MemoryConfig> StateContext for InstructionBuilder<'_, '_, MC> {
    type Value<R> = Value<R>;

    fn read_proj<P>(&mut self, param: P::Parameter) -> Self::Value<P::Target>
    where
        P: MachineCoreProjection,
        P::Target: Typed,
    {
        super::read_proj::<MC, P>(&self.target_config, self.builder, self.core_param, param)
    }

    fn write_proj<P>(&mut self, param: P::Parameter, value: Self::Value<P::Target>)
    where
        P: MachineCoreProjection,
    {
        super::write_proj::<MC, P>(
            &self.target_config,
            self.builder,
            self.core_param,
            param,
            value,
        )
    }

    fn pc_read(&mut self) -> Self::Value<XValue> {
        // SAFETY: `I64` is the valid cranelift representation for an `Address`, and matches
        // the representation of `XValue`.
        unsafe {
            Value::<XValue>::from_discriminant(
                &self.target_config,
                self.builder,
                self.instruction_pc as i64,
            )
        }
    }

    fn xreg_read_nz(&mut self, reg: NonZeroXRegister) -> Self::Value<XValue> {
        self.read_proj::<XRegisterProj>(reg as usize)
    }

    fn xreg_write_nz(&mut self, reg: NonZeroXRegister, value: Self::Value<XValue>) {
        self.write_proj::<XRegisterProj>(reg as usize, value);
    }

    fn freg_read(&mut self, reg: FRegister) -> Self::Value<FValue> {
        self.read_proj::<FRegisterProj>(reg as usize)
    }

    fn freg_write(&mut self, reg: FRegister, value: Self::Value<FValue>) {
        self.write_proj::<FRegisterProj>(reg as usize, value);
    }

    fn reservation_set_read(&mut self) -> Self::Value<u64> {
        self.read_proj::<ReservationSetProj>(())
    }

    fn reservation_set_write(&mut self, value: Self::Value<u64>) {
        self.write_proj::<ReservationSetProj>((), value);
    }
}
