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
//! [`InstructionBuilder::finish`] converts the builder into a [`LoweredInstruction`] with all
//! [outcomes] properly connected at their source, allowing the [sequence builder] to integrate it
//! into the overall sequence control flow.
//!
//! [sequence builder]: super::sequence::SequenceBuilder
//! [`build_next_instruction`]: super::sequence::SequenceBuilder::build_next_instruction
//! [instruction builder]: InstructionBuilder
//! [outcomes]: Outcome

use cranelift::codegen::ir::BlockArg;
use cranelift::prelude::Block;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift::prelude::types::I32;
use cranelift::prelude::types::I64;
use cranelift::prelude::types::I128;

use crate::exceptions::Exception;
use crate::instruction_context::ICB;
use crate::instruction_context::MulHighType;
use crate::instruction_context::StoreLoadInt;
use crate::instruction_context::value::PhiValue;
use crate::interpreter::float::RoundingMode;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;
use crate::jit::state_access::ExceptionCode;
use crate::jit::state_access::JsaCalls;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::XValue;
use crate::machine_state::registers::XValue32;
use crate::parser::instruction::InstrWidth;
use crate::state_backend::owned_backend::Owned;
use crate::state_context::StateContext;
use crate::state_context::projection::MachineCoreProjection;

/// Probability of taking an outcome of a particular instruction.
///
/// As detailed in RISC-V Control Transfer Instructions specification (2.5),
/// backward-branches should be treated as likely taken, while forward-branches
/// should be treated as likely not-taken. Also, exception handlers should be treated
/// as likely not-taken (except for a few instructions, such as `ECall`, which are guaranteed to
/// result in an exception).
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutcomeProbability {
    Guaranteed,
    High,
    Low,
}

/// Instruction execution outcome
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    /// Continue execution
    Next {
        /// The block that the instruction will jump to in order to continue execution with the
        /// next instruction
        hook: Block,
    },

    /// An exception has been raised
    Exception {
        /// The block that the instruction will jump to after an exception in order to exit the
        /// instruction sequence
        hook: Block,
    },

    /// Branch to a known location
    KnownBranch {
        /// Instruction destination relative to the instruction's program counter
        offset: i64,

        /// The block that the instruction will jump to in case of a branch
        hook: Block,
    },

    /// Branch to an unknown location
    UnknownBranch {
        /// Address of the branch destination
        destination: Value<Address>,

        /// The block that the instruction will jump to in case of a branch
        hook: Block,
    },
}

/// Lowered RISC-V instruction
pub struct LoweredInstruction {
    /// Location of the instruction
    program_counter: Address,

    /// Block that runs the instruction
    run_block: Block,

    /// Execution outcomes of the instruction
    outcomes: Vec<Outcome>,

    /// Width of the instruction
    width: InstrWidth,
}

impl LoweredInstruction {
    /// Access the program counter for this instruction.
    pub fn program_counter(&self) -> Address {
        self.program_counter
    }

    /// Return the address of the instruction following this one.
    pub fn next_instruction_address(&self) -> Address {
        self.program_counter.wrapping_add(self.width as u64)
    }

    /// Access the outcomes of the instruction.
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// Build a jump that effectively runs the instruction.
    pub fn build_run(&self, builder: &mut FunctionBuilder) {
        builder.ins().jump(self.run_block, []);
    }
}

/// Result of an instruction execution
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
    core_param: Pointer<MachineCoreState<MC, Owned>>,

    /// Parameter pointing to the sequence result
    result_param: Pointer<ExceptionCode>,

    /// Execution outcomes of the instruction
    outcomes: Vec<Outcome>,

    /// Width of the instruction being built
    width: InstrWidth,
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
        core_param: Pointer<MachineCoreState<MC, Owned>>,
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
            outcomes: Vec::new(),
            width,
        }
    }

    /// Obtain an instruction inserter.
    pub(super) fn ins(&mut self) -> impl InstBuilder {
        self.builder.ins()
    }

    /// Allocate an outcome block for an exception.
    fn create_exception_outcome(&mut self) -> Block {
        let hook = self.builder.create_block();
        self.outcomes.push(Outcome::Exception { hook });
        hook
    }

    /// Allocate an outcome block for a known branch.
    fn create_known_branch_outcome(&mut self, offset: i64) -> Block {
        let hook = self.builder.create_block();
        self.outcomes.push(Outcome::KnownBranch { offset, hook });
        hook
    }

    /// Handle an exception raised by the instruction.
    fn handle_exception<Any>(&mut self, exception: Value<ExceptionCode>) -> InstructionResult<Any> {
        self.result_param.write(self.builder, exception);

        let exception_block = self.create_exception_outcome();
        self.builder.ins().jump(exception_block, []);

        InstructionResult::NoNext
    }

    /// Finalise the instruction building and produce an instruction.
    pub fn finish(
        self,
        result: InstructionResult<ProgramCounterUpdate<Value<XValue>>>,
    ) -> LoweredInstruction {
        let mut lowered = LoweredInstruction {
            program_counter: self.instruction_pc,
            run_block: self.entry_block,
            outcomes: self.outcomes,
            width: self.width,
        };

        // Hook up the end of the instruction.
        match result {
            InstructionResult::NoNext => {
                // When the instruction being built exits regardless, that means that the block
                // we're currently targeting ends in a branching or jump IR instruction.
            }

            InstructionResult::HasNext(update) => {
                // However, when a next instruction is possible, the current block needs to be
                // populated. In this case, we jump to the corresponding outcome hook block. We
                // need to insert this jump instruction to ensure that the block is not empty -
                // otherwise we can't switch away from it.

                let hook = self.builder.create_block();
                self.builder.ins().jump(hook, []);

                let outcome = match update {
                    ProgramCounterUpdate::Set(address) => Outcome::UnknownBranch {
                        destination: address,
                        hook,
                    },
                    ProgramCounterUpdate::Relative(offset) => Outcome::KnownBranch { offset, hook },
                    ProgramCounterUpdate::Next(_width) => Outcome::Next { hook },
                };
                lowered.outcomes.push(outcome);
            }
        }

        lowered
    }
}

impl<MC: MemoryConfig> ICB for InstructionBuilder<'_, '_, MC> {
    type XValue = Value<XValue>;

    type XValue32 = Value<XValue32>;

    type FValue = Value<FValue>;

    type Bool = Value<bool>;

    type IResult<T> = InstructionResult<T>;

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

    fn pc_read(&mut self) -> Self::XValue {
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

    fn branch(
        &mut self,
        condition: Self::Bool,
        offset: i64,
        instr_width: InstrWidth,
    ) -> ProgramCounterUpdate<Self::XValue> {
        let continue_block = self.builder.create_block();
        let branch_block = self.create_known_branch_outcome(offset);

        self.ins()
            .brif(condition.to_value(), branch_block, [], continue_block, []);

        self.builder.seal_block(branch_block);
        self.builder.seal_block(continue_block);

        self.builder.switch_to_block(continue_block);

        ProgramCounterUpdate::Next(instr_width)
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
}
