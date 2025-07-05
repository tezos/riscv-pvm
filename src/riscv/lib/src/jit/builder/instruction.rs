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
use cranelift::prelude::Value;
use cranelift::prelude::types::I32;
use cranelift::prelude::types::I64;
use cranelift::prelude::types::I128;

use crate::instruction_context::ICB;
use crate::instruction_context::MulHighType;
use crate::instruction_context::Predicate;
use crate::instruction_context::StoreLoadInt;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::instruction_context::comparable::Comparable;
use crate::instruction_context::value::PhiValue;
use crate::interpreter::atomics;
use crate::interpreter::atomics::ReservationSetOption;
use crate::interpreter::float::RoundingMode;
use crate::jit::builder::F64;
use crate::jit::builder::X32;
use crate::jit::builder::X64;
use crate::jit::state_access::JsaCalls;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FRegister;
use crate::parser::instruction::InstrWidth;
use crate::state_context::StateContext;
use crate::state_context::projection::MachineCoreProjection;

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
        destination: Value,

        /// The block that the instruction will jump to in case of a branch
        hook: Block,
    },
}

/// Lowered RISC-V instruction
pub struct LoweredInstruction {
    /// Location of the instruction
    program_counter: Value,

    /// Block that runs the instruction
    run_block: Block,

    /// Execution outcomes of the instruction
    outcomes: Vec<Outcome>,
}

impl LoweredInstruction {
    /// Access the program counter for this instruction.
    pub fn program_counter(&self) -> Value {
        self.program_counter
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
    /// IR builder
    builder: &'seq mut FunctionBuilder<'jit>,

    /// External function call manager
    ext_calls: &'seq mut JsaCalls<'jit, MC>,

    /// Block that starts the instruction
    entry_block: Block,

    /// Program counter for the instruction being built
    instruction_pc: Value,

    /// Parameter pointing to the `MachineCoreState`
    core_param: Value,

    /// Parameter pointing to the sequence result
    result_param: Value,

    /// Execution outcomes of the instruction
    outcomes: Vec<Outcome>,
}

impl<'seq, 'jit, MC: MemoryConfig> InstructionBuilder<'seq, 'jit, MC> {
    /// Create a new instruction builder.
    pub(super) fn new(
        builder: &'seq mut FunctionBuilder<'jit>,
        ext_calls: &'seq mut JsaCalls<'jit, MC>,
        entry_block: Block,
        instruction_pc: Value,
        core_param: Value,
        result_param: Value,
    ) -> Self {
        Self {
            builder,
            ext_calls,
            entry_block,
            instruction_pc,
            core_param,
            result_param,
            outcomes: Vec::new(),
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

    /// Allocate an outcome block for an unknown branch.
    fn create_unknown_branch_outcome(&mut self, destination: Value) -> Block {
        let hook = self.builder.create_block();
        self.outcomes
            .push(Outcome::UnknownBranch { destination, hook });
        hook
    }

    /// Handle an exception raised by the instruction.
    fn handle_exception<Any>(&mut self, exception_ptr: Value) -> InstructionResult<Any> {
        let current_pc = self.pc_read();
        let outcome = self.ext_calls.handle_exception(
            self.builder,
            self.core_param,
            exception_ptr,
            self.result_param,
            current_pc,
        );

        let exception_block = self.create_exception_outcome();
        let unknown_branch_block = self.create_unknown_branch_outcome(outcome.new_pc.0);

        self.ins().brif(
            outcome.handled,
            unknown_branch_block,
            [],
            exception_block,
            [],
        );

        // The predecessors of either block are now known
        self.builder.seal_block(exception_block);
        self.builder.seal_block(unknown_branch_block);

        InstructionResult::NoNext
    }

    /// Finalise the instruction building and produce an instruction.
    pub fn finish(
        self,
        result: InstructionResult<ProgramCounterUpdate<X64>>,
    ) -> LoweredInstruction {
        let mut lowered = LoweredInstruction {
            program_counter: self.instruction_pc,
            run_block: self.entry_block,
            outcomes: self.outcomes,
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
                        destination: address.0,
                        hook,
                    },
                    ProgramCounterUpdate::Next(_width) => Outcome::Next { hook },
                };
                lowered.outcomes.push(outcome);
            }
        }

        lowered
    }
}

impl<MC: MemoryConfig> ICB for InstructionBuilder<'_, '_, MC> {
    type XValue = X64;

    type XValue32 = X32;

    type FValue = F64;

    type Bool = Value;

    type IResult<T> = InstructionResult<T>;

    fn xvalue_of_imm(&mut self, imm: i64) -> Self::XValue {
        X64(self.ins().iconst(I64, imm))
    }

    fn xvalue32_of_imm(&mut self, imm: i32) -> Self::XValue32 {
        X32(self.ins().iconst(I32, imm as i64))
    }

    fn xvalue_from_bool(&mut self, value: Self::Bool) -> Self::XValue {
        // Unsigned extension works as boolean can never be negative (only 0 or 1)
        X64(self.ins().uextend(I64, value))
    }

    fn fregister_read(&mut self, reg: FRegister) -> Self::FValue {
        self.ext_calls
            .ir_freg_read(self.builder, self.core_param, reg)
    }

    fn fregister_write(&mut self, reg: FRegister, value: Self::FValue) {
        // The value contained must be a floating-point type.
        self.ext_calls
            .ir_freg_write(self.builder, self.core_param, reg, value)
    }

    fn pc_read(&mut self) -> Self::XValue {
        X64(self.instruction_pc)
    }

    fn bool_and(&mut self, lhs: Self::Bool, rhs: Self::Bool) -> Self::Bool {
        self.ins().band(lhs, rhs)
    }

    fn narrow(&mut self, value: Self::XValue) -> Self::XValue32 {
        X32(self.ins().ireduce(I32, value.0))
    }

    fn extend_signed(&mut self, value: Self::XValue32) -> Self::XValue {
        X64(self.ins().sextend(I64, value.0))
    }

    fn extend_unsigned(&mut self, value: Self::XValue32) -> Self::XValue {
        X64(self.ins().uextend(I64, value.0))
    }

    fn mul_high(
        &mut self,
        lhs: Self::XValue,
        rhs: Self::XValue,
        mul_high_type: crate::instruction_context::MulHighType,
    ) -> Self::XValue {
        let (lhs, rhs) = match mul_high_type {
            MulHighType::Signed => (
                self.ins().sextend(I128, lhs.0),
                self.ins().sextend(I128, rhs.0),
            ),
            MulHighType::Unsigned => (
                self.ins().uextend(I128, lhs.0),
                self.ins().uextend(I128, rhs.0),
            ),
            MulHighType::SignedUnsigned => (
                self.ins().sextend(I128, lhs.0),
                self.ins().uextend(I128, rhs.0),
            ),
        };

        let result = self.ins().imul(lhs, rhs);
        let (_low, high) = self.ins().isplit(result);

        X64(high)
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
            .brif(condition, branch_block, [], continue_block, []);

        self.builder.seal_block(branch_block);
        self.builder.seal_block(continue_block);

        self.builder.switch_to_block(continue_block);

        ProgramCounterUpdate::Next(instr_width)
    }

    fn branch_merge<Phi: PhiValue, OnTrue, OnFalse>(
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

        self.ins().brif(cond, true_block, [], false_block, []);

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

    fn atomic_access_fault_guard<V: StoreLoadInt>(
        &mut self,
        address: Self::XValue,
        reservation_set_option: ReservationSetOption,
    ) -> Self::IResult<()> {
        let width = self.xvalue_of_imm(V::WIDTH as i64);
        let remainder = address.modulus_unsigned(width, self);

        // The steps of taking the comparison are technically not needed, as cranelift will
        // treat any non-zero value as a take-branch (i.e. raise exception) value, so we could
        // pass the remainder directly. However for completeness and clarity, we are keeping the
        // comparison here.
        let zero = self.xvalue_of_imm(0);
        let not_aligned = remainder.compare(zero, Predicate::NotEqual, self);

        let exception_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        self.ins()
            .brif(not_aligned, exception_block, [], success_block, []);

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the address is not aligned
        {
            self.builder.switch_to_block(exception_block);

            if let ReservationSetOption::Reset = reservation_set_option {
                // If the atomic operation was a store_conditional, we reset the reservation.
                atomics::reset_reservation_set(self);
            }

            let exception_ptr = self
                .ext_calls
                .raise_store_amo_access_fault_exception(self.builder, address.0);
            self.handle_exception::<()>(exception_ptr);
        }

        self.builder.switch_to_block(success_block);

        InstructionResult::HasNext(())
    }

    fn ecall(&mut self) -> Self::IResult<ProgramCounterUpdate<Self::XValue>> {
        let exception_ptr = self.ext_calls.ecall(self.builder, self.core_param);
        self.handle_exception(exception_ptr)
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

        self.ins()
            .brif(errno.is_exception, exception_block, [], success_block, []);

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the store failed
        {
            self.builder.switch_to_block(exception_block);
            self.handle_exception::<()>(errno.exception_ptr);
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

        self.ins()
            .brif(errno.is_exception, exception_block, [], success_block, []);

        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when the load failed
        {
            self.builder.switch_to_block(exception_block);
            self.handle_exception::<()>(errno.exception_ptr);
        }

        // Code for when the load succeeded
        {
            self.builder.switch_to_block(success_block);

            let return_value = (errno.on_ok)(self.builder);
            InstructionResult::HasNext(return_value)
        }
    }

    fn reservation_set_write(&mut self, address: Self::XValue) {
        self.ext_calls
            .reservation_set_write(self.builder, self.core_param, address);
    }

    fn reservation_set_read(&mut self) -> Self::XValue {
        self.ext_calls
            .reservation_set_read(self.builder, self.core_param)
    }

    fn ok<Value>(&mut self, val: Value) -> Self::IResult<Value> {
        InstructionResult::HasNext(val)
    }

    fn err_illegal_instruction<In>(&mut self) -> Self::IResult<In> {
        let exception_ptr = self
            .ext_calls
            .raise_illegal_instruction_exception(self.builder);
        self.handle_exception(exception_ptr)
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

    fn f64_from_x64_unsigned_dynamic(&mut self, xval: Self::XValue) -> Self::IResult<Self::FValue> {
        let errno =
            self.ext_calls
                .f64_from_x64_unsigned_dynamic(self.builder, self.core_param, xval);

        let exception_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        self.ins()
            .brif(errno.is_exception, exception_block, [], success_block, []);

        // All inputs to these blocks are already known, so we can seal them immediately.
        self.builder.seal_block(exception_block);
        self.builder.seal_block(success_block);

        // Code for when an exception was raised.
        {
            self.builder.switch_to_block(exception_block);
            self.handle_exception::<()>(errno.exception_ptr);
        }

        // Code for when the conversion succeeded.
        {
            self.builder.switch_to_block(success_block);

            let return_value = (errno.on_ok)(self.builder);
            InstructionResult::HasNext(return_value)
        }
    }

    fn f64_from_x64_unsigned_static(
        &mut self,
        xval: Self::XValue,
        rm: RoundingMode,
    ) -> Self::FValue {
        self.ext_calls
            .f64_from_x64_unsigned_static(self.builder, self.core_param, xval, rm)
    }
}

impl<MC: MemoryConfig> StateContext for InstructionBuilder<'_, '_, MC> {
    type X64 = X64;

    fn read_proj<P>(&mut self, param: P::Parameter) -> Self::X64
    where
        P: MachineCoreProjection<Target = u64>,
    {
        super::read_proj::<MC, P>(self.builder, self.core_param, param)
    }

    fn write_proj<P>(&mut self, param: P::Parameter, value: Self::X64)
    where
        P: MachineCoreProjection<Target = u64>,
    {
        super::write_proj::<MC, P>(self.builder, self.core_param, param, value)
    }
}
