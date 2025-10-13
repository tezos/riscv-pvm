// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Builder for sequences of instructions
//!
//! This module provides the infrastructure for building and compiling sequences of RISC-V
//! instructions using Cranelift IR. The [sequence builder] coordinates the compilation of multiple
//! instructions into a single JIT function, managing control flow, program counter updates,
//! and various [execution outcomes].
//!
//! [sequence builder]: SequenceBuilder
//! [execution outcomes]: Outcome

use cranelift::codegen::Context;
use cranelift::codegen::ir::BlockArg;
use cranelift::prelude::AbiParam;
use cranelift::prelude::Block;
use cranelift::prelude::EntityRef;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::FunctionBuilderContext;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::Variable;
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift::prelude::types::I64;
use cranelift_jit::JITModule;
use cranelift_module::Module;

use super::instruction::Outcome;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Typed;
use crate::jit::builder::typed::Value;
use crate::jit::state_access::ExceptionCode;
use crate::jit::state_access::JsaCalls;
use crate::machine_state::MachineCoreState;
use crate::machine_state::hart_state::write_pc;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::parser::instruction::InstrWidth;
use crate::state_backend::owned_backend::Owned;
use crate::state_context::StateContext;
use crate::state_context::projection::MachineCoreProjection;

const STEPS_REMAINING_VAR_ID: usize = 0;

/// Builder for an instruction sequence
pub struct SequenceBuilder<'jit, MC: MemoryConfig> {
    /// Target configuration for the JIT module
    target_config: TargetFrontendConfig,

    /// IR builder
    builder: FunctionBuilder<'jit>,

    /// External function call manager
    ext_calls: JsaCalls<MC>,

    /// Function entry block
    entry_block: Block,

    /// Parameter pointing to the `MachineCoreState`
    core_param: Pointer<MachineCoreState<MC, Owned>>,

    /// The program counter of the start of the sequence
    program_counter: Address,

    /// Offset to the program counter for the next instruction
    program_counter_offset: i64,

    /// Initial maximum number of steps that can be executed in the sequence
    max_steps_param: Value<usize>,

    /// Parameter pointing to the sequence result
    result_param: Pointer<ExceptionCode>,

    /// Variable storing the maximum number of steps that can be executed in the sequence.
    steps_remaining: Variable,
}

impl<'jit, MC: MemoryConfig> SequenceBuilder<'jit, MC> {
    /// Create a new sequence builder.
    pub fn new(
        module: &'jit mut JITModule,
        context: &'jit mut Context,
        builder_context: &'jit mut FunctionBuilderContext,
        program_counter: Address,
    ) -> Self {
        // The pointer type is host-dependent, hence we need to retrieve it from the module's
        // target configuration.
        let ptr_type = module.target_config().pointer_type();

        // The context is where the function's IR is built. We don't want any left-overs from
        // previous functions, so we clear the context.
        module.clear_context(context);

        // Parameters:
        //   - `self`: Pointer to the `Jitted` block
        //   - `core`: Pointer to the `MachineCoreState`
        //   - `program_counter`: Program counter at the start of the sequence
        //   - `max_steps`: Maximum number of steps that can be executed in the sequence
        //   - `result`: Pointer to the result of the sequence
        //   - `compiler`: Pointer to the `BlockBuilder` that is used to build the sequence
        // Returns:
        //   - `steps`: Number of steps executed in the sequence
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(I64));
        context.func.signature.params.push(AbiParam::new(I64));
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(ptr_type));

        context.func.signature.returns.push(AbiParam::new(I64));

        // The function builder will be used to create basic blocks and to insert IR instructions
        // into them.
        let mut builder = FunctionBuilder::new(&mut context.func, builder_context);

        // [`JsaCalls`] allows us to perform calls to external functions, such as reading registers
        // or writing the program counter to the machine core state.
        let ext_calls = JsaCalls::new(module.target_config());

        // The function entry block is the first basic block in the function. It brings the function
        // parameters values into scope.
        let param_block = builder.create_block();
        builder.seal_block(param_block);
        builder.append_block_params_for_function_params(param_block);
        builder.switch_to_block(param_block);

        // SAFETY: `JitFn` accepts a `&mut MachineCoreState<MC, Owned>` as the 2nd parameter.
        let core_param = unsafe {
            let raw_value = builder.block_params(param_block)[1];
            Pointer::<MachineCoreState<MC, Owned>>::from_raw(raw_value)
        };

        // SAFETY: `JitFn` accepts a `usize` as the 4th parameter.
        let max_steps_param = unsafe {
            let raw_value = builder.block_params(param_block)[3];
            Value::<usize>::from_raw(raw_value)
        };

        // SAFETY: `JitFn` accepts a `&mut ExceptionCode` as the 5th parameter.
        let result_param = unsafe {
            let raw_value = builder.block_params(param_block)[4];
            Pointer::<ExceptionCode>::from_raw(raw_value)
        };

        let steps_remaining = Variable::new(STEPS_REMAINING_VAR_ID);
        builder.declare_var(steps_remaining, I64);

        // Assign the passed in `max_steps` to the `steps_remaining` variable.
        builder.def_var(steps_remaining, max_steps_param.to_value());

        // The entry block is where we will eventually transition to the first instruction basic
        // block. The function's entry block (`param_block` for our purposes) will directly jump to
        // this `entry_block`.
        let entry_block = builder.create_block();
        builder.ins().jump(entry_block, []);

        let target_config = module.target_config();

        Self {
            target_config,
            builder,
            ext_calls,
            entry_block,
            core_param,
            program_counter,
            program_counter_offset: 0,
            max_steps_param,
            result_param,
            steps_remaining,
        }
    }

    /// Abandon building the sequence.
    pub fn abandon(mut self) {
        // We need to finalise the builder but the builder might not be in a valid state.
        // Normally, we would need to insert a block finaliser. This should be done automatically
        // for each instruction. However, this is not the case for the entry block.
        if self.builder.current_block() == Some(self.entry_block) {
            self.builder.ins().return_(&[]);
        }

        // Sealing all blocks is the second step to get the builder into a valid state for
        // finalisation.
        self.builder.seal_all_blocks();

        // Finalisation clears the function builder context.
        self.builder.finalize();
    }

    /// Create an instruction builder for the next instruction in the sequence.
    pub fn build_next_instruction<'seq>(
        &'seq mut self,
        width: InstrWidth,
    ) -> InstructionBuilder<'seq, 'jit, MC> {
        let entry_block = self.builder.create_block();
        self.builder.switch_to_block(entry_block);

        // Compute the program counter for the instruction, if necessary.
        let instruction_pc = self
            .program_counter
            .wrapping_add_signed(self.program_counter_offset);

        let instr_builder = InstructionBuilder::new(
            self.target_config,
            &mut self.builder,
            &mut self.ext_calls,
            entry_block,
            instruction_pc,
            self.core_param,
            self.result_param,
            width,
        );

        // The next instruction needs to be able to compute its program counter based on which
        // instructions came before it.
        self.program_counter_offset += width as i64;

        instr_builder
    }

    /// Update the `steps_remaining` variable and jump to the exit block.
    ///
    /// `final_program_counter` is the program counter that we want to commit back to the
    /// machine core state when exiting the sequence.
    fn jump_to_exit(
        &mut self,
        steps_completed: u64,
        final_program_counter: Value<Address>,
        exit_block: Block,
    ) {
        self.update_steps_remaining(steps_completed);

        self.builder.ins().jump(exit_block, &[BlockArg::Value(
            final_program_counter.to_value(),
        )]);
    }

    /// Decrement 'steps_remaining' by the number of steps taken in the sequence.
    fn update_steps_remaining(&mut self, steps_completed: u64) {
        if steps_completed == 0 {
            return;
        }

        let steps_completed = self.builder.ins().iconst(I64, steps_completed as i64);

        let steps_remaining = self.builder.use_var(self.steps_remaining);
        let result_steps_remaining = self.builder.ins().isub(steps_remaining, steps_completed);

        self.builder
            .def_var(self.steps_remaining, result_steps_remaining);
    }

    /// Finish building the sequence.
    pub fn finish(mut self, instrs: &[LoweredInstruction]) {
        let exit_block = self.builder.create_block();

        // The exit block is used to write the program counter back to the machine core state, as
        // well as returning from the JIT function.
        {
            self.builder.switch_to_block(exit_block);

            // SAFETY: We're declaring the value as a `I64` which has the same representation of
            // `Address`.
            let final_program_counter = unsafe {
                let final_program_counter = self.builder.append_block_param(exit_block, I64);
                Value::<Address>::from_raw(final_program_counter)
            };

            write_pc(&mut self, final_program_counter);

            let steps_remaining = self.builder.use_var(self.steps_remaining);
            let max_steps = self.max_steps_param.to_value();
            let steps = self.builder.ins().isub(max_steps, steps_remaining);

            self.builder.ins().return_(&[steps]);
        }

        let mut peekable_instrs = instrs.iter().enumerate().peekable();

        if let Some((_, first_instr)) = peekable_instrs.peek() {
            // Hook up the entry block to the first instruction.
            self.builder.switch_to_block(self.entry_block);
            first_instr.build_run(&mut self.builder);
        }

        while let Some((instr_index, instr)) = peekable_instrs.next() {
            // Each instruction may have multiple outcomes. Each kind of outcome needs to be
            // handled. This involves populating the hook block, which the instruction jumps to in
            // order to indicate that outcome.
            for outcome in instr.outcomes() {
                match outcome {
                    Outcome::Next { hook } => {
                        self.builder.switch_to_block(*hook);

                        if let Some((_, next_instr)) = peekable_instrs.peek() {
                            // If there is a next instruction, we jump to its entry block.
                            next_instr.build_run(&mut self.builder);
                        } else {
                            // This is a successful outcome, hence +1 step.
                            let steps_completed = instr_index as u64 + 1;

                            let final_program_counter = instr.next_instruction_address();

                            // SAFETY: We are constructing this value directly from an Address type.
                            let final_program_counter = unsafe {
                                Value::<Address>::from_discriminant(
                                    &self.target_config,
                                    &mut self.builder,
                                    final_program_counter as i64,
                                )
                            };

                            // If there is no next instruction, we jump to the exit block.
                            self.jump_to_exit(steps_completed, final_program_counter, exit_block);
                        }
                    }

                    Outcome::Exception { hook } => {
                        self.builder.switch_to_block(*hook);

                        // Exception outcomes do not increment the step counter.
                        let steps_completed = instr_index as u64;

                        // In the case of an exception, the program counter needs to refer to the
                        // instruction that caused the exception.
                        let final_program_counter = instr.program_counter();

                        // SAFETY: We are constructing this value directly from an Address type.
                        let final_program_counter = unsafe {
                            Value::<Address>::from_discriminant(
                                &self.target_config,
                                &mut self.builder,
                                final_program_counter as i64,
                            )
                        };

                        // Exception outcomes do not increment the step counter, as they don't
                        // count as a successful step.
                        self.jump_to_exit(steps_completed, final_program_counter, exit_block);
                    }

                    Outcome::KnownBranch { offset, hook } => {
                        self.builder.switch_to_block(*hook);

                        // This is a successful outcome, hence +1 step.
                        let steps_completed = instr_index as u64 + 1;

                        let final_program_counter: Address =
                            instr.program_counter().wrapping_add_signed(*offset);

                        // SAFETY: We are constructing this value directly from an Address type.
                        let final_program_counter = unsafe {
                            Value::<Address>::from_discriminant(
                                &self.target_config,
                                &mut self.builder,
                                final_program_counter as i64,
                            )
                        };

                        self.jump_to_exit(steps_completed, final_program_counter, exit_block);
                    }

                    Outcome::UnknownBranch { destination, hook } => {
                        self.builder.switch_to_block(*hook);

                        // This is a successful outcome, hence +1 step.
                        let steps_completed = instr_index as u64 + 1;

                        // The instruction wants to jump somewhere, so we take the destination.
                        let final_program_counter = *destination;

                        self.jump_to_exit(steps_completed, final_program_counter, exit_block);
                    }
                }
            }
        }

        self.builder.seal_all_blocks();
        self.builder.finalize();
    }
}

impl<MC: MemoryConfig> StateContext for SequenceBuilder<'_, MC> {
    type Value<R> = Value<R>;

    fn read_proj<P>(&mut self, param: P::Parameter) -> Self::Value<P::Target>
    where
        P: MachineCoreProjection,
        P::Target: Typed,
    {
        super::read_proj::<MC, P>(
            &self.target_config,
            &mut self.builder,
            self.core_param,
            param,
        )
    }

    fn write_proj<P>(&mut self, param: P::Parameter, value: Self::Value<P::Target>)
    where
        P: MachineCoreProjection,
    {
        super::write_proj::<MC, P>(
            &self.target_config,
            &mut self.builder,
            self.core_param,
            param,
            value,
        )
    }
}
