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
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::FunctionBuilderContext;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::Value;
use cranelift::prelude::types::I64;
use cranelift_jit::JITModule;
use cranelift_module::Module;

use super::instruction::Outcome;
use crate::jit::JsaImports;
use crate::jit::builder::X64;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::state_access::JsaCalls;
use crate::machine_state::memory::MemoryConfig;
use crate::parser::instruction::InstrWidth;
use crate::state_context::StateContext;
use crate::state_context::projection::MachineCoreProjection;
use crate::state_context::projection::RegionCons;

/// Builder for an instruction sequence
pub struct SequenceBuilder<'jit, MC: MemoryConfig> {
    /// IR builder
    builder: FunctionBuilder<'jit>,

    /// External function call manager
    ext_calls: JsaCalls<'jit, MC>,

    /// Function entry block
    entry_block: Block,

    /// Parameter pointing to the `MachineCoreState`
    core_param: Value,

    /// Parameter holding the program counter at the start of the sequence
    program_counter_param: Value,

    /// Offset to the program counter for the next instruction
    program_counter_offset: i64,

    /// Parameter pointing to the sequence result
    result_param: Value,
}

impl<'jit, MC: MemoryConfig> SequenceBuilder<'jit, MC> {
    /// Create a new sequence builder.
    pub fn new(
        module: &'jit mut JITModule,
        imports: &'jit JsaImports<MC>,
        context: &'jit mut Context,
        builder_context: &'jit mut FunctionBuilderContext,
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
        //   - `result`: Pointer to the result of the sequence
        //   - `block_builder`: Pointer to the `BlockBuilder` that is used to build the sequence
        // Returns:
        //   - `steps`: Number of steps executed in the sequence
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(I64));
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(ptr_type));

        context.func.signature.returns.push(AbiParam::new(I64));

        // The function builder will be used to create basic blocks and to insert IR instructions
        // into them.
        let mut builder = FunctionBuilder::new(&mut context.func, builder_context);

        // [`JsaCalls`] allows us to perform calls to external functions, such as reading registers
        // or writing the program counter to the machine core state.
        let ext_calls = JsaCalls::func_calls(module, imports, ptr_type);

        // The function entry block is the first basic block in the function. It brings the function
        // parameters values into scope.
        let param_block = builder.create_block();
        builder.seal_block(param_block);
        builder.append_block_params_for_function_params(param_block);
        builder.switch_to_block(param_block);

        let core_param = builder.block_params(param_block)[1];
        let program_counter_param = builder.block_params(param_block)[2];
        let result_param = builder.block_params(param_block)[3];

        // The entry block is where we will eventually transition to the first instruction basic
        // block. The function's entry block (`param_block` for our purposes) will directly jump to
        // this `entry_block`.
        let entry_block = builder.create_block();
        builder.ins().jump(entry_block, []);

        Self {
            builder,
            ext_calls,
            entry_block,
            core_param,
            program_counter_param,
            program_counter_offset: 0,
            result_param,
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
        let instruction_pc = if self.program_counter_offset == 0 {
            self.program_counter_param
        } else {
            self.builder
                .ins()
                .iadd_imm(self.program_counter_param, self.program_counter_offset)
        };

        let instr_builder = InstructionBuilder::new(
            &mut self.builder,
            &mut self.ext_calls,
            entry_block,
            instruction_pc,
            self.core_param,
            self.result_param,
        );

        // The next instruction needs to be able to compute its program counter based on which
        // instructions came before it.
        self.program_counter_offset += width as i64;

        instr_builder
    }

    /// Finish building the sequence.
    pub fn finish(mut self, instrs: &[LoweredInstruction]) {
        let exit_block = self.builder.create_block();

        // The exit block is used to write the program counter back to the machine core state, as
        // well as returning from the JIT function.
        {
            self.builder.switch_to_block(exit_block);
            let steps = self.builder.append_block_param(exit_block, I64);
            let final_program_counter = self.builder.append_block_param(exit_block, I64);

            self.ext_calls.pc_write(
                &mut self.builder,
                self.core_param,
                X64(final_program_counter),
            );

            self.builder.ins().return_(&[steps]);
        }

        let jump_to_exit =
            // `steps` is the number of successful steps that were executed in the sequence. It
            // will be returned as the result of the JIT function.
            // `final_program_counter` is the program counter that we want to commit back to the
            // machine core state when existing the sequence.
            |builder: &mut FunctionBuilder<'_>, steps: i64, final_program_counter: Value| {
                let steps = builder.ins().iconst(I64, steps);
                builder.ins().jump(exit_block, &[
                    BlockArg::Value(steps),
                    BlockArg::Value(final_program_counter),
                ]);
            };

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
                            let step_counter = instr_index as i64 + 1;

                            // At this point `program_counter_offset` is the sum of all instruction
                            // widths. We can add it to the program counter for the start of the
                            // sequence to obtain the final program counter which is just past the
                            // last instruction.
                            let final_program_counter = self
                                .builder
                                .ins()
                                .iadd_imm(self.program_counter_param, self.program_counter_offset);

                            // If there is no next instruction, we jump to the exit block.
                            jump_to_exit(&mut self.builder, step_counter, final_program_counter);
                        }
                    }

                    Outcome::Exception { hook } => {
                        self.builder.switch_to_block(*hook);

                        // Exception outcomes do not increment the step counter, as they don't
                        // count as a successful step.
                        let step_counter = instr_index as i64;

                        // In the case of an exception, the program counter needs to refer to the
                        // instruction that caused the exception.
                        let final_program_counter = instr.program_counter();

                        jump_to_exit(&mut self.builder, step_counter, final_program_counter);
                    }

                    Outcome::KnownBranch { offset, hook } => {
                        self.builder.switch_to_block(*hook);

                        // This is a successful outcome, hence +1 step.
                        let step_counter = instr_index as i64 + 1;

                        // The new program counter is relative to the program counter of the
                        // instruction that is being executed.
                        let final_program_counter = self
                            .builder
                            .ins()
                            .iadd_imm(instr.program_counter(), *offset);

                        jump_to_exit(&mut self.builder, step_counter, final_program_counter);
                    }

                    Outcome::UnknownBranch { destination, hook } => {
                        self.builder.switch_to_block(*hook);

                        // This is a successful outcome, hence +1 step.
                        let step_counter = instr_index as i64 + 1;

                        // The instruction wants to jump somewhere, so we take the destination.
                        let final_program_counter = *destination;

                        jump_to_exit(&mut self.builder, step_counter, final_program_counter);
                    }
                }
            }
        }

        self.builder.seal_all_blocks();
        self.builder.finalize();
    }
}

impl<MC: MemoryConfig> StateContext for SequenceBuilder<'_, MC> {
    type X64 = X64;

    fn read_machine_region<L, const LEN: usize>(&mut self, index: usize) -> Self::X64
    where
        L: MachineCoreProjection<Target = RegionCons<u64, LEN>>,
    {
        super::read_machine_region::<MC, L, LEN>(&mut self.builder, self.core_param, index)
    }

    fn write_machine_region<L, const LEN: usize>(&mut self, index: usize, value: Self::X64)
    where
        L: MachineCoreProjection<Target = RegionCons<u64, LEN>>,
    {
        super::write_machine_region::<MC, L, LEN>(&mut self.builder, self.core_param, index, value)
    }
}
