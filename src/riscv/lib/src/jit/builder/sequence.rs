// SPDX-FileCopyrightText: 2025-2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Builder for sequences of instructions
//!
//! This module provides the infrastructure for building and compiling sequences of RISC-V
//! instructions using Cranelift IR. The [sequence builder] coordinates the compilation of multiple
//! instructions into a single JIT function, managing control flow, program counter updates,
//! and [instruction outcomes].
//!
//! [sequence builder]: SequenceBuilder
//! [instruction outcomes]: crate::jit::builder::instruction::InstructionOutcomes

use cranelift::codegen::Context;
use cranelift::codegen::ir::BlockArg;
use cranelift::prelude::AbiParam;
use cranelift::prelude::Block;
use cranelift::prelude::EntityRef;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::FunctionBuilderContext;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::IntCC::UnsignedGreaterThanOrEqual;
use cranelift::prelude::Variable;
use cranelift::prelude::isa::TargetFrontendConfig;
use cranelift::prelude::types::I64;
use cranelift_jit::JITModule;
use cranelift_module::Module;
use octez_riscv_data::mode::Normal;

use crate::jit::builder::control_flow_graph::ControlFlowGraph;
use crate::jit::builder::control_flow_graph::NodeInfo;
use crate::jit::builder::control_flow_graph::OutcomeData;
use crate::jit::builder::instruction::InstructionBuilder;
use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::outcome_map::ExitKind;
use crate::jit::builder::outcome_map::TargetInstrLoc;
use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::jit::state_access::ExceptionCode;
use crate::jit::state_access::JsaCalls;
use crate::machine_state::MachineCoreState;
use crate::machine_state::hart_state::ProgramCounterProj;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::entrypoint::Page;
use crate::machine_state::registers::XValue;
use crate::parser::instruction::InstrWidth;
use crate::state_context::PcWriteContext;

const STEPS_REMAINING_VAR_ID: usize = 0;

/// Builder for an instruction sequence
pub struct SequenceBuilder<'jit, D, MC: MemoryConfig> {
    /// Target configuration for the JIT module
    target_config: TargetFrontendConfig,

    /// IR builder
    builder: FunctionBuilder<'jit>,

    /// External function call manager
    ext_calls: JsaCalls<MC>,

    /// Function entry block
    entry_block: Block,

    /// Standard exit block to jump to when finishing the sequence
    exit_block: Block,

    /// Exit block to jump to when falling back to interpreted mode.
    fallback_interpreted_block: Block,

    /// Parameter pointing to the code page
    page_param: Pointer<Page<D, MC>>,

    /// Parameter pointing to the `MachineCoreState`
    core_param: Pointer<MachineCoreState<MC, Normal>>,

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

impl<'jit, D, MC: MemoryConfig> SequenceBuilder<'jit, D, MC> {
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
        // Returns:
        //   - `steps`: Number of steps executed in the sequence
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(ptr_type));
        context.func.signature.params.push(AbiParam::new(I64));
        context.func.signature.params.push(AbiParam::new(I64));
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

        // SAFETY: `JitFn` accepts a `&Page` as the 1st parameter.
        let page_param = unsafe {
            let raw_value = builder.block_params(param_block)[0];
            Pointer::<Page<D, MC>>::from_raw(raw_value)
        };

        // SAFETY: `JitFn` accepts a `&mut MachineCoreState<MC, Normal>` as the 2nd parameter.
        let core_param = unsafe {
            let raw_value = builder.block_params(param_block)[1];
            Pointer::<MachineCoreState<MC, Normal>>::from_raw(raw_value)
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

        #[cfg(test)]
        {
            let entrypoint = builder.ins().iconst(I64, program_counter as i64);
            // SAFETY: The value is constructed from an Address type.
            let entrypoint = unsafe { Value::<Address>::from_raw(entrypoint) };
            // Record each JIT function invocation from within the generated function.
            ext_calls.record_jit_call(&mut builder, page_param, entrypoint);
        }

        // The entry block is where we will eventually transition to the first instruction basic
        // block. The function's entry block (`param_block` for our purposes) will directly jump to
        // this `entry_block`.
        let entry_block = builder.create_block();
        builder.ins().jump(entry_block, []);

        let target_config = module.target_config();
        let exit_block = builder.create_block();

        // The fallback exit block is used when we need to switch to interpreted mode and then exit.
        // It requires two parameters: the current PC and the remaining steps.
        let fallback_interpreted_block = builder.create_block();
        builder.append_block_param(fallback_interpreted_block, I64);
        builder.append_block_param(fallback_interpreted_block, I64);

        Self {
            target_config,
            builder,
            ext_calls,
            entry_block,
            exit_block,
            fallback_interpreted_block,
            page_param,
            core_param,
            program_counter,
            program_counter_offset: 0,
            max_steps_param,
            result_param,
            steps_remaining,
        }
    }

    /// The exit block is used to write the program counter back to the machine core state, as
    /// well as returning from the JIT function.
    ///
    /// This function should be called once all blocks leading to the exit block have been built.
    fn fill_exit_block(&mut self) {
        self.builder.seal_block(self.exit_block);
        self.builder.switch_to_block(self.exit_block);

        // SAFETY: We're declaring the value as a `I64` which has the same representation of
        // `Address`.
        let final_program_counter = unsafe {
            let final_program_counter = self.builder.append_block_param(self.exit_block, I64);
            Value::<Address>::from_raw(final_program_counter)
        };

        self.pc_write(final_program_counter);

        let steps_remaining = self.builder.use_var(self.steps_remaining);
        let max_steps = self.max_steps_param.to_value();
        let steps = self.builder.ins().isub(max_steps, steps_remaining);

        #[cfg(test)]
        // SAFETY: The value is constructed from a valid u64 type `steps`.
        unsafe {
            self.ext_calls.debug::<u64>(
                &mut self.builder,
                c"exiting. took steps: ",
                Value::from_raw(steps),
            );
        }

        self.builder.ins().return_(&[steps]);
    }

    /// IR for calling the interpreted mode fallback function when unable to continue in JIT mode.
    /// This exit block writes the current PC back to the machine core state before calling the
    /// interpreter, and returns the total number of steps executed in JIT and interpreted mode.
    ///
    /// This function should be called once all blocks leading to this exiting block have been built.
    // TODO RV-842: Introduce counter update here for number of calls to fallback to interpreter.
    fn fill_fallback_interpreted_block(&mut self) {
        self.builder.seal_block(self.fallback_interpreted_block);
        self.builder
            .switch_to_block(self.fallback_interpreted_block);
        self.builder.set_cold_block(self.fallback_interpreted_block);

        // Before calling the fallback function, we must commit the current PC to the machine core state.

        // SAFETY: The first parameter of the block is the `current_pc`, which is an `Address`.
        let current_pc = unsafe {
            let raw_value = self.builder.block_params(self.fallback_interpreted_block)[0];
            Value::<Address>::from_raw(raw_value)
        };

        self.pc_write(current_pc);

        // SAFETY: The second parameter of the block is the `steps_remaining` value, which is a `usize`.
        let fallback_max_steps = unsafe {
            let raw_value = self.builder.block_params(self.fallback_interpreted_block)[1];
            Value::<usize>::from_raw(raw_value)
        };

        #[cfg(test)]
        {
            let entrypoint = self.builder.ins().iconst(I64, self.program_counter as i64);
            // SAFETY: The value is constructed from an Address type.
            let entrypoint = unsafe { Value::<Address>::from_raw(entrypoint) };
            // Record each fallback to interpreted mode from within the generated function.
            self.ext_calls.record_fallback_to_interpreter(
                &mut self.builder,
                self.page_param,
                entrypoint,
            );
        }

        // Call the interpreter fallback function. This will continue up to `fallback_max_steps` steps
        // in interpreted mode. This syncs the PC to the `machine_core_state` as it executes.
        let final_steps_remaining = self
            .ext_calls
            .run_interpreter_fallback(
                &mut self.builder,
                self.page_param,
                self.core_param,
                fallback_max_steps,
                self.result_param,
            )
            .to_value();

        let steps = self
            .builder
            .ins()
            .isub(self.max_steps_param.to_value(), final_steps_remaining);

        // Exit the JIT function without writing the PC, as it will already be synced.
        self.builder.ins().return_(&[steps]);
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

    /// Decrement 'steps_remaining' by the number of steps taken in the sequence.
    fn insert_step_update_ir(&mut self, step_delta: i64) {
        if step_delta == 0 {
            return;
        }

        let step_delta = self.builder.ins().iconst(I64, step_delta);

        let steps_remaining = self.builder.use_var(self.steps_remaining);
        let result_steps_remaining = self.builder.ins().isub(steps_remaining, step_delta);

        self.builder
            .def_var(self.steps_remaining, result_steps_remaining);
    }

    /// Insert a budget-check into the control flow IR for the current block being built.
    fn insert_budget_check_ir(&mut self, budget: u64, exit_delta: i64, exit_pc: Value<Address>) {
        let steps_remaining = self.builder.use_var(self.steps_remaining);

        let steps_remaining = self.builder.ins().iadd_imm(steps_remaining, -exit_delta);
        let budget = self.builder.ins().iconst(I64, budget as i64);
        let is_enough_budget =
            self.builder
                .ins()
                .icmp(UnsignedGreaterThanOrEqual, steps_remaining, budget);

        let continue_block = self.builder.create_block();

        self.builder.ins().brif(
            is_enough_budget,
            continue_block,
            &[],
            self.fallback_interpreted_block,
            &[
                BlockArg::Value(exit_pc.to_value()),
                BlockArg::Value(steps_remaining),
            ],
        );

        self.builder.seal_block(continue_block);
        self.builder.switch_to_block(continue_block);

        #[cfg(test)]
        {
            let entrypoint = self.builder.ins().iconst(I64, self.program_counter as i64);
            // SAFETY: The value is constructed from an Address type.
            let entrypoint = unsafe { Value::<Address>::from_raw(entrypoint) };
            // Record each budget check pass from within the generated function.
            self.ext_calls
                .record_budget_check_pass(&mut self.builder, self.page_param, entrypoint);
        }
    }

    /// Insert a budget-check into the control flow IR for the current entrypoint.
    fn build_entry_outcome_ir(
        &mut self,
        entry_node: &NodeInfo<OutcomeData, Block>,
        seq_min_budget: u64,
    ) {
        self.builder.switch_to_block(self.entry_block);

        // SAFETY: We are constructing this value directly from an Address type.
        let entry_node_addr = unsafe {
            Value::<Address>::from_discriminant(
                &self.target_config,
                &mut self.builder,
                entry_node.location as i64,
            )
        };

        self.insert_budget_check_ir(seq_min_budget, 0, entry_node_addr);
        entry_node.run_instruction(&mut self.builder);
    }

    /// Insert step counter updates and budget checks into the control flow IR for the
    /// current outcome.
    fn build_outcome_ir(
        &mut self,
        outcome_data: &OutcomeData,
        outcome_target: TargetInstrLoc,
        cfg: &ControlFlowGraph<OutcomeData, Block>,
    ) {
        self.builder.switch_to_block(outcome_data.hook());

        if let Some(step_update) = outcome_data.get_step_delta() {
            self.insert_step_update_ir(step_update as i64);
        }

        if let Some(budget) = outcome_data.get_budget_check() {
            let exit_pc = outcome_data.exit_pc(&mut self.builder, &self.target_config);
            let exit_delta = outcome_data
                .get_exit_delta()
                .expect("Any budget check must have an exit delta.");
            self.insert_budget_check_ir(budget as u64, exit_delta as i64, exit_pc);
        }

        match outcome_target {
            TargetInstrLoc::Internal(target) => {
                // Continue to the target instruction.
                cfg.nodes[target].run_instruction(&mut self.builder);
            }
            TargetInstrLoc::Exit(_) => {
                let exit_pc = outcome_data.exit_pc(&mut self.builder, &self.target_config);
                self.builder
                    .ins()
                    .jump(self.exit_block, &[BlockArg::Value(exit_pc.to_value())]);
            }
        }
    }

    /// Finish building the sequence.
    pub fn finish(mut self, instrs: &[LoweredInstruction]) {
        let node_infos: Vec<NodeInfo<OutcomeData, Block>> = instrs
            .iter()
            .enumerate()
            .map(|(idx, instr)| {
                NodeInfo::<OutcomeData, Block>::from_lowered_instruction(instr, idx == 0)
            })
            .collect();

        let cfg = ControlFlowGraph::<OutcomeData, Block>::new(node_infos.iter());

        let (step_updates, exit_deltas) = cfg.find_step_counter_updates();
        let sequence_budget = cfg.annotate_budget_checks(&exit_deltas);

        // Insert step counter update information into the outcome data.
        for (_outcome_id, outcome) in step_updates.iter() {
            let Some(update) = outcome.data() else {
                // The analysis determined there is nothing to do for this edge.
                continue;
            };

            let step_delta = match outcome.to() {
                TargetInstrLoc::Exit(ExitKind::Exception) => update.exception_delta(),
                TargetInstrLoc::Internal(_) | TargetInstrLoc::Exit(ExitKind::Normal) => {
                    update.success_delta()
                }
            };

            update.edge().info.set_step_delta(step_delta);
        }

        // Insert budget check information into the outcome data.
        for (_, outcome) in sequence_budget.budget_checks().iter() {
            let Some(budget_check) = outcome.data() else {
                // The analysis determined there is nothing to do for this edge.
                continue;
            };

            let bc_info = &budget_check.edge().info;
            bc_info.set_budget_check(budget_check.budget());
            bc_info.set_exit_delta(budget_check.exit_delta());
        }

        for (_, outcome) in cfg.outcomes.iter() {
            let Some(edge) = outcome.data() else {
                // This is an entry outcome, so it is handled with an entry budget check.
                let &entry_node_id = outcome
                    .to()
                    .as_internal()
                    .expect("Entry outcomes must go to a node.");
                let entry_node = cfg.nodes[entry_node_id];
                self.build_entry_outcome_ir(entry_node, sequence_budget.min_budget() as u64);

                continue;
            };

            let outcome_data = &edge.info;
            self.build_outcome_ir(outcome_data, outcome.to(), &cfg);
        }

        self.fill_exit_block();
        self.fill_fallback_interpreted_block();

        self.builder.seal_all_blocks();
        self.builder.finalize();
    }
}

impl<D, MC: MemoryConfig> PcWriteContext for SequenceBuilder<'_, D, MC> {
    type Value<R> = Value<R>;

    fn pc_write(&mut self, value: Self::Value<XValue>) {
        super::write_proj::<MC, ProgramCounterProj>(
            &self.target_config,
            &mut self.builder,
            self.core_param,
            (),
            value,
        );
    }
}
