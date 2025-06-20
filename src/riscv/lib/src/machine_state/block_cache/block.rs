// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Switching of execution strategy for blocks.

pub(crate) mod dispatch;
mod interpreted;
mod jitted;

pub use dispatch::DispatchFn;
pub use dispatch::InlineCompiler;
pub use dispatch::OutlineCompiler;
pub use interpreted::CachedInstruction;
pub use interpreted::Interpreted;
pub use interpreted::InterpretedBlockBuilder;
pub use jitted::Jitted;

use super::run_instr;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerReadWrite;
use crate::traps::EnvironException;
use crate::traps::Exception;

/// Functionality required to construct & execute blocks.
///
/// A block is a sequence of at least one instruction, which may be executed sequentially.
/// Blocks will never contain more than [`super::CACHE_INSTR`] instructions.
pub trait Block<MC: MemoryConfig, M: ManagerBase> {
    /// Block construction may require additional state not kept in storage,
    /// this is then passed as a parameter to [`Block::run_block`].
    ///
    /// `Sized` bound is required to ensure any reference to `BlockBuilder` will be thin -
    /// see [`dispatch::DispatchFn`].
    type BlockBuilder: Default + Sized;

    /// Create a new block instance.
    fn new() -> Self
    where
        M::ManagerRoot: ManagerReadWrite;

    /// Ready a block for construction.
    ///
    /// Previous instructions are removed.
    fn start_block(&mut self);

    /// Push an instruction to the block.
    fn push_instr(&mut self, instr: Instruction)
    where
        M::ManagerRoot: ManagerReadWrite;

    fn num_instr(&self) -> usize;

    /// Invalidate a block, meaning it should no longer be run.
    fn invalidate(&mut self);

    /// Reset a block to the default state, it should no longer be run.
    fn reset(&mut self)
    where
        M::ManagerRoot: ManagerReadWrite;

    /// Returns the underlying slice of instructions stored in the block.
    fn instr(&self) -> &[CachedInstruction<MC, M>];

    /// Run a block against the machine state.
    ///
    /// This function will execute as many instructions from the block as possible
    /// within the given step limit.
    ///
    /// # Safety
    ///
    /// The `block_builder` must be the same as the block builder given to the `compile` call that
    /// (may) have natively compiled this block to machine code.
    ///
    /// This ensures that the builder in question is guaranteed to be alive, for at least as long
    /// as this block may be run.
    unsafe fn run_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        max_steps: usize,
        block_builder: &mut Self::BlockBuilder,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite;
}

fn run_block_inner<MC: MemoryConfig, M: ManagerReadWrite>(
    instr: &[CachedInstruction<MC, M>],
    core: &mut MachineCoreState<MC, M>,
    instr_pc: &mut Address,
    max_steps: usize,
) -> StepManyResult<Exception> {
    let mut result = StepManyResult::ZERO;

    for instr in instr.iter().take(max_steps) {
        match run_instr(instr, core) {
            Ok(ProgramCounterUpdate::Next(width)) => {
                *instr_pc += width as u64;
                core.hart.pc.write(*instr_pc);
                result.steps += 1;
            }

            Ok(ProgramCounterUpdate::Set(instr_pc)) => {
                // Setting the instr_pc implies execution continuing
                // elsewhere - and no longer within the current block.
                core.hart.pc.write(instr_pc);
                result.steps += 1;
                break;
            }

            Err(e) => {
                // Exceptions lead to a new address being set to handle it,
                // with no guarantee of it being the next instruction.
                result.error = Some(e);
                break;
            }
        }
    }

    result
}
