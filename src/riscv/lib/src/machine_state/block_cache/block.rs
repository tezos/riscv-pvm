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
pub use interpreted::Interpreted;
pub use interpreted::InterpretedBlockBuilder;
pub use jitted::Jitted;

use super::CACHE_INSTR;
use super::ICallPlaced;
use super::run_instr;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Atom;
use crate::state_backend::EnrichedCell;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;
use crate::state_backend::Ref;
use crate::traps::EnvironException;
use crate::traps::Exception;

/// State Layout for Blocks
pub type BlockLayout = (Atom<u8>, [Atom<Instruction>; CACHE_INSTR]);

/// Functionality required to construct & execute blocks.
///
/// A block is a sequence of at least one instruction, which may be executed sequentially.
/// Blocks will never contain more than [`CACHE_INSTR`] instructions.
pub trait Block<MC: MemoryConfig, M: ManagerBase>: NewState<M> {
    /// Block construction may require additional state not kept in storage,
    /// this is then passed as a parameter to [`Block::run_block`].
    ///
    /// `Sized` bound is required to ensure any reference to `BlockBuilder` will be thin -
    /// see [`dispatch::DispatchFn`].
    type BlockBuilder: Default + Sized;

    /// Bind the block to the given allocated state.
    fn bind(allocated: AllocatedOf<BlockLayout, M>) -> Self
    where
        M::ManagerRoot: ManagerReadWrite;

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(&'a self) -> AllocatedOf<BlockLayout, F::Output>;

    /// Ready a block for construction.
    ///
    /// Previous instructions are removed.
    fn start_block(&mut self)
    where
        M: ManagerWrite;

    /// Push an instruction to the block.
    fn push_instr(&mut self, instr: Instruction)
    where
        M: ManagerReadWrite;

    fn num_instr(&self) -> usize
    where
        M: ManagerRead;

    /// Invalidate a block, meaning it should no longer be run.
    fn invalidate(&mut self)
    where
        M: ManagerWrite;

    /// Reset a block to the default state, it should no longer be run.
    fn reset(&mut self)
    where
        M: ManagerReadWrite;

    /// Returns the underlying slice of instructions stored in the block.
    fn instr(&self) -> &[EnrichedCell<ICallPlaced<MC, M>, M>]
    where
        M: ManagerRead;

    /// Run a block against the machine state.
    ///
    /// When calling this function, there must be no partial block in progress. To ensure
    /// this, you must always run [`BlockCache::complete_current_block`] prior to fetching
    /// and running a new block.
    ///
    /// There _must_ also be sufficient steps remaining, to execute the block in full.
    ///
    /// # Safety
    ///
    /// The `block_builder` must be the same as the block builder given to the `compile` call that
    /// (may) have natively compiled this block to machine code.
    ///
    /// This ensures that the builder in question is guaranteed to be alive, for at least as long
    /// as this block may be run.
    ///
    /// [`BlockCache::complete_current_block`]: super::BlockCache::complete_current_block
    unsafe fn run_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        block_builder: &mut Self::BlockBuilder,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite;
}

fn run_block_inner<MC: MemoryConfig, M: ManagerReadWrite>(
    instr: &[EnrichedCell<ICallPlaced<MC, M>, M>],
    core: &mut MachineCoreState<MC, M>,
    instr_pc: &mut Address,
) -> StepManyResult<Exception> {
    let mut result = StepManyResult::ZERO;

    for instr in instr.iter() {
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
