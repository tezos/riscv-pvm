// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Interpreted blocks of instructions

use crate::default::ConstDefault;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::block_cache::CACHE_INSTR;
use crate::machine_state::block_cache::ICallPlaced;
use crate::machine_state::block_cache::block::Block;
use crate::machine_state::block_cache::block::BlockLayout;
use crate::machine_state::block_cache::block::run_block_inner;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Cell;
use crate::state_backend::EnrichedCell;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;
use crate::state_backend::Ref;
use crate::traps::EnvironException;

/// Interpreted blocks are built automatically, and require no additional context.
#[derive(Debug, Default)]
pub struct InterpretedBlockBuilder;

/// Blocks that are executed via interpreting the individual instructions.
///
/// Interpreted blocks use the [`EnrichedCell`] mechanism, in order to dispatch
/// opcode to function statically during block construction. This saves time over
/// dispatching on every 'instruction run'. See [`ICall`] for more information.
///
/// [`ICall`]: super::super::ICall
pub struct Interpreted<MC: MemoryConfig, M: ManagerBase> {
    pub(super) instr: [EnrichedCell<ICallPlaced<MC, M>, M>; CACHE_INSTR],
    len_instr: Cell<u8, M>,
}

impl<MC: MemoryConfig, M: ManagerBase> NewState<M> for Interpreted<MC, M> {
    fn new(manager: &mut M) -> Self
    where
        M: ManagerAlloc,
    {
        Self {
            len_instr: Cell::new(manager),
            instr: NewState::new(manager),
        }
    }
}

impl<MC: MemoryConfig, M: ManagerBase> Block<MC, M> for Interpreted<MC, M> {
    type BlockBuilder = InterpretedBlockBuilder;

    fn num_instr(&self) -> usize
    where
        M: ManagerRead,
    {
        self.len_instr.read() as usize
    }

    #[inline]
    fn instr(&self) -> &[EnrichedCell<ICallPlaced<MC, M>, M>]
    where
        M: ManagerRead,
    {
        &self.instr[..self.num_instr()]
    }

    fn invalidate(&mut self)
    where
        M: ManagerWrite,
    {
        self.len_instr.write(0);
    }

    fn push_instr(&mut self, instr: Instruction)
    where
        M: ManagerReadWrite,
    {
        let len = self.len_instr.read();
        self.instr[len as usize].write(instr);
        self.len_instr.write(len + 1);
    }

    fn reset(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.len_instr.write(0);
        self.instr
            .iter_mut()
            .for_each(|lc| lc.write(Instruction::DEFAULT));
    }

    fn start_block(&mut self)
    where
        M: ManagerWrite,
    {
        self.len_instr.write(0);
    }

    fn bind(space: AllocatedOf<BlockLayout, M>) -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        Self {
            len_instr: space.0,
            instr: space.1.map(EnrichedCell::bind),
        }
    }

    fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(&'a self) -> AllocatedOf<BlockLayout, F::Output> {
        (
            self.len_instr.struct_ref::<F>(),
            self.instr.each_ref().map(|entry| entry.struct_ref::<F>()),
        )
    }

    /// # SAFETY
    ///
    /// This function is always safe to call.
    #[inline(always)]
    unsafe fn run_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        mut instr_pc: Address,
        _block_builder: &mut Self::BlockBuilder,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite,
    {
        let mut result = run_block_inner(self.instr(), core, &mut instr_pc);

        if let Some(exc) = result.error {
            if let Err(err) = core.handle_step_result(instr_pc, Err(exc)) {
                return StepManyResult {
                    steps: result.steps,
                    error: Some(err),
                };
            }

            // If we successfully handled an error, need to increment steps one more.
            result.steps += 1;
        }

        StepManyResult {
            steps: result.steps,
            error: None,
        }
    }
}

impl<MC: MemoryConfig, M: ManagerClone> Clone for Interpreted<MC, M> {
    fn clone(&self) -> Self {
        Self {
            len_instr: self.len_instr.clone(),
            instr: self.instr.clone(),
        }
    }
}
