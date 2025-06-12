// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Interpreted blocks of instructions

use crate::default::ConstDefault;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::block_cache::CACHE_INSTR;
use crate::machine_state::block_cache::ICall;
use crate::machine_state::block_cache::block::Block;
use crate::machine_state::block_cache::block::run_block_inner;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;
use crate::traps::EnvironException;

/// Interpreted blocks are built automatically, and require no additional context.
#[derive(Debug, Default)]
pub struct InterpretedBlockBuilder;

/// An instruction in the cache
pub struct CachedInstruction<MC: MemoryConfig, M: ManagerBase> {
    /// Run function for the instruction
    pub runner: ICall<MC, M::ManagerRoot>,

    /// Instruction itself
    pub instr: Instruction,
}

impl<MC: MemoryConfig, M: ManagerBase> CachedInstruction<MC, M> {
    /// Creates a new cached instruction with the given instruction.
    pub fn new(instr: Instruction) -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        let runner = ICall::from(&instr);
        Self { instr, runner }
    }
}

impl<MC: MemoryConfig, M: ManagerBase> Clone for CachedInstruction<MC, M> {
    fn clone(&self) -> Self {
        Self {
            instr: self.instr,
            runner: self.runner,
        }
    }
}

/// Blocks that are executed via interpreting the individual instructions
pub struct Interpreted<MC: MemoryConfig, M: ManagerBase> {
    instr: [CachedInstruction<MC, M>; CACHE_INSTR],
    len_instr: u8,
}

impl<MC: MemoryConfig, M: ManagerBase> Block<MC, M> for Interpreted<MC, M> {
    type BlockBuilder = InterpretedBlockBuilder;

    fn num_instr(&self) -> usize
    where
        M: ManagerRead,
    {
        self.len_instr as usize
    }

    #[inline]
    fn instr(&self) -> &[CachedInstruction<MC, M>]
    where
        M: ManagerRead,
    {
        &self.instr[..self.num_instr()]
    }

    fn invalidate(&mut self)
    where
        M: ManagerWrite,
    {
        self.len_instr = 0;
    }

    fn push_instr(&mut self, instr: Instruction)
    where
        M: ManagerReadWrite,
    {
        self.instr[self.len_instr as usize] = CachedInstruction::new(instr);
        self.len_instr = self.len_instr.saturating_add(1);
    }

    fn reset(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.len_instr = 0;
        self.instr
            .iter_mut()
            .for_each(|entry| *entry = CachedInstruction::new(Instruction::DEFAULT));
    }

    fn start_block(&mut self)
    where
        M: ManagerWrite,
    {
        self.len_instr = 0;
    }

    fn new() -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        Self {
            len_instr: 0,
            instr: std::array::from_fn(|_| CachedInstruction::new(Instruction::DEFAULT)),
        }
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
            len_instr: self.len_instr,
            instr: self.instr.clone(),
        }
    }
}
