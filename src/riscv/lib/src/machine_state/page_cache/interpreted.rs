// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Abstraction over the execution method when dispatching from a [`CodePage`].
//!
//! [`CodePage`]: super::CodePage

use std::marker::PhantomData;
use std::sync::Arc;

use super::INSTRUCTION_ENTRIES;
use super::code_page_entry::CodePageEntry;
use super::code_page_entry::ICall;
use crate::exceptions::Exception;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::listener::MemoryGovernanceListener;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerReadWrite;

/// Interpreted entrypoints are built automatically, and require no additional context.
#[derive(Debug, Default)]
pub struct InterpretedCompiler;

/// Entrypoints that are interpreted only.
#[derive(derive_more::Debug)]
pub struct Interpreted<MC: MemoryConfig, M: ManagerBase> {
    icall: ICall<MC, M>,
}

impl<MC: MemoryConfig, M: ManagerBase> CodePageEntry<MC, M> for Interpreted<MC, M> {
    type Compiler = InterpretedCompiler;

    fn from_instr(instr: Instruction) -> Self
    where
        M: ManagerReadWrite,
    {
        Self {
            icall: ICall::from_instr(instr),
        }
    }

    /// Run an entrypoint in a purely interpreted manner.
    ///
    /// # SAFETY
    ///
    /// This function is always safe to call.
    unsafe fn run_entrypoint(
        page: &Arc<[Self; INSTRUCTION_ENTRIES]>,
        core: &mut MachineCoreState<MC, M>,
        _compiler: &mut Self::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite,
    {
        super::run_code_page_interpreted(page, core, instr_pc, max_steps)
    }
}

impl<MC: MemoryConfig, M: ManagerBase> AsRef<ICall<MC, M>> for Interpreted<MC, M> {
    fn as_ref(&self) -> &ICall<MC, M> {
        &self.icall
    }
}

impl<MC: MemoryConfig, M: ManagerBase> Copy for Interpreted<MC, M> {}

impl<MC: MemoryConfig, M: ManagerBase> Clone for Interpreted<MC, M> {
    fn clone(&self) -> Self {
        *self
    }
}
