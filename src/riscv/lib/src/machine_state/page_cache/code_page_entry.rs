// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Abstraction over the execution method when dispatching from a [`CodePage`].
//!
//! [`CodePage`]: super::CodePage

use std::sync::Arc;

use super::INSTRUCTION_ENTRIES;
use crate::exceptions::Exception;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::instruction::RunInstr;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerReadWrite;

#[derive(derive_more::Debug)]
pub struct ICall<MC: MemoryConfig, M: ManagerBase> {
    instr: Instruction,
    #[debug(skip)]
    run_fun: RunInstr<MC, M>,
}

impl<MC: MemoryConfig, M: ManagerBase> ICall<MC, M> {
    pub(super) fn from_instr(instr: Instruction) -> Self
    where
        M: ManagerReadWrite,
    {
        Self {
            run_fun: instr.opcode.to_run(),
            instr,
        }
    }

    #[inline(always)]
    pub(super) fn run_instr(
        &self,
        core: &mut MachineCoreState<MC, M>,
    ) -> Result<ProgramCounterUpdate<Address>, Exception>
    where
        M: ManagerReadWrite,
    {
        (self.run_fun)(&self.instr.args, core)
    }
}

impl<MC: MemoryConfig, M: ManagerBase> AsRef<Instruction> for ICall<MC, M> {
    fn as_ref(&self) -> &Instruction {
        &self.instr
    }
}

impl<MC: MemoryConfig, M: ManagerBase> Copy for ICall<MC, M> {}

impl<MC: MemoryConfig, M: ManagerBase> Clone for ICall<MC, M> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Functionality required to dispatch entrypoints in a code page.
///
/// Entrypoints are semantically equivalent to instructions - but may
/// take the opportunity to execute multiple instructions in a row before returning.
pub trait CodePageEntry<MC: MemoryConfig, M: ManagerBase>:
    AsRef<ICall<MC, M>> + std::fmt::Debug + Sized
{
    /// Entrypoints may be just-in-time compiled to more efficient code,
    /// if called frequently.
    ///
    /// We require the compiler capable of doing so to be passed in when
    /// dispatching.
    type Compiler: Default;

    fn from_instr(instr: Instruction) -> Self
    where
        M: ManagerReadWrite;

    /// Run a code-page entrypoint against the [`MachineCoreState`].
    ///
    /// This will run for up-to `max_steps`, but never over.
    ///
    /// # SAFETY
    ///
    /// The `compiler` must always be the same instance as passed to any
    /// call to `run_entrypoint` within the same page, for the lifetime of that page. This ensures
    /// that the compiler in question is guaranteed to be alive, for as long as this entrypoint may
    /// be run.
    unsafe fn run_entrypoint(
        page: &Arc<[Self; INSTRUCTION_ENTRIES]>,
        core: &mut MachineCoreState<MC, M>,
        compiler: &mut Self::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite;
}
