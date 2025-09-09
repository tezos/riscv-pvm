// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Abstraction over the execution method when dispatching from a [`CodePage`].
//!
//! [`CodePage`]: super::CodePage

// TODO: RV-765: add support for inline JIT

use super::INSTRUCTION_ENTRIES;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::block_cache::block::InterpretedBlockBuilder;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerReadWrite;
use crate::traps::Exception;

/// Functionality required to dispatch entrypoints in a code page.
///
/// Entrypoints are semantically equivalent to instructions - but may
/// take the opportunity to execute multiple instructions in a row before returning.
pub trait CodePageEntry: AsRef<Instruction> + From<Instruction> + std::fmt::Debug + Sized {
    /// Entrypoints may be just-in-time compiled to more efficient code,
    /// if called frequently.
    ///
    /// We require the compiler capable of doing so to be passed in when
    /// dispatching.
    type Compiler;

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
    unsafe fn run_entrypoint<MC, M>(
        page: &mut [Self; INSTRUCTION_ENTRIES],
        core: &mut MachineCoreState<MC, M>,
        compiler: &mut Self::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        MC: MemoryConfig,
        M: ManagerReadWrite;
}

impl CodePageEntry for Instruction {
    type Compiler = InterpretedBlockBuilder;

    /// Run an entrypoint in a purely interpreted manner.
    ///
    /// # SAFETY
    ///
    /// This function is always safe to call.
    unsafe fn run_entrypoint<MC, M>(
        page: &mut [Self; INSTRUCTION_ENTRIES],
        core: &mut MachineCoreState<MC, M>,
        _compiler: &mut Self::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        MC: MemoryConfig,
        M: ManagerReadWrite,
    {
        super::run_code_page_interpreted(page, core, instr_pc, max_steps)
    }
}

impl AsRef<Instruction> for Instruction {
    fn as_ref(&self) -> &Instruction {
        self
    }
}
