// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Abstraction over the execution method when dispatching from a [`CodePage`].
//!
//! [`CodePage`]: super::CodePage

use std::sync::Arc;

use crate::exceptions::Exception;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;

/// Functionality required to dispatch entrypoints in a code page.
///
/// Entrypoints are semantically equivalent to instructions - but may
/// take the opportunity to execute multiple instructions in a row before returning.
pub trait CodePageEntry<MC: MemoryConfig, M: ManagerBase>:
    AsRef<Instruction> + From<Instruction> + std::fmt::Debug + Sized
{
    /// Entrypoints may be just-in-time compiled to more efficient code,
    /// if called frequently.
    ///
    /// We require the compiler capable of doing so to be passed in when
    /// dispatching.
    type Compiler: Clone;

    /// In some cases, different compilers share a context. We want to instantiate the context
    /// once, rather than a new one for every compiler.
    type CompilerContext: Default;

    /// To create a new compiler we use a compiler context.
    ///
    /// TODO RV-847: It isn't great the way this compiler-creation functionality is present both
    /// here and in the `DispatchCompiler` trait. We plan to unify the different compilers to all
    /// implement something like `DispatchCompiler`, which will mean this can certainly be removed
    /// from here; possibly the whole `CodePageEntry` trait can be removed.
    fn new_compiler(context: &Self::CompilerContext) -> Self::Compiler;

    /// Run a code-page entrypoint against the [`MachineCoreState`].
    ///
    /// This will run for up-to `max_steps`, but never over.
    ///
    /// This entrypoint may be either interpreted or compiled. If compiled, compilation occurs
    /// using the compiler contained in the `PageEntry`.
    fn run_entrypoint(
        page: &Arc<super::state::PageEntry<Self, Self::Compiler>>,
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerRead + ManagerWrite;

    /// For analysis in tests, the entrypoint keeps track of the number of times it has been
    /// called.
    #[cfg(test)]
    fn called_times(&self) -> usize;
}
