// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Abstraction over the execution method when dispatching from a [`CodePage`].
//!
//! [`CodePage`]: super::CodePage

use std::marker::PhantomData;
use std::sync::Arc;

use super::code_page_entry::CodePageEntry;
use crate::exceptions::Exception;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::router::RouterEq;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;

/// Interpreted entrypoints are built automatically, and require no additional context.
#[derive(Debug, Default, Clone)]
pub struct InterpretedCompiler;

/// Since [`InterpretedCompiler`] does no compilation at all, [`router_eq`] should always return
/// false, so that the ranges in the router are kept as small as possible. This helps reduce the
/// number of pages that could get unneccessarily dropped.
///
/// [`router_eq`]: crate::machine_state::page_cache::router::RouterEq::router_eq
impl RouterEq for InterpretedCompiler {
    fn router_eq(&self, _other: &Self) -> bool {
        false
    }
}

/// Entrypoints that are interpreted only.
#[derive(derive_more::Debug)]
pub struct Interpreted<MC, M> {
    instruction: Instruction,
    #[cfg(test)]
    call_count: std::cell::Cell<usize>,
    #[debug(skip)]
    _pd: PhantomData<(MC, M)>,
}

impl<MC: MemoryConfig, M: ManagerBase> CodePageEntry<MC, M> for Interpreted<MC, M> {
    type Compiler = InterpretedCompiler;

    type CompilerContext = ();

    fn new_compiler(_context: &()) -> Self::Compiler {
        InterpretedCompiler
    }

    /// Run an entrypoint in a purely interpreted manner.
    fn run_entrypoint(
        page: &Arc<super::state::PageEntry<Self, InterpretedCompiler>>,
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerRead + ManagerWrite,
    {
        #[cfg(test)]
        {
            let instr_offset = super::address_to_halfword_index(instr_pc);
            page.entries[instr_offset]
                .call_count
                .update(|x| x.saturating_add(1));
        }

        super::run_code_page_interpreted(&page.entries, core, instr_pc, max_steps)
    }

    #[cfg(test)]
    fn called_times(&self) -> usize {
        self.call_count.get()
    }
}

impl<MC, M> From<Instruction> for Interpreted<MC, M> {
    fn from(instruction: Instruction) -> Self {
        Self {
            instruction,
            #[cfg(test)]
            call_count: std::cell::Cell::new(0),
            _pd: PhantomData,
        }
    }
}

impl<MC, M> AsRef<Instruction> for Interpreted<MC, M> {
    fn as_ref(&self) -> &Instruction {
        &self.instruction
    }
}

impl<MC, M> Clone for Interpreted<MC, M> {
    fn clone(&self) -> Self {
        Self {
            instruction: self.instruction,
            #[cfg(test)]
            call_count: self.call_count.clone(),
            _pd: self._pd,
        }
    }
}
