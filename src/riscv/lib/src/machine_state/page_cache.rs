// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Per-page cache of instructions and dispatch targets.
//!
//! The aim is to massively speed up execution two-fold:
//! - first, by bypassing the fetch/parse/run cycle for each instruction
//! - secondly, by allowing hotspots of frequently re-executed sequences of instructions to be
//!   JIT-compiled - thus running even faster.
//!
//! The page cache does this by populating the cache on a per-page basis. This allows the
//! page-cache to _very strictly_ only be populated for pages that have R+X permissions set, but
//! *are not* writable. The pleasant outcome of this is that we no longer have to keep the page
//! cache as part of the PVM state: it is automatically always synced with data memory -- and
//! therefore execution using the page cache is semantically identical to the fetch/parse/run
//! cycle.

pub(crate) mod state;

use std::marker::PhantomData;

use super::MachineCoreState;
use super::ProgramCounterUpdate;
use super::StepManyResult;
use super::instruction::Instruction;
use super::memory;
use super::memory::Address;
use super::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::traps::Exception;

/// Per page, we store exactly the number of instruction halfwords we could fetch from that page's
/// memory.
///
/// This is exactly half the page size - we know the instruction program counter is always
/// halfword-aligned.
const INSTRUCTION_ENTRIES: usize = 1
    << memory::OFFSET_BITS
        .checked_sub(1)
        .expect("OFFSET_BITS is non-zero") as usize;

/// Calculate the offset into a page of instructions from a program counter.
///
/// Since we know the instruction pc to always be halfword-aligned, there are half
/// as many entries as the page size.
const PAGE_OFFSET_MASK: u64 = (1 << memory::OFFSET_BITS) - 1;

/// Instance of the page cache.
#[expect(
    unused,
    reason = "PageCache is not integrated into the rest of the codebase"
)]
pub trait PageCache<MC: MemoryConfig, M: ManagerBase> {
    /// Instantiate a new page cache instance.
    fn new() -> Self;

    /// Retrieve page that is dispatchable against the [`MachineCoreState`].
    fn get_page_dispatch(&mut self, addr: Address) -> Option<PageDispatch<'_, MC, M>>
    where
        M: ManagerRead;

    /// Populate a page with instruction and dispatch information, if the page has R+X permissions.
    fn populate_page(&mut self, address: Address, core: &MachineCoreState<MC, M>) -> Result<(), Exception>
    where
        M: ManagerReadWrite;

    /// Invalidate a range of pages, usually due to the corresponding memory becoming write-able,
    /// or no longer executable.
    fn invalidate_range(&mut self, pages: std::ops::Range<u64>);
}

/// A page containing entrypoints that may then be dispatched against the [`MachineCoreState`].
pub(crate) struct PageDispatch<'a, MC: MemoryConfig, M: ManagerBase> {
    page: &'a [Instruction; INSTRUCTION_ENTRIES],
    _pd: PhantomData<(MC, M)>,
}

impl<MC: MemoryConfig, M: ManagerBase> PageDispatch<'_, MC, M> {
    /// Dispatch instructions from the page against the machine state.
    #[expect(
        unused,
        reason = "PageCache is not integrated into the rest of the codebase"
    )]
    pub(crate) fn run<B>(
        &self,
        core: &mut MachineCoreState<MC, M>,
        _compiler: B,
        mut instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite,
    {
        let mut result = StepManyResult::ZERO;

        // Since we know the instruction pc to always be halfword-aligned, there are half
        // as many entries as the page size.
        let mut instr_offset = (instr_pc & PAGE_OFFSET_MASK) >> 1;

        while max_steps > result.steps && instr_offset < INSTRUCTION_ENTRIES as u64 {
            let instr = &self.page[instr_offset as usize];

            match instr.run(core) {
                Ok(ProgramCounterUpdate::Next(width)) => {
                    instr_pc += width as u64;

                    // we update the offset by half the width, as the offset is halfword aligned
                    instr_offset += (width as u64) >> 1;

                    core.hart.pc.write(instr_pc);
                    result.steps += 1;
                }

                Ok(ProgramCounterUpdate::Set(new_instr_pc)) => {
                    // A jump to a new instruction requires us to exit this loop. The targeted
                    // instruction may not be part of the current block, but also we need to ensure we
                    // don't violate the maximum number of steps allowed to run.
                    core.hart.pc.write(new_instr_pc);
                    result.steps += 1;
                    break;
                }

                Ok(ProgramCounterUpdate::Relative(offset)) => {
                    // While relative jumps are likely to be in the same block, we don't do step
                    // counting within this function, so we can't respect the maximum number of steps
                    // allowed to run.
                    core.hart.pc.write(instr_pc.wrapping_add_signed(offset));
                    result.steps += 1;
                    break;
                }

                Err(exception) => {
                    // Exceptions are handled outside of block execution. So we exit the loop.
                    result.error = Some(exception);
                    break;
                }
            }
        }

        result
    }
}
