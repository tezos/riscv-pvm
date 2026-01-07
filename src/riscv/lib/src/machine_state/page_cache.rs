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

pub(crate) mod code_page_entry;
pub(crate) mod dispatch;
mod empty;
pub(crate) mod interpreted;
pub(crate) mod jitted;
mod router;
pub(crate) mod state;

pub use code_page_entry::CodePageEntry;
pub use dispatch::DispatchTarget;
pub use dispatch::InlineCompiler;
pub use dispatch::OutlineCompiler;
pub use empty::EmptyPageCache;
pub use interpreted::Interpreted;
pub use interpreted::InterpretedCompiler;
pub use jitted::Jitted;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use state::PageCacheImpl;

use super::MachineCoreState;
use super::ProgramCounterUpdate;
use super::StepManyResult;
use super::instruction::Instruction;
use super::memory;
use super::memory::Address;
use super::memory::MemoryConfig;
use super::memory::address_to_page_offset;
use super::memory::listener::MemoryGovernanceListener;
use crate::exceptions::Exception;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;

/// Type alias for the default [`PageCache`] implementation with interpreted dispatch only.
///
/// This is only usable when the [`Normal`] backend mode is selected.
pub type PageCacheInterpreted<MC> = PageCacheImpl<Interpreted<MC, Normal>, MC, Normal>;

/// Type alias for the default [`PageCache`] implementation with inline jit enabled.
///
/// This is only usable when the [`Normal`] backend mode is selected.
pub type PageCacheInlineJit<MC> = PageCacheImpl<Jitted<InlineCompiler, MC>, MC, Normal>;

/// Type alias for the default [`PageCache`] implementation with outline jit enabled.
///
/// This is only usable when the [`Normal`] backend mode is selected.
pub type PageCacheOutlineJit<MC> = PageCacheImpl<Jitted<OutlineCompiler<MC>, MC>, MC, Normal>;

/// Per page, we store exactly the number of instruction halfwords we could fetch from that page's
/// memory.
///
/// This is exactly half the page size - we know the instruction program counter is always
/// halfword-aligned.
const INSTRUCTION_ENTRIES: usize = 1
    << memory::OFFSET_BITS
        .get()
        .checked_sub(1)
        .expect("OFFSET_BITS is non-zero") as usize;

/// Isolate the the 'halfword-index' into a page of an address.
///
/// We only store entries for halfword-aligned addresses, since pc is always halfword aligned.
#[inline]
pub fn address_to_halfword_index(address: Address) -> usize {
    address_to_page_offset(address) >> 1
}

/// Instance of the page cache.
///
/// A page cache is a mapping from _page indexes_ to a page of entrypoints.
///
/// Specifically, a page index is the 'page number':
/// - address `0` is the start of *page 1*
/// - address `PAGE_SIZE` is the start of *page 2*
/// - address `MC::TOTAL_BYTES - 1` is the end of the page at `PAGES - 1`.
///
/// Every page index uniquely corresponds to a slot in the page cache - which may
/// or may not be populated. Specifically, pages that are writable or not-executable
/// will never be populated. Pages that are executable (and not-writable) _may_ be
/// populated.
///
/// Page entrypoints exist at the start of each _halfword_ within a page slot. Since the
/// instruction pc is always halfword-aligned, a populated
pub trait PageCache<MC: MemoryConfig, M: Mode>: MemoryGovernanceListener {
    /// Instantiate a new page cache instance.
    fn new() -> Self;

    /// Retrieve code page that is dispatchable against the [`MachineCoreState`]. If found, such a
    /// page will contain the code for `addr`.
    fn get_code_page(&mut self, addr: Address) -> Option<impl CodePage<'_, MC, M>>
    where
        M: ManagerRead;

    /// Populate a page with instruction and dispatch information, if the page has R+X permissions only.
    fn populate_page(&mut self, address: Address, core: &MachineCoreState<MC, M>)
    where
        M: ManagerRead + ManagerWrite;
}

/// A code page contains instructions that can be executed
/// against the [machine state].
///
/// [machine state]: MachineCoreState
pub trait CodePage<'a, MC: MemoryConfig, M: Mode> {
    /// Execute instructions from a code page against the
    /// machine state.
    ///
    /// Execution begins at the offset into the page given by
    /// the `instr_pc` (program counter) - and will run for
    /// up to `max_steps` steps.
    fn run(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerRead + ManagerWrite;
}

/// In interpreted mode, run up to `max_steps` instructions starting from `instr_pc` in the
/// given `code_page`.
pub(crate) fn run_code_page_interpreted<I, MC, M>(
    code_page: &[I; INSTRUCTION_ENTRIES],
    core: &mut MachineCoreState<MC, M>,
    mut instr_pc: Address,
    max_steps: usize,
) -> StepManyResult<Exception>
where
    I: AsRef<Instruction>,
    MC: MemoryConfig,
    M: ManagerRead + ManagerWrite,
{
    let mut result = StepManyResult::ZERO;

    let page_offset = address_to_page_offset(instr_pc);

    // Since we know the instruction pc to always be halfword-aligned, there are half
    // as many entries as the page size.
    let mut instr_offset = page_offset >> 1;

    while max_steps > result.steps && instr_offset < INSTRUCTION_ENTRIES {
        let instr = code_page[instr_offset].as_ref();

        match instr.run(core) {
            Ok(ProgramCounterUpdate::Next(width)) => {
                instr_pc += width as u64;

                // we update the offset by half the width, as the offset is halfword aligned
                instr_offset += (width as usize) >> 1;

                core.hart.pc.write(instr_pc);
                result.steps += 1;
            }

            Ok(ProgramCounterUpdate::Set(new_instr_pc)) => {
                // A jump to a new instruction requires us to exit this loop. The targeted
                // instruction may not be part of the current page, but either way we should
                // allow the target of the jump to be considered as a potential hot-spot.
                core.hart.pc.write(new_instr_pc);
                result.steps += 1;
                break;
            }

            Ok(ProgramCounterUpdate::Relative(offset)) => {
                // While relative jumps are likely to be in the same page, we exit at this point to allow
                // the jump target to be considered as a potential hot-spot.
                core.hart.pc.write(instr_pc.wrapping_add_signed(offset));
                result.steps += 1;
                break;
            }

            Err(exception) => {
                // Exceptions are handled outside of interpreted entrypoint dispatch. So we exit the loop.
                result.error = Some(exception);
                break;
            }
        }
    }

    result
}
