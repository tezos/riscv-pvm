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
pub(crate) mod interpreted;
pub(crate) mod jitted;
pub(crate) mod state;

use std::sync::Arc;

pub use code_page_entry::CodePageEntry;
pub use dispatch::DispatchTarget;
pub use dispatch::InlineCompiler;
pub use dispatch::OutlineCompiler;
pub use interpreted::Interpreted;
pub use interpreted::InterpretedCompiler;
pub use jitted::Jitted;

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
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;

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
pub trait PageCache<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase>:
    MemoryGovernanceListener
{
    /// Instantiate a new page cache instance.
    fn new() -> Self;

    /// Retrieve code page that is dispatchable against the [`MachineCoreState`]. If found, such a
    /// page will contain the code for `addr`.
    fn get_code_page(&mut self, addr: Address) -> Option<CodePage<'_, CPE>>
    where
        M: ManagerRead;

    /// Populate a page with instruction and dispatch information, if the page has R+X permissions only.
    fn populate_page(&mut self, address: Address, core: &MachineCoreState<MC, M>)
    where
        M: ManagerReadWrite;

    /// Invalidate a range of pages, usually due to the corresponding memory becoming write-able,
    /// or no longer executable.
    fn invalidate_pages(&mut self, pages: std::ops::RangeInclusive<u64>);
}

/// A page containing code that may then be run against the [`MachineCoreState`].
#[derive(Debug)]
pub struct CodePage<'a, CPE> {
    page: &'a Arc<[CPE; INSTRUCTION_ENTRIES]>,
}

impl<CPE> CodePage<'_, CPE> {
    /// Run a code page against the machine state.
    ///
    /// # SAFETY
    ///
    /// The `compiler` must always be the same compiler when dispatching a given page (for the
    /// lifetime of that page).
    ///
    /// This ensures dispatching can ensure the compiler's state is kept alive.
    pub(crate) unsafe fn run<MC, M>(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        compiler: &mut CPE::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        CPE: CodePageEntry<MC, M>,
        MC: MemoryConfig,
        M: ManagerReadWrite,
    {
        // SAFETY: the compiler remains the same for the lifetime of the page this code-page
        // references
        unsafe { CPE::run_entrypoint(self.page, core, compiler, instr_pc, max_steps) }
    }
}

fn run_code_page_interpreted<I, MC, M>(
    code_page: &[I; INSTRUCTION_ENTRIES],
    core: &mut MachineCoreState<MC, M>,
    mut instr_pc: Address,
    max_steps: usize,
) -> StepManyResult<Exception>
where
    I: AsRef<Instruction>,
    MC: MemoryConfig,
    M: ManagerReadWrite,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::CodePage;
    use super::INSTRUCTION_ENTRIES;
    use super::interpreted::Interpreted;
    use crate::array_utils::boxed_from_fn;
    use crate::backend_test;
    use crate::exceptions::Exception;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::listener::NoopMemoryGovernanceListener;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::registers::nz;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;
    use crate::state_backend::test_helpers::TestBackendFactory;

    struct DispatchTest<'a, F: TestBackendFactory> {
        state: &'a std::cell::RefCell<MachineCoreState<M4K, F>>,
        dispatch: &'a std::cell::RefCell<CodePage<'a, Interpreted<M4K, F>>>,
        pc_addr: u64,
        max_steps: usize,
        expected_steps: usize,
        expected_pc_addr: u64,
        expected_a0: u64,
        expected_exception: Option<Exception>,
    }

    fn run_test<F: TestBackendFactory>(test: DispatchTest<'_, F>) {
        let mut state = test.state.borrow_mut();
        state.reset(NoopMemoryGovernanceListener);

        // SAFETY: interpreted mode is always safe to call
        let res = unsafe {
            test.dispatch.borrow_mut().run(
                &mut state,
                &mut InterpretedCompiler,
                test.pc_addr,
                test.max_steps,
            )
        };

        assert_eq!(res.steps, test.expected_steps);
        assert_eq!(res.error, test.expected_exception);

        assert_eq!(state.hart.pc.read(), test.expected_pc_addr);
        assert_eq!(state.hart.xregisters.read_nz(nz::a0), test.expected_a0);
    }

    backend_test!(page_dispatch_respects_max_steps_compressed, F, {
        let mut page_entry: Arc<[Interpreted<_, _>; INSTRUCTION_ENTRIES]> = boxed_from_fn(|| {
            Interpreted::from(Instruction::new_addi(
                nz::a0,
                nz::a0,
                5,
                InstrWidth::Compressed,
            ))
        })
        .into();

        let dispatch = &std::cell::RefCell::new(CodePage {
            page: &mut page_entry,
        });

        let state = MachineCoreState::<M4K, F>::new();
        let state = &std::cell::RefCell::new(state);

        let page_size = memory::PAGE_SIZE.get();

        // run, no branching, within page
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: 0,
            max_steps: 10,
            expected_steps: 10,
            expected_pc_addr: 10 * InstrWidth::Compressed as u64,
            expected_a0: 5 * 10,
            expected_exception: None,
        });

        // run, no branching, within page (differing max_steps count)
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: 10,
            max_steps: 5,
            expected_steps: 5,
            expected_pc_addr: 10 + 5 * InstrWidth::Compressed as u64,
            expected_a0: 5 * 5,
            expected_exception: None,
        });

        // run, no branching, exits at page boundary
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: page_size - 8 * InstrWidth::Compressed as u64,
            max_steps: 300,
            expected_steps: 8,
            expected_pc_addr: page_size,
            expected_a0: 5 * 8,
            expected_exception: None,
        });
    });

    backend_test!(page_dispatch_respects_max_steps_uncompressed, F, {
        let mut page_entry: Arc<[Interpreted<_, _>; INSTRUCTION_ENTRIES]> = boxed_from_fn({
            let mut idx = 0;
            move || {
                // we put uncompressed instructions on 4-byte aligned addresses
                let instr = if idx % 2 == 0 {
                    Instruction::new_addi(nz::a0, nz::a0, 5, InstrWidth::Uncompressed)
                } else {
                    Instruction::new_nop(InstrWidth::Compressed)
                };

                idx += 1;

                Interpreted::from(instr)
            }
        })
        .into();

        let dispatch = &std::cell::RefCell::new(CodePage {
            page: &mut page_entry,
        });

        let state = MachineCoreState::<M4K, F>::new();
        let state = &std::cell::RefCell::new(state);

        let page_size = memory::PAGE_SIZE.get();

        // run, no branching, within page
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: 0,
            max_steps: 10,
            expected_steps: 10,
            expected_pc_addr: 10 * InstrWidth::Uncompressed as u64,
            expected_a0: 5 * 10,
            expected_exception: None,
        });

        // run, no branching, within page (differing max_steps count)
        run_test(DispatchTest {
            state,
            dispatch,
            // start on 2-byte aligned instruction, first step compressed no-op
            pc_addr: 10,
            max_steps: 5,
            expected_steps: 5,
            expected_pc_addr: 10
                + InstrWidth::Compressed as u64
                + 4 * InstrWidth::Uncompressed as u64,
            expected_a0: 5 * 4,
            expected_exception: None,
        });

        // run, no branching, exits at page boundary
        run_test(DispatchTest {
            state,
            dispatch,
            // start on 2-byte aligned instruction, first step compressed no-op
            pc_addr: page_size
                - 8 * InstrWidth::Uncompressed as u64
                - InstrWidth::Compressed as u64,
            max_steps: 300,
            expected_steps: 9,
            expected_pc_addr: page_size,
            expected_a0: 5 * 8,
            expected_exception: None,
        });
    });

    backend_test!(page_dispatch_exits_on_non_next_pc_update, F, {
        let mut page_entry = Vec::with_capacity(INSTRUCTION_ENTRIES);

        let pc_j_absolute_start = 0;
        let pc_j_absolute = 10 * InstrWidth::Compressed as u64;
        for _ in 0..10 {
            page_entry.push(Interpreted::from(Instruction::new_addi(
                nz::a0,
                nz::a0,
                5,
                InstrWidth::Compressed,
            )));
        }
        page_entry.push(Interpreted::from(Instruction::new_j_absolute(
            0,
            InstrWidth::Uncompressed,
        )));

        let pc_jump_pc_start = pc_j_absolute + InstrWidth::Compressed as u64;
        let pc_jump_pc = pc_jump_pc_start + 10 * InstrWidth::Compressed as u64;
        for _ in 0..10 {
            page_entry.push(Interpreted::from(Instruction::new_addi(
                nz::a0,
                nz::a0,
                4,
                InstrWidth::Compressed,
            )));
        }
        page_entry.push(Interpreted::from(Instruction::new_jump_pc(
            0,
            InstrWidth::Uncompressed,
        )));

        let pc_ecall_start = pc_jump_pc + InstrWidth::Compressed as u64;
        let pc_ecall = pc_ecall_start + 10 * InstrWidth::Compressed as u64;
        for _ in 0..10 {
            page_entry.push(Interpreted::from(Instruction::new_addi(
                nz::a0,
                nz::a0,
                3,
                InstrWidth::Compressed,
            )));
        }
        page_entry.push(Interpreted::from(Instruction::new_ecall()));

        while page_entry.len() < page_entry.capacity() {
            page_entry.push(Interpreted::from(Instruction::new_nop(
                InstrWidth::Compressed,
            )));
        }

        let page_entry: Box<[_; INSTRUCTION_ENTRIES]> = page_entry
            .try_into()
            .expect("page_entry has INSTRUCTION_ENTRIES entries");
        let page_entry = Arc::from(page_entry);

        let dispatch = &std::cell::RefCell::new(CodePage { page: &page_entry });

        let state = MachineCoreState::<M4K, F>::new();
        let state = &std::cell::RefCell::new(state);

        // run, exits on PcUpdate::Set
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: pc_j_absolute_start,
            max_steps: 20,
            // jump back to start
            expected_steps: 11,
            expected_pc_addr: pc_j_absolute_start,
            expected_a0: 5 * 10,
            expected_exception: None,
        });

        // run, exits on PcUpdate::Relative
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: pc_jump_pc_start,
            max_steps: 20,
            // jump to current instruction
            expected_steps: 11,
            expected_pc_addr: pc_jump_pc,
            expected_a0: 4 * 10,
            expected_exception: None,
        });

        // run, exits on Exception
        run_test(DispatchTest {
            state,
            dispatch,
            pc_addr: pc_ecall_start,
            max_steps: 20,
            // throwing an exception is not a complete step
            expected_steps: 10,
            expected_pc_addr: pc_ecall,
            expected_a0: 3 * 10,
            expected_exception: Some(Exception::EnvCall),
        });
    });
}
