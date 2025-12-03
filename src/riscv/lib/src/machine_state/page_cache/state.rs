// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of the [PageCache].
//!
//! Crucially, the aim of the page cache is to never alter PVM semantics.
//!
//! We therefore do not raise errors that occur within the page cache externally - as
//! handling these elsewhere inadvertently _could_ cause a divergence in execution.
//!
//! [PageCache]: super::PageCache

use std::sync::Arc;

use super::INSTRUCTION_ENTRIES;
use super::PageCache;
use super::code_page_entry::CodePageEntry;
use super::router::Router;
use crate::default::ConstDefault;
use crate::exceptions::Exception;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::InstructionData;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::PAGE_MASK;
use crate::machine_state::memory::PAGE_SIZE;
use crate::machine_state::memory::address_to_page_index;
use crate::machine_state::memory::listener::MemoryGovernanceListener;
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;

/// Offset from the start of the page, to the last halfword contained within.
const LAST_HALFWORD_PAGE_OFFSET: u64 = PAGE_SIZE
    .get()
    .checked_sub(std::mem::size_of::<u16>() as u64)
    .expect("page-size must contain at least one halfword");

/// A `PageEntry` is an entry in the [page cache].
///
/// It corresponds to a complete page of executable memory, represented as entrypoints at
/// every halfword. These entrypoints are dispatchable: running them is the equivalent to
/// performing a _fetch/run/parse_ cycle, starting at the equivalent address.
///
/// We only have entries for every _halfword_, as the instruction pc is always halfword
/// aligned.
///
/// [page cache]: super::PageCache
#[derive(Debug)]
pub struct PageEntry<CPE, C> {
    pub entries: [CPE; INSTRUCTION_ENTRIES],
    pub(super) compiler: C,
}

impl<CPE: From<Instruction> + std::fmt::Debug, C> PageEntry<CPE, C> {
    /// Construct a new [`PageEntry`].
    ///
    /// `fetch_compiler` should return the compiler that will be attached to this `PageEntry`. This
    /// argument is `FnOnce` because it may be necessary to create a new compiler and add it in the
    /// router before returning it.
    ///
    /// `fetch_instr` should return the instruction that will be placed into the entrypoint
    /// at `offset` in the `PageEntry`.
    ///
    /// This must be able to return an instruction for offsets from zero to [`INSTRUCTION_ENTRIES`].
    /// *NB* the offsets refer to the _halfwords_ of the page, rather than each byte themselves.
    pub(super) fn new<E>(
        fetch_compiler: impl FnOnce() -> C,
        fetch_instr: impl Fn(usize) -> Result<Instruction, E>,
    ) -> Result<Arc<Self>, E> {
        let mut page = Arc::<Self>::new_uninit();
        let page_ptr = Arc::get_mut(&mut page)
            .expect("We just created this arc, there's only one")
            .as_mut_ptr();

        // SAFETY: it's safe to construct a pointer to an uninitialised field of an
        // uninitialised struct.
        let entries_ptr: *mut [CPE; INSTRUCTION_ENTRIES] =
            unsafe { std::ptr::addr_of_mut!((*page_ptr).entries) };

        let start_ptr: *mut CPE = entries_ptr as *mut CPE;

        for offset in 0..INSTRUCTION_ENTRIES {
            let instruction = fetch_instr(offset)?;
            let entrypoint = CPE::from(instruction);

            // SAFETY: we are writing exactly `INSTRUCTION_ENTRIES` into an array of length
            // `INSTRUCTION_ENTRIES`. Additionally, we use `write` rather than `=` to avoid
            // calling `drop` on the old value - which would be UB as it is undefined.
            unsafe {
                start_ptr.add(offset).write(entrypoint);
            }
        }

        // SAFETY: it is safe to construct a pointer to an uninitialised field of an uninitialised
        // struct and then write to it
        unsafe {
            std::ptr::addr_of_mut!((*page_ptr).compiler).write(fetch_compiler());
        }

        // Safety: all `INSTRUCTION_ENTRIES` entrypoints are now initialised. The compiler field is
        // also initialised. It is therefore safe to no longer treat these as unitialised.
        let page = unsafe { page.assume_init() };

        Ok(page)
    }

    /// Construct a new page entry as if it were populated from a memory of entirely zeroes.
    #[cfg(test)]
    pub(crate) fn zeroed(compiler: C) -> Arc<Self> {
        use crate::machine_state::instruction::Instruction;
        use crate::parser::parse_compressed_instruction;

        const ZEROED_HALFWORD: u16 = 0;
        assert!(is_compressed(ZEROED_HALFWORD));

        let instruction = parse_compressed_instruction(ZEROED_HALFWORD);
        let instruction = Instruction::from(&instruction);

        let Ok(page) = Self::new::<std::convert::Infallible>(|| compiler, |_| Ok(instruction));

        page
    }

    /// Push a sequence of instructions to a page entry, starting from the page offset given by the
    /// address.
    ///
    /// The page in question must not have been cloned.
    #[cfg(test)]
    pub(crate) fn push_instructions(
        page: &mut Arc<Self>,
        address: Address,
        instructions: impl Iterator<Item = Instruction>,
    ) {
        let page =
            Arc::get_mut(page).expect("push_instructions can only be called on an uncloned page");

        // we only store entries for halfword-aligned addresses, since pc is always halfword
        // aligned
        let mut offset = super::address_to_halfword_index(address);

        for instr in instructions {
            if offset > INSTRUCTION_ENTRIES {
                panic!(
                    "Instructions cannot all fit within a single page, starting at the given address"
                );
            }

            page.entries[offset] = CPE::from(instr);

            // we update the offset by half the width, as the offset is halfword aligned
            offset += (instr.width() as usize) >> 1;
        }
    }
}

#[cfg(test)]
impl<D, MC> PageCacheImpl<super::Jitted<D, MC>, MC, octez_riscv_data::mode::Normal>
where
    MC: MemoryConfig,
    D: Clone + super::router::RouterEq + super::dispatch::DispatchCompiler<MC>,
{
    /// TEST ONLY
    ///
    /// Get the number of times that an entrypoint has been called
    pub(crate) fn get_entrypoint_called_times(&self, address: Address) -> Option<usize> {
        let page_index = address_to_page_index(address);
        let page = self.pages[page_index].as_ref()?;

        // we only store entries for halfword-aligned addresses, since pc is always halfword
        // aligned
        let offset = super::address_to_halfword_index(address);

        Some(page.entries[offset].dispatch.called_times())
    }
}

/// Type alias to simplify the [`PageCacheImpl`] struct.
type BoxedPages<CPE, C> = Box<[Option<Arc<PageEntry<CPE, C>>>]>;

/// Default implementor of [`PageCache`].
///
/// [`PageCache`]: super::PageCache
// TODO RV-775: this will change to be the default implementor of `PageCache` for
//              `Normal` mode, only.
pub struct PageCacheImpl<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase> {
    pages: BoxedPages<CPE, CPE::Compiler>,
    compiler_context: CPE::CompilerContext,
    router: Router<CPE::Compiler>,
}

#[cfg(test)]
impl<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase> PageCacheImpl<CPE, MC, M> {
    /// TEST ONLY
    ///
    /// Overwrite a page entry within the page cache. The entry overwritten is the one containing
    /// the given address.
    pub(crate) fn overwrite_page(
        &mut self,
        address: Address,
        page_entry: Arc<PageEntry<CPE, CPE::Compiler>>,
    ) {
        let page_index = address_to_page_index(address);

        self.pages[page_index] = Some(page_entry);
    }

    /// In the stepper tests we wish to include a summary of the page cache in the goldenfile, so
    /// that changes to the compilation and caching behaviour are tracked.
    pub(crate) fn write_summary(&self, w: &mut impl std::io::Write) {
        for (page_ix, page) in self.pages.iter().enumerate() {
            let Some(page) = page else {
                continue;
            };
            let call_summary = page
                .entries
                .iter()
                .enumerate()
                .map(|(i, entrypoint)| (i, entrypoint.called_times()))
                .filter(|(_i, calls)| *calls != 0)
                .collect::<Vec<_>>();
            writeln!(w, "page {page_ix}:\n entrypoint calls: {call_summary:?}").unwrap();
        }
    }
}

impl<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase> PageCache<MC, M>
    for PageCacheImpl<CPE, MC, M>
{
    /// Construct a new page cache, which will be entirely unpopulated.
    fn new() -> Self {
        assert!(
            MC::TOTAL_BYTES.get() % memory::PAGE_SIZE.get() as usize == 0,
            "PageCache relies on memory being an exact number of pages in length"
        );
        let pages = MC::TOTAL_BYTES.get() >> memory::OFFSET_BITS.get();

        Self {
            pages: vec![None; pages].into_boxed_slice(),
            compiler_context: CPE::CompilerContext::default(),
            router: Router::<CPE::Compiler>::default(),
        }
    }

    /// Fetch a dispatch call, if the address corresponds to a populated page in the PageCache.
    fn get_code_page(&mut self, address: Address) -> Option<impl super::CodePage<'_, MC, M>>
    where
        M: ManagerRead,
    {
        let page_index = address_to_page_index(address);

        self.pages
            .get_mut(page_index)
            .and_then(|entry| entry.as_mut())
            .map(|page| CodePageImpl { page })
    }

    /// Populates the entry in the page cache, that the given address points to.
    ///
    /// This will only populate the page iff the memory is R+X and *not writeable*.
    fn populate_page(&mut self, address: Address, core: &MachineCoreState<MC, M>)
    where
        M: ManagerRead + ManagerWrite,
    {
        let page_start = address & PAGE_MASK;
        let page_index = address_to_page_index(page_start);

        let Some(page_entry) = self.pages.get_mut(page_index) else {
            #[cfg(feature = "log")]
            crate::log::warning!(
                "Failed to populated page at {page_start:x}: address {address:x} out of bounds of page cache"
            );
            return;
        };

        if page_entry.is_some() {
            // no need to populate the page entry a second time
            return;
        }

        let fetch_compiler = || {
            let idx = page_index as u64;
            if let Some(compiler) = self.router.get(&idx) {
                compiler.clone()
            } else {
                #[cfg(feature = "log")]
                crate::log::warning!(
                    "Populating page requires ad-hoc single page range to be added in router: {page_index}"
                );
                self.router
                    .add_range(idx..=idx, || CPE::new_compiler(&self.compiler_context));
                self.router
                    .get(&idx)
                    .expect("Should find a page we just added to the router")
                    .clone()
            }
        };

        *page_entry = PageEntry::new::<PopulationError>(fetch_compiler, |offset| {
            let offset_bytes = (offset << 1) as u64;

            if offset_bytes < LAST_HALFWORD_PAGE_OFFSET {
                let InstructionData {
                    data: instr,
                    writable: false,
                } = core.fetch_instr(page_start + offset_bytes)?
                else {
                    return Err(PopulationError::Writable);
                };

                return Ok(instr);
            }

            let page_last_halfword = page_start + LAST_HALFWORD_PAGE_OFFSET;

            let InstructionData {
                data: halfword,
                writable: false,
            } = core.fetch_instr_halfword(page_last_halfword)?
            else {
                unreachable!(
                    "If the page was not R+X only, we would have failed to fetch all previous halfwords"
                );
            };

            if is_compressed(halfword) {
                let instr = parse_compressed_instruction(halfword);
                return Ok(Instruction::from(&instr));
            }

            // uncompressed instruction crosses page boundary: fallback to
            // FetchRunParse here.
            Ok(Instruction::DEFAULT)
        }).map_err(|_err| {
            #[cfg(feature = "log")]
            crate::log::warning!("Failed to populated page at {page_start:x}: {_err:?}");
        }).ok()
    }
}

impl<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase> PageCacheImpl<CPE, MC, M> {
    /// Invalidate the ranges of pages in the router which overlap the provided range of memory.
    fn invalidate_pages(&mut self, pages: std::ops::RangeInclusive<u64>) {
        for range in self.router.drain_overlapping(pages) {
            for page_idx in range {
                self.pages[page_idx as usize] = None;
            }
        }
    }

    /// Instantiate a new range of pages in the router, either with a new compiler or by extending
    /// existing ones.
    fn instantiate_pages(&mut self, pages: std::ops::RangeInclusive<u64>) {
        self.router
            .add_range(pages, || CPE::new_compiler(&self.compiler_context));
    }
}

impl<CPE, MC, M> MemoryGovernanceListener for PageCacheImpl<CPE, MC, M>
where
    CPE: CodePageEntry<MC, M>,
    MC: MemoryConfig,
    M: ManagerBase,
{
    /// The PageCache must ensure that it is always synchronised with main memory.
    ///
    /// Specifically, we must only have entries for pages that are executable, possibly readable,
    /// and *not* writeable. Caching writeable pages would mean that entries could easily become
    /// desynchronised from main memory, when said pages are written to - which we must avoid.
    fn handle_permissions_update(
        &mut self,
        pages: std::ops::RangeInclusive<u64>,
        permissions: memory::Permissions,
    ) {
        let needs_invalidation = permissions.can_write() || !permissions.can_exec();
        let needs_instantiation = !permissions.can_write() && permissions.can_exec();

        if needs_invalidation {
            self.invalidate_pages(pages);
        } else if needs_instantiation {
            self.instantiate_pages(pages);
        }
    }
}

/// Error that occurs populating a page
#[derive(thiserror::Error, Debug)]
enum PopulationError {
    #[error("Failed to access given page in memory")]
    Access(#[from] memory::BadMemoryAccess),
    #[error("Failed to fetch instruction")]
    FetchInstr(#[from] crate::machine_state::Exception),
    #[error("Cannot cache writable pages")]
    Writable,
}

/// A page containing code that may then be run against the [`MachineCoreState`].
///
/// This serves as the default implementation of [`CodePage`]
/// to be used with [`PageCacheImpl`].
///
/// [`CodePage`]: super::CodePage
#[derive(Debug)]
struct CodePageImpl<'a, MC: MemoryConfig, M: ManagerBase, CPE: CodePageEntry<MC, M>> {
    page: &'a Arc<PageEntry<CPE, CPE::Compiler>>,
}

impl<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase> super::CodePage<'_, MC, M>
    for CodePageImpl<'_, MC, M, CPE>
{
    #[inline]
    fn run(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        MC: MemoryConfig,
        M: ManagerRead + ManagerWrite,
    {
        CPE::run_entrypoint(self.page, core, instr_pc, max_steps)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::num::NonZeroUsize;

    use octez_riscv_data::mode::Normal;
    use proptest::prelude::*;

    use super::PageCacheImpl;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::exceptions::Exception;
    use crate::machine_state::CodePageEntry;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::memory::Permissions;
    use crate::machine_state::memory::address_to_page_index;
    use crate::machine_state::memory::listener::NoopMemoryGovernanceListener;
    use crate::machine_state::page_cache::CodePage;
    use crate::machine_state::page_cache::INSTRUCTION_ENTRIES;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::page_cache::PageCache;
    use crate::machine_state::page_cache::address_to_halfword_index;
    use crate::machine_state::page_cache::interpreted::Interpreted;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::machine_state::registers::nz;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;
    use crate::state_backend::ManagerBase;
    use crate::state_backend::test_helpers::TestBackendFactory;

    fn count_active_pages<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase>(
        cache: &PageCacheImpl<CPE, MC, M>,
    ) -> usize {
        cache.pages.iter().fold(
            0,
            |acc, page_entry| if page_entry.is_some() { acc + 1 } else { acc },
        )
    }

    #[test]
    fn test_page_invalidation_resets_pages() {
        let mut cache = PageCacheImpl::<Interpreted<_, _>, M1M, Normal>::new();

        // In the router, we instantiate pages 4..=7 as a single range, and the other pages up to
        // 14 as single-page ranges.
        cache.instantiate_pages(4..=7);
        for i in 0..14 {
            cache.instantiate_pages(i..=i);
        }

        for page in cache.pages.iter_mut().take(16) {
            *page = PageEntry::<Interpreted<_, _>, InterpretedCompiler>::new::<
                std::convert::Infallible,
            >(|| InterpretedCompiler, |_| Ok(Instruction::DEFAULT))
            .ok()
        }
        assert_eq!(count_active_pages(&cache), 16);

        // invalidate some pages, the overlap with the 4..=7 range means those extra pages will be
        // dropped too
        cache.invalidate_pages(1..=4);
        assert_eq!(count_active_pages(&cache), 9);
        assert!(cache.get_code_page(0).is_some());

        for i in 1..=7 {
            assert!(cache.get_code_page(i * PAGE_SIZE.get()).is_none());
        }

        assert!(cache.get_code_page(8 * PAGE_SIZE.get()).is_some());

        // invalidate another range
        cache.invalidate_pages(10..=11);
        assert_eq!(count_active_pages(&cache), 7);

        // invalidating pages we already have does nothing
        cache.invalidate_pages(1..=3);
        assert_eq!(count_active_pages(&cache), 7);

        // invalidate all addresses clears all pages that are in the router
        cache.invalidate_pages(0..=u64::MAX);
        assert_eq!(count_active_pages(&cache), 2);
    }

    backend_test!(test_populate_cache, F, {
        let mut state = MachineCoreState::<M4K, F>::new();
        let mut cache = PageCacheImpl::<Interpreted<_, _>, M4K, F>::new();

        // populating a non R+X page should fail
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a read-only page should fail
        let page_size: NonZeroUsize = PAGE_SIZE.try_into().unwrap();
        state
            .main_memory
            .protect_pages(
                0,
                page_size,
                Permissions {
                    read: true,
                    exec: false,
                    write: false,
                },
                &mut cache,
            )
            .unwrap();
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a R+W should fail
        state
            .main_memory
            .protect_pages(0, page_size, Permissions::READ_WRITE, &mut cache)
            .unwrap();
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a R+W+X should fail
        state
            .main_memory
            .protect_pages(0, page_size, Permissions::READ_WRITE_EXEC, &mut cache)
            .unwrap();
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a R+X page should succeed
        state
            .main_memory
            .protect_pages(
                0,
                page_size,
                Permissions {
                    read: true,
                    exec: true,
                    write: false,
                },
                &mut cache,
            )
            .unwrap();
        cache.populate_page(90, &state);
        assert_eq!(count_active_pages(&cache), 1);
    });

    backend_test!(populate_from_memory, F, {
        let state = MachineCoreState::<M1M, F>::new();
        let state = &std::cell::RefCell::new(state);

        let cache = &std::cell::RefCell::new(PageCacheImpl::<Interpreted<_, _>, M1M, F>::new());

        proptest!(|(pc_addr in 0..M1M::TOTAL_BYTES.get() as u64,
                    page: Box<[u8; PAGE_SIZE.get() as usize]>)| {
            // Arrange
            let mut state = state.borrow_mut();
            let mut cache = cache.borrow_mut();

            state.reset(&mut *cache);

            // clear lowest bit - address is always halfword aligned
            let pc_addr = pc_addr & !1;

            let page_start = pc_addr - (pc_addr % PAGE_SIZE.get());

            // sanity check - in case PAGE_SIZE/OFFSET_BITS changes in a very weird way
            assert!((page_start..(page_start + PAGE_SIZE.get())).contains(&pc_addr));

            for (offset, byte) in page.iter().enumerate() {
                state.main_memory.write_instruction_unchecked(page_start + offset as u64, *byte).unwrap();
            }

            // Act
            cache.populate_page(pc_addr, &state);

            // Assert
            assert_eq!(1, count_active_pages(&cache));

            let instr_from_memory = state.fetch_instr(pc_addr);
            let pc_offset = pc_addr % PAGE_SIZE.get();

            let expected_instr = if pc_offset <= (PAGE_SIZE.get() - InstrWidth::Uncompressed as u64) {
                instr_from_memory.expect("instruction guaranteed to not overlap page boundary").data
            } else if let Ok(instr) = instr_from_memory {
                // we must have had a compressed instruction in the last halfword
                assert_eq!(InstrWidth::Compressed, instr.data.width());
                assert_eq!(pc_offset, PAGE_SIZE.get() - InstrWidth::Compressed as u64);
                instr.data
            } else {
                // overlapping page boundary, fallback to force-fetch-run
                assert_eq!(pc_offset, PAGE_SIZE.get() - InstrWidth::Compressed as u64);
                Instruction::DEFAULT
            };

            let entry_idx = address_to_halfword_index(pc_addr);

            let instr_from_code_page = &cache.pages[address_to_page_index(pc_addr)]
                .as_ref()
                .expect("Page is populated")
                .entries[entry_idx];
            assert_eq!(&expected_instr, instr_from_code_page.as_ref());

            let mut code_page = cache.get_code_page(pc_addr).expect("code page populated");

            // double check last halfword
            let pc_last_halfword = page_start + (PAGE_SIZE.get() - InstrWidth::Compressed as u64);
            let last_halfword = state.fetch_instr_halfword(pc_last_halfword).unwrap();
            if !crate::parser::is_compressed(last_halfword.data) {
                let step_res = code_page.run(&mut state, pc_last_halfword, 1);
                assert_eq!(step_res.error, Some(crate::exceptions::Exception::ForceFetchRun));
                assert_eq!(step_res.steps, 0, "raising an exception does not complete a step");
            }
        });
    });

    struct DispatchTest<'a, F: TestBackendFactory> {
        state: &'a RefCell<MachineCoreState<M4K, F>>,
        dispatch: &'a RefCell<super::CodePageImpl<'a, M4K, F, Interpreted<M4K, F>>>,
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
        let res = test
            .dispatch
            .borrow_mut()
            .run(&mut state, test.pc_addr, test.max_steps);

        assert_eq!(res.steps, test.expected_steps);
        assert_eq!(res.error, test.expected_exception);

        assert_eq!(state.hart.pc.read(), test.expected_pc_addr);
        assert_eq!(state.hart.xregisters.read_nz(nz::a0), test.expected_a0);
    }

    backend_test!(page_dispatch_respects_max_steps_compressed, F, {
        let Ok(mut page_entry) =
            PageEntry::<Interpreted<_, _>, InterpretedCompiler>::new::<std::convert::Infallible>(
                || InterpretedCompiler,
                |_| {
                    Ok(Instruction::new_addi(
                        nz::a0,
                        nz::a0,
                        5,
                        InstrWidth::Compressed,
                    ))
                },
            );

        let dispatch = &RefCell::new(super::CodePageImpl {
            page: &mut page_entry,
        });

        let state = MachineCoreState::<M4K, F>::new();
        let state = &RefCell::new(state);

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
        let Ok(mut page_entry) =
            PageEntry::<Interpreted<_, _>, InterpretedCompiler>::new::<std::convert::Infallible>(
                || InterpretedCompiler,
                |idx| {
                    // we put uncompressed instructions on 4-byte aligned addresses
                    let instr = if idx % 2 == 0 {
                        Instruction::new_addi(nz::a0, nz::a0, 5, InstrWidth::Uncompressed)
                    } else {
                        Instruction::new_nop(InstrWidth::Compressed)
                    };

                    Ok(instr)
                },
            );

        let dispatch = &RefCell::new(super::CodePageImpl {
            page: &mut page_entry,
        });

        let state = MachineCoreState::<M4K, F>::new();
        let state = &RefCell::new(state);

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
        let mut instructions = Vec::with_capacity(INSTRUCTION_ENTRIES);

        let pc_j_absolute_start = 0;
        let pc_j_absolute = 10 * InstrWidth::Compressed as u64;
        for _ in 0..10 {
            instructions.push(Instruction::new_addi(
                nz::a0,
                nz::a0,
                5,
                InstrWidth::Compressed,
            ));
        }
        instructions.push(Instruction::new_j_absolute(0, InstrWidth::Uncompressed));

        let pc_jump_pc_start = pc_j_absolute + InstrWidth::Compressed as u64;
        let pc_jump_pc = pc_jump_pc_start + 10 * InstrWidth::Compressed as u64;
        for _ in 0..10 {
            instructions.push(Instruction::new_addi(
                nz::a0,
                nz::a0,
                4,
                InstrWidth::Compressed,
            ));
        }
        instructions.push(Instruction::new_jump_pc(0, InstrWidth::Uncompressed));

        let pc_ecall_start = pc_jump_pc + InstrWidth::Compressed as u64;
        let pc_ecall = pc_ecall_start + 10 * InstrWidth::Compressed as u64;
        for _ in 0..10 {
            instructions.push(Instruction::new_addi(
                nz::a0,
                nz::a0,
                3,
                InstrWidth::Compressed,
            ));
        }
        instructions.push(Instruction::new_ecall());

        while instructions.len() < instructions.capacity() {
            instructions.push(Instruction::new_nop(InstrWidth::Compressed));
        }

        let Ok(mut page_entry) = PageEntry::<Interpreted<_, _>, InterpretedCompiler>::new::<
            std::convert::Infallible,
        >(
            || InterpretedCompiler, |offset| Ok(instructions[offset])
        );

        let dispatch = &RefCell::new(super::CodePageImpl {
            page: &mut page_entry,
        });

        let state = MachineCoreState::<M4K, F>::new();
        let state = &RefCell::new(state);

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
