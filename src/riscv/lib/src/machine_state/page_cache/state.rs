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
use crate::default::ConstDefault;
use crate::machine_state::MachineCoreState;
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
    /// `fetch_instr` should return the instruction that will be placed into the entrypoint
    /// at `offset` in the `PageEntry`.
    ///
    /// This must be able to return an instruction for offsets from zero to [`INSTRUCTION_ENTRIES`].
    /// *NB* the offsets refer to the _halfwords_ of the page, rather than each byte themselves.
    pub(super) fn new<E>(
        compiler: C,
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
            std::ptr::addr_of_mut!((*page_ptr).compiler).write(compiler);
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

        let Ok(page) = Self::new::<std::convert::Infallible>(compiler, |_| Ok(instruction));

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
    D: Clone + super::dispatch::DispatchCompiler<MC>,
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
    compiler: CPE::Compiler,
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

impl<CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase> PageCache<CPE, MC, M>
    for PageCacheImpl<CPE, MC, M>
{
    /// Construct a new page cache, which will be entirely unpopulated.
    fn new() -> Self {
        assert!(
            MC::TOTAL_BYTES.get() % memory::PAGE_SIZE.get() as usize == 0,
            "PageCache relies on memory being an exact number of pages in length"
        );
        let pages = MC::TOTAL_BYTES.get() >> memory::OFFSET_BITS.get();

        let compiler_context = CPE::CompilerContext::default();

        Self {
            pages: vec![None; pages].into_boxed_slice(),
            compiler: CPE::new_compiler(&compiler_context),
        }
    }

    /// Fetch a dispatch call, if the address corresponds to a populated page in the PageCache.
    fn get_code_page(&mut self, address: Address) -> Option<super::CodePage<'_, MC, M, CPE>>
    where
        M: ManagerRead,
    {
        let page_index = address_to_page_index(address);

        self.pages
            .get_mut(page_index)
            .and_then(|entry| entry.as_mut())
            .map(|page| super::CodePage { page })
    }

    /// Populates the entry in the page cache, that the given address points to.
    ///
    /// This will only populate the page iff the memory is R+X and *not writeable*.
    fn populate_page(&mut self, address: Address, core: &MachineCoreState<MC, M>)
    where
        M: ManagerRead + ManagerWrite,
    {
        let page_start = address & PAGE_MASK;

        let Some(page_entry) = self.pages.get_mut(address_to_page_index(page_start)) else {
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

        *page_entry = PageEntry::new::<PopulationError>(self.compiler.clone(), |offset| {
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

    /// Invalidate a range of pages corresponding to the provided range of memory.
    fn invalidate_pages(&mut self, pages: std::ops::RangeInclusive<u64>) {
        let pages_len = self.pages.len();
        for page_idx in pages.take_while(|idx| *idx < pages_len as u64) {
            self.pages[page_idx as usize] = None;
        }
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

        if needs_invalidation {
            self.invalidate_pages(pages);
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use octez_riscv_data::mode::Normal;
    use proptest::prelude::*;

    use super::PageCacheImpl;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::machine_state::CodePageEntry;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::memory::Permissions;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::page_cache::PageCache;
    use crate::machine_state::page_cache::interpreted::Interpreted;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;
    use crate::state_backend::ManagerBase;

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

        for page in cache.pages.iter_mut().take(16) {
            *page = PageEntry::<Interpreted<_, _>, InterpretedCompiler>::new::<
                std::convert::Infallible,
            >(InterpretedCompiler, |_| Ok(Instruction::DEFAULT))
            .ok()
        }
        assert_eq!(count_active_pages(&cache), 16);

        // invalidate some pages
        cache.invalidate_pages(1..=4);
        assert_eq!(count_active_pages(&cache), 12);
        assert!(cache.get_code_page(0).is_some());

        for i in 1..=4 {
            assert!(cache.get_code_page(i * PAGE_SIZE.get()).is_none());
        }

        assert!(cache.get_code_page(5 * PAGE_SIZE.get()).is_some());

        // invalidate another range
        cache.invalidate_pages(10..=11);
        assert_eq!(count_active_pages(&cache), 10);

        // invalidating pages we already have does nothing
        cache.invalidate_pages(1..=3);
        assert_eq!(count_active_pages(&cache), 10);

        // invalidate all addresses clears all pages
        cache.invalidate_pages(0..=u64::MAX);
        assert_eq!(count_active_pages(&cache), 0);
    }

    backend_test!(test_populate_block_cache, F, {
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

            let mut code_page = cache.get_code_page(pc_addr).expect("code page populated");
            let instr_from_code_page = &code_page.page.entries[pc_offset as usize / 2];
            assert_eq!(&expected_instr, instr_from_code_page.as_ref());

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
}
