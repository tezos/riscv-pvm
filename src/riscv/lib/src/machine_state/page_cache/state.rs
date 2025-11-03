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

use std::marker::PhantomData;
use std::sync::Arc;

use super::INSTRUCTION_ENTRIES;
use super::PageCache;
use super::code_page_entry::CodePageEntry;
use crate::array_utils::boxed_from_fn;
use crate::default::ConstDefault;
use crate::machine_state::MachineCoreState;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::InstructionData;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::OFFSET_BITS;
use crate::machine_state::memory::PAGE_MASK;
use crate::machine_state::memory::PAGE_SIZE;
use crate::machine_state::memory::address_to_page_index;
use crate::machine_state::memory::listener::MemoryGovernanceListener;
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;

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
pub(crate) struct PageEntry<CPE> {
    // TODO: RV-790: consider raising this pointer (Arc) out of `PageEntry` to exploit the
    // `Option<Arc<_>>` optimisation.
    entries: Arc<[CPE; INSTRUCTION_ENTRIES]>,
}

#[cfg(test)]
impl<CPE: From<Instruction>> PageEntry<CPE> {
    /// TEST ONLY
    ///
    /// Construct a new page entry as if it were populated from a memory of entirely zeroes.
    pub(crate) fn zeroed() -> Self {
        use crate::machine_state::instruction::Instruction;
        use crate::parser::parse_compressed_instruction;

        const ZEROED_HALFWORD: u16 = 0;
        assert!(is_compressed(ZEROED_HALFWORD));

        let instruction = parse_compressed_instruction(ZEROED_HALFWORD);
        let instruction = Instruction::from(&instruction);

        Self {
            entries: Arc::from(boxed_from_fn(|| CPE::from(instruction))),
        }
    }

    /// TEST ONLY
    ///
    /// Push a sequence of instructions to a page entry, starting from the page offset given by the
    /// address.
    ///
    /// The page in question must not have been cloned.
    pub(crate) fn push_instructions(
        &mut self,
        address: Address,
        instructions: impl Iterator<Item = Instruction>,
    ) {
        use crate::machine_state::memory::address_to_page_offset;

        let entries = Arc::get_mut(&mut self.entries)
            .expect("push_instructions can only be called on an uncloned page");

        // we only store entries for halfword-aligned addresses, since pc is always halfword
        // aligned
        let mut offset = address_to_page_offset(address) >> 1;

        for instr in instructions {
            if offset > INSTRUCTION_ENTRIES {
                panic!(
                    "Instructions cannot all fit within a single page, starting at the given address"
                );
            }

            entries[offset] = CPE::from(instr);

            // we update the offset by half the width, as the offset is halfword aligned
            offset += (instr.width() as usize) >> 1;
        }
    }
}

#[cfg(test)]
impl<const PAGES: usize, D, MC>
    PageCacheImpl<PAGES, super::Jitted<D, MC>, MC, crate::state_backend::owned_backend::Owned>
where
    MC: MemoryConfig,
    D: super::dispatch::DispatchCompiler<MC>,
{
    /// TEST ONLY
    ///
    /// Get the number of times that an entrypoint has been called
    pub(crate) fn get_entrypoint_called_times(&self, address: Address) -> Option<usize> {
        let page_index = address_to_page_index(address);
        let page = self.pages[page_index].as_ref()?;

        // we only store entries for halfword-aligned addresses, since pc is always halfword
        // aligned
        let offset = crate::machine_state::memory::address_to_page_offset(address) >> 1;

        Some(page.entries[offset].dispatch.called_times())
    }
}

/// Default implementor of [`PageCache`].
///
/// This separation mainly to allow us to work around rust's restrictions w.r.t.
/// const-generics. We require the number of `PAGES` to match that passed to the
/// memory config. Rust will not allow us to expose this on the [`MemoryConfig`]
/// and consume it here, however. Therefore, we avoid specifying the const-generic
/// on the trait level, where it's easy to tie everything together - and make the
/// connection with the concrete types that implement these traits.
///
/// [`PageCache`]: super::PageCache
pub struct PageCacheImpl<const PAGES: usize, CPE, MC, M> {
    pages: Box<[Option<PageEntry<CPE>>; PAGES]>,
    _pd: PhantomData<(MC, M)>,
}

#[cfg(test)]
impl<const PAGES: usize, CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase>
    PageCacheImpl<PAGES, CPE, MC, M>
{
    /// TEST ONLY
    ///
    /// Overwrite a page entry within the page cache. The entry overwritten is the one containing
    /// the given address.
    pub(crate) fn overwrite_page(&mut self, address: Address, page_entry: PageEntry<CPE>) {
        let page_index = crate::machine_state::memory::address_to_page_index(address);

        self.pages[page_index] = Some(page_entry);
    }
}

impl<const PAGES: usize, CPE: CodePageEntry<MC, M>, MC: MemoryConfig, M: ManagerBase>
    PageCache<CPE, MC, M> for PageCacheImpl<PAGES, CPE, MC, M>
{
    /// Construct a new page cache, which will be entirely unpopulated.
    fn new() -> Self {
        Self {
            pages: boxed_from_fn(|| None),
            _pd: PhantomData,
        }
    }

    /// Fetch a dispatch call, if the address corresponds to a populated page in the PageCache.
    fn get_code_page(&mut self, address: Address) -> Option<super::CodePage<'_, CPE>>
    where
        M: ManagerRead,
    {
        let page_index = address_to_page_index(address);

        self.pages
            .get_mut(page_index)
            .and_then(|entry| entry.as_mut())
            .map(|page| super::CodePage {
                page: &mut page.entries,
            })
    }

    /// Populates the entry in the page cache, that the given address points to.
    ///
    /// This will only populate the page iff the memory is R+X and *not writeable*.
    fn populate_page(&mut self, address: Address, core: &MachineCoreState<MC, M>)
    where
        M: ManagerReadWrite,
    {
        // Unlike with `Box<[T; LEN]>` - we cannot initialise with a Vec and do an in-place
        // conversion. To avoid copying, we're forced to allocate with `Arc` directly.
        let mut entries = Arc::new_uninit_slice(INSTRUCTION_ENTRIES);
        let page_entries =
            Arc::get_mut(&mut entries).expect("We just created this arc, there's only one");

        let page_start = address & PAGE_MASK;

        let page_last_halfword = page_start + LAST_HALFWORD_PAGE_OFFSET;

        let page_range = page_start..page_last_halfword;

        // fetch and parse all instructions in the page, except the final halfword as
        // it could be the lower halfword of an uncompressed instruction, which would
        // require looking up the next page (which may not be R+X).
        //
        // After this loop, all but the final entrypoint are initialised.
        //
        // TODO: RV-772: remove redundant permission checks/halfword fetches from memory.
        let mut offset = 0;
        for address in page_range.step_by(std::mem::size_of::<u16>()) {
            let Ok(InstructionData {
                data: instr,
                writable: false,
            }) = core.fetch_instr(address)
            else {
                return;
            };

            page_entries[offset].write(CPE::from(instr));
            offset += 1;
        }

        // final halfword may need to use ForceFetchRun mechanism if it overlaps page boundary
        let Ok(InstructionData {
            data: halfword,
            writable: false,
        }) = core.fetch_instr_halfword(page_last_halfword)
        else {
            unreachable!(
                "If the page was not R+X only, we would have failed to fetch all previous halfwords"
            );
        };

        let final_entry = if is_compressed(halfword) {
            let instr = parse_compressed_instruction(halfword);
            Instruction::from(&instr)
        } else {
            // uncompressed instruction crosses page boundary: fallback to
            // FetchRunParse here.
            Instruction::DEFAULT
        };

        // Initialise the final entrypoint (which corresponds to the final halfword).
        page_entries[offset].write(CPE::from(final_entry));

        if let Some(page_entry) = self
            .pages
            .get_mut((page_start >> OFFSET_BITS.get()) as usize)
        {
            // Safety: all `INSTRUCTION_ENTRIES` entrypoints are now initialised. It is therefore
            // safe to no longer treat these as unitialised.
            let entries = unsafe { entries.assume_init() };

            *page_entry = Some(PageEntry {
                entries: entries
                    .try_into()
                    .expect("instructions has exactly the length expected for PageEntry::entries"),
            });
        }
    }

    /// Invalidate a range of pages corresponding to the provided range of memory.
    fn invalidate_pages(&mut self, pages: std::ops::RangeInclusive<u64>) {
        for page_idx in pages.take_while(|idx| *idx < PAGES as u64) {
            self.pages[page_idx as usize] = None;
        }
    }
}

impl<const PAGES: usize, CPE, MC, M> MemoryGovernanceListener for PageCacheImpl<PAGES, CPE, MC, M>
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use proptest::prelude::*;

    use super::PageCacheImpl;
    use crate::array_utils::boxed_array;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::memory::Permissions;
    use crate::machine_state::page_cache::INSTRUCTION_ENTRIES;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::page_cache::PageCache;
    use crate::machine_state::page_cache::interpreted::Interpreted;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;
    use crate::state_backend::owned_backend::Owned;

    fn count_active_pages<const PAGES: usize, CPE, MC, M>(
        cache: &PageCacheImpl<PAGES, CPE, MC, M>,
    ) -> usize {
        cache.pages.iter().fold(
            0,
            |acc, page_entry| if page_entry.is_some() { acc + 1 } else { acc },
        )
    }

    #[test]
    fn test_page_invalidation_resets_pages() {
        const PAGES: usize = M1M::TOTAL_BYTES.get() / PAGE_SIZE.get() as usize;

        let mut cache = PageCacheImpl::<PAGES, Interpreted<_, _>, M1M, Owned>::new();

        let make_page = || PageEntry {
            entries: Arc::from(
                boxed_array![Interpreted::<_, _>::from(Instruction::DEFAULT); INSTRUCTION_ENTRIES],
            ),
        };

        for page in cache.pages.iter_mut().take(16) {
            *page = Some(make_page());
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
        let mut cache = PageCacheImpl::<1, Interpreted<_, _>, M4K, F>::new();

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
        const PAGES: usize = M1M::TOTAL_BYTES.get() / PAGE_SIZE.get() as usize;

        let state = MachineCoreState::<M1M, F>::new();
        let state = &std::cell::RefCell::new(state);

        let cache =
            &std::cell::RefCell::new(PageCacheImpl::<PAGES, Interpreted<_, _>, M1M, F>::new());

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
            let instr_from_code_page = code_page.page[pc_offset as usize / 2];
            assert_eq!(&expected_instr, instr_from_code_page.as_ref());

            // double check last halfword
            let pc_last_halfword = page_start + (PAGE_SIZE.get() - InstrWidth::Compressed as u64);
            let last_halfword = state.fetch_instr_halfword(pc_last_halfword).unwrap();
            if !crate::parser::is_compressed(last_halfword.data) {
                // SAFETY: interpreted is always safe to call
                let step_res = unsafe { code_page.run(&mut state, &mut InterpretedCompiler, pc_last_halfword, 1) };
                assert_eq!(step_res.error, Some(crate::exceptions::Exception::ForceFetchRun));
                assert_eq!(step_res.steps, 0, "raising an exception does not complete a step");
            }
        });
    });
}
