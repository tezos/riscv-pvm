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

use super::INSTRUCTION_ENTRIES;
use crate::array_utils::boxed_from_fn;
use crate::default::ConstDefault;
use crate::machine_state::MachineCoreState;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::InstructionData;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::OFFSET_BITS;
use crate::machine_state::memory::PAGE_MASK;
use crate::machine_state::memory::PAGE_SIZE;
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

struct PageEntry {
    // TODO: RV-773: consider re-using something like the EnrichedCell mechanism for faster
    // interpreted dispatch here.
    entries: Box<[Instruction; INSTRUCTION_ENTRIES]>,
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
pub struct PageCacheImpl<const PAGES: usize> {
    pages: Box<[Option<PageEntry>; PAGES]>,
}

impl<const PAGES: usize, MC: MemoryConfig, M: ManagerBase> super::PageCache<MC, M>
    for PageCacheImpl<PAGES>
{
    /// Construct a new page cache, which will be entirely unpopulated.
    fn new() -> Self {
        Self {
            pages: boxed_from_fn(|| None),
        }
    }

    /// Fetch a dispatch call, if the address corresponds to a populated page in the PageCache.
    fn get_code_page(&mut self, address: Address) -> Option<super::CodePage<'_>>
    where
        M: ManagerRead,
    {
        // TODO: RV-779: add & use utility to retreiving page_idx & offset from address.
        let page_index = address >> OFFSET_BITS.get();

        self.pages
            .get_mut(page_index as usize)
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
        let mut instructions = Vec::with_capacity(INSTRUCTION_ENTRIES);

        let page_start = address & PAGE_MASK;

        let page_last_halfword = page_start + LAST_HALFWORD_PAGE_OFFSET;

        let page_range = page_start..page_last_halfword;

        // fetch and parse all instructions in the page, except the final halfword as
        // it could be the lower halfword of an uncompressed instruction, which would
        // require looking up the next page (which may not be R+X).
        //
        // TODO: RV-772: remove redundant permission checks/halfword fetches from memory.
        for address in page_range.step_by(std::mem::size_of::<u16>()) {
            let Ok(InstructionData {
                data: instr,
                writable: false,
            }) = core.fetch_instr(address)
            else {
                return;
            };

            instructions.push(instr);
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

        instructions.push(final_entry);

        if let Some(page_entry) = self
            .pages
            .get_mut((page_start >> OFFSET_BITS.get()) as usize)
        {
            *page_entry = Some(PageEntry {
                entries: instructions
                    .try_into()
                    .expect("instructions has exactly the length expected for PageEntry::entries"),
            });
        }
    }

    /// Invalidate a range of pages corresponding to the provided range of memory.
    fn invalidate_pages(&mut self, addresses: std::ops::Range<u64>) {
        let start_page = addresses.start >> OFFSET_BITS.get();
        let end_page = addresses.end.wrapping_sub(1) >> OFFSET_BITS.get();

        // shortcut in-case of ranges out of bounds of memory
        let end_page = end_page.min(self.pages.len().saturating_sub(1) as u64);

        for page_idx in start_page..=end_page {
            if let Some(entry) = self.pages.get_mut(page_idx as usize) {
                *entry = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::PageCacheImpl;
    use crate::array_utils::boxed_array;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::block_cache::block::InterpretedBlockBuilder;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::memory::Permissions;
    use crate::machine_state::page_cache::INSTRUCTION_ENTRIES;
    use crate::machine_state::page_cache::PageCache;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;
    use crate::state_backend::owned_backend::Owned;

    fn count_active_pages<const PAGES: usize>(cache: &PageCacheImpl<PAGES>) -> usize {
        cache.pages.iter().fold(
            0,
            |acc, page_entry| if page_entry.is_some() { acc + 1 } else { acc },
        )
    }

    #[test]
    fn test_page_invalidation_resets_pages() {
        const PAGES: usize = M1M::TOTAL_BYTES / PAGE_SIZE.get() as usize;

        let mut cache = <PageCacheImpl<PAGES> as PageCache<M1M, Owned>>::new();

        let make_page = || PageEntry {
            entries: boxed_array![Instruction::DEFAULT; INSTRUCTION_ENTRIES],
        };

        for page in cache.pages.iter_mut().take(16) {
            *page = Some(make_page());
        }
        assert_eq!(count_active_pages(&cache), 16);

        // invalidate a range - page aligned
        //
        // we expect the final page to not be invalidated - as upper bound of the half-open range
        // ends on the first byte of the page.
        PageCache::<M1M, Owned>::invalidate_pages(
            &mut cache,
            PAGE_SIZE.get()..(5 * PAGE_SIZE.get()),
        );
        assert_eq!(count_active_pages(&cache), 12);
        assert!(PageCache::<M1M, Owned>::get_code_page(&mut cache, 0).is_some());

        for i in 1..5 {
            assert!(
                PageCache::<M1M, Owned>::get_code_page(&mut cache, i * PAGE_SIZE.get()).is_none()
            );
        }

        assert!(PageCache::<M1M, Owned>::get_code_page(&mut cache, 5 * PAGE_SIZE.get()).is_some());

        // invalidate a range - non-page aligned
        //
        // in this instance, we expect both the starting and ending pages to be invalidated.
        PageCache::<M1M, Owned>::invalidate_pages(
            &mut cache,
            (10 * PAGE_SIZE.get() + 1)..(11 * PAGE_SIZE.get() + 1),
        );
        assert_eq!(count_active_pages(&cache), 10);

        // invalidate an already invalidated range does nothing
        PageCache::<M1M, Owned>::invalidate_pages(
            &mut cache,
            PAGE_SIZE.get()..(4 * PAGE_SIZE.get()),
        );
        assert_eq!(count_active_pages(&cache), 10);

        // invalidate all addresses clears all pages
        PageCache::<M1M, Owned>::invalidate_pages(&mut cache, 0..u64::MAX);
        assert_eq!(count_active_pages(&cache), 0);
    }

    backend_test!(test_populate_block_cache, F, {
        let mut state = MachineCoreState::<M4K, F>::new();
        let mut cache = <PageCacheImpl<1> as PageCache<M4K, Owned>>::new();

        // populating a non R+X page should fail
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a read-only page should fail
        state
            .main_memory
            .protect_pages(0, PAGE_SIZE.get() as usize, Permissions {
                read: true,
                exec: false,
                write: false,
            })
            .unwrap();
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a R+W should fail
        state
            .main_memory
            .protect_pages(0, PAGE_SIZE.get() as usize, Permissions::READ_WRITE)
            .unwrap();
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a R+W+X should fail
        state
            .main_memory
            .protect_pages(0, PAGE_SIZE.get() as usize, Permissions::READ_WRITE_EXEC)
            .unwrap();
        cache.populate_page(15, &state);
        assert_eq!(count_active_pages(&cache), 0);

        // populating a R+X page should succeed
        state
            .main_memory
            .protect_pages(0, PAGE_SIZE.get() as usize, Permissions {
                read: true,
                exec: true,
                write: false,
            })
            .unwrap();
        cache.populate_page(90, &state);
        assert_eq!(count_active_pages(&cache), 1);
    });

    backend_test!(populate_from_memory, F, {
        const PAGES: usize = M1M::TOTAL_BYTES / PAGE_SIZE.get() as usize;

        let state = MachineCoreState::<M1M, F>::new();
        let state = &std::cell::RefCell::new(state);

        proptest!(|(pc_addr in 0..M1M::TOTAL_BYTES as u64,
                    page: Box<[u8; PAGE_SIZE.get() as usize]>)| {
            // Arrange
            let mut cache = <PageCacheImpl<PAGES> as PageCache<M1M, Owned>>::new();
            let mut state = state.borrow_mut();
            state.reset();

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

            let code_page = PageCache::<M1M, Owned>::get_code_page(&mut cache, pc_addr).expect("code page populated");
            let instr_from_code_page = code_page.page[pc_offset as usize / 2];
            assert_eq!(expected_instr, instr_from_code_page);

            // double check last halfword
            let pc_last_halfword = page_start + (PAGE_SIZE.get() - InstrWidth::Compressed as u64);
            let last_halfword = state.fetch_instr_halfword(pc_last_halfword).unwrap();
            if !crate::parser::is_compressed(last_halfword.data) {
                let step_res = code_page.run(&mut state, InterpretedBlockBuilder, pc_last_halfword, 1);
                assert_eq!(step_res.error, Some(crate::traps::Exception::ForceFetchRun));
                assert_eq!(step_res.steps, 0, "raising an exception does not complete a step");
            }
        });
    });
}
