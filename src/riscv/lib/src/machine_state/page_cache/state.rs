// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;

use super::INSTRUCTION_ENTRIES;
use super::PAGE_MASK;
use crate::array_utils::boxed_from_fn;
use crate::default::ConstDefault;
use crate::machine_state::MachineCoreState;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::InstructionData;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::{self};
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::traps::Exception;

struct PageEntry<MC: MemoryConfig, M: ManagerBase> {
    entries: Box<[Instruction; INSTRUCTION_ENTRIES]>,
    _pd: PhantomData<(MC, M)>,
}

pub struct PageCache<const PAGES: usize, MC: MemoryConfig, M: ManagerBase> {
    pages: Box<[Option<PageEntry<MC, M>>; PAGES]>,
}

impl<const PAGES: usize, MC: MemoryConfig, M: ManagerBase> super::PageCache<MC, M> for PageCache<PAGES, MC, M> {
    /// Construct a new page cache, which will be entirely unpopulated.
    fn new() -> Self {
        Self {
            pages: boxed_from_fn(|| None),
        }
    }

    /// Fetch a dispatch call, if the address corresponds to a populated page in the PageCache.
    fn get_page_dispatch(
        &mut self,
        address: Address,
    ) -> Option<super::PageDispatch<'_, MC, M>>
    where
        M: ManagerRead,
    {
        let page_index = address & PAGE_MASK >> memory::OFFSET_BITS;

        self.pages
            .get_mut(page_index as usize)
            .map(|entry| entry.as_mut())
            .flatten()
            .map(|page| super::PageDispatch {
                page: &mut page.entries,
                _pd: PhantomData,
            })
    }

    /// Populates the entry in the page cache, that the given address points to.
    ///
    /// This will only populate the page iff the memory is R+X and *not writeable*.
    fn populate_page(
        &mut self,
        address: Address,
        core: &MachineCoreState<MC, M>,
    ) -> Result<(), Exception>
    where
        M: ManagerReadWrite,
    {
        let mut instructions = Vec::with_capacity(INSTRUCTION_ENTRIES);

        let page_start = address & PAGE_MASK;
        let page_last_halfword =
            page_start + memory::PAGE_SIZE.get() - std::mem::size_of::<u16>() as u64;

        // Does not overflow
        let page_range = page_start..page_last_halfword;

        for address in page_range.step_by(std::mem::size_of::<u16>()) {
            let InstructionData {
                data: instr,
                writable: false,
            } = core.fetch_instr(address)?
            else {
                return Ok(());
            };

            instructions.push(instr);
        }

        // handle last two bytes
        let InstructionData {
            data: halfword,
            writable: true,
        } = core.fetch_instr_halfword(page_last_halfword)?
        else {
            return Ok(());
        };

        let final_entry = if is_compressed(halfword) {
            let instr = parse_compressed_instruction(halfword);
            Instruction::from(&instr)
        } else {
            Instruction::DEFAULT
        };

        instructions.push(final_entry);

        if let Some(page_entry) = self
            .pages
            .get_mut((page_start >> memory::OFFSET_BITS) as usize)
        {
            *page_entry = Some(PageEntry {
                entries: instructions
                    .try_into()
                    .expect("instructions has exactly the length expected for PageEntry::entries"),
                _pd: PhantomData,
            });
        }

        Ok(())
    }

    /// Invalidate a range of pages corresponding to the provided range of memory.
    fn invalidate_range(&mut self, start_address: Address, length: u64) {
        for page_idx in (start_address >> memory::OFFSET_BITS)
            ..(start_address.saturating_add(length) >> memory::OFFSET_BITS)
        {
            self.pages
                .get_mut(page_idx as usize)
                .map(|entry| *entry = None);
        }
    }
}
