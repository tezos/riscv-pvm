use std::marker::PhantomData;

use super::MachineCoreState;
use super::ProgramCounterUpdate;
use super::StepManyResult;
use super::instruction::Instruction;
use super::instruction::RunInstr;
use super::memory::Address;
use super::memory::Memory;
use super::memory::MemoryConfig;
use super::memory::OFFSET_BITS;
use crate::array_utils::boxed_from_fn;
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::parser::parse_uncompressed_instruction;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerReadWrite;
use crate::traps::EnvironException;
use crate::traps::Exception;

pub const OFFSET_MASK: u64 = 0b1111_1111_1111;

pub struct PageCache<MC: MemoryConfig, M: ManagerBase> {
    // pages for 1GB
    pages: Box<[Option<PageCacheEntry<MC, M>>; 1024 * 1024 * 1024 / 4096]>,
}

impl<MC: MemoryConfig, M: ManagerBase> PageCache<MC, M> {
    pub fn new() -> Self {
        Self {
            pages: boxed_from_fn(|| None),
        }
    }

    pub fn get_block(&mut self, addr: Address) -> Option<BlockCall<'_, MC, M>> {
        let page_index = (addr >> OFFSET_BITS) as usize;
        // aligned to 2 byte boundaries
        let page_offset = (addr & OFFSET_MASK) as usize >> 1;

        self.pages
            .get(page_index)
            .map(|page| page.as_ref())
            .flatten()
            .map(|page| &page.entries[page_offset..])
            .map(|entries| BlockCall { entries })
    }

    pub fn populate(&mut self, address: Address, core: &mut MachineCoreState<MC, M>)
    where
        M: ManagerReadWrite,
    {
        let page_index = (address >> OFFSET_BITS) as usize;

        if self
            .pages
            .get(page_index)
            .map(|page| page.is_some())
            .unwrap_or_default()
        {
            return;
        }

        //eprintln!("populate {page_index}");

        let page_start = address & !OFFSET_MASK;
        self.pages[page_index] = Some(PageCacheEntry::new(page_start, core));
    }
}

pub struct BlockCall<'a, MC: MemoryConfig, M: ManagerBase> {
    entries: &'a [CacheEntry<MC, M>],
}

impl<'a, MC: MemoryConfig, M: ManagerBase> std::fmt::Debug for BlockCall<'a, MC, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockCall")
            .field("entries", &self.entries)
            .finish()
    }
}

impl<'a, MC: MemoryConfig, M: ManagerBase> BlockCall<'a, MC, M> {
    pub fn run_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        //block_builder: &mut B::BlockBuilder,
        mut instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite,
    {
        run_block_inner(
            &self.entries[0..usize::min(max_steps, self.entries.len())],
            core,
            &mut instr_pc,
        )
    }
}

pub struct PageCacheEntry<MC: MemoryConfig, M: ManagerBase> {
    entries: Box<[CacheEntry<MC, M>; 2048]>,
}

impl<MC: MemoryConfig, M: ManagerBase> PageCacheEntry<MC, M> {
    pub fn new(page_start: Address, core: &mut MachineCoreState<MC, M>) -> Self
    where
        M: ManagerReadWrite,
    {
        let mut page_cache = Vec::with_capacity(2048);
        let mut words = vec![0u16; 2048];

        core.main_memory
            .read_all(page_start, words.as_mut_slice())
            .expect("The page is read/exec not write");

        let mut iter = words.into_iter().peekable();

        while let Some(lower_half) = iter.next() {
            let instr = if is_compressed(lower_half) {
                parse_compressed_instruction(lower_half)
            } else {
                let Some(upper_half) = iter.peek() else { break };

                parse_uncompressed_instruction((lower_half as u32) | ((*upper_half as u32) << 16))
            };

            let instr = Instruction::from(&instr);

            page_cache.push(CacheEntry {
                run_fn: instr.opcode.to_run(),
                instr,
            })
        }

        // final instruction is not-compressed
        // todo: exception to raise up to run-instr
        if page_cache.len() == 2047 {
            let bytes: u32 = core.main_memory.read(page_start + 4096 - 2).unwrap();
            let instr = parse_uncompressed_instruction(bytes);
            let instr = Instruction::from(&instr);

            page_cache.push(CacheEntry {
                instr,
                run_fn: instr.opcode.to_run(),
            })
        }

        Self {
            entries: page_cache.try_into().unwrap(),
        }
    }
}

pub struct CacheEntry<MC: MemoryConfig, M: ManagerBase> {
    instr: Instruction,
    run_fn: RunInstr<MC, M>,
}

impl<MC: MemoryConfig, M: ManagerClone> Clone for CacheEntry<MC, M> {
    fn clone(&self) -> Self {
        Self {
            instr: self.instr.clone(),
            run_fn: self.run_fn,
        }
    }
}

impl<MC: MemoryConfig, M: ManagerBase> std::fmt::Debug for CacheEntry<MC, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheEntry")
            .field("instr", &self.instr)
            .finish()
    }
}

fn run_block_inner<'a, MC: MemoryConfig, M: ManagerReadWrite + 'a>(
    entries: &'a [CacheEntry<MC, M>],
    core: &mut MachineCoreState<MC, M>,
    instr_pc: &mut Address,
) -> StepManyResult<Exception> {
    let mut result = StepManyResult::ZERO;

    let mut index = 0;
    while index < entries.len() {
        let entry = &entries[index];

        match (entry.run_fn)(&entry.instr.args, core) {
            Ok(ProgramCounterUpdate::Next(width)) => {
                *instr_pc += width as u64;
                core.hart.pc.write(*instr_pc);
                result.steps += 1;
                index += (width as usize) >> 1;
            }

            Ok(ProgramCounterUpdate::Set(instr_pc)) => {
                // Setting the instr_pc implies execution continuing
                // elsewhere and no longer within the current block, so the
                // current block instr_pc does not need updating.
                core.hart.pc.write(instr_pc);
                result.steps += 1;
                break;
            }

            Ok(ProgramCounterUpdate::Relative(offset)) => {
                core.hart.pc.write(instr_pc.wrapping_add_signed(offset));
                result.steps += 1;
                break;
            }

            Err(e) => {
                // Exceptions lead to a new address being set to handle it,
                // with no guarantee of it being the next instruction.
                result.error = Some(e);
                break;
            }
        }
    }

    result
}
