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
use crate::parser::instruction::InstrWidth;
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::parser::parse_uncompressed_instruction;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerReadWrite;
use crate::traps::EnvironException;
use crate::traps::Exception;

pub const OFFSET_MASK: u64 = 0b1111_1111_1111;

pub struct PageCache<MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> {
    // pages for 1GB
    pages: Box<[Option<PageCacheEntry<MC, BR, M>>; 1024 * 1024 * 1024 / 4096]>,
}

impl<MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> PageCache<MC, BR, M> {
    pub fn new() -> Self {
        Self {
            pages: boxed_from_fn(|| None),
        }
    }

    pub fn get_block(&mut self, addr: Address) -> Option<BlockCall<'_, MC, BR, M>> {
        let page_index = (addr >> OFFSET_BITS) as usize;

        self.pages
            .get_mut(page_index)
            .map(|page| page.as_mut())
            .flatten()
            .map(|page| BlockCall {
                entries: &mut page.entries,
                _pd: PhantomData,
            })
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

pub struct BlockCall<'a, MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> {
    entries: &'a mut [BR; 2048],
    _pd: PhantomData<(MC, M)>,
}

impl<'a, MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> std::fmt::Debug
    for BlockCall<'a, MC, BR, M>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockCall")
            .field("entries", &self.entries)
            .finish()
    }
}

impl<'a, MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> BlockCall<'a, MC, BR, M> {
    pub fn run_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        block_builder: &mut BR::BlockBuilder,
        mut instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite,
    {
        unsafe { BR::run_block(&mut self.entries, core, instr_pc, block_builder) }
    }
}

pub struct PageCacheEntry<MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> {
    entries: Box<[BR; 2048]>,
    _pd: PhantomData<(MC, M)>,
}

impl<MC: MemoryConfig, BR: BlockRunner<MC, M>, M: ManagerBase> PageCacheEntry<MC, BR, M> {
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

            page_cache.push(BR::new(instr));
        }

        // final instruction is not-compressed
        // todo: exception to raise up to run-instr
        if page_cache.len() == 2047 {
            let bytes: u32 = core.main_memory.read(page_start + 4096 - 2).unwrap();
            let instr = parse_uncompressed_instruction(bytes);
            let instr = Instruction::from(&instr);

            page_cache.push(BR::new(instr));
        }

        Self {
            entries: page_cache.try_into().unwrap(),
            _pd: PhantomData,
        }
    }
}

pub struct CacheEntry<MC: MemoryConfig, BR, M: ManagerBase> {
    instr: Instruction,
    run_fn: RunInstr<MC, M>,
    block_run: BR,
}

impl<MC: MemoryConfig, BR: Default, M: ManagerClone> Clone for CacheEntry<MC, BR, M> {
    fn clone(&self) -> Self {
        Self {
            instr: self.instr.clone(),
            run_fn: self.run_fn,
            block_run: Default::default(),
        }
    }
}

impl<MC: MemoryConfig, BR, M: ManagerBase> std::fmt::Debug for CacheEntry<MC, BR, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheEntry")
            .field("instr", &self.instr)
            .finish()
    }
}

fn run_block_inner<'a, MC: MemoryConfig, BR, M: ManagerReadWrite + 'a>(
    entries: &'a [CacheEntry<MC, BR, M>],
    core: &mut MachineCoreState<MC, M>,
    instr_pc: &mut Address,
) -> StepManyResult<Exception> {
    let mut result = StepManyResult::ZERO;

    let mut iter = entries.iter();

    while let Some(entry) = iter.next() {
        let res = (entry.run_fn)(&entry.instr.args, core);
        match res {
            Ok(ProgramCounterUpdate::Next(width)) => {
                *instr_pc += width as u64;
                core.hart.pc.write(*instr_pc);
                result.steps += 1;

                if width == InstrWidth::Uncompressed {
                    // skip the next instruction as it corresponds to the
                    // upper halfword of the current instr
                    let _ = iter.next();
                }
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

pub trait BlockRunner<MC: MemoryConfig, M: ManagerBase>: Sized + std::fmt::Debug {
    type BlockBuilder: Default + Sized;

    fn new(instr: Instruction) -> Self
    where
        M: ManagerReadWrite;

    unsafe fn run_block(
        instr: &mut [Self; 2048],
        core: &mut MachineCoreState<MC, M>,
        instr_pc: Address,
        block_builder: &mut Self::BlockBuilder,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite;
}

pub struct Interpreted;

#[derive(Default)]
pub struct InterpretedBlockBuilder;

impl<MC: MemoryConfig, M: ManagerBase> BlockRunner<MC, M> for CacheEntry<MC, Interpreted, M> {
    type BlockBuilder = InterpretedBlockBuilder;

    fn new(instr: Instruction) -> Self
    where
        M: ManagerReadWrite,
    {
        Self {
            instr,
            run_fn: instr.opcode.to_run(),
            block_run: Interpreted,
        }
    }

    unsafe fn run_block(
        instr: &mut [Self; 2048],
        core: &mut MachineCoreState<MC, M>,
        mut instr_pc: Address,
        _block_builder: &mut Self::BlockBuilder,
    ) -> StepManyResult<Exception>
    where
        M: ManagerReadWrite,
    {
        // aligned
        let offset = (instr_pc & OFFSET_MASK) as usize >> 1;

        run_block_inner(&instr[offset..], core, &mut instr_pc)
    }
}
