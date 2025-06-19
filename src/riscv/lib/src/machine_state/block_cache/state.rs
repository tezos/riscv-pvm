// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;

use crate::cache_utils::FenceCounter;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::StepManyResult;
use crate::machine_state::block_cache::BlockCall;
use crate::machine_state::block_cache::CACHE_INSTR;
use crate::machine_state::block_cache::block::Block;
use crate::machine_state::block_cache::config::BlockCacheConfig;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::OFFSET_MASK;
use crate::machine_state::memory::PAGE_SIZE;
use crate::parser::instruction::InstrWidth;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;
use crate::state_backend::Ref;
use crate::traps::EnvironException;

/// Layout of a partial block.
pub type PartialBlockLayout = (Atom<Address>, Atom<bool>, Atom<u8>);

/// Structure used to remember that a block was only partway executed
/// before needing to pause due to `max_steps == 0`.
///
/// If a block is being partially executed, if either:
/// - an error occurs
/// - a jump or branch occurs then the partial block is reset, and execution will continue with a
///   potentially different block.
pub struct PartialBlock<M: ManagerBase> {
    addr: Cell<Address, M>,
    in_progress: Cell<bool, M>,
    progress: Cell<u8, M>,
}

impl<M: ManagerClone> Clone for PartialBlock<M> {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr.clone(),
            in_progress: self.in_progress.clone(),
            progress: self.progress.clone(),
        }
    }
}

impl<M: ManagerBase> PartialBlock<M> {
    /// Bind the allocated space to produce a [`PartialBlock`].
    pub(super) fn bind(space: AllocatedOf<PartialBlockLayout, M>) -> Self {
        Self {
            addr: space.0,
            in_progress: space.1,
            progress: space.2,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the [`PartialBlockLayout`]'s allocated
    /// structure containing the constituents of `N` that were produced from the constituents of
    /// `&M`.
    pub(super) fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<PartialBlockLayout, F::Output> {
        (
            self.addr.struct_ref::<F>(),
            self.in_progress.struct_ref::<F>(),
            self.progress.struct_ref::<F>(),
        )
    }

    fn reset(&mut self)
    where
        M: ManagerWrite,
    {
        self.in_progress.write(false);
        self.addr.write(0);
        self.progress.write(0);
    }

    /// Run a block against the machine state.
    ///
    /// When calling this function, there must be no partial block in progress. To ensure
    /// this, you must always run [`super::BlockCache::complete_current_block`] prior to fetching
    /// and running a new block.
    #[cold]
    pub(super) fn run_block_partial<B: Block<MC, M>, MC: MemoryConfig>(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        max_steps: usize,
        entry: &mut Cached<MC, B, M>,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite,
    {
        // start a new block
        self.in_progress.write(true);
        self.progress.write(0);
        self.addr.write(entry.address.read());

        self.run_partial_inner(core, max_steps, entry)
    }

    fn run_partial_inner<B: Block<MC, M>, MC: MemoryConfig>(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        max_steps: usize,
        entry: &mut Cached<MC, B, M>,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite,
    {
        let mut result = StepManyResult::ZERO;

        // Protect against partial blocks being executed when
        // no steps are remaining
        if max_steps == 0 {
            return result;
        }

        let mut progress = self.progress.read();
        let mut instr_pc = core.hart.pc.read();

        let range = progress as usize..;
        for instr in entry.block.instr()[range].iter() {
            match super::run_instr(instr, core) {
                Ok(ProgramCounterUpdate::Next(width)) => {
                    instr_pc += width as u64;
                    core.hart.pc.write(instr_pc);
                    result.steps += 1;
                    progress += 1;

                    if result.steps >= max_steps {
                        break;
                    }
                }

                Ok(ProgramCounterUpdate::Set(instr_pc)) => {
                    // Setting the instr_pc implies execution continuing
                    // elsewhere - and no longer within the current block.
                    core.hart.pc.write(instr_pc);
                    result.steps += 1;
                    self.reset();
                    return result;
                }

                Err(e) => {
                    self.reset();

                    // Exceptions lead to a new address being set to handle it,
                    // with no guarantee of it being the next instruction.
                    if let Err(error) = core.handle_step_result(instr_pc, Err(e)) {
                        result.error = Some(error);
                        return result;
                    }

                    // If we succesfully handled an error, need to increment steps one more.
                    result.steps += 1;
                    return result;
                }
            }
        }

        if progress as usize == entry.block.num_instr() {
            // We finished the block in exactly the number of steps left
            self.reset();
        } else {
            // Remember the progress made through the block, when we later
            // continue executing it
            self.progress.write(progress);
        }

        result
    }
}

impl<M: ManagerBase> NewState<M> for PartialBlock<M> {
    fn new(manager: &mut M) -> Self
    where
        M: ManagerAlloc,
    {
        Self {
            addr: Cell::new(manager),
            in_progress: Cell::new(manager),
            progress: Cell::new(manager),
        }
    }
}

/// The layout of block cache entries, see [`Cached`] for more information.
pub type CachedLayout = (
    Atom<Address>,
    Atom<FenceCounter>,
    Atom<u8>,
    [Atom<Instruction>; CACHE_INSTR],
);

/// Block cache entry.
///
/// Contains the physical address & fence counter for validity checks, the
/// underlying [`Block`] state.
pub struct Cached<MC: MemoryConfig, B, M: ManagerBase> {
    pub(super) block: B,
    address: Cell<Address, M>,
    fence_counter: Cell<FenceCounter, M>,
    _pd: PhantomData<MC>,
}

impl<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase> Cached<MC, B, M> {
    /// Bind the allocated space to produce a [`Cached`] entry.
    pub(super) fn bind(space: AllocatedOf<CachedLayout, M>) -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        Self {
            address: space.0,
            fence_counter: space.1,
            block: B::bind((space.2, space.3)),
            _pd: PhantomData,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the [`CachedLayout`]'s allocated
    /// structure containing the constituents of `N` that were produced from the constituents of
    /// `&M`.
    pub(super) fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<CachedLayout, F::Output> {
        let (len_instr, instr) = self.block.struct_ref::<'a, F>();
        (
            self.address.struct_ref::<F>(),
            self.fence_counter.struct_ref::<F>(),
            len_instr,
            instr,
        )
    }

    fn invalidate(&mut self)
    where
        M: ManagerWrite,
    {
        self.address.write(!0);
        self.block.invalidate();
    }

    fn reset(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.address.write(!0);
        self.fence_counter.write(FenceCounter::INITIAL);
        self.block.reset();
    }

    fn start_block(&mut self, block_addr: Address, fence_counter: FenceCounter)
    where
        M: ManagerWrite,
    {
        self.address.write(block_addr);
        self.block.start_block();
        self.fence_counter.write(fence_counter);
    }
}

impl<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase> NewState<M> for Cached<MC, B, M> {
    fn new(manager: &mut M) -> Self
    where
        M: ManagerAlloc,
    {
        Self {
            address: Cell::new_with(manager, !0),
            block: B::new(manager),
            fence_counter: Cell::new(manager),
            _pd: PhantomData,
        }
    }
}

impl<MC: MemoryConfig, B: Clone, M: ManagerClone> Clone for Cached<MC, B, M> {
    fn clone(&self) -> Self {
        Self {
            address: self.address.clone(),
            block: self.block.clone(),
            fence_counter: self.fence_counter.clone(),
            _pd: PhantomData,
        }
    }
}

/// Entries in the block cache
type Entries<const SIZE: usize, MC, B, M> = Box<[Cached<MC, B, M>; SIZE]>;

/// The block cache - caching sequences of instructions by physical address.
///
/// The number of entries is controlled by the `BCL` layout parameter.
pub struct BlockCache<const SIZE: usize, B: Block<MC, M>, MC: MemoryConfig, M: ManagerBase> {
    /// Starting address of the current block
    pub(super) current_block_addr: Cell<Address, M>,

    /// The address of the next instruction to be injected into the current block
    pub(super) next_instr_addr: Cell<Address, M>,

    /// Fence counter for the entire block cache
    pub(super) fence_counter: Cell<FenceCounter, M>,

    /// Current block being executed
    pub(super) partial_block: PartialBlock<M>,

    /// Block entries
    pub(super) entries: Entries<SIZE, MC, B, M>,
}

impl<const SIZE: usize, B: Block<MC, M>, MC: MemoryConfig, M: ManagerBase>
    BlockCache<SIZE, B, MC, M>
{
    fn entry(entries: &Entries<SIZE, MC, B, M>, addr: Address) -> &Cached<MC, B, M> {
        &entries[BlockCacheConfig::<SIZE>::cache_index(addr)]
    }

    fn entry_mut(entries: &mut Entries<SIZE, MC, B, M>, addr: Address) -> &mut Cached<MC, B, M> {
        &mut entries[BlockCacheConfig::<SIZE>::cache_index(addr)]
    }

    fn reset_to(&mut self, addr: Address)
    where
        M: ManagerWrite,
    {
        self.current_block_addr.write(addr);
        self.next_instr_addr.write(0);
    }

    /// Add the instruction into a block.
    ///
    /// If the block is full, a new block will be started.
    ///
    /// If there is a block at the next address, we will
    /// merge it into the current block if possible. We
    /// ensure it is valid (it is part of the same fence,
    /// on the same page and that the combined block will
    /// not exceed the maximum number of instructions).
    fn cache_inner<const WIDTH: u64>(&mut self, addr: Address, instr: Instruction)
    where
        M: ManagerReadWrite,
    {
        let mut block_addr = self.current_block_addr.read();
        let fence_counter = self.fence_counter.read();

        let mut entry = Self::entry_mut(&mut self.entries, block_addr);
        let start = entry.address.read();

        let mut len_instr = entry.block.num_instr();

        if start != block_addr || start == addr {
            entry.start_block(block_addr, fence_counter);
            len_instr = 0;
        } else if len_instr == CACHE_INSTR {
            // The current block is full, start a new one
            self.reset_to(addr);
            block_addr = addr;

            entry = Self::entry_mut(&mut self.entries, block_addr);
            entry.start_block(block_addr, fence_counter);

            len_instr = 0;
        }

        entry.block.push_instr(instr);
        let new_len = len_instr + 1;

        let next_addr = addr + WIDTH;
        self.next_instr_addr.write(next_addr);

        let possible_block = Self::entry(&self.entries, next_addr);
        let adjacent_block_found = possible_block.address.read() == next_addr
            && possible_block.fence_counter.read() == fence_counter
            && next_addr & OFFSET_MASK != 0
            && possible_block.block.num_instr() + new_len <= CACHE_INSTR;

        if adjacent_block_found {
            let num_instr = possible_block.block.num_instr();
            for i in 0..num_instr {
                // Need to resolve the adjacent block again because we may only keep one reference at a time
                // to `self.entries`.
                let new_block = Self::entry_mut(&mut self.entries, next_addr);
                let new_instr = new_block.block.instr()[i].read_stored();
                // Need to resolve the target block again because we may only keep one reference at a time
                // to `self.entries`.
                let current_entry = Self::entry_mut(&mut self.entries, block_addr);
                current_entry.block.push_instr(new_instr);
            }
            self.next_instr_addr.write(!0);
            self.current_block_addr.write(!0);
        }
    }

    /// *TEST ONLY* - retrieve the underlying instructions contained in the entry at the given
    /// address.
    #[cfg(test)]
    pub(crate) fn get_block_instr(&mut self, addr: Address) -> Vec<Instruction>
    where
        M: ManagerRead,
    {
        let entry = Self::entry_mut(&mut self.entries, addr);
        let instr = entry.block.instr();
        instr.iter().map(|cell| cell.read_stored()).collect()
    }
}

impl<const SIZE: usize, MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase>
    super::BlockCache<MC, B, M> for BlockCache<SIZE, B, MC, M>
{
    fn new(manager: &mut M) -> Self
    where
        M: ManagerAlloc,
    {
        Self {
            current_block_addr: Cell::new_with(manager, !0),
            next_instr_addr: Cell::new_with(manager, !0),
            fence_counter: Cell::new(manager),
            partial_block: PartialBlock::new(manager),
            entries: NewState::new(manager),
        }
    }

    fn clone(&self) -> Self
    where
        B: Clone,
        M: ManagerClone,
    {
        Self {
            current_block_addr: self.current_block_addr.clone(),
            fence_counter: self.fence_counter.clone(),
            next_instr_addr: self.next_instr_addr.clone(),
            partial_block: self.partial_block.clone(),
            // This may appear like a wild way to clone a boxed array. But! This way avoids that
            // the array gets temporarily allocated on the stack whhich causes a stack overflow on
            // most platforms.
            entries: self
                .entries
                .to_vec()
                .try_into()
                .map_err(|_| "mismatching vector lengths in block cache")
                .unwrap(),
        }
    }

    fn reset(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.fence_counter.write(FenceCounter::INITIAL);
        self.reset_to(!0);
        self.entries.iter_mut().for_each(Cached::reset);
        self.partial_block.reset();
    }

    fn invalidate(&mut self)
    where
        M: ManagerReadWrite,
    {
        let counter = self.fence_counter.read();
        self.fence_counter.write(counter.next());
        self.reset_to(!0);
        Self::entry_mut(&mut self.entries, counter.0 as Address).invalidate();
    }

    fn get_block(&mut self, addr: Address) -> Option<BlockCall<'_, B, MC, M>>
    where
        M: ManagerRead,
    {
        debug_assert!(
            !self.partial_block.in_progress.read(),
            "Get block was called with a partial block in progress"
        );

        let entry = Self::entry_mut(&mut self.entries, addr);

        if entry.address.read() == addr
            && self.fence_counter.read() == entry.fence_counter.read()
            && entry.block.num_instr() > 0
        {
            Some(BlockCall {
                entry,
                partial: &mut self.partial_block,
            })
        } else {
            None
        }
    }

    fn push_instr_compressed(&mut self, addr: Address, instr: Instruction)
    where
        M: ManagerReadWrite,
    {
        debug_assert_eq!(
            instr.width(),
            InstrWidth::Compressed,
            "expected compressed instruction, found: {instr:?}"
        );

        let next_addr = self.next_instr_addr.read();

        // If the instruction is at the start of the page, we _must_ start a new block,
        // as we cannot allow blocks to cross page boundaries.
        if addr & OFFSET_MASK == 0 || addr != next_addr {
            self.reset_to(addr);
        }

        self.cache_inner::<{ InstrWidth::Compressed as u64 }>(addr, instr);
    }

    fn push_instr_uncompressed(&mut self, addr: Address, instr: Instruction)
    where
        M: ManagerReadWrite,
    {
        debug_assert_eq!(
            instr.width(),
            InstrWidth::Uncompressed,
            "expected uncompressed instruction, found: {instr:?}"
        );

        // ensure uncompressed does not cross page boundaries
        const END_OF_PAGE: Address = PAGE_SIZE.get() - 2;
        if addr % PAGE_SIZE.get() == END_OF_PAGE {
            return;
        }

        let next_addr = self.next_instr_addr.read();

        // If the instruction is at the start of the page, we _must_ start a new block,
        // as we cannot allow blocks to cross page boundaries.
        if addr & OFFSET_MASK == 0 || addr != next_addr {
            self.reset_to(addr);
        }

        self.cache_inner::<{ InstrWidth::Uncompressed as u64 }>(addr, instr);
    }

    fn complete_current_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        max_steps: usize,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite,
    {
        if !self.partial_block.in_progress.read() {
            return StepManyResult::ZERO;
        }

        let entry = Self::entry_mut(&mut self.entries, self.partial_block.addr.read());

        self.partial_block.run_partial_inner(core, max_steps, entry)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::MachineState;
    use crate::machine_state::StepManyResult;
    use crate::machine_state::block_cache::BlockCache;
    use crate::machine_state::block_cache::BlockCacheConfig;
    use crate::machine_state::block_cache::CACHE_INSTR;
    use crate::machine_state::block_cache::block::Block;
    use crate::machine_state::block_cache::block::Interpreted;
    use crate::machine_state::block_cache::block::InterpretedBlockBuilder;
    use crate::machine_state::block_cache::config::TestCacheConfig;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::instruction::OpCode;
    use crate::machine_state::instruction::tagged_instruction::TaggedArgs;
    use crate::machine_state::instruction::tagged_instruction::TaggedInstruction;
    use crate::machine_state::instruction::tagged_instruction::TaggedRegister;
    use crate::machine_state::memory;
    use crate::machine_state::memory::Address;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::registers::XRegister;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::t0;
    use crate::machine_state::registers::t1;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;
    use crate::state_backend::owned_backend::Owned;
    use crate::traps::EnvironException;

    type TestState<M> = <TestCacheConfig as BlockCacheConfig>::State<M4K, Interpreted<M4K, M>, M>;

    // writing CACHE_INSTR to the block cache creates new block
    backend_test!(test_writing_full_block_fetchable_uncompressed, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let uncompressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::X64Store,
            args: TaggedArgs {
                rs1: t1.into(),
                rs2: t0.into(),
                rd: TaggedRegister::X(XRegister::x1),
                imm: 8,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = 10;

        for offset in 0..(CACHE_INSTR as u64) {
            state.push_instr_uncompressed(addr + offset * 4, uncompressed);
        }

        let block = state.get_block(addr);
        assert!(block.is_some());
        assert_eq!(CACHE_INSTR, block.unwrap().entry.block.num_instr());
    });

    backend_test!(test_writing_full_block_fetchable_compressed, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let compressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::Li,
            args: TaggedArgs {
                rd: nz::a0.into(),
                imm: 1,
                rs1: nz::ra.into(),
                rs2: nz::ra.into(),
                width: InstrWidth::Compressed,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = 10;

        for offset in 0..(CACHE_INSTR as u64) {
            state.push_instr_compressed(addr + offset * 2, compressed);
        }

        let block = state.get_block(addr);
        assert!(block.is_some());
        assert_eq!(CACHE_INSTR, block.unwrap().entry.block.num_instr());
    });

    // writing instructions immediately creates block
    backend_test!(test_writing_half_block_fetchable_compressed, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let compressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::Li,
            args: TaggedArgs {
                rd: nz::a0.into(),
                imm: 1,
                rs1: nz::ra.into(),
                rs2: nz::ra.into(),
                width: InstrWidth::Compressed,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = 10;

        for offset in 0..((CACHE_INSTR / 2) as u64) {
            state.push_instr_compressed(addr + offset * 2, compressed);
        }

        let block = state.get_block(addr);
        assert!(block.is_some());
        assert_eq!(CACHE_INSTR / 2, block.unwrap().entry.block.num_instr());
    });

    backend_test!(test_writing_two_blocks_fetchable_compressed, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let compressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::Li,
            args: TaggedArgs {
                rd: nz::a0.into(),
                imm: 1,
                rs1: nz::ra.into(),
                rs2: nz::ra.into(),
                width: InstrWidth::Compressed,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = 10;

        for offset in 0..((CACHE_INSTR * 2) as u64) {
            state.push_instr_compressed(addr + offset * 2, compressed);
        }

        let block = state.get_block(addr);
        assert!(block.is_some());
        assert_eq!(CACHE_INSTR, block.unwrap().entry.block.num_instr());

        let block = state.get_block(addr + 2 * CACHE_INSTR as u64);
        assert!(block.is_some());
        assert_eq!(CACHE_INSTR, block.unwrap().entry.block.num_instr());
    });

    // writing across pages offset two blocks next to each other
    backend_test!(test_crossing_page_exactly_creates_new_block, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let compressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::Li,
            args: TaggedArgs {
                rd: nz::a0.into(),
                imm: 1,
                rs1: nz::ra.into(),
                rs2: nz::ra.into(),
                width: InstrWidth::Compressed,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = PAGE_SIZE.get() - 10;

        for offset in 0..10 {
            state.push_instr_compressed(addr + offset * 2, compressed);
        }

        let block = state.get_block(addr);
        assert!(block.is_some());
        assert_eq!(5, block.unwrap().entry.block.num_instr());

        let block = state.get_block(addr + 10);
        assert!(block.is_some());
        assert_eq!(5, block.unwrap().entry.block.num_instr());
    });

    backend_test!(test_partial_block_executes, F, {
        let mut manager = F::manager();
        let mut core_state = MachineCoreState::<M4K, _>::new(&mut manager);
        let mut block_state = TestState::<F::Manager>::new(&mut manager);

        let addiw = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::AddWordImmediate,
            args: TaggedArgs {
                rd: nz::a1.into(),
                rs1: a1.into(),
                imm: 257,
                rs2: TaggedRegister::X(XRegister::x1),
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let block_addr = memory::FIRST_ADDRESS;

        for offset in 0..10 {
            block_state.push_instr_uncompressed(block_addr + offset * 4, addiw);
        }

        core_state.hart.pc.write(block_addr);

        // Execute the first 5 instructions
        let block = block_state.get_block(block_addr).unwrap();
        let StepManyResult { steps, error: None } =
            block
                .partial
                .run_block_partial(&mut core_state, 5, block.entry)
        else {
            panic!()
        };

        assert_eq!(steps, 5);
        assert!(block_state.partial_block.in_progress.read());
        assert_eq!(5, block_state.partial_block.progress.read());
        assert_eq!(block_addr, block_state.partial_block.addr.read());
        assert_eq!(block_addr + 5 * 4, core_state.hart.pc.read());

        // Execute no steps
        let StepManyResult { steps, error: None } =
            block_state.complete_current_block(&mut core_state, 0)
        else {
            panic!()
        };

        assert_eq!(steps, 0);
        assert!(block_state.partial_block.in_progress.read());
        assert_eq!(5, block_state.partial_block.progress.read());
        assert_eq!(block_addr, block_state.partial_block.addr.read());
        assert_eq!(block_addr + 5 * 4, core_state.hart.pc.read());

        // Execute the next 2 instructions
        let StepManyResult { steps, error: None } =
            block_state.complete_current_block(&mut core_state, 2)
        else {
            panic!()
        };

        assert_eq!(steps, 2);
        assert!(block_state.partial_block.in_progress.read());
        assert_eq!(7, block_state.partial_block.progress.read());
        assert_eq!(block_addr, block_state.partial_block.addr.read());
        assert_eq!(block_addr + 7 * 4, core_state.hart.pc.read());

        // Finish the block. We don't consume all the steps
        let StepManyResult { steps, error: None } =
            block_state.complete_current_block(&mut core_state, 5)
        else {
            panic!()
        };

        assert_eq!(steps, 3);
        assert!(!block_state.partial_block.in_progress.read());
        assert_eq!(0, block_state.partial_block.progress.read());
        assert_eq!(0, block_state.partial_block.addr.read());
        assert_eq!(block_addr + 10 * 4, core_state.hart.pc.read());
    });

    backend_test!(test_concat_blocks_suitable, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let uncompressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::X64Store,
            args: TaggedArgs {
                rs1: t1.into(),
                rs2: t0.into(),
                rd: TaggedRegister::X(XRegister::x1),
                imm: 8,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = 30;
        let preceding_num_instr: u64 = 5;

        for offset in 0..(CACHE_INSTR as u64 - preceding_num_instr) {
            state.push_instr_uncompressed(addr + offset * 4, uncompressed);
        }

        for offset in 0..preceding_num_instr {
            state
                .push_instr_uncompressed(addr - preceding_num_instr * 4 + offset * 4, uncompressed);
        }

        let block = state.get_block(addr - 20);
        assert!(block.is_some());
        assert_eq!(CACHE_INSTR, block.unwrap().entry.block.num_instr());

        let old_block = state.get_block(addr);
        assert!(old_block.is_some());
        assert_eq!(15, old_block.unwrap().entry.block.num_instr());
    });

    backend_test!(test_concat_blocks_too_big, F, {
        let mut state = TestState::<F::Manager>::new(&mut F::manager());

        let uncompressed = Instruction::try_from(TaggedInstruction {
            opcode: OpCode::X64Store,
            args: TaggedArgs {
                rs1: t1.into(),
                rs2: t0.into(),
                rd: TaggedRegister::X(XRegister::x1),
                imm: 8,
                ..TaggedArgs::DEFAULT
            },
        })
        .unwrap();

        let addr = 30;
        let preceding_num_instr: u64 = 5;

        for offset in 0..(CACHE_INSTR as u64 - preceding_num_instr + 1) {
            state.push_instr_uncompressed(addr + offset * 4, uncompressed);
        }

        for offset in 0..preceding_num_instr {
            state
                .push_instr_uncompressed(addr - preceding_num_instr * 4 + offset * 4, uncompressed);
        }

        let first_block = state.get_block(addr - preceding_num_instr * 4);
        assert!(first_block.is_some());
        assert_eq!(
            preceding_num_instr,
            first_block.unwrap().entry.block.num_instr() as u64
        );

        let second_block = state.get_block(addr);
        assert!(second_block.is_some());
        assert_eq!(
            CACHE_INSTR - preceding_num_instr as usize + 1,
            second_block.unwrap().entry.block.num_instr()
        );
    });

    /// The initialised block cache must not return any blocks. This is especially important for
    /// blocks at address 0 which at one point were accidentally valid but empty which caused loops.
    #[test]
    fn test_init_block() {
        // This test only makes sense if the test cache size isn't 0.
        if TestCacheConfig::CACHE_SIZE < 1 {
            panic!("Test cache size must be at least 1");
        }

        let check_block = |block: &mut TestState<Owned>| {
            for i in 0..TestCacheConfig::CACHE_SIZE {
                assert!(block.get_block(i as Address).is_none());
            }
        };

        let populate_block = |block: &mut TestState<Owned>| {
            for i in 0..TestCacheConfig::CACHE_SIZE {
                block.push_instr_uncompressed(
                    i as Address,
                    Instruction::try_from(TaggedInstruction {
                        opcode: OpCode::Add,
                        args: TaggedArgs {
                            rd: nz::a1.into(),
                            rs1: nz::a1.into(),
                            rs2: nz::a2.into(),
                            ..TaggedArgs::DEFAULT
                        },
                    })
                    .unwrap(),
                );
            }
        };

        let mut block: TestState<Owned> = TestState::new(&mut Owned);

        // The initial block cache should not return any blocks.
        check_block(&mut block);

        populate_block(&mut block);
        block.reset();

        // The reset block cache should not return any blocks.
        check_block(&mut block);

        // We check the invalidation logic multiple times because it works progressively, not in
        // one go.
        for _ in 0..TestCacheConfig::CACHE_SIZE {
            populate_block(&mut block);
            block.invalidate();

            // The invalidated block cache should not return any blocks.
            check_block(&mut block);
        }
    }

    /// The initialised block cache has an entry for address 0. The block at address 0 happens to
    /// be empty, which causes the step function to loop indefinitely when it runs the block.
    #[test]
    fn test_run_addr_zero() {
        let mut state: MachineState<M4K, TestCacheConfig, Interpreted<M4K, Owned>, Owned> =
            MachineState::new(&mut Owned, InterpretedBlockBuilder);

        // Encoding of ECALL instruction
        const ECALL: u32 = 0b1110011;

        state.core.hart.pc.write(0);
        state
            .core
            .main_memory
            .write_instruction_unchecked(0, ECALL)
            .unwrap();

        let result = state.step();
        assert_eq!(result, Err(EnvironException::EnvCall));
    }

    /// The initialised block cache has an entry for address 0. The block at address 0 happens to
    /// be empty, which causes the step function to loop indefinitely when it runs the block.
    #[test]
    fn test_get_empty_block_fails() {
        let mut block_cache = TestState::new(&mut Owned);

        // Fetching empty block fails
        assert!(block_cache.get_block(0).is_none());

        block_cache.push_instr_compressed(0, Instruction::new_nop(InstrWidth::Compressed));

        // Fetching non-empty block succeeds
        assert!(block_cache.get_block(0).is_some());
    }
}
