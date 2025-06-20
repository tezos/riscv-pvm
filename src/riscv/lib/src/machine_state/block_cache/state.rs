// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;

use crate::array_utils;
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
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerReadWrite;

/// Block cache entry.
///
/// Contains the physical address & fence counter for validity checks, the
/// underlying [`Block`] state.
pub struct Cached<MC: MemoryConfig, B, M: ManagerBase> {
    pub(super) block: B,
    address: Address,
    _pd: PhantomData<(MC, M)>,
}

impl<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase> Cached<MC, B, M> {
    /// Create a new cache entry.
    pub fn new() -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        Self {
            block: B::new(),
            address: !0,
            _pd: PhantomData,
        }
    }

    fn reset(&mut self)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        self.address = !0;
        self.block.reset();
    }

    fn start_block(&mut self, block_addr: Address) {
        self.address = block_addr;
        self.block.start_block();
    }
}

impl<MC: MemoryConfig, B: Clone, M: ManagerBase> Clone for Cached<MC, B, M> {
    fn clone(&self) -> Self {
        Self {
            address: self.address,
            block: self.block.clone(),
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
    pub(super) current_block_addr: Address,

    /// The address of the next instruction to be injected into the current block
    pub(super) next_instr_addr: Address,

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

    fn reset_to(&mut self, addr: Address) {
        self.current_block_addr = addr;
        self.next_instr_addr = 0;
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
        M::ManagerRoot: ManagerReadWrite,
    {
        let mut block_addr = self.current_block_addr;

        let mut entry = Self::entry_mut(&mut self.entries, block_addr);
        let start = entry.address;

        let mut len_instr = entry.block.num_instr();

        if start != block_addr || start == addr {
            entry.start_block(block_addr);
            len_instr = 0;
        } else if len_instr == CACHE_INSTR {
            // The current block is full, start a new one
            self.reset_to(addr);
            block_addr = addr;

            entry = Self::entry_mut(&mut self.entries, block_addr);
            entry.start_block(block_addr);

            len_instr = 0;
        }

        entry.block.push_instr(instr);
        let new_len = len_instr + 1;

        let next_addr = addr + WIDTH;
        self.next_instr_addr = next_addr;

        let possible_block = Self::entry(&self.entries, next_addr);
        let adjacent_block_found = possible_block.address == next_addr
            && next_addr & OFFSET_MASK != 0
            && possible_block.block.num_instr() + new_len <= CACHE_INSTR;

        if adjacent_block_found {
            let num_instr = possible_block.block.num_instr();
            for i in 0..num_instr {
                // Need to resolve the adjacent block again because we may only keep one reference at a time
                // to `self.entries`.
                let new_block = Self::entry_mut(&mut self.entries, next_addr);
                let new_instr = new_block.block.instr()[i].instr;
                // Need to resolve the target block again because we may only keep one reference at a time
                // to `self.entries`.
                let current_entry = Self::entry_mut(&mut self.entries, block_addr);
                current_entry.block.push_instr(new_instr);
            }
            self.next_instr_addr = !0;
            self.current_block_addr = !0;
        }
    }

    /// *TEST ONLY* - retrieve the underlying instructions contained in the entry at the given
    /// address.
    #[cfg(test)]
    pub(crate) fn get_block_instr(&mut self, addr: Address) -> Vec<Instruction> {
        let entry = Self::entry_mut(&mut self.entries, addr);
        let instr = entry.block.instr();
        instr.iter().map(|cell| cell.instr).collect()
    }
}

impl<const SIZE: usize, MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase>
    super::BlockCache<MC, B, M> for BlockCache<SIZE, B, MC, M>
{
    fn new() -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        Self {
            current_block_addr: !0,
            next_instr_addr: !0,
            entries: array_utils::boxed_from_fn(|| Cached::new()),
        }
    }

    fn clone(&self) -> Self
    where
        B: Clone,
    {
        Self {
            current_block_addr: self.current_block_addr,
            next_instr_addr: self.next_instr_addr,
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
        M::ManagerRoot: ManagerReadWrite,
    {
        self.reset_to(!0);
        self.entries.iter_mut().for_each(Cached::reset);
    }

    fn get_block(&mut self, addr: Address) -> Option<BlockCall<'_, B, MC, M>> {
        let entry = Self::entry_mut(&mut self.entries, addr);

        if entry.address == addr && entry.block.num_instr() > 0 {
            Some(BlockCall { entry })
        } else {
            None
        }
    }

    fn push_instr_compressed(&mut self, addr: Address, instr: Instruction)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        debug_assert_eq!(
            instr.width(),
            InstrWidth::Compressed,
            "expected compressed instruction, found: {instr:?}"
        );

        let next_addr = self.next_instr_addr;

        // If the instruction is at the start of the page, we _must_ start a new block,
        // as we cannot allow blocks to cross page boundaries.
        if addr & OFFSET_MASK == 0 || addr != next_addr {
            self.reset_to(addr);
        }

        self.cache_inner::<{ InstrWidth::Compressed as u64 }>(addr, instr);
    }

    fn push_instr_uncompressed(&mut self, addr: Address, instr: Instruction)
    where
        M::ManagerRoot: ManagerReadWrite,
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

        let next_addr = self.next_instr_addr;

        // If the instruction is at the start of the page, we _must_ start a new block,
        // as we cannot allow blocks to cross page boundaries.
        if addr & OFFSET_MASK == 0 || addr != next_addr {
            self.reset_to(addr);
        }

        self.cache_inner::<{ InstrWidth::Uncompressed as u64 }>(addr, instr);
    }
}

#[cfg(test)]
mod tests {
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::machine_state::MachineState;
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
    use crate::machine_state::memory::Address;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::registers::XRegister;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::t0;
    use crate::machine_state::registers::t1;
    use crate::parser::instruction::InstrWidth;
    use crate::state_backend::owned_backend::Owned;
    use crate::traps::EnvironException;

    type TestState<M> = <TestCacheConfig as BlockCacheConfig>::State<M4K, Interpreted<M4K, M>, M>;

    // writing CACHE_INSTR to the block cache creates new block
    backend_test!(test_writing_full_block_fetchable_uncompressed, F, {
        let mut state = TestState::<F::Manager>::new();

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
        let mut state = TestState::<F::Manager>::new();

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
        let mut state = TestState::<F::Manager>::new();

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
        let mut state = TestState::<F::Manager>::new();

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
        let mut state = TestState::<F::Manager>::new();

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

    backend_test!(test_concat_blocks_suitable, F, {
        let mut state = TestState::<F::Manager>::new();

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
        let mut state = TestState::<F::Manager>::new();

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

        let mut block: TestState<Owned> = TestState::new();

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

            // XXX: Technically, the invalidation logic is no longer the same.
            block.reset();

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
        let mut block_cache = TestState::<Owned>::new();

        // Fetching empty block fails
        assert!(block_cache.get_block(0).is_none());

        block_cache.push_instr_compressed(0, Instruction::new_nop(InstrWidth::Compressed));

        // Fetching non-empty block succeeds
        assert!(block_cache.get_block(0).is_some());
    }
}
