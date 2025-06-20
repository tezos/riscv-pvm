// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;

use range_collections::RangeSet;
use range_collections::RangeSet2;

use crate::machine_state::block_cache::BlockCache;
use crate::machine_state::block_cache::BlockCacheConfig;
use crate::machine_state::block_cache::block::Block;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerReadWrite;

#[derive(Clone)]
pub struct RangedBlockCaches<
    MC: MemoryConfig,
    BCC: BlockCacheConfig,
    B: Block<MC, M>,
    M: ManagerBase,
> {
    allowed_ranges: RangeSet<[Address; 4]>,
    cache: BCC::State<MC, B, M>,
}

pub trait RangedBlockCache<MC: MemoryConfig, M: ManagerBase> {
    fn remove_range(&mut self, start: Address, length: u64)
    where
        M::ManagerRoot: ManagerReadWrite;

    fn insert_range(&mut self, start: Address, length: u64)
    where
        M::ManagerRoot: ManagerReadWrite;
}

pub struct DummyRangedBlockCache<MC: MemoryConfig, M: ManagerBase>(PhantomData<MC>, PhantomData<M>);

impl<MC: MemoryConfig, M: ManagerBase> Default for DummyRangedBlockCache<MC, M> {
    fn default() -> Self {
        DummyRangedBlockCache(PhantomData, PhantomData)
    }
}

impl<MC: MemoryConfig, M: ManagerBase> RangedBlockCache<MC, M> for DummyRangedBlockCache<MC, M> {
    fn remove_range(&mut self, _start: Address, _length: u64) {}

    fn insert_range(&mut self, _start: Address, _length: u64) {}
}

impl<MC: MemoryConfig, BCC: BlockCacheConfig, B: Block<MC, M>, M: ManagerBase>
    RangedBlockCache<MC, M> for RangedBlockCaches<MC, BCC, B, M>
{
    fn remove_range(&mut self, start: Address, length: u64)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        let new_range = RangeSet2::from(start..start.saturating_add(length));

        if self.allowed_ranges.intersects(&new_range) {
            // TODO: RV-XXX Resetting the entire cache is overkill. Perhaps we can optimise this by
            // removing only the affected blocks. However, the block cache interface currently does
            // not support this.
            self.cache.reset();
        }

        self.allowed_ranges.difference_with(&new_range);
    }

    fn insert_range(&mut self, start: Address, length: u64)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        let new_range = RangeSet2::from(start..start.saturating_add(length));
        self.allowed_ranges.union_with(&new_range);
    }
}

impl<const SIZE: usize, B: Block<MC, M>, MC: MemoryConfig, M: ManagerBase>
    RangedBlockCaches<MC, super::config::BlockCacheConfig<SIZE>, B, M>
{
    #[cfg(test)]
    pub(crate) fn get_block_instr(&mut self, addr: Address) -> Vec<Instruction> {
        self.cache.get_block_instr(addr)
    }

    #[cfg(test)]
    pub fn force_single_block_cache(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.insert_range(0, u64::MAX);
    }
}

impl<MC, BCC, B, M> super::BlockCache<MC, B, M> for RangedBlockCaches<MC, BCC, B, M>
where
    MC: MemoryConfig,
    BCC: BlockCacheConfig,
    B: Block<MC, M>,
    M: ManagerBase,
{
    fn new() -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        RangedBlockCaches {
            allowed_ranges: RangeSet::empty(),
            cache: super::BlockCache::new(),
        }
    }

    fn clone(&self) -> Self
    where
        B: Clone,
    {
        RangedBlockCaches {
            allowed_ranges: self.allowed_ranges.clone(),
            cache: self.cache.clone(),
        }
    }

    fn reset(&mut self)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        self.allowed_ranges = RangeSet::empty();
        self.cache.reset();
    }

    #[inline(always)]
    fn get_block(&mut self, addr: Address) -> Option<super::BlockCall<'_, B, MC, M>> {
        self.cache.get_block(addr)
    }

    #[inline(always)]
    fn push_instr_compressed(&mut self, addr: Address, instr: Instruction)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        self.cache.push_instr_compressed(addr, instr);
    }

    #[inline(always)]
    fn push_instr_uncompressed(&mut self, addr: Address, instr: Instruction)
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        self.cache.push_instr_uncompressed(addr, instr);
    }
}
