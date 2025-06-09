// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use super::block::Block;
use crate::cache_utils::FenceCounter;
use crate::machine_state::block_cache::state::BlockCache;
use crate::machine_state::block_cache::state::Cached;
use crate::machine_state::block_cache::state::CachedLayout;
use crate::machine_state::block_cache::state::PartialBlock;
use crate::machine_state::block_cache::state::PartialBlockLayout;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Atom;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::Many;
use crate::state_backend::Ref;

/// Configuration for a block cache
pub struct BlockCacheConfig<const SIZE: usize>;

impl<const SIZE: usize> BlockCacheConfig<SIZE> {
    /// Number of entries in the block cache
    pub const CACHE_SIZE: usize = if SIZE.is_power_of_two() {
        SIZE
    } else {
        panic!("BITS parameter does not match SIZE parameter");
    };

    const CACHE_MASK: usize = {
        Self::fence_counter_wrapping_protection();
        Self::CACHE_SIZE - 1
    };

    /// Compute the index of a cache bucket for a given address.
    #[inline(always)]
    pub const fn cache_index(addr: Address) -> usize {
        // We know that addr here is always u16-aligned. Therefore, we can safely halve the number
        // of buckets we look at.
        (addr >> 1) as usize & Self::CACHE_MASK
    }

    /// Assert that the fence counter would not wrap before every cache entry has been invalidated
    /// _at least_ once.
    const fn fence_counter_wrapping_protection() {
        let invalidation_count_until_wrapping = FenceCounter::MAX.0 as usize;
        let cache_entries = Self::CACHE_SIZE;

        assert!(
            invalidation_count_until_wrapping > cache_entries,
            "The fence counter does a full cycle before all cache entries could be invalidated!"
        );
    }
}

impl<const SIZE: usize> super::BlockCacheConfig for BlockCacheConfig<SIZE> {
    type Layout = (
        Atom<Address>,
        Atom<Address>,
        Atom<FenceCounter>,
        PartialBlockLayout,
        Many<CachedLayout, SIZE>,
    );

    type State<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase> = BlockCache<SIZE, B, MC, M>;

    fn bind<MC, B, M>(
        space: AllocatedOf<Self::Layout, M>,
        block_builder: B::BlockBuilder,
    ) -> Self::State<MC, B, M>
    where
        MC: MemoryConfig,
        B: Block<MC, M>,
        M: ManagerBase,
        M::ManagerRoot: ManagerReadWrite,
    {
        Self::State {
            current_block_addr: space.0,
            next_instr_addr: space.1,
            fence_counter: space.2,
            partial_block: PartialBlock::bind(space.3),
            entries: space
                .4
                .into_iter()
                .map(Cached::bind)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| "mismatching vector lengths for instruction cache")
                .unwrap(),
            block_builder,
        }
    }

    fn struct_ref<'a, MC, B, M, F>(
        instance: &'a Self::State<MC, B, M>,
    ) -> AllocatedOf<Self::Layout, F::Output>
    where
        MC: MemoryConfig,
        B: Block<MC, M>,
        M: ManagerBase,
        F: FnManager<Ref<'a, M>>,
    {
        (
            instance.current_block_addr.struct_ref::<F>(),
            instance.next_instr_addr.struct_ref::<F>(),
            instance.fence_counter.struct_ref::<F>(),
            instance.partial_block.struct_ref::<F>(),
            instance
                .entries
                .iter()
                .map(|entry| entry.struct_ref::<F>())
                .collect(),
        )
    }
}

/// The default block cache index bits
const DEFAULT_CACHE_BITS: usize = 20;

/// The default block cache size
const DEFAULT_CACHE_SIZE: usize = 1 << DEFAULT_CACHE_BITS;

/// The default block cache index bits for tests
const TEST_CACHE_BITS: usize = 12;

/// The default block cache for tests
const TEST_CACHE_SIZE: usize = 1 << TEST_CACHE_BITS;

/// The default configuration of the block cache
pub type DefaultCacheConfig = BlockCacheConfig<{ DEFAULT_CACHE_SIZE }>;

/// The default configuration of the block cache for testing
pub type TestCacheConfig = BlockCacheConfig<{ TEST_CACHE_SIZE }>;
