// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use super::block::Block;
use crate::machine_state::block_cache::state::BlockCache;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;

/// Configuration for a block cache
pub struct BlockCacheConfig<const SIZE: usize>;

impl<const SIZE: usize> BlockCacheConfig<SIZE> {
    /// Number of entries in the block cache
    pub const CACHE_SIZE: usize = if SIZE.is_power_of_two() {
        SIZE
    } else {
        panic!("BITS parameter does not match SIZE parameter");
    };

    const CACHE_MASK: usize = Self::CACHE_SIZE - 1;

    /// Compute the index of a cache bucket for a given address.
    #[inline(always)]
    pub const fn cache_index(addr: Address) -> usize {
        // We know that addr here is always u16-aligned. Therefore, we can safely halve the number
        // of buckets we look at.
        (addr >> 1) as usize & Self::CACHE_MASK
    }
}

impl<const SIZE: usize> super::BlockCacheConfig for BlockCacheConfig<SIZE> {
    type State<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase> = BlockCache<SIZE, B, MC, M>;
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
