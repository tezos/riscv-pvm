// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of Zifencei extension for RISC-V

use crate::machine_state::MachineState;
use crate::machine_state::block_cache::BlockCacheConfig;
use crate::machine_state::block_cache::block::Block;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend;

impl<MC, BCC, B, M> MachineState<MC, BCC, B, M>
where
    MC: MemoryConfig,
    BCC: BlockCacheConfig,
    B: Block<MC, M>,
    M: state_backend::ManagerReadWrite,
{
    /// Execute a `fence.i` instruction.
    #[inline(always)]
    pub fn run_fencei(&mut self) {}
}
