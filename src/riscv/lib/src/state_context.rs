// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

pub(crate) mod projection;

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerReadWrite;
use crate::state_context::projection::MachineCoreProjection;
use crate::state_context::projection::RegionCons;

/// Context for accessing parts of the PVM state
pub trait StateContext {
    /// 64-bit integer type
    type X64;

    /// Read from a region of the machine core state.
    fn read_machine_region<L, const LEN: usize>(&mut self, index: usize) -> Self::X64
    where
        L: MachineCoreProjection<Target = RegionCons<u64, LEN>>;

    /// Write to a region of the machine core state.
    fn write_machine_region<L, const LEN: usize>(&mut self, index: usize, value: Self::X64)
    where
        L: MachineCoreProjection<Target = RegionCons<u64, LEN>>;
}

impl<MC: MemoryConfig, M: ManagerReadWrite> StateContext for MachineCoreState<MC, M> {
    type X64 = u64;

    #[inline]
    fn read_machine_region<L, const LEN: usize>(&mut self, index: usize) -> Self::X64
    where
        L: MachineCoreProjection<Target = RegionCons<u64, LEN>>,
    {
        let region = L::project(self);
        M::region_read(region, index)
    }

    #[inline]
    fn write_machine_region<L, const LEN: usize>(&mut self, index: usize, value: Self::X64)
    where
        L: MachineCoreProjection<Target = RegionCons<u64, LEN>>,
    {
        let region = L::project_mut(self);
        M::region_write(region, index, value);
    }
}
