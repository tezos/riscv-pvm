// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State context abstraction for accessing PVM machine state
//!
//! This module provides the [`StateContext`] trait, enabling type-safe access to different parts
//! of the RISC-V PVM machine state through projections.
//!
//! The state context abstraction allows for:
//! - Reading from specific regions of machine core state
//! - Writing to specific regions of machine core state
//! - Type-safe access patterns through the projection system

pub(crate) mod projection;

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerReadWrite;
use crate::state_context::projection::MachineCoreProjection;

/// Context for accessing parts of the PVM state
pub trait StateContext {
    /// 64-bit integer type
    type X64;

    /// Read from a region of the machine core state.
    fn read_proj<P>(&mut self, param: P::Parameter) -> Self::X64
    where
        P: MachineCoreProjection<Target = u64>;

    /// Write to a region of the machine core state.
    fn write_proj<P>(&mut self, param: P::Parameter, value: Self::X64)
    where
        P: MachineCoreProjection<Target = u64>;
}

impl<MC: MemoryConfig, M: ManagerReadWrite> StateContext for MachineCoreState<MC, M> {
    type X64 = u64;

    #[inline]
    fn read_proj<P>(&mut self, param: P::Parameter) -> Self::X64
    where
        P: MachineCoreProjection<Target = u64>,
    {
        P::project_read(self, param)
    }

    #[inline]
    fn write_proj<P>(&mut self, param: P::Parameter, value: Self::X64)
    where
        P: MachineCoreProjection<Target = u64>,
    {
        P::project_write(self, param, value);
    }
}
