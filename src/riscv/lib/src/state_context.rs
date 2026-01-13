// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State context abstraction for accessing PVM machine state
//!
//! This module provides the [`StateContext`] trait, enabling type-safe access to different parts
//! of the RISC-V PVM machine state.
//!
//! The state context abstraction allows for:
//! - Reading from specific regions of machine core state
//! - Writing to specific regions of machine core state

pub(crate) mod projection;

use octez_riscv_data::components::atom::AtomMode;

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::FRegister;
use crate::machine_state::registers::FValue;
use crate::machine_state::registers::NonZeroXRegister;
use crate::machine_state::registers::XValue;

/// Context for accessing parts of the PVM state
pub trait StateContext {
    /// Value type for this context
    type Value<R>;

    /// Read the program counter.
    fn pc_read(&mut self) -> Self::Value<XValue>;

    /// Read from a non-zero integer register.
    fn xreg_read_nz(&mut self, reg: NonZeroXRegister) -> Self::Value<XValue>;

    /// Write to a non-zero integer register.
    fn xreg_write_nz(&mut self, reg: NonZeroXRegister, value: Self::Value<XValue>);

    /// Read from a floating-point register.
    fn freg_read(&mut self, reg: FRegister) -> Self::Value<FValue>;

    /// Write to a floating-point register.
    fn freg_write(&mut self, reg: FRegister, value: Self::Value<FValue>);

    /// Read the reservation set.
    fn reservation_set_read(&mut self) -> Self::Value<u64>;

    /// Write the reservation set.
    fn reservation_set_write(&mut self, value: Self::Value<u64>);
}

impl<MC: MemoryConfig, M: AtomMode> StateContext for MachineCoreState<MC, M> {
    type Value<R> = R;

    #[inline]
    fn pc_read(&mut self) -> Self::Value<XValue> {
        self.hart.pc.read()
    }

    #[inline]
    fn xreg_read_nz(&mut self, reg: NonZeroXRegister) -> Self::Value<XValue> {
        self.hart.xregisters.read_nz(reg)
    }

    #[inline]
    fn xreg_write_nz(&mut self, reg: NonZeroXRegister, value: Self::Value<XValue>) {
        self.hart.xregisters.write_nz(reg, value);
    }

    #[inline]
    fn freg_read(&mut self, reg: FRegister) -> Self::Value<FValue> {
        self.hart.fregisters.read(reg)
    }

    #[inline]
    fn freg_write(&mut self, reg: FRegister, value: Self::Value<FValue>) {
        self.hart.fregisters.write(reg, value);
    }

    #[inline]
    fn reservation_set_read(&mut self) -> Self::Value<u64> {
        self.hart.reservation_set.start_addr.read()
    }

    #[inline]
    fn reservation_set_write(&mut self, value: Self::Value<u64>) {
        self.hart.reservation_set.start_addr.write(value);
    }
}

/// Context where you can update the program counter.
pub(crate) trait PcWriteContext {
    /// Value type for this context
    type Value<R>;

    /// Write the program counter.
    fn pc_write(&mut self, value: Self::Value<XValue>);
}

impl<MC: MemoryConfig, M: AtomMode> PcWriteContext for MachineCoreState<MC, M> {
    type Value<R> = R;

    #[inline]
    fn pc_write(&mut self, value: Self::Value<XValue>) {
        self.hart.pc.write(value);
    }
}
