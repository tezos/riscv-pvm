// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of RV_64_A extension for RISC-V
//!
//! Chapter 8 - Unprivileged spec

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::BitXor;

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory;
use crate::machine_state::registers::XRegister;
use crate::state_backend as backend;
use crate::traps::Exception;

impl<MC, M> MachineCoreState<MC, M>
where
    MC: memory::MemoryConfig,
    M: backend::ManagerReadWrite,
{
    /// `AMOSWAP.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and writes val(rs2)
    /// back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoswapd(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, |_, value_rs2| value_rs2)
    }

    /// `AMOXOR.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// XORing it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoxord(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, u64::bitxor)
    }

    /// `AMOAND.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// ANDing it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoandd(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, u64::bitand)
    }

    /// `AMOOR.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// ORing it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoord(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, u64::bitor)
    }

    /// `AMOMIN.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the minimum
    /// between it and val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amomind(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, |value_rs1, value_rs2| {
            (value_rs1 as i64).min(value_rs2 as i64) as u64
        })
    }

    /// `AMOMAX.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the maximum
    /// between it and val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amomaxd(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, |value_rs1, value_rs2| {
            (value_rs1 as i64).max(value_rs2 as i64) as u64
        })
    }

    /// `AMOMINU.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the minimum
    /// between it and val(rs2) back to the address in rs1, treating both as
    /// unsigned values.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amominud(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, u64::min)
    }

    /// `AMOMAXU.D` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the maximum
    /// between it and val(rs2) back to the address in rs1, treating both as
    /// unsigned values.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amomaxud(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_d(rs1, rs2, rd, u64::max)
    }
}

#[cfg(test)]
mod test {
    use std::ops::BitAnd;
    use std::ops::BitOr;
    use std::ops::BitXor;

    use proptest::prelude::*;

    use crate::backend_test;
    use crate::interpreter::rv32a::test::test_amo;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;

    test_amo!(run_amoswapd, |_, r2_val| r2_val, 8, u64);

    test_amo!(run_amoxord, u64::bitxor, 8, u64);

    test_amo!(run_amoandd, u64::bitand, 8, u64);

    test_amo!(run_amoord, u64::bitor, 8, u64);

    test_amo!(
        run_amomind,
        |r1_val, r2_val| i64::min(r1_val as i64, r2_val as i64) as u64,
        8,
        u64
    );

    test_amo!(
        run_amomaxd,
        |r1_val, r2_val| i64::max(r1_val as i64, r2_val as i64) as u64,
        8,
        u64
    );

    test_amo!(run_amominud, u64::min, 8, u64);

    test_amo!(run_amomaxud, u64::max, 8, u64);
}
