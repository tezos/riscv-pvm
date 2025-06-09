// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of RV_32_A extension for RISC-V
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
    /// `AMOADD.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// adding it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoaddw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, i32::wrapping_add)
    }

    /// `AMOXOR.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// XORing it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoxorw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, i32::bitxor)
    }

    /// `AMOAND.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// ANDing it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoandw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, i32::bitand)
    }

    /// `AMOOR.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the result of
    /// ORing it to val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amoorw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, i32::bitor)
    }

    /// `AMOMIN.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the minimum
    /// between it and val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amominw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, i32::min)
    }

    /// `AMOMAX.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the maximum
    /// between it and val(rs2) back to the address in rs1.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amomaxw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, i32::max)
    }

    /// `AMOMINU.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the minimum
    /// between it and val(rs2) back to the address in rs1, treating both as
    /// unsigned values.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amominuw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, |value_rs1, value_rs2| {
            (value_rs1 as u32).min(value_rs2 as u32) as i32
        })
    }

    /// `AMOMAXU.W` R-type instruction
    ///
    /// Loads in rd the value from the address in rs1 and stores the maximum
    /// between it and val(rs2) back to the address in rs1, treating both as
    /// unsigned values.
    /// The `aq` and `rl` bits specify additional memory constraints in
    /// multi-hart environments so they are currently ignored.
    pub fn run_amomaxuw(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        _rl: bool,
        _aq: bool,
    ) -> Result<(), Exception> {
        self.run_amo_w(rs1, rs2, rd, |value_rs1, value_rs2| {
            (value_rs1 as u32).max(value_rs2 as u32) as i32
        })
    }
}

#[cfg(test)]
pub(super) mod test {
    use std::ops::BitAnd;
    use std::ops::BitOr;
    use std::ops::BitXor;

    use proptest::prelude::*;

    use crate::backend_test;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;

    macro_rules! test_amo {
        ($instr: ident, $f: expr, $align: expr, $t: ident) => {
            backend_test!($instr, F, {
                use $crate::machine_state::memory::M4K;
                use $crate::state::NewState;

                let state = MachineCoreState::<M4K, _>::new(&mut F::manager());
                let state_cell = std::cell::RefCell::new(state);

                proptest!(|(
                    r1_addr in (0..1023_u64/$align).prop_map(|x| x * $align),
                    r1_val in any::<u64>(),
                    r2_val in any::<u64>(),
                )| {
                    let mut state = state_cell.borrow_mut();
                    state.reset();
                    state.main_memory.set_all_readable_writeable();

                    state.hart.xregisters.write(a0, r1_addr);
                    state.write_to_bus(0, a0, r1_val)?;
                    state.hart.xregisters.write(a1, r2_val);
                    state.$instr(a0, a1, a2, false, false)?;
                    let res: $t = state.read_from_address(r1_addr)?;

                    prop_assert_eq!(
                        state.hart.xregisters.read(a2) as $t, r1_val as $t);
                    // avoids redundant_closure_call warnings
                    let f = $f;
                    prop_assert_eq!(res, f(r1_val, r2_val))
                })
            });

        }
    }

    pub(crate) use test_amo;

    test_amo!(
        run_amoaddw,
        |r1_val, r2_val| (r1_val as i32).wrapping_add(r2_val as i32),
        4,
        i32
    );

    test_amo!(
        run_amoxorw,
        |r1_val, r2_val| (r1_val as i32).bitxor(r2_val as i32),
        4,
        i32
    );

    test_amo!(
        run_amoandw,
        |r1_val, r2_val| (r1_val as i32).bitand(r2_val as i32),
        4,
        i32
    );

    test_amo!(
        run_amoorw,
        |r1_val, r2_val| (r1_val as i32).bitor(r2_val as i32),
        4,
        i32
    );

    test_amo!(
        run_amominw,
        |r1_val, r2_val| (r1_val as i32).min(r2_val as i32),
        4,
        i32
    );

    test_amo!(
        run_amomaxw,
        |r1_val, r2_val| (r1_val as i32).max(r2_val as i32),
        4,
        i32
    );

    test_amo!(
        run_amominuw,
        |r1_val, r2_val| (r1_val as u32).min(r2_val as u32) as i32,
        4,
        i32
    );

    test_amo!(
        run_amomaxuw,
        |r1_val, r2_val| (r1_val as u32).max(r2_val as u32) as i32,
        4,
        i32
    );
}
