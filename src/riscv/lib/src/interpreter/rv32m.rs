// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of RV_32_M extension for RISC-V
//!
//! Chapter 7 - Unprivileged spec

use crate::machine_state::registers::XRegister;
use crate::machine_state::registers::XRegisters;
use crate::state_backend as backend;

impl<M> XRegisters<M>
where
    M: backend::ManagerReadWrite,
{
    #[cfg(test)]
    pub fn run_remu(&mut self, rs1: XRegister, rs2: XRegister, rd: XRegister) {
        let rval1 = self.read(rs1);
        let rval2 = self.read(rs2);

        let result = if rval2 == 0 { rval1 } else { rval1 % rval2 };

        self.write(rd, result);
    }

    /// `DIVU` R-type instruction
    ///
    /// Divide val(rs1) by val(rs2). The result is stored in `rd`. In case val(rs2)
    /// is zero, the result is `u64::MAX`. All values are _unsigned integers_.
    pub fn run_divu(&mut self, rs1: XRegister, rs2: XRegister, rd: XRegister) {
        let rval1 = self.read(rs1);
        let rval2 = self.read(rs2);

        let result = if rval2 == 0 { u64::MAX } else { rval1 / rval2 };

        self.write(rd, result);
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::any;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use crate::backend_test;
    use crate::machine_state::registers::XRegisters;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;
    use crate::machine_state::registers::a3;
    use crate::state::NewState;

    backend_test!(test_divu_remu_invariant, F, {
        proptest!(|(
            r1_val in any::<u64>(),
            r2_val in any::<u64>(),
        )| {
            let mut state = XRegisters::new(&mut F::manager());

            state.write(a0, r1_val);
            state.write(a1, r2_val);
            state.run_divu(a0, a1, a2);
            state.run_remu(a0, a1, a3);

            prop_assert_eq!(
                state.read(a0),
                state.read(a1)
                    .wrapping_mul(state.read(a2))
                    .wrapping_add(state.read(a3)));
        })
    });
}
