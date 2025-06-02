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
    use crate::interpreter::integer::run_x64_rem_unsigned;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;
    use crate::machine_state::registers::a3;
    use crate::machine_state::registers::nz;
    use crate::state::NewState;

    backend_test!(test_divu_remu_invariant, F, {
        proptest!(|(
            r1_val in any::<u64>(),
            r2_val in any::<u64>(),
        )| {
            let mut state = MachineCoreState::<M4K, _>::new(&mut F::manager());

            state.hart.xregisters.write(a0, r1_val);
            state.hart.xregisters.write(a1, r2_val);
            state.hart.xregisters.run_divu(a0, a1, a2);
            run_x64_rem_unsigned(&mut state, a0, a1, nz::a3);

            prop_assert_eq!(
                state.hart.xregisters.read(a0),
                state.hart.xregisters.read(a1)
                    .wrapping_mul(state.hart.xregisters.read(a2))
                    .wrapping_add(state.hart.xregisters.read(a3)));
        })
    });
}
