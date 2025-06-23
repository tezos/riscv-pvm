// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of RV_64_DC extension for RISC-V
//!
//! U:C-16

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory;
use crate::machine_state::registers::FRegister;
use crate::machine_state::registers::sp;
use crate::state_backend as backend;
use crate::traps::Exception;

impl<MC, M> MachineCoreState<MC, M>
where
    MC: memory::MemoryConfig,
    M: backend::ManagerReadWrite,
{
    /// `C.FLDSP` CI-type compressed instruction
    ///
    /// Loads a double-precision floating-point value from memory into
    /// floating-point register `rd`. It computes an effective address by
    /// adding the immediate to the stack pointer.
    /// The immediate is obtained by zero-extending and scaling by 8 the
    /// offset encoded in the instruction (see U:C-16.3).
    pub fn run_cfldsp(&mut self, imm: i64, rd_rs1: FRegister) -> Result<(), Exception> {
        debug_assert!(imm >= 0 && imm % 8 == 0);
        self.run_fld(imm, sp, rd_rs1)
    }

    /// `C.FSDSP` CSS-type compressed instruction
    ///
    /// Stores a double-precision floating-point value in floating-point
    /// register `rs2` to memory. It computes an effective address by adding
    /// the immediate to the stack pointer.
    /// The immediate is obtained by zero-extending and scaling by 8 the
    /// offset encoded in the instruction (see U:C-16.3).
    pub fn run_cfsdsp(&mut self, imm: i64, rs2: FRegister) -> Result<(), Exception> {
        debug_assert!(imm >= 0 && imm % 8 == 0);
        self.run_fsd(imm, sp, rs2)
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    use crate::backend_test;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::registers::fa2;
    use crate::machine_state::registers::fa3;
    use crate::machine_state::registers::sp;
    use crate::state::NewState;
    use crate::traps::Exception;

    const ZERO_OFFSET: i64 = 0;

    type MC = M4K;

    const OUT_OF_BOUNDS_OFFSET: i64 = MC::TOTAL_BYTES as i64;

    backend_test!(test_cfsdsp_cfldsp, F, {
        let state = MachineCoreState::<MC, _>::new(&mut F::manager());
        let state_cell = std::cell::RefCell::new(state);

        proptest!(|(
            base_addr in (0..504_u64),
            base_imm in (0..=64i64).prop_map(|x| x * 8), // multiples of 8 in the 0..512 range
            val in any::<f64>().prop_map(f64::to_bits),
        )| {
            let mut state = state_cell.borrow_mut();
            state.reset();
            state.main_memory.set_all_readable_writeable();

            let mut perform_test = |offset: i64| -> Result<(), Exception> {
                state.hart.fregisters.write(fa2, val.into());
                state.hart.xregisters.write(sp, base_addr);

                let imm = base_imm + offset;
                state.run_cfsdsp(imm, fa2)?;
                state.run_cfldsp(imm, fa3)?;

                assert_eq!(state.hart.fregisters.read(fa3), val.into());
                Ok(())
            };

            // Aligned and unaligned loads / stores
            prop_assert!(perform_test(ZERO_OFFSET).is_ok());

            // Out of bounds loads / stores
            prop_assert!(perform_test(OUT_OF_BOUNDS_OFFSET).is_err_and(|e|
                matches!(e, Exception::StoreAMOAccessFault(_))
            ));
        });
    });
}
