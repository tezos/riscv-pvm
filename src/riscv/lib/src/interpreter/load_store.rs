// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of load and store instructions for RISC-V over the ICB.

use crate::instruction_context::ICB;
use crate::instruction_context::StoreLoadInt;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::machine_state::registers::NonZeroXRegister;
use crate::machine_state::registers::XRegister;

/// Loads the immediate `imm` into register `rd_rs1`.
///
/// Relevant RISC-V opcodes:
/// - C.LI
/// - C.LUI
/// - ADD
/// - ADDI
/// - ANDI
/// - ORI
/// - XORI
/// - SLLI
/// - SRLI
/// - SRAI
/// - AND
/// - C.AND
/// - OR
/// - XOR
/// - SLL
/// - SRL
/// - SRA
/// - SUB
pub fn run_li(icb: &mut impl ICB, imm: i64, rd_rs1: NonZeroXRegister) {
    let imm = icb.xvalue_of_imm(imm);
    icb.xregister_write_nz(rd_rs1, imm)
}

/// Stores a value to the address starting at `val(rs1) + imm`.
///
/// The value is taken from `rs2`, but only the lowest `width` bytes
/// are written to memory.
#[inline(always)]
pub fn run_store<V: StoreLoadInt, I: ICB>(
    icb: &mut I,
    imm: i64,
    rs1: XRegister,
    rs2: XRegister,
) -> I::IResult<()> {
    let base_address = icb.xregister_read(rs1);
    let offset = icb.xvalue_of_imm(imm);

    let address = base_address.add(offset, icb);

    let value = icb.xregister_read(rs2);

    icb.main_memory_store::<V>(address, value)
}

/// Loads a value from the address starting at `val(rs1) + imm`.
///
/// Only `width` bytes are read from memory and then extended to the full register size
/// using the appropriate signed/unsigned extension.
///
/// The result is written to `rd`.
#[inline(always)]
pub fn run_load<V: StoreLoadInt, I: ICB>(
    icb: &mut I,
    imm: i64,
    rs1: XRegister,
    rd: XRegister,
) -> I::IResult<()> {
    let base_address = icb.xregister_read(rs1);
    let offset = icb.xvalue_of_imm(imm);

    let address = base_address.add(offset, icb);

    let value = icb.main_memory_load::<V>(address);
    I::and_then(value, |value| {
        icb.xregister_write(rd, value);
        icb.ok(())
    })
}

#[cfg(test)]
mod test {
    use proptest::arbitrary::any;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::*;
    use crate::backend_test;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;
    use crate::machine_state::registers::a3;
    use crate::machine_state::registers::a4;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::t0;
    use crate::machine_state::registers::t1;
    use crate::machine_state::registers::t2;
    use crate::machine_state::registers::t3;
    use crate::machine_state::registers::t4;
    use crate::state::NewState;
    use crate::traps::Exception;

    backend_test!(test_run_li, F, {
        let imm_rdrs1_res = [
            (0_i64, nz::t3, 0_u64),
            (0xFFF0_0420, nz::t2, 0xFFF0_0420),
            (-1, nz::t4, 0xFFFF_FFFF_FFFF_FFFF),
        ];

        for (imm, rd_rs1, res) in imm_rdrs1_res {
            let mut state = MachineCoreState::<M4K, F>::new();
            super::run_li(&mut state, imm, rd_rs1);
            assert_eq!(state.hart.xregisters.read_nz(rd_rs1), res);
        }
    });

    backend_test!(test_lui, F, {
        proptest!(|(imm in any::<i64>())| {
            let mut state = MachineCoreState::<M4K, F>::new();
            state.hart.xregisters.write(a2, 0);
            state.hart.xregisters.write(a4, 0);

            // U-type immediate sets imm[31:20]
            let imm = imm & 0xFFFF_F000;
            super::run_li(&mut state, imm, nz::a3);
            // read value is the expected one
            prop_assert_eq!(state.hart.xregisters.read(a3), imm as u64);
            // it doesn't modify other registers
            prop_assert_eq!(state.hart.xregisters.read(a2), 0);
            prop_assert_eq!(state.hart.xregisters.read(a4), 0);
        });
    });

    backend_test!(test_load_store, F, {
        let state = MachineCoreState::<M4K, F>::new();
        let state_cell = std::cell::RefCell::new(state);

        proptest!(|(
            v_1 in any::<u8>(),
            v_2 in any::<u16>(),
            v_3 in any::<u32>(),
            v_4 in any::<u64>(),
        )|
        {
            let mut state = state_cell.borrow_mut();
            state.reset();
            state.main_memory.set_all_readable_writeable();

            let mut perform_test = |offset: u64, signed: bool| -> Result<(), Exception> {
                // Save test values v_i in registers ai
                state.hart.xregisters.write(a4, v_4);
                state.hart.xregisters.write(a3, v_3 as u64);
                state.hart.xregisters.write(a2, v_2 as u64);
                state.hart.xregisters.write(a1, v_1 as u64);

                // t0 will hold the "global" offset of all loads / stores we are going to make
                state.hart.xregisters.write(t0, offset);

                // Perform the stores
                run_store::<u64, _>(&mut *state, 0, t0, a4)?;
                run_store::<u32, _>(&mut *state, 8, t0, a3)?;
                run_store::<u16, _>(&mut *state, 12, t0, a2)?;
                run_store::<u8, _>(&mut *state, 14, t0, a1)?;

                run_load::<i64, _>(&mut *state, 0, t0, t4)?;

                if signed {
                    run_load::<i32, _>(&mut *state, 8, t0, t3)?;
                    run_load::<i16, _>(&mut *state, 12, t0, t2)?;
                    run_load::<i8, _>(&mut *state, 14, t0, t1)?;
                } else {
                    run_load::<u32, _>(&mut *state, 8, t0, t3)?;
                    run_load::<u16, _>(&mut *state, 12, t0, t2)?;
                    run_load::<u8, _>(&mut *state, 14, t0, t1)?;
                }

                assert_eq!(state.hart.xregisters.read(t4), v_4);

                // Converting the expected result we are also checking the sign-extension behaviour
                if signed {
                    assert_eq!(state.hart.xregisters.read(t3), v_3 as i32 as u64);
                    assert_eq!(state.hart.xregisters.read(t2), v_2 as i16 as u64);
                    assert_eq!(state.hart.xregisters.read(t1), v_1 as i8 as u64);
                } else {
                    assert_eq!(state.hart.xregisters.read(t3), v_3 as u64);
                    assert_eq!(state.hart.xregisters.read(t2), v_2 as u64);
                    assert_eq!(state.hart.xregisters.read(t1), v_1 as u64);
                };

                Ok(())
            };

            let invalid_offset = 0u64.wrapping_sub(1024);
            let aligned_offset = 512;
            let misaligned_offset = 513;

            // Out of bounds loads / stores
            prop_assert!(perform_test(invalid_offset, true).is_err_and(|e|
                matches!(e, Exception::StoreAMOAccessFault(_))
            ));
            // Aligned loads / stores
            prop_assert!(perform_test(aligned_offset, true).is_ok());
            // Unaligned loads / stores
            prop_assert!(perform_test(misaligned_offset, true).is_ok());

            // Out of bounds loads / stores
            prop_assert!(perform_test(invalid_offset, false).is_err_and(|e|
                matches!(e, Exception::StoreAMOAccessFault(_))
            ));
            // Aligned loads / stores
            prop_assert!(perform_test(aligned_offset, false).is_ok());
            // Unaligned loads / stores
            prop_assert!(perform_test(misaligned_offset, false).is_ok());
        });
    });
}
