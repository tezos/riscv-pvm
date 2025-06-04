// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Core logic for atomic instructions

use std::mem;

use crate::instruction_context::ICB;
use crate::instruction_context::Predicate;
use crate::instruction_context::StoreLoadInt;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::instruction_context::comparable::Comparable;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory;
use crate::machine_state::registers::XRegister;
use crate::machine_state::reservation_set::RES_SET_BITMASK;
use crate::machine_state::reservation_set::UNSET_VALUE;
use crate::state_backend as backend;
use crate::traps::Exception;

pub const SC_SUCCESS: u64 = 0;
pub const SC_FAILURE: u64 = 1;

impl<MC, M> MachineCoreState<MC, M>
where
    MC: memory::MemoryConfig,
    M: backend::ManagerReadWrite,
{
    /// Generic implementation of any atomic memory operation, implementing
    /// read-modify-write operations for multi-processor synchronisation
    /// (Section 8.4)
    fn run_amo<T: backend::Elem>(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        f: fn(T, T) -> T,
        to: fn(u64) -> T,
        from: fn(T) -> u64,
    ) -> Result<(), Exception> {
        let address_rs1 = self.hart.xregisters.read(rs1);

        // "The A extension requires that the address held in rs1 be naturally
        // aligned to the size of the operand (i.e., eight-byte aligned for
        // 64-bit words and four-byte aligned for 32-bit words). If the address
        // is not naturally aligned, an address-misaligned exception or
        // an access-fault exception will be generated."
        if address_rs1 % mem::size_of::<T>() as u64 != 0 {
            return Err(Exception::StoreAMOAccessFault(address_rs1));
        }

        // Load the value from address in rs1
        let value_rs1: T = self.read_from_address(address_rs1)?;

        // Apply the binary operation to the loaded value and the value in rs2
        let value_rs2 = to(self.hart.xregisters.read(rs2));
        let value = f(value_rs1, value_rs2);

        // Write the value read fom the address in rs1 in rd
        self.hart.xregisters.write(rd, from(value_rs1));

        // Store the resulting value to the address in rs1
        self.write_to_address(address_rs1, value)
    }

    /// Generic implementation of an atomic memory operation which works on
    /// 32-bit values
    pub(super) fn run_amo_w(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        f: fn(i32, i32) -> i32,
    ) -> Result<(), Exception> {
        self.run_amo(rs1, rs2, rd, f, |x| x as i32, |x| x as u64)
    }

    /// Generic implementation of an atomic memory operation which works on
    /// 64-bit values
    pub(super) fn run_amo_d(
        &mut self,
        rs1: XRegister,
        rs2: XRegister,
        rd: XRegister,
        f: fn(u64, u64) -> u64,
    ) -> Result<(), Exception> {
        self.run_amo(rs1, rs2, rd, f, |x| x, |x| x)
    }
}

/// Generic implementation of any atomic memory operation which works on 64-bit values,
/// implementing read-modify-write operations for multi-processor synchronisation
/// (Section 8.4)
fn run_x64_atomic<I: ICB>(
    icb: &mut I,
    rs1: XRegister,
    rs2: XRegister,
    rd: XRegister,
    f: fn(I::XValue, I::XValue, &mut I) -> I::XValue,
) -> I::IResult<()> {
    let address_rs1 = icb.xregister_read(rs1);

    // Handle the case where the address is not aligned.
    let result = icb.atomic_access_fault_guard::<u64>(address_rs1, false);

    // Continue with the operation if the address is aligned.
    let val_rs1_result = I::and_then(result, |_| icb.main_memory_load::<u64>(address_rs1));

    // Continue with the operation if the load was successful.
    I::and_then(val_rs1_result, |val_rs1| {
        // Apply the binary operation to the loaded value and the value in rs2
        let val_rs2 = icb.xregister_read(rs2);
        let res = f(val_rs1, val_rs2, icb);

        // Write the value read fom the address in rs1 in rd
        icb.xregister_write(rd, val_rs1);

        // Store the resulting value to the address in rs1
        icb.main_memory_store::<u64>(address_rs1, res)
    })
}

/// Loads in `rd` the value from the address in `rs1` and stores the result of
/// adding it to `val(rs2)` back to the address in `rs1`.
///
/// The `aq` and `rl` bits specify additional memory constraints
/// in multi-hart environments so they are currently ignored.
pub fn run_x64_atomic_add<I: ICB>(
    icb: &mut I,
    rs1: XRegister,
    rs2: XRegister,
    rd: XRegister,
    _aq: bool,
    _rl: bool,
) -> I::IResult<()> {
    run_x64_atomic(icb, rs1, rs2, rd, |x, y, icb| x.add(y, icb))
}

/// Loads a word or a double from the address in `rs1`, places the
/// sign-extended value in `rd`, and registers a reservation set for
/// that address.
/// See also [crate::machine_state::reservation_set].
pub(super) fn run_atomic_load<I: ICB, V: StoreLoadInt>(
    icb: &mut I,
    rs1: XRegister,
    rd: XRegister,
) -> I::IResult<()> {
    let address_rs1 = icb.xregister_read(rs1);

    // "The A extension requires that the address held in rs1 be naturally
    // aligned to the size of the operand (i.e., eight-byte aligned for
    // 64-bit words and four-byte aligned for 32-bit words). If the address
    // is not naturally aligned, an address-misaligned exception or
    // an access-fault exception will be generated."
    let result = icb.atomic_access_fault_guard::<V>(address_rs1, false);

    // Continue with the operation if the address is aligned and load the value from address in rs1.
    let val_rs1_result = I::and_then(result, |_| icb.main_memory_load::<V>(address_rs1));

    // If the load was successful, register a reservation set for the address in rs1
    // and write the value at that address to rd.
    I::and_then(val_rs1_result, |val_rs1| {
        let aligned_address_rs1 = reservation_set_align_address::<V, I>(icb, address_rs1);

        icb.reservation_set_write(aligned_address_rs1);
        icb.xregister_write(rd, val_rs1);

        icb.ok(())
    })
}

/// Conditionally writes a word in `rs2` to the address in `rs1`.
/// This succeeds if the reservation is still valid and
/// the reservation set contains the bytes being written.
/// In case of success, write 0 in `rd`, otherwise write 1.
/// See also [crate::machine_state::reservation_set].
/// The `aq` and `rl` bits specify additional memory constraints in
/// multi-hart environments so they are currently ignored.
pub(super) fn run_atomic_store<I: ICB, V: StoreLoadInt>(
    icb: &mut I,
    rs1: XRegister,
    rs2: XRegister,
    rd: XRegister,
) -> I::IResult<()> {
    let address_rs1 = icb.xregister_read(rs1);

    // "The A extension requires that the address held in rs1 be naturally
    // aligned to the size of the operand (i.e., eight-byte aligned for
    // 64-bit words and four-byte aligned for 32-bit words). If the address
    // is not naturally aligned, an address-misaligned exception or
    // an access-fault exception will be generated."
    // icb.reset_reservation_set();
    let result = icb.atomic_access_fault_guard::<V>(address_rs1, true);

    I::and_then(result, |_| {
        let cond = test_and_unset_reservation_set::<V, I>(icb, address_rs1);

        icb.branch_merge::<Result<(), Exception>, _, _>(
            cond,
            |icb| {
                // If the address in rs1 belongs to a valid reservation, write
                // the value in rs2 to this address and return success.
                let value_rs2 = icb.xregister_read(rs2);
                let sc_success_imm = icb.xvalue_of_imm(SC_SUCCESS as i64);

                icb.xregister_write(rd, sc_success_imm);
                icb.main_memory_store::<V>(address_rs1, value_rs2)
            },
            |icb| {
                // If the address in rs1 does not belong to a valid reservation or
                // there is no valid reservation set on the hart, do not write to
                // memory and return failure.
                let sc_failure_imm = icb.xvalue_of_imm(SC_FAILURE as i64);
                icb.xregister_write(rd, sc_failure_imm);

                icb.ok(())
            },
        )
    })
}

/// Loads a word from the address in `rs1`, places the
/// sign-extended value in `rd`, and registers a reservation set for
/// that address.
///
/// The value in `rs2` is always 0 so is ignored.
/// The `aq` and `rl` bits specify additional memory constraints
/// in multi-hart environments so they are currently ignored.
pub fn run_x32_atomic_load<I: ICB>(
    icb: &mut I,
    rs1: XRegister,
    _rs2: XRegister,
    rd: XRegister,
    _rl: bool,
    _aq: bool,
) -> I::IResult<()> {
    run_atomic_load::<I, i32>(icb, rs1, rd)
}

/// Loads a doubleword from the address in `rs1`, places the value in `rd`,
/// and registers a reservation set for that address.
///
/// The value in `rs2` is always 0 so is ignored.
/// The `aq` and `rl` bits specify additional memory constraints
/// in multi-hart environments so they are currently ignored.
pub fn run_x64_atomic_load<I: ICB>(
    icb: &mut I,
    rs1: XRegister,
    _rs2: XRegister,
    rd: XRegister,
    _rl: bool,
    _aq: bool,
) -> I::IResult<()> {
    run_atomic_load::<I, i64>(icb, rs1, rd)
}

/// Conditionally writes a word in `rs2` to the address in `rs1`.
/// This succeeds if the reservation is still valid and
/// the reservation set contains the bytes being written.
/// In case of success, write 0 in `rd`, otherwise write 1.
/// See also [crate::machine_state::reservation_set].
///
/// The `aq` and `rl` bits specify additional memory constraints
/// in multi-hart environments so they are currently ignored.
pub fn run_x32_atomic_store<I: ICB>(
    icb: &mut I,
    rs1: XRegister,
    rs2: XRegister,
    rd: XRegister,
    _rl: bool,
    _aq: bool,
) -> I::IResult<()> {
    run_atomic_store::<I, i32>(icb, rs1, rs2, rd)
}

/// Conditionally writes a doubleword in `rs2` to the address in `rs1`.
/// This succeeds if the reservation is still valid and
/// the reservation set contains the bytes being written.
/// In case of success, write 0 in `rd`, otherwise write 1.
/// See also [crate::machine_state::reservation_set].
///
/// The `aq` and `rl` bits specify additional memory constraints
/// in multi-hart environments so they are currently ignored.
pub fn run_x64_atomic_store<I: ICB>(
    icb: &mut I,
    rs1: XRegister,
    rs2: XRegister,
    rd: XRegister,
    _rl: bool,
    _aq: bool,
) -> I::IResult<()> {
    run_atomic_store::<I, i64>(icb, rs1, rs2, rd)
}

// Reservation Set Helper Functionss

/// Reset the reservation set to an unset state.
pub(crate) fn reset_reservation_set<I: ICB>(icb: &mut I) {
    let unset = icb.xvalue_of_imm(UNSET_VALUE as i64);
    icb.reservation_set_write(unset);
}

/// Align an address to the size of the reservation set.
fn reservation_set_align_address<V: StoreLoadInt, I: ICB>(
    icb: &mut I,
    address: I::XValue,
) -> I::XValue {
    let bitmask = icb.xvalue_of_imm(RES_SET_BITMASK as i64);
    address.and(bitmask, icb)
}

/// Check whether the given 'address' is within the reservation set and reset the reservation set.
///
/// Returns `true` if the address is within the reservation set, `false` otherwise.
fn test_and_unset_reservation_set<V: StoreLoadInt, I: ICB>(
    icb: &mut I,
    address: I::XValue,
) -> I::Bool {
    let start_addr = icb.reservation_set_read();

    // Regardless of success or failure, executing an SC.x instruction
    // invalidates any reservation held by this hart.
    reset_reservation_set(icb);

    let unset_value = icb.xvalue_of_imm(UNSET_VALUE as i64);
    let is_set = start_addr.compare(unset_value, Predicate::NotEqual, icb);

    let aligned_address = reservation_set_align_address::<V, I>(icb, address);
    let in_reservation_set = aligned_address.compare(start_addr, Predicate::Equal, icb);

    icb.bool_and(is_set, in_reservation_set)
}

#[cfg(test)]
pub(crate) mod test {
    use proptest::prelude::*;

    use super::*;
    use crate::backend_test;
    use crate::interpreter::integer::run_addi;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;
    use crate::machine_state::registers::a3;
    use crate::machine_state::registers::a7;
    use crate::state::NewState;

    macro_rules! test_atomic_loadstore {
        ($name:ident, $lr: expr, $sc: expr, $align: expr, $t: ident) => {
            backend_test!($name, F, {
                use $crate::machine_state::registers::nz;
                use $crate::machine_state::memory::M4K;
                use $crate::state::NewState;

                let state = MachineCoreState::<M4K, _>::new(&mut F::manager());
                let state_cell = std::cell::RefCell::new(state);

                proptest!(|(
                    r1_addr in (0..1023_u64/$align).prop_map(|x| x * $align),
                    r1_val in any::<u64>(),
                    imm in any::<i64>(),
                )| {
                    let mut state = state_cell.borrow_mut();
                    state.reset();
                    state.main_memory.set_all_readable_writeable();

                    state.hart.xregisters.write(a0, r1_addr);
                    state.write_to_bus(0, a0, r1_val)?;

                    // SC.x fails when no reservation is set on the hart
                    $sc(&mut *state, a0, a1, a2, false, false)?;
                    let res = state.hart.xregisters.read(a2);
                    prop_assert_eq!(res, SC_FAILURE);

                    // Correct sequence of LR.x / SC.y instructions
                    // SC.x succeeds and stores the expected value
                    $lr(&mut *state, a0, a1, a2, false, false)?;
                    run_addi(&mut *state, imm, nz::a2, nz::a1);
                    $sc(&mut *state, a0, a1, a2, false, false)?;
                    let res = state.hart.xregisters.read(a2);
                    let val: $t = state.read_from_address(r1_addr)?;
                    prop_assert_eq!(res, SC_SUCCESS);
                    prop_assert_eq!(val, r1_val.wrapping_add(imm as u64) as $t);

                    // SC.x fails when a previous SC.x has been executed
                    $sc(&mut *state, a0, a1, a2, false, false)?;
                    let res = state.hart.xregisters.read(a2);
                    prop_assert_eq!(res, SC_FAILURE);
                })
            });
        }
    }
    pub(crate) use test_atomic_loadstore;

    macro_rules! test_atomic {
        ($(#[$m:meta])* $name: ident, $instr: path, $f: expr, $align: expr, $t: ty) => {
            backend_test!($name, F, {
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
                    match $instr(&mut *state, a0, a1, a2, false, false) {
                        Ok(_) => {}
                        Err(e) => panic!("Error: {:?}", e),
                    }
                    let res: $t = state.read_from_address(r1_addr)?;

                    prop_assert_eq!(
                        state.hart.xregisters.read(a2) as $t, r1_val as $t);

                    let f = $f;
                    let expected = f(r1_val as $t, r2_val as $t);
                    prop_assert_eq!(res, expected);
                })
            });

        }
    }

    test_atomic_loadstore!(
        test_x32_atomic_loadstore,
        run_x32_atomic_load,
        run_x32_atomic_store,
        4,
        u32
    );

    test_atomic!(
        test_run_x64_atomic_add,
        super::run_x64_atomic_add,
        u64::wrapping_add,
        8,
        u64
    );

    test_atomic_loadstore!(
        test_x64_atomic_loadstore,
        run_x64_atomic_load,
        run_x64_atomic_store,
        8,
        u64
    );

    test_atomic_loadstore!(
        test_atomic_loadstore_x64_x32,
        run_x64_atomic_load,
        run_x32_atomic_store,
        8,
        u32
    );
    test_atomic_loadstore!(
        test_atomic_loadstore_x32_x64,
        run_x32_atomic_load,
        run_x64_atomic_store,
        8,
        u32
    );

    backend_test!(test_alignment, F, {
        let mut state = MachineCoreState::<M4K, _>::new(&mut F::manager());
        state.main_memory.set_all_readable_writeable();
        state.hart.xregisters.write(a0, 80); // LR.D starting address.
        state.hart.xregisters.write(a1, 84); // SC.W starting address.
        state.hart.xregisters.write(a2, 200); // Value to store.

        run_x64_atomic_load(&mut state, a0, a7, a3, false, false).unwrap();
        run_x32_atomic_store(&mut state, a1, a2, a3, false, false).unwrap();

        // Check that the value was stored correctly.
        let stored_value: u32 = state.read_from_address(84).unwrap();
        assert_eq!(stored_value, 200);

        // check rd stores the success value 0.
        let rd_value = state.hart.xregisters.read(a3);
        assert_eq!(rd_value, SC_SUCCESS);
    });
}
