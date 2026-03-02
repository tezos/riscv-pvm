//! Implementation of Zicsr extension for RISC-V
//!
//! Chapter 9 - Unprivileged spec

use crate::exceptions::Exception;
use crate::instruction_context::ICB;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::machine_state::csregisters;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::registers;

/// Checks that `csr` is write-able. If it is, executes the `body` closure. Otherwise, raises an
/// [`Exception::IllegalInstruction`].
fn if_writable<I: ICB>(icb: &mut I, csr: CSRegister, body: impl FnOnce(&mut I)) -> I::IResult<()> {
    if csr.is_read_only() {
        return icb.raise_exception(Exception::IllegalInstruction);
    }

    body(icb);

    icb.ok(())
}

/// Replace the value in `csr` with `new_value` and write the previous value to `rd`.
/// When `rd = x0`, no read side effects are triggered.
fn csr_replace<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    new_value: I::XValue,
    rd: registers::XRegister,
) -> I::IResult<()> {
    if_writable(icb, csr, |icb| {
        let old = icb.csr_read(csr);
        icb.csr_write(csr, new_value);
        registers::write_xregister(icb, rd, old);
    })
}

/// Set the specified bits in `csr` and write the previous value to `rd`.
fn csr_set_bits<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    bits: I::XValue,
    rd: registers::XRegister,
) {
    let old = icb.csr_read(csr);
    let new = old.or(bits, icb);
    icb.csr_write(csr, new);
    registers::write_xregister(icb, rd, old);
}

/// Clear the specified bits in `csr` and write the previous value to `rd`.
fn csr_clear_bits<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    bits: I::XValue,
    rd: registers::XRegister,
) {
    let old = icb.csr_read(csr);
    let mask = bits.not(icb);
    let new = old.and(mask, icb);
    icb.csr_write(csr, new);
    registers::write_xregister(icb, rd, old);
}

/// Execute a `CSRRW` instruction.
pub fn run_csrrw<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    rs1: registers::XRegister,
    rd: registers::XRegister,
) -> I::IResult<()> {
    let value = registers::read_xregister(icb, rs1);
    csr_replace(icb, csr, value, rd)
}

/// Execute a `CSRRWI` instruction.
pub fn run_csrrwi<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    imm: registers::XValue,
    rd: registers::XRegister,
) -> I::IResult<()> {
    let imm = icb.xvalue_of_imm((imm & 0b11111) as i64);
    csr_replace(icb, csr, imm, rd)
}

/// Execute the `CSRRS` instruction.
pub fn run_csrrs<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    rs1: registers::XRegister,
    rd: registers::XRegister,
) -> I::IResult<()> {
    // When `rs1 = x0`, we don't want to trigger any CSR write effects.
    if rs1.is_zero() {
        let old = icb.csr_read(csr);
        registers::write_xregister(icb, rd, old);
        return icb.ok(());
    }

    if_writable(icb, csr, |icb| {
        let bits = registers::read_xregister(icb, rs1);
        csr_set_bits(icb, csr, bits, rd);
    })
}

/// Execute the `CSRRSI` instruction.
pub fn run_csrrsi<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    imm: registers::XValue,
    rd: registers::XRegister,
) -> I::IResult<()> {
    let imm = imm & 0b11111;

    // When `imm = 0`, we don't want to trigger any CSR write effects.
    if imm == 0 {
        let old = icb.csr_read(csr);
        registers::write_xregister(icb, rd, old);
        return icb.ok(());
    }

    if_writable(icb, csr, |icb| {
        let bits = icb.xvalue_of_imm(imm as i64);
        csr_set_bits(icb, csr, bits, rd);
    })
}

/// Execute the `CSRRC` instruction.
pub fn run_csrrc<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    rs1: registers::XRegister,
    rd: registers::XRegister,
) -> I::IResult<()> {
    // When `rs1 = x0`, we don't want to trigger any CSR write effects.
    if rs1.is_zero() {
        let old = icb.csr_read(csr);
        registers::write_xregister(icb, rd, old);
        return icb.ok(());
    }

    if_writable(icb, csr, |icb| {
        let bits = registers::read_xregister(icb, rs1);
        csr_clear_bits(icb, csr, bits, rd);
    })
}

/// Execute the `CSRRCI` instruction.
pub fn run_csrrci<I: ICB>(
    icb: &mut I,
    csr: csregisters::CSRegister,
    imm: registers::XValue,
    rd: registers::XRegister,
) -> I::IResult<()> {
    let imm = imm & 0b11111;

    // When `imm = 0`, we don't want to trigger any CSR write effects.
    if imm == 0 {
        let old = icb.csr_read(csr);
        registers::write_xregister(icb, rd, old);
        return icb.ok(());
    }

    if_writable(icb, csr, |icb| {
        let bits = icb.xvalue_of_imm(imm as i64);
        csr_clear_bits(icb, csr, bits, rd);
    })
}
