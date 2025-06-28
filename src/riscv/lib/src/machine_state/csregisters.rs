// SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Shl;
use std::ops::Shr;

use num_enum::TryFromPrimitive;

use crate::bits::Bits64;
use crate::state::NewState;
use crate::state_backend as backend;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::struct_layout;
use crate::traps::Exception;

/// CSR index
#[expect(non_camel_case_types, reason = "Consistent with RISC-V spec")]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    strum::EnumIter,
    TryFromPrimitive,
    strum::Display,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(usize)]
pub enum CSRegister {
    // Unprivileged Floating-Point CSRs
    fflags = 0x001,
    frm = 0x002,
    fcsr = 0x003,

    // Unprivileged Counter/Timers
    cycle = 0xC00,
    time = 0xC01,
    instret = 0xC02,
    hpmcounter3 = 0xC03,
    hpmcounter4 = 0xC04,
    hpmcounter5 = 0xC05,
    hpmcounter6 = 0xC06,
    hpmcounter7 = 0xC07,
    hpmcounter8 = 0xC08,
    hpmcounter9 = 0xC09,
    hpmcounter10 = 0xC0A,
    hpmcounter11 = 0xC0B,
    hpmcounter12 = 0xC0C,
    hpmcounter13 = 0xC0D,
    hpmcounter14 = 0xC0E,
    hpmcounter15 = 0xC0F,
    hpmcounter16 = 0xC10,
    hpmcounter17 = 0xC11,
    hpmcounter18 = 0xC12,
    hpmcounter19 = 0xC13,
    hpmcounter20 = 0xC14,
    hpmcounter21 = 0xC15,
    hpmcounter22 = 0xC16,
    hpmcounter23 = 0xC17,
    hpmcounter24 = 0xC18,
    hpmcounter25 = 0xC19,
    hpmcounter26 = 0xC1A,
    hpmcounter27 = 0xC1B,
    hpmcounter28 = 0xC1C,
    hpmcounter29 = 0xC1D,
    hpmcounter30 = 0xC1E,
    hpmcounter31 = 0xC1F,
}

impl CSRegister {
    /// Determines if the register is read-only
    #[inline(always)]
    pub fn is_read_only(self) -> bool {
        // Rules & Table of read-write / read-only ranges are in section 2.1 & table 2.1
        (self as usize >> 10) & 0b11 == 0b11
    }

    /// Attempt to parse the 32-bit integer as a register identifier.
    pub const fn try_parse(r: u32) -> Option<Self> {
        use CSRegister::*;

        match r {
            // Unprivileged Floating-Point CSRs
            0x001 => Some(fflags),
            0x002 => Some(frm),
            0x003 => Some(fcsr),

            // Unprivileged Counter/Timers
            0xC00 => Some(cycle),
            0xC01 => Some(time),
            0xC02 => Some(instret),
            0xC03 => Some(hpmcounter3),
            0xC04 => Some(hpmcounter4),
            0xC05 => Some(hpmcounter5),
            0xC06 => Some(hpmcounter6),
            0xC07 => Some(hpmcounter7),
            0xC08 => Some(hpmcounter8),
            0xC09 => Some(hpmcounter9),
            0xC0A => Some(hpmcounter10),
            0xC0B => Some(hpmcounter11),
            0xC0C => Some(hpmcounter12),
            0xC0D => Some(hpmcounter13),
            0xC0E => Some(hpmcounter14),
            0xC0F => Some(hpmcounter15),
            0xC10 => Some(hpmcounter16),
            0xC11 => Some(hpmcounter17),
            0xC12 => Some(hpmcounter18),
            0xC13 => Some(hpmcounter19),
            0xC14 => Some(hpmcounter20),
            0xC15 => Some(hpmcounter21),
            0xC16 => Some(hpmcounter22),
            0xC17 => Some(hpmcounter23),
            0xC18 => Some(hpmcounter24),
            0xC19 => Some(hpmcounter25),
            0xC1A => Some(hpmcounter26),
            0xC1B => Some(hpmcounter27),
            0xC1C => Some(hpmcounter28),
            0xC1D => Some(hpmcounter29),
            0xC1E => Some(hpmcounter30),
            0xC1F => Some(hpmcounter31),

            _ => None,
        }
    }
}

/// Representation of a value in a CSR
pub type CSRRepr = u64;

/// Return type of read/write operations
pub type Result<R> = core::result::Result<R, Exception>;

/// Checks that `reg` is write-able.
///
/// Throws [`Exception::IllegalInstruction`] in case of wrong access rights.
/// Section 2.1 - privileged spec
#[inline]
pub fn check_write(reg: CSRegister) -> Result<()> {
    if reg.is_read_only() {
        return Err(Exception::IllegalInstruction);
    }

    Ok(())
}

/// Bit mask for the rounding mode
const FRM_MASK: CSRRepr = 0b111;

/// Rounding mode is bits 5 to 7 in the `fcsr` register
const FRM_SHIFT: usize = 5;

/// Bit mask for the flags
const FFLAGS_MASK: CSRRepr = 0b11111;

// Layout for [`CSRegisters`]
struct_layout! {
    pub struct CSRegistersLayout {
        fflags: Atom<u8>,
        frm: Atom<u8>,
    }
}

/// Cntrol and State Registers (CSRs)
pub struct CSRegisters<M: backend::ManagerBase> {
    fflags: Cell<u8, M>,
    frm: Cell<u8, M>,
}

impl<M: backend::ManagerBase> CSRegisters<M> {
    /// Write to a CSR.
    pub fn write<V: Bits64>(&mut self, reg: CSRegister, value: V)
    where
        M: backend::ManagerWrite,
    {
        match reg {
            CSRegister::fflags => {
                let fflags = value.to_bits().bitand(FFLAGS_MASK) as u8;
                self.fflags.write(fflags);
            }

            CSRegister::frm => {
                let frm = value.to_bits().bitand(FRM_MASK) as u8;
                self.frm.write(frm);
            }

            CSRegister::fcsr => {
                let fflags = value.to_bits().bitand(FFLAGS_MASK) as u8;
                self.fflags.write(fflags);

                let frm = value.to_bits().shr(FRM_SHIFT).bitand(FRM_MASK) as u8;
                self.frm.write(frm);
            }

            CSRegister::cycle
            | CSRegister::time
            | CSRegister::instret
            | CSRegister::hpmcounter3
            | CSRegister::hpmcounter4
            | CSRegister::hpmcounter5
            | CSRegister::hpmcounter6
            | CSRegister::hpmcounter7
            | CSRegister::hpmcounter8
            | CSRegister::hpmcounter9
            | CSRegister::hpmcounter10
            | CSRegister::hpmcounter11
            | CSRegister::hpmcounter12
            | CSRegister::hpmcounter13
            | CSRegister::hpmcounter14
            | CSRegister::hpmcounter15
            | CSRegister::hpmcounter16
            | CSRegister::hpmcounter17
            | CSRegister::hpmcounter18
            | CSRegister::hpmcounter19
            | CSRegister::hpmcounter20
            | CSRegister::hpmcounter21
            | CSRegister::hpmcounter22
            | CSRegister::hpmcounter23
            | CSRegister::hpmcounter24
            | CSRegister::hpmcounter25
            | CSRegister::hpmcounter26
            | CSRegister::hpmcounter27
            | CSRegister::hpmcounter28
            | CSRegister::hpmcounter29
            | CSRegister::hpmcounter30
            | CSRegister::hpmcounter31 => {
                // These register are read-only.
            }
        }
    }

    /// Read from a CSR.
    pub fn read<V: Bits64>(&self, reg: CSRegister) -> V
    where
        M: backend::ManagerRead,
    {
        match reg {
            CSRegister::fflags => {
                let fflags = self.fflags.read() as u64;
                V::from_bits(fflags)
            }

            CSRegister::frm => {
                let frm = self.frm.read() as u64;
                V::from_bits(frm)
            }

            CSRegister::fcsr => {
                let fflags = self.fflags.read() as u64;
                let frm = self.frm.read() as u64;
                let fcsr = frm.shl(FRM_SHIFT).bitor(fflags);
                V::from_bits(fcsr)
            }

            CSRegister::cycle | CSRegister::time | CSRegister::instret => {
                // We don't count those at the moment.
                V::from_bits(0)
            }

            CSRegister::hpmcounter3
            | CSRegister::hpmcounter4
            | CSRegister::hpmcounter5
            | CSRegister::hpmcounter6
            | CSRegister::hpmcounter7
            | CSRegister::hpmcounter8
            | CSRegister::hpmcounter9
            | CSRegister::hpmcounter10
            | CSRegister::hpmcounter11
            | CSRegister::hpmcounter12
            | CSRegister::hpmcounter13
            | CSRegister::hpmcounter14
            | CSRegister::hpmcounter15
            | CSRegister::hpmcounter16
            | CSRegister::hpmcounter17
            | CSRegister::hpmcounter18
            | CSRegister::hpmcounter19
            | CSRegister::hpmcounter20
            | CSRegister::hpmcounter21
            | CSRegister::hpmcounter22
            | CSRegister::hpmcounter23
            | CSRegister::hpmcounter24
            | CSRegister::hpmcounter25
            | CSRegister::hpmcounter26
            | CSRegister::hpmcounter27
            | CSRegister::hpmcounter28
            | CSRegister::hpmcounter29
            | CSRegister::hpmcounter30
            | CSRegister::hpmcounter31 => {
                // We assume that the M-level counters are all not enabled. There is no M-level, so
                // we get to decide.
                V::from_bits(0)
            }
        }
    }

    /// Replace the CSR value, returning the previous value.
    #[inline]
    pub fn replace<V: Bits64>(&mut self, reg: CSRegister, value: V) -> V
    where
        M: backend::ManagerReadWrite,
    {
        let old = self.read::<V>(reg);
        self.write(reg, value);
        old
    }

    /// Set bits in the CSR.
    #[inline]
    pub fn set_bits(&mut self, reg: CSRegister, bits: CSRRepr) -> CSRRepr
    where
        M: backend::ManagerReadWrite,
    {
        let old_value = self.read(reg);
        let new_value = old_value | bits;
        self.write(reg, new_value);
        old_value
    }

    /// Clear bits in the CSR.
    #[inline]
    pub fn clear_bits(&mut self, reg: CSRegister, bits: CSRRepr) -> CSRRepr
    where
        M: backend::ManagerReadWrite,
    {
        let old_value = self.read(reg);
        let new_value = old_value & !bits;
        self.write(reg, new_value);
        old_value
    }

    /// Bind the CSR state to the allocated space.
    pub fn bind(space: backend::AllocatedOf<CSRegistersLayout, M>) -> Self {
        Self {
            fflags: space.fflags,
            frm: space.frm,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: backend::FnManager<backend::Ref<'a, M>>>(
        &'a self,
    ) -> backend::AllocatedOf<CSRegistersLayout, F::Output> {
        CSRegistersLayoutF {
            fflags: self.fflags.struct_ref::<F>(),
            frm: self.frm.struct_ref::<F>(),
        }
    }

    /// Reset the control and state registers.
    pub fn reset(&mut self)
    where
        M: backend::ManagerWrite,
    {
        // Resets accrued floating-point exceptions
        self.fflags.write(0b00000);

        // 000 = RNE aka "round to nearest, ties to even"
        self.frm.write(0b000);
    }
}

impl<M: backend::ManagerBase> NewState<M> for CSRegisters<M> {
    fn new(manager: &mut M) -> Self
    where
        M: backend::ManagerAlloc,
    {
        Self {
            fflags: Cell::new(manager),
            frm: Cell::new(manager),
        }
    }
}

impl<M: backend::ManagerClone> Clone for CSRegisters<M> {
    fn clone(&self) -> Self {
        Self {
            fflags: self.fflags.clone(),
            frm: self.frm.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use crate::backend_test;
    use crate::machine_state::csregisters::CSRRepr;
    use crate::machine_state::csregisters::CSRegister;
    use crate::machine_state::csregisters::CSRegisters;
    use crate::machine_state::csregisters::Exception;
    use crate::machine_state::csregisters::check_write as check;
    use crate::state::NewState;

    #[test]
    fn test_read_write_access() {
        let is_illegal_instr = |e| -> bool { e == Exception::IllegalInstruction };

        // User registers
        assert!(check(CSRegister::fcsr).is_ok());
        assert!(check(CSRegister::instret).is_err_and(is_illegal_instr));
        assert!(check(CSRegister::cycle).is_err_and(is_illegal_instr));
    }

    /// Ensure that parsing CSRs matches the values assigned in the enum.
    #[test]
    fn test_csr_parser_roundtrip() {
        for csr in CSRegister::iter() {
            let value = csr as usize;
            let result = CSRegister::try_parse(value as u32);

            assert_eq!(Some(csr), result, "Expected {csr}, got {result:?}");
        }
    }

    backend_test!(test_fcsr, F, {
        let mut csrs = CSRegisters::new(&mut F::manager());

        // check starting values
        assert_eq!(0, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0, csrs.read::<CSRRepr>(CSRegister::fflags));

        // writing to fcsr is reflected in frm/fflags
        csrs.write(CSRegister::fcsr, 0b111_11111);

        assert_eq!(0b111_11111, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0b111, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0b11111, csrs.read::<CSRRepr>(CSRegister::fflags));

        // writing to frm is reflected in fcsr
        csrs.write(CSRegister::frm, 0b010);

        assert_eq!(0b010_11111, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0b010, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0b11111, csrs.read::<CSRRepr>(CSRegister::fflags));

        // writing to fflags is reflected in fcsr
        csrs.write(CSRegister::fflags, 0b01010);

        assert_eq!(0b010_01010, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0b010, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0b01010, csrs.read::<CSRRepr>(CSRegister::fflags));
    });
}
