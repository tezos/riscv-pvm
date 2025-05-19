// SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

#![allow(non_upper_case_globals)]

pub mod effects;
mod root;
pub mod values;

use num_enum::TryFromPrimitive;
use root::RootCSRegister;
use strum::IntoEnumIterator;
use values::CSRValues;
use values::CSRValuesLayout;

use self::values::CSRValue;
use crate::bits::Bits64;
use crate::state::NewState;
use crate::state_backend as backend;
use crate::traps::Exception;

/// Privilege required to access a CSR
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Privilege {
    Unprivileged = 0,
    Supervisor = 1,
    Hypervisor = 2,
    Machine = 3,
}

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
    /// Determine the priviledge level required to access this CSR.
    #[inline(always)]
    pub fn privilege(self) -> Privilege {
        match self as usize {
            0x000..=0x0FF
            | 0x400..=0x4FF
            | 0x800..=0x8FF
            | 0xC00..=0xC7F
            | 0xC80..=0xCBF
            | 0xCC0..=0xCFF => Privilege::Unprivileged,

            0x100..=0x1FF
            | 0x500..=0x57F
            | 0x580..=0x5BF
            | 0x5C0..=0x5FF
            | 0x900..=0x97F
            | 0x980..=0x9BF
            | 0x9C0..=0x9FF
            | 0xD00..=0xD7F
            | 0xD80..=0xDBF
            | 0xDC0..=0xDFF => Privilege::Supervisor,

            0x200..=0x2FF
            | 0x600..=0x67F
            | 0x680..=0x6BF
            | 0x6C0..=0x6FF
            | 0xA00..=0xA7F
            | 0xA80..=0xABF
            | 0xAC0..=0xAFF
            | 0xE00..=0xE7F
            | 0xE80..=0xEBF
            | 0xEC0..=0xEFF => Privilege::Hypervisor,

            0x300..=0x3FF
            | 0x700..=0x77F
            | 0x780..=0x79F
            | 0x7A0..=0x7AF
            | 0x7B0..=0x7BF
            | 0x7C0..=0x7FF
            | 0xB00..=0xB7F
            | 0xB80..=0xBBF
            | 0xBC0..=0xBFF
            | 0xF00..=0xF7F
            | 0xF80..=0xFBF
            | 0xFC0..=0xFFF => Privilege::Machine,

            reg => unreachable!("Invalid CSR {reg:#x}",),
        }
    }

    /// Determines if the register is read-only
    #[inline(always)]
    pub fn is_read_only(self) -> bool {
        // Rules & Table of read-write / read-only ranges are in section 2.1 & table 2.1
        (self as usize >> 10) & 0b11 == 0b11
    }

    /// Enforce the WPRI and WLRL field specifications.
    ///
    /// Either return the value to be written, or None to signify that no write is necessary,
    /// leaving the existing value in its place.
    #[inline(always)]
    pub fn make_value_writable(self, value: CSRRepr) -> Option<CSRRepr> {
        // respect the reserved WPRI fields, setting them to 0
        let value = self.clear_wpri_fields(value);
        // apply WARL rules
        let value = self.transform_warl_fields(value)?;
        // check if value is legal w.r.t. WLRL fields
        self.is_legal(value).then_some(value)
    }

    const WPRI_MASK_EMPTY: CSRRepr = CSRRepr::MAX;

    /// Return the mask of non reserved bits, (WPRI bits are 0)
    /// Relevant section 2.3 - privileged spec
    #[inline(always)]
    pub fn wpri_mask(self) -> CSRRepr {
        CSRegister::WPRI_MASK_EMPTY
    }

    /// Ensures that WPRI fields are kept read-only zero.
    ///
    /// Conforming to Section 2.3 - privileged spec
    #[inline(always)]
    pub fn clear_wpri_fields(self, new_value: CSRRepr) -> CSRRepr {
        new_value & self.wpri_mask()
    }

    /// If the register is WLRL, return if `new_value` is legal, false otherwise
    ///
    /// Section 2.3 - privileged spec
    #[inline(always)]
    pub fn is_legal(self, _new_value: CSRRepr) -> bool {
        true
    }

    /// Ensures WARL registers / fields are respected
    ///
    /// Section 2.3 - privileged spec
    ///
    /// If `None` is returned, then no update must take place
    #[inline(always)]
    pub fn transform_warl_fields(self, new_value: CSRRepr) -> Option<CSRRepr> {
        Some(new_value)
    }

    /// FCSR mask
    const FCSR_MASK: CSRRepr = Self::FRM_MASK | Self::FFLAGS_MASK;

    /// FRM mask
    const FRM_MASK: CSRRepr = 0b111 << Self::FRM_SHIFT;

    /// FRM is bits 5..7
    const FRM_SHIFT: usize = 5;

    /// FFLAGS mask
    const FFLAGS_MASK: CSRRepr = 0b11111;

    /// Get the default value for the register.
    fn default_value(&self) -> u64 {
        match self {
            CSRegister::cycle | CSRegister::time | CSRegister::instret => {
                // Default is that the machine starts at 0
                0
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
                // All counters shall start at 0 again
                0
            }

            CSRegister::fflags => {
                // Resets accrued floating-point exceptions
                0b00000
            }

            CSRegister::frm => {
                // 000 = RNE aka "round to nearest, ties to even"
                0b000
            }

            CSRegister::fcsr => {
                // fcsr is a combination of fflags and fcsr
                CSRegister::fflags.default_value() & (CSRegister::frm.default_value() << 5)
            }
        }
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
pub use values::CSRRepr;

/// Return type of read/write operations
pub type Result<R> = core::result::Result<R, Exception>;

/// Checks that `reg` is write-able.
///
/// Throws [`Exception::IllegalInstruction`] in case of wrong access rights.
/// Section 2.1 - privileged spec
#[inline(always)]
pub fn check_write(reg: CSRegister) -> Result<()> {
    if reg.is_read_only() {
        return Err(Exception::IllegalInstruction);
    }

    Ok(())
}

/// CSRs
pub struct CSRegisters<M: backend::ManagerBase> {
    registers: CSRValues<M>,
}

impl<M: backend::ManagerBase> CSRegisters<M> {
    /// Transform the write operation to account for shadow registers.
    ///
    /// Sections 3.1.6 & 4.1.1
    #[inline(always)]
    fn transform_write(&self, reg: CSRegister, value: CSRRepr) -> CSRRepr
    where
        M: backend::ManagerRead,
    {
        // the update of a shadow register follows the steps:
        // 1. keep the shadowed fields from [value]
        // 2. all the other, non-shadowed fields are the underlying register
        //    masked with the inverse of the shadowed fields mask
        // Note: This works because currently there are no shadowed WLRL registers
        match reg {
            CSRegister::fcsr => value & CSRegister::FCSR_MASK,
            CSRegister::frm => {
                let fcsr = self.registers.fcsr.read();
                let fcsr = fcsr & !CSRegister::FRM_MASK;
                ((value << CSRegister::FRM_SHIFT) & CSRegister::FRM_MASK) | fcsr
            }
            CSRegister::fflags => {
                let fcsr = self.registers.fcsr.read();
                let fcsr = fcsr & !CSRegister::FFLAGS_MASK;
                (value & CSRegister::FFLAGS_MASK) | fcsr
            }
            _ => value,
        }
    }

    /// Transform a read operation to account for shadow registers.
    ///
    /// `source_reg_value` holds the value of the register which is the ground truth for `reg`
    /// if known, `None` otherwise.
    ///
    /// Sections 3.1.6 & 4.1.1
    #[inline(always)]
    fn transform_read(&self, reg: CSRegister, source_reg_value: Option<CSRRepr>) -> CSRRepr
    where
        M: backend::ManagerRead,
    {
        let source_reg_value = source_reg_value.unwrap_or_else(|| {
            // If reg is a shadow, obtain the underlying ground truth for that register
            self.general_raw_read(reg.into())
        });

        // modify the value according to the shadowing rules of each register
        match reg {
            CSRegister::fcsr => source_reg_value & CSRegister::FCSR_MASK,
            CSRegister::frm => (source_reg_value & CSRegister::FRM_MASK) >> CSRegister::FRM_SHIFT,
            CSRegister::fflags => source_reg_value & CSRegister::FFLAGS_MASK,
            _ => source_reg_value,
        }
    }

    /// Write to a CSR.
    #[inline(always)]
    pub fn write<V: Bits64>(&mut self, reg: CSRegister, value: V)
    where
        M: backend::ManagerReadWrite,
    {
        if let Some(value) = reg.make_value_writable(value.to_bits()) {
            let value = self.transform_write(reg, value);
            let source_reg: RootCSRegister = reg.into();
            self.general_raw_write(source_reg, value);
        }
    }

    /// Read from a CSR.
    #[inline(always)]
    pub fn read<V: Bits64>(&self, reg: CSRegister) -> V
    where
        M: backend::ManagerRead,
    {
        V::from_bits(self.transform_read(reg, None))
    }

    /// Replace the CSR value, returning the previous value.
    #[inline(always)]
    pub fn replace<V: Bits64>(&mut self, reg: CSRegister, value: V) -> V
    where
        M: backend::ManagerReadWrite,
    {
        if let Some(value) = reg.make_value_writable(value.to_bits()) {
            let value = self.transform_write(reg, value);
            let source_reg: RootCSRegister = reg.into();
            let old_value = self.general_raw_replace(source_reg, value);
            let old_value = self.transform_read(reg, Some(old_value));
            V::from_bits(old_value)
        } else {
            self.read(reg)
        }
    }

    /// Set bits in the CSR.
    #[inline(always)]
    pub fn set_bits(&mut self, reg: CSRegister, bits: CSRRepr) -> CSRValue
    where
        M: backend::ManagerReadWrite,
    {
        let old_value: CSRValue = self.read(reg);
        let new_value = old_value.repr() | bits;
        self.write(reg, new_value);
        old_value
    }

    /// Clear bits in the CSR.
    #[inline(always)]
    pub fn clear_bits(&mut self, reg: CSRegister, bits: CSRRepr) -> CSRValue
    where
        M: backend::ManagerReadWrite,
    {
        let old_value: CSRValue = self.read(reg);
        let new_value = old_value.repr() & !bits;
        self.write(reg, new_value);
        old_value
    }
}

/// Layout for [CSRegisters]
pub type CSRegistersLayout = CSRValuesLayout;

impl<M: backend::ManagerBase> CSRegisters<M> {
    /// Bind the CSR state to the allocated space.
    pub fn bind(space: backend::AllocatedOf<CSRegistersLayout, M>) -> Self {
        Self {
            registers: values::CSRValues::bind(space),
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: backend::FnManager<backend::Ref<'a, M>>>(
        &'a self,
    ) -> backend::AllocatedOf<CSRegistersLayout, F::Output> {
        self.registers.struct_ref::<F>()
    }

    /// Reset the control and state registers.
    pub fn reset(&mut self)
    where
        M: backend::ManagerWrite,
    {
        // Try to reset known CSRs to known default values.
        for reg in CSRegister::iter() {
            self.general_raw_write(reg.into(), reg.default_value());
        }
    }
}

impl<M: backend::ManagerBase> NewState<M> for CSRegisters<M> {
    fn new(manager: &mut M) -> Self
    where
        M: backend::ManagerAlloc,
    {
        Self {
            registers: CSRValues::new(manager),
        }
    }
}

impl<M: backend::ManagerClone> Clone for CSRegisters<M> {
    fn clone(&self) -> Self {
        Self {
            registers: self.registers.clone(),
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
    use crate::state::NewState;

    #[test]
    fn test_read_write_access() {
        use crate::machine_state::csregisters::CSRegister as csreg;
        use crate::machine_state::csregisters::Exception;
        use crate::machine_state::csregisters::check_write as check;

        let is_illegal_instr = |e| -> bool { e == Exception::IllegalInstruction };

        // User registers
        assert!(check(csreg::fcsr).is_ok());
        assert!(check(csreg::instret).is_err_and(is_illegal_instr));
        assert!(check(csreg::cycle).is_err_and(is_illegal_instr));
    }

    #[test]
    fn test_wlrl() {
        use crate::machine_state::csregisters::CSRegister as csreg;

        // Additionally check if value remains legal after using `make_value_writable`
        let check =
            |reg: csreg, value| reg.is_legal(value) && reg.make_value_writable(value).is_some();

        // Registers that are not xcause should always be ok
        assert!(check(csreg::time, 0x0));
    }

    #[test]
    fn test_writable_warl() {
        use crate::machine_state::csregisters::CSRegister as csreg;

        let check = |reg: csreg, value| reg.make_value_writable(value).unwrap();

        // non warl register
        assert!(check(csreg::instret, 0x42) == 0x42);
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
        csrs.write(CSRegister::fcsr, u64::MAX);

        assert_eq!(0xff, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0b111, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0b11111, csrs.read::<CSRRepr>(CSRegister::fflags));

        // writing to frm is reflected in fcsr
        csrs.write(CSRegister::frm, 0b010);

        assert_eq!(0b01011111, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0b010, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0b11111, csrs.read::<CSRRepr>(CSRegister::fflags));

        // writing to fflags is reflected in fcsr
        csrs.write(CSRegister::fflags, 0b01010);

        assert_eq!(0b01001010, csrs.read::<CSRRepr>(CSRegister::fcsr));
        assert_eq!(0b010, csrs.read::<CSRRepr>(CSRegister::frm));
        assert_eq!(0b01010, csrs.read::<CSRRepr>(CSRegister::fflags));
    });
}
