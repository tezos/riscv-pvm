// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use super::CSRegister;

/// [`RootCSRegister`] is the set of unshadowed CSRs, whereas
/// the simple enum [`CSRegister`] is the public-facing API which contains shadowed CSRs.
///
/// For example, fflags exists as a [`CSRegister`] but the value is actually derived from
/// fcsr, which is also a [`RootCSRegister`].
#[expect(
    non_camel_case_types,
    reason = "We want to use the register names from the RISC-V specification"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, strum::EnumIter, strum::Display)]
pub(super) enum RootCSRegister {
    // Unprivileged Floating-Point CSRs
    fcsr,

    // Unprivileged Counter/Timers
    cycle,
    time,
    instret,
    hpmcounter3,
    hpmcounter4,
    hpmcounter5,
    hpmcounter6,
    hpmcounter7,
    hpmcounter8,
    hpmcounter9,
    hpmcounter10,
    hpmcounter11,
    hpmcounter12,
    hpmcounter13,
    hpmcounter14,
    hpmcounter15,
    hpmcounter16,
    hpmcounter17,
    hpmcounter18,
    hpmcounter19,
    hpmcounter20,
    hpmcounter21,
    hpmcounter22,
    hpmcounter23,
    hpmcounter24,
    hpmcounter25,
    hpmcounter26,
    hpmcounter27,
    hpmcounter28,
    hpmcounter29,
    hpmcounter30,
    hpmcounter31,
}

impl From<CSRegister> for RootCSRegister {
    #[inline(always)]
    fn from(value: CSRegister) -> Self {
        match value {
            // fflags is a shadow of fcsr
            CSRegister::fflags => RootCSRegister::fcsr,
            // frm is a shadow of fcsr
            CSRegister::frm => RootCSRegister::fcsr,
            CSRegister::fcsr => RootCSRegister::fcsr,
            CSRegister::cycle => RootCSRegister::cycle,
            CSRegister::time => RootCSRegister::time,
            CSRegister::instret => RootCSRegister::instret,
            CSRegister::hpmcounter3 => RootCSRegister::hpmcounter3,
            CSRegister::hpmcounter4 => RootCSRegister::hpmcounter4,
            CSRegister::hpmcounter5 => RootCSRegister::hpmcounter5,
            CSRegister::hpmcounter6 => RootCSRegister::hpmcounter6,
            CSRegister::hpmcounter7 => RootCSRegister::hpmcounter7,
            CSRegister::hpmcounter8 => RootCSRegister::hpmcounter8,
            CSRegister::hpmcounter9 => RootCSRegister::hpmcounter9,
            CSRegister::hpmcounter10 => RootCSRegister::hpmcounter10,
            CSRegister::hpmcounter11 => RootCSRegister::hpmcounter11,
            CSRegister::hpmcounter12 => RootCSRegister::hpmcounter12,
            CSRegister::hpmcounter13 => RootCSRegister::hpmcounter13,
            CSRegister::hpmcounter14 => RootCSRegister::hpmcounter14,
            CSRegister::hpmcounter15 => RootCSRegister::hpmcounter15,
            CSRegister::hpmcounter16 => RootCSRegister::hpmcounter16,
            CSRegister::hpmcounter17 => RootCSRegister::hpmcounter17,
            CSRegister::hpmcounter18 => RootCSRegister::hpmcounter18,
            CSRegister::hpmcounter19 => RootCSRegister::hpmcounter19,
            CSRegister::hpmcounter20 => RootCSRegister::hpmcounter20,
            CSRegister::hpmcounter21 => RootCSRegister::hpmcounter21,
            CSRegister::hpmcounter22 => RootCSRegister::hpmcounter22,
            CSRegister::hpmcounter23 => RootCSRegister::hpmcounter23,
            CSRegister::hpmcounter24 => RootCSRegister::hpmcounter24,
            CSRegister::hpmcounter25 => RootCSRegister::hpmcounter25,
            CSRegister::hpmcounter26 => RootCSRegister::hpmcounter26,
            CSRegister::hpmcounter27 => RootCSRegister::hpmcounter27,
            CSRegister::hpmcounter28 => RootCSRegister::hpmcounter28,
            CSRegister::hpmcounter29 => RootCSRegister::hpmcounter29,
            CSRegister::hpmcounter30 => RootCSRegister::hpmcounter30,
            CSRegister::hpmcounter31 => RootCSRegister::hpmcounter31,
        }
    }
}
