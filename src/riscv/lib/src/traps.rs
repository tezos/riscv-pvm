// SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Traps doc
//! There are 4 types of traps, depending on where they are handled and visibility to the hart.
//! ### Contained:
//! A trap which is handled by the normal procedure of
//! trap handling without interacting with the execution environment.
//! (Software knows a trap is taken e.g. U -> M/S, S -> M/S, M -> M)
//!
//! ### Requested:
//! A trap requested by the software to the execution environment.
//! so the software is aware of traps like U/S/M -> EE -> M/S
//!
//! ### Invisible:
//! A trap is handled by the execution environment without software being aware of this.
//!
//! ### Fatal:
//! A trap which causes the execution environment to halt the machine.
//!

use std::fmt::Formatter;

use tezos_smart_rollup_constants::riscv::SbiError;

use crate::machine_state::memory::Address;

/// RISC-V Exceptions (also known as synchronous exceptions)
#[derive(PartialEq, Eq, thiserror::Error, strum::Display, Debug, Clone, Copy)]
pub enum EnvironException {
    EnvCall,
}

impl TryFrom<&Exception> for EnvironException {
    type Error = &'static str;

    fn try_from(value: &Exception) -> Result<Self, Self::Error> {
        match value {
            Exception::EnvCall => Ok(EnvironException::EnvCall),
            Exception::Breakpoint
            | Exception::IllegalInstruction
            | Exception::InstructionAccessFault(_)
            | Exception::LoadAccessFault(_)
            | Exception::StoreAMOAccessFault(_)
            | Exception::InstructionPageFault(_)
            | Exception::LoadPageFault(_)
            | Exception::StoreAMOPageFault(_) => {
                Err("Execution environment supports only ecall exceptions")
            }
        }
    }
}

/// RISC-V Exceptions (also known as synchronous exceptions)
#[derive(PartialEq, Eq, thiserror::Error, strum::Display, Clone, Copy)]
pub enum Exception {
    /// `InstructionAccessFault(addr)` where `addr` is the faulting instruction address
    InstructionAccessFault(Address),
    IllegalInstruction,
    Breakpoint,
    /// `LoadAccessFault(addr)` where `addr` is the faulting load address
    LoadAccessFault(Address),
    /// `StoreAccessFault(addr)` where `addr` is the faulting store address
    StoreAMOAccessFault(Address),
    EnvCall,
    InstructionPageFault(Address),
    LoadPageFault(Address),
    StoreAMOPageFault(Address),
}

impl core::fmt::Debug for Exception {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Self::InstructionPageFault(adr) => write!(f, "InstructionPageFault({adr:#X})"),
            Self::LoadPageFault(adr) => write!(f, "LoadPageFault({adr:#X})"),
            Self::StoreAMOPageFault(adr) => write!(f, "StoreAMOPageFault({adr:#X})"),
            Self::LoadAccessFault(adr) => write!(f, "LoadAccessFault({adr:#X})"),
            other => write!(f, "{other}"),
        }
    }
}

impl From<Exception> for SbiError {
    fn from(value: Exception) -> Self {
        match value {
            Exception::InstructionAccessFault(_)
            | Exception::InstructionPageFault(_)
            | Exception::LoadAccessFault(_)
            | Exception::LoadPageFault(_)
            | Exception::StoreAMOAccessFault(_)
            | Exception::StoreAMOPageFault(_) => SbiError::InvalidAddress,
            Exception::IllegalInstruction | Exception::Breakpoint | Exception::EnvCall => {
                SbiError::Failed
            }
        }
    }
}
