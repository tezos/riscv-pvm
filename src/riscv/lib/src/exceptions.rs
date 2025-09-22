// SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Exceptions that may arise during execution

/// RISC-V Exceptions (also known as synchronous exceptions)
#[derive(Debug, PartialEq, Eq, thiserror::Error, strum::Display, Clone, Copy)]
#[repr(i64)]
pub enum Exception {
    /// Could not access instruction memory.
    InstructionAccessFault = 1,

    /// Encountered an illegal instruction.
    IllegalInstruction,

    /// Breakpoint has been triggered.
    Breakpoint,

    /// Reading from data memory failed.
    LoadAccessFault,

    /// Writing to data memory failed.
    StoreAMOAccessFault,

    /// Call out to the execution environment.
    EnvCall,

    /// Synchronise data and instruction memory.
    FenceI,

    /// Force the current instruction to be fetched from memory and executed.
    ///
    /// This exception *cannot* occur if executing an instruction fetched directly
    /// from memory.
    ForceFetchRun,
}
