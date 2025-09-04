// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Machine code disassembly functionality using the Capstone engine
//!
//! This module provides functions to disassemble hex-encoded machine code into
//! human-readable assembly instructions for the host architecture.

use anyhow::Context;
use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;
use capstone::arch::arm;
use capstone::arch::arm64;
use capstone::arch::x86;

/// Creates a Capstone disassembler instance for the host architecture.
///
/// # Returns
///
/// * `Ok(Capstone)` - Successfully created disassembler
/// * `Err(anyhow::Error)` - Failed to create disassembler or unsupported architecture
pub fn create_disassembler() -> Result<Capstone> {
    let capstone = if cfg!(target_arch = "x86_64") {
        Capstone::new()
            .x86()
            .mode(x86::ArchMode::Mode64)
            .detail(true)
            .build()
    } else if cfg!(target_arch = "x86") {
        Capstone::new()
            .x86()
            .mode(x86::ArchMode::Mode32)
            .detail(true)
            .build()
    } else if cfg!(target_arch = "aarch64") {
        Capstone::new()
            .arm64()
            .mode(arm64::ArchMode::Arm)
            .detail(true)
            .build()
    } else if cfg!(target_arch = "arm") {
        Capstone::new()
            .arm()
            .mode(arm::ArchMode::Arm)
            .detail(true)
            .build()
    } else {
        anyhow::bail!("Unsupported host architecture")
    }
    .context("Failed to create Capstone disassembler")?;

    Ok(capstone)
}

/// Disassembles hex-encoded machine code into human-readable assembly instructions.
///
/// Takes a hex-encoded string representing machine code and uses the provided Capstone
/// disassembly engine to convert it into readable assembly instructions for the host architecture.
///
/// # Arguments
///
/// * `capstone` - Capstone disassembler instance
/// * `hex_code` - Hex-encoded machine code string (e.g., "48c7c03a000000")
/// * `func_addr` - Function address where the code is located in memory
///
/// # Returns
///
/// * `Ok(String)` - Disassembled assembly instructions, one per line
/// * `Err(anyhow::Error)` - Hex decoding failed or disassembly failed
pub fn disassemble_machine_code(
    capstone: &Capstone,
    hex_code: &str,
    func_addr: u64,
) -> Result<String> {
    // Decode hex string to bytes
    let machine_code = hex::decode(hex_code.trim())?;

    // Disassemble the machine code using the actual function address
    let instrs = capstone
        .disasm_all(&machine_code, func_addr)
        .context("Failed to disassemble machine code")?;

    // Format instructions as strings without leading addresses to make diffing easier
    let assembly = instrs
        .iter()
        .map(|insn| {
            format!(
                "{} {}",
                insn.mnemonic().unwrap_or(""),
                insn.op_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(assembly)
}
