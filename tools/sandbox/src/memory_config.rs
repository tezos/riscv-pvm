// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::str::FromStr;

/// Memory configuration for the PVM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryConfigValue {
    M64M,
    M1G,
    M4G,
    M16G,
    M64G,
}

impl FromStr for MemoryConfigValue {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "64m" => Ok(MemoryConfigValue::M64M),
            "1g" => Ok(MemoryConfigValue::M1G),
            "4g" => Ok(MemoryConfigValue::M4G),
            "16g" => Ok(MemoryConfigValue::M16G),
            "64g" => Ok(MemoryConfigValue::M64G),
            cfg => Err(format!("Unsupported memory configuration: {cfg}")),
        }
    }
}
