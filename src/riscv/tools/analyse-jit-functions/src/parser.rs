// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! JSON log file parsing functionality
//!
//! This module provides structures and functions to parse log files and extract JIT function data
//! from JSON entries.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

/// Top-level structure of a JSON log entry
///
/// Represents the outer structure of log entries in the JSON log file.
/// Each entry must contain a `fields` object with the actual function data.
#[derive(Debug, Deserialize)]
struct LogEntry {
    /// Required fields containing the actual function data
    fields: FunctionData,
}

/// Function data that can be parsed from JSON log entries or used as processed data
///
/// This structure holds all the information about a single function, either
/// directly parsed from JSON log entries or as an intermediate representation
/// before generating the output files.
#[derive(Debug, Deserialize)]
pub struct FunctionData {
    /// Unique function identifier (hash)
    pub func_name: String,

    /// Memory address where the function is located
    pub func_addr: u64,

    /// Unoptimised intermediate representation code
    pub unoptimised_ir: String,

    /// Optimised intermediate representation code
    pub optimised_ir: String,

    /// Cranelift VCode
    pub vcode: String,

    /// Machine code
    pub machine_code: String,
}

/// Parses a JSON log file and extracts function data from all valid entries.
///
/// Reads the specified log file line by line, parsing each line as JSON.
/// For each successfully parsed entry, extracts the function data and builds a `FunctionData` struct.
///
/// # Arguments
///
/// * `log_file_path` - Path to the JSON log file to parse
///
/// # Returns
///
/// * `Ok(Vec<FunctionData>)` - Successfully parsed function definitions
/// * `Err(anyhow::Error)` - File I/O error or JSON parsing error
pub fn parse_log_file(log_file_path: &Path) -> Result<Vec<FunctionData>> {
    let file = File::open(log_file_path)?;
    let reader = BufReader::new(file);

    let functions = reader
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| {
            let entry = serde_json::from_str::<LogEntry>(line.trim()).ok()?;
            Some(entry.fields)
        })
        .collect();

    Ok(functions)
}
