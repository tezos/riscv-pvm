// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Generate Etherlink Store Accesses
//!
//! A command-line tool for extracting storage access patterns from Etherlink execution traces.
//! Parses trace files and condenses storage operations into a structured JSON format with three sections:
//! setup, transaction, and block_creation.
//!
//! # Usage
//!
//! ```bash
//! # Extract from trace with default output
//! gen-etherlink-store-accesses etherlink.trace
//!
//! # Specify output file
//! gen-etherlink-store-accesses etherlink.trace -o output.json
//!
//! # Use a specific transaction index (default is 3 in order to capture an ERC-20 transfer)
//! gen-etherlink-store-accesses etherlink.trace -t 1
//! ```

mod parser;

use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser as ClapParser;

use crate::parser::LineType;
use crate::parser::OutputFormat;
use crate::parser::StoreAccessBuilder;
use crate::parser::classify_line;

/// Extract storage access patterns from Etherlink execution traces.
#[derive(ClapParser)]
#[command(about = "Extract storage access patterns from Etherlink execution traces")]
struct Args {
    /// Path to the trace file
    trace_file: PathBuf,

    /// Output JSON file
    #[arg(short, long, default_value = "store_accesses.json")]
    output: PathBuf,

    /// Transaction index to extract. Default to the first ERC-20 transfer in the trace.
    #[arg(short, long, default_value = "3")]
    tx_index: usize,
}

/// Section of the trace file being processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Setup,
    Transaction,
    BlockCreation,
    Other,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.trace_file.is_file() {
        anyhow::bail!("Trace file not found: {}", args.trace_file.display());
    }

    // Process trace file
    let output = process_trace(&args.trace_file, args.tx_index)?;

    // Write output
    let out_file = fs::File::create(&args.output)?;
    serde_json::to_writer_pretty(out_file, &output)?;

    println!("Extracted storage accesses:");
    println!("  Setup: {} operations", output.setup.len());
    println!("  Transaction: {} operations", output.transaction.len());
    println!(
        "  Block creation: {} operations",
        output.block_creation.len()
    );
    println!("Written to: {}", args.output.display());

    Ok(())
}

/// Process a trace file and extract storage accesses.
fn process_trace(trace_file: &Path, tx_index: usize) -> Result<OutputFormat> {
    // First pass: count total transactions
    let total_transactions = count_transactions(trace_file)?;
    println!("Found {total_transactions} transactions in trace");

    let file = fs::File::open(trace_file)?;
    let reader = BufReader::new(file);

    let mut section = Section::Other;
    let mut setup_accesses = Vec::new();
    let mut transaction_accesses = Vec::new();
    let mut block_creation_accesses = Vec::new();

    let mut condenser = StoreAccessBuilder::new();
    let mut current_tx_index = 0;

    for line in reader.lines() {
        let line = line?;
        let line_type = classify_line(&line);

        // Update section state
        section = update_section_state(
            section,
            &line_type,
            &mut current_tx_index,
            total_transactions,
        );

        // Process storage events based on current section
        if let LineType::StorageEvent(event) = line_type {
            if let Some(access) = condenser.process_event(*event) {
                match section {
                    Section::Setup => setup_accesses.push(access),
                    Section::Transaction if current_tx_index - 1 == tx_index => {
                        transaction_accesses.push(access);
                    }
                    Section::BlockCreation => block_creation_accesses.push(access),
                    _ => {} // Ignore the `Other` section and all other transactions
                }
            }
        }
    }

    Ok(OutputFormat {
        setup: setup_accesses,
        transaction: transaction_accesses,
        block_creation: block_creation_accesses,
    })
}

/// Count total transactions in a trace file.
fn count_transactions(trace_file: &Path) -> Result<usize> {
    let file = fs::File::open(trace_file)?;
    let reader = BufReader::new(file);
    let mut count = 0;

    for line in reader.lines() {
        let line = line?;
        if line.contains("[OTel] [start] apply_transaction") {
            count += 1;
        }
    }

    Ok(count)
}

/// Update the current section based on the line type.
fn update_section_state(
    current: Section,
    line_type: &LineType,
    tx_counter: &mut usize,
    total_transactions: usize,
) -> Section {
    match line_type {
        LineType::EnteringStageOne => Section::Setup,
        LineType::ComputingBlockInProgress => Section::Other,
        LineType::ApplyTransaction => {
            *tx_counter += 1;
            Section::Transaction
        }
        LineType::RegisterValidTransaction => {
            // Switch after the last transaction
            if *tx_counter >= total_transactions {
                Section::BlockCreation
            } else {
                Section::Other
            }
        }
        _ => current,
    }
}
