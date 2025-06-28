// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! JIT Function Analyser
//!
//! A command-line tool for analysing JIT function log files and extracting function data.
//! For each valid log entry found, creates a type-based directory structure where each code type
//! has its own directory with `by-name` and `by-address` subdirectories. Files are created for
//! unoptimised IR, optimised IR, VCode, and disassembled host machine code using the Capstone
//! disassembly engine. Symbolic links provide address-based access to function files.
//!
//! # Usage
//!
//! ```bash
//! # Process log file with default output directory
//! analyse-jit-functions log.json
//!
//! # Process log file with custom output directory
//! analyse-jit-functions log.json -o my_functions
//! ```
//!
//! # Output Format
//!
//! Creates a type-based directory structure:
//!
//! ```text
//! output_dir/
//! ├── unoptimised_ir/
//! │   ├── by-name/
//! │   │   └── {function_name}.txt      # Unoptimised IR for each function
//! │   └── by-address/
//! │       └── 0x{address}.txt -> ../by-name/{function_name}.txt
//! ├── optimised_ir/
//! │   ├── by-name/
//! │   │   └── {function_name}.txt      # Optimised IR for each function
//! │   └── by-address/
//! │       └── 0x{address}.txt -> ../by-name/{function_name}.txt
//! ├── vcode/
//! │   ├── by-name/
//! │   │   └── {function_name}.txt      # Cranelift VCode for each function
//! │   └── by-address/
//! │       └── 0x{address}.txt -> ../by-name/{function_name}.txt
//! └── asm/
//!     ├── by-name/
//!     │   └── {function_name}.txt      # Disassembled machine code for each function
//!     └── by-address/
//!         └── 0x{address}.txt -> ../by-name/{function_name}.txt
//! ```

mod disassembly;
mod parser;

use std::fs;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs as unix_fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;

use crate::disassembly::create_disassembler;
use crate::disassembly::disassemble_machine_code;
use crate::parser::FunctionData;
use crate::parser::parse_log_file;

// Constants for type directory names to ensure consistency
const UNOPTIMISED_IR_DIR: &str = "unoptimised_ir";
const OPTIMISED_IR_DIR: &str = "optimised_ir";
const VCODE_DIR: &str = "vcode";
const ASM_DIR: &str = "asm";
const BY_NAME_DIR: &str = "by-name";
const BY_ADDRESS_DIR: &str = "by-address";

/// Command-line arguments for the JIT function analyser
#[derive(Parser)]
#[command(
    about = "Analyse JIT function log files and extract function data from all valid entries"
)]
struct Args {
    /// Path to the JSON log file
    log_file: PathBuf,

    /// Output directory for function files
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
}

/// Creates the type-based directory structure with by-name and by-address subdirectories.
///
/// Creates the following structure:
/// - unoptimised_ir/by-name/
/// - unoptimised_ir/by-address/
/// - optimised_ir/by-name/
/// - optimised_ir/by-address/
/// - vcode/by-name/
/// - vcode/by-address/
/// - asm/by-name/
/// - asm/by-address/
///
/// # Arguments
///
/// * `output_dir` - Base output directory
///
/// # Returns
///
/// * `Ok(())` - Directories created successfully
/// * `Err(anyhow::Error)` - Directory creation failed
fn create_type_directories(output_dir: &Path) -> Result<()> {
    for type_name in [UNOPTIMISED_IR_DIR, OPTIMISED_IR_DIR, VCODE_DIR, ASM_DIR] {
        let type_dir = output_dir.join(type_name);
        let by_name_dir = type_dir.join(BY_NAME_DIR);
        let by_address_dir = type_dir.join(BY_ADDRESS_DIR);

        fs::create_dir_all(&by_name_dir)?;
        fs::create_dir_all(&by_address_dir)?;
    }

    Ok(())
}

/// Creates a file for a specific code type with the function's content.
///
/// Creates a file in the `by-name` subdirectory and a corresponding symbolic link
/// in the `by-address` subdirectory pointing to the file.
///
/// # Arguments
///
/// * `content` - The content to write to the file
/// * `func_name` - Function name used for the filename (without sanitization)
/// * `func_addr` - Function address used for symlink creation
/// * `type_dir` - The type directory (e.g., unoptimised_ir, optimised_ir, vcode, asm)
///
/// # Returns
///
/// * `Ok(())` - File and symlink created successfully
/// * `Err(anyhow::Error)` - File or symlink creation failed
fn create_type_file(content: &str, func_name: &str, func_addr: u64, type_dir: &Path) -> Result<()> {
    let by_name_dir = type_dir.join(BY_NAME_DIR);
    let by_address_dir = type_dir.join(BY_ADDRESS_DIR);

    // Create file in by-name directory with function name
    let named_base = format!("{func_name}.txt");
    let named_path = by_name_dir.join(&named_base);
    let mut file = File::create(&named_path)?;
    writeln!(file, "{}", content)?;

    // Create symlink in by-address directory with hex address
    let addr_base = format!("{func_addr:#x}.txt");
    let addr_path = by_address_dir.join(addr_base);

    // Remove existing symlink if it exists to avoid conflicts
    if addr_path.exists() {
        fs::remove_file(&addr_path)?;
    }

    // Create relative symlink path pointing to by-name file
    let target_filename = Path::new("..").join(BY_NAME_DIR).join(named_base);
    unix_fs::symlink(&target_filename, &addr_path).context("Failed to create symbolic link")?;

    Ok(())
}

/// Creates files for a single function across all code type directories.
///
/// Creates files in the type-based structure:
/// - unoptimised_ir/by-name/{func_name}.txt and unoptimised_ir/by-address/{address}.txt
/// - optimised_ir/by-name/{func_name}.txt and optimised_ir/by-address/{address}.txt
/// - vcode/by-name/{func_name}.txt and vcode/by-address/{address}.txt
/// - asm/by-name/{func_name}.txt and asm/by-address/{address}.txt
///
/// # Arguments
///
/// * `capstone` - Capstone disassembler instance for machine code disassembly
/// * `func_data` - Function data to write to files
/// * `output_dir` - Base output directory containing type directories
///
/// # Returns
///
/// * `Ok(())` - All files and symlinks successfully created
/// * `Err(anyhow::Error)` - File or symlink creation failed
///
/// # Errors
///
/// * I/O errors - Cannot create files or symlinks
fn create_function_files(
    capstone: &capstone::Capstone,
    func_data: &FunctionData,
    output_dir: &Path,
) -> Result<()> {
    // Create unoptimised IR file and symlink
    let unoptimised_ir_dir = output_dir.join(UNOPTIMISED_IR_DIR);
    create_type_file(
        &func_data.unoptimised_ir,
        &func_data.func_name,
        func_data.func_addr,
        &unoptimised_ir_dir,
    )?;

    // Create optimised IR file and symlink
    let optimised_ir_dir = output_dir.join(OPTIMISED_IR_DIR);
    create_type_file(
        &func_data.optimised_ir,
        &func_data.func_name,
        func_data.func_addr,
        &optimised_ir_dir,
    )?;

    // Create VCode file and symlink
    let vcode_dir = output_dir.join(VCODE_DIR);
    create_type_file(
        &func_data.vcode,
        &func_data.func_name,
        func_data.func_addr,
        &vcode_dir,
    )?;

    // Create disassembled machine code file and symlink
    let asm_dir = output_dir.join(ASM_DIR);
    let disassembly =
        match disassemble_machine_code(capstone, &func_data.machine_code, func_data.func_addr) {
            Ok(disassembly) => disassembly,
            Err(e) => format!("Error disassembling: {}", e),
        };
    create_type_file(
        &disassembly,
        &func_data.func_name,
        func_data.func_addr,
        &asm_dir,
    )?;

    println!(
        "Created files for function: {} ({:#x})",
        func_data.func_name, func_data.func_addr
    );
    Ok(())
}

/// Main entry point for the function analyzer
///
/// Parses command-line arguments, validates input, processes the log file,
/// and generates output files for each function found. Provides progress
/// information and error handling throughout the process.
fn main() -> Result<()> {
    let args = Args::parse();

    if !args.log_file.exists() {
        anyhow::bail!("Log file '{}' not found", args.log_file.display())
    }

    // Parse the log file
    println!("Parsing log file: {}", args.log_file.display());
    let functions = parse_log_file(&args.log_file)?;

    println!("Found {} functions", functions.len());

    // Determine output directory
    let output_dir = args.output_dir.unwrap_or_else(|| {
        args.log_file
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    });

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;
    println!("Output directory: {}", output_dir.display());

    // Create type-based directory structure with constants
    create_type_directories(&output_dir)?;

    // Create Capstone disassembler instance once for reuse
    let capstone = create_disassembler()?;

    // Process each function and create files across all type directories
    for func_data in &functions {
        create_function_files(&capstone, func_data, &output_dir)?;
    }

    println!();
    println!(
        "Processing complete. Created files for {} functions in {}",
        functions.len(),
        output_dir.display()
    );

    Ok(())
}
