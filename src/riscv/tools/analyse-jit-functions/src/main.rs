//! JIT Function Analyser
//!
//! A command-line tool for analysing JIT function log files and extracting function data.
//! For each valid log entry found, creates a directory structure organized into `by-name` and
//! `by-address` subdirectories. Function directories are created in `by-name` with separate files
//! for unoptimised IR, optimised IR, VCode, and disassembled host machine code using the Capstone
//! disassembly engine. Symbolic links are created in `by-address` for easy address-based access.
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
//! Creates an organized directory structure:
//!
//! ```
//! output_dir/
//! ├── by-name/
//! │   └── {function_name}/         # Function directories
//! │       ├── unoptimised_ir.txt   # Unoptimised IR code
//! │       ├── optimised_ir.txt     # Optimised IR code
//! │       ├── vcode.txt            # Cranelift VCode
//! │       └── disasm.txt           # Disassembled machine code
//! └── by-address/
//!     └── 0x{address} -> ../by-name/{function_name}/  # Address-based symlinks
//! ```

use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::os::unix::fs as unix_fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use capstone::Capstone;
use capstone::Insn;
use capstone::arch::BuildsCapstone;
use capstone::arch::arm;
use capstone::arch::arm64;
use capstone::arch::x86;
use clap::Parser;
use serde::Deserialize;

/// Top-level structure of a JSON log entry.
///
/// Represents the outer structure of log entries in the JSON log file.
/// Each entry must contain a `fields` object with the actual function data.
#[derive(Debug, Deserialize)]
struct LogEntry {
    /// Required fields containing the actual function data
    fields: FunctionData,
}

/// Function data that can be parsed from JSON log entries or used as processed data.
///
/// This structure holds all the information about a single function, either
/// directly parsed from JSON log entries or as an intermediate representation
/// before generating the output files.
#[derive(Debug, Deserialize)]
struct FunctionData {
    /// Unique function identifier (hash)
    func_name: String,

    /// Memory address where the function is located
    func_addr: u64,

    /// Unoptimised intermediate representation code
    unoptimised_ir: String,

    /// Optimised intermediate representation code
    optimised_ir: String,

    /// Cranelift VCode
    vcode: String,

    /// Machine code
    machine_code: String,
}

#[derive(Parser)]
#[command(name = "analyse-jit-functions")]
#[command(
    about = "Analyse JIT function log files and extract function data from all valid entries"
)]
struct Args {
    /// Path to the JSON log file
    log_file: PathBuf,

    /// Output directory for function files (default: <log_file_stem>_functions)
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
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
///
/// # Errors
///
/// * I/O errors - File cannot be opened or read
/// * JSON parsing errors - Line contains invalid JSON (logged as warning, not fatal)
fn parse_log_file(log_file_path: &Path) -> Result<Vec<FunctionData>> {
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

/// Disassembles hex-encoded machine code into human-readable assembly instructions.
///
/// Takes a hex-encoded string representing machine code and uses the Capstone
/// disassembly engine to convert it into readable assembly instructions for the host architecture.
///
/// # Arguments
///
/// * `hex_code` - Hex-encoded machine code string (e.g., "48c7c03a000000")
/// * `func_addr` - Function address where the code is located in memory
///
/// # Returns
///
/// * `Ok(String)` - Disassembled assembly instructions, one per line
/// * `Err(anyhow::Error)` - Hex decoding failed or disassembly failed
///
/// # Errors
///
/// * Hex decoding errors - Invalid hex characters in input
/// * Capstone initialization errors - Failed to create disassembler
/// * Disassembly errors - Invalid machine code or unsupported instructions
fn disassemble_machine_code(hex_code: &str, func_addr: u64) -> Result<String> {
    // Decode hex string to bytes
    let machine_code = hex::decode(hex_code.trim())?;

    // Create Capstone disassembler for host architecture
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
        anyhow::bail!("Unsupported host architecture: {}", std::env::consts::ARCH)
    }
    .context("Failed to create Capstone disassembler")?;

    // Disassemble the machine code using the actual function address
    let instrs = capstone
        .disasm_all(&machine_code, func_addr)
        .context("Failed to disassemble machine code")?;

    // Format instructions as strings
    let assembly = instrs
        .iter()
        .map(Insn::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    Ok(assembly)
}

/// Creates a symbolic link from the function address to the function directory.
///
/// # Arguments
///
/// * `func_addr` - Function address to use as symlink name
/// * `target_dir_name` - Name of the target directory (function name)
/// * `by_address_dir` - The by-address directory where symlink should be created
///
/// # Returns
///
/// * `Ok(())` - Symlink created successfully
/// * `Err(anyhow::Error)` - Symlink creation failed
fn create_address_symlink(func_addr: u64, target_dir_name: &str, by_address_dir: &Path) -> Result<()> {
    let symlink_name = format!("{:#x}", func_addr);
    let symlink_path = by_address_dir.join(symlink_name);

    // Remove existing symlink if it exists
    if symlink_path.exists() {
        fs::remove_file(&symlink_path)?;
    }

    // Create relative path from by-address to by-name directory
    let target_path = Path::new("../by-name").join(target_dir_name);

    unix_fs::symlink(&target_path, &symlink_path).context("Failed to create symbolic link")?;

    Ok(())
}

/// Creates a file containing unoptimised IR code.
fn create_unoptimised_ir_file(func_data: &FunctionData, func_dir: &Path) -> Result<()> {
    let filepath = func_dir.join("unoptimised_ir.txt");
    let mut file = File::create(&filepath)?;
    writeln!(file, "{}", func_data.unoptimised_ir)?;
    Ok(())
}

/// Creates a file containing optimised IR code.
fn create_optimised_ir_file(func_data: &FunctionData, func_dir: &Path) -> Result<()> {
    let filepath = func_dir.join("optimised_ir.txt");
    let mut file = File::create(&filepath)?;
    writeln!(file, "{}", func_data.optimised_ir)?;
    Ok(())
}

/// Creates a file containing VCode.
fn create_vcode_file(func_data: &FunctionData, func_dir: &Path) -> Result<()> {
    let filepath = func_dir.join("vcode.txt");
    let mut file = File::create(&filepath)?;
    writeln!(file, "{}", func_data.vcode)?;
    Ok(())
}

/// Creates a file containing disassembled machine code.
fn create_disasm_file(func_data: &FunctionData, func_dir: &Path) -> Result<()> {
    let filepath = func_dir.join("disasm.txt");
    let mut file = File::create(&filepath)?;
    match disassemble_machine_code(&func_data.machine_code, func_data.func_addr) {
        Ok(disassembly) => writeln!(file, "{}", disassembly)?,
        Err(e) => writeln!(file, "Error disassembling: {}", e)?,
    }
    Ok(())
}

/// Creates a directory for a single function containing separate files for each code representation.
///
/// Generates a directory named after the function name in the by-name subdirectory containing:
/// - unoptimised_ir.txt
/// - optimised_ir.txt  
/// - vcode.txt
/// - disasm.txt
///
/// Also creates a symbolic link from the function address to the directory in the by-address subdirectory.
///
/// # Directory Naming Logic
///
/// Directories are created in `by-name/` using function names. A symbolic link named
/// `0x{address}` in `by-address/` points to the function directory for address-based access.
///
/// # Arguments
///
/// * `func_data` - Function data to write to files
/// * `by_name_dir` - The by-name directory where the function directory should be created
/// * `by_address_dir` - The by-address directory where the symlink should be created
///
/// # Returns
///
/// * `Ok(())` - Directory, files, and symlink successfully created
/// * `Err(anyhow::Error)` - Directory, file, or symlink creation failed
///
/// # Errors
///
/// * I/O errors - Cannot create directory, files, or symlinks
fn create_function_directory(func_data: &FunctionData, by_name_dir: &Path, by_address_dir: &Path) -> Result<()> {
    // Create directory using function name in by-name subdirectory
    let func_dir = by_name_dir.join(&func_data.func_name);
    fs::create_dir_all(&func_dir)?;

    // Create individual files
    create_unoptimised_ir_file(func_data, &func_dir)?;
    create_optimised_ir_file(func_data, &func_dir)?;
    create_vcode_file(func_data, &func_dir)?;
    create_disasm_file(func_data, &func_dir)?;

    // Create symbolic link from address to function directory
    create_address_symlink(func_data.func_addr, &func_data.func_name, by_address_dir)?;

    println!(
        "Created: {} (symlink: {:#x})",
        func_dir.display(),
        func_data.func_addr
    );
    Ok(())
}

/// Main entry point for the function analyzer.
///
/// Parses command-line arguments, validates input, processes the log file,
/// and generates output files for each function found. Provides progress
/// information and error handling throughout the process.
///
/// # Process Flow
///
/// 1. Parse command-line arguments
/// 2. Validate that the input log file exists
/// 3. Determine output directory (user-specified or default)
/// 4. Create output directory if it doesn't exist
/// 5. Parse the log file to extract function data
/// 6. Generate individual directories in by-name and symlinks in by-address for each function
/// 7. Report completion statistics
///
/// # Returns
///
/// * `Ok(())` - Processing completed successfully
/// * `Err(anyhow::Error)` - Processing failed due to I/O or parsing errors
///
/// # Exit Codes
///
/// * `0` - Success
/// * `1` - Input file not found (exits before returning error)
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
        let log_file_stem = args
            .log_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("log");
        args.log_file
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}_functions", log_file_stem))
    });

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;
    println!("Output directory: {}", output_dir.display());

    // Create by-name and by-address subdirectories
    let by_name_dir = output_dir.join("by-name");
    let by_address_dir = output_dir.join("by-address");
    fs::create_dir_all(&by_name_dir)?;
    fs::create_dir_all(&by_address_dir)?;

    // Create directories for each function
    for func_data in &functions {
        create_function_directory(func_data, &by_name_dir, &by_address_dir)?;
    }

    println!();
    println!(
        "Processing complete. Created {} function directories in {}",
        functions.len(),
        output_dir.display()
    );

    Ok(())
}
