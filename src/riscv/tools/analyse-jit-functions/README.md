# JIT Function Analyser

A Rust executable for analysing JIT function log files and extracting function data.

## Features

- Parses JSON log files for function data
- Extracts unoptimised IR, optimised IR, and optimised ASM for each function
- Creates individual text files named after the function's hexadecimal address
- Processes all valid log entries with required function fields
- Configurable output directory

## Usage

```bash
# Build the project
cargo build --release --bin analyse-jit-functions

# Run with default output directory
./target/release/analyse-jit-functions log.json

# Run with custom output directory
./target/release/analyse-jit-functions log.json -o my_functions

# Show help
./target/release/analyse-jit-functions --help
```

## Output

For each function found in the log file, creates a text file named `0x{address}.txt` containing:

- Function name (hash)
- Function address (in hexadecimal)
- Unoptimised IR code
- Optimised IR code  
- Optimised ASM code
- Optimised machine code

## Dependencies

- anyhow: Error handling
- clap: Command-line argument parsing
- serde: JSON serialization/deserialization
- serde_json: JSON parsing

## Documentation

Generate and view the full API documentation:

```bash
cargo doc --bin analyse-jit-functions --open
```

The codebase includes comprehensive documentation with:
- Module-level overview and usage examples
- Detailed function and type documentation
- Error handling explanations
- File naming logic and processing flow

## Conversion from Python

This is a Rust port of the original Python script `scripts/analyze_functions.py`, providing:

- Better performance for large log files
- Strong typing and error handling
- Integration with the workspace build system
- No external Python dependencies required
- Comprehensive documentation and examples