// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use object::Object;
use object::ObjectSymbol;
use rustc_demangle::demangle;

/// Used to collect profiling samples and build a profile
pub struct Sampler {
    /// Symbol resolver for mapping addresses to function names
    symbols: Symbols,
    /// Series of samples taken during the execution of the program
    samples: Vec<u64>,
}

impl Sampler {
    /// Create a new sampler with a given symbol table
    pub fn new(symbols: Symbols) -> Result<Self, Box<dyn Error>> {
        Ok(Sampler {
            symbols,
            samples: Vec::with_capacity(1000),
        })
    }

    /// Record a sample from the stepper in its current state
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `pc_ptr` points to a valid, initialised u64 value
    /// - The pointed-to value remains valid for the lifetime of this call
    pub unsafe fn sample_from_ptr(&mut self, pc_ptr: *const u64) {
        // SAFETY: Caller guarantees pc_ptr is valid and concurrent reads are safe
        let pc = unsafe { pc_ptr.read_volatile() };
        self.samples.push(u64::from_le(pc));
    }

    /// Build a profile by resolving the symbol corresponding to each sample
    /// and recording the number of samples which contain it
    pub fn finish(self, output_path: &Path) -> Result<usize, Box<dyn Error>> {
        let mut profile = HashMap::new();
        let n_samples = self.samples.len();

        for pc in self.samples {
            let symbol = self.symbols.resolve_address(pc).unwrap_or("<unknown>");
            profile
                .entry(symbol.to_string())
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }

        let file = File::create(output_path)?;
        let mut output_file = BufWriter::new(file);

        for (symbol, count) in profile.iter() {
            writeln!(output_file, "{symbol} {count}")?;
        }

        Ok(n_samples)
    }
}

/// Symbol table mapping from function start address to function name
pub struct Symbols {
    symbols: BTreeMap<u64, String>,
}

impl Symbols {
    /// Build the symbol table for the given binary
    pub fn new(binary: &[u8]) -> Result<Self, Box<dyn Error>> {
        let file = object::File::parse(binary)?;

        let mut symbols = BTreeMap::new();

        for symbol in file.symbols() {
            if let Ok(name) = symbol.name() {
                // Only include symbols that have a size and are functions
                if symbol.size() > 0 && symbol.kind() == object::SymbolKind::Text {
                    let start = symbol.address();

                    // Formatting to remove the crate disambiguator suffix
                    let demangled = format!("{:#}", demangle(name));

                    symbols.insert(start, demangled);
                }
            }
        }
        Ok(Symbols { symbols })
    }

    /// Resolve an address to a function name
    pub fn resolve_address(&self, address: u64) -> Option<&str> {
        // Find the largest address that is less than or equal to `address`
        self.symbols
            .range(..=address)
            .last()
            .map(|(_, name)| name.as_str())
    }
}
