// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! PVM hooks
//!
//! Hooks are used to handle requests from the PVM inline without needing to suspend it.
//!
//! For example, the PVM might want to write to the debug log. This happens fairly frequently.
//! Hence, it is not ideal to suspend the PVM for every single write. Instead, we use the hooks to
//! handle each request inline (where it is raised) therefore saving the overhead of suspending the
//! PVM.

use std::io::Write;
use std::io::stdout;

use tezos_smart_rollup_utils::console::Console;

/// PVM hooks
pub trait PvmHooks {
    /// Write bytes to the debug output.
    fn write_debug_bytes(&mut self, bytes: &[u8]);

    /// Write a single byte to the debug output.
    fn write_debug_byte(&mut self, char: u8) {
        self.write_debug_bytes(&[char]);
    }
}

impl<H: PvmHooks> PvmHooks for &mut H {
    fn write_debug_byte(&mut self, char: u8) {
        H::write_debug_byte(self, char)
    }

    fn write_debug_bytes(&mut self, bytes: &[u8]) {
        H::write_debug_bytes(self, bytes)
    }
}

impl PvmHooks for Console<'_> {
    fn write_debug_bytes(&mut self, bytes: &[u8]) {
        self.write_all(bytes).unwrap();
    }
}

impl PvmHooks for Vec<u8> {
    fn write_debug_bytes(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// PVM hooks that write debug information to the standard output
pub struct StdoutDebugHooks;

impl PvmHooks for StdoutDebugHooks {
    fn write_debug_bytes(&mut self, bytes: &[u8]) {
        stdout().write_all(bytes).unwrap();
    }
}

/// Do nothing with the hooks
pub struct NoHooks;

impl PvmHooks for NoHooks {
    fn write_debug_byte(&mut self, _char: u8) {}

    fn write_debug_bytes(&mut self, _bytes: &[u8]) {}
}
