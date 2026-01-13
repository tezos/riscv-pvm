// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::ops::ControlFlow;

use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::mode::Mode;
use perfect_derive::perfect_derive;

use crate::machine_state::MachineState;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::PageCache;
use crate::machine_state::registers::a0;
use crate::machine_state::registers::a7;

/// Reason for interrupting execution
pub enum BreakReason {
    /// The program wants to exit
    Exit(u64),

    /// An error occurred
    Error(String),
}

/// Posix execution environment state
#[perfect_derive(Default)]
pub struct PosixState<M: Mode> {
    _pd: std::marker::PhantomData<M>,
}

impl<M: Mode> PosixState<M> {
    /// Handle a POSIX system call. Returns `Ok(true)` if it makes sense to continue execution.
    pub fn handle_call<MC: MemoryConfig, PC: PageCache<MC, M>>(
        &mut self,
        machine: &mut MachineState<MC, PC, M>,
    ) -> ControlFlow<BreakReason>
    where
        M: AtomMode,
    {
        let handle_exit = |code| ControlFlow::Break(BreakReason::Exit(code));

        // Successful physical memory tests set
        //   a7 = 93 & a0 = 0
        // Successful virtual memory tests set
        //   a7 = 0 (a7 never gets set) & a0 = 1
        // Failed physical memory tests set
        //   a7 = 93 & a0 = 1 | (TEST_FAILED << 1)
        // Failed virtual memory tests set
        //   a7 = 0 (a7 never gets set) & a0 = 1 | (TEST_FAILED << 1)
        let a7_val = machine.core.hart.xregisters.read(a7);
        let a0_val = machine.core.hart.xregisters.read(a0);
        match (a7_val, a0_val) {
            // Exit (test pass, physical | virtual)
            (93, 0) | (0, 1) => handle_exit(0),

            // Exit (test fail, physical | virtual)
            (93, code) | (0, code) => handle_exit(code),

            // Unimplemented
            _ => ControlFlow::Break(BreakReason::Error(format!(
                "Unknown system call number {a7_val}"
            ))),
        }
    }
}
