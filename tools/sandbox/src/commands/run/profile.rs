// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Functions for profiling a run of the PVM stepper
//!
//! Produce a folded stacks profile for a given PVM stepper run. This is a simple
//! profile format which lists symbols (either individual symbols or stacks in
//! which symbols are separated by `;`) and the number of times they were sampled,
//! which can then be turned into a flamegraph using tools such as:
//! - <https://flamegraph.com>
//! - <https://github.com/brendangregg/FlameGraph>
//!
//! The profiler works by only sampling the pc of the running PVM stepper, so only
//! the currently running function is recorded. The full call stack information is
//! not yet recorded. The sampler runs in a separate thread and samples the PVM
//! stepper's pc at a regular, configurable interval without interrupting it.
//! The pc value is resolved to a symbol name using a symbol map built using
//! the DWARF information in the binary, which must be compiled in debug mode.

use std::error::Error;
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use octez_riscv::state_backend::owned_backend::Owned;
use octez_riscv::stepper::StepResult;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;

use super::sample::Sampler;
use super::sample::Symbols;

/// Run profiling on the given stepper
pub(crate) fn profile_stepper(
    mut stepper: impl Stepper<Manager = Owned>,
    program: &[u8],
    sample_interval: Duration,
    max_steps: Option<usize>,
    output_path: &Path,
) -> Result<usize, Box<dyn Error>> {
    let symbols = Symbols::new(program)?;

    let is_running = Arc::new(AtomicBool::new(true));
    let is_running_sampler = Arc::clone(&is_running);

    let output_path = output_path.to_path_buf();

    // Get a pointer to the pc, which the sampler thread can sample from. The pointer will
    // remain vaild throughout the lifetime of the thread.
    let pc_ptr = std::ptr::addr_of!(stepper.machine_state().hart.pc) as *const u64 as usize;

    // Spawn thread to run the sampler
    let worker = thread::spawn(move || -> Result<usize, String> {
        let mut sampler = Sampler::new(symbols).map_err(|e| e.to_string())?;

        let pc_ptr = pc_ptr as *const u64;

        loop {
            // Check if PVM stepper is still running
            if !is_running.load(Ordering::Acquire) {
                break;
            }

            thread::sleep(sample_interval);

            // SAFETY: pc_ptr points to the pc cell in the stepper, which is owned
            // by the main thread and remains valid.
            unsafe {
                sampler.sample_from_ptr(pc_ptr);
            }
        }

        let sample_count = sampler.finish(&output_path).map_err(|e| e.to_string())?;
        Ok(sample_count)
    });

    // Run the stepper until completion
    let step_limit = max_steps.map_or(Bound::Unbounded, Bound::Included);
    let result = stepper.step_max(step_limit);

    // Signal sampler thread to stop
    is_running_sampler.store(false, Ordering::Release);

    let steps = match result.to_stepper_status() {
        StepperStatus::Exited {
            success: true,
            steps,
            ..
        } => Ok(steps),
        result => Err(format!("{result:?}")),
    }?;

    let sample_count = worker.join().map_err(|_| "Sampler thread panicked")??;
    eprintln!("Samples collected: {sample_count}");

    Ok(steps)
}
