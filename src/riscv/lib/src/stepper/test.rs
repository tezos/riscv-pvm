// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod interpreter;
mod posix;

use std::collections::BTreeMap;
use std::ops::Bound;

use derive_more::Error;
use derive_more::From;
use posix::PosixState;

use super::StepResult;
use super::Stepper;
use super::StepperStatus;
use crate::exceptions::Exception;
use crate::kernel_loader;
use crate::machine_state::MachineCoreState;
use crate::machine_state::MachineError;
use crate::machine_state::MachineState;
use crate::machine_state::StepManyResult;
use crate::machine_state::memory::M1G;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::Permissions;
use crate::machine_state::page_cache::CodePageEntry;
use crate::machine_state::page_cache::Interpreted;
use crate::machine_state::registers;
use crate::program::Program;
use crate::state::NewState;
use crate::state_backend::owned_backend::Owned;

#[derive(Clone, Debug)]
pub enum TestStepperResult {
    /// Execution has not finished. Returns the number of steps executed.
    Running { steps: usize },
    /// Program exited. Returns exit code and number of steps executed.
    Exit { code: usize, steps: usize },
    /// Execution finished because an unhandled exception has been thrown.
    /// Returns exception and number of steps executed.
    Exception {
        cause: Exception,
        steps: usize,
        message: Option<String>,
    },
}

impl Default for TestStepperResult {
    fn default() -> Self {
        Self::Running { steps: 0 }
    }
}

impl StepResult for TestStepperResult {
    fn to_stepper_status(&self) -> StepperStatus {
        match self {
            Self::Running { steps } => StepperStatus::Running { steps: *steps },
            Self::Exit { code, steps } => StepperStatus::Exited {
                steps: *steps,
                success: *code == 0,
                status: format!("code {code}"),
            },
            Self::Exception {
                cause,
                steps,
                message,
            } => StepperStatus::Errored {
                steps: *steps,
                cause: format!("{cause:?}"),
                message: message.as_deref().unwrap_or("<no message>").to_owned(),
            },
        }
    }
}

#[derive(Debug, From, Error, derive_more::Display)]
pub enum TestStepperError {
    KernelLoadingError(kernel_loader::Error),
    MachineError(MachineError),
}

pub struct TestStepper<
    MC: MemoryConfig = M1G,
    CPE: CodePageEntry<MC, Owned> = Interpreted<MC, Owned>,
> {
    machine_state: MachineState<MC, CPE, Owned>,
    posix_state: PosixState<Owned>,
}

impl<MC: MemoryConfig, CPE: CodePageEntry<MC, Owned>> TestStepper<MC, CPE> {
    /// Initialise an interpreter with a given `program`.
    #[inline]
    pub fn new(program: &[u8], compiler: CPE::Compiler) -> Result<Self, TestStepperError> {
        Ok(Self::new_with_parsed_program(program, compiler)?.0)
    }

    /// Consumes the stepper, returning the [`Compiler`] used internally.
    ///
    /// This allows the compiler to be re-used with a second stepper.
    ///
    /// [`Compiler`]: CodePageEntry::Compiler
    pub fn recover_builder(self) -> CPE::Compiler {
        self.machine_state.compiler
    }

    /// Initialise an interpreter with a given `program`. Returns both the interpreter and the fully
    /// parsed program.
    #[inline]
    pub fn new_with_parsed_program(
        program: &[u8],
        compiler: CPE::Compiler,
    ) -> Result<(Self, BTreeMap<u64, String>), TestStepperError> {
        let mut stepper = Self {
            posix_state: PosixState::<Owned>::new(),
            machine_state: MachineState::new(compiler),
        };

        // The interpreter needs a program to run.
        let elf_program = Program::<MC>::from_elf(program)?;

        stepper.machine_state.setup_boot_program(&elf_program)?;

        // Set booting Hart ID (a0) to 0
        stepper
            .machine_state
            .core
            .hart
            .xregisters
            .write(registers::a0, 0);

        Ok((stepper, elf_program.parsed()))
    }

    /// Allows to override permissions for the entirety of memory.
    ///
    /// For certain tests, relying on permissions as given by the program headers
    /// may not be sufficient. For these tests, it's required to be able to
    /// set permissions more loosely.
    #[cfg(test)]
    pub fn set_all_read_write_exec(&mut self) {
        let (main_memory, listener) = self.machine_state.memory_with_listener();
        main_memory
            .protect_pages(0, MC::TOTAL_BYTES, Permissions::READ_WRITE_EXEC, listener)
            .unwrap();
    }

    fn handle_step_result(
        &mut self,
        result: StepManyResult<posix::BreakReason>,
    ) -> TestStepperResult {
        match result.error {
            // An error was encountered in the evaluation function.
            Some(posix::BreakReason::Error(error)) => TestStepperResult::Exception {
                cause: Exception::EnvCall,
                steps: result.steps,
                message: Some(error),
            },

            // An exit was requested in the evaluation function.
            Some(posix::BreakReason::Exit(code)) => TestStepperResult::Exit {
                code: code as usize,
                steps: result.steps,
            },

            // Evaluation function returned without error.
            None => TestStepperResult::Running {
                steps: result.steps,
            },
        }
    }
}

impl<MC: MemoryConfig, CPE: CodePageEntry<MC, Owned>> Stepper for TestStepper<MC, CPE> {
    type MemoryConfig = MC;

    type Manager = Owned;

    #[inline(always)]
    fn machine_state(&self) -> &MachineCoreState<Self::MemoryConfig, Self::Manager> {
        &self.machine_state.core
    }

    type StepResult = TestStepperResult;

    fn step_max(&mut self, steps: Bound<usize>) -> Self::StepResult {
        let result = self.machine_state.step_max_handle(steps, |machine_state| {
            self.posix_state.handle_call(machine_state)
        });
        self.handle_step_result(result)
    }
}
