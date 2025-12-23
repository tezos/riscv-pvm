// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! JIT-compilation support for entrypoints in pages.

use std::sync::Arc;

use octez_riscv_data::mode::Normal;

use super::dispatch::DispatchCompiler;
use crate::exceptions::Exception;
use crate::jit::state_access::ExceptionCode;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::address_to_page_offset;
use crate::machine_state::page_cache::DispatchTarget;

/// A full page of entrypoints.
pub type Page<D, MC> = Arc<super::state::PageEntry<D, MC>>;

/// Entrypoints that are compiled to native code for execution, when possible & desirable.
///
/// Not all instructions are currently supported, when a sequence contains unsupported
/// instructions, a fallback to interpreted mode may occur (if the unsupported instruction is
/// attempted for compilation).
#[derive(derive_more::Debug)]
pub struct Entrypoint<D, MC> {
    instruction: Instruction,
    pub(crate) dispatch: DispatchTarget<D, MC>,
}

impl<D, MC> AsRef<Instruction> for Entrypoint<D, MC> {
    fn as_ref(&self) -> &Instruction {
        &self.instruction
    }
}

impl<D, MC> Entrypoint<D, MC> {
    /// The default initial dispatcher for jit.
    ///
    /// This will run the entrypoint in interpreted mode by default, but may attempt to JIT-compile
    /// the block.
    pub(super) extern "C" fn run_entrypoint_interpreted(
        page: &Page<D, MC>,
        core: &mut MachineCoreState<MC, Normal>,
        instr_pc: Address,
        max_steps: usize,
        result: &mut ExceptionCode,
    ) -> usize
    where
        D: DispatchCompiler<MC>,
        MC: MemoryConfig,
    {
        match D::compile(page, instr_pc) {
            None => Self::run_entrypoint_not_compiled(page, core, instr_pc, max_steps, result),
            // SAFETY: the compiler is still alive (`page` has a reference) so the function pointer is
            // safe to call
            Some(fun) => unsafe { (fun)(page, core, instr_pc, max_steps, result) },
        }
    }

    /// Dispatch an entrypoint where JIT-compilation has been attempted, but failed for any reason.
    pub(super) extern "C" fn run_entrypoint_not_compiled(
        page: &Page<D, MC>,
        core: &mut MachineCoreState<MC, Normal>,
        instr_pc: Address,
        max_steps: usize,
        result: &mut ExceptionCode,
    ) -> usize
    where
        MC: MemoryConfig,
    {
        let block_result =
            super::run_code_page_interpreted(&page.entries, core, instr_pc, max_steps);

        *result = block_result
            .error
            .map(ExceptionCode::from_exception)
            .unwrap_or(ExceptionCode::NoException);

        block_result.steps
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> From<Instruction> for Entrypoint<D, MC> {
    fn from(instruction: Instruction) -> Self {
        Self {
            instruction,
            dispatch: DispatchTarget::default(),
        }
    }
}

impl<D: Clone + DispatchCompiler<MC>, MC: MemoryConfig> Entrypoint<D, MC> {
    /// Run from an entrypoint, using the currently selected dispatch mechanism
    pub fn run_entrypoint(
        page: &Page<D, MC>,
        core: &mut MachineCoreState<MC, Normal>,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception> {
        let page_offset = address_to_page_offset(instr_pc);

        // Since we know the instruction pc to always be halfword-aligned, there are half
        // as many entries as the page size.
        let instr_offset = page_offset >> 1;

        let entrypoint = &page.entries[instr_offset];

        let fun = entrypoint.dispatch.get();

        let mut result = ExceptionCode::NoException;

        #[cfg(test)]
        entrypoint.dispatch.jit_counters().record_entrypoint_call();

        // SAFETY: the compiler which was used to compile `fun` is still alive (`page` has
        // a reference) so the function pointer is safe to call
        let steps = unsafe { (fun)(page, core, instr_pc, max_steps, &mut result) };

        StepManyResult {
            steps,
            error: result.to_exception(),
        }
    }

    #[cfg(test)]
    pub fn call_statistics(&self) -> &super::dispatch::jit_counters::JitTestCounters {
        self.dispatch.jit_counters()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;

    use super::Entrypoint;
    use crate::exceptions::Exception;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::page_cache::InlineCompiler;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::parser::instruction::InstrWidth;

    /// An arbitrary number of steps to use for testing.
    const DEFAULT_TEST_MAX_STEPS: usize = 40;

    #[test]
    fn test_jitted_entrypoint_called() {
        let Ok(page) = PageEntry::<InlineCompiler, M4K>::new::<std::convert::Infallible>(
            InlineCompiler::default,
            |_| Ok(Instruction::new_nop(InstrWidth::Compressed)),
        );

        let mut core = MachineCoreState::default();

        let result = Entrypoint::run_entrypoint(&page, &mut core, 100, DEFAULT_TEST_MAX_STEPS);

        assert!(result.error.is_none());
        assert_eq!(result.steps, DEFAULT_TEST_MAX_STEPS);

        assert_eq!(
            core.hart.pc.read(),
            100 + (DEFAULT_TEST_MAX_STEPS as u64) * InstrWidth::Compressed as u64
        );

        assert_eq!(page.entries[100 >> 1].dispatch.num_jit_calls(), 1);
    }

    proptest::proptest! {
        #[test]
        fn test_interpreted_fallback_on_insufficient_steps(
            max_steps in 0usize..(DEFAULT_TEST_MAX_STEPS * 2)
        ) {
            let Ok(page) = PageEntry::<InlineCompiler, M4K>::new::<std::convert::Infallible>(
                InlineCompiler::default,
                |_| Ok(Instruction::new_nop(InstrWidth::Compressed)),
            );

            let mut core = MachineCoreState::default();

            let expected_steps = max_steps;
            let start_pc = 100;
            let start_entry = (start_pc >> 1) as usize;

            // Safety: we only ever use the above JIT instance
            let result = Entrypoint::run_entrypoint(&page, &mut core, start_pc, max_steps);

            prop_assert_eq!(result.error, None);
            prop_assert_eq!(result.steps, expected_steps);

            prop_assert_eq!(
                core.hart.pc.read(),
                start_pc + (expected_steps as u64) * InstrWidth::Compressed as u64
            );

            prop_assert_eq!(page.entries[start_entry].dispatch.num_jit_calls(), 1);
        }
    }

    #[test]
    fn test_not_compiled_fallback_on_compilation_failure() {
        let Ok(page) = PageEntry::<InlineCompiler, M4K>::new::<std::convert::Infallible>(
            InlineCompiler::default,
            |_| Ok(Instruction::new_fence_i()),
        );

        let mut core = MachineCoreState::default();

        let max_steps = DEFAULT_TEST_MAX_STEPS;

        let result = Entrypoint::run_entrypoint(&page, &mut core, 100, max_steps);

        assert_eq!(result.error, Some(Exception::FenceI));
        assert_eq!(result.steps, 0);

        // Fence is not JIT-compiled, so we will not enter any JIT function.
        assert_eq!(page.entries[100 >> 1].dispatch.num_jit_calls(), 0);

        let info = format!("{:?}", page.entries[100 >> 1].dispatch);
        assert!(
            info.contains("status: NotCompiled"),
            "unexpected status: \"{info}\""
        );
    }

    /// The compilation we send to the jit must be formed of continguous instructions in memory,
    /// not contiguous instructions in the page cache entry.
    ///
    /// This is because the page cache entry contains entrypoints for _every_ half-word.
    ///
    /// Therefore, naively sending just a 'slice' of entries to the JIT will include an extra
    /// instruction (ie the upper half word) whenever there is an uncompressed instruction in the
    /// request.
    #[test]
    fn test_compilation_request_respects_instruction_width() {
        let Ok(page) = PageEntry::<InlineCompiler, M4K>::new::<std::convert::Infallible>(
            InlineCompiler::default,
            |index| {
                let instruction = if index % 2 == 0 {
                    Instruction::new_nop(InstrWidth::Uncompressed)
                } else {
                    Instruction::new_unknown(InstrWidth::Compressed)
                };
                Ok(instruction)
            },
        );

        let mut core = MachineCoreState::default();

        // Run Noops only
        let result = Entrypoint::run_entrypoint(&page, &mut core, 0, DEFAULT_TEST_MAX_STEPS);

        assert!(result.error.is_none());
        assert_eq!(result.steps, DEFAULT_TEST_MAX_STEPS);

        assert_eq!(
            core.hart.pc.read(),
            (DEFAULT_TEST_MAX_STEPS as u64) * InstrWidth::Uncompressed as u64
        );

        assert_eq!(page.entries[0].dispatch.num_jit_calls(), 1);

        // This time, start with a `Unknown` instruction
        let result = Entrypoint::run_entrypoint(&page, &mut core, 2, DEFAULT_TEST_MAX_STEPS);

        assert_eq!(result.error, Some(Exception::IllegalInstruction));
        assert_eq!(result.steps, 0);

        assert_eq!(core.hart.pc.read(), 2);

        assert_eq!(page.entries[2 >> 1].dispatch.num_jit_calls(), 1);
    }
}
