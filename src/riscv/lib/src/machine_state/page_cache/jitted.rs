// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! JIT-compilation support for entrypoints in pages.

use super::INSTRUCTION_ENTRIES;
use super::PAGE_OFFSET_MASK;
use super::code_page_entry::CodePageEntry;
use crate::exceptions::Exception;
use crate::jit::state_access::ExceptionCode;
use crate::machine_state::MachineCoreState;
use crate::machine_state::StepManyResult;
use crate::machine_state::block_cache::block::dispatch::CodeDispatcher;
use crate::machine_state::block_cache::block::dispatch::DispatchCompiler;
use crate::machine_state::block_cache::block::dispatch::DispatchTarget;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::owned_backend::Owned;

/// Maximum number of instructions we pass to a compilation request
///
/// Doubles as a coarse upper-bound for the minimum number of steps required to safely dispatch
/// the entrypoints.
const MAX_INSTR_COMPILED: usize = 20;

/// A full-page of Jit-supporting entrypoints.
pub type JittedPage<D, MC> = [Jitted<D, MC>; INSTRUCTION_ENTRIES];

/// Entrypoints that are compiled to native code for execution, when possible & desirable.
///
/// Not all instructions are currently supported, when a sequence contains
/// unsupported instructions, a fallback to [`super::Interpreted`] mode may occur (if the
/// unsupported instruction is attempted for compilation).
#[derive(derive_more::Debug)]
pub struct Jitted<D: DispatchCompiler<MC>, MC: MemoryConfig> {
    instruction: Instruction,
    dispatch: DispatchTarget<[Self; INSTRUCTION_ENTRIES], D, MC>,
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> AsRef<Instruction> for Jitted<D, MC> {
    fn as_ref(&self) -> &Instruction {
        &self.instruction
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> CodeDispatcher<D, MC> for JittedPage<D, MC> {
    /// The default initial dispatcher for jit.
    ///
    /// This will run the entrypoint in interpreted mode by default, but may attempt to JIT-compile
    /// the block.
    ///
    /// # SAFETY
    ///
    /// The `compiler` must be the same every time this function is called.
    ///
    /// This ensures that the builder in question is guaranteed to be alive, for at least as long
    /// as this entrypoint may be run via [`CodePageEntry::run_entrypoint`].
    unsafe extern "C" fn run_block_interpreted(
        &mut self,
        core: &mut MachineCoreState<MC, Owned>,
        instr_pc: Address,
        max_steps: usize,
        result: &mut ExceptionCode,
        compiler: &mut D,
    ) -> usize {
        let offset = (instr_pc & PAGE_OFFSET_MASK) >> 1;

        if !compiler.should_compile(&mut self[offset as usize].dispatch) {
            // Safety: the compiler passed to this function is always the same for the
            // lifetime of the entrypoint
            return unsafe {
                self.run_block_not_compiled(core, instr_pc, max_steps, result, compiler)
            };
        }

        // trigger JIT compilation
        let instr = self
            .iter()
            .skip(offset as usize)
            .take(MAX_INSTR_COMPILED)
            .map(|entry| entry.instruction)
            .collect::<Vec<_>>();

        let fun = compiler.compile(&mut self[offset as usize].dispatch, instr, instr_pc);

        // Safety: the compiler passed to this function is always the same for the
        // lifetime of the entrypoint
        unsafe { (fun)(self, core, instr_pc, max_steps, result, compiler) }
    }

    /// Run a block where JIT-compilation has been attempted, but failed for any reason.
    ///
    /// # SAFETY
    ///
    /// The `compiler` must be the same every time this function is called.
    ///
    /// This ensures that the builder in question is guaranteed to be alive, for at least as long
    /// as this entrypoint may be run via [`CodePageEntry::run_entrypoint`].
    unsafe extern "C" fn run_block_not_compiled(
        &mut self,
        core: &mut MachineCoreState<MC, Owned>,
        instr_pc: Address,
        max_steps: usize,
        result: &mut ExceptionCode,
        _compiler: &mut D,
    ) -> usize {
        let block_result = super::run_code_page_interpreted(self, core, instr_pc, max_steps);

        *result = block_result
            .error
            .map(ExceptionCode::from_exception)
            .unwrap_or(ExceptionCode::NoException);

        block_result.steps
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> From<Instruction> for Jitted<D, MC> {
    fn from(instruction: Instruction) -> Self {
        Self {
            instruction,
            dispatch: DispatchTarget::default(),
        }
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> CodePageEntry<MC, Owned> for Jitted<D, MC> {
    type Compiler = D;

    /// Run from an entrypoint, using the currently selected dispatch mechanism
    ///
    /// # SAFETY
    ///
    /// The `compiler` must be the same every time this function is called.
    ///
    /// This ensures that the builder in question is guaranteed to be alive, for at least as long
    /// as this entrypoint may be run via [`CodePageEntry::run_entrypoint`].
    unsafe fn run_entrypoint(
        page: &mut [Self; INSTRUCTION_ENTRIES],
        core: &mut MachineCoreState<MC, Owned>,
        compiler: &mut Self::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception> {
        if max_steps < MAX_INSTR_COMPILED {
            return super::run_code_page_interpreted(page, core, instr_pc, max_steps);
        }

        // Since we know the instruction pc to always be halfword-aligned, there are half
        // as many entries as the page size.
        let instr_offset = (instr_pc & PAGE_OFFSET_MASK) >> 1;

        let entrypoint = &mut page[instr_offset as usize];

        let fun = entrypoint.dispatch.get();

        #[cfg(test)]
        entrypoint.dispatch.record_called();

        let mut result = ExceptionCode::NoException;

        // SAFETY: The compiler builder is always the same instance, guaranteeing that any JIT-compiled
        // function is still alive.
        let steps = unsafe { (fun)(page, core, instr_pc, max_steps, &mut result, compiler) };

        StepManyResult {
            steps,
            error: result.to_exception(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Jitted;
    use super::MAX_INSTR_COMPILED;
    use crate::array_utils::boxed_from_fn;
    use crate::exceptions::Exception;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::block_cache::block::InlineCompiler;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::page_cache::CodePageEntry;
    use crate::machine_state::page_cache::INSTRUCTION_ENTRIES;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;

    #[test]
    fn test_jitted_entrypoint_called() {
        let mut page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>(|| {
            Jitted::<_, M4K>::from(Instruction::new_nop(InstrWidth::Compressed))
        });

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        // Safety: we only ever use the above JIT instance
        let result = unsafe {
            CodePageEntry::run_entrypoint(&mut page, &mut core, &mut jit, 100, MAX_INSTR_COMPILED)
        };

        assert!(result.error.is_none());
        assert_eq!(result.steps, MAX_INSTR_COMPILED);

        assert_eq!(
            core.hart.pc.read(),
            100 + (MAX_INSTR_COMPILED as u64) * InstrWidth::Compressed as u64
        );

        assert_eq!(page[100 >> 1].dispatch.called_times(), 1);
    }

    #[test]
    fn test_interpreted_fallback_on_insufficient_steps() {
        let mut page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>(|| {
            Jitted::<_, M4K>::from(Instruction::new_nop(InstrWidth::Compressed))
        });

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        let max_steps = MAX_INSTR_COMPILED - 1;

        // Safety: we only ever use the above JIT instance
        let result = unsafe {
            CodePageEntry::run_entrypoint(&mut page, &mut core, &mut jit, 100, max_steps)
        };

        assert!(result.error.is_none());
        assert_eq!(result.steps, max_steps);

        assert_eq!(
            core.hart.pc.read(),
            100 + max_steps as u64 * InstrWidth::Compressed as u64
        );

        assert_eq!(page[100 >> 1].dispatch.called_times(), 0);
    }

    #[test]
    fn test_not_compiled_fallback_on_compilation_failure() {
        let mut page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>(|| {
            Jitted::<_, M4K>::from(Instruction::new_fence_i())
        });

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        let max_steps = MAX_INSTR_COMPILED;

        // Safety: we only ever use the above JIT instance
        let result = unsafe {
            CodePageEntry::run_entrypoint(&mut page, &mut core, &mut jit, 100, max_steps)
        };

        assert_eq!(result.error, Some(Exception::FenceI));
        assert_eq!(result.steps, 0);

        // we have attempted compilation
        assert_eq!(page[100 >> 1].dispatch.called_times(), 1);

        let info = format!("{:?}", page[100 >> 1].dispatch);
        assert!(
            info.contains("status: NotCompiled"),
            "unexpected status: \"{info}\""
        );
    }
}
