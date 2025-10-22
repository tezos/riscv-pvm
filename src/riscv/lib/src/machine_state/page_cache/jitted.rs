// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! JIT-compilation support for entrypoints in pages.

use std::sync::Arc;

use super::INSTRUCTION_ENTRIES;
use super::code_page_entry::CodePageEntry;
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
use crate::state_backend::owned_backend::Owned;

/// Maximum number of instructions we pass to a compilation request
///
/// Doubles as a coarse upper-bound for the minimum number of steps required to safely dispatch
/// the entrypoints.
pub(crate) const MAX_INSTR_COMPILED: usize = 40;

/// A full-page of Jit-supporting entrypoints.
pub type JittedPage<D, MC> = Arc<[Jitted<D, MC>; INSTRUCTION_ENTRIES]>;

/// Entrypoints that are compiled to native code for execution, when possible & desirable.
///
/// Not all instructions are currently supported, when a sequence contains
/// unsupported instructions, a fallback to [`super::Interpreted`] mode may occur (if the
/// unsupported instruction is attempted for compilation).
#[derive(derive_more::Debug)]
pub struct Jitted<D: DispatchCompiler<MC>, MC: MemoryConfig> {
    instruction: Instruction,
    pub(super) dispatch: DispatchTarget<D, MC>,
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> AsRef<Instruction> for Jitted<D, MC> {
    fn as_ref(&self) -> &Instruction {
        &self.instruction
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> Jitted<D, MC> {
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
    pub(super) unsafe extern "C" fn run_entrypoint_interpreted(
        page: &Arc<[Self; INSTRUCTION_ENTRIES]>,
        core: &mut MachineCoreState<MC, Owned>,
        instr_pc: Address,
        max_steps: usize,
        result: &mut ExceptionCode,
        compiler: &mut D,
    ) -> usize {
        let page_offset = address_to_page_offset(instr_pc);

        // instr_pc is always halfword aligned
        let offset = page_offset >> 1;

        if !compiler.should_compile(&page[offset].dispatch) {
            // Safety: the compiler passed to this function is always the same for the
            // lifetime of the entrypoint
            return unsafe {
                Self::run_entrypoint_not_compiled(page, core, instr_pc, max_steps, result, compiler)
            };
        }

        let fun = compiler.compile(page, instr_pc);

        // Safety: the compiler passed to this function is always the same for the
        // lifetime of the entrypoint
        unsafe { (fun)(page, core, instr_pc, max_steps, result, compiler) }
    }

    /// Dispatch an entrypoint where JIT-compilation has been attempted, but failed for any reason.
    ///
    /// # SAFETY
    ///
    /// The `compiler` must be the same every time this function is called.
    ///
    /// This ensures that the builder in question is guaranteed to be alive, for at least as long
    /// as this entrypoint may be run via [`CodePageEntry::run_entrypoint`].
    pub(super) unsafe extern "C" fn run_entrypoint_not_compiled(
        page: &Arc<[Self; INSTRUCTION_ENTRIES]>,
        core: &mut MachineCoreState<MC, Owned>,
        instr_pc: Address,
        max_steps: usize,
        result: &mut ExceptionCode,
        _compiler: &mut D,
    ) -> usize {
        let block_result = super::run_code_page_interpreted(page, core, instr_pc, max_steps);

        *result = block_result
            .error
            .map(ExceptionCode::from_exception)
            .unwrap_or(ExceptionCode::NoException);

        block_result.steps
    }

    /// Returns up to [MAX_INSTR_COMPILED] instructions, that would be contiguous in memory,
    /// starting from the page offset given by `instr_pc`.
    ///
    /// These instructions can be passed to the JIT compiler for entrypoint dispatch optimisation.
    pub(super) fn compilation_request_instructions(
        page: &[Self; INSTRUCTION_ENTRIES],
        instr_pc: Address,
    ) -> Vec<Instruction> {
        let page_offset = address_to_page_offset(instr_pc);

        // instr_pc is always halfword aligned
        let mut offset = page_offset >> 1;

        let mut instructions = Vec::with_capacity(MAX_INSTR_COMPILED);
        while offset < INSTRUCTION_ENTRIES && instructions.len() < MAX_INSTR_COMPILED {
            let entry = &page[offset];

            offset += (entry.instruction.width() as usize) >> 1;

            instructions.push(entry.instruction);
        }

        instructions
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
        page: &Arc<[Self; INSTRUCTION_ENTRIES]>,
        core: &mut MachineCoreState<MC, Owned>,
        compiler: &mut Self::Compiler,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<Exception> {
        if max_steps < MAX_INSTR_COMPILED {
            return super::run_code_page_interpreted(page, core, instr_pc, max_steps);
        }

        let page_offset = address_to_page_offset(instr_pc);

        // Since we know the instruction pc to always be halfword-aligned, there are half
        // as many entries as the page size.
        let instr_offset = page_offset >> 1;

        let entrypoint = &page[instr_offset];

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
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::page_cache::CodePageEntry;
    use crate::machine_state::page_cache::INSTRUCTION_ENTRIES;
    use crate::machine_state::page_cache::InlineCompiler;
    use crate::parser::instruction::InstrWidth;
    use crate::state::NewState;

    #[test]
    fn test_jitted_entrypoint_called() {
        let page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>(|| {
            Jitted::<_, M4K>::from(Instruction::new_nop(InstrWidth::Compressed))
        })
        .into();

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        // Safety: we only ever use the above JIT instance
        let result = unsafe {
            CodePageEntry::run_entrypoint(&page, &mut core, &mut jit, 100, MAX_INSTR_COMPILED)
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
        let page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>(|| {
            Jitted::<_, M4K>::from(Instruction::new_nop(InstrWidth::Compressed))
        })
        .into();

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        let max_steps = MAX_INSTR_COMPILED - 1;

        // Safety: we only ever use the above JIT instance
        let result =
            unsafe { CodePageEntry::run_entrypoint(&page, &mut core, &mut jit, 100, max_steps) };

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
        let page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>(|| {
            Jitted::<_, M4K>::from(Instruction::new_fence_i())
        })
        .into();

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        let max_steps = MAX_INSTR_COMPILED;

        // Safety: we only ever use the above JIT instance
        let result =
            unsafe { CodePageEntry::run_entrypoint(&page, &mut core, &mut jit, 100, max_steps) };

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
        let page = boxed_from_fn::<_, INSTRUCTION_ENTRIES>({
            let mut index = 0;
            move || {
                let instruction = if index % 2 == 0 {
                    Instruction::new_nop(InstrWidth::Uncompressed)
                } else {
                    Instruction::new_unknown(InstrWidth::Compressed)
                };
                index += 1;
                Jitted::<_, M4K>::from(instruction)
            }
        })
        .into();

        let mut jit = InlineCompiler::default();
        let mut core = MachineCoreState::new();

        // Run Noops only

        // Safety: we only ever use the above JIT instance
        let result = unsafe {
            CodePageEntry::run_entrypoint(&page, &mut core, &mut jit, 0, MAX_INSTR_COMPILED)
        };

        assert!(result.error.is_none());
        assert_eq!(result.steps, MAX_INSTR_COMPILED);

        assert_eq!(
            core.hart.pc.read(),
            (MAX_INSTR_COMPILED as u64) * InstrWidth::Uncompressed as u64
        );

        assert_eq!(page[0].dispatch.called_times(), 1);

        // This time, start with a `Unknown` instruction

        // Safety: we only ever use the above JIT instance
        let result = unsafe {
            CodePageEntry::run_entrypoint(&page, &mut core, &mut jit, 2, MAX_INSTR_COMPILED)
        };

        assert_eq!(result.error, Some(Exception::IllegalInstruction));
        assert_eq!(result.steps, 0);

        assert_eq!(core.hart.pc.read(), 2);

        assert_eq!(page[2 >> 1].dispatch.called_times(), 1);
    }
}
