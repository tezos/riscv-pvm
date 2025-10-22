// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! A JIT library for compilation of sequences (or blocks) of RISC-V
//! instructions to native code.

pub(crate) mod builder;
pub(crate) mod state_access;
use std::collections::HashMap;
use std::ffi::c_void;

use cranelift::codegen::CodegenError;
use cranelift::codegen::settings::SetError;
use cranelift::frontend::FunctionBuilderContext;
use cranelift::prelude::*;
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_module::Linkage;
use cranelift_module::Module;
use cranelift_module::ModuleError;
use thiserror::Error;

use crate::jit::builder::sequence::SequenceBuilder;
use crate::jit::state_access::ExceptionCode;
use crate::log;
use crate::machine_state::MachineCoreState;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::hash::Hash;
use crate::state_backend::owned_backend::Owned;

/// Alias for the function signature produced by the JIT compilation.
///
/// This must have the same Abi as [`DispatchFn`], which is used by
/// the entrypoint dispatch mechanism in the page cache.
///
/// The JitFn does not inspect the first and last parameters here, however.
/// These parameters are needed by the initial dispatch mechanism to enable
/// JIT-compilation & hot-swapping. To avoid over-specifying these parameters here
/// (which can among other things cause type-checking issues), we replace the parameters
/// with pointers to `c_void` - which in the C abi map to the same parameter type as the
/// thin-references to the actual variables passed.
///
/// It also does not inspect the third parameter as it is hard-coded in the sequence building.
///
/// [`DispatchFn`]: crate::machine_state::page_cache::dispatch::DispatchFn
pub type JitFn<MC> = unsafe extern "C" fn(
    // ignored
    *const c_void,
    &mut MachineCoreState<MC, Owned>,
    // ignored
    u64,
    usize,
    &mut ExceptionCode,
    // ignored
    *const c_void,
    // TODO: RV-751 - Move the unused parameters for the JIT function to the end.
) -> usize;

/// Errors that may arise from the initialisation of the JIT.
#[derive(Debug, Error)]
pub enum JitError {
    /// Failures setting flags.
    #[error("Failed to set flag {0}")]
    Setting(#[from] SetError),
    /// Native compilation unsupported on the current arch/os.
    #[error("Native platform unsupported: {0}")]
    UnsupportedPlatform(&'static str),
    /// Constructing the Cranelift builder failed.
    #[error("Unable to initialise builder {0}")]
    BuilderFailure(#[from] CodegenError),
    /// Unable to register external state access functionality.
    #[error("Unable to register external state access functions: {0}")]
    JsaRegistration(#[from] Box<ModuleError>),
}

/// The JIT is responsible for compiling blocks of instructions to machine code,
/// returning a function that can be run over the [`MachineCoreState`].
pub struct JIT<MC: MemoryConfig> {
    /// The function builder context, which is reused across multiple
    /// [`FunctionBuilder`] instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: JITModule,

    /// Cache of compilation results.
    cache: HashMap<Hash, Option<JitFn<MC>>>,
}

impl<MC: MemoryConfig> JIT<MC> {
    /// Create a new instance of the JIT, which will be able to
    /// produce functions that can be run over the current
    /// memory configuration and manager.
    pub fn new() -> Result<Self, JitError> {
        if std::mem::size_of::<usize>() != std::mem::size_of::<u64>() {
            return Err(JitError::UnsupportedPlatform(
                "octez-riscv JIT only supports 64-bit architectures",
            ));
        }

        let mut flag_builder = settings::builder();

        // Optimization level for generated code. Configured to Generate the fastest possible code
        // https://docs.wasmtime.dev/api/cranelift/prelude/settings/struct.Flags.html#method.opt_level
        flag_builder.set("opt_level", "speed")?;

        let isa_builder = cranelift_native::builder().map_err(JitError::UnsupportedPlatform)?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: codegen::Context::new(),
            module,
            cache: Default::default(),
        })
    }

    /// Compile a sequence of instructions to a callable native function.
    ///
    /// Not all instructions are currently supported. For blocks containing
    /// unsupported instructions, `None` will be returned.
    pub fn compile(
        &mut self,
        instr: &[Instruction],
        program_counter: Address,
    ) -> Option<JitFn<MC>> {
        let Ok(hash) = Hash::blake3_hash((instr, program_counter)) else {
            return None;
        };

        if let Some(compilation_result) = self.cache.get(&hash) {
            return *compilation_result;
        }

        let mut builder = self.start(program_counter);
        let mut lowered_instrs = Vec::with_capacity(instr.len());

        // Check if the opcode of the instruction is supported in JIT and stop compilation in JIT if not.
        for i in instr {
            let Some(lower) = i.opcode.to_lowering() else {
                builder.abandon();
                self.clear();
                self.cache.insert(hash, None);
                return None;
            };

            let mut instr_builder = builder.build_next_instruction(i.width());

            let instr_result = unsafe {
                // # SAFETY: lower is called with args from the same instruction that it
                // was derived
                (lower)(i.args(), &mut instr_builder)
            };

            let lowered_instr = instr_builder.finish(instr_result);
            lowered_instrs.push(lowered_instr);
        }

        if lowered_instrs.is_empty() {
            builder.abandon();
            self.clear();
            return None;
        }

        builder.finish(&lowered_instrs);

        self.produce_function(&hash)
            .inspect_err(|error| {
                let opcodes = instr.iter().map(|i| i.opcode).collect::<Vec<_>>();
                log::warning!("Failed to compile {:?}: {:?}", opcodes, error);
            })
            .ok()
    }

    /// Start building a new sequence of instructions.
    fn start(&mut self, program_counter: Address) -> SequenceBuilder<'_, MC> {
        SequenceBuilder::new(
            &mut self.module,
            &mut self.ctx,
            &mut self.builder_context,
            program_counter,
        )
    }

    /// Finalise and cache the function under construction.
    fn produce_function(&mut self, hash: &Hash) -> Result<JitFn<MC>, Box<ModuleError>> {
        let name = hex::encode(hash);

        let fun = self.finalise(&name)?;

        self.cache.insert(*hash, Some(fun));

        Ok(fun)
    }

    /// Finalise the function currently under construction.
    fn finalise(&mut self, name: &str) -> Result<JitFn<MC>, Box<ModuleError>> {
        let id = self.module.declare_function(
            name.as_ref(),
            Linkage::Export,
            &self.ctx.func.signature,
        )?;

        // The context is currently in the state where we left it. That means it contains the IR
        // that we produced without any modifications or optimisations.
        #[cfg(feature = "log")]
        let unoptimised_ir = self.ctx.func.display().to_string();

        // Request a textual representation of the host assembly. It appears the "disassembly" is
        // actually Cranelift's VCode which contains additional information which we won't find if
        // we disassemble the native machine code. Hence it is useful to include.
        #[cfg(feature = "log")]
        self.ctx.set_disasm(true);

        // Populate the function in the JIT module.
        self.module.define_function(id, &mut self.ctx)?;

        // Finalise the definitions, ensuring that everything is ready to be called.
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(id);

        // The `define_function` will indirectly call `Context::optimize` to optimise the IR
        // representation. This allows us to extract the optimised IR. The native assembly is also
        // produced as part of the definition process as that triggers a compilation.
        //
        // We keep the expressions scoped to the macro to avoid evaluating them when logging is
        // turned off.
        log::trace!(
            func_name = name,
            func_addr = code as usize,
            unoptimised_ir,
            optimised_ir = self.ctx.func.display().to_string(),
            vcode = {
                self.ctx
                    .compiled_code()
                    .and_then(|code| code.vcode.as_deref())
                    .unwrap_or("<no vcode>")
            },
            machine_code = {
                let data = self
                    .ctx
                    .compiled_code()
                    .map(|code| code.buffer.data())
                    .unwrap_or(&[]);
                hex::encode(data)
            },
            "Defined function"
        );

        self.clear();

        // SAFETY: the signature of a JitFn matches exactly the abi we specified in the
        //         entry block. Compilation has succeeded & therefore this produced code
        //         is safe to call.
        Ok(unsafe { std::mem::transmute::<*const u8, JitFn<MC>>(code) })
    }

    /// Clear the current context to allow a new function to be compiled
    fn clear(&mut self) {
        self.module.clear_context(&mut self.ctx)
    }
}

// TODO: https://linear.app/tezos/issue/RV-496
//       `Block::BlockBuilder` should not require Default, as it
//         does not allow for potential fallilibility
impl<MC: MemoryConfig> Default for JIT<MC> {
    fn default() -> Self {
        Self::new().expect("JIT is supported on all octez-riscv supported platforms")
    }
}

#[cfg(test)]
mod tests {
    use std::ops::BitAnd;
    use std::ops::BitOr;
    use std::ops::BitXor;
    use std::ptr::null;

    use Instruction as I;
    use proptest::prelude::proptest;
    use rustc_apfloat::Float;
    use rustc_apfloat::Round;
    use rustc_apfloat::ieee::Double;

    use super::*;
    use crate::exceptions::Exception;
    use crate::instruction_context::LoadStoreWidth;
    use crate::interpreter::float::RoundingMode;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::MachineState;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::memory::PAGE_SIZE;
    use crate::machine_state::memory::listener::NoopMemoryGovernanceListener;
    use crate::machine_state::page_cache::InlineCompiler;
    use crate::machine_state::page_cache::Interpreted;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::page_cache::Jitted;
    use crate::machine_state::page_cache::jitted::MAX_INSTR_COMPILED;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::machine_state::registers::FValue;
    use crate::machine_state::registers::NonZeroXRegister;
    use crate::machine_state::registers::XRegister;
    use crate::machine_state::registers::nz;
    use crate::parser::instruction::InstrRoundingMode;
    use crate::parser::instruction::InstrWidth;
    use crate::parser::instruction::InstrWidth::*;
    use crate::state::NewState;
    use crate::state_backend::FnManagerIdent;

    type SetupHook = dyn Fn(&mut MachineCoreState<M4K, Owned>);
    type AssertHook = dyn Fn(&MachineCoreState<M4K, Owned>);

    /// Machine state for test scenarios with a configurable [`Block`] type.
    type TestMachineState<CPE> = MachineState<M4K, CPE, Owned>;

    enum ScenarioSteps {
        /// Steps equal to the instruction sequence.
        Sequence,

        /// Steps equal to `MAX_INSTR_COMPILED`.
        Max,

        /// A specific expected step count.
        Specific(usize),
    }

    struct Scenario {
        initial_pc: Option<u64>,
        expected_steps: ScenarioSteps,
        instructions: Vec<Instruction>,
        setup_hook: Option<Box<SetupHook>>,
        xregisters: Vec<(NonZeroXRegister, u64)>,
        assert_hook: Option<Box<AssertHook>>,
        expected_exception: Option<Exception>,
    }

    impl Scenario {
        fn simple(instructions: &[Instruction]) -> Self {
            Scenario {
                initial_pc: None,
                expected_steps: ScenarioSteps::Sequence,
                instructions: instructions.to_vec(),
                setup_hook: None,
                xregisters: vec![],
                assert_hook: None,
                // TODO: RV-803:
                //     due to the overestimate we are forced to make in terms of budget checks,
                //     we expect most scenarios to move past the end of the instructions and encounter
                //     this exception when running `Unknown`.
                //
                //     We should revert to `None` once this is tackled
                expected_exception: Some(Exception::IllegalInstruction),
            }
        }

        fn check_compilable(&self) {
            // Ensure the set of instructions can be compiled in JIT.
            let mut test_jit = JIT::<M4K>::new().unwrap();
            test_jit
                .compile(&self.instructions, self.initial_pc.unwrap_or_default())
                .expect("JIT compilation should succeed.");
        }

        /// Run a test scenario in three modes:
        ///
        /// A) With the `xregisters` preset (as if in a `setup_hook`)
        /// B) With the `xregisters` set in `Compressed` instructions
        /// C) With the `xregisters` set in `Uncompressed` instructions
        fn run(mut self) {
            self.run_inner();

            if self.xregisters.is_empty() {
                return;
            }

            let instr_count = self.instructions.len() + self.xregisters.len();
            let mut add_compressed = Vec::with_capacity(instr_count);
            let mut add_uncompressed = Vec::with_capacity(instr_count);

            if let ScenarioSteps::Specific(exp_steps) = self.expected_steps {
                let new = exp_steps + self.xregisters.len();
                assert!(new <= MAX_INSTR_COMPILED);
                self.expected_steps = ScenarioSteps::Specific(new);
            }

            for (reg, value) in self.xregisters.drain(..) {
                add_compressed.push(I::new_li(reg, value as i64, Compressed));
                add_uncompressed.push(I::new_li(reg, value as i64, Uncompressed));
            }

            add_compressed.extend(&self.instructions);
            add_uncompressed.extend(&self.instructions);

            self.instructions = add_compressed;
            self.run_inner();

            self.instructions = add_uncompressed;
            self.run_inner();
        }

        /// Run a test scenario over both the Interpreted & JIT modes of compilation,
        /// to ensure they behave identically.
        fn run_inner(&self) {
            // ensure the set of instructions can be compiled in JIT.
            self.check_compilable();

            // Create the states for the interpreted and jitted runs.
            let mut interpreted_state: TestMachineState<Interpreted<_, _>> =
                MachineState::new(InterpretedCompiler);
            interpreted_state
                .core
                .main_memory
                .set_all_readable_writeable(NoopMemoryGovernanceListener);

            let mut jitted_state: TestMachineState<Jitted<InlineCompiler<_>, _>> =
                MachineState::new(InlineCompiler::default());
            jitted_state
                .core
                .main_memory
                .set_all_readable_writeable(NoopMemoryGovernanceListener);

            let initial_pc = self.initial_pc.unwrap_or_default();

            // Push the given instructions to the correct page
            let mut interpreted_page = PageEntry::<Interpreted<M4K, Owned>>::zeroed();
            interpreted_page.push_instructions(initial_pc, self.instructions.iter().cloned());

            interpreted_state
                .page_cache
                .overwrite_page(initial_pc, interpreted_page);

            let mut jitted_page = PageEntry::<Jitted<InlineCompiler<_>, M4K>>::zeroed();
            jitted_page.push_instructions(initial_pc, self.instructions.iter().cloned());

            jitted_state
                .page_cache
                .overwrite_page(initial_pc, jitted_page);

            // Run the setup hook.
            if let Some(hook) = &self.setup_hook {
                (hook)(&mut interpreted_state.core);
                (hook)(&mut jitted_state.core)
            }

            // Preset the setup registers.
            for &(reg, value) in &self.xregisters {
                interpreted_state.core.hart.xregisters.write_nz(reg, value);
                jitted_state.core.hart.xregisters.write_nz(reg, value);
            }

            // initialise starting parameters: pc and expected_steps

            // TODO: RV-803:
            //     we are forced to run for `MAX_INSTR_COMPILED` steps as the jitted code page
            //     entry is extremely coarse in terms of the upper bound for size of entrypoints
            //
            //     Once we move the initial budget check into the JIT-compiled functions directly,
            //     we can remove this coarse bound
            //
            //     As a result, by default we expect most scenarios to end with an
            //     `Exception::IllegalInstruction` error as they move past the compiled entrypoint
            let max_steps = self.instructions.len().max(MAX_INSTR_COMPILED);
            let expected_steps = match self.expected_steps {
                ScenarioSteps::Sequence => self.instructions.len(),
                ScenarioSteps::Specific(n) => n,
                ScenarioSteps::Max => MAX_INSTR_COMPILED,
            };

            interpreted_state.core.hart.pc.write(initial_pc);
            jitted_state.core.hart.pc.write(initial_pc);

            // Run the sequence in interpreted mode and Jitted mode.
            let interpreted_res = interpreted_state.step_max_inner(max_steps);
            let jitted_res = jitted_state.step_max_inner(max_steps);

            // Assert the JIT-compiled entrypoint was called once.
            let jit_called_counter = jitted_state
                .page_cache
                .get_entrypoint_called_times(initial_pc)
                .expect("Entrypoint at initial_pc should be valid");
            assert_eq!(
                jit_called_counter, 1,
                "Expected JIT-compiled entrypoint to be called exactly once"
            );

            // Run the assert hook. We do this on both states for easier debugging.
            if let Some(hook) = &self.assert_hook {
                (hook)(&interpreted_state.core);
                (hook)(&jitted_state.core);
            }

            // Check steps. We do this on both steps for easier debugging
            assert_eq!(
                interpreted_res.steps, expected_steps,
                "Expected {expected_steps} steps; interpreted scenario ran for {}",
                interpreted_res.steps
            );
            assert_eq!(
                jitted_res.steps, expected_steps,
                "Expected {expected_steps} steps; jitted scenario ran for {}",
                jitted_res.steps
            );

            // Finally check state equality. We do this last as the earlier checks provide better
            // clues for debugging when they fail.
            assert!(
                interpreted_state.struct_ref::<FnManagerIdent>()
                    == jitted_state.struct_ref::<FnManagerIdent>(),
                "Interpreted and Jitted states should be equal."
            );
            assert_eq!(
                jitted_res, interpreted_res,
                "JittedRes {jitted_res:?} should equal InterpretedRes {interpreted_res:?}"
            );
            assert_eq!(
                jitted_res.error, self.expected_exception,
                "Expected exception: {:?}, got {:?}",
                self.expected_exception, jitted_res.error
            );
        }
    }

    /// A builder for creating scenarios.
    struct ScenarioBuilder {
        initial_pc: Option<u64>,
        expected_steps: ScenarioSteps,
        instructions: Vec<Instruction>,
        setup_hook: Option<Box<SetupHook>>,
        xregisters: Vec<(NonZeroXRegister, u64)>,
        assert_hook: Option<Box<AssertHook>>,
        expected_exception: Option<Exception>,
    }

    impl ScenarioBuilder {
        fn set_instructions(mut self, instructions: &[Instruction]) -> Self {
            self.instructions = instructions.to_vec();
            self
        }

        fn set_initial_pc(mut self, initial_pc: u64) -> Self {
            self.initial_pc = Some(initial_pc);
            self
        }

        fn set_expected_steps(mut self, expected_steps: usize) -> Self {
            self.expected_steps = ScenarioSteps::Specific(expected_steps);
            self
        }

        fn set_expect_max_steps(mut self) -> Self {
            self.expected_steps = ScenarioSteps::Max;
            // TODO: RV-803: remove workaround for assuming exception by default once
            //     JIT has better budget checks
            self.expected_exception = None;
            self
        }

        fn set_assert_hook(mut self, assert_hook: Box<AssertHook>) -> Self {
            self.assert_hook = Some(assert_hook);
            self
        }

        fn set_setup_hook(mut self, setup_hook: Box<SetupHook>) -> Self {
            self.setup_hook = Some(setup_hook);
            self
        }

        fn with_xreg(mut self, reg: NonZeroXRegister, value: u64) -> Self {
            self.xregisters.push((reg, value));
            self
        }

        fn set_expected_exception(mut self, exception: Exception) -> Self {
            self.expected_exception = Some(exception);
            self
        }

        fn build(self) -> Scenario {
            Scenario {
                initial_pc: self.initial_pc,
                expected_steps: self.expected_steps,
                instructions: self.instructions,
                setup_hook: self.setup_hook,
                xregisters: self.xregisters,
                assert_hook: self.assert_hook,
                expected_exception: self.expected_exception,
            }
        }
    }

    impl Default for ScenarioBuilder {
        fn default() -> Self {
            Self {
                initial_pc: None,
                expected_steps: ScenarioSteps::Sequence,
                instructions: vec![],
                setup_hook: None,
                xregisters: vec![],
                assert_hook: None,
                // TODO: RV-803:
                //     due to the overestimate we are forced to make in terms of budget checks,
                //     we expect most scenarios to move past the end of the instructions and encounter
                //     this exception when running `Unknown`.
                //
                //     We should revert to `None` once this is tackled
                expected_exception: Some(Exception::IllegalInstruction),
            }
        }
    }

    macro_rules! setup_hook {
        (|$core:ident| $block:block) => {
            Box::new(move |$core: &mut MachineCoreState<M4K, Owned>| $block)
        };
    }

    macro_rules! assert_hook {
        (|$core:ident| $block:block) => {
            Box::new(move |$core: &MachineCoreState<M4K, Owned>| $block)
        };
    }

    #[test]
    fn test_cnop() {
        let scenarios = vec![
            Scenario::simple(&[I::new_nop(Compressed)]),
            Scenario::simple(&[I::new_nop(Compressed), I::new_nop(Uncompressed)]),
            Scenario::simple(&[
                I::new_nop(Uncompressed),
                I::new_nop(Compressed),
                I::new_nop(Uncompressed),
            ]),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_cmv() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x2_is_one = assert_hook!(|core| {
            assert_eq!(core.hart.xregisters.read_nz(x2), 1);
        });

        // Arrange
        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(x1, 1)
                .set_instructions(&[I::new_mv(x2, x1, Compressed)])
                .set_assert_hook(assert_x2_is_one.clone())
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 1)
                .set_instructions(&[I::new_mv(x2, x1, Uncompressed)])
                .set_assert_hook(assert_x2_is_one.clone())
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 1)
                .set_instructions(&[I::new_mv(x2, x1, Compressed), I::new_mv(x3, x2, Compressed)])
                .set_assert_hook(assert_x2_is_one)
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_negate() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x1_x2_equal = assert_hook!(|core| {
            assert_eq!(
                core.hart.xregisters.read_nz(x1),
                core.hart.xregisters.read_nz(x2)
            );
        });

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(x1, -1_i64 as u64)
                .with_xreg(x3, 1)
                .set_instructions(&[I::new_neg(x2, x3, Compressed)])
                .set_assert_hook(assert_x1_x2_equal.clone())
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 1)
                .set_instructions(&[
                    I::new_neg(x3, x1, Uncompressed),
                    I::new_neg(x2, x3, Compressed),
                ])
                .set_assert_hook(assert_x1_x2_equal)
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, i64::MIN as u64)
                .set_instructions(&[I::new_neg(x2, x1, Uncompressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), i64::MIN as u64);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_jit_x64_add() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x1_is_five = assert_hook!(|core| {
            assert_eq!(core.hart.xregisters.read_nz(x1), 5);
        });

        let scenario: Scenario = ScenarioBuilder::default()
            .with_xreg(x1, 1)
            .set_instructions(&[
                I::new_x64_add(x2, x2, x1, Compressed),
                I::new_x64_add(x1, x1, x2, Uncompressed),
                I::new_x64_add(x2, x2, x1, Uncompressed),
                I::new_x64_add(x1, x1, x2, Compressed),
            ])
            .set_assert_hook(assert_x1_is_five)
            .build();

        scenario.run();
    }

    #[test]
    fn test_add_word() {
        use Instruction as I;

        use crate::machine_state::registers::a0;
        use crate::machine_state::registers::a1;

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(nz::a0, 10)
                .with_xreg(nz::a1, 1)
                .set_instructions(&[I::new_add_word(nz::a2, a0, a1, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(nz::a2), 11);
                }))
                .build(),
            // Test that we wrap around and truncate before sign extending. This
            // operation 0xFFFFFFFF + 0xFFFFFFFF should produce a different result
            // for 32-bit (truncated sum with sign extension) vs 64-bit operations.
            ScenarioBuilder::default()
                .with_xreg(nz::a0, 0xFFFFFFFF)
                .with_xreg(nz::a1, 0xFFFFFFFF)
                .set_instructions(&[I::new_add_word(nz::a2, a0, a1, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    // In 32-bit addition:
                    // 0xFFFFFFFF + 0xFFFFFFFF = 0x1FFFFFFFE
                    // Truncated to 32 bits: 0xFFFFFFFE
                    // Sign extended to 64 bits: 0xFFFFFFFFFFFFFFFE
                    assert_eq!(core.hart.xregisters.read_nz(nz::a2), -2i64 as u64);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_add_word_i() {
        use Instruction as I;

        use crate::machine_state::registers::a0;

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(nz::a0, 10)
                .set_instructions(&[I::new_add_word_immediate(nz::a1, a0, 1_i64, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(nz::a1), 11);
                }))
                .build(),
            // Test that we wrap around and truncate before sign extending. This
            // operation 0xFFFFFFFF + 0xFFFFFFFF should produce a different result
            // for 32-bit (truncated sum with sign extension) vs 64-bit operations.
            ScenarioBuilder::default()
                .with_xreg(nz::a0, 0xFFFFFFFF)
                .set_instructions(&[I::new_add_word_immediate(
                    nz::a1,
                    a0,
                    0xFFFFFFFF_i64,
                    Compressed,
                )])
                .set_assert_hook(assert_hook!(|core| {
                    // In 32-bit addition:
                    // 0xFFFFFFFF + 0xFFFFFFFF = 0x1FFFFFFFE
                    // Truncated to 32 bits: 0xFFFFFFFE
                    // Sign extended to 64 bits: 0xFFFFFFFFFFFFFFFE
                    assert_eq!(core.hart.xregisters.read_nz(nz::a1), -2i64 as u64);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_jit_x64_sub() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(x1, 10)
                .set_instructions(&[I::new_x64_sub(x2, x1, x1, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), 0);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 10)
                .with_xreg(x3, -10_i64 as u64)
                .set_instructions(&[I::new_x64_sub(x2, x1, x3, Uncompressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), 20);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 10)
                .with_xreg(x3, 100)
                .set_instructions(&[I::new_x64_sub(x2, x1, x3, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), (-90_i64) as u64);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_sub_word() {
        use Instruction as I;

        use crate::machine_state::registers::a0;
        use crate::machine_state::registers::a1;

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(nz::a0, 10)
                .with_xreg(nz::a1, 1)
                .set_instructions(&[I::new_sub_word(nz::a2, a0, a1, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(nz::a2), 9);
                }))
                .build(),
            // Test that we wrap around 0 and truncate before sign extending. This
            // operation 0xFFFFFFFFFFFFFFFF - 0xFFFFFFFF00000000 should produce a
            // different result for 32-bit (all 1s) and 64-bit operations (only lower 32-bits
            // as 1s).
            ScenarioBuilder::default()
                .with_xreg(nz::a0, !0)
                .with_xreg(nz::a1, 0xFFFFFFFF00000000u64)
                .set_instructions(&[I::new_sub_word(nz::a2, a0, a1, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(nz::a2), !0);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x64_and() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x1_and_x2_equal = assert_hook!(|core| {
            assert_eq!(
                core.hart.xregisters.read_nz(x1),
                core.hart.xregisters.read_nz(x2)
            );
        });

        let scenarios = vec![
            ScenarioBuilder::default()
                // Bitwise and with all ones is self.
                .with_xreg(x1, 13872)
                .with_xreg(x3, !0)
                .set_instructions(&[I::new_x64_and(x2, x1, x3, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal.clone())
                .build(),
            ScenarioBuilder::default()
                // Bitwise and with itself is self.
                .with_xreg(x1, 49666)
                .set_instructions(&[I::new_x64_and(x2, x1, x1, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal.clone())
                .build(),
            ScenarioBuilder::default()
                // Bitwise and with 0 is 0.
                .with_xreg(x1, 0)
                .with_xreg(x3, 540921)
                .set_instructions(&[I::new_x64_and(x2, x1, x3, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal)
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x64_or() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x1_and_x2_equal = assert_hook!(|core| {
            assert_eq!(
                core.hart.xregisters.read_nz(x1),
                core.hart.xregisters.read_nz(x2)
            );
        });

        let scenarios = vec![
            // Bitwise or with all ones is all-ones.
            ScenarioBuilder::default()
                .with_xreg(x1, !0)
                .with_xreg(x3, 13872)
                .set_instructions(&[I::new_x64_or(x2, x1, x3, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal.clone())
                .build(),
            // Bitwise or with itself is self.
            ScenarioBuilder::default()
                .with_xreg(x1, 49666)
                .set_instructions(&[I::new_x64_or(x2, x1, x1, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal.clone())
                .build(),
            // Bitwise or with 0 is self.
            ScenarioBuilder::default()
                .with_xreg(x1, 540921)
                .with_xreg(x3, 0)
                .set_instructions(&[I::new_x64_or(x2, x1, x3, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal)
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 0xF0F0)
                .set_instructions(&[I::new_x64_or_immediate(x2, x1, 0x0F0F, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), 0xFFFF);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, 0x0000)
                .set_instructions(&[I::new_x64_or_immediate(x2, x1, 0x5555, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), 0x5555);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x64_mul() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(x1, 5)
                .with_xreg(x3, 10)
                .set_instructions(&[I::new_mul(x2, x1, x3, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), 50);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, !0)
                .set_instructions(&[I::new_mul(x2, x1, x1, Uncompressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(
                        core.hart.xregisters.read_nz(x2),
                        u64::MAX.wrapping_mul(u64::MAX)
                    );
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(x1, -20_i64 as u64)
                .with_xreg(x3, 40)
                .set_instructions(&[I::new_mul(x2, x1, x3, Uncompressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), -800i64 as u64);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x32_mul() {
        use crate::machine_state::registers::a0;
        use crate::machine_state::registers::a1;

        let test_x32_mul = |value1: i64,
                            value2: i64,
                            expected_result: u64,
                            instruction_width: InstrWidth|
         -> Scenario {
            ScenarioBuilder::default()
                .with_xreg(nz::a0, value1 as u64)
                .with_xreg(nz::a1, value2 as u64)
                .set_instructions(&[I::new_x32_mul(nz::a2, a0, a1, instruction_width)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(nz::a2), expected_result);
                }))
                .build()
        };

        let scenarios = vec![
            test_x32_mul(10, 5, 50, Uncompressed),
            // Test that we truncate to 32 bits before sign extending
            // 2^32 * 2 = 2^33, but truncated to 32 bits = 0
            test_x32_mul(0x1_0000_0000, 2, 0, Compressed),
            // Test with negative numbers
            // -10 * 5 = -50, sign extended to 64 bits
            test_x32_mul(-10, 5, -50i64 as u64, Compressed),
            // Test with large 32-bit values that overflow
            // INT32_MAX * 2 = 0xFFFFFFFE (truncated to 32 bits)
            // Sign extended to 64 bits: 0xFFFFFFFFFFFFFFFE
            test_x32_mul(0x7FFFFFFF, 2, -2i64 as u64, Compressed),
            // Test with values that would produce different results in 32-bit vs 64-bit multiplication
            // In 32-bit: INT32_MIN * INT32_MIN = 0 (truncated)
            // In 64-bit: would be 0x4000000000000000
            test_x32_mul(0x80000000, 0x80000000, 0, Compressed),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x32_div_rem_signed() {
        use crate::machine_state::registers::*;

        let test_fn = |dividend, divisor| {
            let expected = match (dividend, divisor) {
                (_, 0) => -1_i32 as u32,
                (i32::MIN, -1) => i32::MIN as u32,
                _ => (dividend / divisor) as u32,
            };
            ScenarioBuilder::default()
                .with_xreg(nz::a0, dividend as u64)
                .with_xreg(nz::a1, divisor as u64)
                .set_instructions(&[
                    I::new_x32_div_signed(nz::a2, a0, a1, Compressed),
                    I::new_x32_rem_signed(nz::a3, a0, a1, Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let quotient = core.hart.xregisters.read_nz(nz::a2) as i32;
                    let remainder = core.hart.xregisters.read_nz(nz::a3) as i32;
                    assert_eq!(quotient, expected as i32);
                    if !(dividend == i32::MIN && divisor == -1) {
                        assert_eq!(dividend, divisor * quotient + remainder);
                    }
                }))
                .build()
                .run();
        };

        proptest!(|(x: i32, y: i32)| test_fn(x, y));

        // Test edge cases
        test_fn(i32::MIN, i32::MIN);
        // Division by zero
        test_fn(i32::MIN, 0);
        // Signed division overflow case
        test_fn(i32::MIN, -1);
    }

    #[test]
    fn test_x64_div_rem_signed() {
        use crate::machine_state::registers::*;

        let test_fn = |dividend, divisor| {
            let expected = match (dividend, divisor) {
                (_, 0) => -1_i64 as u64,
                (i64::MIN, -1) => i64::MIN as u64,
                _ => (dividend / divisor) as u64,
            };
            ScenarioBuilder::default()
                .with_xreg(nz::a0, dividend as u64)
                .with_xreg(nz::a1, divisor as u64)
                .set_instructions(&[
                    I::new_x64_div_signed(nz::a2, a0, a1, Compressed),
                    I::new_x64_rem_signed(nz::a3, a0, a1, Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let quotient = core.hart.xregisters.read_nz(nz::a2) as i64;
                    let remainder = core.hart.xregisters.read_nz(nz::a3) as i64;
                    assert_eq!(quotient, expected as i64);
                    if !(dividend == i64::MIN && divisor == -1) {
                        assert_eq!(dividend, divisor * quotient + remainder);
                    }
                }))
                .build()
                .run();
        };

        proptest!(|(x: i64, y: i64)| test_fn(x, y));

        // Test edge cases
        test_fn(i64::MIN, i64::MIN);
        // Division by zero
        test_fn(i64::MIN, 0);
        // Signed division overflow case
        test_fn(i64::MIN, -1);
    }

    #[test]
    fn test_x32_div_rem_unsigned() {
        use crate::machine_state::registers::*;

        let test_fn = |dividend, divisor| {
            let expected = match (dividend, divisor) {
                (_, 0) => u32::MAX,
                _ => dividend / divisor,
            };
            ScenarioBuilder::default()
                .with_xreg(nz::a0, dividend as u64)
                .with_xreg(nz::a1, divisor as u64)
                .set_instructions(&[
                    I::new_x32_div_unsigned(nz::a2, a0, a1, Compressed),
                    I::new_x32_rem_unsigned(nz::a3, a0, a1, Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let quotient = core.hart.xregisters.read_nz(nz::a2) as u32;
                    let remainder = core.hart.xregisters.read_nz(nz::a3) as u32;
                    assert_eq!(quotient, expected);
                    assert_eq!(dividend, divisor * quotient + remainder);
                }))
                .build()
                .run();
        };

        proptest!(|(x: u32, y: u32)| test_fn(x, y));

        // Division by zero
        test_fn(0, 0);
        test_fn(u32::MAX, 0);
    }

    #[test]
    fn test_x64_div_rem_unsigned() {
        use crate::machine_state::registers::*;

        let test_fn = |dividend, divisor| {
            let expected = match (dividend, divisor) {
                (_, 0) => u64::MAX,
                _ => dividend / divisor,
            };
            ScenarioBuilder::default()
                .with_xreg(nz::a0, dividend)
                .with_xreg(nz::a1, divisor)
                .set_instructions(&[
                    I::new_x64_div_unsigned(nz::a2, a0, a1, Compressed),
                    I::new_x64_rem_unsigned(nz::a3, a0, a1, Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let quotient = core.hart.xregisters.read_nz(nz::a2);
                    let remainder = core.hart.xregisters.read_nz(nz::a3);
                    assert_eq!(quotient, expected);
                    assert_eq!(dividend, divisor * quotient + remainder);
                }))
                .build()
                .run();
        };

        proptest!(|(x: u64, y: u64)| test_fn(x, y));

        // Division by zero
        test_fn(0, 0);
        test_fn(u64::MAX, 0);
    }

    #[test]
    fn test_jump_pc() {
        let scenarios = vec![
            ScenarioBuilder::default()
                // Jumping to the next instruction should exit the block
                .set_instructions(&[
                    I::new_nop(Compressed),
                    I::new_nop(Compressed),
                    I::new_jump_pc(2, Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 6);
                }))
                .build(),
            ScenarioBuilder::default()
                // Jump past 0 - in both worlds we should wrap around.
                .set_instructions(&[I::new_jump_pc(-4, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), u64::MAX - 3);
                }))
                // we jump to outside of main memory
                .set_expected_exception(Exception::InstructionAccessFault)
                .build(),
            ScenarioBuilder::default()
                // Jump past u64::MAX - in both worlds we should wrap around but not
                // execute functions past the end of the instruction sequence (the jump).
                .set_instructions(&[
                    I::new_nop(Uncompressed),
                    I::new_nop(Uncompressed),
                    I::new_jump_pc((u64::MAX - 1) as i64, Uncompressed),
                    I::new_nop(Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_initial_pc(PAGE_SIZE.get() - 18)
                .set_assert_hook(assert_hook!(|core| {
                    let expected =
                        (PAGE_SIZE.get() - 18 - 1 + 2 * Uncompressed as u64).wrapping_add(u64::MAX);
                    assert_eq!(core.hart.pc.read(), expected);
                }))
                .set_expected_steps(3)
                // we jump, back to the page but outside of instructions we pushed.
                // we therefore encounter an illegal instruction
                .set_expected_exception(Exception::IllegalInstruction)
                .build(),
            ScenarioBuilder::default()
                // jump by nothing
                .set_instructions(&[
                    I::new_nop(Compressed),
                    I::new_jump_pc(0, Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // we jump, but repeatedly to the current jump instruction
                    // this will run until the end of the scenario
                    assert_eq!(core.hart.pc.read(), 2);
                }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
            ScenarioBuilder::default()
                // jumping to start of the instruction sequence should exit the instruction sequence in both interpreted and jitted world
                //
                // since we jump to the start of the instruction sequence, however, we will fallback to partial
                // instruction sequence evaluation on the 4th step. Therefore, we do not expect an
                // IllegalInstruction for executing an `Unknown` instruction.
                .set_instructions(&[
                    I::new_nop(Compressed),
                    I::new_nop(Compressed),
                    I::new_jump_pc(-4, Compressed),
                    I::new_nop(Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // after 40 steps we will be executing the second no-op
                    assert_eq!(core.hart.pc.read(), 2);
                }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_jr() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            // JR not to start of instruction sequence should exit
            ScenarioBuilder::default()
                .with_xreg(x2, 10)
                .set_instructions(&[I::new_jr(x2, Compressed), I::new_nop(Compressed)])
                .set_assert_hook(assert_hook!(|core| { assert_eq!(core.hart.pc.read(), 10) }))
                .set_expected_steps(1)
                .set_expected_exception(Exception::IllegalInstruction)
                .build(),
            // JR to start of instruction sequence should continue with evaluating the same instruction sequence
            ScenarioBuilder::default()
                .with_xreg(x6, 0)
                .set_instructions(&[I::new_jr(x6, Compressed), I::new_nop(Compressed)])
                // after 40 steps we will be evaluating the jump for the second time
                .set_assert_hook(assert_hook!(|core| { assert_eq!(core.hart.pc.read(), 0) }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run()
        }
    }

    #[test]
    fn test_jr_imm() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            // JR_IMM not to start of instruction sequence should exit
            ScenarioBuilder::default()
                .with_xreg(x2, 10)
                .set_instructions(&[I::new_jr_imm(x2, 10, Compressed), I::new_nop(Compressed)])
                .set_assert_hook(assert_hook!(|core| { assert_eq!(core.hart.pc.read(), 20) }))
                .set_expected_steps(1)
                .set_expected_exception(Exception::IllegalInstruction)
                .build(),
            // JR_IMM to start of instruction sequence should continue with evaluating the same instruction sequence
            ScenarioBuilder::default()
                .with_xreg(x6, 10)
                .set_instructions(&[I::new_jr_imm(x6, -10, Uncompressed), I::new_nop(Compressed)])
                // after 40 steps we will be evaluating the jump for the second time
                .set_assert_hook(assert_hook!(|core| { assert_eq!(core.hart.pc.read(), 0) }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run()
        }
    }

    #[test]
    fn test_jalr() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            // JALR not to start of instruction sequence should exit
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_li(x2, 100_000, Compressed),
                    I::new_jalr(x1, x2, Compressed),
                    I::new_nop(Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 100_000);
                    assert_eq!(core.hart.xregisters.read_nz(x1), 4);
                }))
                .set_expected_steps(2)
                .set_expected_exception(Exception::InstructionAccessFault)
                .build(),
            // JALR to start of instruction sequence should continue with evaluating the same instruction sequence
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_li(x6, 0, Uncompressed),
                    I::new_jalr(x3, x6, Uncompressed),
                    I::new_nop(Compressed),
                ])
                // after 40 steps we will be evaluating the first instruction
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 0);
                    assert_eq!(core.hart.xregisters.read_nz(x3), 8);
                }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run()
        }
    }

    #[test]
    fn test_jalr_imm() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            // JALR_IMM not to start of instruction sequence should exit
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_li(x2, 10, Compressed),
                    I::new_jalr_imm(x1, x2, 10, Compressed),
                    I::new_nop(Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 20);
                    assert_eq!(core.hart.xregisters.read_nz(x1), 4);
                }))
                .set_expected_steps(2)
                .set_expected_exception(Exception::IllegalInstruction)
                .build(),
            // JALR_IMM to start of instruction sequence should continue with evaluating the same instruction sequence
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_li(x1, 1000, Uncompressed),
                    I::new_jalr_imm(x6, x1, -1000, Uncompressed),
                    I::new_nop(Compressed),
                ])
                // after 40 steps we will be evaluating the jump for the second time
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 0);
                    assert_eq!(core.hart.xregisters.read_nz(x6), 8);
                }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run()
        }
    }

    #[test]
    fn test_jalr_absolute() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            // JALR_ABSOLUTE not to start of instruction sequence should exit
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_nop(Uncompressed),
                    I::new_jalr_absolute(x1, 10, Compressed),
                    I::new_nop(Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 10);
                    assert_eq!(core.hart.xregisters.read_nz(x1), 6);
                }))
                .set_expected_steps(2)
                .set_expected_exception(Exception::IllegalInstruction)
                .build(),
            // JALR_ABSOLUTE to start of instruction sequence should continue with evaluating the same instruction sequence
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_nop(Compressed),
                    I::new_jalr_absolute(x3, 0, Uncompressed),
                    I::new_nop(Compressed),
                ])
                // after 40 steps we will be evaluating the jump for the second time
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 0);
                    assert_eq!(core.hart.xregisters.read_nz(x3), 6);
                }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run()
        }
    }

    #[test]
    fn test_j_absolute() {
        let scenarios = vec![
            // J_ABSOLUTE not to start of instruction sequence should exit
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_nop(Uncompressed),
                    I::new_j_absolute(10, Compressed),
                    I::new_nop(Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 10);
                }))
                .set_expected_steps(2)
                .set_expected_exception(Exception::IllegalInstruction)
                .build(),
            // J_ABSOLUTE to start of instruction sequence should continue with evaluating the same instruction sequence
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_nop(Compressed),
                    I::new_j_absolute(0, Uncompressed),
                    I::new_nop(Compressed),
                ])
                // after 40 steps we will be evaluating first instruction
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), 0);
                }))
                // TODO: RV-803: remove workarounds once JIT budget no longer overestimated
                .set_expect_max_steps()
                .build(),
        ];

        for scenario in scenarios {
            scenario.run()
        }
    }

    #[test]
    fn test_jump_and_link_pc() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let test_jump_and_link_pc = |offset: i64,
                                     initial_pc: u64,
                                     expected_pc: u64,
                                     expected_x1: u64,
                                     intruction_width: InstrWidth|
         -> Scenario {
            ScenarioBuilder::default()
                .set_instructions(&[I::new_jump_and_link_pc(x1, offset, intruction_width)])
                .set_initial_pc(initial_pc)
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.pc.read(), expected_pc);
                    assert_eq!(core.hart.xregisters.read_nz(x1), expected_x1);
                }))
                .build()
        };

        let scenarios = vec![
            test_jump_and_link_pc(10, 0, 10, 2, Compressed),
            test_jump_and_link_pc(-10, 10, 0, 12, Compressed),
            test_jump_and_link_pc(1000, 1000, 2000, 1004, Uncompressed),
            test_jump_and_link_pc(-((u64::MAX - 1) as i64), 500, 502, 504, Uncompressed),
            test_jump_and_link_pc(
                (u64::MAX - 1) as i64,
                PAGE_SIZE.get() - 2,
                4092,
                4096,
                Compressed,
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_addi() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x1_is_five = assert_hook!(|core| {
            assert_eq!(core.hart.xregisters.read_nz(x1), 5);
        });

        let scenarios = vec![
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_addi(x1, x1, 2, Compressed),
                    I::new_addi(x1, x1, 3, Uncompressed),
                ])
                .set_assert_hook(assert_x1_is_five.clone())
                .build(),
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_addi(x1, x1, i64::MAX, Compressed),
                    I::new_addi(x1, x1, i64::MAX, Compressed),
                    I::new_addi(x1, x1, 7, Uncompressed),
                ])
                .set_assert_hook(assert_x1_is_five.clone())
                .build(),
            ScenarioBuilder::default()
                .set_instructions(&[
                    I::new_addi(x1, x3, 7, Compressed),
                    I::new_addi(x1, x1, -2, Uncompressed),
                ])
                .set_assert_hook(assert_x1_is_five)
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_andi() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister::*;

        let assert_x1_and_x2_equal = assert_hook!(|core| {
            assert_eq!(
                core.hart.xregisters.read_nz(x1),
                core.hart.xregisters.read_nz(x2)
            );
        });

        let scenarios = vec![
            // Bitwise and with all ones is self.
            ScenarioBuilder::default()
                .with_xreg(x1, 13872)
                .set_instructions(&[I::new_andi(x2, x1, !0, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal.clone())
                .build(),
            // Bitwise and with itself is self.
            ScenarioBuilder::default()
                .with_xreg(x1, 49666)
                .set_instructions(&[I::new_andi(x2, x1, 49666, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal.clone())
                .build(),
            // Bitwise and with 0 is 0.
            ScenarioBuilder::default()
                .with_xreg(x1, 0)
                .set_instructions(&[I::new_andi(x2, x1, 50230, Compressed)])
                .set_assert_hook(assert_x1_and_x2_equal)
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_set_less_than() {
        use crate::machine_state::registers::XRegister::*;

        const TRUE: u64 = 1;
        const FALSE: u64 = 0;

        let with_maybe_zero_xreg =
            |scenario_builder: ScenarioBuilder, (reg, val): (XRegister, i64)| -> ScenarioBuilder {
                match reg.try_into() {
                    Ok(nz) => scenario_builder.with_xreg(nz, val as u64),
                    Err(_) => {
                        assert_eq!(val, 0);
                        scenario_builder
                    }
                }
            };

        let test_slt = |constructor: fn(NonZeroXRegister, XRegister, XRegister) -> I,
                        lhs: (XRegister, i64),
                        rhs: (XRegister, i64),
                        expected: u64|
         -> Scenario {
            let scenario_builder = ScenarioBuilder::default();
            let scenario_builder = with_maybe_zero_xreg(scenario_builder, lhs);
            let scenario_builder = with_maybe_zero_xreg(scenario_builder, rhs);
            scenario_builder
                .set_instructions(&[constructor(nz::ra, lhs.0, rhs.0)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(
                        expected,
                        core.hart.xregisters.read_nz(nz::ra),
                        "Expected {expected} for Slt* lhs: {lhs:?}, rhs: {rhs:?}"
                    )
                }))
                .build()
        };

        let test_slt_imm = |constructor: fn(NonZeroXRegister, XRegister, i64) -> I,
                            lhs: (XRegister, i64),
                            rhs: i64,
                            expected: u64|
         -> Scenario {
            let scenario_builder = ScenarioBuilder::default();
            let scenario_builder = with_maybe_zero_xreg(scenario_builder, lhs);
            scenario_builder
                .set_instructions(&[constructor(nz::ra, lhs.0, rhs)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(
                        expected,
                        core.hart.xregisters.read_nz(nz::ra),
                        "Expected {expected} for Slt* lhs: {lhs:?}, rhs: {rhs:?}"
                    )
                }))
                .build()
        };

        let scenarios = vec![
            // -------------------------
            // equal values always false
            // -------------------------
            // Slt
            test_slt(I::new_set_less_than_signed, (x1, 1), (x2, 1), FALSE),
            test_slt(I::new_set_less_than_signed, (x0, 0), (x2, 0), FALSE),
            test_slt(I::new_set_less_than_signed, (x3, -1), (x2, -1), FALSE),
            // Sltu
            test_slt(I::new_set_less_than_unsigned, (x1, 1), (x2, 1), FALSE),
            test_slt(I::new_set_less_than_unsigned, (x0, 0), (x2, 0), FALSE),
            test_slt(I::new_set_less_than_unsigned, (x3, -1), (x2, -1), FALSE),
            // Slti
            test_slt_imm(I::new_set_less_than_immediate_signed, (x1, 1), 1, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_signed, (x0, 0), 0, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_signed, (x3, -1), -1, FALSE),
            // Sltiu
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x1, 1), 1, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x0, 0), 0, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x3, -1), -1, FALSE),
            // --------------------------------
            // greater than values always false
            // --------------------------------
            // Slt
            test_slt(I::new_set_less_than_signed, (x1, 3), (x2, 1), FALSE),
            test_slt(I::new_set_less_than_signed, (x0, 0), (x2, -2), FALSE),
            test_slt(I::new_set_less_than_signed, (x3, -1), (x2, -5), FALSE),
            // Sltu
            test_slt(I::new_set_less_than_unsigned, (x1, 1), (x2, 1), FALSE),
            test_slt(I::new_set_less_than_unsigned, (x2, 5), (x0, 0), FALSE),
            test_slt(I::new_set_less_than_unsigned, (x3, -1), (x2, 2), FALSE),
            // Slti
            test_slt_imm(I::new_set_less_than_immediate_signed, (x1, 2), 1, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_signed, (x5, 1), 0, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_signed, (x3, -5), -6, FALSE),
            // Sltiu
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x1, 5), 1, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x3, -1), 15, FALSE),
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x3, -1), -6, FALSE),
            // ----------------------------
            // less than values always true
            // ----------------------------
            // Slt
            test_slt(I::new_set_less_than_signed, (x1, 2), (x2, 5), TRUE),
            test_slt(I::new_set_less_than_signed, (x0, 0), (x2, 3), TRUE),
            test_slt(I::new_set_less_than_signed, (x3, -5), (x2, -3), TRUE),
            // Sltu
            test_slt(I::new_set_less_than_unsigned, (x1, 1), (x2, -1), TRUE),
            test_slt(I::new_set_less_than_unsigned, (x0, 0), (x3, 5), TRUE),
            test_slt(I::new_set_less_than_unsigned, (x3, -2), (x2, -1), TRUE),
            // Slti
            test_slt_imm(I::new_set_less_than_immediate_signed, (x1, 2), 5, TRUE),
            test_slt_imm(I::new_set_less_than_immediate_signed, (x5, 0), 3, TRUE),
            test_slt_imm(I::new_set_less_than_immediate_signed, (x3, -6), -5, TRUE),
            // Sltiu
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x1, 3), 5, TRUE),
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x3, 5), -15, TRUE),
            test_slt_imm(I::new_set_less_than_immediate_unsigned, (x3, -7), -6, TRUE),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_branch() {
        let test_branch =
            |non_branch: fn(NonZeroXRegister, NonZeroXRegister, i64, InstrWidth) -> I,
             branch: fn(NonZeroXRegister, NonZeroXRegister, i64, InstrWidth) -> I,
             lhs: i64,
             rhs: i64|
             -> Scenario {
                let initial_pc: u64 = 0x100;
                let imm: i64 = -0x2000;
                let expected_pc_branch = initial_pc.wrapping_add_signed(imm).wrapping_add(8);

                ScenarioBuilder::default()
                    .set_initial_pc(initial_pc)
                    .set_instructions(&[
                        I::new_li(nz::a1, lhs, InstrWidth::Compressed),
                        I::new_li(nz::a2, rhs, InstrWidth::Compressed),
                        non_branch(nz::a1, nz::a2, imm, InstrWidth::Uncompressed),
                        branch(nz::a1, nz::a2, imm, InstrWidth::Uncompressed),
                        I::new_nop(InstrWidth::Compressed),
                    ])
                    .set_expected_steps(4)
                    // we branch, and all memory is set as non-executable.
                    // since we exit the instruction sequence, we fall back to fetch/run - which will fail
                    .set_expected_exception(Exception::InstructionAccessFault)
                    .set_assert_hook(assert_hook!(|core| {
                        assert_eq!(
                            expected_pc_branch,
                            core.hart.pc.read(),
                            "Expected {expected_pc_branch} pc for B*Zero cmp {lhs}, {rhs}"
                        )
                    }))
                    .build()
            };

        let scenarios = vec![
            // Equality
            test_branch(I::new_branch_equal, I::new_branch_not_equal, 2, 3),
            test_branch(I::new_branch_not_equal, I::new_branch_equal, 2, 2),
            test_branch(I::new_branch_equal, I::new_branch_not_equal, 2, -3),
            // LessThanUnsigned + GreaterThanOrEqualUnsigned
            test_branch(
                I::new_branch_less_than_unsigned,
                I::new_branch_greater_than_or_equal_unsigned,
                3,
                2,
            ),
            test_branch(
                I::new_branch_less_than_unsigned,
                I::new_branch_greater_than_or_equal_unsigned,
                2,
                2,
            ),
            test_branch(
                I::new_branch_greater_than_or_equal_unsigned,
                I::new_branch_less_than_unsigned,
                2,
                -3,
            ),
            // LessThanSigned + GreaterThanOrEqualSigned
            test_branch(
                I::new_branch_less_than_signed,
                I::new_branch_greater_than_or_equal_signed,
                3,
                2,
            ),
            test_branch(
                I::new_branch_less_than_signed,
                I::new_branch_greater_than_or_equal_signed,
                2,
                2,
            ),
            test_branch(
                I::new_branch_less_than_signed,
                I::new_branch_greater_than_or_equal_signed,
                2,
                -3,
            ),
            test_branch(
                I::new_branch_greater_than_or_equal_signed,
                I::new_branch_less_than_signed,
                -4,
                -3,
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_branch_compare_zero() {
        let test_branch_compare_zero = |non_branch: fn(NonZeroXRegister, i64, InstrWidth) -> I,
                                        branch: fn(NonZeroXRegister, i64, InstrWidth) -> I,
                                        val: i64|
         -> Scenario {
            let initial_pc: u64 = 0x100;
            let imm: i64 = 0x2000;
            let expected_pc_branch = initial_pc + imm as u64 + 4;

            ScenarioBuilder::default()
                .set_initial_pc(initial_pc)
                .set_instructions(&[
                    I::new_li(nz::ra, val, InstrWidth::Compressed),
                    non_branch(nz::ra, imm, InstrWidth::Compressed),
                    branch(nz::ra, imm, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(3)
                // we branch, and all memory is set as non-executable
                .set_expected_exception(Exception::InstructionAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(
                        expected_pc_branch,
                        core.hart.pc.read(),
                        "Expected {expected_pc_branch} pc for B*Zero cmp {val:?}"
                    )
                }))
                .build()
        };

        let scenarios = vec![
            // Equality
            test_branch_compare_zero(I::new_branch_equal_zero, I::new_branch_not_equal_zero, 12),
            test_branch_compare_zero(I::new_branch_not_equal_zero, I::new_branch_equal_zero, 0),
            test_branch_compare_zero(I::new_branch_equal_zero, I::new_branch_not_equal_zero, -12),
            // LessThan + GreaterThanOrEqual
            test_branch_compare_zero(
                I::new_branch_less_than_zero,
                I::new_branch_greater_than_or_equal_zero,
                12,
            ),
            test_branch_compare_zero(
                I::new_branch_less_than_zero,
                I::new_branch_greater_than_or_equal_zero,
                0,
            ),
            test_branch_compare_zero(
                I::new_branch_greater_than_or_equal_zero,
                I::new_branch_less_than_zero,
                -12,
            ),
            // LessThanOrEqual + GreaterThan
            test_branch_compare_zero(
                I::new_branch_less_than_or_equal_zero,
                I::new_branch_greater_than_zero,
                12,
            ),
            test_branch_compare_zero(
                I::new_branch_greater_than_zero,
                I::new_branch_less_than_or_equal_zero,
                0,
            ),
            test_branch_compare_zero(
                I::new_branch_greater_than_zero,
                I::new_branch_less_than_or_equal_zero,
                -12,
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_unknown() {
        let scenarios = vec![
            ScenarioBuilder::default()
                .set_expected_steps(
                    // The unknown instruction raises an exception. This does not count as a full step.
                    1,
                )
                .set_expected_exception(Exception::IllegalInstruction)
                .set_instructions(&[
                    I::new_nop(Uncompressed),
                    I::new_unknown(Compressed),
                    I::new_nop(Uncompressed),
                ])
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_ecall() {
        let scenario: Scenario = ScenarioBuilder::default()
            .set_expected_steps(1)
            .set_expected_exception(Exception::EnvCall)
            .set_instructions(&[
                I::new_nop(Uncompressed),
                I::new_ecall(),
                I::new_nop(Uncompressed),
            ])
            .build();

        scenario.run();
    }

    #[test]
    fn test_jit_recovers_from_compilation_failure() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        // Arrange
        let failure_scenarios: &[&[I]] = &[
            &[
                // does not currently support lowering.
                I::new_fadds(x1, x1, x1, Uncompressed),
            ],
            &[
                I::new_nop(Uncompressed),
                // does not currently support lowering.
                I::new_fadds(x1, x1, x1, Uncompressed),
            ],
        ];

        let success: &[I] = &[I::new_nop(Compressed)];

        for failure in failure_scenarios.iter() {
            let mut jit = JIT::<M4K>::new().unwrap();

            let mut jitted = MachineCoreState::<M4K, _>::new();

            let initial_pc = 0;
            jitted.hart.pc.write(initial_pc);

            jitted.hart.xregisters.write_nz(x1, 1);

            // Act
            let res = jit.compile(failure, initial_pc);

            assert!(
                res.is_none(),
                "Compilation of unsupported instruction should fail"
            );

            let fun = jit
                .compile(success, initial_pc)
                .expect("Compilation of subsequent functions should succeed");

            let mut jitted_err = ExceptionCode::NoException;
            let max_steps = usize::MAX;
            let jitted_steps = unsafe {
                // # Safety - the jit is not dropped until after we
                //            exit the instruction sequence.
                (fun)(
                    null(),
                    &mut jitted,
                    initial_pc,
                    max_steps,
                    &mut jitted_err,
                    null(),
                )
            };

            assert_eq!(jitted_err, ExceptionCode::NoException);
            assert_eq!(jitted_steps, success.len());
        }
    }

    #[test]
    fn test_add_immediate_to_pc() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let scenarios = vec![
            ScenarioBuilder::default()
                .set_initial_pc(1000)
                .set_instructions(&[I::new_add_immediate_to_pc(x1, 4096, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x1), 5096);
                }))
                .build(),
            ScenarioBuilder::default()
                .set_instructions(&[I::new_add_immediate_to_pc(x1, 0xFFFFF000, Uncompressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x1), 0xFFFFF000);
                }))
                .build(),
            ScenarioBuilder::default()
                .set_initial_pc(1000)
                .set_instructions(&[I::new_add_immediate_to_pc(x1, -4096, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x1), -3096_i64 as u64);
                }))
                .build(),
            ScenarioBuilder::default()
                .set_initial_pc(1000)
                .set_instructions(&[I::new_add_immediate_to_pc(x1, 20, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x1), 1020);
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_shift_reg() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let x64_shift_reg = |constructor: fn(
            NonZeroXRegister,
            NonZeroXRegister,
            NonZeroXRegister,
            InstrWidth,
        ) -> I,
                             lhs: (NonZeroXRegister, i64),
                             rhs: (NonZeroXRegister, i64),
                             expected: u64|
         -> Scenario {
            ScenarioBuilder::default()
                .with_xreg(lhs.0, lhs.1 as u64)
                .with_xreg(rhs.0, rhs.1 as u64)
                .set_instructions(&[constructor(x2, lhs.0, rhs.0, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(
                        expected,
                        core.hart.xregisters.read_nz(x2),
                        "Expected {expected} for Shift* lhs: {lhs:?}, rhs: {rhs:?}"
                    )
                }))
                .build()
        };

        let x32_shift_reg =
            |constructor: fn(NonZeroXRegister, XRegister, XRegister, InstrWidth) -> I,
             lhs: (NonZeroXRegister, i64),
             rhs: (NonZeroXRegister, i64),
             expected: u64|
             -> Scenario {
                ScenarioBuilder::default()
                    .with_xreg(lhs.0, lhs.1 as u64)
                    .with_xreg(rhs.0, rhs.1 as u64)
                    .set_instructions(&[constructor(x2, lhs.0.into(), rhs.0.into(), Compressed)])
                    .set_assert_hook(assert_hook!(|core| {
                        assert_eq!(
                            expected,
                            core.hart.xregisters.read_nz(x2),
                            "Expected {expected} for Shift* lhs: {lhs:?}, rhs: {rhs:?}"
                        )
                    }))
                    .build()
            };

        let scenarios = vec![
            x64_shift_reg(I::new_x64_shift_left, (x1, 1), (x3, 1), 2),
            x64_shift_reg(
                I::new_x64_shift_left,
                (x1, 1),
                (x3, 63),
                0x8000_0000_0000_0000,
            ),
            x64_shift_reg(I::new_x64_shift_left, (x1, 2), (x3, 63), 0),
            x64_shift_reg(
                I::new_x64_shift_left,
                (x1, 1),
                (x3, 126),
                0x4000_0000_0000_0000,
            ),
            x64_shift_reg(I::new_x64_shift_left, (x1, -16), (x3, 2), -64_i64 as u64),
            x64_shift_reg(I::new_x64_shift_right_unsigned, (x1, 2), (x3, 1), 1),
            x64_shift_reg(I::new_x64_shift_right_unsigned, (x1, !0), (x3, 63), 1),
            x64_shift_reg(
                I::new_x64_shift_right_unsigned,
                (x1, 0x7FFF_FFFF_FFFF_FFFF),
                (x3, 63),
                0,
            ),
            x64_shift_reg(I::new_x64_shift_right_unsigned, (x1, !0), (x3, 126), 3),
            x64_shift_reg(
                I::new_x64_shift_right_unsigned,
                (x1, -8),
                (x3, 2),
                0x3FFF_FFFF_FFFF_FFFE,
            ),
            x64_shift_reg(I::new_x64_shift_right_signed, (x1, 2), (x3, 1), 1),
            x64_shift_reg(I::new_x64_shift_right_signed, (x1, !0), (x3, 63), !0),
            x64_shift_reg(
                I::new_x64_shift_right_signed,
                (x1, 0x7FFF_FFFF_FFFF_FFFF),
                (x3, 62),
                1,
            ),
            x64_shift_reg(I::new_x64_shift_right_signed, (x1, !0), (x3, 126), !0),
            x64_shift_reg(
                I::new_x64_shift_right_signed,
                (x1, -8),
                (x3, 2),
                -2_i64 as u64,
            ),
            // X32ShiftLeft tests
            x32_shift_reg(I::new_x32_shift_left, (x1, 1), (x3, 1), 2),
            x32_shift_reg(
                I::new_x32_shift_left,
                (x1, 1),
                (x3, 31),
                0xFFFF_FFFF_8000_0000,
            ),
            x32_shift_reg(I::new_x32_shift_left, (x1, 2), (x3, 31), 0),
            x32_shift_reg(
                I::new_x32_shift_left,
                (x1, 1),
                (x3, 95),
                0xFFFF_FFFF_8000_0000,
            ),
            // X32ShiftRightUnsigned tests
            x32_shift_reg(I::new_x32_shift_right_unsigned, (x1, 2), (x3, 1), 1),
            x32_shift_reg(
                I::new_x32_shift_right_unsigned,
                (x1, 0x80000000),
                (x3, 31),
                1,
            ),
            x32_shift_reg(I::new_x32_shift_right_unsigned, (x1, 1), (x3, 31), 0),
            x32_shift_reg(
                I::new_x32_shift_right_unsigned,
                (x1, 0x80000000),
                (x3, 95),
                1,
            ),
            // X32ShiftRightSigned tests
            x32_shift_reg(I::new_x32_shift_right_signed, (x1, 2), (x3, 1), 1),
            x32_shift_reg(
                I::new_x32_shift_right_signed,
                (x1, 0x80000000),
                (x3, 31),
                0xFFFF_FFFF_FFFF_FFFF,
            ),
            x32_shift_reg(I::new_x32_shift_right_signed, (x1, 0x40000000), (x3, 31), 0),
            x32_shift_reg(
                I::new_x32_shift_right_signed,
                (x1, 0x80000000),
                (x3, 95),
                0xFFFF_FFFF_FFFF_FFFF,
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_shift_imm() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let shift_imm =
            |constructor: fn(NonZeroXRegister, NonZeroXRegister, i64, InstrWidth) -> I,
             lhs: (NonZeroXRegister, i64),
             imm: i64,
             expected: u64|
             -> Scenario {
                ScenarioBuilder::default()
                    .with_xreg(lhs.0, lhs.1 as u64)
                    .set_instructions(&[constructor(x2, lhs.0, imm, Compressed)])
                    .set_assert_hook(assert_hook!(|core| {
                        assert_eq!(
                            expected,
                            core.hart.xregisters.read_nz(x2),
                            "Expected {expected} for Shift* lhs: {lhs:?}, imm: {imm}"
                        )
                    }))
                    .build()
            };

        let scenarios = vec![
            shift_imm(I::new_x64_shift_left_imm, (x1, 1), 1, 2),
            shift_imm(
                I::new_x64_shift_left_imm,
                (x1, 1),
                63,
                0x8000_0000_0000_0000,
            ),
            shift_imm(I::new_x64_shift_left_imm, (x1, 2), 63, 0),
            shift_imm(I::new_x64_shift_left_imm, (x1, -16), 2, -64_i64 as u64),
            shift_imm(I::new_x64_shift_right_imm_unsigned, (x1, 2), 1, 1),
            shift_imm(I::new_x64_shift_right_imm_unsigned, (x1, !0), 63, 1),
            shift_imm(
                I::new_x64_shift_right_imm_unsigned,
                (x1, 0x7FFF_FFFF_FFFF_FFFF),
                63,
                0,
            ),
            shift_imm(
                I::new_x64_shift_right_imm_unsigned,
                (x1, -8),
                2,
                0x3FFF_FFFF_FFFF_FFFE,
            ),
            shift_imm(I::new_x64_shift_right_imm_signed, (x1, 2), 1, 1),
            shift_imm(I::new_x64_shift_right_imm_signed, (x1, !0), 63, !0),
            shift_imm(
                I::new_x64_shift_right_imm_signed,
                (x1, 0x7FFF_FFFF_FFFF_FFFF),
                62,
                1,
            ),
            shift_imm(
                I::new_x64_shift_right_imm_signed,
                (x1, -8),
                2,
                -2_i64 as u64,
            ),
            // X32ShiftLeftImmediate tests
            shift_imm(I::new_x32_shift_left_immediate, (x1, 1), 1, 2),
            shift_imm(
                I::new_x32_shift_left_immediate,
                (x1, 1),
                31,
                0xFFFF_FFFF_8000_0000,
            ),
            shift_imm(I::new_x32_shift_left_immediate, (x1, 2), 31, 0),
            // X32ShiftRightImmediateUnsigned tests
            shift_imm(I::new_x32_shift_right_immediate_unsigned, (x1, 2), 1, 1),
            shift_imm(
                I::new_x32_shift_right_immediate_unsigned,
                (x1, 0x80000000),
                31,
                1,
            ),
            shift_imm(I::new_x32_shift_right_immediate_unsigned, (x1, 1), 31, 0),
            // X32ShiftRightImmediateSigned tests
            shift_imm(I::new_x32_shift_right_immediate_signed, (x1, 2), 1, 1),
            shift_imm(
                I::new_x32_shift_right_immediate_signed,
                (x1, 0x80000000),
                31,
                0xFFFF_FFFF_FFFF_FFFF,
            ),
            shift_imm(
                I::new_x32_shift_right_immediate_signed,
                (x1, 0x40000000),
                31,
                0,
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_store() {
        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::*;

        type ConstructStoreFn =
            fn(rs1: XRegister, rs2: XRegister, imm: i64, width: InstrWidth) -> I;

        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;
        const XREG_VALUE: u64 = 0xFFEEDDCCBBAA9988;

        let valid_store = |constructor: ConstructStoreFn, imm: u64, expected: u64| {
            const STORE_ADDRESS_BASE: u64 = MEMORY_SIZE / 2;

            ScenarioBuilder::default()
                .with_xreg(NZ::x1, STORE_ADDRESS_BASE)
                .with_xreg(NZ::x2, XREG_VALUE)
                .set_instructions(&[
                    constructor(x1, x2, imm as i64, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.main_memory.read(STORE_ADDRESS_BASE + imm).unwrap();

                    assert_eq!(value, expected, "Found {value:x}, expected {expected:x}");
                }))
                .build()
        };

        let invalid_store = |constructor: ConstructStoreFn, width: LoadStoreWidth| {
            // an address that, with the immediate of 16, will be out of bounds by one byte
            let store_address_base = MEMORY_SIZE - 15 - width as u64;
            let store_address_offset = 16;

            ScenarioBuilder::default()
                .with_xreg(NZ::x1, store_address_base)
                .with_xreg(NZ::x2, XREG_VALUE)
                .set_instructions(&[
                    constructor(
                        x1,
                        x2,
                        store_address_offset as i64,
                        InstrWidth::Uncompressed,
                    ),
                    I::new_nop(InstrWidth::Compressed),
                ])
                // the load will fail due to being out of bounds
                .set_expected_steps(
                    // A failed store does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.main_memory.read(MEMORY_SIZE - 8).unwrap();

                    assert_eq!(value, 0, "Found {value:x}, but expected store to fail");
                }))
                .build()
        };

        let scenarios = vec![
            // check stores - differing imm value to ensure both
            // aligned & unaligned stores are supported
            valid_store(I::new_x64_store, 8, XREG_VALUE),
            valid_store(I::new_x64_store, 5, XREG_VALUE),
            valid_store(I::new_x32_store, 4, XREG_VALUE as u32 as u64),
            valid_store(I::new_x32_store, 3, XREG_VALUE as u32 as u64),
            valid_store(I::new_x16_store, 2, XREG_VALUE as u16 as u64),
            valid_store(I::new_x16_store, 1, XREG_VALUE as u16 as u64),
            // byte load always aligned
            valid_store(I::new_x8_store, 0, XREG_VALUE as u8 as u64),
            // invalid stores: out of bounds
            invalid_store(I::new_x64_store, LoadStoreWidth::Double),
            invalid_store(I::new_x32_store, LoadStoreWidth::Word),
            invalid_store(I::new_x16_store, LoadStoreWidth::Half),
            invalid_store(I::new_x8_store, LoadStoreWidth::Byte),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_load() {
        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::*;

        type ConstructLoadFn = fn(rd: XRegister, rs1: XRegister, imm: i64, width: InstrWidth) -> I;

        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;

        let valid_load = |new_load: ConstructLoadFn, imm: u64, expected: u64| {
            const LOAD_ADDRESS_BASE: u64 = MEMORY_SIZE / 2;

            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(LOAD_ADDRESS_BASE + imm, expected)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, LOAD_ADDRESS_BASE)
                .set_instructions(&[
                    new_load(x2, x1, imm as i64, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x2);
                    assert_eq!(value, expected, "Found {value:x}, expected {expected:x}");
                }))
                .build()
        };

        let invalid_load = |new_load: ConstructLoadFn, width: LoadStoreWidth| {
            // an address that, with the immediate of 16, will be out of bounds by one byte
            let load_address_base = MEMORY_SIZE - 15 - width as u64;
            let load_address_offset = 16;

            ScenarioBuilder::default()
                .with_xreg(NZ::x1, load_address_base)
                .set_instructions(&[
                    new_load(x2, x1, load_address_offset as i64, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                // the load will fail due to being out of bounds
                .set_expected_steps(
                    // A failed load does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::LoadAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x2);
                    assert_eq!(value, 0, "Found {value:x}, but expected load to fail");
                }))
                .build()
        };

        const XREG_VALUE: u64 = 0xFFEEDDCCBBAA9988;

        let scenarios = vec![
            // check loads - differing imm value to ensure both
            // aligned & unaligned loads are supported
            valid_load(I::new_x64_load_signed, 8, XREG_VALUE),
            valid_load(I::new_x64_load_signed, 5, XREG_VALUE),
            valid_load(I::new_x32_load_signed, 4, XREG_VALUE as i32 as u64),
            valid_load(I::new_x32_load_unsigned, 4, XREG_VALUE as u32 as u64),
            valid_load(I::new_x32_load_signed, 3, XREG_VALUE as i32 as u64),
            valid_load(I::new_x32_load_unsigned, 3, XREG_VALUE as u32 as u64),
            valid_load(I::new_x16_load_signed, 2, XREG_VALUE as i16 as u64),
            valid_load(I::new_x16_load_unsigned, 2, XREG_VALUE as u16 as u64),
            valid_load(I::new_x16_load_signed, 1, XREG_VALUE as i16 as u64),
            valid_load(I::new_x16_load_unsigned, 1, XREG_VALUE as u16 as u64),
            // byte load always aligned
            valid_load(I::new_x8_load_signed, 0, XREG_VALUE as i8 as u64),
            valid_load(I::new_x8_load_unsigned, 0, XREG_VALUE as u8 as u64),
            // invalid loads: out of bounds
            invalid_load(I::new_x64_load_signed, LoadStoreWidth::Double),
            invalid_load(I::new_x32_load_signed, LoadStoreWidth::Word),
            invalid_load(I::new_x32_load_unsigned, LoadStoreWidth::Word),
            invalid_load(I::new_x16_load_signed, LoadStoreWidth::Half),
            invalid_load(I::new_x16_load_unsigned, LoadStoreWidth::Half),
            invalid_load(I::new_x8_load_signed, LoadStoreWidth::Byte),
            invalid_load(I::new_x8_load_unsigned, LoadStoreWidth::Byte),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_xor() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let bitwise_test_xor_immediate =
            |lhs_reg: NonZeroXRegister, lhs_val: u64, imm: i64, expected: u64| -> Scenario {
                ScenarioBuilder::default()
                    .with_xreg(lhs_reg, lhs_val)
                    .set_instructions(&[I::new_x64_xor_immediate(x2, lhs_reg, imm, Compressed)])
                    .set_assert_hook(assert_hook!(|core| {
                        assert_eq!(core.hart.xregisters.read_nz(x2), expected);
                    }))
                    .build()
            };

        let bitwise_test_xor = |lhs_reg: NonZeroXRegister,
                                lhs_val: u64,
                                rhs_reg: NonZeroXRegister,
                                rhs_val: u64,
                                expected: u64|
         -> Scenario {
            ScenarioBuilder::default()
                .with_xreg(lhs_reg, lhs_val)
                .with_xreg(rhs_reg, rhs_val)
                .set_instructions(&[I::new_x64_xor(x2, lhs_reg, rhs_reg, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), expected);
                }))
                .build()
        };

        let scenarios = vec![
            // XOR immediate tests
            bitwise_test_xor_immediate(x1, 0xF0F0, 0x0F0F, 0xFFFF),
            bitwise_test_xor_immediate(x1, 0xAAAA, 0x5555, 0xFFFF),
            bitwise_test_xor_immediate(x1, 0xFFF0, 0x0FFF, 0xF00F),
            // XOR register tests
            bitwise_test_xor(x1, 0xF0F0, x3, 0x0F0F, 0xFFFF),
            bitwise_test_xor(x1, 0xAAAA, x3, 0x5555, 0xFFFF),
            bitwise_test_xor(x1, 0xFFF0, x3, 0x0FFF, 0xF00F),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x64_atomic() {
        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::*;

        type ConstructAtomicFn = fn(
            rd: XRegister,
            rs1: XRegister,
            rs2: XRegister,
            aq: bool,
            rl: bool,
            width: InstrWidth,
        ) -> I;

        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;

        const ADDRESS_BASE_ATOMICS: u64 = MEMORY_SIZE / 2;

        let valid_x64_atomic_signed = |constructor: ConstructAtomicFn,
                                       val1: i64,
                                       val2: i64,
                                       fun: fn(i64, i64) -> i64|
         -> Scenario {
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory.write(ADDRESS_BASE_ATOMICS, val1).unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x2, val2 as u64)
                .set_instructions(&[
                    constructor(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value as i64, val1);

                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS).unwrap();
                    let expected = fun(val1, val2);
                    assert_eq!(
                        res as i64, expected,
                        "Found {value:x}, expected {expected:x}"
                    );
                }))
                .build()
        };

        let valid_x64_atomic_unsigned = |constructor: ConstructAtomicFn,
                                         val1: u64,
                                         val2: u64,
                                         fun: fn(u64, u64) -> u64|
         -> Scenario {
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory.write(ADDRESS_BASE_ATOMICS, val1).unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x2, val2)
                .set_instructions(&[
                    constructor(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, val1);

                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS).unwrap();
                    let expected = fun(val1, val2);
                    assert_eq!(res, expected, "Found {value:x}, expected {expected:x}");
                }))
                .build()
        };

        let invalid_x64_atomic_signed = |constructor: ConstructAtomicFn,
                                         val1: i64,
                                         val2: i64,
                                         fun: fn(i64, i64) -> i64|
         -> Scenario {
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(ADDRESS_BASE_ATOMICS + 4, val1)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS + 4)
                .with_xreg(NZ::x2, val2 as u64)
                .set_instructions(&[
                    constructor(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // A failed atomic operation does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value as i64, 0);

                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 4).unwrap();
                    let expected = fun(val1, val2);
                    assert_eq!(res as i64, val1, "Found {value:x}, expected {expected:x}");
                }))
                .build()
        };

        let invalid_x64_atomic_unsigned = |constructor: ConstructAtomicFn,
                                           val1: u64,
                                           val2: u64,
                                           fun: fn(u64, u64) -> u64|
         -> Scenario {
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(ADDRESS_BASE_ATOMICS + 4, val1)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS + 4)
                .with_xreg(NZ::x2, val2)
                .set_instructions(&[
                    constructor(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // A failed atomic operation does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);

                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 4).unwrap();
                    let expected = fun(val1, val2);
                    assert_eq!(res, val1, "Found {value:x}, expected {expected:x}");
                }))
                .build()
        };

        let scenarios = vec![
            valid_x64_atomic_unsigned(I::new_x64_atomic_add, 10, 30, u64::wrapping_add),
            invalid_x64_atomic_unsigned(I::new_x64_atomic_add, 10, 30, u64::wrapping_add),
            valid_x64_atomic_unsigned(I::new_x64_atomic_and, 10, 30, u64::bitand),
            invalid_x64_atomic_unsigned(I::new_x64_atomic_and, 10, 30, u64::bitand),
            valid_x64_atomic_unsigned(I::new_x64_atomic_or, 10, 30, u64::bitor),
            invalid_x64_atomic_unsigned(I::new_x64_atomic_or, 10, 30, u64::bitor),
            valid_x64_atomic_unsigned(I::new_x64_atomic_xor, 10, 30, u64::bitxor),
            invalid_x64_atomic_unsigned(I::new_x64_atomic_xor, 10, 30, u64::bitxor),
            valid_x64_atomic_signed(I::new_x64_atomic_min_signed, -10, 30, i64::min),
            invalid_x64_atomic_signed(I::new_x64_atomic_min_signed, 10, -30, i64::min),
            valid_x64_atomic_unsigned(I::new_x64_atomic_min_unsigned, 10, 30, u64::min),
            valid_x64_atomic_unsigned(I::new_x64_atomic_min_unsigned, 10, -30_i64 as u64, u64::min),
            invalid_x64_atomic_unsigned(I::new_x64_atomic_min_unsigned, 10, 30, u64::min),
            valid_x64_atomic_signed(I::new_x64_atomic_max_signed, -10, 30, i64::max),
            invalid_x64_atomic_signed(I::new_x64_atomic_max_signed, 10, -30, i64::max),
            valid_x64_atomic_unsigned(I::new_x64_atomic_max_unsigned, 10, 30, u64::max),
            valid_x64_atomic_unsigned(I::new_x64_atomic_max_unsigned, 10, -30_i64 as u64, u64::max),
            invalid_x64_atomic_unsigned(I::new_x64_atomic_max_unsigned, 10, 30, u64::max),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_mul_high() {
        use crate::machine_state::registers::NonZeroXRegister::*;

        let test_mul_high = |constructor: fn(
            NonZeroXRegister,
            NonZeroXRegister,
            NonZeroXRegister,
            InstrWidth,
        ) -> I,
                             lhs_reg: NonZeroXRegister,
                             lhs_val: u64,
                             rhs_reg: NonZeroXRegister,
                             rhs_val: u64,
                             expected: u64|
         -> Scenario {
            ScenarioBuilder::default()
                .with_xreg(lhs_reg, lhs_val)
                .with_xreg(rhs_reg, rhs_val)
                .set_instructions(&[constructor(x2, lhs_reg, rhs_reg, Compressed)])
                .set_assert_hook(assert_hook!(|core| {
                    assert_eq!(core.hart.xregisters.read_nz(x2), expected);
                }))
                .build()
        };

        let scenarios = vec![
            // MULH (Signed × Signed)
            test_mul_high(
                I::new_x64_mul_high_signed,
                x1,
                i64::MAX as u64,
                x3,
                i64::MAX as u64,
                (((i64::MAX as i128) * (i64::MAX as i128)) >> 64) as u64,
            ),
            test_mul_high(
                I::new_x64_mul_high_signed,
                x1,
                i64::MIN as u64,
                x3,
                i64::MIN as u64,
                (((i64::MIN as i128) * (i64::MIN as i128)) >> 64) as u64,
            ),
            test_mul_high(
                I::new_x64_mul_high_signed,
                x1,
                i64::MIN as u64,
                x3,
                i64::MAX as u64,
                (((i64::MIN as i128) * (i64::MAX as i128)) >> 64) as u64,
            ),
            // MULHSU (Signed × Unsigned)
            test_mul_high(
                I::new_x64_mul_high_signed_unsigned,
                x1,
                i64::MAX as u64,
                x3,
                u64::MAX,
                (((i64::MAX as i128) * (u64::MAX as u128) as i128) >> 64) as u64,
            ),
            test_mul_high(
                I::new_x64_mul_high_signed_unsigned,
                x1,
                i64::MIN as u64,
                x3,
                u64::MAX,
                (((i64::MIN as i128) * (u64::MAX as u128) as i128) >> 64) as u64,
            ),
            test_mul_high(
                I::new_x64_mul_high_signed_unsigned,
                x1,
                -1i64 as u64,
                x3,
                u64::MAX,
                (-((u64::MAX as u128) as i128) >> 64) as u64,
            ),
            // MULHU (Unsigned × Unsigned)
            test_mul_high(
                I::new_x64_mul_high_unsigned,
                x1,
                u64::MAX,
                x3,
                u64::MAX,
                (((u64::MAX as u128) * (u64::MAX as u128)) >> 64) as u64,
            ),
            test_mul_high(
                I::new_x64_mul_high_unsigned,
                x1,
                i64::MIN as u64,
                x3,
                2u64,
                (((i64::MIN as u64 as u128) * (2u128)) >> 64) as u64,
            ),
            test_mul_high(I::new_x64_mul_high_unsigned, x1, 0u64, x3, u64::MAX, 0u64),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_jit_x32_atomic_loadstore() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::*;
        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;
        const ADDRESS_BASE_ATOMICS: u64 = MEMORY_SIZE / 2;

        let scenarios = vec![
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory.write(ADDRESS_BASE_ATOMICS, 100).unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // normal x32-atomic-load followed by x32-atomic-store.
                    I::new_x32_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x32_atomic_store(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);

                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS).unwrap();
                    assert_eq!(res, 200, "Found {res}, expected 200");
                }))
                .build(),
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(ADDRESS_BASE_ATOMICS + 4, 100)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS + 2)
                .set_instructions(&[
                    // x32-atomic-load with an address that is not aligned.
                    I::new_x32_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // A failed atomic load does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x4, ADDRESS_BASE_ATOMICS + 2)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // x32-atomic-store with an address not aligned.
                    I::new_x32_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x32_atomic_store(x3, x4, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // The failed atomic operation does not count as a full step
                    1,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    // Failure due to unaligned address should not modify the value in `rd`.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x4, ADDRESS_BASE_ATOMICS + 100)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // x32-atomic-store with an address outside the reservation set.
                    I::new_x32_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x32_atomic_store(x3, x4, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // Failure due to address outside the reservation set
                    // should set the value in `rd` to 1.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 1);

                    // The value in memory should not be modified.
                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 100).unwrap();
                    assert_ne!(res, 200, "Found {res}, expected value not to be modified");
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x4, ADDRESS_BASE_ATOMICS + 100)
                .with_xreg(NZ::x3, 200)
                .set_instructions(&[
                    // x32-atomic-store with an address inside an expired reservation set.
                    I::new_x32_atomic_load(x3, x4, false, false, InstrWidth::Uncompressed),
                    I::new_x32_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x32_atomic_store(x3, x4, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // Failure due to address outside the current reservation set
                    // should set the value in `rd` to 1.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 1);

                    // The value in memory should not be modified.
                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 100).unwrap();
                    assert_ne!(res, 200, "Found {res}, expected value not to be modified");
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_jit_x64_atomic_loadstore() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::*;
        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;
        const ADDRESS_BASE_ATOMICS: u64 = MEMORY_SIZE / 2;

        let scenarios = vec![
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory.write(800, 100).unwrap();
                }))
                .with_xreg(NZ::x1, 800)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // normal x64-atomic-load followed by x64-atomic-store.
                    I::new_x64_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x64_atomic_store(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);

                    let res: u64 = core.main_memory.read(800).unwrap();
                    assert_eq!(res, 200, "Found {res}, expected 200");
                }))
                .build(),
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(ADDRESS_BASE_ATOMICS + 4, 100)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS + 4)
                .set_instructions(&[
                    // x64-atomic-load with an address that is not aligned.
                    I::new_x64_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // A failed atomic load does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x4, ADDRESS_BASE_ATOMICS + 4)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // x64-atomic-store with an address not aligned.
                    I::new_x64_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x64_atomic_store(x3, x4, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // The failed atomic operation does not count as a full step
                    1,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    // Failure due to unaligned address should not modify the value in `rd`.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x4, ADDRESS_BASE_ATOMICS + 80)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // x64-atomic-store with an address outside the reservation set.
                    I::new_x64_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x64_atomic_store(x3, x4, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // Failure due to address outside the reservation set
                    // should set the value in `rd` to 1.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 1);

                    // The value in memory should not be modified.
                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 100).unwrap();
                    assert_ne!(res, 200, "Found {res}, expected value not to be modified");
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x4, ADDRESS_BASE_ATOMICS + 80)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // x64-atomic-store with an address inside an expired reservation set.
                    I::new_x64_atomic_load(x3, x4, false, false, InstrWidth::Uncompressed),
                    I::new_x64_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x64_atomic_store(x3, x4, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // Failure due to address outside the current reservation set
                    // should set the value in `rd` to 1.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 1);

                    // The value in memory should not be modified.
                    let res: u64 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 100).unwrap();
                    assert_ne!(res, 200, "Found {res}, expected value not to be modified");
                }))
                .build(),
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, 800)
                .with_xreg(NZ::x4, 804)
                .with_xreg(NZ::x2, 200)
                .set_instructions(&[
                    // x32-atomic-store with an address in an x64-atomic-load reservation set.
                    I::new_x64_atomic_load(x3, x1, false, false, InstrWidth::Uncompressed),
                    I::new_x32_atomic_store(x3, x4, x2, false, false, InstrWidth::Compressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    // Success due to address outside the current reservation set
                    // should set the value in `rd` to 0.
                    let value: u64 = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);

                    // The value in memory should be modified.
                    let res: u64 = core.main_memory.read(804).unwrap();
                    assert_eq!(res, 200, "Found {res}, expected value to be modified");
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_rem() {
        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::x1;
        use crate::machine_state::registers::XRegister::x3;

        let test_rem =
            |constructor: fn(NonZeroXRegister, XRegister, XRegister, InstrWidth) -> I,
             lhs_val: u64,
             rhs_val: u64,
             expected: u64|
             -> Scenario {
                ScenarioBuilder::default()
                    .with_xreg(NZ::x1, lhs_val)
                    .with_xreg(NZ::x3, rhs_val)
                    .set_instructions(&[constructor(NZ::x2, x1, x3, Compressed)])
                    .set_assert_hook(assert_hook!(|core| {
                        assert_eq!(core.hart.xregisters.read_nz(NZ::x2), expected);
                    }))
                    .build()
            };

        let scenarios = vec![
            // REM (Signed 64-bit) tests
            test_rem(I::new_x64_rem_signed, 20, 6, 2),
            test_rem(
                I::new_x64_rem_signed,
                (-20i64) as u64,
                (-6i64) as u64,
                (-2i64) as u64,
            ),
            test_rem(I::new_x64_rem_signed, 20, (-6i64) as u64, 2),
            test_rem(I::new_x64_rem_signed, (-20i64) as u64, 6, (-2i64) as u64),
            // Special cases for signed remainder
            test_rem(I::new_x64_rem_signed, i64::MIN as u64, (-1i64) as u64, 0),
            test_rem(I::new_x64_rem_signed, 5, 0, 5),
            // REMU (Unsigned 64-bit) tests
            test_rem(I::new_x64_rem_unsigned, 20, 6, 2),
            test_rem(I::new_x64_rem_unsigned, 7, 3, 1),
            test_rem(I::new_x64_rem_unsigned, u64::MAX, 2, 1),
            test_rem(I::new_x64_rem_unsigned, 5, 0, 5),
            // REM (Signed 32-bit) tests
            test_rem(I::new_x32_rem_signed, 20, 6, 2),
            test_rem(
                I::new_x32_rem_signed,
                (-20i32) as u32 as u64,
                (-6i32) as u32 as u64,
                (-2i32) as u64,
            ),
            test_rem(I::new_x32_rem_signed, 20, (-6i32) as u32 as u64, 2),
            test_rem(
                I::new_x32_rem_signed,
                (-20i32) as u32 as u64,
                6,
                (-2i32) as u64,
            ),
            // Special cases for 32-bit signed remainder
            test_rem(
                I::new_x32_rem_signed,
                i32::MIN as u32 as u64,
                (-1i32) as u32 as u64,
                0,
            ),
            test_rem(I::new_x32_rem_signed, 5, 0, 5),
            // Test truncation and sign extension
            test_rem(
                I::new_x32_rem_signed,
                0xFFFFFFFF_00000005, // Should be truncated to 5
                3,
                2,
            ),
            test_rem(
                I::new_x32_rem_signed,
                (-5i32) as u32 as u64,
                3,
                (-2i32) as u64,
            ),
            // REMU (Unsigned 32-bit) tests
            test_rem(I::new_x32_rem_unsigned, 20, 6, 2),
            test_rem(I::new_x32_rem_unsigned, 7, 3, 1),
            test_rem(I::new_x32_rem_unsigned, u32::MAX as u64, 2, 1),
            test_rem(I::new_x32_rem_unsigned, 5, 0, 5),
            // Test truncation
            test_rem(
                I::new_x32_rem_unsigned,
                0xFFFFFFFF_00000005, // Should be truncated to 5
                3,
                2,
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_atomic_swap() {
        use crate::machine_state::instruction::Instruction as I;
        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::x1;
        use crate::machine_state::registers::XRegister::x2;
        use crate::machine_state::registers::XRegister::x3;

        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;
        const ADDRESS_BASE_ATOMICS: u64 = MEMORY_SIZE / 2;

        let test_atomic_swap =
            |constructor: fn(XRegister, XRegister, XRegister, bool, bool, InstrWidth) -> I,
             addr: u64,
             val: u64,
             expected_rd: u64,
             expected_mem: u64|
             -> Scenario {
                ScenarioBuilder::default()
                    .with_xreg(NZ::x1, addr)
                    .with_xreg(NZ::x3, val)
                    .set_setup_hook(setup_hook!(|core| {
                        core.main_memory
                            .set_all_readable_writeable(NoopMemoryGovernanceListener);
                        core.main_memory.write(addr, expected_rd).unwrap();
                    }))
                    .set_instructions(&[constructor(
                        x2,
                        x1,
                        x3,
                        false,
                        false,
                        InstrWidth::Uncompressed,
                    )])
                    .set_assert_hook(assert_hook!(|core| {
                        // Check rd gets the original memory value
                        assert_eq!(
                            core.hart.xregisters.read(x2),
                            expected_rd,
                            "rd value mismatch"
                        );
                        // Check memory gets the new value from rs2
                        let mem_val: u64 = core.main_memory.read(addr).unwrap();
                        assert_eq!(mem_val, expected_mem, "memory value mismatch");
                    }))
                    .build()
            };

        let scenarios = vec![
            // 32-bit atomic swap (4-byte aligned address)
            test_atomic_swap(
                I::new_x32_atomic_swap,
                ADDRESS_BASE_ATOMICS, // 4-byte aligned address
                0x200,                // New value to swap
                0x100,                // Expected original value in rd
                0x200,                // Expected new value in memory
            ),
            // 64-bit atomic swap (8-byte aligned address)
            test_atomic_swap(
                I::new_x64_atomic_swap,
                ADDRESS_BASE_ATOMICS + 8, // 8-byte aligned address
                0x200,                    // New value to swap
                0x100,                    // Expected original value in rd
                0x200,                    // Expected new value in memory
            ),
            // 32-bit atomic swap with truncation
            test_atomic_swap(
                I::new_x32_atomic_swap,
                ADDRESS_BASE_ATOMICS + 16, // Another 4-byte aligned address
                0xFFFFFFFF_00000200,       // Value that will be truncated to 32-bits
                0x100,                     // Expected original value in rd
                0x200,                     // Expected truncated value in memory
            ),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_x32_atomic_arithmetic() {
        use Instruction as I;

        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::*;

        const MEMORY_SIZE: u64 = M4K::TOTAL_BYTES.get() as u64;

        const ADDRESS_BASE_ATOMICS: u64 = MEMORY_SIZE / 2;

        type ConstructAtomicFn = fn(
            rd: XRegister,
            rs1: XRegister,
            rs2: XRegister,
            aq: bool,
            rl: bool,
            width: InstrWidth,
        ) -> I;

        let valid_x32_atomic = |constructor: ConstructAtomicFn,
                                val1: u32,
                                val2: u32,
                                fun: fn(u32, u32) -> u32|
         -> Scenario {
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(ADDRESS_BASE_ATOMICS, val1 as i32)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS)
                .with_xreg(NZ::x2, val2 as u64)
                .set_instructions(&[
                    constructor(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let value = core.hart.xregisters.read(x3);
                    assert_eq!(value, val1 as i32 as u64);
                    let expected = fun(val1, val2);
                    let res: u32 = core.main_memory.read(ADDRESS_BASE_ATOMICS).unwrap();
                    assert_eq!(res, expected);
                }))
                .build()
        };

        let invalid_x32_atomic = |constructor: ConstructAtomicFn,
                                  val1: u32,
                                  val2: u32,
                                  fun: fn(u32, u32) -> u32|
         -> Scenario {
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    core.main_memory
                        .write(ADDRESS_BASE_ATOMICS + 2, val1 as i32)
                        .unwrap();
                }))
                .with_xreg(NZ::x1, ADDRESS_BASE_ATOMICS + 2)
                .with_xreg(NZ::x2, val2 as u64)
                .set_instructions(&[
                    constructor(x3, x1, x2, false, false, InstrWidth::Uncompressed),
                    I::new_nop(InstrWidth::Compressed),
                ])
                .set_expected_steps(
                    // A failed atomic operation does not count as a full step
                    0,
                )
                .set_expected_exception(Exception::StoreAMOAccessFault)
                .set_assert_hook(assert_hook!(|core| {
                    let value = core.hart.xregisters.read(x3);
                    assert_eq!(value, 0);
                    let expected = fun(val1, val2);
                    let res: u32 = core.main_memory.read(ADDRESS_BASE_ATOMICS + 2).unwrap();
                    assert_eq!(res, val1, "Found {value:x}, expected {expected:x}");
                }))
                .build()
        };

        let bitwise_xor = |x: u32, y: u32| x ^ y;
        let bitwise_and = |x: u32, y: u32| x & y;
        let bitwise_or = |x: u32, y: u32| x | y;
        let signed_min = |x: u32, y: u32| (x as i32).min(y as i32) as u32;
        let signed_max = |x: u32, y: u32| (x as i32).max(y as i32) as u32;
        let unsigned_min = |x: u32, y: u32| x.min(y);
        let unsigned_max = |x: u32, y: u32| x.max(y);

        let scenarios = vec![
            valid_x32_atomic(I::new_x32_atomic_add, 10, 20, u32::wrapping_add),
            invalid_x32_atomic(I::new_x32_atomic_add, 10, 20, u32::wrapping_add),
            valid_x32_atomic(
                I::new_x32_atomic_add,
                0xFFFF_FFFF,
                0xFFFF_FFFF,
                u32::wrapping_add,
            ),
            invalid_x32_atomic(
                I::new_x32_atomic_add,
                0xFFFF_FFFF,
                0xFFFF_FFFF,
                u32::wrapping_add,
            ),
            valid_x32_atomic(I::new_x32_atomic_xor, 10, 20, bitwise_xor),
            invalid_x32_atomic(I::new_x32_atomic_xor, 10, 20, bitwise_xor),
            valid_x32_atomic(I::new_x32_atomic_xor, 0xFF00_00FF, 0x00FF_FF00, bitwise_xor),
            invalid_x32_atomic(I::new_x32_atomic_xor, 0xFFFF_FFFF, 0xFFFF_FFFF, bitwise_xor),
            valid_x32_atomic(I::new_x32_atomic_xor, 10, 20, bitwise_xor),
            invalid_x32_atomic(I::new_x32_atomic_xor, 10, 20, bitwise_xor),
            valid_x32_atomic(I::new_x32_atomic_xor, 0xFFFF_FFFF, 0xFFFF_FFFF, bitwise_xor),
            valid_x32_atomic(I::new_x32_atomic_and, 10, 20, bitwise_and),
            invalid_x32_atomic(I::new_x32_atomic_and, 10, 20, bitwise_and),
            valid_x32_atomic(I::new_x32_atomic_and, 0xFF00_00FF, 0x00FF_FF00, bitwise_and),
            invalid_x32_atomic(I::new_x32_atomic_and, 0xFFFF_FFFF, 0xFFFF_FFFF, bitwise_and),
            valid_x32_atomic(I::new_x32_atomic_or, 10, 20, bitwise_or),
            invalid_x32_atomic(I::new_x32_atomic_or, 10, 20, bitwise_or),
            valid_x32_atomic(I::new_x32_atomic_min_unsigned, 10, 20, unsigned_min),
            invalid_x32_atomic(I::new_x32_atomic_min_unsigned, 10, 20, unsigned_min),
            valid_x32_atomic(I::new_x32_atomic_min_signed, 10, 20, signed_min),
            invalid_x32_atomic(I::new_x32_atomic_min_signed, 10, 20, signed_min),
            valid_x32_atomic(I::new_x32_atomic_max_unsigned, 10, 20, unsigned_max),
            invalid_x32_atomic(I::new_x32_atomic_max_unsigned, 10, 20, unsigned_max),
            valid_x32_atomic(I::new_x32_atomic_max_signed, 10, 20, signed_max),
            invalid_x32_atomic(I::new_x32_atomic_max_signed, 10, 20, signed_max),
        ];

        for scenario in scenarios {
            scenario.run();
        }
    }

    #[test]
    fn test_f64_from_x64_unsigned_jit() {
        use Instruction as I;

        use crate::machine_state::registers::FRegister::*;
        use crate::machine_state::registers::NonZeroXRegister as NZ;
        use crate::machine_state::registers::XRegister::*;

        let scenarios = vec![
            ScenarioBuilder::default()
                .with_xreg(NZ::x1, 13872)
                .set_instructions(&[
                    I::new_f64_from_x64_unsigned(
                        f2,
                        x1,
                        InstrRoundingMode::Static(RoundingMode::RTZ),
                        Compressed,
                    ),
                    I::new_nop(InstrWidth::Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let res = core.hart.fregisters.read(f2);
                    let expected: FValue = (Double::from_u128_r(13872u128, Round::TowardZero))
                        .value
                        .into();
                    assert_eq!(res, expected, "Expected {expected:?}, found {res:?}");
                }))
                .build(),
            ScenarioBuilder::default()
                .set_setup_hook(setup_hook!(|core| {
                    // resets the rounding mode in `frm` of `fcsr` register to NTE.
                    core.hart.csregisters.reset();
                }))
                .with_xreg(NZ::x1, 13872)
                .set_instructions(&[
                    I::new_f64_from_x64_unsigned(f2, x1, InstrRoundingMode::Dynamic, Compressed),
                    I::new_nop(InstrWidth::Uncompressed),
                ])
                .set_assert_hook(assert_hook!(|core| {
                    let res = core.hart.fregisters.read(f2);
                    let expected: FValue =
                        (Double::from_u128_r(13872u128, Round::NearestTiesToEven))
                            .value
                            .into();
                    assert_eq!(res, expected, "Expected {expected:?}, found {res:?}");
                }))
                .build(),
        ];

        for scenario in scenarios {
            scenario.run();
        }

        //invalid csr repr
    }
}
