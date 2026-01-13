// SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

pub(crate) mod csregisters;
pub(crate) mod hart_state;
pub mod instruction;
pub mod memory;
pub mod page_cache;
pub(crate) mod registers;
pub(crate) mod reservation_set;

use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::ops::Bound;
use std::ops::ControlFlow;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use hart_state::HartState;
use instruction::Instruction;
use memory::Address;
use memory::BadMemoryAccess;
use memory::Memory;
use memory::MemoryConfig;
use memory::MemoryGovernanceError;
use memory::Permissions;
use memory::listener::MemoryGovernanceListener;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::components::data_space::CloneDataSpaceMode;
use octez_riscv_data::components::data_space::DataSpaceMode;
use octez_riscv_data::components::data_space::EncodeDataSpaceMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use page_cache::CodePage;
use page_cache::EmptyPageCache;
use page_cache::PageCache;
use perfect_derive::perfect_derive;

use crate::bits::u64;
use crate::exceptions::Exception;
use crate::log;
pub use crate::machine_state::registers::NonZeroXRegister;
use crate::parser::instruction::InstrWidth;
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::parser::parse_uncompressed_instruction;
use crate::program::Program;
use crate::pvm::linux::signals::Signal;
use crate::pvm::linux::signals::SignalActions;
use crate::range_utils::bound_saturating_sub;
use crate::range_utils::less_than_bound;
use crate::range_utils::unwrap_bound;

/// The part of the machine state required to run (almost all) instructions.
///
/// Namely, things that are required to fetch instructions, but not run them, should be placed
/// elsewhere in [`MachineState`].
///
/// Certain instructions (e.g. `FENCE.I` may invalidate other parts of the state, but this are
/// small in number).
#[perfect_derive(PartialEq, Eq)]
pub struct MachineCoreState<MC: memory::MemoryConfig, M: Mode> {
    pub hart: HartState<M>,
    pub main_memory: MC::State<M>,
    pub signal_actions: SignalActions<M>,
}

impl<MC: memory::MemoryConfig, M: Mode> MachineCoreState<MC, M> {
    /// Update the hart's pc given the update and explicitly given the current value of pc
    #[inline]
    pub(crate) fn update_pc(&mut self, instr_pc: Address, update: ProgramCounterUpdate<Address>)
    where
        M: AtomMode,
    {
        let pc = match update {
            ProgramCounterUpdate::Set(address) => address,
            ProgramCounterUpdate::Next(width) => instr_pc.wrapping_add(width as u64),
            ProgramCounterUpdate::Relative(offset) => instr_pc.wrapping_add_signed(offset),
        };

        self.hart.pc.write(pc);
    }

    /// Reset the machine state.
    pub fn reset(&mut self, listener: impl MemoryGovernanceListener)
    where
        M: AtomMode + DataSpaceMode,
    {
        self.hart.reset(memory::FIRST_ADDRESS);
        self.main_memory.reset(listener);
        self.signal_actions.reset();
    }

    /// Fetch the 16 bits of an instruction at the given physical address.
    #[inline(always)]
    fn fetch_instr_halfword(
        &self,
        phys_addr: Address,
    ) -> Result<memory::InstructionData<u16>, Exception>
    where
        M: AtomMode + DataSpaceMode,
    {
        self.main_memory
            .read_exec(phys_addr)
            .map_err(|_: BadMemoryAccess| Exception::InstructionAccessFault)
    }

    /// Fetch instruction from the address given by program counter
    /// The spec stipulates translation is performed for each byte respectively.
    /// However, we assume the `raw_pc` is 2-byte aligned.
    #[inline]
    fn fetch_instr(&self, addr: Address) -> Result<memory::InstructionData<Instruction>, Exception>
    where
        M: AtomMode + DataSpaceMode,
    {
        let lower = self.fetch_instr_halfword(addr)?;

        // The reasons to provide the second half in the lambda is
        // because those bytes may be inaccessible or may trigger an exception when read.
        // Hence we can't read all 4 bytes eagerly.
        let instruction_data = if is_compressed(lower.data) {
            let instr = parse_compressed_instruction(lower.data);
            let instr = Instruction::from(&instr);

            memory::InstructionData {
                data: instr,
                writable: lower.writable,
            }
        } else {
            let next_addr = addr + 2;
            let upper = self.fetch_instr_halfword(next_addr)?;

            let combined = lower.combine_with_upper(upper);
            let instr = parse_uncompressed_instruction(combined.data);
            let instr = Instruction::from(&instr);

            memory::InstructionData {
                data: instr,
                writable: combined.writable,
            }
        };

        Ok(instruction_data)
    }
}

impl<MC, M, F> Foldable<F> for MachineCoreState<MC, M>
where
    MC: MemoryConfig,
    M: Mode,
    F: Fold,
    HartState<M>: Foldable<F>,
    MC::State<M>: Foldable<F>,
    SignalActions<M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.hart);
        builder.add(&self.main_memory);
        builder.add(&self.signal_actions);
        builder.done()
    }
}

impl<MC> FromProof for MachineCoreState<MC, Verify>
where
    MC: MemoryConfig,
{
    fn from_proof<D: octez_riscv_data::merkle_proof::Deserialiser>(
        proof: D,
    ) -> octez_riscv_data::merkle_proof::SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, hart) = proof.next_branch()?;
        let (proof, main_memory) = proof.next_branch_with(MC::state_from_proof)?;
        let (proof, signal_actions) = proof.next_branch()?;

        proof.done(Self {
            hart,
            main_memory,
            signal_actions,
        })
    }
}

/// The alignment of a stack pointer in RISC-V's ABI
/// See RISC-V ABIs Specification Chapter 2.1
pub const RISCV_ABI_SP_ALIGNMENT: NonZeroU64 =
    NonZeroU64::new(16).expect("Alignment must be non-zero");

impl<MC: memory::MemoryConfig, M: AtomMode + DataSpaceMode> Default for MachineCoreState<MC, M> {
    fn default() -> Self {
        Self {
            hart: HartState::default(),
            main_memory: Memory::default(),
            signal_actions: SignalActions::default(),
        }
    }
}

impl<MC: memory::MemoryConfig> MachineCoreState<MC, Normal> {
    /// Return a proof-generating version of this MachineCoreState.
    pub fn start_proof(&self) -> MachineCoreState<MC, Prove<'_>> {
        MachineCoreState {
            hart: self.hart.start_proof(),
            main_memory: MC::start_proof(&self.main_memory),
            signal_actions: self.signal_actions.start_proof(),
        }
    }
}

impl<MC: memory::MemoryConfig, M: CloneAtomMode + CloneDataSpaceMode> Clone
    for MachineCoreState<MC, M>
{
    fn clone(&self) -> Self {
        Self {
            hart: self.hart.clone(),
            main_memory: self.main_memory.clone_state(),
            signal_actions: self.signal_actions.clone(),
        }
    }
}

impl<MC: memory::MemoryConfig, M: CloneAtomMode + CloneDataSpaceMode> CloneState
    for MachineCoreState<MC, M>
{
    fn clone_state(&self) -> Self {
        Self {
            hart: self.hart.clone_state(),
            main_memory: self.main_memory.clone_state(),
            signal_actions: self.signal_actions.clone_state(),
        }
    }
}

impl<MC: memory::MemoryConfig, M: EncodeAtomMode + EncodeDataSpaceMode> Encode
    for MachineCoreState<MC, M>
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.hart.encode(encoder)?;
        self.main_memory.encode(encoder)?;
        self.signal_actions.encode(encoder)?;
        Ok(())
    }
}

impl<C, MC: memory::MemoryConfig> Decode<C> for MachineCoreState<MC, Normal>
where
    MC::State<Normal>: Decode<C>,
{
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            hart: Decode::decode(decoder)?,
            main_memory: Decode::decode(decoder)?,
            signal_actions: Decode::decode(decoder)?,
        })
    }
}

/// RISC-V machine state
///
/// The machine state contains everything required to fetch & run instructions.
pub struct MachineState<MC: memory::MemoryConfig, PC, M: Mode> {
    pub core: MachineCoreState<MC, M>,
    pub page_cache: PC,
}

impl<MC: memory::MemoryConfig, PC: PageCache<MC, M>, M: CloneAtomMode + CloneDataSpaceMode> Clone
    for MachineState<MC, PC, M>
{
    // TODO: RV-806: implement Clone on PageCache
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            page_cache: PC::new(),
        }
    }
}

impl<MC: memory::MemoryConfig, PC: PageCache<MC, Normal>> MachineState<MC, PC, Normal> {
    /// Return a proof-generating version of this MachineState.
    pub fn start_proof(&self) -> MachineState<MC, EmptyPageCache, Prove<'_>> {
        MachineState {
            core: self.core.start_proof(),
            page_cache: <EmptyPageCache as PageCache<MC, Prove<'_>>>::new(),
        }
    }
}

impl<MC: memory::MemoryConfig, PC: PageCache<MC, M>, M: CloneAtomMode + CloneDataSpaceMode>
    CloneState for MachineState<MC, PC, M>
{
    fn clone_state(&self) -> Self {
        Self {
            core: self.core.clone_state(),
            page_cache: PC::new(),
        }
    }
}

impl<MC> FromProof for MachineState<MC, EmptyPageCache, Verify>
where
    MC: MemoryConfig,
{
    fn from_proof<D: octez_riscv_data::merkle_proof::Deserialiser>(
        proof: D,
    ) -> octez_riscv_data::merkle_proof::SuspendedResult<D, Self> {
        let result = MachineCoreState::from_proof(proof)?;
        let result = result.map(|core| Self {
            core,
            page_cache: <EmptyPageCache as PageCache<MC, Verify>>::new(),
        });
        Ok(result)
    }
}

impl<MC, PC, M> Encode for MachineState<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: EncodeAtomMode + EncodeDataSpaceMode,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.core.encode(encoder)
    }
}

impl<C, MC, PC> Decode<C> for MachineState<MC, PC, Normal>
where
    MC: MemoryConfig,
    PC: PageCache<MC, Normal>,
    MC::State<Normal>: Decode<C>,
{
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            core: Decode::decode(decoder)?,
            page_cache: PC::new(),
        })
    }
}

impl<MC, PC, M> PartialEq for MachineState<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: Mode,
    MachineCoreState<MC, M>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.core == other.core
    }
}

impl<MC, PC, M> Eq for MachineState<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: Mode,
    MachineCoreState<MC, M>: PartialEq,
{
}

/// How to modify the program counter
#[derive(Debug, PartialEq)]
pub enum ProgramCounterUpdate<AddressRepr> {
    /// Jump to a fixed address
    Set(AddressRepr),
    /// Offset to the next instruction by current instruction width
    Next(InstrWidth),
    /// Jump to an address relative to the current program counter
    Relative(i64),
}

/// Result type when running multiple steps at a time with [`MachineState::step_max`]
#[derive(Debug, PartialEq, Eq)]
pub struct StepManyResult<E> {
    pub steps: usize,
    pub error: Option<E>,
}

impl<E> StepManyResult<E> {
    /// Initial/zero result - no steps taken, no error
    const ZERO: Self = Self {
        steps: 0,
        error: None,
    };

    /// Merge the result of two step results, returning true if an error occurred.
    ///
    /// # Behaviour
    ///
    /// Combines the number of steps from both results. The `error` field is taken from the `other`
    /// instance, overwriting the error stored in `self`.
    ///
    /// # Returns
    ///
    /// Returns `true` if an error has been merged.
    fn merge_and_return(&mut self, other: StepManyResult<E>) -> bool {
        self.steps += other.steps;
        self.error = other.error;
        self.error.is_some()
    }
}

impl<E> Default for StepManyResult<E> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<MC: memory::MemoryConfig, PC: PageCache<MC, M>, M: AtomMode + DataSpaceMode> Default
    for MachineState<MC, PC, M>
{
    fn default() -> Self {
        Self {
            core: MachineCoreState::default(),
            page_cache: PC::new(),
        }
    }
}

impl<MC: memory::MemoryConfig, PC: PageCache<MC, M>, M: Mode> MachineState<MC, PC, M> {
    /// Reset the machine state.
    pub fn reset(&mut self)
    where
        M: AtomMode + DataSpaceMode,
    {
        let listener = &mut self.page_cache;
        self.core.reset(listener);
    }

    /// Get access to the main memory, with a listener to hook into any permission updates.
    ///
    /// This is required to keep parts of [`MachineState`], that do not form part of the PVM state,
    /// synchronised with main memory so as to ensure determinism regardless of whether these
    /// additional parts of the state are populated or not.
    pub(crate) fn memory_with_listener(
        &mut self,
    ) -> (&mut MC::State<M>, impl MemoryGovernanceListener) {
        (&mut self.core.main_memory, &mut self.page_cache)
    }

    /// Fetch & run the instruction located at address `instr_pc`.
    ///
    /// Additionally, this will populate the relevant page in the page cache, iff the memory address is
    /// *not* writable.
    fn run_instr_at(&mut self, addr: Address) -> Result<ProgramCounterUpdate<Address>, Exception>
    where
        M: AtomMode + DataSpaceMode,
    {
        let memory::InstructionData {
            data: instr,
            writable,
        } = self.core.fetch_instr(addr)?;

        // If the memory backing the instruction is writable, we do not cache it as it may
        // change at any time.
        if !writable {
            self.page_cache.populate_page(addr, &self.core);
        }

        instr.run(&mut self.core)
    }

    /// Take an interrupt if available, and then
    /// perform precisely one [`Instr`] and handle the traps that may rise as a side-effect.
    ///
    /// The [`Err`] case represents an [`Exception`] that ought to be handled at a higher level.
    ///
    /// [`Instr`]: crate::parser::instruction::Instr
    #[inline]
    pub fn step(&mut self) -> Result<(), Exception>
    where
        M: AtomMode + DataSpaceMode,
    {
        match self.step_max_inner(1).error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) fn step_max_inner(&mut self, max_steps: usize) -> StepManyResult<Exception>
    where
        M: AtomMode + DataSpaceMode,
    {
        let mut result = StepManyResult::ZERO;

        while result.steps < max_steps {
            // Obtain the pc for the next instruction to be executed
            let instr_pc = self.core.hart.pc.read();

            if let Some(mut code_page) = self.page_cache.get_code_page(instr_pc) {
                let steps_remaining = max_steps - result.steps;

                let entrypoint_result = code_page.run(&mut self.core, instr_pc, steps_remaining);

                // Short-circuit if the entrypoint call failed
                if result.merge_and_return(entrypoint_result) {
                    return result;
                }

                continue;
            }

            match self.run_instr_at(instr_pc) {
                Ok(update) => {
                    self.core.update_pc(instr_pc, update);
                    result.steps += 1;
                }

                Err(exc) => {
                    result.error = Some(exc);
                    return result;
                }
            }
        }

        result
    }

    /// Perform as many steps as the given `max_steps` bound allows. Returns the number of retired
    /// instructions.
    #[inline]
    pub fn step_max(&mut self, max_steps: Bound<usize>) -> StepManyResult<Exception>
    where
        M: AtomMode + DataSpaceMode,
    {
        let mut result = StepManyResult::ZERO;

        loop {
            let iter_result = self.step_max_inner(unwrap_bound(max_steps));

            let errored = result.merge_and_return(iter_result);
            let out_of_steps = !less_than_bound(result.steps, max_steps);

            if errored || out_of_steps {
                break;
            }
        }

        result
    }

    /// Similar to [`Self::step_max`] but lets the user handle environment exceptions inside the
    /// inner step loop.
    ///
    /// The `handle` closure is called whenever an environment exception is raised. Its return value
    /// indicates whether to exit the step loop, and thereby this function. The break reason is made
    /// available via the [`StepManyResult::error`] field.
    #[inline]
    pub fn step_max_handle<E>(
        &mut self,
        mut step_bounds: Bound<usize>,
        mut handle: impl FnMut(&mut Self) -> ControlFlow<E>,
    ) -> StepManyResult<E>
    where
        M: AtomMode + DataSpaceMode,
    {
        let mut steps = 0usize;

        let error = loop {
            let result = self.step_max(step_bounds);

            steps = steps.saturating_add(result.steps);
            step_bounds = bound_saturating_sub(step_bounds, result.steps);

            match result.error {
                Some(cause) => {
                    // Raising the exception is not a completed step. Trying to handle it is.
                    // We don't have to check against `max_steps` because running the
                    // instruction that triggered the exception meant that `max_steps > 0`.
                    steps = steps.saturating_add(1);
                    step_bounds = bound_saturating_sub(step_bounds, 1);

                    match self.handle_exception(cause, &mut handle) {
                        ControlFlow::Continue(()) => {}
                        ControlFlow::Break(error) => break Some(error),
                    }
                }

                None => break None,
            }
        };

        StepManyResult { steps, error }
    }

    #[inline]
    fn handle_exception<E>(
        &mut self,
        cause: Exception,
        mut handle: impl FnMut(&mut Self) -> ControlFlow<E>,
    ) -> ControlFlow<E>
    where
        M: AtomMode + DataSpaceMode,
    {
        match cause {
            Exception::EnvCall => return handle(self),

            Exception::FenceI => {
                // We have no caches/state that is sensitive to a `fence.i` instruction.

                // We need to advance pc by width of the Fence.I instruction because raising the exception does not do it for us.
                let pc = self
                    .core
                    .hart
                    .pc
                    .read()
                    .wrapping_add(InstrWidth::Uncompressed as u64);
                self.core.hart.pc.write(pc);
            }

            Exception::ForceFetchRun => return self.handle_force_fetch_run(handle),

            Exception::IllegalInstruction => {
                self.dispatch_signal_or_trap(Signal::Sigill);
            }

            Exception::InstructionAccessFault
            | Exception::LoadAccessFault
            | Exception::StoreAMOAccessFault => {
                self.dispatch_signal_or_trap(Signal::Sigsegv);
            }

            // There's currently no support for breakpoints - it requires SIGTRAP
            Exception::Breakpoint => {
                self.core.hart.pc.write(0);
            }
        }

        ControlFlow::Continue(())
    }

    /// Handle [`Exception::ForceFetchRun`] by fetching instruction data from memory directly,
    /// then executing it.
    ///
    /// If this itself results in an exception, we handle this. *NB* this subsequent exception
    /// cannot be a `ForceFetchRun`.
    fn handle_force_fetch_run<E>(
        &mut self,
        handle: impl FnMut(&mut Self) -> ControlFlow<E>,
    ) -> ControlFlow<E>
    where
        M: AtomMode + DataSpaceMode,
    {
        let instr_pc = self.core.hart.pc.read();
        let result = self
            .core
            .fetch_instr(instr_pc)
            .and_then(|memory::InstructionData { data: instr, .. }| instr.run(&mut self.core));

        let exception = match result {
            Ok(update) => {
                self.core.update_pc(instr_pc, update);
                return ControlFlow::Continue(());
            }
            // this should never happen (as we do not parse instruction data into an instruction
            // with OpCode::ForceFetchRun). If it does though, we shouldn't crash the PVM. Instead
            // this indicates an illegal instruction has been executed.
            Err(Exception::ForceFetchRun) => {
                log::warning!(
                    "handling ForceFetchRun exception resulted in another ForceFetchRun exception"
                );

                Exception::IllegalInstruction
            }
            Err(cause) => cause,
        };

        // SAFETY: this recursive call cannot overflow the stack as the exception has
        // changed.
        self.handle_exception(exception, handle)
    }

    /// Install a program and set the program counter to its start.
    ///
    /// Returns the `program_start` and `program_end`, if successful.
    pub fn setup_boot_program(
        &mut self,
        program: &Program<MC>,
    ) -> Result<(Address, Address), MachineError>
    where
        M: AtomMode + DataSpaceMode,
    {
        let program_start = program.segments.keys().min().copied().unwrap_or(0);
        let program_end = program
            .segments
            .iter()
            .map(|(addr, data)| addr.saturating_add(data.len() as u64))
            .max()
            .unwrap_or(0);

        let (main_memory, mut listener) = self.memory_with_listener();

        let program_length = program_end.saturating_sub(program_start) as usize;
        if let Some(program_length) = NonZeroUsize::new(program_length) {
            // Allow the program to be written to main memory
            main_memory.protect_pages(
                program_start,
                program_length,
                Permissions::WRITE,
                &mut listener,
            )?;

            // Write program to main memory
            for (&addr, data) in program.segments.iter() {
                main_memory.write_all(addr, data)?;
            }

            // Remove access to the program that has just been placed into memory
            main_memory.protect_pages(
                program_start,
                program_length,
                Permissions::NONE,
                &mut listener,
            )?;
        };

        // Configure memory permissions using the ELF program headers, if present
        if let Some(program_headers) = &program.program_headers {
            for mem_perms in program_headers.permissions.iter() {
                let Some(length) = NonZeroUsize::new(mem_perms.length as usize) else {
                    continue;
                };

                main_memory.protect_pages(
                    mem_perms.start_address,
                    length,
                    mem_perms.permissions,
                    &mut listener,
                )?;
            }
        }

        Ok((program_start, program_end))
    }

    fn dispatch_signal_or_trap(&mut self, signal: Signal)
    where
        M: AtomMode + DataSpaceMode,
    {
        if self.core.dispatch_signal(signal).is_err() {
            self.core.hart.pc.write(0);
        }
    }

    #[cfg(test)]
    pub(crate) fn set_all_readable_writeable<const PAGES: usize, const TOTAL_BYTES: usize, MB>(
        &mut self,
    ) where
        MB: memory::buddy::Buddy<M>,
        MC: MemoryConfig<State<M> = memory::state::MemoryImpl<PAGES, TOTAL_BYTES, MB, M>>,
        M: AtomMode + DataSpaceMode,
    {
        let (main_memory, listener) = self.memory_with_listener();
        main_memory.set_all_readable_writeable(listener);
    }
}

impl<MC, PC, M, F> Foldable<F> for MachineState<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: Mode,
    F: Fold,
    MachineCoreState<MC, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.core.fold(builder)
    }
}

/// Errors that occur from interacting with the [MachineState]
#[derive(Debug, thiserror::Error)]
pub enum MachineError {
    #[error("Error while accessing memory")]
    MemoryError(#[from] BadMemoryAccess),

    #[error("Error while govering memory")]
    MemoryGovernanceError(#[from] MemoryGovernanceError),

    #[error("Device tree error: {0}")]
    DeviceTreeError(#[from] vm_fdt::Error),

    #[error("Memory too small to properly configure the machine")]
    MemoryTooSmall,
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::cell::RefMut;
    use std::ops::Deref;
    use std::ops::DerefMut;

    use octez_riscv_data::mode::Mode;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;

    use super::MachineState;
    use super::page_cache::EmptyPageCache;
    use crate::machine_state::memory::M4K;

    /// A wrapper to use a type `T` from either a mutable reference or an owned value.
    pub enum RefMutOrOwned<'a, T> {
        RefMut(RefMut<'a, T>),
        Owned(T),
    }

    impl<T> Deref for RefMutOrOwned<'_, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            match self {
                RefMutOrOwned::RefMut(r) => r,
                RefMutOrOwned::Owned(t) => t,
            }
        }
    }

    impl<T> DerefMut for RefMutOrOwned<'_, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            match self {
                RefMutOrOwned::RefMut(r) => r,
                RefMutOrOwned::Owned(t) => t,
            }
        }
    }

    /// Type alias for the machine state used in some tests.
    pub type TestMachineOf<M> = MachineState<M4K, EmptyPageCache, M>;

    /// Trait used to initialise a specific object - [`TestMachineStateOf<M>`] - with respect to a
    /// backend type.
    pub trait ReinitMachine<M: Mode> {
        /// Provided a dirty state, reinitialize the machine state as if it was just created.
        fn reinit_machine_state(
            dirty_state: RefMut<TestMachineOf<M>>,
        ) -> RefMutOrOwned<TestMachineOf<M>>;
    }

    /// Trait used to obtain initial values for testing purposes.
    ///
    /// It is useful to choose how objects are reinitialised in tests to improve performance, for
    /// example the [`Prove`] would rather be newly created than be reset or cloned.
    pub trait ManagerTestInit: Mode {
        /// This type is used to downcast the initialisation function to the actual type that is the
        /// subject of initialisation.
        type TestMachine: ReinitMachine<Self>;

        /// Passthrough function to reinitialize the machine state.
        fn reinit_machine_state(
            dirty_state: RefMut<TestMachineOf<Self>>,
        ) -> RefMutOrOwned<TestMachineOf<Self>> {
            Self::TestMachine::reinit_machine_state(dirty_state)
        }
    }

    // This is the place where we choose _what_ we are interested in initialising.
    impl<M: Mode> ManagerTestInit for M
    where
        TestMachineOf<M>: ReinitMachine<M>,
    {
        type TestMachine = TestMachineOf<Self>;
    }

    impl ReinitMachine<Normal> for TestMachineOf<Normal> {
        fn reinit_machine_state(mut dirty_state: RefMut<Self>) -> RefMutOrOwned<Self> {
            dirty_state.reset();
            RefMutOrOwned::RefMut(dirty_state)
        }
    }

    impl<'normal> ReinitMachine<Prove<'normal>> for TestMachineOf<Prove<'normal>> {
        fn reinit_machine_state(_dirty_state: RefMut<Self>) -> RefMutOrOwned<Self> {
            let new_state = MachineState::default();
            RefMutOrOwned::Owned(new_state)
        }
    }

    impl ReinitMachine<Verify> for TestMachineOf<Verify> {
        fn reinit_machine_state(mut dirty_state: RefMut<Self>) -> RefMutOrOwned<Self> {
            dirty_state.reset();
            RefMutOrOwned::RefMut(dirty_state)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::ops::Bound;
    use std::ops::ControlFlow;

    use octez_riscv_data::clone::CloneState;
    use octez_riscv_data::mode::Normal;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::MachineState;
    use super::instruction::Instruction;
    use super::memory::Address;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::exceptions::Exception;
    use crate::machine_state::RISCV_ABI_SP_ALIGNMENT;
    use crate::machine_state::StepManyResult;
    use crate::machine_state::memory;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::M8K;
    use crate::machine_state::memory::M64M;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::memory::Permissions;
    use crate::machine_state::memory::listener::MemoryGovernanceListener;
    use crate::machine_state::page_cache::EmptyPageCache;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::page_cache::PageCacheInterpreted;
    use crate::machine_state::page_cache::state::PageEntry;
    use crate::machine_state::registers::a7;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::sp;
    use crate::machine_state::registers::t0;
    use crate::machine_state::registers::t2;
    use crate::machine_state::test_helpers::ManagerTestInit;
    use crate::machine_state::test_helpers::TestMachineOf;
    use crate::parser::instruction::InstrWidth;
    use crate::parser::parse_uncompressed_instruction;
    use crate::program::Program;
    use crate::pvm::Pvm;
    use crate::pvm::handle_system_call;
    use crate::pvm::hooks::StdoutDebugHooks;
    use crate::pvm::linux::signals::Signal;
    use crate::pvm::linux::signals::SignalError;

    backend_test!(test_step, F, {
        let state = TestMachineOf::<F>::default();

        let state_cell = std::cell::RefCell::new(state);

        proptest!(|(
            pc_addr_offset in 0..250_u64,
            jump_addr in 0..250_u64,
        )| {
            let mut state = <F as ManagerTestInit>::reinit_machine_state(
                state_cell.borrow_mut(),
            );

            let init_pc_addr = memory::FIRST_ADDRESS + pc_addr_offset * 4;
            let jump_addr = memory::FIRST_ADDRESS + jump_addr * 4;

            // Instruction which performs a unit op (AUIPC with t0)
            const T2_ENC: u32 = 0b0_0111; // x7

            state.core.hart.pc.write(init_pc_addr);
            state.core.main_memory.write_instruction_unchecked::<u32>(init_pc_addr, (T2_ENC << 7) |
                0b0010111).expect("Storing instruction should succeed");
            state.step().expect("should not raise trap to EE");
            prop_assert_eq!(state.core.hart.xregisters.read(t2), init_pc_addr);
            prop_assert_eq!(state.core.hart.pc.read(), init_pc_addr + 4);

            // Instruction which updates pc by returning an address
            // t3 = jump_addr, (JALR imm=0, rs1=t3, rd=t0)
            const T0_ENC: u32 = 0b00101; // x5
            const OP_JALR: u32 = 0b110_0111;
            const F3_0: u32 = 0b000;

            state.core.hart.pc.write(init_pc_addr);
            state.core.main_memory.write_instruction_unchecked(init_pc_addr, (T2_ENC << 15) | (F3_0 << 12) | (T0_ENC << 7) | OP_JALR).unwrap();

            state.core.hart.xregisters.write(t2, jump_addr);
            let page_index = crate::machine_state::memory::address_to_page_index(init_pc_addr) as u64;

            // Since we've written a new instruction to the init_pc_addr - we need to
            // invalidate the page cache entry for that page.
            state.page_cache.handle_permissions_update(
                page_index..=page_index,
                Permissions::WRITE,
            );

            state.core.hart.xregisters.write(t2, jump_addr);

            state.step().expect("should not raise trap to EE");
            prop_assert_eq!(state.core.hart.xregisters.read(t0), init_pc_addr + 4);
            prop_assert_eq!(state.core.hart.pc.read(), jump_addr);
        });
    });

    backend_test!(test_step_env_exc, F, {
        let state = TestMachineOf::<F>::default();

        let state_cell = std::cell::RefCell::new(state);

        proptest!(|(
            pc_addr_offset in 0..200_u64
        )| {
            let mut state = <F as ManagerTestInit>::reinit_machine_state(
                state_cell.borrow_mut(),
            );

            let init_pc_addr = memory::FIRST_ADDRESS + pc_addr_offset * 4;

            const ECALL: u32 = 0b111_0011;

            // TEST: Raise ECALL exception ==>> environment exception
            state.core.hart.pc.write(init_pc_addr);
            state.core.main_memory.write_instruction_unchecked(init_pc_addr, ECALL).unwrap();
            let e = state.step()
                .expect_err("should raise Environment Exception");
            assert_eq!(e, Exception::EnvCall);
            prop_assert_eq!(state.core.hart.pc.read(), init_pc_addr);
        });
    });

    backend_test!(test_step_access_exception, F, {
        let state = TestMachineOf::<F>::default();
        let state_cell = std::cell::RefCell::new(state);

        proptest!(|(
            pc_addr_offset in 0..200_u64,
        )| {
            let mut state = <F as ManagerTestInit>::reinit_machine_state(
                state_cell.borrow_mut(),
            );

            let bad_address = memory::FIRST_ADDRESS.wrapping_sub((pc_addr_offset + 10) * 4);
            state.core.hart.pc.write(bad_address);

            let result = state.step();
            assert_eq!(result, Err(Exception::InstructionAccessFault));
            assert_eq!(state.core.hart.pc.read(), bad_address);
        });
    });

    // This test checks that view on the instruction memory is synchronised with data memory,
    // including across rebindings of the PVM state. There are more details in the
    // `page-cache-tester` kernel's source that is used to test this.
    #[test]
    fn test_page_cache_state() {
        let base_state = {
            let mut state = Pvm::<M64M, PageCacheInterpreted<_>, Normal>::default();

            // The `page-cache-tester` kernel is a simple kernel that needs to be built before
            // this test can run. It is located in the `/kernels/page-cache-tester` directory.
            let contents = fs::read("../../../kernels/page-cache-tester/target/riscv64gc-unknown-linux-musl/debug/page-cache-tester").expect("Could not find `page-cache-tester` kernel. Perhaps you need to build it via `make -C kernels/page-cache-tester build`?");
            let program = Program::from_elf(&contents).unwrap();

            state.setup_linux_process(&program).unwrap();

            let res = state
                .machine_state
                .step_max_handle::<()>(Bound::Unbounded, |machine| {
                    let syscall_number = machine.core.hart.xregisters.read(a7) as i64;

                    // The `page-cache-tester` kernel will issue a system call with number -1
                    // to indicate that it has reached the signal point. This is our cue that we
                    // have produced the right "start" state.
                    if syscall_number == -1 {
                        return ControlFlow::Break(());
                    }

                    handle_system_call(
                        machine,
                        &mut state.system_state,
                        &mut state.status,
                        &mut state.reveal_request,
                        StdoutDebugHooks,
                    )
                });

            assert_eq!(
                res.error,
                Some(()),
                "Program didn't make it to the signal point"
            );

            state
        };

        let alt_state = {
            // Clone the base state to get rid of any ephemeral state that was created.
            let mut state = base_state.clone_state();

            // We want to run the kernel until it exits as that is a good point to compare.
            loop {
                let _steps = state.eval_max(StdoutDebugHooks, Bound::Unbounded);

                if unsafe { state.has_exited().is_some() } {
                    break;
                }
            }

            state
        };

        let state = {
            // Take ownership within this code block.
            let mut state = base_state;

            // We want to run the kernel until it exits as that is a good point to compare.
            loop {
                let _steps = state.eval_max(StdoutDebugHooks, Bound::Unbounded);

                if unsafe { state.has_exited().is_some() } {
                    break;
                }
            }

            state
        };

        assert_eq!(
            unsafe { state.has_exited() },
            Some(0),
            "State didn't exit cleanly"
        );

        assert!(state == alt_state, "States aren't equal");
    }

    // Ensure that cloning the machine state does not result in a stack overflow
    backend_test!(test_machine_state_cloneable, F, {
        let state = MachineState::<M1M, EmptyPageCache, F>::default();

        let second = state.clone();

        assert!(state == second, "State equality expected");
    });

    // Ensure that the force-fetch-run mechanism correctly fetches instructions directly from
    // memory, and executes them.
    #[test]
    fn test_force_fetch_run() {
        // li a1, 1
        let li_bytes: u32 = 0x00100593;
        const IMMEDIATE: i64 = 1;

        let li_instr = Instruction::from(&parse_uncompressed_instruction(li_bytes));
        assert_eq!(
            li_instr,
            Instruction::new_li(nz::a1, IMMEDIATE, InstrWidth::Uncompressed),
            "Incorrect bytes {li_bytes:x} for instruction"
        );

        let li_lower = li_bytes as u16;
        let li_upper = (li_bytes >> 16) as u16;

        let run_test = |initial_pc: Address,
                        write_lower: bool,
                        write_upper: bool,
                        expected_pc: Address,
                        succeeds: bool| {
            let mut state = MachineState::<M8K, PageCacheInterpreted<_>, Normal>::default();

            state.core.hart.pc.write(initial_pc);

            if write_lower {
                state
                    .core
                    .main_memory
                    .write_instruction_unchecked(initial_pc, li_lower)
                    .unwrap();
            }

            if write_upper {
                state
                    .core
                    .main_memory
                    .write_instruction_unchecked(initial_pc + 2, li_upper)
                    .unwrap();
            }

            let mut page = PageEntry::zeroed(InterpretedCompiler);
            PageEntry::push_instructions(&mut page, initial_pc, [Instruction::DEFAULT].into_iter());

            state.page_cache.overwrite_page(initial_pc, page);

            let res: StepManyResult<()> =
                state.step_max_handle(Bound::Included(1), |_| panic!("unexpected ECall"));

            assert_eq!(state.core.hart.pc.read(), expected_pc);
            assert_eq!(res.steps, 1);
            assert_eq!(res.error, None);
            let expected_a1 = if succeeds { IMMEDIATE as u64 } else { 0 };
            assert_eq!(state.core.hart.xregisters.read_nz(nz::a1), expected_a1);
        };

        // TESTS
        // we use as failure indication pc == 0 and a1 not updated
        // -----

        let pc_within_page = 100;

        let pc_across_pages = memory::PAGE_SIZE.get().checked_sub(2).unwrap();

        // force fetch run within an executable page succeeds
        run_test(
            pc_within_page,
            true,
            true,
            pc_within_page + InstrWidth::Uncompressed as u64,
            true,
        );

        // force fetch run within a non executable page fails
        run_test(pc_within_page, false, false, 0, false);

        // force fetch run across two executable pages succeeds
        run_test(
            pc_across_pages,
            true,
            true,
            pc_across_pages + InstrWidth::Uncompressed as u64,
            true,
        );

        // force fetch run across one executable page followed by a non executable page
        run_test(pc_across_pages, true, false, 0, false);

        // force fetch run across one non-executable page followed by an executable page
        run_test(pc_across_pages, false, true, 0, false);
    }

    backend_test!(test_signal_context, F, {
        let mut state = MachineState::<M4K, EmptyPageCache, F>::default();

        state.reset();
        state.set_all_readable_writeable();

        let stack_top = M4K::TOTAL_BYTES.get() as u64;
        state.core.hart.xregisters.write(sp, stack_top);

        let init_pc = 0xFE;
        assert!(init_pc % RISCV_ABI_SP_ALIGNMENT.get() != 0);
        state.core.hart.pc.write(init_pc);

        state.core.push_signal_context(Signal::Sigstop).unwrap();
        let pc = state.core.pop_signal_context().unwrap();
        assert_eq!(pc, init_pc);
    });

    // RV-757: Test for bugfix where previously a modified stack could cause a panic.
    backend_test!(test_signal_index_fix, F, {
        let mut state = MachineState::<M4K, EmptyPageCache, F>::default();

        state.reset();
        state.set_all_readable_writeable();

        let stack_top = M4K::TOTAL_BYTES.get() as u64;
        state.core.hart.xregisters.write(sp, stack_top);

        let init_pc = 0x100;
        state.core.hart.pc.write(init_pc);

        state.core.push_signal_context(Signal::Sigfpe).unwrap();

        let signal_index_address = stack_top - 32;
        let bad_signal_index: u64 = 42;

        state
            .core
            .main_memory
            .write(signal_index_address, bad_signal_index)
            .unwrap();

        let result = state.core.pop_signal_context();
        assert_eq!(result, Err(SignalError::BadContext));
    });
}
