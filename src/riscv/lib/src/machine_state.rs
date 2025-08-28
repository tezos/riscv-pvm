// SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

pub mod block_cache;
pub(crate) mod csregisters;
pub(crate) mod hart_state;
pub mod instruction;
pub mod memory;
pub(crate) mod registers;
pub(crate) mod reservation_set;

use std::num::NonZeroU64;
use std::ops::Bound;

use block_cache::BlockCache;
use block_cache::block::Block;
use hart_state::HartState;
use hart_state::HartStateLayout;
use instruction::Instruction;
use memory::Address;
use memory::BadMemoryAccess;
use memory::Memory;
use memory::MemoryConfig;
use memory::MemoryGovernanceError;

use crate::bits::u64;
use crate::machine_state::block_cache::BlockCacheConfig;
use crate::parser::instruction::InstrWidth;
use crate::parser::is_compressed;
use crate::parser::parse_compressed_instruction;
use crate::parser::parse_uncompressed_instruction;
use crate::program::Program;
use crate::pvm::linux::signals::NO_HANDLER;
use crate::pvm::linux::signals::Signal;
use crate::pvm::linux::signals::SignalActions;
use crate::pvm::linux::signals::SignalActionsLayout;
use crate::range_utils::bound_saturating_sub;
use crate::range_utils::less_than_bound;
use crate::range_utils::unwrap_bound;
use crate::state::NewState;
use crate::state_backend as backend;
use crate::state_backend::ManagerReadWrite;
use crate::traps::Exception;

/// Layout for the machine 'run state' - which contains everything required for the running of
/// instructions.
pub type MachineCoreStateLayout<MC> = (
    HartStateLayout,
    <MC as MemoryConfig>::Layout,
    SignalActionsLayout,
);

/// The part of the machine state required to run (almost all) instructions.
///
/// Namely, things that are required to fetch instructions, but not run them, should be placed
/// elsewhere in [`MachineState`].
///
/// Certain instructions (e.g. `FENCE.I` may invalidate other parts of the state, but this are
/// small in number).
pub struct MachineCoreState<MC: memory::MemoryConfig, M: backend::ManagerBase> {
    pub hart: HartState<M>,
    pub main_memory: MC::State<M>,
    pub signal_actions: SignalActions<M>,
}

impl<MC: memory::MemoryConfig, M: backend::ManagerBase> MachineCoreState<MC, M> {
    #[inline]
    fn update_pc(&mut self, instr_pc: Address, update: ProgramCounterUpdate<Address>)
    where
        M: backend::ManagerWrite,
    {
        let pc = match update {
            ProgramCounterUpdate::Set(address) => address,
            ProgramCounterUpdate::Next(width) => instr_pc.wrapping_add(width as u64),
            ProgramCounterUpdate::Relative(offset) => instr_pc.wrapping_add_signed(offset),
        };

        self.hart.pc.write(pc);
    }

    /// Bind the machine state to the given allocated space.
    pub fn bind(space: backend::AllocatedOf<MachineCoreStateLayout<MC>, M>) -> Self {
        Self {
            hart: HartState::bind(space.0),
            main_memory: MC::bind(space.1),
            signal_actions: SignalActions::bind(space.2),
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: backend::FnManager<backend::Ref<'a, M>>>(
        &'a self,
    ) -> backend::AllocatedOf<MachineCoreStateLayout<MC>, F::Output> {
        (
            self.hart.struct_ref::<F>(),
            MC::struct_ref::<_, F>(&self.main_memory),
            self.signal_actions.struct_ref::<F>(),
        )
    }

    /// Reset the machine state.
    pub fn reset(&mut self)
    where
        M: backend::ManagerReadWrite,
    {
        self.hart.reset(memory::FIRST_ADDRESS);
        self.main_memory.reset();
        self.signal_actions.reset();
    }

    /// Fetch the 16 bits of an instruction at the given physical address.
    #[inline(always)]
    fn fetch_instr_halfword(
        &self,
        phys_addr: Address,
    ) -> Result<memory::InstructionData<u16>, Exception>
    where
        M: backend::ManagerRead,
    {
        self.main_memory
            .read_exec(phys_addr)
            .map_err(|_: BadMemoryAccess| Exception::InstructionAccessFault)
    }

    /// Fetch instruction from the address given by program counter
    /// The spec stipulates translation is performed for each byte respectively.
    /// However, we assume the `raw_pc` is 2-byte aligned.
    #[inline]
    fn fetch_instr(
        &mut self,
        addr: Address,
    ) -> Result<memory::InstructionData<Instruction>, Exception>
    where
        M: backend::ManagerReadWrite,
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

/// The alignment of a stack pointer in RISC-V's ABI
/// See RISC-V ABIs Specification Chapter 2.1
pub const RISCV_ABI_SP_ALIGNMENT: NonZeroU64 =
    NonZeroU64::new(16).expect("Alignment must be non-zero");

impl<MC: memory::MemoryConfig, M: backend::ManagerBase> NewState<M> for MachineCoreState<MC, M> {
    fn new() -> Self
    where
        M: backend::ManagerAlloc,
    {
        Self {
            hart: HartState::new(),
            main_memory: NewState::new(),
            signal_actions: SignalActions::new(),
        }
    }
}

impl<MC: memory::MemoryConfig, M: backend::ManagerClone> Clone for MachineCoreState<MC, M> {
    fn clone(&self) -> Self {
        Self {
            hart: self.hart.clone(),
            main_memory: self.main_memory.clone(),
            signal_actions: self.signal_actions.clone(),
        }
    }
}

/// Layout for the machine state - everything required to fetch & run instructions.
pub type MachineStateLayout<MC, BCC> = (
    MachineCoreStateLayout<MC>,
    <BCC as BlockCacheConfig>::Layout,
);

/// The machine state contains everything required to fetch & run instructions.
pub struct MachineState<
    MC: memory::MemoryConfig,
    BCC: block_cache::BlockCacheConfig,
    B: Block<MC, M>,
    M: backend::ManagerBase,
> {
    pub core: MachineCoreState<MC, M>,
    pub block_cache: BCC::State<MC, B, M>,

    /// The block builder is used to optimise block execution. For example, just-in-time compiling
    /// them to the host architecture.
    pub block_builder: B::BlockBuilder,
}

impl<
    MC: memory::MemoryConfig,
    BCC: BlockCacheConfig,
    B: Block<MC, M> + Clone,
    M: backend::ManagerClone,
> Clone for MachineState<MC, BCC, B, M>
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            block_cache: self.block_cache.clone(),
            block_builder: B::BlockBuilder::default(),
        }
    }
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

impl<MC: memory::MemoryConfig, BCC: BlockCacheConfig, B: Block<MC, M>, M: backend::ManagerBase>
    MachineState<MC, BCC, B, M>
{
    /// Allocate a new machine state.
    pub fn new(block_builder: B::BlockBuilder) -> Self
    where
        M: backend::ManagerAlloc,
    {
        Self {
            core: MachineCoreState::new(),
            block_cache: BlockCache::new(),
            block_builder,
        }
    }

    /// Bind the block cache to the given allocated state and the given [block builder].
    ///
    /// [block builder]: Block::BlockBuilder
    pub fn bind(
        space: backend::AllocatedOf<MachineStateLayout<MC, BCC>, M>,
        block_builder: B::BlockBuilder,
    ) -> Self
    where
        M::ManagerRoot: ManagerReadWrite,
    {
        Self {
            core: MachineCoreState::bind(space.0),
            block_cache: BCC::bind(space.1),
            block_builder,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: backend::FnManager<backend::Ref<'a, M>>>(
        &'a self,
    ) -> backend::AllocatedOf<MachineStateLayout<MC, BCC>, F::Output> {
        (
            self.core.struct_ref::<F>(),
            BCC::struct_ref::<MC, B, M, F>(&self.block_cache),
        )
    }

    /// Reset the machine state.
    pub fn reset(&mut self)
    where
        M: backend::ManagerReadWrite,
    {
        self.core.reset();
        self.block_cache.reset();
    }

    /// Fetch & run the instruction located at address `instr_pc`.
    ///
    /// Additionally, this will push the instruction to the block cache, iff the memory address is
    /// *not* writable.
    fn run_instr_at(&mut self, addr: Address) -> Result<ProgramCounterUpdate<Address>, Exception>
    where
        M: backend::ManagerReadWrite,
    {
        let memory::InstructionData {
            data: instr,
            writable,
        } = self.core.fetch_instr(addr)?;

        // If the memory backing the instruction is writable, we do not cache it as it may
        // change at any time.
        if !writable {
            self.block_cache.push_instruction(addr, instr);
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
        M: ManagerReadWrite,
    {
        match self.step_max_inner(1).error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) fn step_max_inner(&mut self, max_steps: usize) -> StepManyResult<Exception>
    where
        M: backend::ManagerReadWrite,
    {
        let mut result = self
            .block_cache
            .complete_current_block(&mut self.core, max_steps);

        // Executing the current block may fail
        if result.error.is_some() {
            return result;
        }

        while result.steps < max_steps {
            // Obtain the pc for the next instruction to be executed
            let instr_pc = self.core.hart.pc.read();

            match self.block_cache.get_block(instr_pc) {
                Some(mut block) => {
                    let steps_remaining = max_steps - result.steps;
                    let block_result = block.run_block(
                        &mut self.core,
                        &mut self.block_builder,
                        instr_pc,
                        steps_remaining,
                    );

                    // Short-circuit if the block failed
                    if result.merge_and_return(block_result) {
                        return result;
                    }
                }

                None => match self.run_instr_at(instr_pc) {
                    Ok(update) => {
                        self.core.update_pc(instr_pc, update);
                        result.steps += 1;
                    }

                    Err(exc) => {
                        result.error = Some(exc);
                        return result;
                    }
                },
            }
        }

        result
    }

    /// Perform as many steps as the given `max_steps` bound allows. Returns the number of retired
    /// instructions.
    #[inline]
    pub fn step_max(&mut self, max_steps: Bound<usize>) -> StepManyResult<Exception>
    where
        M: backend::ManagerReadWrite,
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
    #[inline]
    pub fn step_max_handle<E>(
        &mut self,
        mut step_bounds: Bound<usize>,
        mut handle: impl FnMut(&mut Self) -> Result<bool, E>,
    ) -> StepManyResult<E>
    where
        M: backend::ManagerReadWrite,
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
                        Ok(may_continue) => {
                            if !may_continue {
                                break None;
                            }
                        }

                        Err(error) => break Some(error),
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
        mut handle: impl FnMut(&mut Self) -> Result<bool, E>,
    ) -> Result<bool, E>
    where
        M: ManagerReadWrite,
    {
        match cause {
            Exception::EnvCall => return handle(self),

            Exception::FenceI => {
                self.invalidate_instruction_caches();

                // We need to advance pc by width of the Fence.I instruction because raising the exception does not do it for us.
                self.core.hart.pc.write(
                    self.core
                        .hart
                        .pc
                        .read()
                        .wrapping_add(InstrWidth::Uncompressed as u64),
                );
            }

            Exception::IllegalInstruction => {
                self.dispatch_signal_or_trap(Signal::Sigill);
            }

            Exception::InstructionAccessFault
            | Exception::LoadAccessFault
            | Exception::StoreAMOAccessFault
            | Exception::InstructionPageFault
            | Exception::LoadPageFault
            | Exception::StoreAMOPageFault => {
                self.dispatch_signal_or_trap(Signal::Sigsegv);
            }

            // There's currently no support for breakpoints - it requires SIGTRAP
            Exception::Breakpoint => {
                self.core.hart.pc.write(0);
            }
        }

        Ok(true)
    }

    /// Install a program and set the program counter to its start.
    pub fn setup_boot(&mut self, program: &Program<MC>) -> Result<(), MachineError>
    where
        M: backend::ManagerReadWrite,
    {
        // Reset hart state & set pc to entrypoint
        self.core.hart.reset(program.entrypoint);

        // Write program to main memory and point the PC at its start
        for (addr, data) in program.segments.iter() {
            self.core.main_memory.write_all(*addr, data)?;
        }

        // Set booting Hart ID (a0) to 0
        self.core.hart.xregisters.write(registers::a0, 0);

        Ok(())
    }

    /// Invalidate all instruction caches that are sensitive to
    /// instruction fences.
    fn invalidate_instruction_caches(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.block_cache.invalidate();
    }

    fn dispatch_signal_or_trap(&mut self, signal: Signal)
    where
        M: ManagerReadWrite,
    {
        // We only currently support the `action` field.
        // TODO RV-732 Parse `sa_flags` subset.
        let handler = self.core.signal_actions.read_action(signal);

        if handler == NO_HANDLER || self.core.dispatch_signal(signal).is_err() {
            self.core.hart.pc.write(0);
        }
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

    use super::MachineState;
    use crate::machine_state::block_cache::TestCacheConfig;
    use crate::machine_state::block_cache::block::Interpreted;
    use crate::machine_state::block_cache::block::InterpretedBlockBuilder;
    use crate::machine_state::memory::M4K;
    use crate::state_backend::ManagerBase;
    use crate::state_backend::owned_backend::Owned;
    use crate::state_backend::proof_backend::ProofGen;
    use crate::state_backend::verify_backend::Verifier;

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
    pub type TestMachineOf<M> = MachineState<M4K, TestCacheConfig, Interpreted<M4K, M>, M>;

    /// Trait used to initialise a specific object - [`TestMachineStateOf<M>`] - with respect to a
    /// backend type.
    pub trait ReinitMachine<M: ManagerBase>: Sized {
        /// Provided a dirty state, reinitialize the machine state as if it was just created.
        fn reinit_machine_state(
            dirty_state: RefMut<TestMachineOf<M>>,
        ) -> RefMutOrOwned<TestMachineOf<M>>;
    }

    /// Trait used to obtain initial values for testing purposes.
    ///
    /// It is useful to choose how objects are reinitialised in tests to improve performance, for
    /// example the [`ProofGen`] would rather be newly created than be reset or cloned.
    pub trait ManagerTestInit: ManagerBase {
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
    impl<M: ManagerBase> ManagerTestInit for M
    where
        TestMachineOf<M>: ReinitMachine<M>,
    {
        type TestMachine = TestMachineOf<Self>;
    }

    impl ReinitMachine<Owned> for TestMachineOf<Owned> {
        fn reinit_machine_state(mut dirty_state: RefMut<Self>) -> RefMutOrOwned<Self> {
            dirty_state.reset();
            RefMutOrOwned::RefMut(dirty_state)
        }
    }

    impl ReinitMachine<ProofGen<Owned>> for TestMachineOf<ProofGen<Owned>> {
        fn reinit_machine_state(_dirty_state: RefMut<Self>) -> RefMutOrOwned<Self> {
            let new_state = MachineState::new(InterpretedBlockBuilder);
            RefMutOrOwned::Owned(new_state)
        }
    }

    impl ReinitMachine<Verifier> for TestMachineOf<Verifier> {
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

    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::MachineState;
    use super::block_cache::block::Interpreted;
    use super::block_cache::block::InterpretedBlockBuilder;
    use super::instruction::Instruction;
    use super::instruction::OpCode;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::machine_state::RISCV_ABI_SP_ALIGNMENT;
    use crate::machine_state::block_cache::BlockCache;
    use crate::machine_state::block_cache::TestCacheConfig;
    use crate::machine_state::instruction::Args;
    use crate::machine_state::memory;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::M64M;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a7;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::sp;
    use crate::machine_state::registers::t0;
    use crate::machine_state::registers::t2;
    use crate::machine_state::test_helpers::ManagerTestInit;
    use crate::machine_state::test_helpers::TestMachineOf;
    use crate::program::Program;
    use crate::pvm::Pvm;
    use crate::pvm::PvmLayout;
    use crate::pvm::handle_system_call;
    use crate::pvm::hooks::StdoutDebugHooks;
    use crate::pvm::linux::signals::Signal;
    use crate::pvm::linux::signals::SignalError;
    use crate::state_backend::CloneLayout;
    use crate::state_backend::FnManagerIdent;
    use crate::traps::Exception;

    backend_test!(test_step, F, {
        let state = TestMachineOf::<F>::new(InterpretedBlockBuilder);

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

            // Since we've written a new instruction to the init_pc_addr - we need to
            // invalidate the instruction cache.
            state.invalidate_instruction_caches();

            state.step().expect("should not raise trap to EE");
            prop_assert_eq!(state.core.hart.xregisters.read(t0), init_pc_addr + 4);
            prop_assert_eq!(state.core.hart.pc.read(), jump_addr);
        });
    });

    backend_test!(test_step_env_exc, F, {
        let state = TestMachineOf::<F>::new(InterpretedBlockBuilder);

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
        let state = TestMachineOf::<F>::new(InterpretedBlockBuilder);
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

    // This test checks that view on the instruction memory is preserved across rebindings of the
    // PVM state. There are more details in the `block-cache-tester` kernel's source that is used
    // to test this.
    backend_test!(test_block_cache_state, F, {
        let base_state = {
            let mut state =
                Pvm::<M64M, TestCacheConfig, Interpreted<M64M, F>, F>::new(InterpretedBlockBuilder);

            // The `block-cache-tester` kernel is a simple kernel that needs to be built before
            // this test can run. It is located in the `/kernels/block-cache-tester` directory.
            let contents = fs::read("../../../kernels/block-cache-tester/target/riscv64gc-unknown-linux-musl/debug/block-cache-tester").expect("Could not find `block-cache-tester` kernel. Perhaps you need to build it via `make -C kernels/block-cache-tester build`?");
            let program = Program::from_elf(&contents).unwrap();

            state.setup_linux_process(&program).unwrap();

            let res = state
                .machine_state
                .step_max_handle::<()>(Bound::Unbounded, |machine| {
                    let syscall_number = machine.core.hart.xregisters.read(a7) as i64;

                    // The `block-cache-tester` kernel will issue a system call with number -1
                    // to indicate that it has reached the signal point. This is our cue that we
                    // have produced the right "start" state.
                    if syscall_number == -1 {
                        return Err(());
                    }

                    let shall_continue = handle_system_call(
                        machine,
                        &mut state.system_state,
                        &mut state.status,
                        &mut state.reveal_request,
                        StdoutDebugHooks,
                    );
                    Ok(shall_continue)
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
            let refs = base_state.struct_ref::<FnManagerIdent>();
            let refs = PvmLayout::<M64M, TestCacheConfig>::clone_allocated(refs);
            let mut state = Pvm::<M64M, TestCacheConfig, Interpreted<M64M, F>, F>::bind(
                refs,
                InterpretedBlockBuilder,
            );

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

        assert!(
            state.struct_ref::<FnManagerIdent>() == alt_state.struct_ref::<FnManagerIdent>(),
            "States aren't equal"
        );
    });

    // Ensure that cloning the machine state does not result in a stack overflow
    backend_test!(test_machine_state_cloneable, F, {
        let state = MachineState::<M1M, TestCacheConfig, Interpreted<M1M, F>, F>::new(
            InterpretedBlockBuilder,
        );

        let second = state.clone();

        assert!(
            state.struct_ref::<FnManagerIdent>() == second.struct_ref::<FnManagerIdent>(),
            "State equality expected"
        );
    });

    // This test exercises the "partial block" feature of the block cache which is needed to ensure
    // divergence of the state machine does not occur when the instructions in memory are changed
    // but the state is rebound.
    //
    // The scenario where problems could arise is when running a cached block, but we run fewer
    // instructions than the block contains. In the absence of the partial block feature, we would
    // run instructions from the memory and not from the block cache. This means depending on how
    // many instructions we run, we run potentially different instructions.
    backend_test!(test_block_cache_partial_determinism, F, {
        const CODE1: [u32; 4] = [
            0x00150513, // addi a0, a0, 1
            0x00200293, // li t0, 2
            0x02550533, // mul a0, a0, t0
            0xff5ff06f, // j -12
        ];

        const CODE1_RESULT: u64 = 15358;

        let code1_instrs: &[Instruction] = &[
            Instruction {
                opcode: OpCode::Addi,
                args: Args {
                    rd: nz::a0.into(),
                    rs1: nz::a0.into(),
                    imm: 1,
                    ..Args::DEFAULT
                },
            },
            Instruction {
                opcode: OpCode::Li,
                args: Args {
                    rd: nz::t0.into(),
                    imm: 2,
                    ..Args::DEFAULT
                },
            },
            Instruction {
                opcode: OpCode::Mul,
                args: Args {
                    rd: nz::a0.into(),
                    rs1: nz::a0.into(),
                    rs2: nz::t0.into(),
                    ..Args::DEFAULT
                },
            },
            Instruction {
                opcode: OpCode::JumpPC,
                args: Args {
                    imm: -12,
                    ..Args::DEFAULT
                },
            },
        ];

        const CODE2: [u32; 4] = [
            0x00150513, // addi a0, a0, 1
            0x00300293, // li t0, 3
            0x02550533, // mul a0, a0, t0
            0xff5ff06f, // j -12
        ];

        const CODE2_RESULT: u64 = 856209;

        let code2_instrs: &[Instruction] = &[
            Instruction {
                opcode: OpCode::Addi,
                args: Args {
                    rd: nz::a0.into(),
                    rs1: nz::a0.into(),
                    imm: 1,
                    ..Args::DEFAULT
                },
            },
            Instruction {
                opcode: OpCode::Li,
                args: Args {
                    rd: nz::t0.into(),
                    imm: 3,
                    ..Args::DEFAULT
                },
            },
            Instruction {
                opcode: OpCode::Mul,
                args: Args {
                    rd: nz::a0.into(),
                    rs1: nz::a0.into(),
                    rs2: nz::t0.into(),
                    ..Args::DEFAULT
                },
            },
            Instruction {
                opcode: OpCode::JumpPC,
                args: Args {
                    imm: -12,
                    ..Args::DEFAULT
                },
            },
        ];

        let mut state = MachineState::<M4K, TestCacheConfig, Interpreted<M4K, F>, F>::new(
            InterpretedBlockBuilder,
        );

        // Write instructions to a known part in memory.
        state
            .core
            .main_memory
            .write_instruction_unchecked(0x100, CODE1)
            .unwrap();

        // Run the block of instructions 10 times. Make sure it ran correctly. Validate that there
        // is a block entry in the block cache.
        {
            state.core.hart.pc.write(0x100);
            state.core.hart.xregisters.write(a0, 13);

            let steps = CODE1.len() * 10;
            let result = state.step_max(Bound::Included(steps));

            assert_eq!(result.steps, steps);
            assert_eq!(result.error, None);

            let param0 = state.core.hart.xregisters.read(a0);
            assert_eq!(param0, CODE1_RESULT);

            assert!(state.block_cache.get_block(0x100).is_some());
            assert_eq!(state.block_cache.get_block_instr(0x100), code1_instrs);
        }

        // Replace the instructions in memory with different ones.
        state
            .core
            .main_memory
            .write_instruction_unchecked(0x100, CODE2)
            .unwrap();

        // Run the block again 10 times but this time do it one step at a time to ensure the block
        // doesn't get run in its entirety. This exercises the "partial block" feature of the cache
        // which is needed to avoid divergence of the state machine.
        // Despite rewriting the instructions in memory, the block should run as before because we
        // haven't synchronised instruction and data memory yet.
        {
            state.core.hart.pc.write(0x100);
            state.core.hart.xregisters.write(a0, 13);

            for _ in 0..CODE2.len() * 10 {
                let result = state.step_max(Bound::Included(1));
                assert_eq!(result.steps, 1);
                assert_eq!(result.error, None);
            }

            let param0 = state.core.hart.xregisters.read(a0);
            assert_eq!(param0, CODE1_RESULT);

            assert!(state.block_cache.get_block(0x100).is_some());
            assert_eq!(state.block_cache.get_block_instr(0x100), code1_instrs);
        }

        // Flush the block cache.
        state.invalidate_instruction_caches();

        // Run the block again 10 times, this time it should run the new instructions thanks to the
        // previous flush.
        {
            state.core.hart.pc.write(0x100);
            state.core.hart.xregisters.write(a0, 13);

            for _ in 0..CODE2.len() * 10 {
                let result = state.step_max(Bound::Included(1));
                assert_eq!(result.steps, 1);
                assert_eq!(result.error, None);
            }

            let param0 = state.core.hart.xregisters.read(a0);
            assert_eq!(param0, CODE2_RESULT);

            assert!(state.block_cache.get_block(0x100).is_some());
            assert_eq!(state.block_cache.get_block_instr(0x100), code2_instrs);
        }
    });

    backend_test!(test_signal_context, F, {
        let mut state = MachineState::<M4K, TestCacheConfig, Interpreted<M4K, F>, F>::new(
            InterpretedBlockBuilder,
        );

        state.reset();
        state.core.main_memory.set_all_readable_writeable();

        let stack_top = M4K::TOTAL_BYTES as u64;
        state.core.hart.xregisters.write(sp, stack_top);

        let init_pc = 0xFE;
        assert!(init_pc % RISCV_ABI_SP_ALIGNMENT.get() != 0);
        state.core.hart.pc.write(init_pc);

        state.core.push_signal_context(Signal::Sigfpe).unwrap();
        let pc = state.core.pop_signal_context().unwrap();
        assert_eq!(pc, init_pc);
    });

    // RV-757: Test for bugfix where previously a modified stack could cause a panic.
    backend_test!(test_signal_index_fix, F, {
        let mut state = MachineState::<M4K, TestCacheConfig, Interpreted<M4K, F>, F>::new(
            InterpretedBlockBuilder,
        );

        state.reset();
        state.core.main_memory.set_all_readable_writeable();

        let stack_top = M4K::TOTAL_BYTES as u64;
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
