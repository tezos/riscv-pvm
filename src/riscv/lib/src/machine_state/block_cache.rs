// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! The block cache maps certain physical addresses to contiguous sequences of instructions, or
//! 'blocks'. These blocks will never cross page boundaries.
//!
//! Blocks are sets of instructions that are often - but not always - executed immediately after each
//! other. The main exceptions to this are:
//! - instructions may trigger exceptions (e.g. `BadMemoryAccess` when interacting with memory)
//! - branching instructions
//!
//! Specifically, blocks that contain a backwards branching instruction will often terminate with
//! that instruction, and blocks that contain a forwards jump will often contain those internally.
//!
//! This is due to the recommended behaviour mentioned in the ISA:
//! > Software should also assume that backward branches will be predicted taken and forward branches
//! > as not taken, at least the first time they are encountered.
//!
//! Therefore, sets of instructions that form tight loops will naturally form a block, and
//! sequences that may branch elsewhere - but normally fall-through - also form a block.
//!
//! Blocks must never cross page boundaries, as two pages that lie next to each other in physical
//! memory have no guarantees of being consecutive in virtual memory - even if they are at one
//! point.
//!
//! # Determinism
//!
//! Some special care is required when thinking about block execution
//! with respect to determinism.
//!
//! Blocks fundamentally rely on being executed 'as a whole'. This is
//! required to be able to make the most of various optimisations we
//! can apply to groups of instructions, while ensuring that we are
//! compatible with the remaining infrastructure of the stepper at block
//! entry/exit.
//! For example, this could include something as simple as not
//! updating the step counter until we exit a block.
//!
//! Only ever running blocks when sufficient steps remain, however,
//! causes determinism issues. Namely, when insufficient steps remain
//! to run a block, what do we do?
//!
//! The obvious solution is to fall back to the instruction cache, and
//! continue our *fetch/parse/run* cycle. This exact solution, however,
//! causes divergence: the instructions we end up executing from the
//! instruction cache, could be completely different to those stored in
//! the block cache, for the same physical address.
//!
//! This can occur due to the differring nature of cache entries between
//! the instruction & block cache: entries in the instruction cache are
//! 1 instruction per slot, whereas in the block cache it's instead a
//! block (a sequence of instructions) per slot. Therefore, an entry
//! getting overriden in the instruction/block cache at address *A*, does
//! not invalidate all instructions in the block cache that correspond
//! to *A*, as they could also exist in blocks at nearby preceding
//! addresses.
//!
//! ## Solution
//!
//! Instead, we introduce the notion of a [`state::PartialBlock`], that can
//! remember that we were executing a _specific_ entry in the block cache
//! - and indeed the progress made.
//!
//! Then, when insufficient steps are remaining to run a block in full,
//! we proceed to run the block anyway, but step-by-step. Once we
//! exhaust any remaining steps, we save progress in a partial block,
//! and execute the remainder of it with [`BlockCache::complete_current_block`]
//! on the next iteration.
//!
//! Since we now guarantee that we always execute the _same_ set of instructions,
//! no matter how many steps are remaining, we solve this possible divergence.
//!
//! # Dispatch
//!
//! The method of dispatch for Blocks can be one of several mechanisms, the current
//! default being [`Interpreted`].
//!
//! [`Interpreted`]: block::Interpreted

pub mod block;
mod config;
pub mod metrics;
mod state;

use std::marker::PhantomData;

use self::block::Block;
pub use self::config::DefaultCacheConfig;
pub use self::config::TestCacheConfig;
use super::MachineCoreState;
use super::ProgramCounterUpdate;
use super::StepManyResult;
use super::instruction::Instruction;
use super::instruction::RunInstr;
use super::memory::Address;
use super::memory::MemoryConfig;
use crate::machine_state::instruction::Args;
use crate::state_backend;
use crate::state_backend::AllocatedOf;
use crate::state_backend::EnrichedCell;
use crate::state_backend::EnrichedValue;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::Ref;
use crate::traps::EnvironException;
use crate::traps::Exception;

/// The maximum number of instructions that may be contained in a block.
pub const CACHE_INSTR: usize = 20;

/// Bindings for deriving an [`ICall`] from an [`Instruction`] via the [`EnrichedCell`] mechanism.
pub struct ICallPlaced<MC: MemoryConfig, M: ManagerBase> {
    _pd0: PhantomData<MC>,
    _pd1: PhantomData<M>,
}

impl<MC: MemoryConfig, M: ManagerBase> EnrichedValue for ICallPlaced<MC, M> {
    type E = Instruction;

    type D = ICall<MC, M::ManagerRoot>;
}

/// A function derived from an [OpCode] that can be directly run over the [MachineCoreState].
///
/// This allows static dispatch of this function during block construction,
/// rather than for each instruction, during each block execution.
///
/// [OpCode]: super::instruction::OpCode
pub struct ICall<MC: MemoryConfig, M: ManagerBase> {
    run_instr: RunInstr<MC, M>,
}

impl<MC: MemoryConfig, M: ManagerBase> Clone for ICall<MC, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<MC: MemoryConfig, M: ManagerBase> Copy for ICall<MC, M> {}

impl<MC: MemoryConfig, M: ManagerReadWrite> ICall<MC, M> {
    // SAFETY: This function must be called with an `Args` belonging to the same `OpCode` as
    // the one used to dispatch this function.
    #[inline(always)]
    unsafe fn run(
        &self,
        args: &Args,
        core: &mut MachineCoreState<MC, M>,
    ) -> Result<ProgramCounterUpdate<Address>, Exception> {
        unsafe { (self.run_instr)(args, core) }
    }
}

impl<'a, MC: MemoryConfig, M: ManagerReadWrite> From<&'a Instruction> for ICall<MC, M> {
    fn from(value: &'a Instruction) -> Self {
        let run_instr = value.opcode.to_run::<MC, M>();
        Self { run_instr }
    }
}

/// A block that is available to be run.
///
/// If there are sufficiently many steps remaining, the entire block is executed in one go.
/// Otherwise, it will fall back to partial evaluation.
///
/// As a result, before starting to run blocks from the block cache, you must first ensure that
/// any left-over partially-run block is cleared up with [`BlockCache::complete_current_block`].
pub struct BlockCall<'a, B: Block<MC, M>, MC: MemoryConfig, M: ManagerBase> {
    entry: &'a mut state::Cached<MC, B, M>,
    partial: &'a mut state::PartialBlock<M>,
}

impl<B: Block<MC, M>, MC: MemoryConfig, M: ManagerReadWrite> BlockCall<'_, B, MC, M> {
    /// Run a block, either fully or partially, depending on the number of steps remaining.
    #[inline(always)]
    pub fn run_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        block_builder: &mut B::BlockBuilder,
        instr_pc: Address,
        max_steps: usize,
    ) -> StepManyResult<EnvironException> {
        if self.entry.block.num_instr() <= max_steps {
            // Safety: the same block builder is passed through every time.
            unsafe { self.entry.block.run_block(core, instr_pc, block_builder) }
        } else {
            self.partial.run_block_partial(core, max_steps, self.entry)
        }
    }
}

#[inline(always)]
fn run_instr<MC: MemoryConfig, M: ManagerReadWrite>(
    instr: &EnrichedCell<ICallPlaced<MC, M>, M>,
    core: &mut MachineCoreState<MC, M>,
) -> Result<ProgramCounterUpdate<Address>, Exception> {
    let args = instr.read_ref_stored().args();
    let icall = instr.read_derived();

    // SAFETY: This is safe, as the function we are calling is derived directly from the
    // same instruction as the `Args` we are calling with. Therefore `args` will be of the
    // required shape.
    unsafe { icall.run(args, core) }
}

/// Block cache implementation
pub trait BlockCache<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase> {
    /// Instantiate a new block cache instance.
    fn new(manager: &mut M) -> Self
    where
        M: ManagerAlloc;

    /// Clone the entire block cache.
    fn clone(&self) -> Self
    where
        B: Clone,
        M: ManagerClone;

    /// Invalidate the entire block cache. This is more efficient than [`BlockCache::reset`],
    fn invalidate(&mut self)
    where
        M: ManagerReadWrite;

    /// Reset the entire block cache to its initial state. This is less efficient than
    /// [`BlockCache::invalidate`].
    fn reset(&mut self)
    where
        M: ManagerReadWrite;

    /// Retrieve a block at the given address with the purpose of executing it.
    fn get_block(&mut self, addr: Address) -> Option<BlockCall<'_, B, MC, M>>
    where
        M: ManagerRead;

    /// Insert a compressed instruction into the block cache at the given address.
    fn push_instr_compressed(&mut self, addr: Address, instr: Instruction)
    where
        M: ManagerReadWrite;

    /// Insert an uncompressed instruction into the block cache at the given  address.
    fn push_instr_uncompressed(&mut self, addr: Address, instr: Instruction)
    where
        M: ManagerReadWrite;

    /// Mark the current block as complete, and reset the partial block state.
    fn complete_current_block(
        &mut self,
        core: &mut MachineCoreState<MC, M>,
        max_steps: usize,
    ) -> StepManyResult<EnvironException>
    where
        M: ManagerReadWrite;
}

/// Configuration for a block cache
pub trait BlockCacheConfig {
    /// Layout for the block cache instance's state
    type Layout: state_backend::CommitmentLayout + state_backend::ProofLayout;

    /// Block cache instance
    type State<MC: MemoryConfig, B: Block<MC, M>, M: ManagerBase>: BlockCache<MC, B, M>;

    /// Bind the allocated regions to produce a memory instance.
    fn bind<MC, B, M>(space: AllocatedOf<Self::Layout, M>) -> Self::State<MC, B, M>
    where
        MC: MemoryConfig,
        B: Block<MC, M>,
        M: ManagerBase,
        M::ManagerRoot: ManagerReadWrite;

    /// Given a manager morphism `f : &M -> N`, return the block cache instance layout's allocated
    /// structure containing the constituents of `N` that were produced from the constituents of
    /// `&M`.
    fn struct_ref<'a, MC, B, M, F>(
        instance: &'a Self::State<MC, B, M>,
    ) -> AllocatedOf<Self::Layout, F::Output>
    where
        MC: MemoryConfig,
        B: Block<MC, M>,
        M: ManagerBase,
        F: FnManager<Ref<'a, M>>;
}
