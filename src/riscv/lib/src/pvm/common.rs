// SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::fmt;
use std::ops::Bound;
use std::ops::ControlFlow;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::components::data_space::CloneDataSpaceMode;
use octez_riscv_data::components::data_space::DataSpaceMode;
use octez_riscv_data::components::data_space::EncodeDataSpaceMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashError;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_tree::MerkleTree;
use octez_riscv_data::merkle_tree::MerkleTreeFold;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;
use tezos_smart_rollup_constants::riscv::SbiError;

use super::linux;
use super::reveals::RevealRequest;
use crate::default::ConstDefault;
use crate::machine_state;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::EmptyPageCache;
use crate::machine_state::page_cache::PageCache;
use crate::machine_state::registers::a0;
use crate::pvm::hooks::PvmHooks;
use crate::pvm::tezos;
use crate::range_utils::less_than_bound;
use crate::state_backend::ProofTree;
use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
use crate::state_backend::proof_backend::proof::Proof;
use crate::state_backend::proof_backend::proof::deserialise_owned;

/// Type of input that can be passed to the PVM
pub enum PvmInput<'a> {
    InboxMessage {
        inbox_level: u32,
        message_counter: u64,
        payload: &'a [u8],
    },
    Reveal(&'a [u8]),
}

/// PVM status
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Encode, Decode, strum::EnumCount)]
#[repr(u8)]
pub enum PvmStatus {
    Evaluating,
    WaitingForInput,
    WaitingForReveal,
}

impl ConstDefault for PvmStatus {
    const DEFAULT: Self = Self::Evaluating;
}

impl Default for PvmStatus {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for PvmStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self {
            PvmStatus::Evaluating => "Evaluating",
            PvmStatus::WaitingForInput => "Waiting for input message",
            PvmStatus::WaitingForReveal => "Waiting for reveal",
        };
        f.write_str(status)
    }
}

/// Value for the initial version
const INITIAL_VERSION: u64 = 0;

/// Proof generator for the PVM.
pub(crate) type PvmProve<'a, MC> = Pvm<MC, EmptyPageCache, Prove<'a>>;

/// Proof-generating virtual machine
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct Pvm<MC: MemoryConfig, PC, M: Mode> {
    pub(crate) machine_state: machine_state::MachineState<MC, PC, M>,
    pub(crate) reveal_request: RevealRequest<M>,
    pub(crate) system_state: linux::SupervisorState<M>,
    version: Atom<u64, M>,
    pub(crate) tick: Atom<u64, M>,
    pub(crate) message_counter: Atom<u64, M>,
    pub(crate) level: Atom<u32, M>,
    pub(crate) level_is_set: Atom<bool, M>,
    pub(crate) status: Atom<PvmStatus, M>,
}

impl<MC, PC, M> Default for Pvm<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: AtomMode + DataSpaceMode,
{
    fn default() -> Self {
        Self {
            machine_state: machine_state::MachineState::default(),
            reveal_request: RevealRequest::default(),
            system_state: linux::SupervisorState::default(),
            version: Atom::new(INITIAL_VERSION),
            status: Atom::default(),
            tick: Atom::default(),
            message_counter: Atom::default(),
            level: Atom::default(),
            level_is_set: Atom::default(),
        }
    }
}

impl<MC: MemoryConfig, PC: PageCache<MC, M>, M: Mode> Pvm<MC, PC, M> {
    /// Reset the PVM.
    pub fn reset(&mut self)
    where
        M: AtomMode + DataSpaceMode,
    {
        self.machine_state.reset();
        self.version.write(INITIAL_VERSION);
        self.tick.write(0);
        self.message_counter.write(0);
        self.level.write(0);
        self.level_is_set.write(false);
        self.status.write(PvmStatus::DEFAULT);
    }

    /// Used for testing, corrupt the state so the following proofs will be incorrect.
    pub fn insert_failure(&mut self)
    where
        M: AtomMode,
    {
        // We want to just slightly modify the state without interfering with normal execution.
        let csregs = &mut self.machine_state.core.hart.csregisters;

        // `fflags` is a writable CSR
        let fflags = csregs.read(CSRegister::fflags);
        csregs.write(CSRegister::fflags, fflags + 1);
    }

    /// Perform one evaluation step.
    pub(crate) fn eval_one(&mut self, hooks: impl PvmHooks)
    where
        M: AtomMode + DataSpaceMode,
    {
        self.eval_max(hooks, Bound::Included(1));
    }

    /// Perform a range of evaluation steps. Returns the actual number of steps
    /// performed.
    ///
    /// If an environment trap is raised, handle it and
    /// return the number of retired instructions until the raised trap
    ///
    /// NOTE: instructions which raise exceptions / are interrupted are NOT retired
    ///       See section 3.3.1 for context on retired instructions.
    /// e.g: a load instruction raises an exception but the first instruction
    /// of the trap handler will be executed and retired,
    /// so in the end the load instruction which does not bubble its exception up to
    /// the execution environment will still retire an instruction, just not itself.
    /// (a possible case: the privilege mode access violation is treated in EE,
    /// but a page fault is not)
    pub(crate) fn eval_max(&mut self, mut hooks: impl PvmHooks, step_bounds: Bound<usize>) -> usize
    where
        M: AtomMode + DataSpaceMode,
    {
        // Do nothing if step_bounds is less than 1
        if !less_than_bound(0, step_bounds) {
            return 0;
        }

        // When the status is WaitingForReveal during evaluation, we know that
        // nothing has been returned by the rollup node and the reveal request
        // is invalid.
        if let PvmStatus::WaitingForReveal = self.status.read() {
            self.provide_reveal_error_response();
            return 1;
        }

        let steps = self
            .machine_state
            .step_max_handle(step_bounds, |machine_state| {
                handle_system_call(
                    machine_state,
                    &mut self.system_state,
                    &mut self.status,
                    &mut self.reveal_request,
                    &mut hooks,
                )
            })
            .steps;

        let tick = self.tick.read().wrapping_add(steps as u64);
        self.tick.write(tick);

        steps
    }

    /// Provide input. Returns `false` if the machine state is not expecting input.
    pub(crate) fn provide_input(&mut self, input: PvmInput) -> bool
    where
        M: AtomMode + DataSpaceMode,
    {
        // TODO RV-615: Remove `as u32` conversion
        match input {
            PvmInput::InboxMessage {
                inbox_level,
                message_counter,
                payload,
            } => self.provide_inbox_message(inbox_level, message_counter as u32, payload),
            PvmInput::Reveal(reveal_data) => self.provide_reveal_response(reveal_data),
        }
    }

    /// Provide an inbox message. Returns `false` if the machine state is not
    /// expecting a message.
    pub(crate) fn provide_inbox_message(&mut self, level: u32, counter: u32, payload: &[u8]) -> bool
    where
        M: AtomMode + DataSpaceMode,
    {
        if !tezos::provide_input(
            &mut self.status,
            &mut self.machine_state.core,
            level,
            counter,
            payload,
        ) {
            return false;
        }

        let tick = self.tick.read().wrapping_add(1u64);
        self.tick.write(tick);

        self.message_counter.write(counter as u64);
        self.level_is_set.write(true);
        self.level.write(level);
        true
    }

    /// Provide reveal data in response to a reveal request.
    /// Returns `false` if the machine is not expecting a reveal.
    pub(crate) fn provide_reveal_response(&mut self, reveal_data: &[u8]) -> bool
    where
        M: AtomMode + DataSpaceMode,
    {
        if !tezos::provide_reveal_response(
            &mut self.status,
            &mut self.machine_state.core,
            reveal_data,
        ) {
            return false;
        }

        let tick = self.tick.read().wrapping_add(1u64);
        self.tick.write(tick);

        true
    }

    /// Get the reveal request in the machine state.
    pub(crate) fn reveal_request(&self) -> Vec<u8>
    where
        M: AtomMode,
    {
        self.reveal_request.to_vec()
    }

    /// Provide a reveal error response to the PVM
    pub fn provide_reveal_error_response(&mut self)
    where
        M: AtomMode + DataSpaceMode,
    {
        self.machine_state
            .core
            .hart
            .xregisters
            .write(a0, SbiError::InvalidParam as u64);

        let tick = self.tick.read().wrapping_add(1u64);
        self.tick.write(tick);

        self.status.write(PvmStatus::Evaluating);
    }

    /// Get the current machine status.
    pub fn status(&self) -> PvmStatus
    where
        M: AtomMode,
    {
        self.status.read()
    }

    /// Construct an [`InputRequest`] based on the PVM's current status and level.
    pub fn input_request(&self) -> InputRequest
    where
        M: AtomMode,
    {
        match self.status.read() {
            PvmStatus::Evaluating => InputRequest::NoInputRequired,
            PvmStatus::WaitingForReveal => {
                InputRequest::NeedsReveal(self.reveal_request().into_boxed_slice())
            }
            PvmStatus::WaitingForInput => {
                if self.level_is_set.read() {
                    InputRequest::Initial
                } else {
                    InputRequest::FirstAfter {
                        level: self.level.read(),
                        counter: self.message_counter.read(),
                    }
                }
            }
        }
    }
}

impl<MC: MemoryConfig, PC: PageCache<MC, Normal>> Pvm<MC, PC, Normal> {
    /// Return a proof-generating version of this PVM.
    pub(crate) fn start_proof(&self) -> PvmProve<'_, MC> {
        Pvm {
            machine_state: self.machine_state.start_proof(),
            reveal_request: self.reveal_request.start_proof(),
            system_state: self.system_state.start_proof(),
            version: self.version.start_proof(),
            tick: self.tick.start_proof(),
            message_counter: self.message_counter.start_proof(),
            level: self.level.start_proof(),
            level_is_set: self.level_is_set.start_proof(),
            status: self.status.start_proof(),
        }
    }
}

impl<'a, MC: MemoryConfig> Pvm<MC, EmptyPageCache, Prove<'a>>
where
    MC::State<Prove<'a>>: Foldable<MerkleTreeFold> + Foldable<HashFold>,
{
    /// Produce a proof.
    pub(crate) fn produce_proof(&self) -> Result<Proof, HashError> {
        // This read guarantees that the input request can be recovered from the proof.
        let _ = self.input_request();

        let merkle_tree = MerkleTree::from_foldable(self);
        let merkle_proof = merkle_tree_to_merkle_proof(merkle_tree);

        let final_hash = Hash::from_foldable(self);
        let proof = Proof::new(merkle_proof, final_hash);

        Ok(proof)
    }
}

impl<MC: MemoryConfig, PC: PageCache<MC, M>, M: CloneAtomMode + CloneDataSpaceMode> CloneState
    for Pvm<MC, PC, M>
{
    fn clone_state(&self) -> Self {
        Self {
            machine_state: self.machine_state.clone_state(),
            reveal_request: self.reveal_request.clone_state(),
            system_state: self.system_state.clone_state(),
            version: self.version.clone_state(),
            tick: self.tick.clone_state(),
            message_counter: self.message_counter.clone_state(),
            level: self.level.clone_state(),
            level_is_set: self.level_is_set.clone_state(),
            status: self.status.clone_state(),
        }
    }
}

impl<MC, PC, M, F> Foldable<F> for Pvm<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: Mode,
    F: Fold,
    machine_state::MachineState<MC, PC, M>: Foldable<F>,
    RevealRequest<M>: Foldable<F>,
    linux::SupervisorState<M>: Foldable<F>,
    Atom<PvmStatus, M>: Foldable<F>,
    Atom<bool, M>: Foldable<F>,
    Atom<u32, M>: Foldable<F>,
    Atom<u64, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.machine_state);
        builder.add(&self.reveal_request);
        builder.add(&self.system_state);
        builder.add(&self.version);
        builder.add(&self.tick);
        builder.add(&self.message_counter);
        builder.add(&self.level);
        builder.add(&self.level_is_set);
        builder.add(&self.status);
        builder.done()
    }
}

impl<MC: MemoryConfig> FromProof for Pvm<MC, EmptyPageCache, Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, machine_state) = proof.next_branch()?;
        let (proof, reveal_request) = proof.next_branch()?;
        let (proof, system_state) = proof.next_branch()?;
        let (proof, version) = proof.next_branch()?;
        let (proof, tick) = proof.next_branch()?;
        let (proof, message_counter) = proof.next_branch()?;
        let (proof, level) = proof.next_branch()?;
        let (proof, level_is_set) = proof.next_branch()?;
        let (proof, status) = proof.next_branch()?;

        proof.done(Self {
            machine_state,
            reveal_request,
            system_state,
            version,
            tick,
            message_counter,
            level,
            level_is_set,
            status,
        })
    }
}

impl<MC, PC, M> Encode for Pvm<MC, PC, M>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: EncodeAtomMode + EncodeDataSpaceMode,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.machine_state.encode(encoder)?;
        self.reveal_request.encode(encoder)?;
        self.system_state.encode(encoder)?;
        self.version.encode(encoder)?;
        self.tick.encode(encoder)?;
        self.message_counter.encode(encoder)?;
        self.level.encode(encoder)?;
        self.level_is_set.encode(encoder)?;
        self.status.encode(encoder)?;
        Ok(())
    }
}

impl<C, MC, PC> Decode<C> for Pvm<MC, PC, Normal>
where
    MC: MemoryConfig,
    PC: PageCache<MC, Normal>,
    MC::State<Normal>: Decode<C>,
{
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            machine_state: Decode::decode(decoder)?,
            reveal_request: Decode::decode(decoder)?,
            system_state: Decode::decode(decoder)?,
            version: Decode::decode(decoder)?,
            tick: Decode::decode(decoder)?,
            message_counter: Decode::decode(decoder)?,
            level: Decode::decode(decoder)?,
            level_is_set: Decode::decode(decoder)?,
            status: Decode::decode(decoder)?,
        })
    }
}

impl<MC: MemoryConfig> Pvm<MC, EmptyPageCache, Verify> {
    /// Construct a PVM state from a Merkle proof.
    pub fn from_proof(proof: &MerkleProof) -> Option<Self> {
        let (pvm, _) = deserialise_owned::deserialise(ProofTree::Present(proof)).ok()?;
        Some(pvm)
    }
}

/// An [`InputRequest`] is what the PVM expects as input for a specific tick.
pub enum InputRequest {
    /// No input is required at the moment, normal execution can continue.
    NoInputRequired,
    /// The PVM is waiting for an initial inbox input.
    Initial,
    /// The PVM is waiting for a message at a specific level and inbox index.
    /// `FirstAfter(level, counter)` represents the message at `level`, and index `counter + 1`.
    FirstAfter { level: u32, counter: u64 },
    /// The PVM is asking for a reveal response. The arguments are encoded by the payload bytes.
    NeedsReveal(Box<[u8]>),
}

/// Handle a system call in the PVM.
pub(crate) fn handle_system_call<MC, PC, M>(
    machine: &mut machine_state::MachineState<MC, PC, M>,
    system_state: &mut linux::SupervisorState<M>,
    status: &mut Atom<PvmStatus, M>,
    reveal_request: &mut RevealRequest<M>,
    hooks: impl PvmHooks,
) -> ControlFlow<()>
where
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    M: AtomMode + DataSpaceMode,
{
    system_state.handle_system_call(machine, hooks, |core| {
        tezos::handle_tezos(core, status, reveal_request);

        if status.read() == PvmStatus::Evaluating {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    })
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode_test;
    use proptest::proptest;
    use rand::Fill;
    use rand::rng;
    use tezos_smart_rollup_constants::riscv::REVEAL_REQUEST_MAX_SIZE;
    use tezos_smart_rollup_constants::riscv::SBI_FIRMWARE_TEZOS;
    use tezos_smart_rollup_constants::riscv::SBI_TEZOS_INBOX_NEXT;
    use tezos_smart_rollup_constants::riscv::SBI_TEZOS_REVEAL;

    use super::*;
    use crate::machine_state::memory;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::Memory;
    use crate::machine_state::page_cache::EmptyPageCache;
    use crate::machine_state::registers::a0;
    use crate::machine_state::registers::a1;
    use crate::machine_state::registers::a2;
    use crate::machine_state::registers::a3;
    use crate::machine_state::registers::a6;
    use crate::machine_state::registers::a7;
    use crate::pvm::common::tests::memory::Address;
    use crate::pvm::hooks::StdoutDebugHooks;
    use crate::pvm::linux;

    impl<MC: MemoryConfig, PC: PageCache<MC, M>, M: Mode> Pvm<MC, PC, M> {
        /// Handle an exception using the defined Execution Environment.
        // The conditional compilation below causes some warnings.
        fn handle_exception(&mut self, hooks: impl PvmHooks) -> bool
        where
            M: AtomMode + DataSpaceMode,
        {
            handle_system_call(
                &mut self.machine_state,
                &mut self.system_state,
                &mut self.status,
                &mut self.reveal_request,
                hooks,
            )
            .is_continue()
        }
    }

    #[test]
    fn test_read_input() {
        type MC = M1M;
        type PC = EmptyPageCache;

        // Setup PVM
        let mut pvm = Pvm::<MC, PC, Normal>::default();
        pvm.reset();
        pvm.machine_state.set_all_readable_writeable();

        let level_addr = memory::FIRST_ADDRESS;
        let counter_addr = level_addr + 4;
        let buffer_addr = counter_addr + 4;

        const BUFFER_LEN: usize = 1024;

        // Configure machine for 'sbi_tezos_inbox_next'
        pvm.machine_state
            .core
            .hart
            .xregisters
            .write(a0, buffer_addr);
        pvm.machine_state
            .core
            .hart
            .xregisters
            .write(a1, BUFFER_LEN as u64);
        pvm.machine_state.core.hart.xregisters.write(a2, level_addr);
        pvm.machine_state
            .core
            .hart
            .xregisters
            .write(a3, counter_addr);
        pvm.machine_state
            .core
            .hart
            .xregisters
            .write(a7, SBI_FIRMWARE_TEZOS);
        pvm.machine_state
            .core
            .hart
            .xregisters
            .write(a6, SBI_TEZOS_INBOX_NEXT);

        // Should be in evaluating mode
        assert_eq!(pvm.status(), PvmStatus::Evaluating);

        // Handle the ECALL successfully
        let outcome = pvm.handle_exception(StdoutDebugHooks);
        assert!(!outcome);

        // After the ECALL we should be waiting for input
        assert_eq!(pvm.status(), PvmStatus::WaitingForInput);

        // Respond to the request for input
        let level = rand::random();
        let counter = rand::random();
        let mut payload = [0u8; BUFFER_LEN + 10];
        payload.fill(&mut rng());
        assert!(pvm.provide_inbox_message(level, counter, &payload));

        // The status should switch from WaitingForInput to Evaluating
        assert_eq!(pvm.status(), PvmStatus::Evaluating);

        // Returned data is as expected
        assert_eq!(
            pvm.machine_state.core.hart.xregisters.read(a0) as usize,
            BUFFER_LEN
        );
        assert_eq!(
            pvm.machine_state.core.main_memory.read(level_addr),
            Ok(level)
        );
        assert_eq!(
            pvm.machine_state.core.main_memory.read(counter_addr),
            Ok(counter)
        );

        // Payload in memory should be as expected
        for (offset, &byte) in payload[..BUFFER_LEN].iter().enumerate() {
            let addr = buffer_addr + offset as u64;
            let byte_written: u8 = pvm.machine_state.core.main_memory.read(addr).unwrap();
            assert_eq!(
                byte, byte_written,
                "Byte at {addr:x} (offset {offset}) is not the same"
            );
        }

        // Data after the buffer should be untouched
        assert!(
            (BUFFER_LEN..4096)
                .map(|offset| {
                    let addr = buffer_addr + offset as u64;
                    pvm.machine_state.core.main_memory.read(addr).unwrap()
                })
                .all(|b: u8| b == 0)
        );
    }

    #[test]
    fn test_write_debug() {
        const WRITTEN_SIZE: usize = 100;
        proptest!(|(
            address in 0u64 as Address..(1024 * 1024 - WRITTEN_SIZE) as Address,
            written: [u8; WRITTEN_SIZE],
        )|{
            type MC = M1M;
            type PC = EmptyPageCache;

            let mut buffer = Vec::new();

            // Setup PVM
            let mut pvm = Pvm::<MC, PC, Normal>::default();
            pvm.reset();
            pvm.machine_state
                .set_all_readable_writeable();

            // Write characters
            pvm.machine_state
                .core
                .main_memory
                .write_all(address, &written)
                .unwrap();

            // Write the `write` system call number for `ecall`
            pvm.machine_state.core.hart.xregisters.write(a7, linux::WRITE);

            // Write `stdout` as the file descriptor parameter
            pvm.machine_state.core.hart.xregisters.write(a0, 1);

            // Write the address for the string to be read from
            pvm.machine_state
                .core
                .hart
                .xregisters
                .write(a1, address);

            // Write the length of the string
            pvm.machine_state
                .core
                .hart
                .xregisters
                .write(a2, written.len() as u64);

            pvm.handle_exception(&mut buffer);

            // Compare what characters have been passed to the hook versus what we
            // intended to write
            assert_eq!(written.to_vec(), buffer);
        });
    }

    mode_test!(test_reveal, F, {
        type MC = M1M;
        type PC = EmptyPageCache;

        // Setup PVM
        let mut pvm = Pvm::<MC, PC, F>::default();
        pvm.reset();
        pvm.machine_state.set_all_readable_writeable();

        let input_address = memory::FIRST_ADDRESS;
        let buffer = [1u8, 2, 3, 4];
        let output_address = input_address + buffer.len() as u64;
        let xregisters = &mut pvm.machine_state.core.hart.xregisters;

        // Configure machine for 'sbi_tezos_reveal'
        xregisters.write(a7, SBI_FIRMWARE_TEZOS);

        xregisters.write(a6, SBI_TEZOS_REVEAL);

        xregisters.write(a0, input_address);

        xregisters.write(a1, buffer.len() as u64);

        xregisters.write(a2, output_address);

        xregisters.write(a3, REVEAL_REQUEST_MAX_SIZE as u64);

        pvm.machine_state
            .core
            .main_memory
            .write_all(input_address, &buffer)
            .unwrap();

        assert_eq!(pvm.status(), PvmStatus::Evaluating);

        // Handle the ECALL successfully
        let outcome = pvm.handle_exception(StdoutDebugHooks);
        assert!(!outcome);

        // After the ECALL we should be waiting for reveal
        assert_eq!(pvm.status(), PvmStatus::WaitingForReveal);

        // After ECALL the reveal_request field should be set correctly
        assert_eq!(pvm.reveal_request.to_vec(), buffer);

        const REVEAL_DATA_SIZE: usize = 1000;
        let reveal_data = [1u8; REVEAL_DATA_SIZE];

        // Handle Reveal successfully
        let outcome = pvm.provide_reveal_response(&reveal_data);
        assert!(outcome, "Failed to provide reveal data to the PVM");

        // After the reveal the size of the data revealed should be written to a0
        assert_eq!(
            pvm.machine_state.core.hart.xregisters.read(a0) as usize,
            REVEAL_DATA_SIZE
        );

        let mut reveal_result_buffer = [0u8; REVEAL_DATA_SIZE];

        pvm.machine_state
            .core
            .main_memory
            .read_all(output_address, &mut reveal_result_buffer)
            .unwrap();

        // Reveal data returned correctly
        assert_eq!(reveal_result_buffer, reveal_data);
    });

    mode_test!(test_reveal_insufficient_buffer_size, F, {
        type MC = M1M;
        type PC = EmptyPageCache;

        // Setup PVM
        let mut pvm = Pvm::<MC, PC, F>::default();
        pvm.reset();
        pvm.machine_state.set_all_readable_writeable();

        const OUTPUT_BUFFER_SIZE: usize = 10;
        let input_address = memory::FIRST_ADDRESS;
        let buffer = [1u8, 2, 3, 4];
        let output_address = input_address + buffer.len() as u64;

        let xregisters = &mut pvm.machine_state.core.hart.xregisters;

        // Configure machine for 'sbi_tezos_reveal'
        xregisters.write(a7, SBI_FIRMWARE_TEZOS);

        xregisters.write(a6, SBI_TEZOS_REVEAL);

        xregisters.write(a0, input_address);

        xregisters.write(a1, buffer.len() as u64);

        xregisters.write(a2, output_address);

        xregisters.write(a3, OUTPUT_BUFFER_SIZE as u64);

        pvm.machine_state
            .core
            .main_memory
            .write_all(input_address, &buffer)
            .unwrap();

        assert_eq!(pvm.status(), PvmStatus::Evaluating);

        // Handle the ECALL successfully
        let outcome = pvm.handle_exception(StdoutDebugHooks);
        assert!(!outcome);

        // After the ECALL we should be waiting for reveal
        assert_eq!(pvm.status(), PvmStatus::WaitingForReveal);

        // After ECALL the reveal_request field should be set correctly
        assert_eq!(pvm.reveal_request.to_vec(), buffer);

        const REVEAL_DATA_SIZE: usize = 1000;
        let reveal_data = [1u8; REVEAL_DATA_SIZE];

        // Handle Reveal successfully
        let outcome = pvm.provide_reveal_response(&reveal_data);
        assert!(outcome, "Failed to provide reveal data to the PVM");

        // After the reveal the size of the data revealed should be written to a0
        assert_eq!(
            pvm.machine_state.core.hart.xregisters.read(a0) as usize,
            OUTPUT_BUFFER_SIZE
        );

        let mut reveal_result_buffer = [0u8; OUTPUT_BUFFER_SIZE];

        pvm.machine_state
            .core
            .main_memory
            .read_all(output_address, &mut reveal_result_buffer)
            .unwrap();

        // Reveal data returned correctly
        assert_eq!(reveal_result_buffer, reveal_data[..OUTPUT_BUFFER_SIZE]);
    });
}
