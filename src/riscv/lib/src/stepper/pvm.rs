// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

mod reveals;

use std::ops::Bound;
use std::path::Path;

use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::data_space::CloneDataSpaceMode;
use octez_riscv_data::components::data_space::DataSpaceMode;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_tree::MerkleTreeFold;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::mode::utils::catch_not_found_and_more;
use octez_riscv_durable_storage::registry::CloneRegistryMode;
use reveals::RevealRequestResponseMap;
use tezos_smart_rollup_utils::inbox::Inbox;

use super::Stepper;
use super::StepperStatus;
use crate::kernel_loader;
use crate::machine_state::MachineCoreState;
use crate::machine_state::MachineError;
use crate::machine_state::memory::M1G;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::EmptyPageCache;
use crate::machine_state::page_cache::PageCache;
use crate::machine_state::page_cache::PageCacheInterpreted;
use crate::program::Program;
use crate::pvm::Pvm;
use crate::pvm::PvmStatus;
use crate::pvm::durable_storage::DurableStorage;
use crate::pvm::durable_storage::DurableStorageDummy;
use crate::pvm::errors::OperationalError;
use crate::pvm::hooks::NoHooks;
use crate::pvm::hooks::PvmHooks;
use crate::pvm::outbox::OutboxProof;
use crate::pvm::outbox::OutboxProofError;
use crate::pvm::outbox::OutputInfo;
use crate::range_utils;
use crate::range_utils::bound_saturating_sub;
use crate::state_backend::OwnedProofPart;
use crate::state_backend::ProofPart;
use crate::state_backend::ProofTree;
use crate::state_backend::proof_backend::proof::Proof;
use crate::state_backend::proof_backend::proof::deserialise_owned;
use crate::state_backend::proof_backend::proof::deserialise_stream::{self};
use crate::state_backend::proof_backend::proof::serialise_merkle_tree;
use crate::state_backend::verify_backend::ProofVerificationFailure;

/// Error during PVM stepping
#[derive(Debug, derive_more::From, thiserror::Error, derive_more::Display)]
pub enum PvmStepperError {
    /// Errors related to the machine state
    MachineError(MachineError),

    /// Errors arising from loading the kernel
    KernelError(kernel_loader::Error),
}

/// Wrapper over a PVM that lets you step through it
pub struct PvmStepper<
    H,
    MC: MemoryConfig = M1G,
    DS = DurableStorageDummy,
    PC: PageCache<MC, M> = PageCacheInterpreted<MC>,
    M: Mode = Normal,
> {
    pvm: Pvm<MC, PC, DS, M>,
    hooks: H,
    inbox: Inbox,
    rollup_address: [u8; 20],
    origination_level: u32,
    reveal_request_response_map: RevealRequestResponseMap,
    /// Whether the pvm stepper has exited. If true,
    /// attempting to step the pvm will fail
    has_exited: bool,
}

/// Variant of the [`PvmStepper`] used for verifying proofs
type PvmVerify<MC, DS> = PvmStepper<NoHooks, MC, DS, EmptyPageCache, Verify>;

impl<H, MC: MemoryConfig, PC: PageCache<MC, Normal>, DS: DurableStorage<Normal>>
    PvmStepper<H, MC, DS, PC, Normal>
{
    /// Create a new PVM stepper.
    pub fn new(
        program: &[u8],
        inbox: Inbox,
        hooks: H,
        rollup_address: [u8; 20],
        origination_level: u32,
        preimages_dir: Option<Box<Path>>,
    ) -> Result<Self, PvmStepperError>
    where
        DS: Default,
    {
        let mut pvm = Pvm::default();

        let program = Program::<MC>::from_elf(program)?;

        pvm.setup_linux_process(&program)?;

        let reveal_request_response_map =
            RevealRequestResponseMap::new(rollup_address, origination_level, preimages_dir);

        Ok(Self {
            pvm,
            hooks,
            inbox,
            rollup_address,
            origination_level,
            reveal_request_response_map,
            has_exited: false,
        })
    }

    /// Obtain the root hash for the PVM state.
    pub fn hash(&self) -> Hash
    where
        MC::State<Normal>: Foldable<HashFold>,
        DS: Foldable<HashFold>,
    {
        Hash::from_foldable(&self.pvm)
    }
}

impl<H, MC: MemoryConfig, PC: PageCache<MC, Normal>, DS: DurableStorage<Normal>>
    PvmStepper<H, MC, DS, PC, Normal>
{
    /// Create a new stepper in which the existing PVM is put into [`Prove`] mode.
    pub fn start_proof_mode<'normal>(
        &'normal self,
    ) -> PvmStepper<NoHooks, MC, DS::Prover, EmptyPageCache, Prove<'normal>>
    where
        DS: Provable<'normal>,
    {
        PvmStepper {
            pvm: self.pvm.start_proof(),
            rollup_address: self.rollup_address,
            origination_level: self.origination_level,

            // The inbox needs to be cloned because we should not mutate it through the new stepper
            // instance.
            inbox: self.inbox.clone(),

            // We don't want to re-use the same hooks to avoid polluting logs with refutation game
            // output. Instead we use hooks that don't do anything.
            hooks: NoHooks,

            reveal_request_response_map: self.reveal_request_response_map.clone(),
            has_exited: false,
        }
    }

    /// Produce the Merkle proof for evaluating one step on the given PVM state.
    /// The given stepper takes one step.
    pub fn produce_proof<'normal>(&'normal mut self) -> Option<Proof>
    where
        MC::State<Prove<'normal>>: Foldable<HashFold> + Foldable<MerkleTreeFold>,
        DS: Provable<'normal>,
        DS::Prover: DurableStorage<Prove<'normal>> + Foldable<HashFold> + Foldable<MerkleTreeFold>,
    {
        // Step using the proof mode stepper in order to obtain the proof
        let mut proof_stepper = self.start_proof_mode();

        proof_stepper.try_step().then_some(())?;

        let proof = proof_stepper.pvm.produce_proof().ok()?;
        Some(proof)
    }

    /// Produce an outbox proof by recording the Merkle proof of a state transition
    /// in which the outbox message at the given level and index is read.
    pub fn produce_outbox_proof<'normal>(
        &'normal self,
        output_info: OutputInfo,
    ) -> Result<OutboxProof, OutboxProofError>
    where
        MC::State<Prove<'normal>>: Foldable<HashFold> + Foldable<MerkleTreeFold>,
        DS: Provable<'normal>,
        DS::Prover: DurableStorage<Prove<'normal>> + Foldable<HashFold> + Foldable<MerkleTreeFold>,
    {
        let proof_stepper = self.start_proof_mode();
        proof_stepper.pvm.produce_outbox_proof(output_info)
    }
}

impl<
    H: PvmHooks,
    MC: MemoryConfig,
    PC: PageCache<MC, M>,
    DS: DurableStorage<M>,
    M: AtomMode + DataSpaceMode,
> PvmStepper<H, MC, DS, PC, M>
{
    /// Non-continuing variant of [`Stepper::step_max`]
    // TODO RV-573:
    // There is a divergence from the semantics of the PVM. Namely, when failing to provide input
    // we actually still take a step whilst transitioning to the 'error mode'. This is needed to
    // allow the semantics of `run(Unbounded) -> Exited { steps: X }` and `run(X) -> Exited { steps: X }`
    // to be equivalent. Otherwise, if returning `Exited` for the inbox being drained takes no steps,
    // we'd actually return `Running { X }` still in the second case. RV-573 will address this by no
    // longer returning stepper status, but rather allowing parts of the stepper harness, and the pvm, to
    // be queried directly.
    fn step_max_once(&mut self, steps: Bound<usize>) -> StepperStatus {
        // SAFETY: We're in a stepper context where divergence (e.g. early exit) is allowed.
        unsafe {
            if let Some(exit_code) = self.pvm.has_exited() {
                self.has_exited = true;
                return StepperStatus::Exited {
                    steps: 0,
                    success: exit_code == 0,
                    status: format!("Exited with code {exit_code}"),
                };
            }
        }

        if range_utils::unwrap_bound(steps) == 0 {
            // we cannot eval steps or provide input
            return StepperStatus::Running { steps: 0 };
        }

        match self.pvm.status() {
            PvmStatus::Evaluating => {
                let steps = self.pvm.eval_max(&mut self.hooks, steps);
                StepperStatus::Running { steps }
            }

            PvmStatus::WaitingForInput => match self.inbox.next() {
                Some((level, counter, payload)) => {
                    let success =
                        self.pvm
                            .provide_inbox_message(level, counter, payload.as_slice());

                    if success {
                        StepperStatus::Running { steps: 1 }
                    } else {
                        // TODO RV-573: the stepper should take no steps here. We should not return status
                        // but have it be separately queryable
                        self.has_exited = true;
                        StepperStatus::Errored {
                            steps: 1,
                            cause: "PVM was waiting for input".to_owned(),
                            message: "Providing input did not succeed".to_owned(),
                        }
                    }
                }

                None => {
                    // TODO RV-573: the stepper should take no steps here. We should not return status
                    // but have it be separately queryable
                    self.has_exited = true;
                    StepperStatus::Exited {
                        steps: 1,
                        success: true,
                        status: "Inbox has been drained".to_owned(),
                    }
                }
            },

            PvmStatus::WaitingForReveal => {
                let reveal_request = self.pvm.reveal_request();

                let Some(reveal_response) = self
                    .reveal_request_response_map
                    .get_response(reveal_request.as_slice())
                else {
                    // TODO: RV-573: Handle incorrectly encoded request/ Unavailable data differently in the sandbox.
                    // When the PVM sends an incorrectly encoded reveal request, the stepper should return an error.
                    // When the PVM sends a request for unavailable data, the stepper should exit.
                    self.pvm.provide_reveal_error_response();

                    return StepperStatus::Running { steps: 1 };
                };

                let success = self.pvm.provide_reveal_response(&reveal_response);
                if success {
                    StepperStatus::Running { steps: 1 }
                } else {
                    // TODO RV-573: the stepper should take no steps here. We should not return status
                    // but have it be separately queryable
                    self.has_exited = true;
                    StepperStatus::Errored {
                        steps: 1,
                        cause: "PVM was waiting for reveal response".to_owned(),
                        message: "Providing reveal response did not succeed".to_owned(),
                    }
                }
            }
        }
    }

    /// Try to take one step and return true if the PVM is not in an errored state.
    fn try_step(&mut self) -> bool {
        match self.step_max_once(Bound::Included(1)) {
            // We don't include errors in this case because errors count as 0 steps. That means if
            // we receive a [`StepperStatus::Errored`] with `steps: 1` then it tried to run 2 steps
            // but failed at the second.
            StepperStatus::Running { steps: 1 } | StepperStatus::Exited { steps: 1, .. } => true,
            _ => false,
        }
    }

    /// Re-bind the PVM type by cloning the underlying regions.
    pub fn rebind_via_clone(&mut self) -> Result<(), OperationalError>
    where
        M: CloneAtomMode + CloneDataSpaceMode + CloneRegistryMode,
        DS: CloneState,
    {
        self.pvm = self.pvm.try_clone_state()?;
        Ok(())
    }
}

impl<H, MC: MemoryConfig, M: AtomMode + DataSpaceMode, PC: PageCache<MC, M>, DS>
    PvmStepper<H, MC, DS, PC, M>
{
    /// Similar to [`PvmStepper::verify_proof`] but constructs the allocated space by using the raw deserialisation.
    ///
    /// Useful for testing the stream deserialisation.
    pub fn verify_proof_using_raw_bytes(&self, proof: Proof) -> Result<(), ProofVerificationFailure>
    where
        for<'a> MC::State<Verify>: Foldable<PartialHashFold<'a>>,
        for<'a> DS: Foldable<PartialHashFold<'a>>,
        DS: FromProof + DurableStorage<Verify>,
    {
        let tree_serialisation: Box<[u8]> = serialise_merkle_tree(proof.tree()).into_boxed_slice();
        let (pvm, merkle_tree) = deserialise_stream::deserialise(&tree_serialisation)
            .map_err(ProofVerificationFailure::BadDeserialisation)?;

        let deserialised_proof_tree = match merkle_tree {
            OwnedProofPart::Present(ref merkle_tree) => ProofTree::Present(merkle_tree),
            OwnedProofPart::Absent => ProofTree::Absent,
        };
        debug_assert_eq!(
            ProofTree::Present(proof.tree()),
            deserialised_proof_tree,
            "The Merkle proof tree obtained through deserialisation should match the original proof tree"
        );

        let stepper = self.to_verify_stepper(pvm)?;

        stepper.verify_proof_internal(ProofPart::Present(proof.tree()), proof.final_state_hash())
    }

    /// Verify a Merkle proof. The [`PvmStepper`] is used for inbox information.
    pub fn verify_proof(&self, proof: Proof) -> Result<(), ProofVerificationFailure>
    where
        for<'a> MC::State<Verify>: Foldable<PartialHashFold<'a>>,
        for<'a> DS: Foldable<PartialHashFold<'a>>,
        DS: FromProof + DurableStorage<Verify>,
    {
        let proof_tree = ProofTree::Present(proof.tree());
        let (pvm, deserialised_proof_tree) = deserialise_owned::deserialise(proof_tree)
            .map_err(ProofVerificationFailure::BadDeserialisation)?;

        let deserialised_proof_tree = match deserialised_proof_tree {
            OwnedProofPart::Present(ref merkle_tree) => ProofTree::Present(merkle_tree),
            OwnedProofPart::Absent => ProofTree::Absent,
        };
        debug_assert_eq!(
            proof_tree, deserialised_proof_tree,
            "The Merkle proof tree obtained through deserialisation should match the original proof tree"
        );

        let stepper = self.to_verify_stepper(pvm)?;
        stepper.verify_proof_internal(proof_tree, proof.final_state_hash())
    }

    fn to_verify_stepper(
        &self,
        pvm: Pvm<MC, EmptyPageCache, DS, Verify>,
    ) -> Result<PvmVerify<MC, DS>, ProofVerificationFailure> {
        Ok(PvmStepper {
            pvm,
            rollup_address: self.rollup_address,
            origination_level: self.origination_level,

            // The inbox needs to be cloned because we should not mutate it through the new stepper
            // instance.
            inbox: self.inbox.clone(),

            // We don't want to re-use the same hooks to avoid polluting logs with refutation game
            // output. Instead we use hooks that don't do anything.
            hooks: NoHooks,

            reveal_request_response_map: self.reveal_request_response_map.clone(),
            has_exited: false,
        })
    }
}

impl<
    H: PvmHooks,
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode,
    PC: PageCache<MC, M>,
    DS: DurableStorage<M>,
> PvmStepper<H, MC, DS, PC, M>
{
    /// Perform one evaluation step.
    pub fn eval_one(&mut self) {
        self.pvm.eval_one(&mut self.hooks)
    }

    /// Get the current level of the PVM
    pub fn level(&self) -> Option<u32> {
        if !self.pvm.level_is_set.read() {
            return None;
        }
        Some(self.pvm.level.read())
    }
}

impl<H: PvmHooks, MC: MemoryConfig, DS: DurableStorage<Verify>>
    PvmStepper<H, MC, DS, EmptyPageCache, Verify>
{
    /// Try to take one step. Stepping in the [`Verify`] mode may panic
    /// when attempting to access absent data. Catches the case of verifying an invalid proof, as
    /// [`ProofVerificationFailure::AbsentDataAccess`] and all other panics
    /// as [`ProofVerificationFailure::StepperPanic`].
    fn try_step_partial(self) -> Result<Self, ProofVerificationFailure> {
        // Wrapping the stepper in a Mutex, which implements poisoning, in order to pass it
        // across the unwind boundary.
        let mutex = std::sync::Mutex::new(self);
        catch_not_found_and_more(move || {
            {
                let mut stepper = mutex.lock().expect("Mutex was poisoned on initialisation");
                if !stepper.try_step() {
                    return Err(ProofVerificationFailure::StepperError);
                };
            }

            // Since all panics were handled and returned as errors, at this point
            // the mutex cannot be poisoned.
            Ok(mutex.into_inner().expect("Unexpected poisoned mutex"))
        })?
    }

    fn verify_proof_internal(
        self,
        proof_tree: ProofTree,
        expected_final_hash: Hash,
    ) -> Result<(), ProofVerificationFailure>
    where
        for<'a> MC::State<Verify>: Foldable<PartialHashFold<'a>>,
        for<'a> DS: Foldable<PartialHashFold<'a>>,
    {
        let stepper = self.try_step_partial()?;

        let proof_tree = match proof_tree {
            ProofTree::Present(tree) => Some(tree),
            ProofTree::Absent => None,
        };
        let final_hash = PartialHash::from_foldable(proof_tree, &stepper.pvm)
            .to_hash()
            .ok_or(ProofVerificationFailure::BadProofForHashing)?;

        if final_hash != expected_final_hash {
            return Err(ProofVerificationFailure::FinalHashMismatch {
                expected: expected_final_hash,
                computed: final_hash,
            });
        }

        Ok(())
    }
}

impl<H: PvmHooks, MC: MemoryConfig, PC: PageCache<MC, Normal>, DS: DurableStorage<Normal>> Stepper
    for PvmStepper<H, MC, DS, PC, Normal>
{
    type MemoryConfig = MC;

    type Mode = Normal;

    fn machine_state(&self) -> &MachineCoreState<Self::MemoryConfig, Self::Mode> {
        &self.pvm.machine_state.core
    }

    type StepResult = StepperStatus;

    fn step_max(&mut self, mut step_bounds: Bound<usize>) -> Self::StepResult {
        if self.has_exited {
            return StepperStatus::Errored {
                steps: 0,
                cause: "PVM stepper exited previously".to_owned(),
                message: "Cannot step a PVM Stepper that previously exited".to_owned(),
            };
        }

        let mut total_steps = 0usize;

        loop {
            match self.step_max_once(step_bounds) {
                StepperStatus::Running { steps } => {
                    total_steps = total_steps.saturating_add(steps);
                    step_bounds = bound_saturating_sub(step_bounds, steps);

                    if steps < 1 {
                        // Break if no progress has been made.
                        break StepperStatus::Running { steps: total_steps };
                    }
                }

                StepperStatus::Exited {
                    steps,
                    success,
                    status,
                } => {
                    break StepperStatus::Exited {
                        steps: total_steps.saturating_add(steps),
                        success,
                        status,
                    };
                }

                StepperStatus::Errored {
                    steps,
                    cause,
                    message,
                } => {
                    break StepperStatus::Errored {
                        steps: total_steps.saturating_add(steps),
                        cause,
                        message,
                    };
                }
            }
        }
    }
}
