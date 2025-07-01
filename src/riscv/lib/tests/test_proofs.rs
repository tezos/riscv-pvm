// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod common;

use std::io::Write;
use std::ops::Bound;
use std::time::Instant;

use common::*;
use octez_riscv::machine_state::block_cache::BlockCacheConfig;
use octez_riscv::machine_state::block_cache::DefaultCacheConfig;
use octez_riscv::machine_state::block_cache::TestCacheConfig;
use octez_riscv::machine_state::block_cache::block::Interpreted;
use octez_riscv::machine_state::memory::M64M;
use octez_riscv::machine_state::memory::MemoryConfig;
use octez_riscv::pvm::hooks::NoHooks;
use octez_riscv::state_backend::AllocatedOf;
use octez_riscv::state_backend::hash;
use octez_riscv::state_backend::owned_backend::Owned;
use octez_riscv::state_backend::proof_backend::proof::Proof;
use octez_riscv::state_backend::proof_backend::proof::serialise_proof;
use octez_riscv::state_backend::verify_backend::ProofVerificationFailure;
use octez_riscv::state_backend::verify_backend::Verifier;
use octez_riscv::stepper::Stepper;
use octez_riscv::stepper::StepperStatus;
use octez_riscv::stepper::pvm::PvmStepper;
use rand::Rng;

#[test]
#[ignore]
fn test_jstz_proofs_one_step() {
    test_jstz_proofs(false, PvmStepper::verify_proof)
}

#[test]
#[ignore]
fn test_jstz_proofs_one_step_stream() {
    test_jstz_proofs(false, PvmStepper::verify_proof_using_raw_bytes)
}

#[test]
#[ignore]
fn test_jstz_proofs_full() {
    test_jstz_proofs(true, PvmStepper::verify_proof)
}

#[test]
#[ignore]
fn test_jstz_proofs_full_stream() {
    test_jstz_proofs(true, PvmStepper::verify_proof_using_raw_bytes)
}

#[test]
fn test_jstz_initial_proof_regression() {
    // Configuring the stepper with `TestCachelayouts` to match the node PVM
    // and make the test run faster.
    let make_stepper = make_stepper_factory::<TestCacheConfig>();
    let mut stepper = make_stepper();

    eprintln!("> Producing proof ...");
    let proof = stepper.produce_proof().unwrap();
    let proof_serialisation: Vec<u8> = serialise_proof(&proof).collect();

    // This file is also used in the tests for the OCaml `lib_riscv` library
    let mut mint = goldenfile::Mint::new("tests/expected/jstz");
    let mut proof_capture = mint.new_goldenfile("proof_initial").unwrap();

    let proof_bytes = hex::encode(proof_serialisation);
    writeln!(proof_capture, "{proof_bytes}").unwrap();
}

fn test_jstz_proofs(full: bool, verify_fn: StepperVerifyFn<M64M, DefaultCacheConfig, Owned>) {
    let make_stepper = make_stepper_factory::<DefaultCacheConfig>();

    let mut base_stepper = make_stepper();
    let base_result = base_stepper.step_max(Bound::Unbounded);
    assert!(matches!(base_result, StepperStatus::Exited { .. }));

    let steps = base_result.steps();
    let base_hash = base_stepper.hash();

    if full {
        // For each step `s`, the stepper will initially step `s-1` steps, then
        // produce a proof of the `s` step. The minimum step size thus needs to be 1.
        let ladder = dissect_steps(steps, 1);
        run_steps_ladder(&make_stepper, &ladder, Some(base_hash), verify_fn);
    } else {
        // Run a number of steps `s` and produce a proof
        let mut rng = rand::rng();
        let step = [rng.random_range(1..steps)];
        run_steps_ladder(&make_stepper, &step, None, verify_fn)
    }
}

fn run_steps_ladder<F>(
    make_stepper: F,
    ladder: &[usize],
    expected_hash: Option<hash::Hash>,
    verify_fn: StepperVerifyFn<M64M, DefaultCacheConfig, Owned>,
) where
    F: Fn() -> PvmStepper<NoHooks, M64M, DefaultCacheConfig>,
{
    let expected_steps = ladder.iter().sum::<usize>();
    let mut stepper = make_stepper();

    let mut steps_done = 0;
    for &steps in ladder {
        // Run one step short of `steps`, then produce a proof of the following step.
        let steps = steps.checked_sub(1).expect("minimum step size is 1");
        eprintln!("> Running {} steps ...", steps);
        let result = stepper.step_max(Bound::Included(steps));
        steps_done += steps;

        if steps_done != expected_steps {
            assert!(matches!(result, StepperStatus::Running { .. }));

            eprintln!("> Producing proof ...");
            let start = Instant::now();
            let proof = stepper.produce_proof().unwrap();
            let time = start.elapsed();
            let serialisation: Vec<u8> = serialise_proof(&proof).collect();
            eprintln!(
                "> Proof of size {} KiB produced in {:?}",
                serialisation.len() / 1024,
                time
            );

            eprintln!("> Checking initial proof hash ...");
            let initial_state_hash = proof.initial_state_hash();
            assert_eq!(initial_state_hash, stepper.hash());

            let final_state_hash = proof.final_state_hash();

            eprintln!("> Attempting to verify basic invalid proofs ...");
            basic_invalid_proofs_are_rejected(&stepper, &proof, initial_state_hash, verify_fn);

            eprintln!("> Verifying proof ...");
            assert!(verify_fn(&stepper, proof).is_ok());

            // Run one final step, which is the step proven by `proof`, and check that its
            // state hash matches the final state hash of `proof`.
            eprintln!("> Running 1 step ...");
            stepper.eval_one();
            steps_done += 1;

            eprintln!("> Checking final proof hash ...");
            assert_eq!(final_state_hash, stepper.hash());
        } else {
            // Can't generate a proof on the next step if execution has ended
            assert!(matches!(result, StepperStatus::Exited { .. }));
            assert!(ladder.is_empty())
        };

        eprintln!(
            "> Done {:.2}%",
            (steps_done as f64 / expected_steps as f64) * 100.0
        );
    }

    if let Some(hash) = expected_hash {
        assert_eq!(stepper.hash(), hash)
    }
}

type StepperVerifyFn<MC, BCC, M> = fn(
    &PvmStepper<NoHooks, MC, BCC, M, Interpreted<MC, M>>,
    proof: Proof,
) -> Result<(), ProofVerificationFailure>;

fn basic_invalid_proofs_are_rejected<MC: MemoryConfig, BCC: BlockCacheConfig>(
    stepper: &PvmStepper<NoHooks, MC, BCC>,
    proof: &Proof,
    state_hash: hash::Hash,
    verify_fn: StepperVerifyFn<MC, BCC, Owned>,
) where
    AllocatedOf<BCC::Layout, Verifier>: 'static,
{
    // A fully blinded proof could only be valid if every single leaf
    // in the state is written to and proof compression were to optimise
    // for this case.
    let fully_blinded_proof = proof_helpers::fully_blinded(state_hash);
    assert!(
        verify_fn(stepper, fully_blinded_proof)
            .is_err_and(|e| matches!(e, ProofVerificationFailure::AbsentDataAccess(_)))
    );

    let empty_proof = proof_helpers::empty(state_hash);
    assert!(
        verify_fn(stepper, empty_proof)
            .is_err_and(|e| matches!(e, ProofVerificationFailure::UnexpectedProofShape))
    );

    let invalid_final_hash_proof = proof_helpers::with_final_hash(proof, state_hash);
    assert!(
        verify_fn(stepper, invalid_final_hash_proof)
            .is_err_and(|e| matches!(e, ProofVerificationFailure::FinalHashMismatch { .. }))
    );
}

mod proof_helpers {
    use octez_riscv::state_backend::hash::Hash;
    use octez_riscv::state_backend::proof_backend::proof::MerkleProofLeaf;
    use octez_riscv::state_backend::proof_backend::proof::Proof;
    use octez_riscv::state_backend::proof_backend::tree::Tree;

    pub fn fully_blinded(hash: Hash) -> Proof {
        Proof::new(Tree::Leaf(MerkleProofLeaf::Blind(hash)), hash)
    }

    pub(crate) fn empty(hash: Hash) -> Proof {
        Proof::new(Tree::Node(Vec::new()), hash)
    }

    pub(crate) fn with_final_hash(proof: &Proof, hash: Hash) -> Proof {
        Proof::new(proof.tree().clone(), hash)
    }
}
