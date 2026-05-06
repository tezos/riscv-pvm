// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use humansize::BINARY;
use humansize::format_size;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::components::bytes::test_utils::BytesMutOp;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::merkle_proof::proof_binary;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::serialisation::serialise;
use proptest::prelude::Strategy;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// 64 MiB: the maximum size of a `Bytes` component in the durable storage
const LENGTH: usize = 1024 * 1024 * 64;

/// Given a `strategy` to generate instances of type `A` and an `eval` function to evaluate those
/// according to some metric. We always make at least one attempt, followed by some number of
/// `repeats`, to find the worst case, returning the example with the worst outcome we found.
///
/// Another function `setup` is provided to create some background state once instead of on every
/// repeat, in case this provides a performance benefit.
fn find_worst<A, State, Metric: Ord>(
    strategy: impl Strategy<Value = A>,
    setup: impl Fn() -> State,
    eval: impl Fn(&State, &A) -> Metric,
    repeats: usize,
) -> (A, Metric) {
    let mut runner = TestRunner::default();

    let state = setup();

    let mut gen_a = || {
        strategy
            .new_tree(&mut runner)
            .expect("can use `strategy` to generate values")
            .current()
    };
    let mut max_a = gen_a();
    let mut max_eval = eval(&state, &max_a);
    for _ in 0..repeats {
        let a = gen_a();
        let ev = eval(&state, &a);

        if ev > max_eval {
            max_a = a;
            max_eval = ev;
        }
    }

    (max_a, max_eval)
}

/// Initialise a `Bytes<Normal>` component for generating proofs over.
fn init_state() -> Bytes<Normal> {
    let v = [1u8, 2, 3, 4, 5]
        .iter()
        .copied()
        .cycle()
        .take(LENGTH)
        .collect::<Vec<_>>();
    Bytes::from(&v[..])
}

/// Produce the proof for a given bytes component and operation.
fn produce_proof(state: &Bytes<Normal>, op: &BytesMutOp) -> Vec<u8> {
    let mut bytes_prove = state.start_proof();
    let _result = op.run(&mut bytes_prove);

    let proof_tree = MerkleProof::from_foldable(&bytes_prove);
    serialise(proof_tree).expect("can serialise proof")
}

/// An evaluation function that measures the length of the proof for a given `op`.
fn proof_size(state: &Bytes<Normal>, op: &BytesMutOp) -> usize {
    produce_proof(state, op).len()
}

/// A function that runs all the code necessary to verify a proof (for benchmarking purposes),
/// which includes generating both hashes, without actually checking the proof is valid.
fn verify_proof(proof: &[u8], op: &BytesMutOp) {
    let (mut bytes_verify, parsed_proof_tree) =
        proof_binary::deserialise(proof).expect("can deserialise proof");
    let parsed_proof_tree = parsed_proof_tree.into_present();

    // To verify the proof the initial hash and final hash must both be computed. We won't actually
    // check the values here, this benchmark assumes the proof is correct.
    let _init_verify_hash =
        black_box(PartialHash::from_foldable(parsed_proof_tree.clone(), &bytes_verify).to_hash());
    let _verify_result = black_box(op.run(&mut bytes_verify));
    let _after_verify_hash =
        black_box(PartialHash::from_foldable(parsed_proof_tree.clone(), &bytes_verify).to_hash());
}

/// An evaluation function that measures the time taken by the proof verification.
fn proof_time(state: &Bytes<Normal>, op: &BytesMutOp) -> Duration {
    let proof = produce_proof(state, op);

    // one warm-up iteration
    verify_proof(black_box(&proof), black_box(op));

    let mut durations = vec![];

    // run ten iterations
    for _ in 0..10 {
        let start = Instant::now();
        verify_proof(black_box(&proof), black_box(op));
        durations.push(Instant::now().duration_since(start));
    }

    // we remove the quickest and slowest out of the samples and then take the average
    durations.sort();
    durations.into_iter().skip(1).take(8).sum::<Duration>() / 8
}

fn main() {
    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_size, 1000);
    println!("Biggest: {worst_op:?}, {}", format_size(eval, BINARY));

    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_time, 1000);
    println!("Slowest: {worst_op:?}, {eval:?}");
}
