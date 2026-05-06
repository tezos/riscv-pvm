// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use humansize::BINARY;
use humansize::format_size;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::components::bytes::test_utils::BytesMutOp;
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

fn main() {
    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_size, 1000);
    println!("Biggest: {worst_op:?}, {}", format_size(eval, BINARY),);
}
