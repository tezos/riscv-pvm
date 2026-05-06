// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::hint::black_box;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
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

/// A struct to count the allocations or deallocations, compiles summary statistics from a sequence
/// of `Layout`s.
#[derive(Debug)]
struct Counter {
    layouts: AtomicUsize,
    bytes: AtomicUsize,
    max: AtomicUsize,
}

impl Counter {
    const fn new() -> Self {
        Counter {
            layouts: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
        }
    }

    /// Reset the counter to all zero.
    fn reset(&self) {
        self.layouts.store(0, Ordering::SeqCst);
        self.bytes.store(0, Ordering::SeqCst);
        self.max.store(0, Ordering::SeqCst);
    }

    /// Add another `Layout` to the summary counts.
    fn add(&self, layout: Layout) {
        self.layouts.fetch_add(1, Ordering::SeqCst);
        self.bytes.fetch_add(layout.size(), Ordering::SeqCst);
        self.max.fetch_max(layout.size(), Ordering::SeqCst);
    }

    /// Subtract a `Layout` from a counter. Does not behave correctly if the counter would go
    /// negative.
    fn subtract(&self, layout: Layout) {
        self.layouts.fetch_sub(1, Ordering::SeqCst);
        self.bytes.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    /// Increases the counts in `self` to match `other`, if necessary.
    fn max(&self, other: &Counter) {
        let layouts = other.layouts.load(Ordering::SeqCst);
        let bytes = other.bytes.load(Ordering::SeqCst);
        let max = other.max.load(Ordering::SeqCst);

        self.layouts.fetch_max(layouts, Ordering::SeqCst);
        self.bytes.fetch_max(bytes, Ordering::SeqCst);
        self.max.fetch_max(max, Ordering::SeqCst);
    }
}

/// A drop-in replacement for the global allocator, which wraps the system allocator while counting
/// the allocations and deallocations made.
#[derive(Debug)]
struct CountingAllocator {
    /// Counts total allocations requested so far.
    allocs: Counter,

    /// Counts total deallocations requested so far.
    deallocs: Counter,

    /// Counts the allocations currently needed (decreases on deallocation).
    rolling: Counter,

    /// Records the maximum values attained by `rolling` so far.
    max: Counter,
}

impl CountingAllocator {
    /// Set all the counts to zero.
    fn reset(&self) {
        self.allocs.reset();
        self.deallocs.reset();
        self.rolling.reset();
        self.max.reset();
    }

    /// Assert that the counted allocations and deallocations are equal and return the agreed on
    /// values.
    ///
    /// While in a few cases you might want to interrogate the allocations and deallocations
    /// separately, most of the time calling this method is what you want: the idea is to `reset`
    /// the allocator before a scope begins and `check` it afterwards. This means we know that we
    /// are only counting the allocations made (and subsequently dropped) within that scope.
    fn check(&self) -> (usize, usize, usize) {
        let a_layouts = self.allocs.layouts.load(Ordering::SeqCst);
        let a_bytes = self.allocs.bytes.load(Ordering::SeqCst);
        let a_max = self.allocs.max.load(Ordering::SeqCst);
        let d_layouts = self.deallocs.layouts.load(Ordering::SeqCst);
        let d_bytes = self.deallocs.bytes.load(Ordering::SeqCst);
        let d_max = self.deallocs.max.load(Ordering::SeqCst);
        let r_layouts = self.rolling.layouts.load(Ordering::SeqCst);
        let r_bytes = self.rolling.bytes.load(Ordering::SeqCst);

        assert_eq!(r_layouts, 0);
        assert_eq!(r_bytes, 0);

        assert_eq!(a_layouts, d_layouts);
        assert_eq!(a_bytes, d_bytes);
        assert_eq!(a_max, d_max);

        (a_layouts, a_bytes, a_max)
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = unsafe { System.alloc(layout) };
        self.allocs.add(layout);
        self.rolling.add(layout);
        self.max.max(&self.rolling);
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
        }
        self.deallocs.add(layout);
        self.rolling.subtract(layout);
    }
}

/// In order to replace the global allocator we need a static instance of our struct.
#[global_allocator]
static GLOB_ALLOC: CountingAllocator = CountingAllocator {
    allocs: Counter::new(),
    deallocs: Counter::new(),
    rolling: Counter::new(),
    max: Counter::new(),
};

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
        black_box(PartialHash::from_foldable(parsed_proof_tree, &bytes_verify).to_hash());
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

/// An evaluation function that counts the total number of allocations made by the proof
/// verification.
fn proof_allocs(state: &Bytes<Normal>, op: &BytesMutOp) -> usize {
    let proof = produce_proof(state, op);

    GLOB_ALLOC.reset();
    verify_proof(&proof, op);
    let (allocs, _, _) = GLOB_ALLOC.check();
    allocs
}

/// An evaluation function that measures the number of bytes allocated by the proof verification.
fn proof_alloc_bytes(state: &Bytes<Normal>, op: &BytesMutOp) -> usize {
    let proof = produce_proof(state, op);

    GLOB_ALLOC.reset();
    verify_proof(&proof, op);
    let (_, alloc_bytes, _) = GLOB_ALLOC.check();
    alloc_bytes
}

/// An evaluation function that returns the single largest allocation made by the proof
/// verification.
fn proof_biggest_alloc(state: &Bytes<Normal>, op: &BytesMutOp) -> usize {
    let proof = produce_proof(state, op);

    GLOB_ALLOC.reset();
    verify_proof(&proof, op);
    let (_, _, biggest_alloc) = GLOB_ALLOC.check();
    biggest_alloc
}

/// An evaluation function that returns the maximum number of allocations at any one time during
/// verification.
fn proof_max_rolling_allocs(state: &Bytes<Normal>, op: &BytesMutOp) -> usize {
    let proof = produce_proof(state, op);

    GLOB_ALLOC.reset();
    verify_proof(&proof, op);
    GLOB_ALLOC.max.layouts.load(Ordering::SeqCst)
}

/// An evaluation function that returns the maximum number of bytes allocated at any one time
/// during verification.
fn proof_max_rolling_bytes(state: &Bytes<Normal>, op: &BytesMutOp) -> usize {
    let proof = produce_proof(state, op);

    GLOB_ALLOC.reset();
    verify_proof(&proof, op);
    GLOB_ALLOC.max.bytes.load(Ordering::SeqCst)
}

fn main() {
    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_size, 1000);
    println!("Biggest: {worst_op:?}, {}", format_size(eval, BINARY));

    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_time, 1000);
    println!("Slowest: {worst_op:?}, {eval:?}");

    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_allocs, 1000);
    println!("Most allocs: {worst_op:?}, {eval:?}");

    let (worst_op, eval) = find_worst(BytesMutOp::any(LENGTH), init_state, proof_alloc_bytes, 1000);
    println!("Most bytes: {worst_op:?}, {}", format_size(eval, BINARY));

    let (worst_op, eval) = find_worst(
        BytesMutOp::any(LENGTH),
        init_state,
        proof_biggest_alloc,
        1000,
    );
    println!("Biggest alloc: {worst_op:?}, {}", format_size(eval, BINARY));

    let (worst_op, eval) = find_worst(
        BytesMutOp::any(LENGTH),
        init_state,
        proof_max_rolling_allocs,
        1000,
    );
    println!("Most allocs at once: {worst_op:?}, {eval:?}");

    let (worst_op, eval) = find_worst(
        BytesMutOp::any(LENGTH),
        init_state,
        proof_max_rolling_bytes,
        1000,
    );
    println!(
        "Most bytes allocated at once: {worst_op:?}, {}",
        format_size(eval, BINARY)
    );
}
