// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-License-Identifier: MIT

//! Benchmark for hashing `Bytes` values of various sizes with arity 2 vs arity 4.
//!
//! The page tree of [`octez_riscv_data::components::bytes::Bytes`] currently uses an arity of 4.
//! This benchmark measures the wall-clock cost of hashing values of multiple sizes using either
//! arity, so the trade-off between proof structural size and hashing throughput can be evaluated.

use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::mode::Normal;

const SIZES: &[(&str, usize)] = &[
    ("16KiB", 16 * 1024),
    // ("32KiB", 32 * 1024),
    // ("64KiB", 64 * 1024),
    // ("4MiB", 4 * 1024 * 1024),
    // ("32MiB", 32 * 1024 * 1024),
    // ("64MiB", 64 * 1024 * 1024),
];

fn build_value(size: usize) -> Bytes<Normal> {
    Bytes::from(&vec![0u8; size][..])
}

fn bench_bytes_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes_hash");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for (label, size) in SIZES {
        let value = build_value(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("arity_2", label), &value, |b, value| {
            b.iter(|| {
                let hash = black_box(value).hash_with_arity(black_box(2));
                black_box(hash);
            });
        });

        group.bench_with_input(BenchmarkId::new("arity_4", label), &value, |b, value| {
            b.iter(|| {
                let hash = black_box(value).hash_with_arity(black_box(4));
                black_box(hash);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bytes_hash);
criterion_main!(benches);
