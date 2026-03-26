// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod erc20_core;
mod random;

use std::time::Duration;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_durable_storage::database::Database;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_test_utils::TestableTmpdir;

use crate::erc20_core::BenchmarkState;
use crate::erc20_core::bench_run;
use crate::erc20_core::build_template;

fn database_checkout_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("Creating a Tokio runtime should succeed");

    let tmpdir = TestableTmpdir::new();
    let repo = DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
    let (template, database) = build_template(runtime.handle(), &repo);
    let commit = database
        .commit(&repo)
        .expect("Committing the initial benchmark state should succeed");

    let setup = || {
        let database = Database::checkout(runtime.handle(), &repo, commit)
            .expect("Checking out the committed database should succeed");

        BenchmarkState {
            database,
            operations: template.operations.clone(),
            random_data: template.random_data.clone(),
            read_buffer: vec![0u8; template.read_buffer_len],
            handle: runtime.handle(),
            repo: &repo,
        }
    };

    let mut group = c.benchmark_group("ERC-20 with checkout");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));
    group.bench_function("Transactions", |b| {
        b.iter_batched(setup, bench_run, BatchSize::SmallInput)
    });

    group.finish();
}
criterion_group!(benches, database_checkout_benchmark);
criterion_main!(benches);
