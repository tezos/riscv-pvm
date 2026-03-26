// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

mod erc20_core;
mod random;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_durable_storage::database::DirectoryManager;
use octez_riscv_test_utils::TestableTmpdir;
use tokio::runtime::Handle;

use crate::erc20_core::BenchmarkState;
use crate::erc20_core::bench_run;
use crate::erc20_core::build_template;

impl<'a> BenchmarkState<'a> {
    /// Clones the benchmark state onto the given runtime handle and repository.
    ///
    /// This is used by Criterion setup closures so each timed iteration starts
    /// from an identical database snapshot and operation list.
    pub fn clone_with(&self, handle: &'a Handle, repo: &'a DirectoryManager) -> Self {
        Self {
            database: self
                .database
                .try_clone_with(handle, repo)
                .expect("Cloning the database should work"),
            operations: self.operations.clone(),
            read_buffer: vec![0u8; self.read_buffer.len()],
            random_data: self.random_data.clone(),
            handle,
            repo,
        }
    }
}

fn setup_benchmark_state<'a>(handle: &'a Handle, repo: &'a DirectoryManager) -> BenchmarkState<'a> {
    let (template, database) = build_template(handle, repo);
    BenchmarkState {
        database,
        operations: template.operations,
        random_data: template.random_data,
        read_buffer: vec![0u8; template.read_buffer_len],
        handle,
        repo,
    }
}

fn database_benchmark(c: &mut Criterion) {
    // Two threads are needed so the thread for receiving the hashes isn't blocked
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("Creating a Tokio runtime should succeed");

    let tmpdir = TestableTmpdir::new();
    let repo = DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
    let initial_state = setup_benchmark_state(runtime.handle(), &repo);

    let setup = || initial_state.clone_with(runtime.handle(), &repo);

    let mut group = c.benchmark_group("ERC-20");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(120));
    group.bench_function("Transactions", |b| {
        b.iter_batched(setup, bench_run, BatchSize::SmallInput)
    });

    group.finish();
}

criterion_group!(benches, database_benchmark);
criterion_main!(benches);
