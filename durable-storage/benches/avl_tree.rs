// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::time::Duration;

use bytes::Bytes;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_durable_storage::bench::ArcResolver;
use octez_riscv_durable_storage::bench::Tree;
use octez_riscv_durable_storage::bench::generate_keys;
use octez_riscv_durable_storage::bench::generate_random_bytes_in_range;
use octez_riscv_durable_storage::key::Key;
use rand::prelude::*;

const KEY_COUNT: usize = 10_000_000;
const OPERATIONS_PER_SAMPLE: usize = 10_000;

#[derive(Debug, Clone)]
enum Operation {
    Upsert(Key, Bytes),
    Delete(Key),
}

fn get_operations_batch(mut rng: &mut impl Rng, keys: &[Key], batch_size: usize) -> Vec<Operation> {
    (0..batch_size)
        .map(|_| {
            let key: Key = keys
                .choose(rng)
                .expect("The keys array is not empty")
                .clone();
            match rng.random_range(0..2) {
                0 => Operation::Upsert(key, generate_random_bytes_in_range(&mut rng, 1..20).into()),
                _ => Operation::Delete(key),
            }
        })
        .collect()
}

/// This bench inserts half of the [`KEY_COUNT`]
/// generated keys into an AVL tree and samples from
/// all of them for the set and delete operations
/// on the tree.
fn bench_avl_tree_operations(c: &mut Criterion) {
    let mut rng = rand::rng();
    let keys = generate_keys(&mut rng, KEY_COUNT);
    let mut resolver = ArcResolver;

    // Setting up the tree
    let mut tree = Tree::default();
    for key in &keys[..keys.len() / 2] {
        let random_data = generate_random_bytes_in_range(&mut rng, 1..20);
        tree.set(key, &random_data, &mut resolver);
    }

    c.bench_function("Bench AVL tree with operations", |b| {
        b.iter_batched(
            || get_operations_batch(&mut rng, keys.as_slice(), OPERATIONS_PER_SAMPLE),
            |operations| {
                for operation in operations {
                    match operation {
                        Operation::Upsert(key, value) => {
                            tree.set(&key, &value, &mut resolver);
                        }
                        Operation::Delete(key) => {
                            tree.delete(&key, &mut resolver);
                        }
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

/// This benchmark is using the same workload
/// against a B-tree from the standard library
/// so we can compare our solution to an
/// optimized ordered container.
fn reference(c: &mut Criterion) {
    let mut rng = rand::rng();
    let keys = generate_keys(&mut rng, KEY_COUNT);

    let mut tree = BTreeMap::<Key, Bytes>::new();
    for key in &keys[..keys.len() / 2] {
        tree.insert(
            key.clone(),
            generate_random_bytes_in_range(&mut rng, 1..20).into(),
        );
    }

    c.bench_function("BTreeMap reference", |b| {
        b.iter_batched(
            || get_operations_batch(&mut rng, keys.as_slice(), OPERATIONS_PER_SAMPLE),
            |operations| {
                for operation in operations {
                    match operation {
                        Operation::Upsert(key, value) => {
                            tree.insert(key, value);
                        }
                        Operation::Delete(key) => {
                            tree.remove(&key);
                        }
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(30))
        .warm_up_time(Duration::from_secs(2))
        .sample_size(100)
}

criterion_group!(
    name = benches;
    config = configure_criterion();
    targets = bench_avl_tree_operations, reference
);
criterion_main!(benches);
