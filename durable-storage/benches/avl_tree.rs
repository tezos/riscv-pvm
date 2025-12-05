// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::time::Duration;

use bytes::Bytes;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use octez_riscv_durable_storage::merkle_layer::KEY_MAX_SIZE;
use octez_riscv_durable_storage::merkle_layer::Key;
use octez_riscv_durable_storage::merkle_layer::tree::Avl;
use rand::prelude::*;

const KEY_COUNT: usize = 10_000_000;
const OPERATIONS_PER_SAMPLE: usize = 10_000;

#[derive(Debug, Clone)]
pub enum Operation {
    Get(Key),
    Upsert(Key, Bytes),
    Delete(Key),
}

fn generate_byte_vector(rng: &mut impl Rng, length: usize) -> Vec<u8> {
    let mut ret = vec![0u8; length];
    rng.fill(ret.as_mut_slice());
    ret
}

fn generate_keys(rng: &mut impl Rng, length: usize) -> Vec<Key> {
    let mut tmp: HashSet<Key> = HashSet::new();
    while tmp.len() < length {
        let key_length = rng.random_range(1..KEY_MAX_SIZE);
        tmp.insert(
            Key::new(generate_byte_vector(rng, key_length).as_slice())
                .expect("The key should be created"),
        );
    }
    tmp.into_iter().collect()
}

fn get_random_data(rng: &mut impl Rng) -> Bytes {
    let length: usize = rng.random_range(1..20);
    let bytes = generate_byte_vector(rng, length);
    Bytes::from(bytes)
}

fn get_operations_batch(rng: &mut impl Rng, keys: &[Key], batch_size: usize) -> Vec<Operation> {
    (0..batch_size)
        .map(|_| {
            let key: Key = keys
                .choose(rng)
                .expect("The keys array is not empty")
                .clone();
            match rng.random_range(0..3) {
                0 => Operation::Get(key),
                1 => Operation::Upsert(key, get_random_data(rng)),
                _ => Operation::Delete(key),
            }
        })
        .collect()
}

/// This bench inserts half of the [`KEY_COUNT`]
/// generated keys into an AVL tree and samples from
/// all of them for the get, set and delete operations
/// on the tree.
fn bench_avl_tree_operations(c: &mut Criterion) {
    let mut rng = rand::rng();
    let keys = generate_keys(&mut rng, KEY_COUNT);

    // Setting up the tree
    let mut tree = Avl::default();
    for key in &keys[..keys.len() / 2] {
        tree.set(key, get_random_data(&mut rng));
    }

    c.bench_function("Bench AVL tree with operations", |b| {
        b.iter_batched(
            || get_operations_batch(&mut rng, keys.as_slice(), OPERATIONS_PER_SAMPLE),
            |operations| {
                for operation in operations {
                    match operation {
                        Operation::Get(key) => {
                            tree.get(&key);
                        }
                        Operation::Upsert(key, value) => {
                            tree.set(&key, value);
                        }
                        Operation::Delete(key) => {
                            tree.delete(&key);
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
        tree.insert(key.clone(), get_random_data(&mut rng));
    }

    c.bench_function("BTreeMap reference", |b| {
        b.iter_batched(
            || get_operations_batch(&mut rng, keys.as_slice(), OPERATIONS_PER_SAMPLE),
            |operations| {
                for operation in operations {
                    match operation {
                        Operation::Get(key) => {
                            tree.get(&key);
                        }
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
