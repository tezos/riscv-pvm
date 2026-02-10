// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::collections::HashSet;
use std::ops::Range;

use octez_riscv_durable_storage::key::KEY_MAX_SIZE;
use octez_riscv_durable_storage::key::Key;
use rand::prelude::*;

pub fn generate_keys(rng: &mut impl Rng, length: usize) -> Vec<Key> {
    let mut tmp: HashSet<Key> = HashSet::with_capacity(length);
    while tmp.len() < length {
        tmp.insert(
            Key::new(generate_random_bytes_in_range(rng, 1..KEY_MAX_SIZE).as_slice())
                .expect("The key should be created"),
        );
    }
    tmp.into_iter().collect()
}

/// Generates a vector of random bytes with a given length
pub fn generate_random_bytes(rng: &mut impl Rng, length: usize) -> Vec<u8> {
    let mut vec = Vec::with_capacity(length);
    unsafe {
        rng.fill(std::slice::from_raw_parts_mut(vec.as_mut_ptr(), length));
        vec.set_len(length);
    }
    vec
}

/// Generates a vector of random bytes with a length in the range `length_range`.
pub fn generate_random_bytes_in_range(rng: &mut impl Rng, length_range: Range<usize>) -> Vec<u8> {
    let len = rng.random_range(length_range);
    generate_random_bytes(rng, len)
}
