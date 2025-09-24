// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Data structures for managing addressed instructions
//!
//! This module provides data structures and utilities to maintain a mapping between
//! instruction addresses and their associated information. It includes [`AddrMap`] for storing
//! addresses, [`InstrMap`] for storing instruction information, and [`InstrMapBuilder`] for
//! constructing these maps in a consistent manner.

// TODO: RV-703 - `InstrMap` is currently only used in tests and has not yet been integrated into the JIT.
#![cfg(test)]

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::Index;
use std::ops::IndexMut;

use crate::machine_state::memory::Address;

/// A unique identifier for an instruction in an [`InstrMap`] or the associated [`AddrMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::Into)]
pub struct InstrId(usize);

/// An array type holding [`Address`]es of instr-info types
/// in a matching [`InstrMap`].
///
/// Paired items across the types are located by the same [`InstrId`].
/// Addresses are stored in sorted order to allow binary searching.
#[derive(Debug, Clone)]
pub struct AddrMap {
    addresses: Vec<Address>,
}

impl AddrMap {
    /// Find an instruction with the given address and translate it to an instruction identifier.
    pub fn translate(&self, address: Address) -> Option<InstrId> {
        self.addresses.binary_search(&address).ok().map(InstrId)
    }
}

impl Index<InstrId> for AddrMap {
    type Output = Address;

    fn index(&self, index: InstrId) -> &Self::Output {
        &self.addresses[index.0]
    }
}

/// An array holding instr-info types with a matching [`AddrMap`].
///
/// Paired items across the types are located by the same [`InstrId`].
#[derive(Debug, Clone)]
pub struct InstrMap<V> {
    instructions: Vec<V>,
}

impl<V> InstrMap<V> {
    /// Get the number of instructions in the map.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Iterate over the instructions, yielding pairs of (`InstrId`, `&V`).
    pub fn iter(&self) -> impl Iterator<Item = (InstrId, &V)> {
        self.instructions
            .iter()
            .enumerate()
            .map(|(idx, value)| (InstrId(idx), value))
    }

    /// Iterate over the instructions, yielding pairs of (`InstrId`, `&mut V`).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (InstrId, &mut V)> {
        self.instructions
            .iter_mut()
            .enumerate()
            .map(|(idx, value)| (InstrId(idx), value))
    }

    /// Transform the values in the `InstrMap` using the provided function, producing a new `InstrMap`.
    pub fn map<F, U>(&self, mut f: F) -> InstrMap<U>
    where
        F: FnMut(InstrId, &V) -> U,
    {
        let instructions = self
            .iter()
            .map(|(instr_id, val)| f(instr_id, val))
            .collect();
        InstrMap { instructions }
    }
}

impl<V> Index<InstrId> for InstrMap<V> {
    type Output = V;

    fn index(&self, index: InstrId) -> &Self::Output {
        &self.instructions[index.0]
    }
}

impl<V> IndexMut<InstrId> for InstrMap<V> {
    fn index_mut(&mut self, index: InstrId) -> &mut Self::Output {
        &mut self.instructions[index.0]
    }
}

/// Builder for initialising an ([`AddrMap`], [`InstrMap`]) pair.
/// The builder ensures that the two maps are consistent with each other.
/// The keys of the instruction BTreeMap are the addresses, and the values are the
/// associated instruction information.
pub struct InstrMapBuilder<T> {
    instructions: BTreeMap<Address, T>,
}

impl<T> InstrMapBuilder<T> {
    /// Create a new builder with the specified capacity.
    pub fn new() -> Self {
        Self {
            instructions: BTreeMap::new(),
        }
    }

    /// Insert a new address and instruction pair to the builder. If the address
    /// already exists, the existing instruction will be replaced with the new one.
    pub fn insert(&mut self, address: Address, instruction: T) {
        self.instructions.insert(address, instruction);
    }

    /// Finalise the builder and produce the (`AddrMap`, `InstrMap`) pair.
    pub fn build(self) -> (AddrMap, InstrMap<T>) {
        let (addresses, instructions) = self.instructions.into_iter().unzip();

        let addr_map = AddrMap { addresses };
        let instr_map = InstrMap { instructions };

        (addr_map, instr_map)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::*;

    proptest! {
        #[test]
        fn resolution(addresses: Vec<u64>) {
            let mut instr_map_builder = InstrMapBuilder::new();

            for &address in addresses.iter() {
                instr_map_builder.insert(address, address);
            }

            let (addr_map, instr_map) = instr_map_builder.build();

            for (idx, &addr) in instr_map.iter() {
                // Every index in the instruction map should be resolvable to an address.
                let found_addr = addr_map[idx];
                prop_assert_eq!(found_addr, addr);

                // Vice versa, every address in the instruction map should be resolvable to an
                // index.
                let found_idx = addr_map.translate(addr);
                prop_assert_eq!(found_idx, Some(idx));
            }

            for address in addresses {
                // Every address that was inserted should be resolvable to an index.
                let found_idx = addr_map.translate(address).expect("Inserted address should be recoverable");
                let found_addr = addr_map[found_idx];
                assert_eq!(found_addr, address);

                // The same address should exist in the instruction map at the same index.
                let instr_address = instr_map[found_idx];
                assert_eq!(instr_address, address);
            }
        }

        #[test]
        fn instr_map_iterators_and_map_function_work(instructions: Vec<u64>) {
            let addresses = (0..instructions.len() as u64).collect::<Vec<_>>(); // Dummy addresses

            let mut instr_map_builder = InstrMapBuilder::new();
            for (address, instruction) in addresses.iter().cloned().zip(instructions.iter().cloned()) {
                instr_map_builder.insert(address, instruction);
            }
            let (_addr_map, instr_map) = instr_map_builder.build();

            // Test iter
            let mut count = 0;
            for (instr_id, &value) in instr_map.iter() {
                prop_assert_eq!(value, instructions[instr_id.0]);
                count += 1;
            }
            prop_assert_eq!(count, instructions.len());

            // Test iter_mut
            let mut instr_map_mut = instr_map.clone();
            count = 0;
            for (instr_id, value) in instr_map_mut.iter_mut() {
                *value += 1; // Modify the value
                prop_assert_eq!(*value, instructions[instr_id.0] + 1);
                count += 1;
            }
            prop_assert_eq!(count, instructions.len());

            // Test map function
            let mapped_instr_map = instr_map.map(|_, &val| val.wrapping_add(1));
            prop_assert_eq!(mapped_instr_map.instructions.len(), instructions.len());
            for (instr_id, &val) in mapped_instr_map.iter() {
                prop_assert_eq!(val, instructions[instr_id.0].wrapping_add(1));
            }
        }

        #[test]
        fn instr_map_builder_inserts_and_builds_correctly(addr_instr_pairs: Vec<(u64, u64)>) {
            let mut expected_pairs = addr_instr_pairs.clone();
            expected_pairs.sort_by_key(|&(addr, _)| addr);
            expected_pairs.dedup_by_key(|&mut (addr, _)| addr);

            let mut builder = InstrMapBuilder::new();
            for (address, instruction) in addr_instr_pairs {
                builder.insert(address, instruction);
            }

            let (addr_map, instr_map) = builder.build();

            // Check we have the same number of addresses and instructions.
            prop_assert_eq!(addr_map.addresses.len(), instr_map.instructions.len());

            for (id, &instr) in instr_map.iter() {
                let addr = addr_map[id];
                prop_assert_eq!(addr, expected_pairs[id.0].0);
                prop_assert_eq!(instr, expected_pairs[id.0].1);
            }
        }
    }
}
