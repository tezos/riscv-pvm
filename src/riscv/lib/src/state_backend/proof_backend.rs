// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Proof-generating backend
//!
//! Generic backend used for PVM proof generation, which wraps a manager and
//! records all state accesses performed during an evaluation step.
//! After evaluation, a [`MerkleTree`] over the PVM state can be obtained,
//! which can be partially blinded to produce a proof as a partial Merkle tree.
//! The structure of the Merkle tree is informed by the layout of the state.
//!
//! [`MerkleTree`]: octez_riscv_data::merkle_tree::MerkleTree

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::EncodeError;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::utils::Source;
use octez_riscv_data::serialisation::elem::Elem;

use super::ManagerBase;
use super::ManagerRead;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerClone;

pub mod merkle;
pub mod proof;

impl<'normal> ManagerBase for Prove<'normal> {
    type DynRegion = ProofDynRegion<'normal>;
}

impl<'normal> ManagerAlloc for Prove<'normal> {
    fn allocate_dyn_region(len: usize) -> Self::DynRegion {
        let source = Normal::allocate_dyn_region(len);
        ProofDynRegion {
            source: Source::from(source),
            reads: RefCell::default(),
            writes: BTreeMap::new(),
            did_access_length: Cell::new(false),
        }
    }
}

/// Implementation of [`ManagerRead`] which wraps another manager and
/// additionally records read locations.
impl<'normal> ManagerRead for Prove<'normal> {
    fn dyn_region_len(region: &Self::DynRegion) -> usize {
        region.len()
    }

    unsafe fn dyn_region_read<E: Elem>(region: &Self::DynRegion, address: usize) -> E {
        region.reads.borrow_mut().insert::<E>(address);
        region.unrecorded_read(address)
    }
}

/// Implementation of [`ManagerWrite`] which wraps another manager and
/// records written locations but does not write to the wrapped region directly.
impl<'normal> ManagerWrite for Prove<'normal> {
    unsafe fn dyn_region_write<E: Elem>(region: &mut Self::DynRegion, address: usize, value: E) {
        debug_assert!(address + E::STORED_SIZE.get() <= region.unrecorded_len());

        let mut buffer = vec![0u8; E::STORED_SIZE.get()];
        Self::dyn_region_read_all(region, address, &mut buffer);

        // SAFETY: The buffer has been allocated with sufficient space.
        unsafe {
            value.write_unaligned(buffer.as_mut_ptr());
        }

        for (offset, byte) in buffer.into_iter().enumerate() {
            region.writes.insert(address + offset, byte);
        }
    }
}

/// Implementation of [`ManagerSerialise`] which wraps another manager and
/// serialises data as recorded during the `Prove` mode, reconstructed
/// via variants of [`ManagerRead`] functions which do not record access
/// information.
impl<'normal> ManagerSerialise for Prove<'normal> {
    fn serialise_dyn_region<E: Encoder>(
        region: &Self::DynRegion,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        if region.writes.is_empty() {
            // If no writes were recorded, we can serialise the underlying dynamic region as is.
            return Normal::serialise_dyn_region(&region.source, encoder);
        }

        // This variable keeps the index of the next item from the region that should be written.
        let mut write_index = 0;

        for (&index, value) in region.writes.iter() {
            // There are items before the current index that have not been written yet.
            if write_index < index {
                let to_be_written = index - write_index;
                let mut buffer = vec![0u8; to_be_written];
                Normal::dyn_region_read_all(&region.source, write_index, &mut buffer);
                encoder.writer().write(&buffer)?;
            }

            encoder.writer().write(&[*value])?;

            // Make sure we expect to write the next item after the current.
            write_index = index.saturating_add(1);
        }

        // Write the remaining items from the region that were not written yet.
        let to_be_written = region.unrecorded_len().saturating_sub(write_index);
        if to_be_written > 0 {
            let mut buffer = vec![0u8; to_be_written];
            Normal::dyn_region_read_all(&region.source, write_index, &mut buffer);
            encoder.writer().write(&buffer)?;
        }

        Ok(())
    }
}

impl<'normal> ManagerClone for Prove<'normal> {
    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion {
        region.clone()
    }
}

/// Proof dynamic region which wraps a dynamic region managed by another manager.
///
/// When Merkleising a [`ManagerBase::DynRegion`], its data can be split into multiple leaves.
/// Accesses are thus recorded for each address.
/// The underlying dynamic region is never mutated, but all written bytes are
/// recorded in order to preserve the integrity of subsequent reads.
pub struct ProofDynRegion<'normal> {
    source: Source<'normal, <Normal as ManagerBase>::DynRegion>,
    reads: RefCell<DynAccess>,
    writes: BTreeMap<usize, u8>,

    /// Was the length of the dynamic region accessed?
    did_access_length: Cell<bool>,
}

impl<'normal> ProofDynRegion<'normal> {
    /// Bind a pre-existing dynamic region.
    pub fn bind(source: &'normal <Normal as ManagerBase>::DynRegion) -> Self {
        Self {
            source: Source::Borrowed(source),
            reads: RefCell::default(),
            writes: BTreeMap::new(),
            did_access_length: Cell::new(false),
        }
    }

    /// Get the set of addresses of the region that were read from.
    /// This function is meant to be called once when Merkleising the region.
    pub fn get_read(&self) -> DynAccess {
        self.reads.take()
    }

    /// Get the set of addresses of the region that were written to.
    /// This function is meant to be called once when Merkleising the region.
    pub fn get_write(&self) -> DynAccess {
        let writes: BTreeSet<_> = self.writes.keys().copied().collect();
        DynAccess(writes)
    }

    /// Figure out whether the part of the Merkle tree that contains the dynamic region length
    /// needs to be present.
    ///
    /// Generally that is when the length was retrieved, or when any other read or write occurred
    /// within the dynamic region.
    pub(crate) fn need_length_in_proof(&self) -> bool {
        self.did_access_length.get() || !self.reads.borrow().is_empty() || !self.writes.is_empty()
    }
}

impl<'normal> ProofDynRegion<'normal> {
    /// Read from the wrapped dynamic region.
    ///
    /// # Safety
    ///
    /// See [`ManagerRead::dyn_region_read`] for safety requirements.
    pub unsafe fn inner_dyn_region_read<E: Elem>(&self, address: usize) -> E {
        unsafe { Normal::dyn_region_read(&self.source, address) }
    }

    /// Version of [`ManagerRead::dyn_region_read`] which does not record
    /// the access as a read.
    fn unrecorded_read<E: Elem>(&self, address: usize) -> E {
        assert!(address + E::STORED_SIZE.get() <= self.unrecorded_len());

        // Read the underlying bytes of the value.
        let mut value_bytes = vec![0u8; E::STORED_SIZE.get()];
        Normal::dyn_region_read_all(&self.source, address, &mut value_bytes);

        // Overwrite any byte that has been written during the proof step.
        for (&i, &byte) in self.writes.range(address..address + E::STORED_SIZE.get()) {
            value_bytes[i - address] = byte;
        }

        // SAFETY: The vector has been allocated with sufficient space.
        unsafe { E::read_unaligned(value_bytes.as_ptr()) }
    }

    /// Get the length of the dynamic region.
    fn len(&self) -> usize {
        self.did_access_length.set(true);
        self.unrecorded_len()
    }

    /// Like [`Self::len`], but does not record the access as a read.
    pub(crate) fn unrecorded_len(&self) -> usize {
        // XXX: This implies the size can't change in a proof.
        Normal::dyn_region_len(&self.source)
    }
}

impl<'normal> Clone for ProofDynRegion<'normal> {
    fn clone(&self) -> Self {
        Self {
            source: match &self.source {
                Source::Borrowed(source) => Source::Borrowed(source),
                Source::Owned(source) => Source::from(Normal::clone_dyn_region(source)),
            },
            reads: self.reads.clone(),
            writes: self.writes.clone(),
            did_access_length: self.did_access_length.clone(),
        }
    }
}

/// A record of accessed addresses in a dynamic region
#[derive(Default, Clone)]
pub struct DynAccess(BTreeSet<usize>);

impl DynAccess {
    /// Insert all addresses touched while accessing an element of a given size.
    pub fn insert<E: Elem>(&mut self, address: usize) {
        self.0.extend(address..address + E::STORED_SIZE.get())
    }

    /// Check whether any address within a given range of addresses
    /// has been accessed.
    pub fn includes_range(&self, r: std::ops::Range<usize>) -> bool {
        self.0.range(r).next().is_some()
    }

    /// Check whether no address has been accessed.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
