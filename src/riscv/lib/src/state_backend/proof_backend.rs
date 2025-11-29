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
use std::ops::Deref;

use bincode::enc::Encode;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::EncodeError;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;

use super::FnManager;
use super::ManagerBase;
use super::ManagerRead;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::state_backend::Elem;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerClone;
use crate::state_backend::elem_bytes;

pub mod merkle;
pub mod proof;

impl<'normal> ManagerBase for Prove<'normal> {
    type Region<E: 'static, const LEN: usize> = ProofRegion<'normal, E, LEN>;

    type DynRegion = ProofDynRegion<'normal>;

    type ManagerRoot = Self;
}

impl<'normal> ManagerAlloc for Prove<'normal> {
    fn allocate_region<E, const LEN: usize>(init_value: [E; LEN]) -> Self::Region<E, LEN> {
        let source = Normal::allocate_region::<E, LEN>(init_value);
        ProofRegion {
            source: Source::Owned(source),
            writes: BTreeMap::new(),
            access: Cell::new(false),
        }
    }

    fn allocate_dyn_region(len: usize) -> Self::DynRegion {
        let source = Normal::allocate_dyn_region(len);
        ProofDynRegion {
            source: Source::Owned(source),
            reads: RefCell::default(),
            writes: BTreeMap::new(),
            did_access_length: Cell::new(false),
        }
    }
}

/// Implementation of [`ManagerRead`] which wraps another manager and
/// additionally records read locations.
impl<'normal> ManagerRead for Prove<'normal> {
    fn region_read<E: Copy, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> E {
        *Self::region_ref(region, index)
    }

    fn region_ref<E: 'static, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> &E {
        region.set_access_info();
        region.unrecorded_ref(index)
    }

    fn region_read_all<E: Copy, const LEN: usize>(region: &Self::Region<E, LEN>) -> Vec<E> {
        (0..LEN).map(|i| Self::region_read(region, i)).collect()
    }

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
    fn region_write<E, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    ) {
        region.set_access_info();
        region.writes.insert(index, value);
    }

    fn region_write_all<E: Copy, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        values: &[E],
    ) {
        for (i, value) in values.iter().enumerate() {
            Self::region_write(region, i, *value);
        }
    }

    unsafe fn dyn_region_write<E: Elem>(region: &mut Self::DynRegion, address: usize, value: E) {
        debug_assert!(address + E::STORED_SIZE.get() <= region.unrecorded_len());

        for (offset, byte) in elem_bytes(value).into_iter().enumerate() {
            region.writes.insert(address + offset, byte);
        }
    }
}

/// Implementation of [`ManagerSerialise`] which wraps another manager and
/// serialises data as recorded during the `Prove` mode, reconstructed
/// via variants of [`ManagerRead`] functions which do not record access
/// information.
impl<'normal> ManagerSerialise for Prove<'normal> {
    fn serialise_region<T: Encode, const LEN: usize, E: Encoder>(
        region: &Self::Region<T, LEN>,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        if region.writes.is_empty() {
            // If no writes were recorded, we can serialise the underlying region as is.
            return Normal::serialise_region(&region.source, encoder);
        }

        // This variable keeps the index of the next item from the region that should be written.
        let mut write_index = 0;

        for (&index, value) in region.writes.iter() {
            // There are items before the current index that have not been written yet.
            if write_index < index {
                for i in write_index..index {
                    Normal::region_ref(&region.source, i).encode(&mut encoder)?;
                }
            }

            value.encode(&mut encoder)?;

            // Make sure we expect to write the next item after the current.
            write_index = index.saturating_add(1);
        }

        // Write the remaining items from the region that were not written yet.
        let source = &region.source;
        for i in write_index..LEN {
            Normal::region_ref(source, i).encode(&mut encoder)?;
        }

        Ok(())
    }

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
    fn clone_region<E: 'static + Clone, const LEN: usize>(
        region: &Self::Region<E, LEN>,
    ) -> Self::Region<E, LEN> {
        region.clone()
    }

    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion {
        region.clone()
    }
}

/// Source for a region or dynamic region, either borrowed or owned
///
/// We use this type to store the source version of the underlying proof data. Merkle tree proofs
/// usually encode the initial state. Hence we need to figure out what parts of the initial state
/// need to go into the proof.
///
/// Normally the source is borrowed from state in [`Normal`] mode. In those cases, it is enough to
/// obtain a reference to the source.
///
/// However, there are cases when no source can be borrowed, because it is owned. During the
/// proof generation, we might create a new region. This can happen when a dynamic state
/// component holds regions. E.g. it can grow and therefore create new regions. During proof
/// generation we don't differentiate between modes, hence you can't create a region in [`Normal`]
/// mode and then borrow it.
///
/// Wherever newly created regions are allowed, the upstream state which allows this to happen is in
/// charge of preventing those regions from being included in the Merkle tree proof. In the example
/// above, that would be the growable state component. It needs to differentiate between regions that
/// existed before the proof generation started, and those that were created during proof.
///
/// It is possible to override regions when we allow region creation. Overriding isn't the same as
/// writing the underlying values from one region to the other as it would override the region state
/// that is specific to the mode. This would be caught by producing an invalid Merkle tree proof in
/// our test suite. In such cases, the claimed initial state hash of the Merkle tree would not match
/// that of the actual initial state in [`Normal`] mode.
///
/// Another benefit of allowing regions to be created from an owned source is that it simplifies our
/// test suite where we want to instantiate a state using the [`Prove`] mode without a previous
/// state in [`Normal`] mode. This simplifies the API for tests and allows us to write one test and
/// test it against 3 different backends.
///
/// It is possible to simplify this type by only allowing owned regions. This has some downsides,
/// as it would require all [`Normal`] regions to be cloned when wrapping them into [`Prove`]. This
/// could be costly and unnecessary if the region isn't even used.
#[derive(Clone)]
enum Source<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> Deref for Source<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Source::Borrowed(value) => value,
            Source::Owned(value) => value,
        }
    }
}

/// Proof region which wraps a region managed by another manager.
///
/// A [`ManagerBase::Region`] is never split across multiple leaves when Merkleised.
/// An access to any part of the region is thus recorded as an access to the region as a whole.
/// The underlying region is never mutated, but all written values are recorded
/// in order to preserve the integrity of subsequent reads.
#[derive(Clone)]
pub struct ProofRegion<'normal, E: 'static, const LEN: usize> {
    /// See documentation of [`Source`].
    source: Source<'normal, <Normal as ManagerBase>::Region<E, LEN>>,
    writes: BTreeMap<usize, E>,
    access: Cell<bool>,
}

impl<'normal, E: 'static, const LEN: usize> ProofRegion<'normal, E, LEN> {
    /// Bind a pre-existing region.
    pub fn bind(source: &'normal <Normal as ManagerBase>::Region<E, LEN>) -> Self {
        Self {
            source: Source::Borrowed(source),
            writes: BTreeMap::new(),
            access: Cell::new(false),
        }
    }

    /// Get a copy of the access log.
    pub fn get_access_info(&self) -> bool {
        self.access.get()
    }

    /// Record that the regions has been accessed
    pub fn set_access_info(&self) {
        self.access.set(true)
    }

    /// Get a reference to the wrapper region.
    pub fn inner_region_ref(&self) -> &<Normal as ManagerBase>::Region<E, LEN> {
        &self.source
    }
}

impl<'normal, E: 'static, const LEN: usize> ProofRegion<'normal, E, LEN> {
    /// Version of [`ManagerRead::region_ref`] which does not record
    /// the access as a read.
    fn unrecorded_ref(&self, index: usize) -> &E {
        self.writes
            .get(&index)
            .unwrap_or_else(|| Normal::region_ref(&self.source, index))
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
                Source::Owned(source) => Source::Owned(Normal::clone_dyn_region(source)),
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

/// Natural transformation from a [`Normal`] to a proof-generating mode [`Prove`]
pub enum ProofWrapper {}

impl<'normal> FnManager<'normal, Normal> for ProofWrapper {
    type Output = Prove<'normal>;

    fn map_region<E: 'static, const LEN: usize>(
        input: &'normal <Normal as ManagerBase>::Region<E, LEN>,
    ) -> <Prove<'normal> as ManagerBase>::Region<E, LEN> {
        ProofRegion::bind(input)
    }

    fn map_dyn_region(
        input: &'normal <Normal as ManagerBase>::DynRegion,
    ) -> <Prove<'normal> as ManagerBase>::DynRegion {
        ProofDynRegion::bind(input)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Normal;
    use proptest::array;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::merkle::MERKLE_LEAF_SIZE;
    use super::*;
    use crate::state_backend::Cells;
    use crate::state_backend::DynCells;
    use crate::state_backend::ManagerAlloc;
    use crate::state_backend::Ref;

    const CELLS_SIZE: usize = 32;

    #[test]
    fn test_proof_gen_region() {
        proptest!(|(value_before: u64, value_after: u64, i in 0..CELLS_SIZE)| {
            // A read followed by a write
            let cells = [value_before; CELLS_SIZE];
            let region: ProofRegion<'_, u64, CELLS_SIZE> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(region);

            prop_assert!(!region.region_ref().get_access_info());
            let value = region.read(i);
            prop_assert_eq!(value, value_before);
            prop_assert!(region.region_ref().get_access_info());
            region.write(i, value_after);
            prop_assert!(region.region_ref().get_access_info());

            // A write followed by a read
            let cells = [value_before; CELLS_SIZE];
            let region: ProofRegion<'_, u64, CELLS_SIZE> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            region.write(i, value_after);
            prop_assert!(region.region_ref().get_access_info());
            let value = region.read(i);
            prop_assert_eq!(value, value_after);
            prop_assert!(region.region_ref().get_access_info());

            let data_before = [value_before; CELLS_SIZE];
            let data_after = [value_after; CELLS_SIZE];

            // A read_all followed by a write_all
            let cells = data_before;
            let region: ProofRegion<'_, u64, CELLS_SIZE> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            let values = region.read_all();
            prop_assert_eq!(values.as_slice(), data_before);
            prop_assert!(region.region_ref().get_access_info());
            region.write_all(&data_after);
            prop_assert!(region.region_ref().get_access_info());

            // A write_all followed by a read_all
            let cells = data_before;
            let region: ProofRegion<'_, u64, CELLS_SIZE> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            region.write_all(&data_after);
            prop_assert!(region.region_ref().get_access_info());
            let values = region.read_all();
            prop_assert_eq!(values.as_slice(), data_after);
            prop_assert!(region.region_ref().get_access_info());

            // Check correct Merkleisation
            let cells = [value_before; CELLS_SIZE];
            let cells_owned: Cells<u64, CELLS_SIZE, Normal> = Cells::bind(cells);
            let initial_root_hash = Hash::from_foldable(&cells_owned);

            let mut proof_region: ProofRegion<'_, u64, CELLS_SIZE> = ProofRegion::bind(&cells);
            Prove::region_write(&mut proof_region, i, value_after);
            let proof_cells: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(proof_region);

            let merkle_tree = MerkleTree::from_foldable(&proof_cells);
            merkle_tree.check_root_hash();
            match merkle_tree {
                MerkleTree::Leaf(hash, access_info, _) => {
                    prop_assert_eq!(hash, initial_root_hash);
                    prop_assert!(access_info);
                }
                _ => panic!("Expected Merkle tree to contain a single written leaf"),
            }
        });
    }

    const LEAVES: usize = 8;
    const DYN_REGION_SIZE: usize = MERKLE_LEAF_SIZE.get() * LEAVES;
    const ELEM_SIZE: usize = u64::STORED_SIZE.get();

    #[test]
    fn test_proof_gen_dyn_region() {
        if ELEM_SIZE > MERKLE_LEAF_SIZE.get() {
            unreachable!(
                "This test assumes that a single element does not span more than 2 leaves"
            );
        }
        let address_range = 0..DYN_REGION_SIZE - ELEM_SIZE;

        // Check that writing to an address in the proof region makes subsequent reads return
        // the overwritten value.
        proptest!(|(byte_before: u8,
                    bytes_after: [u8; ELEM_SIZE],
                    write_address in &address_range)| {
            let mut cells = Normal::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let dyn_region: ProofDynRegion = ProofDynRegion::bind(&cells);
            let mut dyn_cells: DynCells<Prove> = DynCells::bind(dyn_region);

            // Perform static memory accesses
            let value_before = u64::from_le_bytes([byte_before; ELEM_SIZE]);
            let value_after = u64::from_le_bytes(bytes_after);

            let value: u64 = unsafe { dyn_cells.read(write_address) };
            assert_eq!(value, value_before);
            unsafe { dyn_cells.write(write_address, value_after); }
            let value: u64 = unsafe { dyn_cells.read(write_address) };
            assert_eq!(value, value_after);

            let mut cells = Normal::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let dyn_region: ProofDynRegion = ProofDynRegion::bind(&cells);
            let mut dyn_cells: DynCells<Prove> = DynCells::bind(dyn_region);

            // Perform dynamic memory accesses as `u16`
            let value_before = [u16::from_le_bytes([byte_before; 2]); ELEM_SIZE / 2];
            let value_after = [
                u16::from_le_bytes([bytes_after[0], bytes_after[1]]),
                u16::from_le_bytes([bytes_after[2], bytes_after[3]]),
                u16::from_le_bytes([bytes_after[4], bytes_after[5]]),
                u16::from_le_bytes([bytes_after[6], bytes_after[7]]),
            ];

            let mut value = [0u16; ELEM_SIZE / 2];
            dyn_cells.read_all(write_address, &mut value);
            assert_eq!(value, value_before);
            dyn_cells.write_all(write_address, &value_after);
            dyn_cells.read_all(write_address, &mut value);
            assert_eq!(value, value_after);

            let mut cells = Normal::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let dyn_region: ProofDynRegion = ProofDynRegion::bind(&cells);
            let mut dyn_cells: DynCells<Prove> = DynCells::bind(dyn_region);

            // Perform dynamic memory accesses as bytes
            let value_before = [byte_before; ELEM_SIZE];

            let mut value = [0u8; ELEM_SIZE];
            dyn_cells.read_all(write_address, &mut value);
            assert_eq!(value, value_before);
            dyn_cells.write_all(write_address, &bytes_after);
            dyn_cells.read_all(write_address, &mut value);
            assert_eq!(value, bytes_after);
        });

        // Check correct Merkleisation of a dynamic region which was read from and written to
        proptest!(|(byte_before: u8,
                    bytes_after: [u8; ELEM_SIZE],
                    reads in array::uniform2(&address_range),
                    writes in array::uniform2(&address_range))| {
            let mut cells = Normal::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let owned_dyn_cells: DynCells<Ref<'_, Normal>> = DynCells::bind(&cells);
            let initial_root_hash = Hash::from_foldable(&owned_dyn_cells);

            let mut proof_dyn_region: ProofDynRegion = ProofDynRegion::bind(&cells);

            // Perform memory accesses
            let value_before = [byte_before; ELEM_SIZE];
            reads.iter().try_for_each(|i| {
                let mut value = [0u8; ELEM_SIZE];
                Prove::dyn_region_read_all(&proof_dyn_region, *i, &mut value);
                prop_assert_eq!(value, value_before);
                Ok::<(), proptest::test_runner::TestCaseError>(())
            })?;
            writes.iter().for_each(|i| {
                Prove::dyn_region_write_all(
                    &mut proof_dyn_region,
                    *i,
                    &bytes_after,
                );
            });

            // Build the Merkle tree and check that it has the root hash of the
            // initial wrapped region.
            let proof_dyn_cells: DynCells<Prove> = DynCells::bind(proof_dyn_region);
            let merkle_tree = MerkleTree::from_foldable(&proof_dyn_cells);
            merkle_tree.check_root_hash();
            prop_assert_eq!(merkle_tree.root_hash(), initial_root_hash);

            // Compute expected access info for each leaf, assuming that an access
            // cannot span more than 2 leaves.
            let expected_leaves = |accesses: &[usize]| {
                let mut leaves = BTreeSet::<usize>::new();
                for i in accesses {
                    leaves.insert(*i / MERKLE_LEAF_SIZE);
                    leaves.insert((i + ELEM_SIZE - 1) / MERKLE_LEAF_SIZE);
                }
                leaves
            };
            let read_leaves = expected_leaves(&reads);
            let written_leaves = expected_leaves(&writes);

            // Traverse the generated Merkle tree and check each leaf's access log
            let mut queue = VecDeque::with_capacity(LEAVES + 1);

            let pages_tree = match merkle_tree {
                MerkleTree::Leaf(_, _, _) => panic!("Did not expect leaf"),
                MerkleTree::Node(_, mut children) => {
                    // The node for the pages is the second child.
                    children.remove(1)
                },
            };
            queue.push_back(pages_tree);

            let mut leaf: usize = 0;
            while let Some(node) = queue.pop_front() {
                match node {
                    MerkleTree::Node(_, children) => queue.extend(children),
                    MerkleTree::Leaf(_, access_info, _) => {
                        prop_assert_eq!(
                            access_info,
                            read_leaves.contains(&leaf) ||
                                written_leaves.contains(&leaf)
                        );
                        leaf += 1;
                    }
                }
            }
        });
    }
}
