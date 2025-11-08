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
//! The structure of the Merkle tree is informed by the layout of the state,
//! which needs to implement [`ProofLayout`].
//!
//! [`MerkleTree`]: merkle::MerkleTree
//! [`ProofLayout`]: super::ProofLayout

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use bincode::enc::Encode;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::EncodeError;

use super::FnManager;
use super::ManagerBase;
use super::ManagerRead;
use super::ManagerReadWrite;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::state_backend::Elem;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerClone;
use crate::state_backend::elem_bytes;

pub mod merkle;
pub mod proof;
pub mod tree;

/// Proof-generating backend
pub struct ProofGen<M: ManagerBase> {
    _pd: std::marker::PhantomData<M>,
}

impl<M: ManagerBase> ManagerBase for ProofGen<M> {
    type Region<E: 'static, const LEN: usize> = ProofRegion<E, LEN, M>;

    type DynRegion = ProofDynRegion<M>;

    type ManagerRoot = Self;
}

impl<M: ManagerAlloc> ManagerAlloc for ProofGen<M> {
    fn allocate_region<E, const LEN: usize>(init_value: [E; LEN]) -> Self::Region<E, LEN> {
        ProofRegion::bind(M::allocate_region::<E, LEN>(init_value))
    }

    fn allocate_dyn_region(len: usize) -> Self::DynRegion {
        ProofDynRegion::bind(M::allocate_dyn_region(len))
    }
}

/// Implementation of [`ManagerRead`] which wraps another manager and
/// additionally records read locations.
impl<M: ManagerRead> ManagerRead for ProofGen<M> {
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
impl<M: ManagerRead> ManagerWrite for ProofGen<M> {
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

/// Implementation of [`ManagerReadWrite`] which wraps another manager and
/// additionally records read and written locations.
impl<M: ManagerRead> ManagerReadWrite for ProofGen<M> {
    fn region_replace<E: Copy, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    ) -> E {
        let old = Self::region_read(region, index);
        Self::region_write(region, index, value);
        old
    }
}

/// Implementation of [`ManagerSerialise`] which wraps another manager and
/// serialises data as recorded by the `ProofGen` backend, reconstructed
/// via variants of [`ManagerRead`] functions which do not record access
/// information.
impl<M: ManagerSerialise> ManagerSerialise for ProofGen<M> {
    fn serialise_region<T: Encode, const LEN: usize, E: Encoder>(
        region: &Self::Region<T, LEN>,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        if region.writes.is_empty() {
            // If no writes were recorded, we can serialise the underlying region as is.
            return M::serialise_region(&region.source, encoder);
        }

        // This variable keeps the index of the next item from the region that should be written.
        let mut write_index = 0;

        for (&index, value) in region.writes.iter() {
            // There are items before the current index that have not been written yet.
            if write_index < index {
                for i in write_index..index {
                    M::region_ref(&region.source, i).encode(&mut encoder)?;
                }
            }

            value.encode(&mut encoder)?;

            // Make sure we expect to write the next item after the current.
            write_index = index.saturating_add(1);
        }

        // Write the remaining items from the region that were not written yet.
        for i in write_index..LEN {
            M::region_ref(&region.source, i).encode(&mut encoder)?;
        }

        Ok(())
    }

    fn serialise_dyn_region<E: Encoder>(
        region: &Self::DynRegion,
        mut encoder: E,
    ) -> Result<(), EncodeError> {
        if region.writes.is_empty() {
            // If no writes were recorded, we can serialise the underlying dynamic region as is.
            return M::serialise_dyn_region(&region.source, encoder);
        }

        // This variable keeps the index of the next item from the region that should be written.
        let mut write_index = 0;

        for (&index, value) in region.writes.iter() {
            // There are items before the current index that have not been written yet.
            if write_index < index {
                let to_be_written = index - write_index;
                let mut buffer = vec![0u8; to_be_written];
                M::dyn_region_read_all(&region.source, write_index, &mut buffer);
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
            M::dyn_region_read_all(&region.source, write_index, &mut buffer);
            encoder.writer().write(&buffer)?;
        }

        Ok(())
    }
}

impl<M: ManagerClone> ManagerClone for ProofGen<M> {
    fn clone_region<E: 'static + Clone, const LEN: usize>(
        region: &Self::Region<E, LEN>,
    ) -> Self::Region<E, LEN> {
        region.clone()
    }

    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion {
        region.clone()
    }
}

/// Proof region which wraps a region managed by another manager.
///
/// A [`ManagerBase::Region`] is never split across multiple leaves when Merkleised.
/// An access to any part of the region is thus recorded as an access to the region as a whole.
/// The underlying region is never mutated, but all written values are recorded
/// in order to preserve the integrity of subsequent reads.
pub struct ProofRegion<E: 'static, const LEN: usize, M: ManagerBase> {
    source: M::Region<E, LEN>,
    writes: BTreeMap<usize, E>,
    access: Cell<bool>,
}

impl<M: ManagerBase, E: 'static, const LEN: usize> ProofRegion<E, LEN, M> {
    /// Bind a pre-existing region.
    pub fn bind(source: M::Region<E, LEN>) -> Self {
        Self {
            source,
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
    pub fn inner_region_ref(&self) -> &M::Region<E, LEN> {
        &self.source
    }
}

impl<M: ManagerRead, E: 'static, const LEN: usize> ProofRegion<E, LEN, M> {
    /// Version of [`ManagerRead::region_ref`] which does not record
    /// the access as a read.
    fn unrecorded_ref(&self, index: usize) -> &E {
        self.writes
            .get(&index)
            .unwrap_or_else(|| M::region_ref(&self.source, index))
    }
}

impl<E: Clone, const LEN: usize, M: ManagerClone> Clone for ProofRegion<E, LEN, M> {
    fn clone(&self) -> Self {
        Self {
            source: M::clone_region(&self.source),
            writes: self.writes.clone(),
            access: self.access.clone(),
        }
    }
}

/// Proof dynamic region which wraps a dynamic region managed by another manager.
///
/// When Merkleising a [`ManagerBase::DynRegion`], its data can be split into multiple leaves.
/// Accesses are thus recorded for each address.
/// The underlying dynamic region is never mutated, but all written bytes are
/// recorded in order to preserve the integrity of subsequent reads.
pub struct ProofDynRegion<M: ManagerBase> {
    source: M::DynRegion,
    reads: RefCell<DynAccess>,
    writes: BTreeMap<usize, u8>,

    /// Was the length of the dynamic region accessed?
    did_access_length: Cell<bool>,
}

impl<M: ManagerBase> ProofDynRegion<M> {
    /// Bind a pre-existing dynamic region.
    pub fn bind(source: M::DynRegion) -> Self {
        Self {
            source,
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

impl<M: ManagerRead> ProofDynRegion<M> {
    /// Read from the wrapped dynamic region.
    ///
    /// # Safety
    ///
    /// See [`ManagerRead::dyn_region_read`] for safety requirements.
    pub unsafe fn inner_dyn_region_read<E: Elem>(&self, address: usize) -> E {
        unsafe { M::dyn_region_read(&self.source, address) }
    }

    /// Version of [`ManagerRead::dyn_region_read`] which does not record
    /// the access as a read.
    fn unrecorded_read<E: Elem>(&self, address: usize) -> E {
        assert!(address + E::STORED_SIZE.get() <= self.unrecorded_len());

        // Read the underlying bytes of the value.
        let mut value_bytes = vec![0u8; E::STORED_SIZE.get()];
        M::dyn_region_read_all(&self.source, address, &mut value_bytes);

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
    fn unrecorded_len(&self) -> usize {
        // XXX: This implies the size can't change in a proof.
        M::dyn_region_len(&self.source)
    }
}

impl<M: ManagerClone> Clone for ProofDynRegion<M> {
    fn clone(&self) -> Self {
        Self {
            source: M::clone_dyn_region(&self.source),
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

/// Natural transformation from a manager `M` to a proof-generating manager `ProofGen<M>`
pub enum ProofWrapper {}

impl<M: ManagerBase> FnManager<M> for ProofWrapper {
    type Output = ProofGen<M>;

    fn map_region<E: 'static, const LEN: usize>(
        input: <M as ManagerBase>::Region<E, LEN>,
    ) -> <ProofGen<M> as ManagerBase>::Region<E, LEN> {
        ProofRegion::bind(input)
    }

    fn map_dyn_region(
        input: <M as ManagerBase>::DynRegion,
    ) -> <ProofGen<M> as ManagerBase>::DynRegion {
        ProofDynRegion::bind(input)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use proptest::array;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use tests::merkle::MerkleTree;

    use super::merkle::MERKLE_LEAF_SIZE;
    use super::*;
    use crate::state_backend::Cells;
    use crate::state_backend::CommitmentLayout;
    use crate::state_backend::DynArray;
    use crate::state_backend::DynCells;
    use crate::state_backend::ManagerAlloc;
    use crate::state_backend::ProofLayout;
    use crate::state_backend::Ref;
    use crate::state_backend::layout::Array;
    use crate::state_backend::owned_backend::Owned;

    const CELLS_SIZE: usize = 32;

    #[test]
    fn test_proof_gen_region() {
        proptest!(|(value_before: u64, value_after: u64, i in 0..CELLS_SIZE)| {
            // A read followed by a write
            let cells = [value_before; CELLS_SIZE];
            let region: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, ProofGen<Ref<'_, Owned>>> = Cells::bind(region);

            prop_assert!(!region.region_ref().get_access_info());
            let value = region.read(i);
            prop_assert_eq!(value, value_before);
            prop_assert!(region.region_ref().get_access_info());
            region.write(i, value_after);
            prop_assert!(region.region_ref().get_access_info());

            // A write followed by a read
            let cells = [value_before; CELLS_SIZE];
            let region: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, ProofGen<Ref<'_, Owned>>> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            region.write(i, value_after);
            prop_assert!(region.region_ref().get_access_info());
            let value = region.read(i);
            prop_assert_eq!(value, value_after);
            prop_assert!(region.region_ref().get_access_info());

            // Replace
            let cells = [value_before; CELLS_SIZE];
            let region: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, ProofGen<Ref<'_, Owned>>> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            let value = region.replace(i, value_after);
            prop_assert_eq!(value, value_before);
            prop_assert!(region.region_ref().get_access_info());

            let data_before = [value_before; CELLS_SIZE];
            let data_after = [value_after; CELLS_SIZE];

            // A read_all followed by a write_all
            let cells = data_before;
            let region: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, ProofGen<Ref<'_, Owned>>> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            let values = region.read_all();
            prop_assert_eq!(values.as_slice(), data_before);
            prop_assert!(region.region_ref().get_access_info());
            region.write_all(&data_after);
            prop_assert!(region.region_ref().get_access_info());

            // A write_all followed by a read_all
            let cells = data_before;
            let region: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> = ProofRegion::bind(&cells);
            let mut region: Cells<u64, CELLS_SIZE, ProofGen<Ref<'_, Owned>>> = Cells::bind(region);
            prop_assert!(!region.region_ref().get_access_info());
            region.write_all(&data_after);
            prop_assert!(region.region_ref().get_access_info());
            let values = region.read_all();
            prop_assert_eq!(values.as_slice(), data_after);
            prop_assert!(region.region_ref().get_access_info());

            // Check correct Merkleisation
            let cells = [value_before; CELLS_SIZE];
            let cells_owned: Cells<u64, CELLS_SIZE, Ref<'_, Owned>> = Cells::bind(&cells);
            let initial_root_hash =
                <Array<u64, CELLS_SIZE> as CommitmentLayout>::state_hash(cells_owned).unwrap();

            let mut proof_region: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> =
                ProofRegion::bind(&cells);
            ProofGen::<Ref<'_, Owned>>::region_write(&mut proof_region, i, value_after);
            let proof_cells: Cells<u64, CELLS_SIZE, Ref<'_, ProofGen<Ref<'_, Owned>>>> =
                Cells::bind(&proof_region);

            let merkle_tree =
                <Array<u64, CELLS_SIZE> as ProofLayout>::to_merkle_tree(proof_cells).unwrap();
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
            let mut cells = Owned::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let dyn_region: ProofDynRegion<Owned> = ProofDynRegion::bind(cells);
            let mut dyn_cells: DynCells<ProofGen<Owned>> = DynCells::bind(dyn_region);

            // Perform static memory accesses
            let value_before = u64::from_le_bytes([byte_before; ELEM_SIZE]);
            let value_after = u64::from_le_bytes(bytes_after);

            let value: u64 = unsafe { dyn_cells.read(write_address) };
            assert_eq!(value, value_before);
            unsafe { dyn_cells.write(write_address, value_after); }
            let value: u64 = unsafe { dyn_cells.read(write_address) };
            assert_eq!(value, value_after);

            let mut cells = Owned::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let dyn_region: ProofDynRegion<Owned> = ProofDynRegion::bind(cells);
            let mut dyn_cells: DynCells<ProofGen<Owned>> = DynCells::bind(dyn_region);

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

            let mut cells = Owned::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let dyn_region: ProofDynRegion<Owned> = ProofDynRegion::bind(cells);
            let mut dyn_cells: DynCells<ProofGen<Owned>> = DynCells::bind(dyn_region);

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
            let mut cells = Owned::allocate_dyn_region(DYN_REGION_SIZE);
            cells.fill(byte_before);
            let owned_dyn_cells: DynCells<Ref<'_, Owned>> = DynCells::bind(&cells);
            let initial_root_hash =
                <DynArray as CommitmentLayout>::state_hash(owned_dyn_cells).unwrap();

            let mut proof_dyn_region: ProofDynRegion<Ref<'_, Owned>> = ProofDynRegion::bind(&cells);

            // Perform memory accesses
            let value_before = [byte_before; ELEM_SIZE];
            reads.iter().try_for_each(|i| {
                let mut value = [0u8; ELEM_SIZE];
                ProofGen::<Ref<'_, Owned>>::dyn_region_read_all(&proof_dyn_region, *i, &mut value);
                prop_assert_eq!(value, value_before);
                Ok::<(), proptest::test_runner::TestCaseError>(())
            })?;
            writes.iter().for_each(|i| {
                ProofGen::<Ref<'_, Owned>>::dyn_region_write_all(
                    &mut proof_dyn_region,
                    *i,
                    &bytes_after,
                );
            });

            // Build the Merkle tree and check that it has the root hash of the
            // initial wrapped region.
            let proof_dyn_cells: DynCells<Ref<'_, ProofGen<Ref<'_, Owned>>>> =
                DynCells::bind(&proof_dyn_region);
            let merkle_tree =
                <DynArray as ProofLayout>::to_merkle_tree(proof_dyn_cells).unwrap();
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

    #[test]
    fn test_proof_gen_region_replace() {
        let region: ProofRegion<u64, 1, Owned> = ProofRegion::bind([0u64; 1]);
        let mut cells: Cells<u64, 1, ProofGen<Owned>> = Cells::bind(region);

        cells.write(0, 13);

        let old = cells.replace(0, 37);
        assert_eq!(old, 13);

        let value = cells.read(0);
        assert_eq!(value, 37);
    }
}
