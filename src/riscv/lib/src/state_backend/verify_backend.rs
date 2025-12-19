// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::mode::utils::CaughtNotFoundOrPanic;
use octez_riscv_data::mode::utils::NotFound;
use octez_riscv_data::mode::utils::not_found;
use range_collections::RangeSet2;

use super::ManagerBase;
use super::ManagerClone;
use super::ManagerRead;
use super::ManagerWrite;
use crate::state_backend::Elem;
use crate::state_backend::ProofError;
use crate::state_backend::elem_bytes;
use crate::state_backend::proof_backend::merkle::MERKLE_LEAF_SIZE;

/// Error during proof verification
#[derive(Debug, thiserror::Error)]
pub enum ProofVerificationFailure {
    #[error("Deserialisation error: {0}")]
    BadDeserialisation(#[from] ProofError),

    #[error("Stepper error")]
    StepperError,

    #[error("Stepper panic")]
    StepperPanic(Box<dyn std::any::Any + Send>),

    #[error("Attempted to access absent data")]
    AbsentDataAccess(#[from] NotFound),

    #[error("Proof cannot be used for hashing after the verification step")]
    BadProofForHashing,

    #[error("Final state hash mismatch (expected {expected}, computed {computed})")]
    FinalHashMismatch { expected: Hash, computed: Hash },
}

impl From<CaughtNotFoundOrPanic> for ProofVerificationFailure {
    fn from(error: CaughtNotFoundOrPanic) -> Self {
        match error {
            CaughtNotFoundOrPanic::NotFound(not_found) => Self::AbsentDataAccess(not_found),
            CaughtNotFoundOrPanic::Other(panic_info) => Self::StepperPanic(panic_info),
        }
    }
}

impl ManagerBase for Verify {
    type DynRegion = DynRegion<{ MERKLE_LEAF_SIZE.get() }>;

    type ManagerRoot = Self;
}

#[cfg(test)]
mod test_helpers {
    use crate::state_backend::ManagerAlloc;
    use crate::state_backend::verify_backend::DynRegion;
    use crate::state_backend::verify_backend::PageId;
    use crate::state_backend::verify_backend::Verify;

    impl<const LEAF_SIZE: usize> DynRegion<LEAF_SIZE> {
        /// Construct a zero-initialized dynamic region.
        pub(crate) fn zero_initialized(length: usize) -> Self {
            let nr_pages = length.div_ceil(LEAF_SIZE);

            Self::from_pages(
                Some(length),
                (0..nr_pages).map(|page_id| {
                    let page_index = PageId::<LEAF_SIZE>::from_address(page_id * LEAF_SIZE);
                    (page_index, Box::new([0; LEAF_SIZE]))
                }),
            )
        }

        /// Like [`Self::zero_initialized`] but all pages are absent.
        pub(crate) fn absent(length: usize) -> Self {
            Self::from_pages(Some(length), std::iter::empty())
        }
    }

    impl ManagerAlloc for Verify {
        fn allocate_dyn_region(length: usize) -> Self::DynRegion {
            // Since this implementation is only for testing purposes, we can allocate the regions
            // as zero initialized to mimic what the normal mode would do (to pass tests).
            DynRegion::zero_initialized(length)
        }
    }
}

impl ManagerRead for Verify {
    fn dyn_region_len(region: &Self::DynRegion) -> usize {
        region.len()
    }

    unsafe fn dyn_region_read<E: Elem>(region: &Self::DynRegion, address: usize) -> E {
        let mut raw_data = vec![0u8; E::STORED_SIZE.get()];
        region.read_bytes(address, &mut raw_data);

        // SAFETY: The byte vector has been allocated with sufficient space.
        unsafe { E::read_unaligned(raw_data.as_ptr()) }
    }
}

impl ManagerWrite for Verify {
    unsafe fn dyn_region_write<E: Elem>(region: &mut Self::DynRegion, address: usize, value: E) {
        let raw_data = elem_bytes(value);
        region.write_bytes(address, &raw_data);
    }
}

impl ManagerClone for Verify {
    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion {
        region.clone()
    }
}

/// Represents either a present and complete region in `Verify` mode
/// or specifies whether it is only partially present or completely absent.
pub enum PartialState<T> {
    /// A region is fully present
    Complete(T),
    /// A region is absent
    Absent,
    /// A region is only partially present
    Incomplete,
}

/// Page identifier
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct PageId<const LEAF_SIZE: usize>(usize);

impl<const LEAF_SIZE: usize> PageId<LEAF_SIZE> {
    const LEAF_SIZE: usize = {
        if LEAF_SIZE.count_ones() != 1 {
            panic!("LEAF_SIZE must be a power of 2");
        }

        LEAF_SIZE
    };

    const LEAF_MASK: usize = !(Self::LEAF_SIZE - 1);

    /// Construct a page identifier from an address.
    pub fn from_address(address: usize) -> Self {
        PageId(address & Self::LEAF_MASK)
    }

    /// Calculate the offset of an address relative to the start of the identified page.
    pub fn offset(&self, address: usize) -> Option<usize> {
        address.checked_sub(self.0)
    }
}

/// Error indicating that a page ID cannot be represented
#[derive(Debug, thiserror::Error)]
#[error("Page ID overflow")]
pub struct PageIdOverflow;

/// Page of a dynamic region where sub-ranges may not be present
#[derive(Clone, Debug)]
pub struct Page<const LEAF_SIZE: usize> {
    data: Box<[u8; LEAF_SIZE]>,
    available: RangeSet2<usize>,
}

impl<const LEAF_SIZE: usize> Page<LEAF_SIZE> {
    /// Construct a page where the entire data is present.
    fn from_full(data: Box<[u8; LEAF_SIZE]>) -> Self {
        let available = RangeSet2::from(0..LEAF_SIZE);
        Page { data, available }
    }

    /// Read a sub-range of the page. Only returns `Some` if the entire range is present.
    fn get(&self, start: usize, len: usize) -> Option<&[u8]> {
        if len > LEAF_SIZE.saturating_sub(start) {
            return None;
        }

        let range = start..start.saturating_add(len);

        // Superset means that `self.available` fully covers `range`. In other words, everything in
        // `range` is also in `self.available`.
        if !self.available.is_superset(&RangeSet2::<usize>::from(range)) {
            return None;
        }

        Some(&self.data[start..][..len])
    }

    /// Write to a range in the page. This makes that range available to subsequent reads.
    fn put(&mut self, start: usize, data: &[u8]) -> bool {
        if data.len() > LEAF_SIZE.saturating_sub(start) {
            return false;
        }

        self.available.union_with(&RangeSet2::<usize>::from(
            start..data.len().saturating_add(start),
        ));

        self.data[start..][..data.len()].copy_from_slice(data);

        true
    }

    /// Returns true if every byte of the page is available.
    fn is_fully_available(&self) -> bool {
        self.available.boundaries() == [0, LEAF_SIZE]
    }
}

impl<const LEAF_SIZE: usize> Default for Page<LEAF_SIZE> {
    fn default() -> Self {
        Page {
            data: Box::new([0; LEAF_SIZE]),
            available: RangeSet2::empty(),
        }
    }
}

/// Verifier dynamic region
#[derive(Clone, Debug)]
pub struct DynRegion<const LEAF_SIZE: usize> {
    length: Option<usize>,
    pages: BTreeMap<PageId<LEAF_SIZE>, Page<LEAF_SIZE>>,
}

impl<const LEAF_SIZE: usize> DynRegion<LEAF_SIZE> {
    /// Get the length of the dynamic region.
    fn len(&self) -> usize {
        match self.length {
            Some(len) => len,
            None => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            }
        }
    }

    /// Like [`Self::len`] but returns `None` if the length is not known, instead of panicking.
    pub(crate) fn len_opt(&self) -> Option<usize> {
        self.length
    }

    /// Construct a verifier dynamic region using the given known pages.
    pub fn from_pages(
        length: Option<usize>,
        pages: impl IntoIterator<Item = (PageId<LEAF_SIZE>, Box<[u8; LEAF_SIZE]>)>,
    ) -> Self {
        let pages = pages
            .into_iter()
            .map(|(id, data)| (id, Page::from_full(data)))
            .collect();

        DynRegion { length, pages }
    }

    /// Read bytes from the dynamic region.
    pub fn read_bytes(&self, mut address: usize, mut buffer: &mut [u8]) {
        if buffer.is_empty() {
            return;
        }

        if buffer.len() > self.len().saturating_sub(address) {
            // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
            unsafe { not_found() }
        }

        while !buffer.is_empty() {
            let page_index = PageId::from_address(address);

            let Some(page) = self.pages.get(&page_index) else {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            };

            let Some(offset) = page_index.offset(address) else {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            };

            let chunk_length = buffer.len().min(LEAF_SIZE.saturating_sub(offset));

            let dst = &mut buffer[..chunk_length];
            let Some(src) = page.get(offset, chunk_length) else {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            };
            dst.copy_from_slice(src);

            address = address.saturating_add(chunk_length);
            buffer = &mut buffer[chunk_length..];
        }
    }

    /// Write bytes to the dynamic region.
    pub fn write_bytes(&mut self, mut address: usize, mut buffer: &[u8]) {
        if buffer.is_empty() {
            return;
        }

        if buffer.len() > self.len().saturating_sub(address) {
            // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
            unsafe { not_found() }
        }

        while !buffer.is_empty() {
            let page_index = PageId::from_address(address);
            let page = self.pages.entry(page_index).or_default();

            let Some(offset) = page_index.offset(address) else {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            };

            let chunk_length = buffer.len().min(LEAF_SIZE.saturating_sub(offset));

            let src = &buffer[..chunk_length];
            if !page.put(offset, src) {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode.
                unsafe { not_found() }
            };

            address = address.saturating_add(chunk_length);
            buffer = &buffer[chunk_length..];
        }
    }

    /// Get the contents of a page if it is fully present or its status otherwise.
    pub fn get_partial_page(&self, id: PageId<LEAF_SIZE>) -> PartialState<&[u8; LEAF_SIZE]> {
        match self.pages.get(&id) {
            Some(page) if page.is_fully_available() => PartialState::Complete(page.data.as_ref()),
            Some(_) => PartialState::Incomplete,
            None => PartialState::Absent,
        }
    }

    /// Check whether no pages, not even the length is available.
    ///
    /// This would be the case when the dynamic region represents an absent or blinded node from
    /// the compressed partial Merkle proof tree, and no data has been written to it.
    pub(crate) fn is_completely_absent(&self) -> bool {
        self.length.is_none() && self.pages.is_empty()
    }
}

impl<const LEAF_SIZE: usize> Default for DynRegion<LEAF_SIZE> {
    fn default() -> Self {
        DynRegion {
            length: Some(LEAF_SIZE),
            pages: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::components::atom::Atom;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::mode::utils::NotFound;
    use octez_riscv_data::mode::utils::catch_not_found;

    use super::*;
    use crate::state_backend::DynCells;

    /// Ensures that page indices are properly calculated.
    #[test]
    fn page_index() {
        let page0 = [
            PageId::<4>::from_address(0),
            PageId::<4>::from_address(1),
            PageId::<4>::from_address(2),
            PageId::<4>::from_address(3),
        ];

        let page4 = [
            PageId::<4>::from_address(4),
            PageId::<4>::from_address(5),
            PageId::<4>::from_address(6),
            PageId::<4>::from_address(7),
        ];

        let page8 = [
            PageId::<4>::from_address(8),
            PageId::<4>::from_address(9),
            PageId::<4>::from_address(10),
            PageId::<4>::from_address(11),
        ];

        page0.into_iter().fold((), |_, item| {
            assert_eq!(item, page0[0]);
            assert!(item < page4[0]);
            assert!(item < page8[0]);
        });

        page4.into_iter().fold((), |_, item| {
            assert!(item > page0[0]);
            assert_eq!(item, page4[0]);
            assert!(item < page8[0]);
        });

        page8.into_iter().fold((), |_, item| {
            assert!(item > page0[0]);
            assert!(item > page4[0]);
            assert_eq!(item, page8[0]);
        });
    }

    /// Check functionality of an Atom that is present.
    #[test]
    fn atom_present() {
        proptest::proptest!(|(reg: [u64; 32])| {
            let mut atoms: Atom<[_; 32], Verify> = Atom::new(reg);

            for i in 0..32 {
                let value = catch_not_found(|| atoms[i]).ok();
                proptest::prop_assert_eq!(value, Some(reg[i]));

                let new_value = rand::random();
                atoms[i] = new_value;

                let read_value = atoms[i];
                proptest::prop_assert_eq!(read_value, new_value);
            }
        });
    }

    /// Check functionality of an Atom that is absent.
    #[test]
    fn atom_absent() {
        let cells: Atom<[u64; 32], Verify> = Atom::absent();

        for i in 0..32 {
            let value = catch_not_found(|| cells[i]).ok();
            assert_eq!(value, None);
        }
    }

    macro_rules! assert_eq_found {
        ( $left:expr, $right:expr ) => {
            assert!(catch_not_found(|| { $left }).is_ok_and(|v| v == $right))
        };
    }

    macro_rules! assert_not_found {
        ( $body:expr ) => {{
            let result = catch_not_found(|| $body).expect_err("computation should fail");
            assert!(matches!(result, NotFound), "unexpected error: {result:?}");
        }};
    }

    /// Check the read functionality of a region that has no gaps between its pages.
    #[test]
    fn dyn_region_continuous() {
        const LEAF_SIZE: usize = MERKLE_LEAF_SIZE.get();

        let mut dyn_region = DynRegion::absent(3 * LEAF_SIZE);
        dyn_region.write_bytes(
            0,
            [1, 3, 3, 7]
                .into_iter()
                .cycle()
                .take(LEAF_SIZE)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        dyn_region.write_bytes(
            LEAF_SIZE,
            [11, 14, 14, 15]
                .into_iter()
                .cycle()
                .take(LEAF_SIZE)
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let mut dyn_cells: DynCells<Verify> = DynCells::bind(dyn_region);

        // Read things that are contained in the first leaf.
        unsafe {
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(0), [1, 3, 3, 7]);
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(1), [3, 3, 7, 1]);
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE - 4), [1, 3, 3, 7]);
        }

        // Read things that span the first and second leaf.
        unsafe {
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE - 2), [3, 7, 11, 14]);
        }

        // Read things that are contained in the second leaf.
        unsafe {
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE), [11, 14, 14, 15]);
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE + 1), [14, 14, 15, 11]);
        }

        // Read more than is available.
        unsafe {
            assert_not_found!(dyn_cells.read::<[u8; LEAF_SIZE * 3 + 1]>(0));
        }

        // Read at an offset that is out of bounds.
        unsafe {
            assert_not_found!(dyn_cells.read::<u8>(LEAF_SIZE * 2));
        }

        // Write to an index that is out of bounds.
        unsafe {
            assert_not_found!(dyn_cells.clone().write(LEAF_SIZE * 3, 0u8));
        }

        // Add more to the third leaf.
        let dyn_cells = catch_not_found(move || unsafe {
            dyn_cells.write(LEAF_SIZE * 2, [255u8, 0]);
            dyn_cells
        })
        .unwrap();
        unsafe {
            assert_eq_found!(dyn_cells.read::<[u8; 6]>(LEAF_SIZE * 2 - 4), [
                11, 14, 14, 15, 255, 0
            ]);
            assert_eq_found!(dyn_cells.read::<[u8; 2]>(LEAF_SIZE * 2), [255, 0]);
            assert_not_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE * 2));
            assert_not_found!(dyn_cells.read::<[u8; 2]>(LEAF_SIZE * 2 + 2));
        }

        // Read at an offset that is out of bounds.
        unsafe {
            assert_not_found!(dyn_cells.read::<u8>(LEAF_SIZE * 3));
        }
    }

    /// Check the functionality of a region that has gaps between its pages.
    #[test]
    fn dyn_region_gaps() {
        const LEAF_SIZE: usize = MERKLE_LEAF_SIZE.get();

        let mut dyn_region = DynRegion::absent(3 * LEAF_SIZE);
        dyn_region.write_bytes(
            0,
            [7, 3, 3]
                .into_iter()
                .cycle()
                .take(LEAF_SIZE)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        dyn_region.write_bytes(
            LEAF_SIZE * 2,
            [42, 41]
                .into_iter()
                .cycle()
                .take(LEAF_SIZE)
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let mut dyn_cells: DynCells<Verify> = DynCells::bind(dyn_region);

        unsafe {
            assert_eq_found!(dyn_cells.read::<[u8; 3]>(0), [7, 3, 3]);
            assert_eq_found!(dyn_cells.read::<[u8; 2]>(1), [3, 3]);
            assert_eq_found!(dyn_cells.read::<[u8; 1]>(LEAF_SIZE * 2), [42]);
            assert_eq_found!(dyn_cells.read::<[u8; 1]>(LEAF_SIZE * 2 + 1), [41]);
        }

        // Read a range that covers a gap.
        unsafe {
            assert_not_found!(dyn_cells.read::<[u8; LEAF_SIZE + 4]>(LEAF_SIZE - 2));
            assert_not_found!(dyn_cells.read::<[u8; LEAF_SIZE]>(LEAF_SIZE));
        }

        // Write within the gap.
        let dyn_cells = catch_not_found(move || unsafe {
            dyn_cells.write(LEAF_SIZE - 1, [1u8, 1, 3]);
            dyn_cells
        })
        .unwrap();

        unsafe {
            assert_eq_found!(dyn_cells.read::<[u8; 3]>(LEAF_SIZE - 1), [1, 1, 3]);
            assert_eq_found!(dyn_cells.read::<[u8; 2]>(LEAF_SIZE), [1, 3]);
            assert_eq_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE - 2), [3, 1, 1, 3]);
        }

        unsafe {
            assert_not_found!(dyn_cells.read::<[u8; 6]>(LEAF_SIZE - 1));
            assert_not_found!(dyn_cells.read::<[u8; 4]>(LEAF_SIZE));
        }
    }

    #[test]
    fn test_partial_hash_absent() {
        let verify_cell: Atom<u64, Verify> = Atom::absent();
        let proof = None;

        let hash = PartialHash::from_foldable(proof, &verify_cell);
        assert_eq!(hash, PartialHash::Previous);
    }

    #[test]
    fn test_partial_hash_absent_written() {
        let mut verify_cell: Atom<u64, Verify> = Atom::absent();
        let proof = None;

        let written_value = 1337;
        verify_cell.write(written_value);

        let value_hash = Hash::blake3_hash(written_value).unwrap();
        let expected_state_hash = PartialHash::Present(value_hash);
        let hash = PartialHash::from_foldable(proof, &verify_cell);
        assert_eq!(hash, expected_state_hash);
    }

    #[test]
    fn test_partial_hash_present_written() {
        let mut verify_cell: Atom<u64, Verify> = Atom::new(42);
        let proof = None;

        let written_value = 1337;
        verify_cell.write(written_value);

        let value_hash = Hash::blake3_hash(written_value).unwrap();
        let expected_state_hash = PartialHash::Present(value_hash);
        let hash = PartialHash::from_foldable(proof, &verify_cell);
        assert_eq!(hash, expected_state_hash);
    }
}
