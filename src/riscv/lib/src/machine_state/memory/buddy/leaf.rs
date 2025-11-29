// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Leaf of a tree that forms a Buddy-style memory manager

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use super::Buddy;
use super::BuddyLayout;
use crate::bits::ones;
use crate::state::NewState;
use crate::state_backend::Cell;
use crate::state_backend::FnManager;
use crate::state_backend::Layout;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerDeserialise;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerSerialise;
use crate::state_backend::ManagerWrite;

/// Layout for a leaf of a tree that forms a Buddy-style memory manager
pub struct BuddyLeafLayout<const PAGES: u64>;

impl<const PAGES: u64> Layout for BuddyLeafLayout<PAGES> {
    type Allocated<M: ManagerBase> = BuddyLeaf<PAGES, M>;
}

impl<const PAGES: u64> BuddyLayout for BuddyLeafLayout<PAGES> {
    type Buddy<M: ManagerBase> = BuddyLeaf<PAGES, M>;

    fn bind<M: ManagerBase>(space: Self::Allocated<M>) -> Self::Buddy<M> {
        space
    }

    fn struct_ref<'a, F, M: ManagerBase>(space: &'a Self::Buddy<M>) -> Self::Allocated<F::Output>
    where
        F: FnManager<'a, M>,
    {
        BuddyLeaf {
            set: space.set.struct_ref::<F>(),
        }
    }

    fn buddy_from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self::Buddy<Verify>> {
        let result = Cell::from_proof(proof, ())?;
        let result = result.map(|set| BuddyLeaf { set });
        Ok(result)
    }
}

/// Leaf of a tree that forms a Buddy-style memory manager
#[perfect_derive(PartialEq, Eq)]
pub struct BuddyLeaf<const PAGES: u64, M: ManagerBase> {
    /// Each bit of the `u64` represents a page.
    /// The least significant bit is the page with index 0.
    set: Cell<u64, M>,
}

impl<const PAGES: u64, M: ManagerBase> NewState<M> for BuddyLeaf<PAGES, M> {
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        Self { set: Cell::new() }
    }
}

impl<const PAGES: u64, M: ManagerBase> Buddy<M> for BuddyLeaf<PAGES, M> {
    const PAGES: u64 = PAGES;

    fn allocate(&mut self, pages: u64) -> Option<u64>
    where
        M: ManagerRead + ManagerWrite,
    {
        if pages == 0 || pages > Self::PAGES {
            return None;
        }

        let set = self.set.read();

        for start in 0..=(Self::PAGES - pages) {
            let mask = ones(pages) << start;

            // Since the mask projects only the bits representing the current page range, none of
            // the bits may be set. If they are, then there is an existing overlapping allocation
            // in place already.
            if (set & mask) == 0 {
                self.set.write(set | mask);
                return Some(start);
            }
        }

        None
    }

    fn allocate_fixed(&mut self, idx: u64, pages: u64, replace: bool) -> Option<()>
    where
        M: ManagerRead + ManagerWrite,
    {
        if pages == 0 || pages > Self::PAGES.saturating_sub(idx) {
            return None;
        }

        // Shortcut to avoid state reads
        if idx == 0 && pages == Self::PAGES && replace {
            self.set.write(!0);
            return Some(());
        }

        // Sequence of `pages` 1s starting at bit `idx`
        let mask = ones(pages) << idx;

        let set = self.set.read();

        if !replace && (set & mask) != 0 {
            // If none of the bits representing the to-be-mapped pages are set, then
            // `already_mapped` should be 0 after applying the mask
            return None;
        }

        self.set.write(set | mask);

        Some(())
    }

    fn deallocate(&mut self, idx: u64, pages: u64)
    where
        M: ManagerRead + ManagerWrite,
    {
        if pages == 0 || pages > Self::PAGES.saturating_sub(idx) {
            return;
        }

        // Shortcut to avoid state reads
        if idx == 0 && pages == Self::PAGES {
            self.set.write(0);
            return;
        }

        // Sequence of `pages` 1s starting at bit `idx`
        let mask = ones(pages) << idx;

        // Clear the bits representing the page range
        let set = self.set.read();
        self.set.write(set & !mask);
    }

    fn longest_free_sequence(&self) -> u64
    where
        M: ManagerRead,
    {
        let set = self.set.read();

        if set == 0 {
            return Self::PAGES;
        }

        // Find the longest sequence of 0s
        (0..Self::PAGES).fold(0, |longest_seq, shift| {
            let free_max_pages = Self::PAGES - shift;
            let free_end = (set >> shift).trailing_zeros() as u64;
            free_end.min(free_max_pages).max(longest_seq)
        })
    }

    fn count_free_start(&self) -> u64
    where
        M: ManagerRead,
    {
        Self::PAGES.min(self.set.read().trailing_zeros() as u64)
    }

    fn count_free_end(&self) -> u64
    where
        M: ManagerRead,
    {
        let leading_unused_bits = (u64::BITS as u64)
            .checked_sub(Self::PAGES)
            .expect("PAGES must not be larger than 64");
        (self.set.read().leading_zeros() as u64).saturating_sub(leading_unused_bits)
    }

    fn clone_state(&self) -> Self
    where
        M: ManagerClone,
    {
        Self {
            set: self.set.clone(),
        }
    }
}

impl<const PAGES: u64, M: ManagerSerialise> Encode for BuddyLeaf<PAGES, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.set.encode(encoder)
    }
}

impl<const PAGES: u64, M: ManagerDeserialise> Decode<()> for BuddyLeaf<PAGES, M> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let set = Decode::decode(decoder)?;
        Ok(Self { set })
    }
}

impl<const PAGES: u64, M, F> Foldable<F> for BuddyLeaf<PAGES, M>
where
    M: ManagerBase,
    F: Fold,
    Cell<u64, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.set.fold(builder)
    }
}
