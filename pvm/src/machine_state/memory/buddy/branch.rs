// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Branch of a tree that forms a Buddy-style memory manager

use std::marker::PhantomData;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use super::Buddy;
use super::BuddyConfig;

/// Information about what is free in each buddy
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct FreeInfo {
    /// Length of the longest sequence of free pages in the left buddy
    left_longest_free_sequence: u64,

    /// Number of free pages at the start of the left buddy
    left_free_start: u64,

    /// Number of free pages at the end of the left buddy
    left_free_end: u64,

    /// Length of the longest sequence of free pages in the right buddy
    right_longest_free_sequence: u64,

    /// Number of free pages at the start of the right buddy
    right_free_start: u64,

    /// Number of free pages at the end of the right buddy
    right_free_end: u64,
}

/// Config for a branch in a Buddy-style memory manager tree with 2 children
pub struct BuddyBranch2Config<B>(PhantomData<B>);

impl<B: BuddyConfig> BuddyConfig for BuddyBranch2Config<B> {
    type Buddy<M: Mode> = BuddyBranch2<B::Buddy<M>, M>;

    fn start_proof(instance: &Self::Buddy<Normal>) -> Self::Buddy<Prove<'_>> {
        BuddyBranch2 {
            free_info: instance.free_info.start_proof(),
            left: Box::new(B::start_proof(instance.left.as_ref())),
            right: Box::new(B::start_proof(instance.right.as_ref())),
        }
    }

    fn buddy_from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self::Buddy<Verify>> {
        let proof = proof.into_node()?;

        let (proof, free_info) = proof.next_branch()?;
        let (proof, left) = proof.next_branch_with(B::buddy_from_proof)?;
        let (proof, right) = proof.next_branch_with(B::buddy_from_proof)?;

        proof.done(Self::Buddy {
            free_info,
            left: Box::new(left),
            right: Box::new(right),
        })
    }
}

/// Branch in a Buddy-style memory manager tree
#[perfect_derive(PartialEq, Eq)]
pub struct BuddyBranch2<B, M: Mode> {
    free_info: Atom<FreeInfo, M>,
    left: Box<B>,
    right: Box<B>,
}

impl<B: Buddy<M>, M: Mode> BuddyBranch2<B, M> {
    fn refresh(&mut self)
    where
        M: AtomMode,
    {
        self.free_info.write(FreeInfo {
            left_longest_free_sequence: self.left.longest_free_sequence(),
            left_free_start: self.left.count_free_start(),
            left_free_end: self.left.count_free_end(),
            right_longest_free_sequence: self.right.longest_free_sequence(),
            right_free_start: self.right.count_free_start(),
            right_free_end: self.right.count_free_end(),
        });
    }
}

impl<B, M> Buddy<M> for BuddyBranch2<B, M>
where
    B: Buddy<M>,
    M: Mode,
{
    const PAGES: u64 = B::PAGES * 2;

    fn default() -> Self
    where
        M: AtomMode,
    {
        Self {
            free_info: Atom::new(FreeInfo {
                left_longest_free_sequence: B::PAGES,
                left_free_start: B::PAGES,
                left_free_end: B::PAGES,
                right_longest_free_sequence: B::PAGES,
                right_free_start: B::PAGES,
                right_free_end: B::PAGES,
            }),
            left: Box::new(B::default()),
            right: Box::new(B::default()),
        }
    }

    fn allocate(&mut self, pages: u64) -> Option<u64>
    where
        M: AtomMode,
    {
        if !(1..=Self::PAGES).contains(&pages) {
            return None;
        }

        // The requested range can be allocated entirely in the left buddy
        if pages <= self.free_info.left_longest_free_sequence {
            let idx = self.left.allocate(pages)?;
            self.refresh();
            return Some(idx);
        }

        let left_free_end = self.free_info.left_free_end;
        let right_free_start = self.free_info.right_free_start;
        let overlap = left_free_end.saturating_add(right_free_start);

        // There might be space that covers the end of the left buddy and the start of the right
        // buddy. We may use this space as it is continuous.
        if pages <= overlap && left_free_end > 0 {
            let idx = B::PAGES.checked_sub(left_free_end)?;
            self.left.allocate_fixed(idx, left_free_end, true)?;

            let right_pages = pages.saturating_sub(left_free_end);

            // Allocate the right buddy. Be aware, if that fails, we need to back out of the left
            // buddy allocation.
            if right_pages > 0 && self.right.allocate_fixed(0, right_pages, true).is_none() {
                self.left.deallocate(idx, left_free_end);
                return None;
            }

            self.refresh();
            return Some(idx);
        }

        // Try allocating in the right buddy if there's enough space
        if pages <= self.free_info.right_longest_free_sequence {
            let idx = self.right.allocate(pages)?;
            self.refresh();

            // Return adjusted index to reflect position in the combined buddy
            return Some(idx + B::PAGES);
        }

        None
    }

    fn allocate_fixed(&mut self, idx: u64, pages: u64, replace: bool) -> Option<()>
    where
        M: AtomMode,
    {
        if pages == 0 || pages > Self::PAGES.saturating_sub(idx) {
            return None;
        }

        let left_pages = B::PAGES.saturating_sub(idx).min(pages);
        let right_pages = pages.saturating_sub(left_pages);

        // The range covers the left buddy
        if left_pages > 0 {
            self.left.allocate_fixed(idx, left_pages, replace)?;
        }

        // The range covers the right buddy
        if right_pages > 0 {
            let right_idx = idx.saturating_sub(B::PAGES);
            let right_res = self.right.allocate_fixed(right_idx, right_pages, replace);

            // If the right allocation failed, we might need to do some clean up on the left buddy
            if right_res.is_none() {
                if left_pages > 0 {
                    self.left.deallocate(idx, left_pages);
                }

                return None;
            }
        }

        // Need to refresh the free counters
        self.refresh();

        Some(())
    }

    fn deallocate(&mut self, idx: u64, mut pages: u64)
    where
        M: AtomMode,
    {
        // Defer to only the right buddy if the range does not cover the left side
        if idx >= B::PAGES {
            self.right.deallocate(idx - B::PAGES, pages);
        } else {
            let end = pages.saturating_add(idx);

            // If the range crosses from left to right buddy
            if end > B::PAGES {
                let overhang = end.saturating_sub(B::PAGES);
                self.right.deallocate(0, overhang);
                pages = pages.saturating_sub(overhang);
            }

            self.left.deallocate(idx, pages);
        }

        // Need to refresh the free counters
        self.refresh();
    }

    fn longest_free_sequence(&self) -> u64
    where
        M: AtomMode,
    {
        self.free_info
            .left_free_end
            .saturating_add(self.free_info.right_free_start)
            .max(self.free_info.left_longest_free_sequence)
            .max(self.free_info.right_longest_free_sequence)
    }

    fn count_free_start(&self) -> u64
    where
        M: AtomMode,
    {
        let free_start = self.free_info.left_free_start;

        if free_start < B::PAGES {
            return free_start;
        }

        self.free_info.right_free_start.saturating_add(B::PAGES)
    }

    fn count_free_end(&self) -> u64
    where
        M: AtomMode,
    {
        let free_end = self.free_info.right_free_end;

        if free_end < B::PAGES {
            return free_end;
        }

        self.free_info.left_free_end.saturating_add(B::PAGES)
    }

    fn clone_state(&self) -> Self
    where
        M: CloneAtomMode,
    {
        Self {
            free_info: self.free_info.clone(),
            left: Box::new(self.left.clone_state()),
            right: Box::new(self.right.clone_state()),
        }
    }

    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError>
    where
        M: EncodeAtomMode,
    {
        Encode::encode(self, encoder)
    }
}

impl<B: Buddy<M>, M: EncodeAtomMode> Encode for BuddyBranch2<B, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.free_info.encode(encoder)?;
        self.left.encode(encoder)?;
        self.right.encode(encoder)?;
        Ok(())
    }
}

impl<C, B: Decode<C>> Decode<C> for BuddyBranch2<B, Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            free_info: Decode::decode(decoder)?,
            left: Box::new(Decode::decode(decoder)?),
            right: Box::new(Decode::decode(decoder)?),
        })
    }
}

impl<B, M, F> Foldable<F> for BuddyBranch2<B, M>
where
    B: Foldable<F>,
    M: Mode,
    F: Fold,
    Atom<FreeInfo, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.free_info);
        builder.add(self.left.as_ref());
        builder.add(self.right.as_ref());
        builder.done()
    }
}
