// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Simplified [`BuddyLayout`] selection using const-generics

use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof;
use octez_riscv_data::mode::Verify;

use super::BuddyLayout;
use super::branch_combinations::BuddyBranch1KiLayout;
use super::branch_combinations::BuddyBranch4Layout;
use super::branch_combinations::BuddyBranch16Layout;
use super::branch_combinations::BuddyBranch256Layout;
use super::leaf::BuddyLeafLayout;
use crate::machine_state::memory::buddy::branch_combinations::BuddyBranch8Layout;
use crate::machine_state::memory::buddy::branch_combinations::BuddyBranch32Layout;
use crate::machine_state::memory::buddy::branch_combinations::BuddyBranch64Layout;
use crate::state_backend::FnManager;
use crate::state_backend::Layout;
use crate::state_backend::ManagerBase;
use crate::state_backend::PartialHashError;
use crate::state_backend::ProofLayout;
use crate::state_backend::ProofTree;
use crate::state_backend::RefVerifyAlloc;
use crate::state_backend::VerifyAllocResult;

/// Proxy for a [`BuddyLayout`] that manages the specified number of `PAGES`
pub struct BuddyLayoutProxy<const PAGES: usize>;

impl<const PAGES: usize> Layout for BuddyLayoutProxy<PAGES>
where
    (): BuddyLayoutMatch<PAGES>,
{
    type Allocated<M: ManagerBase> = <PickLayout<PAGES> as Layout>::Allocated<M>;
}

impl<const PAGES: usize> ProofLayout for BuddyLayoutProxy<PAGES>
where
    (): BuddyLayoutMatch<PAGES>,
{
    fn into_verify_alloc<D: merkle_proof::Deserialiser>(proof: D) -> VerifyAllocResult<D, Self> {
        <PickLayout<PAGES> as ProofLayout>::into_verify_alloc(proof)
    }

    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        <PickLayout<PAGES> as ProofLayout>::partial_state_hash(state, proof)
    }
}

impl<const PAGES: usize> BuddyLayout for BuddyLayoutProxy<PAGES>
where
    (): BuddyLayoutMatch<PAGES>,
{
    type Buddy<M: ManagerBase> = <PickLayout<PAGES> as BuddyLayout>::Buddy<M>;

    fn bind<M: ManagerBase>(space: Self::Allocated<M>) -> Self::Buddy<M> {
        <PickLayout<PAGES> as BuddyLayout>::bind(space)
    }

    fn struct_ref<'a, F, M: ManagerBase + 'a>(
        space: &'a Self::Buddy<M>,
    ) -> Self::Allocated<F::Output>
    where
        F: FnManager<'a, M>,
    {
        <PickLayout<PAGES> as BuddyLayout>::struct_ref::<F, M>(space)
    }

    fn buddy_from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self::Buddy<Verify>> {
        <PickLayout<PAGES> as BuddyLayout>::buddy_from_proof(proof)
    }
}

/// Picks a [`BuddyLayout`] given a number of pages
type PickLayout<const PAGES: usize, T = ()> = <T as BuddyLayoutMatch<PAGES>>::AssocLayout;

/// Link between a number of pages and a specific [`BuddyLayout`]
pub trait BuddyLayoutMatch<const PAGES: usize> {
    type AssocLayout: BuddyLayout;
}

impl<T> BuddyLayoutMatch<1> for T {
    type AssocLayout = BuddyLeafLayout<1>;
}

impl<T> BuddyLayoutMatch<2> for T {
    type AssocLayout = BuddyLeafLayout<2>;
}

impl<T> BuddyLayoutMatch<64> for T {
    type AssocLayout = BuddyLeafLayout<64>;
}

impl<T> BuddyLayoutMatch<256> for T {
    type AssocLayout = BuddyBranch4Layout<BuddyLeafLayout<64>>;
}

impl<T> BuddyLayoutMatch<1024> for T {
    type AssocLayout = BuddyBranch4Layout<BuddyLayoutProxy<256>>;
}

impl<T> BuddyLayoutMatch<{ 16 * 1024 }> for T {
    type AssocLayout = BuddyBranch16Layout<BuddyLayoutProxy<1024>>;
}

impl<T> BuddyLayoutMatch<{ 256 * 1024 }> for T {
    type AssocLayout = BuddyBranch256Layout<BuddyLayoutProxy<1024>>;
}

impl<T> BuddyLayoutMatch<{ 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch1KiLayout<BuddyLayoutProxy<1024>>;
}

impl<T> BuddyLayoutMatch<{ 4 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch4Layout<BuddyLayoutProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyLayoutMatch<{ 8 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch8Layout<BuddyLayoutProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyLayoutMatch<{ 16 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch16Layout<BuddyLayoutProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyLayoutMatch<{ 1024 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch1KiLayout<BuddyLayoutProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyLayoutMatch<{ 16 * 1024 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch16Layout<BuddyLayoutProxy<{ 1024 * 1024 * 1024 }>>;
}

impl<T> BuddyLayoutMatch<{ 32 * 1024 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch32Layout<BuddyLayoutProxy<{ 1024 * 1024 * 1024 }>>;
}

impl<T> BuddyLayoutMatch<{ 64 * 1024 * 1024 * 1024 }> for T {
    type AssocLayout = BuddyBranch64Layout<BuddyLayoutProxy<{ 1024 * 1024 * 1024 }>>;
}
