// SPDX-FileCopyrightText: 2025-2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Simplified [`BuddyConfig`] selection using const-generics

use octez_riscv_data::merkle_proof;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;

use super::BuddyConfig;
use super::branch_combinations::BuddyBranch1KiConfig;
use super::branch_combinations::BuddyBranch4Config;
use super::branch_combinations::BuddyBranch16Config;
use super::branch_combinations::BuddyBranch256Config;
use super::leaf::BuddyLeafConfig;
use crate::machine_state::memory::buddy::branch_combinations::BuddyBranch8Config;
use crate::machine_state::memory::buddy::branch_combinations::BuddyBranch32Config;
use crate::machine_state::memory::buddy::branch_combinations::BuddyBranch64Config;

/// Proxy for a [`BuddyConfig`] that manages the specified number of `PAGES`
pub struct BuddyConfigProxy<const PAGES: usize>;

impl<const PAGES: usize> BuddyConfig for BuddyConfigProxy<PAGES>
where
    (): BuddyConfigMatch<PAGES>,
{
    type Buddy<M: Mode> = <PickConfig<PAGES> as BuddyConfig>::Buddy<M>;

    fn start_proof(instance: &Self::Buddy<Normal>) -> Self::Buddy<Prove<'_>> {
        <PickConfig<PAGES> as BuddyConfig>::start_proof(instance)
    }

    fn buddy_from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self::Buddy<Verify>> {
        <PickConfig<PAGES> as BuddyConfig>::buddy_from_proof(proof)
    }
}

/// Picks a [`BuddyConfig`] given a number of pages
type PickConfig<const PAGES: usize, T = ()> = <T as BuddyConfigMatch<PAGES>>::AssocConfig;

/// Link between a number of pages and a specific [`BuddyConfig`]
pub trait BuddyConfigMatch<const PAGES: usize> {
    type AssocConfig: BuddyConfig;
}

impl<T> BuddyConfigMatch<1> for T {
    type AssocConfig = BuddyLeafConfig<1>;
}

impl<T> BuddyConfigMatch<2> for T {
    type AssocConfig = BuddyLeafConfig<2>;
}

impl<T> BuddyConfigMatch<64> for T {
    type AssocConfig = BuddyLeafConfig<64>;
}

impl<T> BuddyConfigMatch<256> for T {
    type AssocConfig = BuddyBranch4Config<BuddyLeafConfig<64>>;
}

impl<T> BuddyConfigMatch<1024> for T {
    type AssocConfig = BuddyBranch4Config<BuddyConfigProxy<256>>;
}

impl<T> BuddyConfigMatch<{ 16 * 1024 }> for T {
    type AssocConfig = BuddyBranch16Config<BuddyConfigProxy<1024>>;
}

impl<T> BuddyConfigMatch<{ 256 * 1024 }> for T {
    type AssocConfig = BuddyBranch256Config<BuddyConfigProxy<1024>>;
}

impl<T> BuddyConfigMatch<{ 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch1KiConfig<BuddyConfigProxy<1024>>;
}

impl<T> BuddyConfigMatch<{ 4 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch4Config<BuddyConfigProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyConfigMatch<{ 8 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch8Config<BuddyConfigProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyConfigMatch<{ 16 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch16Config<BuddyConfigProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyConfigMatch<{ 1024 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch1KiConfig<BuddyConfigProxy<{ 1024 * 1024 }>>;
}

impl<T> BuddyConfigMatch<{ 16 * 1024 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch16Config<BuddyConfigProxy<{ 1024 * 1024 * 1024 }>>;
}

impl<T> BuddyConfigMatch<{ 32 * 1024 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch32Config<BuddyConfigProxy<{ 1024 * 1024 * 1024 }>>;
}

impl<T> BuddyConfigMatch<{ 64 * 1024 * 1024 * 1024 }> for T {
    type AssocConfig = BuddyBranch64Config<BuddyConfigProxy<{ 1024 * 1024 * 1024 }>>;
}
