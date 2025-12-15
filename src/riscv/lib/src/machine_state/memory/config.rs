// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;

use super::buddy::BuddyConfig;
use super::buddy::BuddyConfigProxy;
use super::protection::PagePermissions;
use super::state::MemoryImpl;
use crate::state::NewState;
use crate::state_backend::DynCells;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;

/// State layout for the memory component
pub struct MemoryConfig<const PAGES: usize, const TOTAL_BYTES: usize>;

impl<const PAGES: usize, const TOTAL_BYTES: usize, B, M> NewState<M>
    for MemoryImpl<PAGES, TOTAL_BYTES, B, M>
where
    B: NewState<M>,
    M: ManagerBase,
{
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        MemoryImpl {
            data: DynCells::new(TOTAL_BYTES),
            readable_pages: PagePermissions::new(),
            writable_pages: PagePermissions::new(),
            executable_pages: PagePermissions::new(),
            allocated_pages: B::new(),
        }
    }
}

impl<const PAGES: usize, const TOTAL_BYTES: usize> super::MemoryConfig
    for MemoryConfig<PAGES, TOTAL_BYTES>
where
    BuddyConfigProxy<PAGES>: BuddyConfig + 'static,
{
    const TOTAL_BYTES: NonZeroUsize = NonZeroUsize::new(TOTAL_BYTES)
        .expect("size of memory `TOTAL_BYTES` must be greater than zero");

    type State<M: ManagerBase> =
        MemoryImpl<PAGES, TOTAL_BYTES, <BuddyConfigProxy<PAGES> as BuddyConfig>::Buddy<M>, M>;

    fn state_from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self::State<Verify>> {
        let proof = proof.into_node()?;

        let (proof, data) = proof.next_branch()?;
        let (proof, readable_pages) = proof.next_branch()?;
        let (proof, writable_pages) = proof.next_branch()?;
        let (proof, executable_pages) = proof.next_branch()?;
        let (proof, allocated_pages) =
            proof.next_branch_with(<BuddyConfigProxy<PAGES>>::buddy_from_proof)?;

        proof.done(MemoryImpl {
            data,
            readable_pages,
            writable_pages,
            executable_pages,
            allocated_pages,
        })
    }

    fn start_proof(instance: &Self::State<Normal>) -> Self::State<Prove<'_>> {
        MemoryImpl {
            data: instance.data.start_proof(),
            readable_pages: instance.readable_pages.start_proof(),
            writable_pages: instance.writable_pages.start_proof(),
            executable_pages: instance.executable_pages.start_proof(),
            allocated_pages: <BuddyConfigProxy<PAGES> as BuddyConfig>::start_proof(
                &instance.allocated_pages,
            ),
        }
    }
}

/// Generates a valid memory configuration.
macro_rules! gen_memory_layout {
    ($name:ident = $size_in_g:literal GiB) => {
        pub type $name =
            MemoryConfig<{ $size_in_g * 1024 * 256 }, { $size_in_g * 1024 * 1024 * 1024 }>;
    };

    ($name:ident = $size_in_m:literal MiB) => {
        pub type $name = MemoryConfig<{ $size_in_m * 256 }, { $size_in_m * 1024 * 1024 }>;
    };

    ($name:ident = $size_in_k:literal KiB) => {
        pub type $name = MemoryConfig<{ $size_in_k / 4 }, { $size_in_k * 1024 }>;
    };
}

gen_memory_layout!(M4K = 4 KiB);
gen_memory_layout!(M8K = 8 KiB);
gen_memory_layout!(M1M = 1 MiB);
gen_memory_layout!(M64M = 64 MiB);
gen_memory_layout!(M1G = 1 GiB);
gen_memory_layout!(M4G = 4 GiB);
gen_memory_layout!(M16G = 16 GiB);
gen_memory_layout!(M32G = 32 GiB);
gen_memory_layout!(M64G = 64 GiB);
