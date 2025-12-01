// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;

use super::buddy::BuddyLayout;
use super::buddy::BuddyLayoutProxy;
use super::protection::PagePermissions;
use super::protection::PagePermissionsLayout;
use super::state::MemoryImpl;
use crate::machine_state::page_cache::state::PageCacheImpl;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::DynArray;
use crate::state_backend::DynCells;
use crate::state_backend::FnManager;
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
    BuddyLayoutProxy<PAGES>: BuddyLayout + 'static,
{
    const TOTAL_BYTES: NonZeroUsize = NonZeroUsize::new(TOTAL_BYTES)
        .expect("size of memory `TOTAL_BYTES` must be greater than zero");

    type Layout = (
        DynArray,
        PagePermissionsLayout<PAGES>,
        PagePermissionsLayout<PAGES>,
        PagePermissionsLayout<PAGES>,
        BuddyLayoutProxy<PAGES>,
    );

    type State<M: ManagerBase> =
        MemoryImpl<PAGES, TOTAL_BYTES, <BuddyLayoutProxy<PAGES> as BuddyLayout>::Buddy<M>, M>;

    type PageCache<
        CPE: crate::machine_state::page_cache::code_page_entry::CodePageEntry<Self, M>,
        M: ManagerBase,
    > = PageCacheImpl<PAGES, CPE, Self, M>;

    fn bind<M: ManagerBase>(space: AllocatedOf<Self::Layout, M>) -> Self::State<M> {
        if TOTAL_BYTES == 0 {
            panic!("Memory size must be positive");
        }

        if PAGES.checked_mul(super::PAGE_SIZE.get() as usize) != Some(TOTAL_BYTES) {
            panic!(
                "Memory size {} must be a non-overflowing multiple of the page size {}",
                TOTAL_BYTES,
                super::PAGE_SIZE
            );
        }

        MemoryImpl {
            data: space.0,
            readable_pages: PagePermissions::bind(space.1),
            writable_pages: PagePermissions::bind(space.2),
            executable_pages: PagePermissions::bind(space.3),
            allocated_pages: <BuddyLayoutProxy<PAGES> as BuddyLayout>::bind(space.4),
        }
    }

    fn struct_ref<'a, M, F>(instance: &'a Self::State<M>) -> AllocatedOf<Self::Layout, F::Output>
    where
        M: ManagerBase,
        F: FnManager<'a, M>,
    {
        (
            instance.data.struct_ref::<F>(),
            instance.readable_pages.struct_ref::<F>(),
            instance.writable_pages.struct_ref::<F>(),
            instance.executable_pages.struct_ref::<F>(),
            <BuddyLayoutProxy<PAGES> as BuddyLayout>::struct_ref::<F, M>(&instance.allocated_pages),
        )
    }

    fn state_from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self::State<Verify>> {
        let proof = proof.into_node()?;

        let (proof, data) = proof.next_branch()?;
        let (proof, readable_pages) = proof.next_branch()?;
        let (proof, writable_pages) = proof.next_branch()?;
        let (proof, executable_pages) = proof.next_branch()?;
        let (proof, allocated_pages) =
            proof.next_branch_with(<BuddyLayoutProxy<PAGES>>::buddy_from_proof)?;

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
            allocated_pages: <BuddyLayoutProxy<PAGES> as BuddyLayout>::start_proof(
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
