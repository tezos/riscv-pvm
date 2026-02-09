// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Linux-style memory management
//!
//! # Address Space Layout
//!
//! For a memory config `MC`, the address space consists of the following areas:
//!
//! - `0..program_start` is inaccessible
//! - `program_start..program_end` is the program code and data area
//! - `program_end..heap_start` is the area available for the program break
//! - `heap_start..stack_guard_start` is the heap area
//! - `stack_guard_start..stack_guard_start+PAGE_SIZE` is the stack guard page
//! - `stack_guard_start+PAGE_SIZE..MC::TOTAL_BYTES` is the stack area

use std::num::NonZeroU64;
use std::num::NonZeroUsize;

use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::data_space::DataSpaceMode;
use octez_riscv_data::mode::Mode;

use super::SupervisorState;
use super::addr::PageAligned;
use super::addr::VirtAddr;
use super::error::Error;
use super::parameters::AddressHint;
use super::parameters::Backend;
use super::parameters::Flags;
use super::parameters::NoFileDescriptor;
use super::parameters::Visibility;
use super::parameters::Zero;
use crate::machine_state::MachineState;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::memory::PAGE_SIZE;
use crate::machine_state::memory::Permissions;
use crate::machine_state::page_cache::PageCache;

/// Number of pages that make up the stack
const STACK_PAGES: u64 = 0x2000;

/// Maximum stack size in bytes
pub const STACK_SIZE: u64 = PAGE_SIZE.get() * STACK_PAGES;

impl<M: Mode> SupervisorState<M> {
    /// Handle `brk` system call.
    ///
    /// We do not allow moving the program break. This system call can only be used to query the
    /// position of the program break.
    ///
    /// What does this mean for the user kernel? Musl's mallocng doesn't strictly need `brk` to
    /// work. If it detects that the program break can't be moved it will default to `mmap` to
    /// allocate smaller areas and allocator metadata.
    ///
    /// See: <https://man7.org/linux/man-pages/man2/brk.2.html>
    pub(super) fn handle_brk(&self) -> Result<u64, Error>
    where
        M: AtomMode,
    {
        // The program break may not be moved
        Ok(self.program.end.to_machine_address())
    }

    /// Handle `madvise` system call.
    ///
    /// See: <https://man7.org/linux/man-pages/man2/madvise.2.html>
    pub(super) fn handle_madvise(&mut self) -> Result<u64, Error> {
        // We don't make use of advice yet. We just return 0 to indicate success.
        Ok(0)
    }

    /// Handle `mprotect` system call.
    ///
    /// See: <https://man7.org/linux/man-pages/man2/mprotect.2.html>
    ///
    /// A length of 0 means no protections need to be changed.
    pub(super) fn handle_mprotect<MC, PC>(
        &mut self,
        state: &mut MachineState<MC, PC, M>,
        addr: PageAligned<VirtAddr>,
        length: u64,
        perms: Permissions,
    ) -> Result<u64, Error>
    where
        MC: MemoryConfig,
        PC: PageCache<MC, M>,
        M: AtomMode,
    {
        if let Some(length) = NonZeroUsize::new(length as usize) {
            let (main_memory, listener) = state.memory_with_listener();

            main_memory.protect_pages(addr.to_machine_address(), length, perms, listener)?;
        }

        // Return 0 to indicate success.
        Ok(0)
    }

    /// Handle `mmap` system call.
    ///
    /// See: <https://man7.org/linux/man-pages/man2/mmap.2.html>
    #[expect(
        clippy::too_many_arguments,
        reason = "The system call dispatch mechanism needs these arguments to exist, they can't be on a nested structure"
    )]
    pub(super) fn handle_mmap<MC, PC>(
        &mut self,
        state: &mut MachineState<MC, PC, M>,
        addr: VirtAddr,
        length: NonZeroU64,
        perms: Permissions,
        flags: Flags,
        _fd: NoFileDescriptor,
        _offset: Zero,
    ) -> Result<u64, Error>
    where
        MC: MemoryConfig,
        PC: PageCache<MC, M>,
        M: AtomMode + DataSpaceMode,
    {
        // We don't allow shared mappings
        match flags.visibility {
            Visibility::Private => {}
            Visibility::Shared => return Err(Error::NoSystemCall),
        }

        // We don't support file descriptors yet
        match flags.backend {
            Backend::None => {}
            Backend::File => return Err(Error::NoSystemCall),
        }

        // TODO: RV-561: use u64 everywhere in the PVM
        let length: NonZeroUsize = length.try_into().expect("expect length to fit into usize");
        let (main_memory, listener) = state.memory_with_listener();

        let res_addr: VirtAddr = match flags.addr_hint {
            AddressHint::Hint => {
                main_memory.allocate_and_protect_pages(None, length, perms, false, listener)?
            }

            AddressHint::Fixed { allow_replace } => {
                if !addr.is_aligned(PAGE_SIZE) {
                    return Err(Error::InvalidArgument);
                }

                main_memory.allocate_and_protect_pages(
                    Some(addr.to_machine_address()),
                    length,
                    perms,
                    allow_replace,
                    listener,
                )?
            }
        }
        .into();

        Ok(res_addr.to_machine_address())
    }

    /// Handle `munmap` system call.
    ///
    /// See: <https://man7.org/linux/man-pages/man2/mmap.2.html>
    pub(super) fn handle_munmap<MC, PC>(
        &mut self,
        state: &mut MachineState<MC, PC, M>,
        addr: u64,
        // while not explicitly required to be non-zero, this does partially match the
        // linux implementation which requires both page-aligned addresses and length > 0
        //
        // see <https://github.com/torvalds/linux/blob/50c19e20ed2ef359cf155a39c8462b0a6351b9fa/mm/vma.c#L1573>
        length: NonZeroU64,
    ) -> Result<u64, Error>
    where
        MC: MemoryConfig,
        PC: PageCache<MC, M>,
        M: AtomMode + DataSpaceMode,
    {
        // TODO: RV-561: use u64 everywhere in the PVM
        let length: NonZeroUsize = length.try_into().expect("expect length to fit into usize");
        let (main_memory, listener) = state.memory_with_listener();

        main_memory
            .deallocate_and_protect_pages(addr, length, listener)
            .map_err(|_| Error::InvalidArgument)?;

        Ok(0)
    }
}
