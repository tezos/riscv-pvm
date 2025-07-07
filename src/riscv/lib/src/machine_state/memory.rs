// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod buddy;
mod config;
mod protection;
mod state;

use std::num::NonZeroU64;

use tezos_smart_rollup_constants::riscv::SbiError;

use super::registers::XValue;
use crate::pvm::linux;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::CommitmentLayout;
use crate::state_backend::Elem;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;
use crate::state_backend::ProofLayout;
use crate::state_backend::Ref;

/// Number of bits needed so you can address every byte in a page
pub const OFFSET_BITS: u64 = 12;

/// Bit mask to keep only the page offset
pub const OFFSET_MASK: u64 = (1 << OFFSET_BITS) - 1;

/// Size of a page
pub const PAGE_SIZE: NonZeroU64 = {
    const PAGE_SIZE: u64 = 1 << OFFSET_BITS;

    // Compile-time check: Page size must be positive
    const _: () = {
        if PAGE_SIZE < 1 {
            panic!()
        }
    };

    match NonZeroU64::new(PAGE_SIZE) {
        Some(page_size) => page_size,
        None => {
            // SAFETY: The compile-time check above ensures this branch cannot be reached
            unsafe { std::hint::unreachable_unchecked() }
        }
    }
};

/// Memory address
pub type Address = XValue;

/// Lowest address
pub const FIRST_ADDRESS: Address = 0;

/// Memory access permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permissions {
    /// Readable?
    pub(crate) read: bool,

    /// Writable?
    pub(crate) write: bool,

    /// Executable?
    pub(crate) exec: bool,
}

impl Permissions {
    /// Allow nothing
    pub(crate) const NONE: Self = Self {
        read: false,
        write: false,
        exec: false,
    };

    /// Allow write
    pub(crate) const WRITE: Self = Self {
        read: false,
        write: true,
        exec: false,
    };

    /// Allow read and write
    pub(crate) const READ_WRITE: Self = Self {
        read: true,
        write: true,
        exec: false,
    };

    /// Allow everything
    #[cfg(test)]
    pub(crate) const READ_WRITE_EXEC: Self = Self {
        read: true,
        write: true,
        exec: true,
    };

    /// Do the permissions allow reading?
    pub const fn can_read(&self) -> bool {
        self.read
    }

    /// Do the permissions allow writing?
    pub const fn can_write(&self) -> bool {
        self.write
    }

    /// Do the permissions allow execution?
    pub const fn can_exec(&self) -> bool {
        self.exec
    }
}

impl TryFrom<XValue> for Permissions {
    type Error = linux::error::Error;

    fn try_from(value: XValue) -> Result<Self, Self::Error> {
        // Validate that no bits beyond the low three are set
        if value & !0b111 != 0 {
            return Err(linux::error::Error::InvalidArgument);
        }

        let read = value & 0b001 != 0;
        let write = value & 0b010 != 0;
        let exec = value & 0b100 != 0;

        Ok(Self { read, write, exec })
    }
}

/// Represents data fetched from instruction memory.
///
/// This struct encapsulates the raw instruction data and its associated metadata,
/// such as whether the instruction memory is writable. It is used to manage and
/// process instructions in the RISC-V machine state.
pub struct InstructionData<E> {
    /// The raw instruction data.
    ///
    /// This field contains the actual instruction or part of an instruction
    /// fetched from memory. The type `E` allows flexibility in representing
    /// the data, such as a half-word (`u16`) or a full word (`u32`).
    pub data: E,

    /// Indicates whether the instruction memory is writable.
    ///
    /// If `true`, the instruction memory can be modified by the program.
    /// Writable instructions are generally not ideal for caching, as any
    /// changes to the memory would require invalidating or updating the cache.
    pub writable: bool,
}

impl InstructionData<u16> {
    /// Combines two half-word pieces of instruction data into a full word.
    ///
    /// This method takes the lower and upper half-word parts of an instruction
    /// and combines them into a single 32-bit word. The `writable` flag of the
    /// resulting instruction is set to `true` if either of the input parts is
    /// writable.
    #[inline]
    pub fn combine_with_upper(self, upper: Self) -> InstructionData<u32> {
        let data = (self.data as u32) | ((upper.data as u32) << 16);

        // If either part of the instruction is writable, then we consider the whole instruction
        // writable as it may be changed by the program at any time. Writable instructions are not
        // ideal for caching as we would need to track every write to ensure correctness.
        let writable = self.writable || upper.writable;

        InstructionData { data, writable }
    }
}

/// Something went wrong when accessing the memory
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, thiserror::Error)]
#[error("Bad memory access")]
pub struct BadMemoryAccess;

impl From<BadMemoryAccess> for SbiError {
    fn from(_value: BadMemoryAccess) -> Self {
        SbiError::InvalidAddress
    }
}

/// Something went wrong when allocating memory or changing permissions
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, thiserror::Error)]
#[error("Error during memory governance")]
pub struct MemoryGovernanceError;

/// Instance of memory
pub trait Memory<M: ManagerBase>: NewState<M> + Sized {
    /// Read an element in the region. `address` is in bytes.
    fn read<E>(&self, address: Address) -> Result<E, BadMemoryAccess>
    where
        E: Elem,
        M: ManagerRead;

    /// Read an element in the region that will be used in execution. `address` is in bytes.
    fn read_exec<E>(&self, address: Address) -> Result<InstructionData<E>, BadMemoryAccess>
    where
        E: Elem,
        M: ManagerRead;

    /// Read elements from the region. `address` is in bytes.
    fn read_all<E>(&self, address: Address, values: &mut [E]) -> Result<(), BadMemoryAccess>
    where
        E: Elem,
        M: ManagerRead;

    /// Update an element in the region. `address` is in bytes.
    fn write<E>(&mut self, address: Address, value: E) -> Result<(), BadMemoryAccess>
    where
        E: Elem,
        M: ManagerReadWrite;

    /// Update multiple elements in the region. `address` is in bytes.
    fn write_all<E>(&mut self, address: Address, values: &[E]) -> Result<(), BadMemoryAccess>
    where
        E: Elem + Copy,
        M: ManagerReadWrite;

    /// Clone all memory.
    fn clone(&self) -> Self
    where
        M: ManagerClone;

    /// Zero-out all memory.
    fn reset(&mut self)
    where
        M: ManagerWrite;

    /// Protect the pages that belong to the given address range.
    fn protect_pages(
        &mut self,
        address: Address,
        length: usize,
        perms: Permissions,
    ) -> Result<(), MemoryGovernanceError>
    where
        M: ManagerWrite;

    /// Allocate pages for the given address range.
    fn allocate_pages(
        &mut self,
        address_hint: Option<Address>,
        length: usize,
        allow_replace: bool,
    ) -> Result<Address, MemoryGovernanceError>
    where
        M: ManagerReadWrite;

    /// Allocate pages for the given address range.
    fn deallocate_pages(
        &mut self,
        address: Address,
        length: usize,
    ) -> Result<(), MemoryGovernanceError>
    where
        M: ManagerReadWrite;

    /// Allocate pages for the given address range and amend the protections for them.
    fn allocate_and_protect_pages(
        &mut self,
        address_hint: Option<Address>,
        length: usize,
        perms: Permissions,
        allow_replace: bool,
    ) -> Result<Address, MemoryGovernanceError>
    where
        M: ManagerReadWrite;

    /// Free the pages in that address range and make sure the range is no longer accessible.
    fn deallocate_and_protect_pages(
        &mut self,
        address: Address,
        length: usize,
    ) -> Result<(), MemoryGovernanceError>
    where
        M: ManagerReadWrite,
    {
        self.deallocate_pages(address, length)?;
        self.protect_pages(address, length, Permissions::NONE)
    }
}

/// Memory configuration
pub trait MemoryConfig: 'static {
    /// Number of bytes in the memory
    const TOTAL_BYTES: usize;

    /// Layout for memory instance's state
    type Layout: CommitmentLayout + ProofLayout;

    /// Memory instance
    type State<M: ManagerBase>: Memory<M>;

    /// Bind the allocated regions to produce a memory instance.
    fn bind<M: ManagerBase>(space: AllocatedOf<Self::Layout, M>) -> Self::State<M>;

    /// Given a manager morphism `f : &M -> N`, return the memory instance layout's allocated
    /// structure containing the constituents of `N` that were produced from the constituents of
    /// `&M`.
    fn struct_ref<'a, M, F>(instance: &'a Self::State<M>) -> AllocatedOf<Self::Layout, F::Output>
    where
        M: ManagerBase,
        F: FnManager<Ref<'a, M>>;
}

// Re-export memory configurations
pub use config::M1G;
pub use config::M1M;
pub use config::M4G;
pub use config::M4K;
pub use config::M8K;
pub use config::M64M;
