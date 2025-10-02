// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroU64;

use super::Address;
use super::BadMemoryAccess;
use super::Memory;
use super::PAGE_SIZE;
use super::Permissions;
use super::buddy::Buddy;
use super::protection::PagePermissions;
use crate::num::NonZeroLength;
use crate::state_backend::DynCells;
use crate::state_backend::Elem;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;

/// Machine's memory
pub struct MemoryImpl<const PAGES: usize, const TOTAL_BYTES: usize, B, M: ManagerBase> {
    /// Memory contents
    pub(super) data: DynCells<TOTAL_BYTES, M>,

    /// Read permissions per page
    pub(super) readable_pages: PagePermissions<PAGES, M>,

    /// Write permissions per page
    pub(super) writable_pages: PagePermissions<PAGES, M>,

    /// Execute permissions per page
    pub(super) executable_pages: PagePermissions<PAGES, M>,

    /// Allocation tracker
    pub(super) allocated_pages: B,
}

impl<const PAGES: usize, const TOTAL_BYTES: usize, B, M: ManagerBase>
    MemoryImpl<PAGES, TOTAL_BYTES, B, M>
{
    const TOTAL_BYTES: NonZeroU64 = NonZeroU64::new(TOTAL_BYTES as u64)
        .expect("memory size `TOTAL_BYTES` must be greater than zero");

    /// Ensure the access is within bounds.
    #[inline]
    fn check_bounds<E>(address: Address, length: NonZeroLength, error: E) -> Result<(), E> {
        if length.get() > Self::TOTAL_BYTES.get().saturating_sub(address) {
            return Err(error);
        }

        Ok(())
    }

    /// Mark the whole memory as readable and writeable
    #[cfg(test)]
    pub(crate) fn set_all_readable_writeable(&mut self)
    where
        B: Buddy<M>,
        M: ManagerReadWrite,
    {
        self.protect_pages(0, Self::TOTAL_BYTES, Permissions::READ_WRITE)
            .unwrap();
    }

    /// Update an element in the region without checking memory protections. `address` is in bytes.
    #[cfg(test)]
    pub(crate) fn write_instruction_unchecked<E>(
        &mut self,
        address: Address,
        value: E,
    ) -> Result<(), BadMemoryAccess>
    where
        E: Elem,
        M: ManagerWrite,
    {
        let length = E::STORED_SIZE;

        Self::check_bounds(address, length, BadMemoryAccess)?;

        self.data.write(address as usize, value);
        self.readable_pages.modify_access(address, length, true);
        self.executable_pages.modify_access(address, length, true);
        Ok(())
    }
}

impl<const PAGES: usize, const TOTAL_BYTES: usize, B, M> Memory<M>
    for MemoryImpl<PAGES, TOTAL_BYTES, B, M>
where
    B: Buddy<M>,
    M: ManagerBase,
{
    #[inline]
    fn read<E>(&self, address: Address) -> Result<E, BadMemoryAccess>
    where
        E: Elem,
        M: ManagerRead,
    {
        let length = E::STORED_SIZE;

        Self::check_bounds(address, length, BadMemoryAccess)?;

        // SAFETY: The bounds check above ensures the access check below is safe
        unsafe {
            if !self.readable_pages.can_access_narrow::<E>(address) {
                return Err(BadMemoryAccess);
            }
        }

        Ok(self.data.read(address as usize))
    }

    #[inline]
    fn read_exec<E>(&self, address: Address) -> Result<super::InstructionData<E>, BadMemoryAccess>
    where
        E: Elem,
        M: ManagerRead,
    {
        let length = E::STORED_SIZE;

        Self::check_bounds(address, length, BadMemoryAccess)?;

        // SAFETY: The bounds check above ensures the access check below is safe
        unsafe {
            // Checking for executable access is sufficient as that implies read access
            if !self.executable_pages.can_access_narrow::<E>(address) {
                return Err(BadMemoryAccess);
            }
        }

        let data = self.data.read(address as usize);

        // SAFETY: The bounds check above ensures the access check below is safe
        let writable = unsafe { self.writable_pages.can_access_narrow::<E>(address) };

        Ok(super::InstructionData { data, writable })
    }

    fn read_all<E>(&self, address: Address, values: &mut [E]) -> Result<(), BadMemoryAccess>
    where
        E: Elem,
        M: ManagerRead,
    {
        let Some(length) =
            NonZeroU64::new(E::STORED_SIZE.get().saturating_mul(values.len() as u64))
        else {
            // nothing to read
            return Ok(());
        };

        Self::check_bounds(address, NonZeroLength::wrap(length), BadMemoryAccess)?;

        // SAFETY: The bounds check above ensures the access check below is safe
        unsafe {
            if !self.readable_pages.can_access(address, length) {
                return Err(BadMemoryAccess);
            }
        }

        self.data.read_all(address as usize, values);
        Ok(())
    }

    #[inline]
    fn write<E>(&mut self, address: Address, value: E) -> Result<(), BadMemoryAccess>
    where
        E: Elem,
        M: ManagerReadWrite,
    {
        let length = E::STORED_SIZE;

        Self::check_bounds(address, length, BadMemoryAccess)?;

        // SAFETY: The bounds check above ensures the access check below is safe
        unsafe {
            if !self.writable_pages.can_access_narrow::<E>(address) {
                return Err(BadMemoryAccess);
            }
        }

        self.data.write(address as usize, value);
        Ok(())
    }

    fn write_all<E>(&mut self, address: Address, values: &[E]) -> Result<(), BadMemoryAccess>
    where
        E: Elem + Copy,
        M: ManagerReadWrite,
    {
        let Some(length) =
            NonZeroU64::new(E::STORED_SIZE.get().saturating_mul(values.len() as u64))
        else {
            // nothing to write
            return Ok(());
        };

        Self::check_bounds(address, NonZeroLength::wrap(length), BadMemoryAccess)?;

        // SAFETY: The bounds check above ensures the access check below is safe
        unsafe {
            if !self.writable_pages.can_access(address, length) {
                return Err(BadMemoryAccess);
            }
        }

        self.data.write_all(address as usize, values);
        Ok(())
    }

    fn clone(&self) -> Self
    where
        M: ManagerClone,
    {
        Self {
            data: self.data.clone(),
            readable_pages: self.readable_pages.clone(),
            writable_pages: self.writable_pages.clone(),
            executable_pages: self.executable_pages.clone(),
            allocated_pages: self.allocated_pages.clone(),
        }
    }

    fn reset(&mut self)
    where
        M: ManagerWrite,
    {
        const SIZE_OF_U64: usize = u64::STORED_SIZE.get() as usize;

        let mut address = 0;
        let mut outstanding = TOTAL_BYTES;

        // Write 64-bit chunks
        while outstanding >= SIZE_OF_U64 {
            self.data.write(address, 0u64);
            address += SIZE_OF_U64;
            outstanding -= SIZE_OF_U64;
        }

        // Write remaining bytes
        for i in 0..outstanding {
            self.data.write(address.saturating_add(i), 0u8);
        }

        self.readable_pages.reset();
        self.writable_pages.reset();
        self.executable_pages.reset();
    }

    fn protect_pages(
        &mut self,
        address: Address,
        length: NonZeroU64,
        perms: Permissions,
    ) -> Result<(), super::MemoryGovernanceError>
    where
        M: ManagerWrite,
    {
        Self::check_bounds(
            address,
            NonZeroLength::wrap(length),
            super::MemoryGovernanceError,
        )?;

        self.readable_pages
            .modify_access(address, length, perms.can_read());
        self.writable_pages
            .modify_access(address, length, perms.can_write());
        self.executable_pages
            .modify_access(address, length, perms.can_exec());

        Ok(())
    }

    fn deallocate_pages(
        &mut self,
        address: Address,
        length: NonZeroU64,
    ) -> Result<(), super::MemoryGovernanceError>
    where
        M: ManagerReadWrite,
    {
        Self::check_bounds(
            address,
            NonZeroLength::wrap(length),
            super::MemoryGovernanceError,
        )?;

        // TODO: RV-799: use `NonZeroU64::div_ceil` once stabilised.
        let pages = length.get().div_ceil(super::PAGE_SIZE.get());

        // Buddy memory manager works on page indices, not addresses
        let idx = address >> super::OFFSET_BITS.get();
        self.allocated_pages.deallocate(idx, pages);

        Ok(())
    }

    fn allocate_pages(
        &mut self,
        address_hint: Option<Address>,
        length: NonZeroU64,
        allow_replace: bool,
    ) -> Result<Address, super::MemoryGovernanceError>
    where
        M: ManagerReadWrite,
    {
        // TODO: RV-799: use `NonZeroU64::div_ceil` once stabilised.
        let pages = length.get().div_ceil(super::PAGE_SIZE.get());

        match address_hint {
            // Caller wants to allocate at a specific address
            Some(address) => {
                Self::check_bounds(
                    address,
                    NonZeroLength::wrap(length),
                    super::MemoryGovernanceError,
                )?;

                // Buddy memory manager works on page indices, not addresses
                let idx = address >> super::OFFSET_BITS.get();
                self.allocated_pages
                    .allocate_fixed(idx, pages, allow_replace)
                    .map(|()| address)
            }

            // Allocate anywhere
            None => self.allocated_pages.allocate(pages).map(|idx| {
                // Convert page index to address
                idx << super::OFFSET_BITS.get()
            }),
        }
        .ok_or(super::MemoryGovernanceError)
    }

    fn allocate_and_protect_pages(
        &mut self,
        address_hint: Option<Address>,
        length: NonZeroU64,
        perms: Permissions,
        allow_replace: bool,
    ) -> Result<Address, super::MemoryGovernanceError>
    where
        M: ManagerReadWrite,
    {
        // Mark the page range as occupied
        let address = self.allocate_pages(address_hint, length, allow_replace)?;

        // Configure the permissions on the page range
        if self.protect_pages(address, length, perms).is_err() {
            self.deallocate_pages(address, length)?;
        }

        // Zero initialise in 8-byte chunks. Using larger writes first, means we do fewer writes
        // altogether. This speeds things up.
        // As we allocate in multiples of pages, we must also clear in multiples of pages.
        //
        // TODO: RV-799: use `NonZeroU64::div_ceil` once stabilised.
        let mut remaining = length
            .get()
            .div_ceil(PAGE_SIZE.get())
            .saturating_mul(PAGE_SIZE.get());

        while remaining >= 8 {
            remaining -= 8;
            let address = address.saturating_add(remaining);
            self.data.write(address as usize, 0u64);
        }

        // Zero initialise the tail byte by byte
        for i in 0..remaining {
            let address = address.saturating_add(i);
            self.data.write(address as usize, 0u8);
        }

        Ok(address)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::backend_test;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::memory::MemoryConfig;
    use crate::state::NewState;
    use crate::state_backend::FnManagerIdent;
    use crate::state_backend::owned_backend::Owned;

    #[test]
    fn bounds_check() {
        type OwnedM4K = <M4K as MemoryConfig>::State<Owned>;

        // Bounds checks
        assert!(OwnedM4K::check_bounds(4095, NonZeroU64::new(1).unwrap(), ()).is_ok());
        assert!(OwnedM4K::check_bounds(4096, NonZeroU64::new(1).unwrap(), ()).is_err());
        assert!(OwnedM4K::check_bounds(2 * 4096, NonZeroU64::new(1).unwrap(), ()).is_err());
    }

    // ensure read/write of empty arrays is ok, even if permissions
    // are not set
    backend_test!(test_read_write_all_empty_ok, F, {
        let mut memory = <<M4K as MemoryConfig>::State<F>>::new();

        let res = memory.write_all::<u8>(0, &[]);
        assert_eq!(Ok(()), res);

        let res = memory.read_all::<u8>(0, &mut []);
        assert_eq!(Ok(()), res);

        // double check permissions are in fact, not set
        let res = memory.write_all::<u8>(0, &[1]);
        assert!(res.is_err());

        let buff = &mut [1];
        let res = memory.read_all::<u8>(0, buff);
        assert!(res.is_err());
        assert_eq!(&[1], buff);
    });

    // This test verifies that memory is fully zeroed up to the page boundary, not just the
    // requested length, when allocating memory.
    backend_test!(test_memory_fully_zeroed_on_allocation, F, {
        use crate::machine_state::memory::PAGE_SIZE;
        use crate::machine_state::memory::Permissions;

        let mut memory = <<M4K as MemoryConfig>::State<F>>::new();

        // Write a pattern to ensure memory contains non-zero values
        for i in 0..PAGE_SIZE.get() {
            memory.data.write(i as usize, 0xFFu8);
        }

        // Request size that's not a multiple of page size
        let requested_size = NonZeroU64::new(PAGE_SIZE.get() - 100).unwrap();
        let address = memory
            .allocate_and_protect_pages(None, requested_size, Permissions::READ_WRITE, false)
            .expect("Memory allocation should succeed");

        // Verify that memory is zeroed for the entire page, not just the requested length
        for i in 0..PAGE_SIZE.get() {
            let offset = i as usize;
            let value = memory.data.read::<u8>((address as usize) + offset);
            assert_eq!(
                value,
                0,
                "Memory at offset {} (address: {:#x}) should be zero, found {:#x}",
                offset,
                address + i,
                value
            );
        }
    });

    backend_test!(test_endianess, F, {
        let mut memory = <<M4K as MemoryConfig>::State<F>>::new();

        memory
            .write_instruction_unchecked(0, 0x1122334455667788u64)
            .unwrap();

        macro_rules! check_address {
            ($ty:ty, $addr:expr, $value:expr) => {
                assert_eq!(memory.read::<$ty>($addr), Ok($value));
            };
        }

        check_address!(u64, 0, 0x1122334455667788);

        check_address!(u32, 0, 0x55667788);
        check_address!(u32, 4, 0x11223344);

        check_address!(u16, 0, 0x7788);
        check_address!(u16, 2, 0x5566);
        check_address!(u16, 4, 0x3344);
        check_address!(u16, 6, 0x1122);

        check_address!(u8, 0, 0x88);
        check_address!(u8, 1, 0x77);
        check_address!(u8, 2, 0x66);
        check_address!(u8, 3, 0x55);
        check_address!(u8, 4, 0x44);
        check_address!(u8, 5, 0x33);
        check_address!(u8, 6, 0x22);
        check_address!(u8, 7, 0x11);
    });

    backend_test!(test_memory_reset, F, {
        let clean_memory = <<M4K as MemoryConfig>::State<F>>::new();

        let mut memory = <<M4K as MemoryConfig>::State<F>>::new();

        // setting readable permissions should reset
        memory.set_all_readable_writeable();
        memory.write(5, 0xFFu8).unwrap();
        memory.reset();

        assert!(
            M4K::struct_ref::<F, FnManagerIdent>(&clean_memory)
                == M4K::struct_ref::<F, FnManagerIdent>(&memory),
            "RW memory did not reset correctly"
        );

        // setting executable permissions should reset
        memory
            .write_instruction_unchecked(17, 0x1122334455667788u64)
            .unwrap();
        memory.reset();

        assert!(
            M4K::struct_ref::<F, FnManagerIdent>(&clean_memory)
                == M4K::struct_ref::<F, FnManagerIdent>(&memory),
            "X memory did not reset correctly"
        );
    });
}
