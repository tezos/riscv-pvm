// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use tezos_smart_rollup_constants::riscv::SBI_FIRMWARE_TEZOS;
use tezos_smart_rollup_constants::riscv::SbiError;

const MAX_DURABLE_VALUE_IO: usize = 4096;
const DURABLE_HASH_SIZE: usize = 32;

const SBI_TEZOS_DURABLE_REGISTRY_LEN: u64 = 0x1000;
const SBI_TEZOS_DURABLE_REGISTRY_RESIZE_TICK: u64 = 0x1001;
const SBI_TEZOS_DURABLE_REGISTRY_COPY_DATABASE: u64 = 0x1002;
const SBI_TEZOS_DURABLE_REGISTRY_MOVE_DATABASE: u64 = 0x1003;
const SBI_TEZOS_DURABLE_REGISTRY_CLEAR_DATABASE: u64 = 0x1004;
const SBI_TEZOS_DURABLE_DATABASE_EXISTS: u64 = 0x1010;
const SBI_TEZOS_DURABLE_DATABASE_DELETE: u64 = 0x1011;
const SBI_TEZOS_DURABLE_DATABASE_VALUE_LENGTH: u64 = 0x1012;
const SBI_TEZOS_DURABLE_DATABASE_READ: u64 = 0x1013;
const SBI_TEZOS_DURABLE_DATABASE_SET: u64 = 0x1014;
const SBI_TEZOS_DURABLE_DATABASE_WRITE: u64 = 0x1015;
const SBI_TEZOS_DURABLE_DATABASE_HASH: u64 = 0x1016;

#[derive(Debug, Clone)]
pub enum Error {
    Sbi(SbiError),
    ValueTooLarge,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sbi(error) => write!(f, "durable storage SBI call failed: {error:?}"),
            Self::ValueTooLarge => f.write_str("durable storage value exceeds 4 KiB"),
        }
    }
}

impl From<SbiError> for Error {
    fn from(value: SbiError) -> Self {
        Self::Sbi(value)
    }
}

#[inline]
unsafe fn tezos_ecall(
    a0_in: usize,
    a1_in: usize,
    a2_in: usize,
    a3_in: usize,
    a4_in: usize,
    a5_in: usize,
    function: u64,
) -> Result<usize, SbiError> {
    let result: isize;

    core::arch::asm!(
        "ecall",
        in("a0") a0_in,
        in("a1") a1_in,
        in("a2") a2_in,
        in("a3") a3_in,
        in("a4") a4_in,
        in("a5") a5_in,
        in("a6") function,
        in("a7") SBI_FIRMWARE_TEZOS,
        lateout("a0") result,
    );

    match SbiError::from_result(result) {
        Some(error) => Err(error),
        None => Ok(result as usize),
    }
}

pub fn registry_len() -> Result<usize, Error> {
    unsafe { tezos_ecall(0, 0, 0, 0, 0, 0, SBI_TEZOS_DURABLE_REGISTRY_LEN) }.map_err(Into::into)
}

pub fn registry_resize_tick(new_size: usize) -> Result<(), Error> {
    unsafe {
        tezos_ecall(
            new_size,
            0,
            0,
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_REGISTRY_RESIZE_TICK,
        )
    }
    .map(|_| ())
    .map_err(Into::into)
}

pub fn registry_copy_database(src_index: usize, dst_index: usize) -> Result<(), Error> {
    unsafe {
        tezos_ecall(
            src_index,
            dst_index,
            0,
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_REGISTRY_COPY_DATABASE,
        )
    }
    .map(|_| ())
    .map_err(Into::into)
}

pub fn registry_move_database(src_index: usize, dst_index: usize) -> Result<(), Error> {
    unsafe {
        tezos_ecall(
            src_index,
            dst_index,
            0,
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_REGISTRY_MOVE_DATABASE,
        )
    }
    .map(|_| ())
    .map_err(Into::into)
}

pub fn registry_clear_database(index: usize) -> Result<(), Error> {
    unsafe {
        tezos_ecall(
            index,
            0,
            0,
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_REGISTRY_CLEAR_DATABASE,
        )
    }
    .map(|_| ())
    .map_err(Into::into)
}

pub fn database_exists(index: usize, key: &[u8]) -> Result<bool, Error> {
    unsafe {
        tezos_ecall(
            index,
            key.as_ptr() as usize,
            key.len(),
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_DATABASE_EXISTS,
        )
    }
    .map(|result| result != 0)
    .map_err(Into::into)
}

pub fn database_delete(index: usize, key: &[u8]) -> Result<bool, Error> {
    unsafe {
        tezos_ecall(
            index,
            key.as_ptr() as usize,
            key.len(),
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_DATABASE_DELETE,
        )
    }
    .map(|result| result != 0)
    .map_err(Into::into)
}

pub fn database_value_length(index: usize, key: &[u8]) -> Result<usize, Error> {
    unsafe {
        tezos_ecall(
            index,
            key.as_ptr() as usize,
            key.len(),
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_DATABASE_VALUE_LENGTH,
        )
    }
    .map_err(Into::into)
}

pub fn database_read(
    index: usize,
    key: &[u8],
    offset: usize,
    buffer: &mut [u8],
) -> Result<usize, Error> {
    if buffer.len() > MAX_DURABLE_VALUE_IO {
        return Err(Error::ValueTooLarge);
    }

    unsafe {
        tezos_ecall(
            index,
            key.as_ptr() as usize,
            key.len(),
            offset,
            buffer.as_mut_ptr() as usize,
            buffer.len(),
            SBI_TEZOS_DURABLE_DATABASE_READ,
        )
    }
    .map_err(Into::into)
}

pub fn database_set(index: usize, key: &[u8], data: &[u8]) -> Result<(), Error> {
    if data.len() > MAX_DURABLE_VALUE_IO {
        return Err(Error::ValueTooLarge);
    }

    unsafe {
        tezos_ecall(
            index,
            key.as_ptr() as usize,
            key.len(),
            data.as_ptr() as usize,
            data.len(),
            0,
            SBI_TEZOS_DURABLE_DATABASE_SET,
        )
    }
    .map(|_| ())
    .map_err(Into::into)
}

pub fn database_write(
    index: usize,
    key: &[u8],
    offset: usize,
    data: &[u8],
) -> Result<usize, Error> {
    if data.len() > MAX_DURABLE_VALUE_IO {
        return Err(Error::ValueTooLarge);
    }

    unsafe {
        tezos_ecall(
            index,
            key.as_ptr() as usize,
            key.len(),
            offset,
            data.as_ptr() as usize,
            data.len(),
            SBI_TEZOS_DURABLE_DATABASE_WRITE,
        )
    }
    .map_err(Into::into)
}

pub fn database_hash(index: usize) -> Result<[u8; DURABLE_HASH_SIZE], Error> {
    let mut hash = [0u8; DURABLE_HASH_SIZE];
    let bytes_written = unsafe {
        tezos_ecall(
            index,
            hash.as_mut_ptr() as usize,
            0,
            0,
            0,
            0,
            SBI_TEZOS_DURABLE_DATABASE_HASH,
        )
    }
    .map_err(Error::from)?;

    if bytes_written != DURABLE_HASH_SIZE {
        return Err(Error::Sbi(SbiError::Failed));
    }

    Ok(hash)
}
