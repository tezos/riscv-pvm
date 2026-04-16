// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use tezos_smart_rollup_constants::riscv::SBI_FIRMWARE_TEZOS;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_KECCAK256_HASH;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_SECP256K1_VERIFY;
use tezos_smart_rollup_constants::riscv::SbiError;

// FIXME: these constants should live in tezos_smart_rollup_constants::riscv once moved out of tezos.rs
const SBI_TEZOS_KECCAK256_ENQUEUE: u64 = 0x0c;
const SBI_TEZOS_KECCAK256_DEQUEUE: u64 = 0x0d;

pub unsafe fn secp256k1_verify(
    public_key: &[u8; 65],
    signature: &[u8; 64],
    message_hash: &[u8; 32],
) -> bool {
    let result: isize;

    core::arch::asm!(
        "ecall",
        in("a6") SBI_TEZOS_SECP256K1_VERIFY,
        in("a7") SBI_FIRMWARE_TEZOS,
        in("a0") public_key.as_ptr(),
        in("a1") signature.as_ptr(),
        in("a2") message_hash.as_ptr(),
        lateout("a0") result,
    );

    result == 1
}

pub unsafe fn keccak256_hash(message: &[u8]) -> Result<[u8; 32], SbiError> {
    let mut out = [0u8; 32];
    let result: isize;

    core::arch::asm!(
        "ecall",
        in("a6") SBI_TEZOS_KECCAK256_HASH,
        in("a7") SBI_FIRMWARE_TEZOS,
        in("a0") out.as_mut_ptr(),
        in("a1") message.as_ptr(),
        in("a2") message.len(),
        lateout("a0") result,
    );

    match SbiError::from_result(result) {
        Some(error) => Err(error),
        None => Ok(out),
    }
}

/// Enqueue a keccak-256 hash request.
///
/// The PVM immediately returns; the hash is computed asynchronously in Normal mode.
/// Call [`keccak256_dequeue`] later to retrieve the result.
pub unsafe fn keccak256_enqueue(message: &[u8]) -> Result<(), SbiError> {
    let result: isize;

    core::arch::asm!(
        "ecall",
        in("a6") SBI_TEZOS_KECCAK256_ENQUEUE,
        in("a7") SBI_FIRMWARE_TEZOS,
        in("a0") message.as_ptr(),
        in("a1") message.len(),
        lateout("a0") result,
    );

    match SbiError::from_result(result) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Dequeue the oldest pending keccak-256 result.
///
/// In Normal mode, this blocks until the background worker returns the result.
/// In Prove/Verify mode, the hash is computed synchronously from the stored request.
///
/// Returns an error if the queue is empty.
pub unsafe fn keccak256_dequeue(out: &mut [u8; 32]) -> Result<(), SbiError> {
    let result: isize;

    core::arch::asm!(
        "ecall",
        in("a6") SBI_TEZOS_KECCAK256_DEQUEUE,
        in("a7") SBI_FIRMWARE_TEZOS,
        in("a0") out.as_mut_ptr(),
        lateout("a0") result,
    );

    match SbiError::from_result(result) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
