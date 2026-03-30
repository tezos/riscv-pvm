// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use tezos_smart_rollup_constants::riscv::SBI_FIRMWARE_TEZOS;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_KECCAK256_HASH;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_SECP256K1_VERIFY;
use tezos_smart_rollup_constants::riscv::SbiError;

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
