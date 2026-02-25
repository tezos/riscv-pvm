// SPDX-FileCopyrightText: 2023 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

#![no_std]
#![no_main]

use tezos_smart_rollup_constants::core::MAX_INPUT_MESSAGE_SIZE;
use tezos_smart_rollup_constants::core::PREIMAGE_HASH_SIZE;
use tezos_smart_rollup_constants::riscv::SBI_FIRMWARE_TEZOS;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_INBOX_NEXT;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_REVEAL;

/// Entrypoint of the dummy-no-std
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    debug_out(b"hello world\n");

    let mut msg_level: i32 = 0;
    let mut msg_id: i32 = 0;
    let mut buffer = [0u8; MAX_INPUT_MESSAGE_SIZE];

    debug_out(b"revealing...\n");
    let revealed_len = do_reveal(&mut buffer);
    debug_out(&buffer[..revealed_len]);

    loop {
        debug_out(b"(read input) ");
        let read = read_input(&mut msg_level, &mut msg_id, &mut buffer);
        log_input_as_hex(msg_level, msg_id, &buffer[..read]);
    }
}

fn do_reveal(buf: &mut [u8; MAX_INPUT_MESSAGE_SIZE]) -> usize {
    let mut reveal_request = [0u8; PREIMAGE_HASH_SIZE + 1];
    let _ = hex::decode_to_slice(
        "00252c5b35ebf535d729b5a4a7cfde85c08b43d50e26d60a38ac2a3569c30896c9",
        &mut reveal_request[1..],
    );

    let result: isize;
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") reveal_request.as_ptr(),
            in("a1") reveal_request.len(),
            in("a2") buf.as_mut_ptr(),
            in("a3") buf.len(),
            in("a6") SBI_TEZOS_REVEAL,
            in("a7") SBI_FIRMWARE_TEZOS,
            lateout("a0") result,
        );
    }

    // valid request
    result as usize
}

fn read_input(level: &mut i32, id: &mut i32, buf: &mut [u8; MAX_INPUT_MESSAGE_SIZE]) -> usize {
    let result: isize;

    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") buf.as_mut_ptr(),
            in("a1") buf.len(),
            in("a2") level,
            in("a3") id,
            in("a6") SBI_TEZOS_INBOX_NEXT,
            in("a7") SBI_FIRMWARE_TEZOS,
            lateout("a0") result,
        );
    }

    // valid request
    result as usize
}

fn log_input_as_hex(level: i32, msg_index: i32, input: &[u8]) {
    const HEX_MAX_BYTES_ENCODED: usize = 20;
    const HEX_BUFF_SIZE: usize = 2 * HEX_MAX_BYTES_ENCODED;
    const I32_SIZE: usize = core::mem::size_of::<i32>();
    const HEX_I32_SIZE: usize = 2 * I32_SIZE;

    let mut hex = [0; HEX_BUFF_SIZE];

    debug_out(b"level: 0x");
    let _ = hex::encode_to_slice(level.to_le_bytes(), &mut hex[..HEX_I32_SIZE]);
    debug_out(&hex[..HEX_I32_SIZE]);

    debug_out(b", index: 0x");
    let _ = hex::encode_to_slice(msg_index.to_le_bytes(), &mut hex[..HEX_I32_SIZE]);
    debug_out(&hex[..HEX_I32_SIZE]);

    debug_out(b", input: ");
    let (chunks, last) = input.as_chunks::<{ HEX_MAX_BYTES_ENCODED }>();

    for chunk in chunks {
        let _ = hex::encode_to_slice(chunk, hex.as_mut_slice());
        debug_out(&hex);
    }

    let _ = hex::encode_to_slice(last, &mut hex[..last.len() * 2]);
    debug_out(&hex[..last.len() * 2]);
    debug_out(b"\n");
}

fn debug_out(msg: &[u8]) {
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") 1, /* STDOUT FD */
            in("a1") msg.as_ptr(), /* ptr */
            in("a2") msg.len(), /* size */
            in("a7") 64 /* WRITE syscall */
        );
    }
}

#[panic_handler]
#[inline(never)]
fn panic(_panic: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}
