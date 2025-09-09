// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Signal tester kernel for RISC-V PVM
#![no_std]

use core::mem::MaybeUninit;

pub fn main() {
    // Silences error about lack of `#[panic_handler]` function for `#![no_std]`
    memmap2::MmapOptions::new().len(0x1000).map_anon().unwrap();

    // SAFETY: The sigset is initialised and the sigaction parameters are tested
    unsafe {
        let mut mask = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigemptyset(mask.as_mut_ptr());

        let new = libc::sigaction {
            sa_sigaction: 0x1111,
            sa_mask: mask.assume_init(),
            sa_flags: 42,
            sa_restorer: None,
        };

        libc::sigaction(libc::SIGUSR1, &new, core::ptr::null_mut());
    }

    // SAFETY: The sigset is initialised and the sigaction parameters are tested
    unsafe {
        let mut mask = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigfillset(mask.as_mut_ptr());

        let new = libc::sigaction {
            sa_sigaction: 0xAAAA,
            sa_mask: mask.assume_init(),
            sa_flags: 1337,
            sa_restorer: None,
        };

        libc::sigaction(libc::SIGUSR2, &new, core::ptr::null_mut());
    }

    unsafe { libc::exit(0) }
}
