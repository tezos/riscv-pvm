// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Signal tester kernel for RISC-V PVM
#![no_std]

use core::mem::MaybeUninit;
use core::mem::transmute;

extern "C" fn foo() -> u32 {
    42
}

pub fn main() {
    let mut code_page = memmap2::MmapOptions::new().len(0x1000).map_anon().unwrap();
    let code_ptr = code_page.as_ptr();

    // SAFETY: We rely on the RISC-V code generator to produce simple relocatable functions.
    unsafe {
        code_page
            .as_mut_ptr()
            .copy_from_nonoverlapping(foo as *const u8, 0x1000);
    }

    let code_page = code_page.make_exec().unwrap();
    assert_eq!(code_ptr, code_page.as_ptr());

    // SAFETY: The beginning of `code_page` is a valid function pointer now that we've copied the
    // code and made the page executable.
    let code = unsafe { transmute::<*const u8, extern "C" fn() -> u32>(code_page.as_ptr()) };

    let value = code();
    assert_eq!(value, foo());

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
