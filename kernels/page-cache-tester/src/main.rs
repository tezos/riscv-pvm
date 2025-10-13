// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Page cache tester kernel for RISC-V PVM
//!
//! This module provides a test kernel that demonstrates dynamic code execution and cache
//! invalidation behavior in the RISC-V PVM.
//!
//! # How it works
//!
//! We compile two functions `foo` and `bar` that return different values.
//!
//! Upon starting the kernel, we allocate a memory page and copy the code of `foo` into it. Then
//! the page is made executable. The address at start of the page is effectively a function pointer
//! to a copy of `foo`. We call this function pointer and check that it returns the expected value.
//!
//! Once we have established that the code works, we make the page mutable again and copy the code
//! of `bar` into it. We make the page executable and call the function pointer again. Again, we
//! check the return value. The code has changed, and we expect the return value
//! to now be the value of `bar` because the PVM's instruction memory is automatically
//! synchronised with data memory.
//!
//! The calling of `fence.i` therefore has no effect.

#![no_std]

use core::arch::asm;
use core::mem::transmute;

extern "C" fn foo() -> u32 {
    42
}

extern "C" fn bar() -> u32 {
    1337
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

    let mut code_page = code_page.make_mut().unwrap();
    assert_eq!(code_ptr, code_page.as_ptr());

    // SAFETY: We rely on the RISC-V code generator to produce simple relocatable functions.
    unsafe {
        code_page
            .as_mut_ptr()
            .copy_from_nonoverlapping(bar as *const u8, 0x1000);
    }

    let code_page = code_page.make_exec().unwrap();
    assert_eq!(code_ptr, code_page.as_ptr());

    // SAFETY: The beginning of `code_page` is a valid function pointer now that we've copied the
    // code and made the page executable.
    let code = unsafe { transmute::<*const u8, extern "C" fn() -> u32>(code_page.as_ptr()) };

    // SAFETY: Unknown system calls will be ignored by the PVM.
    unsafe {
        // This system call acts as a signal. The test case which uses this kernel will use it to
        // rebind the PVM state, for example.
        libc::syscall(-1);
    }

    // instruction memory is implicitly synchronized with data memory
    let value = code();
    assert_eq!(value, bar());

    // SAFETY: This assembly is safe to run.
    //
    // This will have no effect.
    unsafe {
        asm!("fence.i");
    }

    let value = code();
    assert_eq!(value, bar());

    unsafe { libc::exit(0) }
}
