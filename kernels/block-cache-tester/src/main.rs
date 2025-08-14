// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Block cache tester kernel for RISC-V PVM
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
//! check the return value. Despite the fact that the code has changed, we expect the return value
//! to still be the value of `foo` because the PVM has not yet invalidated the block cache. In
//! other words, the instruction memory is not synchronised with the data memory. Only after calling
//! `fence.i` to invalidate the block cache and thereby implicitly synchronising the instruction
//! memory with the data memory, we expect the return value to be that of `bar`.
//!
//! A really dangerous bug here would be if the PVM were to be rebound (e.g. by saving it to disk
//! and restoring it), and the instruction memory were to be implicitly synchronised with the data
//! memory. That would mean, stopping the PVM, serialising it, and restoring it would change the
//! semantics of the code that is executed. This is a bug that we want to avoid. You can think of a
//! proof generation as such serialisation; hence it is a realistic risk.
//!
//! # Note for future semantics of the PVM
//!
//! In the future, we may want to change the semantics of the PVM such that instruction memory
//! always synchronises with data memory immediately and automatically.

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

    let value = code();
    assert_eq!(value, foo());

    // SAFETY: This assembly is safe to run.
    unsafe {
        asm!("fence.i");
    }

    let value = code();
    assert_eq!(value, bar());

    unsafe { libc::exit(0) }
}
