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

extern "C" fn handle_sigusr1(
    sig: libc::c_int,
    info: *mut libc::siginfo_t,
    ucontext: *mut libc::ucontext_t,
) {
    assert_eq!(sig, libc::SIGUSR1);

    unsafe {
        assert!(libc::kill(1, libc::SIGUSR2) != 0);
    }

    unsafe {
        assert_eq!((*info).si_signo, libc::SIGUSR1);
        assert_eq!((*info).si_code, 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGHUP), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGINT), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGQUIT), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGILL), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGTRAP), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGABRT), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGIOT), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGBUS), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGFPE), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGKILL), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGUSR1), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGSEGV), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGUSR2), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGPIPE), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGALRM), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGTERM), 0);
        assert_eq!(
            libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGSTKFLT),
            0
        );
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGCHLD), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGCONT), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGSTOP), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGTSTP), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGTTIN), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGTTOU), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGURG), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGXCPU), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGXFSZ), 0);
        assert_eq!(
            libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGVTALRM),
            0
        );
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGPROF), 0);
        assert_eq!(
            libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGWINCH),
            0
        );
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGIO), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGPOLL), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGPWR), 0);
        assert_eq!(libc::sigismember(&(*ucontext).uc_sigmask, libc::SIGSYS), 0);
    }
}

extern "C" fn handle_sigusr1_nested(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ucontext: *mut libc::ucontext_t,
) {
    unsafe {
        assert!(libc::kill(1, libc::SIGUSR2) == 0);
    }
}

extern "C" fn unhandled_sigusr2(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ucontext: *mut libc::ucontext_t,
) {
    // This shouldn't be called
    panic!()
}

extern "C" fn handle_sigusr2(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ucontext: *mut libc::ucontext_t,
) {
    // noop
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

    // Set `SIGUSR1` to `handle_sigusr1` and an empty mask
    //
    // SAFETY: The sigset is initialised and the sigaction parameters are tested
    unsafe {
        let mut mask = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigfillset(mask.as_mut_ptr());

        let new = libc::sigaction {
            sa_sigaction: handle_sigusr1 as libc::sighandler_t,
            sa_mask: mask.assume_init(),
            sa_flags: 42,
            sa_restorer: None,
        };

        libc::sigaction(libc::SIGUSR1, &new, core::ptr::null_mut());
    }

    // Set `SIGUSR2` to `unhandled_sigusr2`
    //
    // SAFETY: The sigset is initialised and the sigaction parameters are tested
    unsafe {
        let mut mask = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigemptyset(mask.as_mut_ptr());

        let new = libc::sigaction {
            sa_sigaction: unhandled_sigusr2 as libc::sighandler_t,
            sa_mask: mask.assume_init(),
            sa_flags: 1337,
            sa_restorer: None,
        };

        libc::sigaction(libc::SIGUSR2, &new, core::ptr::null_mut());
    }

    // Call the `SIGUSR1` handler which will block calling `SIGUSR2`
    unsafe {
        libc::kill(1, libc::SIGUSR1);
    }

    // Set `SIGUSR1` to `handle_sigusr1_nested` and a mask that allows `SIGUSR2`
    //
    // SAFETY: The sigset is initialised and the sigaction parameters are tested
    unsafe {
        let mut mask = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigfillset(mask.as_mut_ptr());
        let mut mask = mask.assume_init();

        // Remove SIGUSR2 from the set of blocked signals
        libc::sigdelset(&mut mask, libc::SIGUSR2);

        let new = libc::sigaction {
            sa_sigaction: handle_sigusr1_nested as libc::sighandler_t,
            sa_mask: mask,
            sa_flags: 42,
            sa_restorer: None,
        };

        libc::sigaction(libc::SIGUSR1, &new, core::ptr::null_mut());
    }

    // Set `SIGUSR2` to `handle_sigusr2`
    //
    // SAFETY: The sigset is initialised and the sigaction parameters are tested.
    unsafe {
        let mut mask = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigemptyset(mask.as_mut_ptr());

        let new = libc::sigaction {
            sa_sigaction: handle_sigusr2 as libc::sighandler_t,
            sa_mask: mask.assume_init(),
            sa_flags: 1337,
            sa_restorer: None,
        };

        libc::sigaction(libc::SIGUSR2, &new, core::ptr::null_mut());
    }

    // Call the `SIGUSR1` handler which then calls the `SIGUSR2` handler
    unsafe {
        libc::kill(1, libc::SIGUSR1);
    }

    unsafe { libc::exit(0) }
}
