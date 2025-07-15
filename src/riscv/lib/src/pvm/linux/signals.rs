// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT
use core::num::NonZeroU64;
use std::fmt;

use arbitrary_int::u7;

use super::addr::VirtAddr;
use super::error::Error;
use crate::default::ConstDefault;

/// `sizeof(struct sigaction)` on the Kernel side
pub const SIZE_SIGACTION: usize = 32;

/// The size left over from this reduced implementation of `struct sigaction`
const SIZE_SIGACTION_BUFFER: usize = SIZE_SIGACTION - size_of::<VirtAddr>();

/// An action taken by a process upon receipt of a given signal
#[expect(unused, reason = "The use of this has not been implemented yet")]
struct SigAction {
    action: VirtAddr,
    buffer: [u8; SIZE_SIGACTION_BUFFER],
}

/// A mapping of (supported) signals to their signal actions
#[expect(unused, reason = "The use of this has not been implemented yet")]
pub struct SignalActions {
    action: [SigAction; SUPPORTED_SIGNALS.len()],
}

impl ConstDefault for SignalActions {
    const DEFAULT: Self = SignalActions { action: [] };
}

/// Size of the `sigset_t` type in bytes
///
/// As we're building a 64-bit system, the sigset should be 64-bit wide as well.
pub const SIGSET_SIZE: u64 = 8;

/// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
#[expect(unused, reason = "Some variants are not supported")]
pub enum Signum {
    Sighup = 1,
    Sigint,
    Sigquit,
    Sigill,
    Sigabrt,
    Sigiot,
    Sigbus,
    Sigfpe,
    Sigkill,
    Sigusr1,
    Sigsegv,
    Sigusr2,
    Sigpipe,
    Sigalrm,
    Sigterm,
    Sigstkflt,
    Sigchld,
    Sigcont,
    Sigstop,
    Sigtstp,
    Sigttin,
    Sigttou,
    Sigurg,
    Sigxcpu,
    Sigxfsz,
    Sigvtalrm,
    Sigprof,
    Sigwinch,
    Sigio,
    Sigpwr,
    Sigsys,
}

/// The signums supported for handling
pub const SUPPORTED_SIGNALS: [Signum; 0] = [];

/// A signal passed to a thread, see `tkill(2)`
#[derive(Debug, Clone, Copy)]
pub struct Signal(u7);

impl TryFrom<u64> for Signal {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(Signal(u7::try_new(value.try_into()?)?))
    }
}

impl Signal {
    /// Extract the exit code from the signal stored in this type
    pub fn exit_code(&self) -> u64 {
        // Setting bit 2^7 of the exit code indicates that the process was killed by a signal
        const EXIT_BY_SIGNAL: u8 = 1 << 7;

        (EXIT_BY_SIGNAL | self.0.value()) as u64
    }
}

/// An address of a signal action in the VM memory
#[derive(Clone, Copy)]
pub struct SignalActionPtr(pub Option<NonZeroU64>);

impl fmt::Debug for SignalActionPtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0.map(|nz| nz.get()).unwrap_or(0))
    }
}

impl From<u64> for SignalActionPtr {
    fn from(value: u64) -> Self {
        SignalActionPtr(NonZeroU64::new(value))
    }
}

impl SignalActionPtr {
    /// Extract the address of the signal action in the VM memory
    pub fn address(&self) -> Option<u64> {
        self.0.map(|nz| nz.get())
    }
}

/// A valid size of `sigset_t`
#[derive(Clone, Copy, Debug)]
pub struct SigsetTSizeEightBytes;

impl TryFrom<u64> for SigsetTSizeEightBytes {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        // As we're implementing a 64-bit system, the size of `sigset_t` must be 8 bytes.
        // This is an assumption which is used in the remainder of the function body.
        match value {
            SIGSET_SIZE => Ok(SigsetTSizeEightBytes),
            _ => Err(Error::InvalidArgument),
        }
    }
}
