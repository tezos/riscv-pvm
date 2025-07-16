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

impl ConstDefault for SigAction {
    const DEFAULT: Self = SigAction {
        action: VirtAddr::new(0),
        buffer: [0; SIZE_SIGACTION_BUFFER],
    };
}

/// A mapping of (supported) signals to their signal actions
#[expect(unused, reason = "The use of this has not been implemented yet")]
pub struct SignalActions {
    action: [SigAction; INDEXED_SIGNUM_COUNT],
}

impl ConstDefault for SignalActions {
    const DEFAULT: Self = SignalActions {
        action: [SigAction::DEFAULT; INDEXED_SIGNUM_COUNT],
    };
}

/// Size of the `sigset_t` type in bytes
///
/// As we're building a 64-bit system, the sigset should be 64-bit wide as well.
pub const SIGSET_SIZE: u64 = 8;

/// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
#[derive(Clone, Copy, Debug)]
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

impl TryFrom<u64> for Signum {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Signum::Sighup),
            2 => Ok(Signum::Sigint),
            3 => Ok(Signum::Sigquit),
            4 => Ok(Signum::Sigill),
            5 => Ok(Signum::Sigabrt),
            6 => Ok(Signum::Sigiot),
            7 => Ok(Signum::Sigbus),
            8 => Ok(Signum::Sigfpe),
            9 => Ok(Signum::Sigkill),
            10 => Ok(Signum::Sigusr1),
            11 => Ok(Signum::Sigsegv),
            12 => Ok(Signum::Sigusr2),
            13 => Ok(Signum::Sigpipe),
            14 => Ok(Signum::Sigalrm),
            15 => Ok(Signum::Sigterm),
            16 => Ok(Signum::Sigstkflt),
            17 => Ok(Signum::Sigchld),
            18 => Ok(Signum::Sigcont),
            19 => Ok(Signum::Sigstop),
            20 => Ok(Signum::Sigtstp),
            21 => Ok(Signum::Sigttin),
            22 => Ok(Signum::Sigttou),
            23 => Ok(Signum::Sigurg),
            24 => Ok(Signum::Sigxcpu),
            25 => Ok(Signum::Sigxfsz),
            26 => Ok(Signum::Sigvtalrm),
            27 => Ok(Signum::Sigprof),
            28 => Ok(Signum::Sigwinch),
            29 => Ok(Signum::Sigio),
            30 => Ok(Signum::Sigpwr),
            31 => Ok(Signum::Sigsys),
            _ => Err(Error::InvalidArgument),
        }
    }
}

/// Maps an identifier into a literal used for counting. By using this on repeated literals and
/// summing, the total number of literals can be found
macro_rules! count_variant {
    ($variant:ident) => {
        1usize
    };
}

/// Marks an enum that is a subset of [Signum] as being supported by the PVM Provides an array of
/// supported [Signum]s as Indexed<enum> which can then be used as an index into the
/// SignalActions array.
macro_rules! supported {
    (
        $(
            #[$attributes:meta]
        )*
        $visibility:vis enum $enum_identifier:ident {
            $(
                $enum_variant:ident$(= $value:expr)?$(,)?
            )*
        }
    ) => {
        /// The variants supported
        $(
            #[$attributes]
        )*
        $visibility enum $enum_identifier {
            $(
                $enum_variant $(= $value)?
            ),*
        }

        paste::paste! {
            /// The total number of supported variants
            $visibility const [<$enum_identifier:snake:upper _COUNT>]: usize = 0usize $(
                + count_variant!($enum_variant)
            )*;
        }

        paste::paste! {
            /// An array of the supported variants
            #[expect(unused, reason = "The use of this has not been implemented yet")]
            $visibility const [<$enum_identifier:snake:upper S>]:
                [$enum_identifier; [<$enum_identifier:snake:upper _COUNT>]] = [
                $(
                    $enum_identifier::$enum_variant,
                )*
            ];
        }

        impl TryFrom<Signum> for $enum_identifier {
            type Error = Error;

            fn try_from(value: Signum) -> Result<Self, Self::Error> {
                let value = value.try_into()?;
                match value {
                    $(
                        Signum::$enum_variant => Ok($enum_identifier::$enum_variant),
                    )*
                    _ => Err(Error::InvalidArgument),
                }
            }
        }

        impl TryFrom<u64> for Signal {
            type Error = Error;

            /// Only supported signals can be used to create a valid Signal
            fn try_from(value: u64) -> Result<Self, Self::Error> {
                let value = value.try_into()?;
                match value {
                    $(
                        Signum::$enum_variant => Ok(Signal(value)),
                    )*
                    _ => Err(Error::InvalidArgument),
                }
            }
        }
    };
}

supported!(
    /// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
    #[derive(Debug, Clone, Copy)]
    #[repr(usize)]
    pub enum IndexedSignum {
        Sigill = 0,
        Sigabrt,
        Sigiot,
        Sigbus,
        Sigfpe,
        Sigkill,
        Sigusr1,
        Sigsegv,
        Sigusr2,
        Sigpipe,
        Sigterm,
        Sigstop,
        Sigsys,
    }
);

/// A signal that is supported by the PVM
#[derive(Debug, Clone, Copy)]
pub struct Signal(Signum);

impl Signal {
    pub fn index(self) -> usize {
        // SAFETY: Only supported signals can be used to construct a Signal.
        // The macro ensures this is in-bounds.
        let value: IndexedSignum = self.0.try_into().unwrap();
        unsafe { std::mem::transmute(value) }
    }
}

/// A signal passed to a thread, see `tkill(2)`
#[derive(Debug, Clone, Copy)]
pub struct TkillSignal(u7);

impl TryFrom<u64> for TkillSignal {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(TkillSignal(u7::try_new(value.try_into()?)?))
    }
}

impl TkillSignal {
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
    pub fn address(&self) -> Option<VirtAddr> {
        self.0.map(|nz| VirtAddr::new(nz.get()))
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
