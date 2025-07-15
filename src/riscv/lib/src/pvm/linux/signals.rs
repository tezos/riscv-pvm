// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fmt;
use std::num::NonZeroU64;

use arbitrary_int::u7;

use super::error::Error;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::pvm::linux::SupervisorState;
use crate::pvm::linux::VirtAddr;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::Ref;
use crate::struct_layout;

const SIZE_SIGACTION: usize = 32;

/// Convert an identifier into the literal `1`, so it can be used to count the number of
/// repetitions
macro_rules! count_identifiers {
    ($identifer:ident) => {
        1usize
    };
}

// Marks an enum that is a subset of `Signum` as being supported by the PVM. Provides an array of
// supported `Signum`s as Indexed<enum> which can then be used as an index into the SignalActions
// array.
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
        // Reproduce the enum passed in
        $(
            #[$attributes]
        )*
        $visibility enum $enum_identifier {
            $(
                $enum_variant $(= $value)?
            ),*
        }

        paste::paste! {
            /// The total number of supported signums
            const [<$enum_identifier:snake:upper _COUNT>]: usize = 0 $( + count_identifiers!($enum_variant))*;

            /// An array of [[<$enum_identifier:upper _COUNT>]] [VirtAddr]s, one action for each
            /// supported signal
            $visibility struct SignalActions<M: ManagerBase> {
                signal_actions: [Cell<VirtAddr, M>; [<$enum_identifier:snake:upper _COUNT>]],
            }

            struct_layout! {
                /// Layout for [SignalActions]
                $visibility struct SignalActionsLayout {
                    signal_actions: [Atom<VirtAddr>; [<$enum_identifier:snake:upper _COUNT>]],
                }
            }
        }

        impl TryFrom<Signum> for $enum_identifier {
            type Error = Error;

            /// Convert a [Signum] into a supported [$enum_identifier], or return an error
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
    }
}

impl<M: ManagerAlloc> Default for SignalActions<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: ManagerBase> SignalActions<M> {
    /// Allocate a new [SignalActions]
    pub fn new() -> Self
    where
        M: ManagerAlloc,
    {
        SignalActions::<M> {
            signal_actions: core::array::from_fn(|_| Cell::new_with(VirtAddr::new(0))),
        }
    }

    /// Bind the given allocated regions to the supervisor state.
    pub fn bind(space: AllocatedOf<SignalActionsLayout, M>) -> Self {
        SignalActions::<M> {
            signal_actions: space.signal_actions,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<SignalActionsLayout, F::Output> {
        SignalActionsLayoutF {
            signal_actions: self
                .signal_actions
                .each_ref()
                .map(|sig_action| Cell::struct_ref::<F>(sig_action)),
        }
    }

    pub fn reset(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.signal_actions
            .iter_mut()
            .for_each(|sig_action| sig_action.write(VirtAddr::new(0)));
    }
}

impl<M: ManagerClone> Clone for SignalActions<M> {
    fn clone(&self) -> Self {
        SignalActions::<M> {
            signal_actions: self.signal_actions.clone(),
        }
    }
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
#[expect(
    dead_code,
    reason = "The use of the signum hasn't been implemented yet"
)]
pub struct Signal(Signum);

/// A signal passed to a thread, see `tkill(2)`
#[derive(Debug, Clone, Copy)]
pub struct TkillSignal(u7);

impl TkillSignal {
    /// Extract the exit code from the signal stored in this type
    pub fn exit_code(&self) -> u64 {
        // Setting bit 2^7 of the exit code indicates that the process was killed by a signal
        const EXIT_BY_SIGNAL: u8 = 1 << 7;

        (EXIT_BY_SIGNAL | self.0.value()) as u64
    }
}

impl TryFrom<u64> for TkillSignal {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(TkillSignal(u7::try_new(value.try_into()?)?))
    }
}

/// An address of a signal action in the VM memory
#[derive(Clone, Copy)]
pub struct SignalActionPtr(pub Option<NonZeroU64>);

impl SignalActionPtr {
    /// Extract the address of the signal action in the VM memory
    pub fn address(&self) -> Option<VirtAddr> {
        self.0.map(|nz| VirtAddr::new(nz.get()))
    }
}

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

impl<M: ManagerBase> SupervisorState<M> {
    /// Handle `sigaltstack` system call. The new signal stack configuration is discarded. If the
    /// old signal stack configuration is requested, it will be zeroed out.
    pub(super) fn handle_sigaltstack(
        &mut self,
        core: &mut MachineCoreState<impl MemoryConfig, M>,
        _: u64,
        old: SignalActionPtr,
    ) -> Result<u64, Error>
    where
        M: ManagerReadWrite,
    {
        /// `sizeof(struct sigaltstack)` on the Kernel side
        const SIZE_SIGALTSTACK: usize = 24;

        if let Some(old) = old.address() {
            core.main_memory
                .write(old.to_machine_address(), [0u8; SIZE_SIGALTSTACK])?;
        }

        // Return 0 as an indicator of success
        Ok(0)
    }

    /// Handle `rt_sigaction` system call. This does nothing effectively. It does not support
    /// retrieving the previous handler for a signal - it just zeroes out the memory.
    ///
    /// See: <https://www.man7.org/linux/man-pages/man2/rt_sigaction.2.html>
    pub(super) fn handle_rt_sigaction(
        &mut self,
        core: &mut MachineCoreState<impl MemoryConfig, M>,
        _: Signal,
        _: SignalActionPtr,
        old: SignalActionPtr,
        _: SigsetTSizeEightBytes,
    ) -> Result<u64, Error>
    where
        M: ManagerReadWrite,
    {
        if let Some(old) = old.address() {
            // As we don't store the previous signal handler, we just zero out the memory
            core.main_memory
                .write(old.to_machine_address(), [0u8; SIZE_SIGACTION])?;
        }

        // Return 0 as an indicator of success
        Ok(0)
    }

    /// Handle `rt_sigprocmask` system call. This does nothing effectively. If the previous mask is
    /// requested, it will simply be zeroed out.
    pub(super) fn handle_rt_sigprocmask(
        &mut self,
        core: &mut MachineCoreState<impl MemoryConfig, M>,
        _: u64,
        _: u64,
        old: SignalActionPtr,
        _: SigsetTSizeEightBytes,
    ) -> Result<u64, Error>
    where
        M: ManagerReadWrite,
    {
        if let Some(old) = old.address() {
            // As we don't store the previous mask, we just zero out the memory
            core.main_memory
                .write(old.to_machine_address(), [0u8; SIGSET_SIZE as usize])?;
        }

        // Return 0 as an indicator of success
        Ok(0)
    }
}
