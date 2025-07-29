// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use arbitrary_int::u7;
use strum::EnumCount;
use strum::FromRepr;

use super::error::Error;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::pvm::linux::SupervisorState;
use crate::pvm::linux::VirtAddr;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::state_backend::Elem;
use crate::state_backend::FnManager;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::Ref;
use crate::struct_layout;

#[repr(C)]
#[derive(Clone, Debug)]
/// Linux sigaction struct, see <https://man7.org/linux/man-pages/man2/sigaction.2.html>
pub struct LinuxSigAction {
    sa_handler: VirtAddr,
    sa_sigaction: VirtAddr,
    sa_mask: u32,
    sa_flags: u32,
    sa_restorer: VirtAddr,
}

#[cfg(test)]
// Currently we only support the `sa_sigaction` field, so in tests it's useful to be able to
// create a sigaction using only this field.
impl LinuxSigAction {
    pub(crate) fn new(sa_sigaction: VirtAddr) -> Self {
        Self {
            sa_handler: VirtAddr::new(0),
            sa_sigaction,
            sa_mask: 0,
            sa_flags: 0,
            sa_restorer: VirtAddr::new(0),
        }
    }
}

#[cfg(test)]
impl Default for LinuxSigAction {
    fn default() -> Self {
        Self {
            sa_handler: VirtAddr::new(0),
            sa_sigaction: VirtAddr::new(0),
            sa_mask: 0,
            sa_flags: 0,
            sa_restorer: VirtAddr::new(0),
        }
    }
}

#[cfg(test)]
impl PartialEq for LinuxSigAction {
    fn eq(&self, other: &Self) -> bool {
        self.sa_handler == other.sa_handler
            && self.sa_sigaction == other.sa_sigaction
            && self.sa_mask == other.sa_mask
            && self.sa_flags == other.sa_flags
            && self.sa_restorer == other.sa_restorer
    }
}

/// `size_of(struct sigaction)` on the Kernel side
const SIZE_SIGACTION: usize = 32;

impl Elem for LinuxSigAction {
    const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(SIZE_SIGACTION).unwrap();

    unsafe fn read_unaligned(source: *const u8) -> Self {
        unsafe {
            let sa_handler_bits = source.cast::<u64>().read();
            let sa_sigaction_bits = source.add(size_of::<u64>()).cast::<u64>().read();
            let sa_mask_bits = source
                .add(size_of::<u64>())
                .add(size_of::<u64>())
                .cast::<u32>()
                .read();
            let sa_flags_bits = source
                .add(size_of::<u64>())
                .add(size_of::<u64>())
                .add(size_of::<u32>())
                .cast::<u32>()
                .read();
            let sa_restorer_bits = source
                .add(size_of::<u64>())
                .add(size_of::<u64>())
                .add(size_of::<u32>())
                .add(size_of::<u32>())
                .cast::<u64>()
                .read();
            Self {
                sa_handler: VirtAddr::new(u64::from_le(sa_handler_bits)),
                sa_sigaction: VirtAddr::new(u64::from_le(sa_sigaction_bits)),
                sa_mask: u32::from_le(sa_mask_bits),
                sa_flags: u32::from_le(sa_flags_bits),
                sa_restorer: VirtAddr::new(u64::from_le(sa_restorer_bits)),
            }
        }
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        unsafe {
            dest.cast::<u64>()
                .write(self.sa_handler.to_machine_address().to_le());
            dest.add(size_of::<u64>())
                .cast::<u64>()
                .write(self.sa_sigaction.to_machine_address().to_le());
            dest.add(size_of::<u64>())
                .add(size_of::<u64>())
                .cast::<u32>()
                .write(self.sa_mask.to_le());
            dest.add(size_of::<u64>())
                .add(size_of::<u64>())
                .add(size_of::<u32>())
                .cast::<u32>()
                .write(self.sa_flags.to_le());
            dest.add(size_of::<u64>())
                .add(size_of::<u64>())
                .add(size_of::<u32>())
                .add(size_of::<u32>())
                .cast::<u64>()
                .write(self.sa_restorer.to_machine_address().to_le());
        }
    }
}

// For [Cell]<E, _>, `E` must be 'static. For this reason, each field of the [LinuxSigAction]
// struct will have its own array of the primitives or wrappers around primitives (e.g. [VirtAddr])
// used for the member's type.

/// Information to support handling each supported signal
pub struct SignalActions<M: ManagerBase> {
    /// An array of [VirtAddr]s, unused
    handlers: [Cell<VirtAddr, M>; SignalIndex::COUNT],
    /// An array of [VirtAddr]s, one action for each supported signal
    actions: [Cell<VirtAddr, M>; SignalIndex::COUNT],
    /// An array of bitmasks, one mask for each supported signal
    masks: [Cell<u32, M>; SignalIndex::COUNT],
    /// An array of bitmasks, one set of flags for each supported signal
    flags: [Cell<u32, M>; SignalIndex::COUNT],
    /// An array of [VirtAddr]s, restorers for each signal, see
    /// <https://www.man7.org/linux/man-pages/man2/sigreturn.2.html>
    restorers: [Cell<VirtAddr, M>; SignalIndex::COUNT],
}

struct_layout! {
    /// Layout for [SignalActions]
    pub struct SignalActionsLayout {
        handlers: [Atom<VirtAddr>; SignalIndex::COUNT],
        actions: [Atom<VirtAddr>; SignalIndex::COUNT],
        masks: [Atom<u32>; SignalIndex::COUNT],
        flags: [Atom<u32>; SignalIndex::COUNT],
        restorers: [Atom<VirtAddr>; SignalIndex::COUNT],
    }
}

impl<MC: MemoryConfig, M: ManagerBase> MachineCoreState<MC, M> {
    fn signal_action(&self, signal: Signal) -> LinuxSigAction
    where
        M: ManagerRead,
    {
        let index = signal_index(signal);
        let sa_sigaction = self.signal_actions.actions[index].read();

        LinuxSigAction {
            sa_handler: VirtAddr::new(0),
            sa_sigaction,
            sa_mask: 0,
            sa_flags: 0,
            sa_restorer: VirtAddr::new(0),
        }
    }

    fn set_signal_action(&mut self, signal: Signal, action: LinuxSigAction)
    where
        M: ManagerReadWrite,
    {
        let index = signal_index(signal);
        self.signal_actions.actions[index].write(action.sa_sigaction);
    }
}

impl<M: ManagerAlloc> Default for SignalActions<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: ManagerBase> SignalActions<M> {
    /// Bind the given allocated regions to the supervisor state.
    pub fn bind(space: AllocatedOf<SignalActionsLayout, M>) -> Self {
        SignalActions::<M> {
            handlers: space.handlers,
            actions: space.actions,
            masks: space.masks,
            flags: space.flags,
            restorers: space.restorers,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<SignalActionsLayout, F::Output> {
        SignalActionsLayoutF {
            handlers: self
                .handlers
                .each_ref()
                .map(|handler| Cell::struct_ref::<F>(handler)),
            actions: self
                .actions
                .each_ref()
                .map(|sig_action| Cell::struct_ref::<F>(sig_action)),
            masks: self
                .masks
                .each_ref()
                .map(|mask| Cell::struct_ref::<F>(mask)),
            flags: self
                .flags
                .each_ref()
                .map(|flag| Cell::struct_ref::<F>(flag)),
            restorers: self
                .restorers
                .each_ref()
                .map(|restorer| Cell::struct_ref::<F>(restorer)),
        }
    }

    /// Reset to the default state
    pub fn reset(&mut self)
    where
        M: ManagerReadWrite,
    {
        self.actions
            .iter_mut()
            .for_each(|sig_action| sig_action.write(VirtAddr::new(0)));
    }
}

impl<M: ManagerBase> NewState<M> for SignalActions<M> {
    /// Allocate a new [SignalActions]
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        SignalActions::<M> {
            handlers: core::array::from_fn(|_| Cell::new_with(VirtAddr::new(0))),
            actions: core::array::from_fn(|_| Cell::new_with(VirtAddr::new(0))),
            masks: core::array::from_fn(|_| Cell::new_with(0u32)),
            flags: core::array::from_fn(|_| Cell::new_with(0u32)),
            restorers: core::array::from_fn(|_| Cell::new_with(VirtAddr::new(0))),
        }
    }
}

impl<M: ManagerClone> Clone for SignalActions<M> {
    fn clone(&self) -> Self {
        SignalActions::<M> {
            handlers: self.handlers.clone(),
            actions: self.actions.clone(),
            masks: self.masks.clone(),
            flags: self.flags.clone(),
            restorers: self.restorers.clone(),
        }
    }
}

/// Size of the `sigset_t` type in bytes
///
/// As we're building a 64-bit system, the sigset should be 64-bit wide as well.
pub const SIGSET_SIZE: u64 = 8;

/// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
#[derive(Debug, Clone, Copy, FromRepr)]
#[repr(u64)]
pub enum Signal {
    Sigill = 4,
    Sigabrt = 5,
    Sigiot = 6,
    Sigbus = 7,
    Sigfpe = 8,
    Sigkill = 9,
    Sigusr1 = 10,
    Sigsegv = 11,
    Sigusr2 = 12,
    Sigpipe = 13,
    Sigterm = 15,
    Sigstop = 19,
    Sigsys = 31,
}

fn signal_index(signal: Signal) -> usize {
    let signal_index: SignalIndex = signal.into();
    unsafe { std::mem::transmute::<SignalIndex, usize>(signal_index) }
}

impl TryFrom<u64> for Signal {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(Error::InvalidArgument)
    }
}

/// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
/// The representation of these enums are used for indices into signal action storage.
#[derive(Debug, Clone, Copy, EnumCount)]
#[repr(usize)]
pub enum SignalIndex {
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

impl From<Signal> for SignalIndex {
    fn from(value: Signal) -> Self {
        match value {
            Signal::Sigill => SignalIndex::Sigill,
            Signal::Sigabrt => SignalIndex::Sigabrt,
            Signal::Sigiot => SignalIndex::Sigiot,
            Signal::Sigbus => SignalIndex::Sigbus,
            Signal::Sigfpe => SignalIndex::Sigfpe,
            Signal::Sigkill => SignalIndex::Sigkill,
            Signal::Sigusr1 => SignalIndex::Sigusr1,
            Signal::Sigsegv => SignalIndex::Sigsegv,
            Signal::Sigusr2 => SignalIndex::Sigusr2,
            Signal::Sigpipe => SignalIndex::Sigpipe,
            Signal::Sigterm => SignalIndex::Sigterm,
            Signal::Sigstop => SignalIndex::Sigstop,
            Signal::Sigsys => SignalIndex::Sigsys,
        }
    }
}

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
#[derive(Clone, Copy, Debug)]
pub struct SignalActionPtr(Option<VirtAddr>);

impl SignalActionPtr {
    /// Extract the address of the signal action in the VM memory
    pub fn address(&self) -> Option<u64> {
        self.0.map(|addr| addr.to_machine_address())
    }
}

impl From<u64> for SignalActionPtr {
    fn from(value: u64) -> Self {
        SignalActionPtr(Some(VirtAddr::new(value)))
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
            core.main_memory.write(old, [0u8; SIZE_SIGALTSTACK])?;
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
        signal: Signal,
        action: SignalActionPtr,
        old: SignalActionPtr,
        _: SigsetTSizeEightBytes,
    ) -> Result<u64, Error>
    where
        M: ManagerReadWrite,
    {
        if let Some(old) = old.address() {
            let old_action = core.signal_action(signal);
            core.main_memory.write(old, old_action)?;
        }

        if let Some(action) = action.address() {
            let new_action: LinuxSigAction = core.main_memory.read(action)?;
            core.set_signal_action(signal, new_action);
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
            core.main_memory.write(old, [0u8; SIGSET_SIZE as usize])?;
        }

        // Return 0 as an indicator of success
        Ok(0)
    }
}
