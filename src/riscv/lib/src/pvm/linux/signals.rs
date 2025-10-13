// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;
use std::ops::Add;
use std::ops::Sub;
use std::slice::from_raw_parts;
use std::slice::from_raw_parts_mut;

use arbitrary_int::u7;
use strum::EnumCount;
use strum::FromRepr;
use zerocopy::FromBytes;
use zerocopy::IntoBytes;
use zerocopy::byteorder::LittleEndian;
use zerocopy::byteorder::U32;
use zerocopy::byteorder::U64;
use zerocopy_derive::FromBytes;
use zerocopy_derive::Immutable;
use zerocopy_derive::IntoBytes;
use zerocopy_derive::KnownLayout;

use super::MachineError;
use super::PAGE_SIZE;
use super::Permissions;
use super::RT_SIGRETURN;
use super::error::Error;
use crate::machine_state::MachineCoreState;
use crate::machine_state::ProgramCounterUpdate;
use crate::machine_state::RISCV_ABI_SP_ALIGNMENT;
use crate::machine_state::memory::BadMemoryAccess;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::code_page_entry::CodePageEntry;
use crate::machine_state::registers::nz;
use crate::machine_state::registers::sp;
use crate::pvm::Pvm;
use crate::pvm::linux::Address;
use crate::pvm::linux::SupervisorState;
use crate::pvm::linux::VirtAddr;
use crate::pvm::linux::parameters::SystemCallResultExecution;
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
use crate::state_backend::ManagerWrite;
use crate::state_backend::Ref;
use crate::struct_layout;

/// Errors relating to handling signals
#[derive(Debug, Eq, thiserror::Error, PartialEq)]
pub enum SignalError {
    #[error("Bad signal context")]
    BadContext,
    #[error("Signals of this type cannot have their disposition changed")]
    ImmutableDisposition,
    #[error(transparent)]
    Memory(#[from] BadMemoryAccess),
    #[error("Misaligned stack pointer")]
    MisalignedStackPointer,
    #[error("Execution shall terminate")]
    Terminate,
}

/// Linux sigaction struct, see <https://man7.org/linux/man-pages/man2/sigaction.2.html>
/// and <https://github.com/torvalds/linux/blob/155a3c003e555a7300d156a5252c004c392ec6b0/include/linux/signal_types.h#L37>
#[repr(C)]
#[derive(Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(Default, PartialEq))]
pub struct LinuxSigAction {
    pub sa_sigaction: U64<LittleEndian>,
    pub sa_flags: U32<LittleEndian>,
    /// The kernel struct has padding between the flags and mask that would be used by the restorer if
    /// RISC-V's ABI didn't use the vDSO for a restorer.
    __sigreturn_padding: [u8; 12],
    pub sa_mask: U64<LittleEndian>,
}

/// In tests it's useful to be able to create a sigaction using only this field.
#[cfg(test)]
impl LinuxSigAction {
    pub(crate) fn new(sa_sigaction: VirtAddr) -> Self {
        Self {
            sa_sigaction: sa_sigaction.to_machine_address().into(),
            ..Default::default()
        }
    }
}

/// Set the default signal disposition
///
/// Value:
/// <https://github.com/torvalds/linux/blob/b320789d6883cc00ac78ce83bccbfe7ed58afcf0/include/uapi/asm-generic/signal-defs.h#L88>
pub const SIG_DFL: VirtAddr = VirtAddr::new(0u64);

/// Set the signal disposition to ignore
///
/// Value:
/// <https://github.com/torvalds/linux/blob/b320789d6883cc00ac78ce83bccbfe7ed58afcf0/include/uapi/asm-generic/signal-defs.h#L89>
pub const SIG_IGN: VirtAddr = VirtAddr::new(1u64);

/// Flag that declares a signal handler has been set with `rt_sigaction(2)` so is called with
/// parameters.
pub const SA_SIGINFO: u32 = 0x4000000;

/// `size_of(struct sigaction)` on the Kernel side
const SIZE_SIGACTION: usize = 32;

impl Elem for LinuxSigAction {
    const STORED_SIZE: NonZeroUsize = { NonZeroUsize::new(SIZE_SIGACTION).unwrap() };

    unsafe fn read_unaligned(source: *const u8) -> Self {
        // SAFETY: The bitwise representation is the same as `write_unaligned` and matches each
        // field.
        unsafe {
            LinuxSigAction::read_from_bytes(from_raw_parts(source, Self::STORED_SIZE.get()))
                .expect("Bad sigaction")
        }
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        // SAFETY: The bitwise representation is the same as `read_unaligned` and matches each
        // field.
        unsafe {
            let _ = self.write_to(from_raw_parts_mut(dest, Self::STORED_SIZE.get()));
        }
    }
}

// For [Cell]<E, _>, `E` must be 'static. For this reason, each field of the [LinuxSigAction]
// struct will have its own array of the primitives or wrappers around primitives (e.g. [VirtAddr])
// used for the member's type.

/// Information to support handling each supported signal
pub struct SignalActions<M: ManagerBase> {
    /// An array of [VirtAddr]s, one action for each supported signal
    actions: [Cell<VirtAddr, M>; SignalIndex::COUNT],
    /// An array of bitmasks, one set of flags for each supported signal
    flags: [Cell<u32, M>; SignalIndex::COUNT],
    /// An array of bitmasks, one mask for each supported signal
    masks: [Cell<u64, M>; SignalIndex::COUNT],
    /// A pointer to a restorer address to jump to after returning from a signal handler. A
    /// function to call `rt_sigreturn` should be loaded into here.
    ///
    /// <https://www.man7.org/linux/man-pages/man2/sigreturn.2.html>
    restorer: Cell<VirtAddr, M>,
    /// A per-thread mask for all signals
    thread_mask: Cell<u64, M>,
}

impl<M: ManagerRead> SignalActions<M> {
    /// Read the action for a given signal
    pub(crate) fn read_action<T: Into<SignalIndex>>(&self, signal: T) -> VirtAddr {
        let signal_index: SignalIndex = signal.into();
        self.actions[signal_index as usize].read()
    }

    /// Read the flags for a given signal
    pub(crate) fn read_flags<T: Into<SignalIndex>>(&self, signal: T) -> u32 {
        let signal_index: SignalIndex = signal.into();
        self.flags[signal_index as usize].read()
    }

    /// Read the mask for a given signal
    pub(crate) fn read_mask<T: Into<SignalIndex>>(&self, signal: T) -> u64 {
        let signal_index: SignalIndex = signal.into();
        self.masks[signal_index as usize].read()
    }

    /// Has the [SA_SIGINFO] flag been set?
    ///
    /// i.e., was this signal action registered using `rt_sigaction(2)`?
    pub fn sa_siginfo<T: Into<SignalIndex>>(&self, signal: T) -> bool {
        self.read_flags(signal) & SA_SIGINFO != 0
    }
}

impl<M: ManagerWrite> SignalActions<M> {
    /// Write the action for a given signal
    pub(crate) fn write_action<T: Into<SignalIndex>>(&mut self, signal: T, action: VirtAddr) {
        let signal_index: SignalIndex = signal.into();
        self.actions[signal_index as usize].write(action);
    }

    /// Write the flags for a given signal
    pub(crate) fn write_flags<T: Into<SignalIndex>>(&mut self, signal: T, flags: u32) {
        let signal_index: SignalIndex = signal.into();
        self.flags[signal_index as usize].write(flags);
    }

    /// Write the mask for a given signal
    pub(crate) fn write_mask<T: Into<SignalIndex>>(&mut self, signal: T, mask: u64) {
        let signal_index: SignalIndex = signal.into();
        self.masks[signal_index as usize].write(mask);
    }
}

struct_layout! {
    /// Layout for [SignalActions]
    pub struct SignalActionsLayout {
        actions: [Atom<VirtAddr>; SignalIndex::COUNT],
        flags: [Atom<u32>; SignalIndex::COUNT],
        masks: [Atom<u64>; SignalIndex::COUNT],
        restorer: Atom<VirtAddr>,
        thread_mask: Atom<u64>,
    }
}

impl<MC: MemoryConfig, M: ManagerReadWrite> MachineCoreState<MC, M> {
    /// Set the hart state to a signal handler
    pub fn dispatch_signal(&mut self, signal: Signal) -> Result<(), SignalError> {
        let handler = self.signal_actions.read_action(signal);

        match handler {
            SIG_IGN => return Ok(()),
            SIG_DFL => {
                match Disposition::default(signal) {
                    Disposition::Term | Disposition::Core => {
                        return Err(SignalError::Terminate);
                    }
                    Disposition::Stop => return Ok(()),
                };
            }
            _ => (),
        }

        let restorer = self.push_signal_context(signal)?;

        self.hart
            .xregisters
            .write_nz(nz::ra, restorer.to_machine_address());

        // Write handler arguments
        // Signal number
        self.hart.xregisters.write_nz(nz::a0, signal as u64);

        if self.signal_actions.sa_siginfo(signal) {
            // TODO RV-754: Implement storing signal information as `siginfo_t`
            // Pointer to siginfo_t
            self.hart.xregisters.write_nz(nz::a1, 0u64);

            // TODO RV-754: Implement storing signal execution context as `ucontext_t`
            // Pointer to ucontext_t
            self.hart.xregisters.write_nz(nz::a2, 0u64);
        }

        // Update the program counter to the handler
        self.hart.pc.write(handler.to_machine_address());
        Ok(())
    }

    /// Pushes the context needed to resume after handling a signal
    pub fn push_signal_context(&mut self, signal: Signal) -> Result<VirtAddr, SignalError> {
        let signal_index: SignalIndex = signal.into();
        let mask = self.signal_actions.read_mask(signal_index);
        let pc = self.hart.pc.read();

        let restorer = self.signal_actions.restorer.read();

        let prev_stack_pointer = self.hart.xregisters.read(sp);
        let stack_pointer = VirtAddr::new(prev_stack_pointer)
            .sub(32)
            .align_down(RISCV_ABI_SP_ALIGNMENT);

        self.hart
            .xregisters
            .write(sp, stack_pointer.to_machine_address());

        self.main_memory
            .write(stack_pointer.to_machine_address(), signal_index as u64)?;
        self.main_memory
            .write(stack_pointer.add(8).to_machine_address(), mask)?;
        self.main_memory
            .write(stack_pointer.add(16).to_machine_address(), pc)?;
        self.main_memory.write(
            stack_pointer.add(24).to_machine_address(),
            prev_stack_pointer,
        )?;

        Ok(restorer)
    }

    /// Pops the context needed to resume after handling a signal
    pub fn pop_signal_context(&mut self) -> Result<Address, SignalError> {
        let stack_pointer = VirtAddr::new(self.hart.xregisters.read(sp));

        let prev_stack_pointer = self
            .main_memory
            .read(stack_pointer.add(24).to_machine_address())?;
        let pc: u64 = self
            .main_memory
            .read(stack_pointer.add(16).to_machine_address())?;
        let mask: u64 = self
            .main_memory
            .read(stack_pointer.add(8).to_machine_address())?;
        let signal_index: u64 = self.main_memory.read(stack_pointer.to_machine_address())?;

        // SAFETY: This was stored by converting from a SignalIndex
        let signal_index =
            SignalIndex::from_repr(signal_index as usize).ok_or(SignalError::BadContext)?;
        self.signal_actions.write_mask(signal_index, mask);
        // TODO RV-734 Restore the alternative stack
        // TODO RV-755 Store and restore registers in push/pop_signal_context

        let stack_pointer = VirtAddr::new(prev_stack_pointer)
            .add(32)
            .align_up(RISCV_ABI_SP_ALIGNMENT)
            .ok_or(SignalError::MisalignedStackPointer)?;

        self.hart
            .xregisters
            .write(sp, stack_pointer.to_machine_address());

        Ok(pc)
    }
}

impl<MC: MemoryConfig, M: ManagerBase> MachineCoreState<MC, M> {
    fn signal_action(&self, signal: Signal) -> LinuxSigAction
    where
        M: ManagerRead,
    {
        let index: SignalIndex = signal.into();
        let sa_sigaction = self.signal_actions.read_action(index);
        let sa_flags = self.signal_actions.read_flags(index);
        let sa_mask = self.signal_actions.read_mask(index);

        LinuxSigAction {
            sa_sigaction: sa_sigaction.to_machine_address().into(),
            sa_flags: sa_flags.into(),
            __sigreturn_padding: Default::default(),
            sa_mask: sa_mask.into(),
        }
    }

    fn set_signal_action(
        &mut self,
        signal: Signal,
        action: LinuxSigAction,
    ) -> Result<(), SignalError>
    where
        M: ManagerWrite,
    {
        // These signals cannot have their dispositions changed, see
        // <https://www.man7.org/linux/man-pages/man7/signal.7.html>
        if signal == Signal::Sigkill || signal == Signal::Sigstop {
            return Err(SignalError::ImmutableDisposition);
        }

        let index: SignalIndex = signal.into();
        self.signal_actions
            .write_action(index, VirtAddr::new(action.sa_sigaction.into()));
        self.signal_actions
            .write_flags(index, action.sa_flags.into());
        self.signal_actions.write_mask(index, action.sa_mask.into());
        Ok(())
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
            actions: space.actions,
            flags: space.flags,
            restorer: space.restorer,
            masks: space.masks,
            thread_mask: space.thread_mask,
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<SignalActionsLayout, F::Output> {
        SignalActionsLayoutF {
            actions: self
                .actions
                .each_ref()
                .map(|sig_action| Cell::struct_ref::<F>(sig_action)),
            flags: self
                .flags
                .each_ref()
                .map(|flag| Cell::struct_ref::<F>(flag)),
            restorer: self.restorer.struct_ref::<F>(),
            masks: self
                .masks
                .each_ref()
                .map(|mask| Cell::struct_ref::<F>(mask)),
            thread_mask: self.thread_mask.struct_ref::<F>(),
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
            actions: core::array::from_fn(|_| Cell::new_with(VirtAddr::new(0))),
            flags: core::array::from_fn(|_| Cell::new_with(0u32)),
            restorer: Cell::new_with(VirtAddr::new(0)),
            masks: core::array::from_fn(|_| Cell::new_with(0u64)),
            thread_mask: Cell::new_with(0u64),
        }
    }
}

impl<M: ManagerClone> Clone for SignalActions<M> {
    fn clone(&self) -> Self {
        SignalActions::<M> {
            actions: self.actions.clone(),
            flags: self.flags.clone(),
            restorer: self.restorer.clone(),
            masks: self.masks.clone(),
            thread_mask: self.thread_mask.clone(),
        }
    }
}

/// Size of the `sigset_t` type in bytes
///
/// As we're building a 64-bit system, the sigset should be 64-bit wide as well.
pub const SIGSET_SIZE: u64 = 8;

#[derive(Debug, Eq, PartialEq)]
enum Disposition {
    Term,
    Core,
    Stop,
}

impl Disposition {
    const fn default(signal: Signal) -> Self {
        match signal {
            Signal::Sigill => Disposition::Core,
            Signal::Sigabrt => Disposition::Core,
            Signal::Sigiot => Disposition::Core,
            Signal::Sigbus => Disposition::Core,
            Signal::Sigfpe => Disposition::Core,
            Signal::Sigkill => Disposition::Term,
            Signal::Sigusr1 => Disposition::Term,
            Signal::Sigsegv => Disposition::Core,
            Signal::Sigusr2 => Disposition::Term,
            Signal::Sigpipe => Disposition::Term,
            Signal::Sigterm => Disposition::Term,
            Signal::Sigstop => Disposition::Stop,
            Signal::Sigsys => Disposition::Core,
        }
    }
}

/// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
#[derive(Debug, Clone, Copy, Eq, FromRepr, PartialEq)]
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

impl TryFrom<u64> for Signal {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(Error::InvalidArgument)
    }
}

/// Linux signal signums in RISC-V, see <https://www.man7.org/linux/man-pages/man7/signal.7.html>
/// The representation of these enums are used for indices into signal action storage.
#[derive(Debug, Clone, Copy, EnumCount, FromRepr)]
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
        if self.0?.is_null() {
            return None;
        }

        self.0.map(|addr| addr.to_machine_address())
    }
}

impl From<u64> for SignalActionPtr {
    fn from(value: u64) -> Self {
        SignalActionPtr(Some(VirtAddr::new(value)))
    }
}

/// The behaviour of `rt_sigprocmask(2)`
///
/// See: <https://github.com/torvalds/linux/blob/32b7144f806e231a3fb619d4ddc5a6bffb731715/include/uapi/asm-generic/signal-defs.h#L72>
///      <https://man7.org/linux/man-pages/man2/rt_sigprocmask.2.html>
#[derive(Debug, Clone, Copy, FromRepr)]
#[repr(u64)]
pub enum SigProcMaskHow {
    Block = 0,
    Unblock = 1,
    SetMask = 2,
}

impl TryFrom<u64> for SigProcMaskHow {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(Error::InvalidArgument)
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

    /// Handle `rt_sigaction` system call.
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
            core.set_signal_action(signal, new_action)
                .map_err(|_| Error::InvalidArgument)?;
        }

        // Return 0 as an indicator of success
        Ok(0)
    }

    /// Handle `rt_sigprocmask` system call.
    ///
    /// See: <https://man7.org/linux/man-pages/man2/rt_sigprocmask.2.html>
    pub(super) fn handle_rt_sigprocmask(
        &mut self,
        core: &mut MachineCoreState<impl MemoryConfig, M>,
        how: SigProcMaskHow,
        old: VirtAddr,
        set: VirtAddr,
        _: SigsetTSizeEightBytes,
    ) -> Result<u64, Error>
    where
        M: ManagerReadWrite,
    {
        let old_mask = core.signal_actions.thread_mask.read();

        if !old.is_null() {
            core.main_memory.write(old.to_machine_address(), old_mask)?;
        }

        if !set.is_null() {
            let mask: u64 = core.main_memory.read(set.to_machine_address())?;
            let mask = match how {
                SigProcMaskHow::Block => old_mask | mask,
                SigProcMaskHow::Unblock => old_mask & !mask,
                SigProcMaskHow::SetMask => mask,
            };

            core.signal_actions.thread_mask.write(mask);
        }

        // Return 0 as an indicator of success
        Ok(0)
    }

    /// Handle `rt_sigreturn` system call.
    /// While `rt_sigreturn` is needed for implementing signal handlers, it should never be called
    /// directly by userspace code. The libc wrapper simply returns an error.
    ///
    /// See: <https://www.man7.org/linux/man-pages/man2/rt_sigreturn.2.html>
    pub(super) fn handle_rt_sigreturn(
        &self,
        core: &mut MachineCoreState<impl MemoryConfig, M>,
    ) -> Result<SystemCallResultExecution, Error>
    where
        M: ManagerReadWrite,
    {
        let pc = core.pop_signal_context().map_err(|_| Error::Fault)?;
        Ok(SystemCallResultExecution {
            result: 0, // indicator of success
            pc_update: ProgramCounterUpdate::Set(pc),
            ..SystemCallResultExecution::default()
        })
    }
}

impl<MC, CPE, M> Pvm<MC, CPE, M>
where
    MC: MemoryConfig,
    CPE: CodePageEntry<MC, M>,
    M: ManagerBase,
{
    /// Writes a small function to call the `rt_sigreturn` system call to a provided address, then
    /// writes the address to [self.signal_actions]. This is used on returning from a signal
    /// handler.
    ///
    /// Returns the address of the start of the next page, the next writeable address.
    ///
    /// In x86 this is a libc function but in RISC-V it is part of Linux's vDSO, a library of small
    /// functions that are dynamically written to memory by the kernel when a process is loaded.
    pub fn write_restorer(&mut self, address: VirtAddr) -> Result<VirtAddr, MachineError>
    where
        M: ManagerReadWrite,
    {
        // Encoding to write RT_SIGRETURN to a7
        // ADDI imm=RT_SIGRETURN, rs1=x0, funct3=0, rd=a7
        const IMM: u32 = RT_SIGRETURN as u32;
        const X0_ENC: u32 = 0b00000;
        const F3_0: u32 = 0b000;
        const A7_ENC: u32 = 0b10001;
        const OP_ADDI: u32 = 0b001_0011;
        const LOAD_SIGRETURN: u32 =
            (IMM << 20) | (X0_ENC << 15) | (F3_0 << 12) | (A7_ENC << 7) | OP_ADDI;

        // Encoding of ECALL instruction
        const ECALL: u32 = 0b1110011;

        const RESTORER_FUNCTION: [u32; 2] = [LOAD_SIGRETURN, ECALL];
        const RESTORER_LENGTH: NonZeroUsize = NonZeroUsize::new(size_of_val(&RESTORER_FUNCTION))
            .expect("size of `RESTORER_FUNCTION` is greater than zero");

        // Ensure the restorer is in its own page
        let address = address
            .align_up(PAGE_SIZE)
            .ok_or(MachineError::MemoryTooSmall)?;

        // Allow the restorer function to be written to main memory
        let (main_memory, mut listener) = self.machine_state.memory_with_listener();
        main_memory.protect_pages(
            address.to_machine_address(),
            RESTORER_LENGTH,
            Permissions::WRITE,
            &mut listener,
        )?;

        // Write the function
        main_memory.write(address.to_machine_address(), RESTORER_FUNCTION)?;

        // Make the restorer page R+X
        main_memory.protect_pages(
            address.to_machine_address(),
            // TODO: RV-561: use u64 everywhere in the PVM
            PAGE_SIZE
                .try_into()
                .expect("`PAGE_SIZE` fits into usize width"),
            Permissions {
                read: true,
                exec: true,
                write: false,
            },
            listener,
        )?;

        let restorer_end = address + RESTORER_LENGTH.get() as u64;

        // Store where the restorer is kept so that the signal handlers can find it
        self.machine_state
            .core
            .signal_actions
            .restorer
            .write(address);

        restorer_end
            .align_up(PAGE_SIZE)
            .ok_or(MachineError::MemoryTooSmall)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::ops::Bound;
    use std::ops::ControlFlow;

    use super::Signal;
    use crate::backend_test;
    use crate::exceptions::Exception;
    use crate::machine_state::memory::M1M;
    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::page_cache::Interpreted;
    use crate::machine_state::page_cache::InterpretedCompiler;
    use crate::machine_state::registers::sp;
    use crate::pvm::Pvm;
    use crate::pvm::linux::VirtAddr;

    backend_test!(test_step_into_handler, F, {
        type MC = M1M;
        type Cpe = Interpreted<MC, Owned>;

        let mut pvm = Pvm::<MC, Cpe, Owned>::new(InterpretedCompiler);

        pvm.machine_state.reset();

        pvm.machine_state.set_all_readable_writeable();

        // Write the initial stack pointer and program counter.
        let stack_top = M1M::TOTAL_BYTES.get() as u64;
        pvm.machine_state.core.hart.xregisters.write(sp, stack_top);
        let init_pc = 10;
        pvm.machine_state.core.hart.pc.write(init_pc);

        // The address of a pseudo handler.
        let handler_address = VirtAddr::new(42);

        // Instruction to return from a function.
        // JALR imm=0, rs1=ra, rd=x0
        const RA_ENC: u32 = 0b00001; // ra(x1)
        const F3_0: u32 = 0b000;
        const X0_ENC: u32 = 0b00000;
        const OP_JALR: u32 = 0b110_0111;

        // Write just an instruction to return to the handler address.
        pvm.machine_state
            .core
            .main_memory
            .write_instruction_unchecked(
                handler_address.to_machine_address(),
                (RA_ENC << 15) | (F3_0 << 12) | (X0_ENC << 7) | OP_JALR,
            )
            .unwrap();

        let restorer_address = VirtAddr::new(84);
        pvm.write_restorer(restorer_address)
            .expect("Failed to write restorer");
        let restorer_address = pvm.machine_state.core.signal_actions.restorer.read();

        // Setup the signal handler
        pvm.machine_state
            .core
            .signal_actions
            .write_action(Signal::Sigill, handler_address);

        // Cause the [`Exception::IllegalInstruction`].
        // `unimp`, an illegal instruction.
        const UNIMPLEMENTED: u32 = 0b_0000;

        pvm.machine_state
            .core
            .main_memory
            .write_instruction_unchecked(init_pc, UNIMPLEMENTED)
            .unwrap();
        let step_result = pvm
            .machine_state
            .step_max_handle::<Infallible>(Bound::Included(1), |_| ControlFlow::Continue(()));
        assert_eq!(step_result.error, None);

        // Check that the program counter is now at the handler.
        assert_eq!(
            pvm.machine_state.core.hart.pc.read(),
            handler_address.to_machine_address()
        );

        pvm.machine_state.step().unwrap();

        // Check that the program counter returned to the restorer address.
        assert_eq!(
            pvm.machine_state.core.hart.pc.read(),
            restorer_address.to_machine_address()
        );
    });

    backend_test!(test_jump_to_restorer, F, {
        type MC = M1M;
        type Cpe = Interpreted<MC, Owned>;

        let mut pvm = Pvm::<MC, Cpe, Owned>::new(InterpretedCompiler);

        pvm.machine_state.reset();

        pvm.machine_state.set_all_readable_writeable();

        // Write the initial stack pointer and program counter.
        let stack_top = M1M::TOTAL_BYTES.get() as u64;
        pvm.machine_state.core.hart.xregisters.write(sp, stack_top);
        let init_pc = 10;
        pvm.machine_state.core.hart.pc.write(init_pc);

        // Push a signal context so the program counter is stored
        pvm.machine_state
            .core
            .push_signal_context(Signal::Sigusr1)
            .expect("Bad signal context");

        // Write and jump to a restorer
        let restorer_address = VirtAddr::new(84);

        pvm.write_restorer(restorer_address)
            .expect("Failed to write restorer");

        let restorer_address = pvm.machine_state.core.signal_actions.restorer.read();
        pvm.machine_state
            .core
            .hart
            .pc
            .write(restorer_address.to_machine_address());
        assert_eq!(
            pvm.machine_state.core.hart.pc.read(),
            restorer_address.to_machine_address()
        );

        let step_result = pvm
            .machine_state
            .step_max_handle::<()>(Bound::Included(100), |_| ControlFlow::Break(()));

        assert_eq!(step_result.error, Some(()));
        assert_eq!(pvm.machine_state.step(), Err(Exception::EnvCall));
    });
}
