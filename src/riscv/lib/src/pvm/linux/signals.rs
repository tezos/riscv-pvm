// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;
use std::ops::Add;
use std::ops::Sub;

use strum::EnumCount;
use strum::FromRepr;

use super::error::Error;
use crate::machine_state::MachineCoreState;
use crate::machine_state::RISCV_ABI_SP_ALIGNMENT;
use crate::machine_state::csregisters::CSRRepr;
use crate::machine_state::csregisters::CSRegister;
use crate::machine_state::memory::BadMemoryAccess;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::nz;
use crate::machine_state::registers::sp;
use crate::pvm::linux::Address;
use crate::pvm::linux::SupervisorState;
use crate::pvm::linux::VirtAddr;
use crate::pvm::linux::registers::FRegister;
use crate::pvm::linux::registers::FValue;
use crate::pvm::linux::registers::XRegister;
use crate::pvm::linux::registers::XValue;
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

/// Linux `si_code`s, see <https://man7.org/linux/man-pages/man2/sigaction.2.html>
///
/// Values: <https://git.musl-libc.org/cgit/musl/tree/include/signal.h>
///
/// These values are not discrete but are differentiated by the signal they are passed with.
pub(crate) mod si_code {
    /// A signal raised by the kernel
    #[cfg(test)]
    pub(crate) const SI_KERNEL: i32 = 128;

    /// [super::Signal::Sigill] raised by an illegal opcode
    pub(crate) const ILL_ILLOPC: i32 = 1;

    /// [super::Signal::Sigsegv] raised by an access error
    pub(crate) const SEGV_ACCERR: i32 = 2;

    /// [super::Signal::Sigsegv] raised by an out-of-bounds access
    pub(crate) const SEGV_BNDERR: i32 = 3;
}

/// Linux sigaction struct, see <https://man7.org/linux/man-pages/man2/sigaction.2.html>
/// and <https://github.com/torvalds/linux/blob/155a3c003e555a7300d156a5252c004c392ec6b0/include/linux/signal_types.h#L37>
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(Default, PartialEq))]
pub struct LinuxSigAction {
    pub sa_sigaction: VirtAddr,
    pub sa_flags: u32,
    pub sa_mask: u64,
}

/// In tests it's useful to be able to create a sigaction using only this field.
#[cfg(test)]
impl LinuxSigAction {
    pub(crate) fn new(sa_sigaction: VirtAddr) -> Self {
        Self {
            sa_sigaction,
            ..Default::default()
        }
    }
}

/// The kernel struct has padding between the flags and mask that would be used by the restorer if
/// RISC-V's ABI didn't use the vDSO for a restorer.
const SIGRETURN_PADDING: usize = 12;

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
            let offset = 0;

            let sa_sigaction_bits = source.add(offset).cast::<u64>().read();
            let offset = offset + size_of_val(&sa_sigaction_bits);

            let sa_flags_bits = source.add(offset).cast::<u32>().read();
            let offset = offset + size_of_val(&sa_flags_bits);

            let offset = offset + SIGRETURN_PADDING;
            let sa_mask_bits = source.add(offset).cast::<u64>().read();

            Self {
                sa_flags: u32::from_le(sa_flags_bits),
                sa_sigaction: VirtAddr::new(u64::from_le(sa_sigaction_bits)),
                sa_mask: u64::from_le(sa_mask_bits),
            }
        }
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        // SAFETY: The bitwise representation is the same as `read_unaligned` and matches each
        // field.
        unsafe {
            let offset = 0;

            dest.add(offset)
                .cast::<u64>()
                .write(self.sa_sigaction.to_machine_address().to_le());
            let offset = offset + size_of_val(&self.sa_sigaction);

            dest.add(offset).cast::<u32>().write(self.sa_flags.to_le());
            let offset = offset + size_of_val(&self.sa_flags);

            let offset = offset + SIGRETURN_PADDING;
            dest.add(offset).cast::<u64>().write(self.sa_mask.to_le());
        }
    }
}

/// Linux siginfo_t, see <https://man7.org/linux/man-pages/man2/sigaction.2.html>
///
/// This struct only contains the fields that are currently used.
#[repr(C)]
pub struct LinuxSigInfo {
    pub si_signo: Signal,

    // Unused in Linux
    si_errno: i32,
    pub si_code: i32,
}

impl LinuxSigInfo {
    pub fn new(si_signo: Signal, si_code: i32) -> Self {
        Self {
            si_signo,
            si_errno: Default::default(),
            si_code,
        }
    }
}

const ALIGNED_SIGINFO_SIZE: u64 = size_of::<LinuxSigInfo>() as u64 + RISCV_ABI_SP_ALIGNMENT.get()
    - (RISCV_ABI_SP_ALIGNMENT.get() % size_of::<LinuxSigInfo>() as u64);

const ALIGNED_CONTEXT_SIZE: u64 = size_of::<LinuxUContext>() as u64 + RISCV_ABI_SP_ALIGNMENT.get()
    - (RISCV_ABI_SP_ALIGNMENT.get() % size_of::<LinuxUContext>() as u64);

const CONTEXT_STACK_SIZE: u64 = ALIGNED_SIGINFO_SIZE + ALIGNED_CONTEXT_SIZE;

impl Elem for LinuxSigInfo {
    const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(size_of::<LinuxSigInfo>()).unwrap();

    unsafe fn read_unaligned(source: *const u8) -> Self {
        // SAFETY: The bitwise representation is the same as `write_unaligned` and matches each
        // field.
        unsafe { source.cast::<LinuxSigInfo>().read() }
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        // SAFETY: The bitwise representation is the same as `read_unaligned` and matches each
        // field.
        unsafe { dest.cast::<LinuxSigInfo>().write(self) }
    }
}

/// Linux ucontext_t, see <https://man7.org/linux/man-pages/man3/getcontext.3.html>
///
/// This struct only contains the fields that are currently used.
#[repr(C)]
pub(crate) struct LinuxUContext {
    uc_sigmask: u64,
    uc_mcontext: MachineContext,
}

impl Elem for LinuxUContext {
    const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(size_of::<LinuxUContext>()).unwrap();

    unsafe fn read_unaligned(source: *const u8) -> Self {
        // SAFETY: The bitwise representation is the same as `write_unaligned` and matches each
        // field.
        unsafe { source.cast::<LinuxUContext>().read() }
    }

    unsafe fn write_unaligned(self, dest: *mut u8) {
        // SAFETY: The bitwise representation is the same as `read_unaligned` and matches each
        // field.
        unsafe { dest.cast::<LinuxUContext>().write(self) }
    }
}

struct CSRegisterContext {
    fflags: CSRRepr,
    frm: CSRRepr,
}

struct MachineContext {
    pc: u64,
    xregisters: [XValue; 31],
    fregisters: [FValue; 32],
    csregisters: CSRegisterContext,
}

/// Pointers to the locations on the stack for the restorer, the context of []LinuxUContext] and
/// the siginfo of [LinuxSigInfo].
pub struct StackPointers {
    restorer: VirtAddr,
    context: VirtAddr,
    siginfo: VirtAddr,
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
    pub restorer: Cell<VirtAddr, M>,
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
    pub fn dispatch_signal(&mut self, siginfo: LinuxSigInfo) -> Result<(), SignalError> {
        let signal: Signal = siginfo.si_signo;
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

        let stack_pointers = self.push_signal_context(siginfo)?;

        self.hart
            .xregisters
            .write_nz(nz::ra, stack_pointers.restorer.to_machine_address());

        // Write handler arguments
        // Signal number
        self.hart.xregisters.write_nz(nz::a0, signal as u64);

        if self.signal_actions.sa_siginfo(signal) {
            // Pointer to siginfo_t
            self.hart
                .xregisters
                .write_nz(nz::a1, stack_pointers.siginfo.to_machine_address());

            // Pointer to ucontext_t
            self.hart
                .xregisters
                .write_nz(nz::a2, stack_pointers.context.to_machine_address());
        }

        // Update the program counter to the handler
        self.hart.pc.write(handler.to_machine_address());
        Ok(())
    }

    /// Pushes the context needed to resume after handling a signal
    pub fn push_signal_context(
        &mut self,
        siginfo: LinuxSigInfo,
    ) -> Result<StackPointers, SignalError> {
        let signal: Signal = siginfo.si_signo;

        let signal_index: SignalIndex = signal.into();

        let mask = self.signal_actions.read_mask(signal_index);
        let pc = self.hart.pc.read();

        let restorer = self.signal_actions.restorer.read();

        // SAFETY: x is bounded between 0..=30, which fits in both [u8] and [XRegister]
        //         f is bounded between 0..=31, which fits in both [u8] and [FRegister]
        let mcontext = MachineContext {
            pc,
            xregisters: std::array::from_fn(|x| {
                self.hart
                    .xregisters
                    .read(unsafe { std::mem::transmute::<u8, XRegister>(x as u8) })
            }),
            fregisters: std::array::from_fn(|f| {
                self.hart
                    .fregisters
                    .read(unsafe { std::mem::transmute::<u8, FRegister>(f as u8) })
            }),
            csregisters: CSRegisterContext {
                fflags: self.hart.csregisters.read(CSRegister::fflags),
                frm: self.hart.csregisters.read(CSRegister::frm),
            },
        };

        let context = LinuxUContext {
            uc_sigmask: mask,
            uc_mcontext: mcontext,
        };

        let prev_stack_pointer = self.hart.xregisters.read(sp);
        let stack_pointer = VirtAddr::new(prev_stack_pointer)
            .sub(CONTEXT_STACK_SIZE)
            .align_down(RISCV_ABI_SP_ALIGNMENT);

        self.hart
            .xregisters
            .write(sp, stack_pointer.to_machine_address());

        let stack_pointers = StackPointers {
            restorer,
            context: stack_pointer,
            siginfo: stack_pointer.add(ALIGNED_CONTEXT_SIZE),
        };

        self.main_memory
            .write::<LinuxUContext>(stack_pointers.context.to_machine_address(), context)?;

        self.main_memory
            .write::<LinuxSigInfo>(stack_pointers.siginfo.to_machine_address(), siginfo)?;

        Ok(stack_pointers)
    }

    /// Pops the context needed to resume after handling a signal
    pub fn pop_signal_context(&mut self) -> Result<Address, SignalError> {
        let stack_pointer = VirtAddr::new(self.hart.xregisters.read(sp));

        let siginfo = self
            .main_memory
            .read::<LinuxSigInfo>(stack_pointer.add(ALIGNED_CONTEXT_SIZE).to_machine_address())?;

        let context = self
            .main_memory
            .read::<LinuxUContext>(stack_pointer.to_machine_address())?;

        // SAFETY: i is bounded between 0..=30, which fits in [u8]
        for (i, xvalue) in context.uc_mcontext.xregisters.iter().enumerate() {
            unsafe {
                self.hart
                    .xregisters
                    .write(std::mem::transmute::<u8, XRegister>(i as u8), *xvalue);
            }
        }

        // SAFETY: i is bounded between 0..=31, which fits in both [u8] and [FRegister]
        for (i, fvalue) in context.uc_mcontext.fregisters.iter().enumerate() {
            unsafe {
                self.hart
                    .fregisters
                    .write(std::mem::transmute::<u8, FRegister>(i as u8), *fvalue);
            }
        }

        self.hart
            .csregisters
            .write(CSRegister::fflags, context.uc_mcontext.csregisters.fflags);
        self.hart
            .csregisters
            .write(CSRegister::frm, context.uc_mcontext.csregisters.frm);

        let signal: Signal = siginfo.si_signo;

        self.signal_actions.write_mask(signal, context.uc_sigmask);

        // TODO RV-734 Restore the alternative stack

        let prev_stack_pointer = self.hart.xregisters.read(sp);
        let stack_pointer = VirtAddr::new(prev_stack_pointer)
            .add(CONTEXT_STACK_SIZE)
            .align_up(RISCV_ABI_SP_ALIGNMENT)
            .ok_or(SignalError::MisalignedStackPointer)?;

        self.hart
            .xregisters
            .write(sp, stack_pointer.to_machine_address());

        Ok(context.uc_mcontext.pc)
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
            sa_sigaction,
            sa_flags,
            sa_mask,
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
        self.signal_actions.write_action(index, action.sa_sigaction);
        self.signal_actions.write_flags(index, action.sa_flags);
        self.signal_actions.write_mask(index, action.sa_mask);
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
#[repr(i32)]
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

impl TryFrom<i32> for Signal {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(Error::InvalidArgument)
    }
}

impl TryFrom<u64> for Signal {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_repr(value.try_into()?).ok_or(Error::InvalidArgument)
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
    ) -> Result<u64, Error>
    where
        M: ManagerReadWrite,
    {
        let pc = core.pop_signal_context().map_err(|_| Error::Fault)?;
        core.hart.pc.write(pc);
        // Return 0 as an indicator of success
        Ok(0)
    }
}
