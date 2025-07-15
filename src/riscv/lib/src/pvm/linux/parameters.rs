// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use core::num::NonZeroU64;
use std::fmt;

use super::MAIN_THREAD_ID;
use super::error::Error;

/// A type coupling the result of the system call with how the program should continue.
#[derive(Debug, Clone, Copy)]
pub struct SystemCallResultExecution {
    pub result: u64,
    pub control_flow: bool,
}

impl<T: Into<u64>> From<T> for SystemCallResultExecution {
    fn from(value: T) -> Self {
        // The default action is to continue execution after the system call. In cases where the
        // execution should halt, this should be specified.
        SystemCallResultExecution {
            result: value.into(),
            control_flow: true,
        }
    }
}

/// The status of the program upon exit. While the C standard specifies that this should be equal
/// to EXIT_SUCCESS or EXIT_FAILURE, this is rarely enforced, and can be any int - where `0`
/// indicates success.
#[derive(Debug, Clone, Copy)]
pub struct ExitStatus(u64);

impl From<ExitStatus> for u64 {
    fn from(value: ExitStatus) -> Self {
        value.0
    }
}

impl From<u64> for ExitStatus {
    fn from(value: u64) -> Self {
        ExitStatus(value)
    }
}

impl ExitStatus {
    /// Extract the exit code from the status stored in this type
    pub fn exit_code(&self) -> u64 {
        self.0
    }
}

/// Known to be a valid thread ID. As we currently only support single-thread execution, this will
/// be the main thread.
#[derive(Debug, Clone, Copy)]
pub struct MainThreadId;

impl TryFrom<u64> for MainThreadId {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        // We only support exiting the main thread
        if value != MAIN_THREAD_ID {
            return Err(Error::Search);
        }
        Ok(MainThreadId)
    }
}

/// Known to be a valid process ID. As we currently only support one hart, this will be that hart,
/// or zero to represent no hart.
#[derive(Debug, Clone, Copy)]
pub struct ProcessId;

impl TryFrom<u64> for ProcessId {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        // We only support the single hart or a null value
        match value {
            0 | 1 => Ok(ProcessId),
            _ => Err(Error::Search),
        }
    }
}

/// Hard limit on CPU time (s)
pub(crate) const RLIMIT_CPU: u64 = u64::MAX;

/// Hard limit for maximum file size (we don't support writing to a filesystem)
pub(crate) const RLIMIT_FSIZE: u64 = 0;

/// Hard limit for core dumps (we don't support core dumps)
pub(crate) const RLIMIT_CORE: u64 = 0;

/// Hard limit on processes (we only support one)
pub(crate) const RLIMIT_NPROC: u64 = 1;

/// Hard limit on the number of file descriptors that a system call can work with
///
/// We also use this constant to implictly limit how much memory can be associated with a system
/// call. For example, `ppoll` takes a pointer to an array of `struct pollfd`. If we don't limit
/// the length of that array, then we might read an arbitrary amount of memory. This impacts the
/// proof size dramatically as everything read would also be in the proof.
pub const RLIMIT_NOFILE: u64 = 512;

/// Hard limit on the number of flock locks plus fcntl locks. Unused by us (and we only support one hart and no filesystem)
pub(crate) const RLIMIT_LOCKS: u64 = 0;

/// Hard limit on the number of signals for a user ID. Signals are not currently implemented.
pub(crate) const RLIMIT_SIGPENDING: u64 = 0;

/// Hard limit on the number of bytes that may be allocated for message queues. Message queues are not currently implemented.
pub(crate) const RLIMIT_MSGQUEUE: u64 = 0;

/// Hard limit on the niceness of a process. This is inverted and the actual niceness is 20 - the
/// rlimit. We only support one hart
pub(crate) const RLIMIT_NICE: u64 = 20;

/// Hard limit on the real time priority of a process. We only support one hart.
pub(crate) const RLIMIT_RTPRIO: u64 = 0;

/// Hard limit on CPU time without making a blocking system call. We only support one hart.
pub(crate) const RLIMIT_RTTIME: u64 = u64::MAX;

#[derive(Debug)]
pub enum Rlimit {
    // Hard limit on CPU time (s)
    Cpu,
    // Hard limit for maximum file size (we don't support writing to a filesystem)
    Fsize,
    // Hard limit on the size of the data segment
    Data,
    // Hard limit on the size of the process stack
    Stack,
    // Hard limit for core dumps (we don't support core dumps)
    Core,
    // Hard limit on resident set size. Not used.
    Rss,
    // Hard limit on processes (we only support one)
    Nproc,
    // Hard limit on the number of file descriptors that a system call can work with
    Nofile,
    // Hard limit on the memory that may be locked into RAM (B)
    Memlock,
    // Hard limit on the size of a process's virtual memory (B)
    As,
    // Hard limit on the number of flock locks plus fcntl locks
    Locks,
    // Hard limit on the number of signals for a user ID
    Sigpending,
    // Hard limit on the number of bytes that may be allocated for message queues
    Msgqueue,
    // Hard limit on the niceness of a process
    Nice,
    // Hard limit on the real time priority of a process
    Rtprio,
    // Hard limit on CPU time without making a blocking system call. We only support one hart.
    Rttime,
}

// Resource limits for system calls
impl TryFrom<u64> for Rlimit {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let value = value.try_into()?;

        let limit = match value {
            // Hard limit on CPU time (s)
            0 => Rlimit::Cpu,

            // Hard limit for maximum file size (we don't support writing to a filesystem)
            1 => Rlimit::Fsize,

            // Hard limit on the size of the data segment
            2 => Rlimit::Data,

            // Hard limit on the size of the process stack
            3 => Rlimit::Stack,

            // Hard limit for core dumps (we don't support core dumps)
            4 => Rlimit::Core,

            // Hard limit on resident set size. Not used.
            5 => Rlimit::Rss,

            // Hard limit on processes (we only support one)
            6 => Rlimit::Nproc,

            // Hard limit on the number of file descriptors that a system call can work with
            7 => Rlimit::Nofile,

            // Hard limit on the memory that may be locked into RAM (B)
            8 => Rlimit::Memlock,

            // Hard limit on the size of a process's virtual memory (B)
            9 => Rlimit::As,

            // Hard limit on the number of flock locks plus fcntl locks
            10 => Rlimit::Locks,

            // Hard limit on the number of signals for a user ID
            11 => Rlimit::Sigpending,

            // Hard limit on the number of bytes that may be allocated for message queues
            12 => Rlimit::Msgqueue,

            // Hard limit on the niceness of a process
            13 => Rlimit::Nice,

            // Hard limit on the real time priority of a process
            14 => Rlimit::Rtprio,

            // Hard limit on CPU time without making a blocking system call. We only support one hart.
            15 => Rlimit::Rttime,
            _ => return Err(Error::Search),
        };
        Ok(limit)
    }
}

/// A valid size for the cpu set struct.
///
/// see <https://man7.org/linux/man-pages/man3/CPU_SET.3.html>
#[derive(Debug, Clone, Copy)]
pub struct CpuSetSize(pub NonZeroU64);

impl TryFrom<u64> for CpuSetSize {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        /// See: <https://man7.org/linux/man-pages/man2/sched_getaffinity.2.html>
        const MAX_CPU_SET_SIZE: u64 = 1024 / u8::BITS as u64;
        if value > MAX_CPU_SET_SIZE {
            return Err(Error::InvalidArgument);
        }

        Ok(CpuSetSize(
            NonZeroU64::new(value).ok_or(Error::InvalidArgument)?,
        ))
    }
}

/// A size parameter passed to `set_robust_list(2)`
#[derive(Debug, Clone, Copy)]
pub struct RobustListHeadSize;

impl TryFrom<u64> for RobustListHeadSize {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let value: usize = value.try_into()?;

        const ROBUST_LIST_HEAD_SIZE: usize = size_of::<u64>();
        if value != ROBUST_LIST_HEAD_SIZE {
            return Err(Error::InvalidArgument);
        }

        Ok(RobustListHeadSize)
    }
}

/// A valid file descriptor, see write(2)
#[derive(Clone, Copy, Debug)]
pub enum FileDescriptorWriteable {
    StandardOutput,
    StandardError,
}

impl TryFrom<u64> for FileDescriptorWriteable {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        // Check if the file descriptor is valid and can be written to.
        // In our case, it's just standard output (1) and standard error (2).
        match value {
            1 => Ok(FileDescriptorWriteable::StandardOutput),
            2 => Ok(FileDescriptorWriteable::StandardError),
            _ => Err(Error::BadFileDescriptor),
        }
    }
}

/// The number (count) of file descriptors
#[derive(Clone, Copy)]
pub struct FileDescriptorCount(u64);

impl FileDescriptorCount {
    /// Extract the file descriptor count as a [`u64`].
    pub fn count(&self) -> u64 {
        self.0
    }
}

impl fmt::Debug for FileDescriptorCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<u64> for FileDescriptorCount {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        // Enforce a limit on the number of file descriptors to prevent proof-size explosion.
        // This is akin to enforcing RLIMIT_NOFILE in a real system.
        if value > RLIMIT_NOFILE {
            return Err(Error::InvalidArgument);
        }
        Ok(FileDescriptorCount(value))
    }
}

/// Definitely zero
#[derive(Debug, Clone, Copy)]
pub struct Zero;

impl TryFrom<u64> for Zero {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value != 0 {
            return Err(Error::InvalidArgument);
        }

        Ok(Zero)
    }
}

/// Special file descriptor meaning no file descriptor
#[derive(Debug, Clone, Copy)]
pub struct NoFileDescriptor;

impl TryFrom<u64> for NoFileDescriptor {
    type Error = Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value != -1i64 as u64 {
            return Err(Error::BadFileDescriptor);
        }

        Ok(NoFileDescriptor)
    }
}

/// Visibility of a memory mapping
#[derive(Debug, Clone, Copy)]
pub enum Visibility {
    /// Only visible to the current task
    Private,

    // Shared with other tasks
    Shared,
}

/// Backing of a memory mapping
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// Just memory
    None,

    /// File-backend memory
    File,
}

/// Hint on how to interpret the address argument when memory mapping
#[derive(Debug, Clone, Copy)]
pub enum AddressHint {
    /// May ignore the address hint
    Hint,

    /// Fixed address
    Fixed { allow_replace: bool },
}

/// Memory mapping request flags
#[derive(Debug, Clone)]
pub struct Flags {
    /// Visibility of the memory mapping
    pub visibility: Visibility,

    /// Memory backing
    pub backend: Backend,

    /// How to interpret the address hint
    pub addr_hint: AddressHint,
}

impl TryFrom<u64> for Flags {
    type Error = Error;

    fn try_from(mut flags: u64) -> Result<Self, Self::Error> {
        // Check if a bit is set, and clear it if it is
        let mut probe_and_clear = |mask: u64| {
            let r = flags & mask == mask;
            flags &= !mask;
            r
        };

        let visibility = {
            const MAP_SHARED: u64 = 0x1;
            const MAP_PRIVATE: u64 = 0x2;

            let shared = probe_and_clear(MAP_SHARED);

            // `MAP_SHARED_VALIDATE` translates to `MAP_PRIVATE | MAP_SHARED` for some reason. We
            // must make sure that private is requested only when `MAP_SHARED` is not set.
            let private = probe_and_clear(MAP_PRIVATE) && !shared;

            if private {
                Visibility::Private
            } else if shared {
                Visibility::Shared
            } else {
                return Err(Error::InvalidArgument);
            }
        };

        let backend = {
            const MAP_ANON: u64 = 0x20;

            if probe_and_clear(MAP_ANON) {
                Backend::None
            } else {
                Backend::File
            }
        };

        let addr_hint = {
            const MAP_FIXED: u64 = 0x10;
            const MAP_FIXED_NOREPLACE: u64 = 0x100000;

            if probe_and_clear(MAP_FIXED) {
                AddressHint::Fixed {
                    allow_replace: !probe_and_clear(MAP_FIXED_NOREPLACE),
                }
            } else if probe_and_clear(MAP_FIXED_NOREPLACE) {
                AddressHint::Fixed {
                    allow_replace: false,
                }
            } else {
                AddressHint::Hint
            }
        };

        // `MAP_NORESERVE` does nothing for us as we have no swap
        const MAP_NORESERVE: u64 = 0x4000;
        probe_and_clear(MAP_NORESERVE);

        // If there are other bits set, that means we likely don't support them
        if flags != 0 {
            return Err(Error::InvalidArgument);
        }

        Ok(Self {
            visibility,
            backend,
            addr_hint,
        })
    }
}
