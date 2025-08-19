// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>

//! Dispatching of blocks under JIT is done via hot-swappable
//! function pointers.
//!
//! This module exposes wrappers for the style of dispatch and compilation that is done.
//!
//! Currently, this is only 'inline' jit, but will soon be expanded to 'outline' jit also;
//! where 'outline' means any JIT compilation occurs in a separate thread.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::Sender;

use super::CacheEntry;
use super::OFFSET_MASK;
use crate::jit::JIT;
use crate::jit::JitFn;
use crate::jit::state_access::ExceptionCode;
use crate::machine_state::MachineCoreState;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::owned_backend::Owned;

const THRESHOLD: usize = 700;

/// The function signature for dispatching a block run.
///
/// Internally, this may be interpreted, just-in-time compiled, or do
/// additional work over just execution.
///
/// The first and last parameters must be thin-references, for ABI-compatability reasons.
pub type DispatchFn<D, MC> = unsafe extern "C" fn(
    &Arc<[CacheEntry<MC, DispatchTarget<D, MC>, Owned>; 2048]>,
    &mut MachineCoreState<MC, Owned>,
    Address,
    usize,
    &mut ExceptionCode,
    &mut D,
) -> usize;

/// Dispatch target that wraps a [`DispatchFn`].
///
/// This is the target used for compilation - see [`DispatchCompiler::compile`].
pub struct DispatchTarget<D: DispatchCompiler<MC>, MC: MemoryConfig> {
    /// Function pointer stored as an atomic usize.
    ///
    /// This will allow the `fun` to be updated from a background thread.
    /// See <https://doc.rust-lang.org/std/primitive.fn.html#casting-to-and-from-integers> for
    /// considerations taken whilst converting pointer <--> usize.
    fun: AtomicUsize,
    remaining_calls: internal_corro::UnsafeSyncCell<usize>,
    _pd: PhantomData<(D, MC)>,
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> DispatchTarget<D, MC> {
    /// Reset the dispatch target to the interpreted dispatch mechanism.
    pub fn reset(&mut self) {
        // in resetting the block, we must allocated a new Arc<AtomicUsize>.
        //
        // If we just reset the current arc, outline jit could update it from the background thread
        // after reset it - meaning a reset/under construction block could now have a jitted function for
        // a completely different set of instructions.
        self.fun = AtomicUsize::new(CacheEntry::<MC, Self, Owned>::run_block_interpreted as usize);

        unsafe { self.remaining_calls.get().write(THRESHOLD) };
    }

    /// Set the dispatch target to use the given `block_run` function.
    pub fn set(&self, fun: DispatchFn<D, MC>) {
        // casting a function pointer as usize is ok to do.
        let fun = fun as usize;

        // store using Release ordering - any subsequent loading with Acquire will see the new ptr.
        self.fun.store(fun, Ordering::Release);
    }

    /// Get the dispatch target's current `block_run` function.
    pub fn get(&self) -> DispatchFn<D, MC> {
        // load using Acquire ordering - so that it will see the previous store which was with
        // Release.
        let fun = self.fun.load(Ordering::Acquire);

        // to avoid problematic integer -> pointer conversion, we must cast it as a pointer first.
        let fun = fun as *const ();

        // Safety: the pointer is indeed a function pointer with an ABI matching `DispatchFn`.
        unsafe { std::mem::transmute::<*const (), DispatchFn<D, MC>>(fun) }
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> Default for DispatchTarget<D, MC> {
    fn default() -> Self {
        Self {
            fun: AtomicUsize::new(CacheEntry::<MC, Self, Owned>::run_block_interpreted as usize),
            remaining_calls: internal_corro::UnsafeSyncCell::new(THRESHOLD),
            _pd: PhantomData,
        }
    }
}

/// A compiler that can JIT-compile blocks of instructions, and hot-swap the execution of
/// said block in the given dispatch target.
pub trait DispatchCompiler<MC: MemoryConfig>: Default + Sized {
    /// Whether compilation should be attempted for the block.
    fn should_compile(&self, target: &DispatchTarget<Self, MC>) -> bool;

    /// Compile a block, hot-swapping the `run_block` function contained in `target` in
    /// the process. This could be to an interpreted execution method, and/or jit-compiled
    /// function.
    ///
    /// NB - the hot-swapping of JIT-compiled blocks may occur at any time, and is not
    /// guaranteed to be contained within the call-time of this function. (This is true for
    /// outline jit, especially).
    fn compile(
        &mut self,
        entries: Arc<[CacheEntry<MC, DispatchTarget<Self, MC>, Owned>; 2048]>,
        program_counter: Address,
    ) -> DispatchFn<Self, MC>;
}

/// JIT compiler for blocks that performs compilation inline, in the same thread as execution.
pub struct InlineCompiler<MC: MemoryConfig> {
    jit: JIT<MC>,
}

impl<MC: MemoryConfig> Default for InlineCompiler<MC> {
    fn default() -> Self {
        Self {
            jit: JIT::default(),
        }
    }
}

impl<MC: MemoryConfig> DispatchCompiler<MC> for InlineCompiler<MC> {
    #[inline]
    fn should_compile(&self, _target: &DispatchTarget<Self, MC>) -> bool {
        // every block must be compiled immediately for inline jit, as it's used for testing
        // jit compatibility with interpreted mode.
        true
    }

    fn compile(
        &mut self,
        entries: Arc<[CacheEntry<MC, DispatchTarget<Self, MC>, Owned>; 2048]>,
        program_counter: Address,
    ) -> DispatchFn<Self, MC> {
        let mut instructions = Vec::with_capacity(40);
        let offset = ((program_counter & OFFSET_MASK) >> 1) as usize;
        let mut index = offset;

        while index < 2048 && instructions.len() < instructions.capacity() {
            let i = entries[index].instr;
            index += i.width() as usize >> 1;
            instructions.push(i);
        }

        let fun = match self.jit.compile(&instructions, program_counter) {
            Some(jitfn) => {
                // Safety: the two function signatures are identical, apart from the first and
                // last parameters. These are both thin-pointers, and ignored by the JitFn.
                //
                // It's therefore safe to cast this function pointer to an identical ABI, where
                // this first and last parameter are thin-references to any value. This is the
                // case for both `CacheEntry` and `Jitted::BlockBuilder` which are both Sized.
                //
                // See <https://doc.rust-lang.org/std/primitive.fn.html#abi-compatibility> for more
                // information on ABI compatability.
                unsafe { std::mem::transmute::<JitFn<MC>, DispatchFn<Self, MC>>(jitfn) }
            }
            None => CacheEntry::run_block_not_compiled,
        };

        entries[offset].block_run.set(fun);

        fun
    }
}

/// Unsafe Rust escape hatches
mod internal_corro {

    #[derive(Default)]
    pub(super) struct UnsafeSyncCell<T> {
        cell: std::cell::UnsafeCell<T>,
    }

    unsafe impl<T> Sync for UnsafeSyncCell<T> {}

    impl<T> UnsafeSyncCell<T> {
        pub unsafe fn get(&self) -> *mut T {
            self.cell.get()
        }

        pub fn new(v: T) -> Self {
            Self {
                cell: std::cell::UnsafeCell::new(v),
            }
        }
    }

    /// A wrapper to make a value `Send`
    #[derive(Default)]
    pub(super) struct SendWrapper<T> {
        /// Do not use directly! Use [`Self::as_mut`] instead.
        _no_please_no: T,
    }

    impl<T> SendWrapper<T> {
        /// Obtain a mutable reference to the inner value.
        ///
        /// # Safety
        ///
        /// Ensure that this is only called from the thread that owns the [`SendWrapper`]. There must
        /// not be any other thread that uses its reference to a [`SendWrapper`].
        pub(super) unsafe fn as_mut(&mut self) -> &mut T {
            &mut self._no_please_no
        }
    }

    // We know that the main thread does not actually use the JIT compilation state. The only thing it
    // may do is drop it when it is the only owner left.
    unsafe impl<T> Send for SendWrapper<T> {}
}

/// JIT compiler for blocks that performs compilation in a
/// background thread.
pub struct OutlineCompiler<MC: MemoryConfig> {
    // We will not touch the jit from the execution thread, however we must maintain
    // a reference to it - to ensure it is not dropped before we are done with execution,
    // even if the background compilation thread panics.
    _do_not_use_this_is_for_drop_only: Arc<Mutex<internal_corro::SendWrapper<JIT<MC>>>>,
    sender: Sender<CompilationRequest<MC, Self>>,
}

impl<MC: MemoryConfig + Send + Sync> OutlineCompiler<MC> {
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        let jit: Arc<Mutex<internal_corro::SendWrapper<JIT<MC>>>> = Default::default();

        let compiler = Self {
            _do_not_use_this_is_for_drop_only: jit.clone(),
            sender,
        };

        std::thread::spawn(move || {
            {
                let mut jit_guard = jit.lock().expect("Only this thread locks the JIT");

                // SAFETY: We are the only thread that may access the JIT compilation state.
                let jit = unsafe { jit_guard.as_mut() };

                while let Ok(msg) = receiver.recv() {
                    let instr = msg.instr();
                    let fun = &msg.entries[((msg.program_counter & OFFSET_MASK) >> 1) as usize]
                        .block_run
                        .fun;

                    if let Some(jitfn) = jit.compile(&instr, msg.program_counter) {
                        debug_assert_eq!(
                            fun.load(Ordering::Acquire),
                            CacheEntry::<MC, DispatchTarget<Self, MC>, Owned>::run_block_not_compiled as usize,
                            "Unexpected function pointer in dispatch target"
                        );

                        // Safety: this function will be retrieved as a DispatchFn, rather than a
                        // JitFn. The two function signatures are identical, apart from the first and
                        // last parameters. These are both thin-pointers, and ignored by the JitFn.
                        //
                        // It's therefore safe to cast this function pointer to an identical ABI, where
                        // this first and last parameter are thin-references to any value. This is the
                        // case for both `CacheEntry` and `Jitted::BlockBuilder` which are both Sized.
                        //
                        // See <https://doc.rust-lang.org/std/primitive.fn.html#abi-compatibility> for more
                        // information on ABI compatability.
                        fun.store(jitfn as usize, Ordering::Release);
                    };
                }
            }
            // because we used blocking recv with an asynchronous channel, this only fails when the
            // other end of the channel has been dropped.
            //
            // This means the BlockBuilder has been dropped - and thus execution has stopped.
            // We are therefore safe to drop the JIT.
        });

        compiler
    }
}

impl<MC: MemoryConfig + Send> Default for OutlineCompiler<MC> {
    fn default() -> Self {
        Self::new()
    }
}

impl<MC: MemoryConfig + Send> DispatchCompiler<MC> for OutlineCompiler<MC> {
    fn should_compile(&self, target: &DispatchTarget<Self, MC>) -> bool {
        unsafe {
            let x = target.remaining_calls.get();
            match *x {
                0 => true,
                a => {
                    //SAFETY: `remaining_calls` (a usize) can only be a positive, non-zero integer
                    x.write(a.unchecked_sub(1));
                    false
                }
            }
        }
    }

    fn compile(
        &mut self,
        entries: Arc<[CacheEntry<MC, DispatchTarget<Self, MC>, Owned>; 2048]>,
        program_counter: Address,
    ) -> DispatchFn<Self, MC> {
        let fun = CacheEntry::run_block_not_compiled;

        let offset = ((program_counter & OFFSET_MASK) >> 1) as usize;
        entries[offset].block_run.set(fun);

        // Single instruction blocks don't perform as well when compiled
        //if instr.len() <= 1 {
        //    return fun;
        //}

        let request = CompilationRequest {
            entries,
            program_counter,
        };

        // This will always succeed, unless the compilation thread has panicked
        // (as this would result in the receiving end of the channel being closed).
        //
        // If it has, execution must still continue - but everything will fallback
        // to interpreted mode.
        //
        // NB - any blocks already JIT compiled are safe to keep calling, as the
        // data behind the mutex (the JIT) is kept alive for as long as we maintain
        // our reference to it, despite the lock itself being poisoned.
        let _ = self.sender.send(request);

        fun
    }
}

struct CompilationRequest<MC: MemoryConfig, D: DispatchCompiler<MC>> {
    entries: Arc<[CacheEntry<MC, DispatchTarget<D, MC>, Owned>; 2048]>,
    program_counter: Address,
}

impl<MC: MemoryConfig, D: DispatchCompiler<MC>> CompilationRequest<MC, D> {
    fn instr(&self) -> Vec<Instruction> {
        let mut instructions = Vec::with_capacity(40);
        let mut index = ((self.program_counter & OFFSET_MASK) >> 1) as usize;

        while index < 2048 && instructions.len() < instructions.capacity() {
            let i = self.entries[index].instr;
            index += i.width() as usize >> 1;
            instructions.push(i);
        }

        instructions
    }
}
