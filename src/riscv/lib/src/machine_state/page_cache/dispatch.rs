// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>

//! Dispatching of entrypoints under JIT is done via hot-swappable
//! function pointers.
//!
//! This module exposes wrappers for the style of dispatch and compilation that is done.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::Sender;

use super::INSTRUCTION_ENTRIES;
use crate::jit::JIT;
use crate::jit::JitFn;
use crate::jit::state_access::ExceptionCode;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::jitted::Jitted;
use crate::machine_state::page_cache::jitted::JittedPage;
use crate::state_backend::owned_backend::Owned;

/// The function signature for dispatching an entrypoint run.
///
/// Internally, this may be interpreted, just-in-time compiled, or do
/// additional work over just execution.
///
/// The first and last parameters must be thin-references, for ABI-compatibility reasons.
pub type DispatchFn<D, MC> = unsafe extern "C" fn(
    &JittedPage<D, MC>,
    &mut MachineCoreState<MC, Owned>,
    Address,
    usize,
    &mut ExceptionCode,
    &mut D,
) -> usize;

/// Dispatch target that wraps a [`DispatchFn`].
///
/// This is the target used for compilation - see [`DispatchCompiler::compile`].
pub struct DispatchTarget<D, MC> {
    /// Function pointer stored as an atomic usize.
    ///
    /// This will allow the `fun` to be updated from a background thread.
    /// See <https://doc.rust-lang.org/std/primitive.fn.html#casting-to-and-from-integers> for
    /// considerations taken whilst converting pointer <--> usize.
    fun: AtomicUsize,
    remaining_calls: AtomicUsize,
    /// A test only counter for the number of times this entrypoint has been called.
    ///
    /// This is used in the `jit.rs` tests to ensure that when running a scenario over InlineJit,
    /// we *do* actually run the entrypoint. Previously this check did not exist, and a change resulted
    /// in the tests using the interpreted fallback mechanism instead for certain classes of test, rather
    /// than actually running the JIT-compiled function as intended.
    #[cfg(test)]
    call_counter: AtomicUsize,
    _pd: PhantomData<(D, MC)>,
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> DispatchTarget<D, MC> {
    /// Set the dispatch target to use the given `run_entrypoint` function.
    pub fn set(&self, fun: DispatchFn<D, MC>) {
        // casting a function pointer as usize is ok to do.
        let fun = fun as usize;

        // store using Release ordering - any subsequent loading with Acquire will see the new ptr.
        self.fun.store(fun, Ordering::Release);
    }

    /// Get the dispatch target's current `run_entrypoint` function.
    pub fn get(&self) -> DispatchFn<D, MC> {
        // load using Acquire ordering - so that it will see the previous store which was with
        // Release.
        let fun = self.fun.load(Ordering::Acquire);

        // to avoid problematic integer -> pointer conversion, we must cast it as a pointer first.
        let fun = fun as *const ();

        // Safety: the pointer is indeed a function pointer with an ABI matching `DispatchFn`.
        unsafe { std::mem::transmute::<*const (), DispatchFn<D, MC>>(fun) }
    }

    /// Increase the call counter to keep track of how often it was dispatched for verification in tests.
    #[cfg(test)]
    pub(crate) fn record_called(&self) {
        self.call_counter.fetch_add(1, Ordering::SeqCst);
    }

    /// Get the number of times this dispatch target has been called for verification in tests.
    #[cfg(test)]
    pub(crate) fn called_times(&self) -> usize {
        self.call_counter.load(Ordering::SeqCst)
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> Default for DispatchTarget<D, MC> {
    fn default() -> Self {
        Self {
            fun: AtomicUsize::new(Jitted::<D, MC>::run_entrypoint_interpreted as usize),
            remaining_calls: AtomicUsize::new(1000),
            #[cfg(test)]
            call_counter: AtomicUsize::new(0),
            _pd: PhantomData,
        }
    }
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> std::fmt::Debug for DispatchTarget<D, MC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fun = self.get();

        #[derive(Debug)]
        enum Status {
            Interpreted,
            NotCompiled,
            Compiled,
        }

        let status = if fun as usize == Jitted::<D, MC>::run_entrypoint_interpreted as usize {
            Status::Interpreted
        } else if fun as usize == Jitted::<D, MC>::run_entrypoint_not_compiled as usize {
            Status::NotCompiled
        } else {
            Status::Compiled
        };

        f.debug_struct("DispatchTarget")
            .field("status", &status)
            .field("fun", &fun)
            .field("remaining_calls", &self.remaining_calls)
            .finish()
    }
}

/// A compiler that can JIT-compile sequences of instructions, and hot-swap the execution of
/// said entrypoint in the given dispatch target.
pub trait DispatchCompiler<MC: MemoryConfig>: Default + Sized {
    /// Whether compilation should be attempted for the instruction sequence.
    fn should_compile(&self, target: &DispatchTarget<Self, MC>) -> bool;

    /// Compile an instruction sequence, hot-swapping the `run_entrypoint` function contained in `target` in
    /// the process. This could be to an interpreted execution method, and/or jit-compiled
    /// function.
    ///
    /// NB - the hot-swapping of JIT-compiled entrypoints may occur at any time, and is not
    /// guaranteed to be contained within the call-time of this function. (This is true for
    /// outline jit, especially).
    fn compile(
        &mut self,
        target: &JittedPage<Self, MC>,
        program_counter: Address,
    ) -> DispatchFn<Self, MC>;
}

/// JIT compiler for entrypoints that performs compilation inline, in the same thread as execution.
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
        // every entrypoint must be compiled immediately for inline jit, as it's used for testing
        // jit compatibility with interpreted mode.
        true
    }

    fn compile(
        &mut self,
        target: &JittedPage<Self, MC>,
        program_counter: Address,
    ) -> DispatchFn<Self, MC> {
        let instr = Jitted::compilation_request_instructions(target.as_ref(), program_counter);

        let fun = match self.jit.compile(&instr, program_counter) {
            Some(jitfn) => {
                // Safety: the two function signatures are identical, apart from the first and
                // last parameters. These are both thin-pointers, and ignored by the JitFn.
                //
                // It's therefore safe to cast this function pointer to an identical ABI, where
                // this first and last parameter are thin-references to any value. This is the
                // case for both `Jitted` and `Jitted::BlockBuilder` which are both Sized.
                //
                // See <https://doc.rust-lang.org/std/primitive.fn.html#abi-compatibility> for more
                // information on ABI compatibility.
                unsafe { std::mem::transmute::<JitFn<MC>, DispatchFn<Self, MC>>(jitfn) }
            }
            None => Jitted::run_entrypoint_not_compiled,
        };

        let offset = memory::address_to_page_offset(program_counter) >> 1;
        target[offset].dispatch.set(fun);

        fun
    }
}

/// Unsafe Rust escape hatches
mod internal_corro {
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

/// JIT compiler for entrypoints that performs compilation in a
/// background thread.
pub struct OutlineCompiler<MC: MemoryConfig> {
    // We will not touch the jit from the execution thread, however we must maintain
    // a reference to it - to ensure it is not dropped before we are done with execution,
    // even if the background compilation thread panics.
    _do_not_use_this_is_for_drop_only: Arc<Mutex<internal_corro::SendWrapper<JIT<MC>>>>,
    sender: Sender<CompilationRequest<Self, MC>>,
}

impl<MC: MemoryConfig + Send> OutlineCompiler<MC> {
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
                    if Arc::strong_count(&msg.page) == 1 {
                        // The main thread already dropped the page (we are the only reference)
                        // - possibly because the memory the page corresponds to is now writable.
                        // There's no need to handle this request.
                        continue;
                    }

                    let instr = Jitted::compilation_request_instructions(
                        msg.page.as_ref(),
                        msg.program_counter,
                    );

                    if let Some(jitfn) = jit.compile(&instr, msg.program_counter) {
                        let offset = memory::address_to_page_offset(msg.program_counter) >> 1;
                        let dispatch = &msg.page[offset].dispatch;

                        debug_assert_eq!(
                            dispatch.fun.load(Ordering::Acquire),
                            Jitted::<Self, MC>::run_entrypoint_not_compiled as usize,
                            "Unexpected function pointer in dispatch target"
                        );

                        // Safety: this function will be retrieved as a DispatchFn, rather than a
                        // JitFn. The two function signatures are identical, apart from the first and
                        // last parameters. These are both thin-pointers, and ignored by the JitFn.
                        //
                        // It's therefore safe to cast this function pointer to an identical ABI, where
                        // this first and last parameter are thin-references to any value. This is the
                        // case for both `Jitted` and `Jitted::BlockBuilder` which are both Sized.
                        //
                        // See <https://doc.rust-lang.org/std/primitive.fn.html#abi-compatibility> for more
                        // information on ABI compatibility.
                        dispatch.fun.store(jitfn as usize, Ordering::Release);
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
        target.remaining_calls.fetch_sub(1, Ordering::SeqCst) == 1
    }

    fn compile(
        &mut self,
        target: &JittedPage<Self, MC>,
        program_counter: Address,
    ) -> DispatchFn<Self, MC> {
        let fun = Jitted::run_entrypoint_not_compiled;

        let offset = memory::address_to_page_offset(program_counter) >> 1;
        target[offset].dispatch.set(fun);

        let request = CompilationRequest {
            page: target.clone(),
            program_counter,
        };

        // This will always succeed, unless the compilation thread has panicked
        // (as this would result in the receiving end of the channel being closed).
        //
        // If it has, execution must still continue - but everything will fallback
        // to interpreted mode.
        //
        // NB - any entrypoints already JIT compiled are safe to keep calling, as the
        // data behind the mutex (the JIT) is kept alive for as long as we maintain
        // our reference to it, despite the lock itself being poisoned.
        let _ = self.sender.send(request);

        fun
    }
}

/// Compilation request sent to the background JIT-thread of the [`OutlineCompiler`].
struct CompilationRequest<D: DispatchCompiler<MC>, MC: MemoryConfig> {
    /// Reference to the page containing the entrypoint to compile.
    page: Arc<[Jitted<D, MC>; INSTRUCTION_ENTRIES]>,
    /// The program counter of the entrypoint we wish to compile.
    program_counter: Address,
}

#[cfg(test)]
mod tests {
    use super::DispatchTarget;
    use super::InlineCompiler;
    use crate::jit::state_access::ExceptionCode;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::memory::Address;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::page_cache::dispatch::DispatchCompiler;
    use crate::machine_state::page_cache::jitted::Jitted;
    use crate::machine_state::page_cache::jitted::JittedPage;
    use crate::state_backend::owned_backend::Owned;

    #[test]
    fn test_dispatch_debug_classification() {
        let dispatch = DispatchTarget::<InlineCompiler<_>, M4K>::default();
        let format = format!("{dispatch:?}");

        assert!(
            format.contains("status: Interpreted"),
            "unexpected formatting \"{format}\""
        );

        dispatch.set(Jitted::<_, _>::run_entrypoint_not_compiled);
        let format = format!("{dispatch:?}");

        assert!(
            format.contains("status: NotCompiled"),
            "unexpected formatting \"{format}\""
        );

        unsafe extern "C" fn compiled_dummy<D: DispatchCompiler<M4K>>(
            _: &JittedPage<D, M4K>,
            _: &mut MachineCoreState<M4K, Owned>,
            _: Address,
            _: usize,
            _: &mut ExceptionCode,
            _: &mut D,
        ) -> usize {
            0
        }

        dispatch.set(compiled_dummy);
        let format = format!("{dispatch:?}");

        assert!(
            format.contains("status: Compiled"),
            "unexpected formatting \"{format}\""
        );
    }
}
