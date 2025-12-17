// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>

//! Dispatching of entrypoints under JIT is done via hot-swappable
//! function pointers.
//!
//! This module exposes wrappers for the style of dispatch and compilation that is done.

use std::cell::LazyCell;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::Sender;

use perfect_derive::perfect_derive;

use crate::jit::JIT;
use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::address_to_halfword_index;
#[cfg(test)]
use crate::machine_state::page_cache::dispatch::jit_counters::JitTestCounters;
use crate::machine_state::page_cache::jitted::Jitted;
use crate::machine_state::page_cache::jitted::JittedPage;
use crate::machine_state::page_cache::router::RouterEq;

/// The function signature for dispatching an entrypoint run.
///
/// Internally, this may be interpreted, just-in-time compiled, or do
/// additional work over just execution.
///
/// The function signature is identical to that of the [jit compiler's]
/// outputs.
///
/// [jit compiler's]: JIT
pub type DispatchFn<D, MC> = crate::jit::JitFn<D, MC>;

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
    /// A test-only set of counters used for validating entry and
    /// exit of JIT execution in `jit.rs` unit tests.
    #[cfg(test)]
    pub(crate) jit_counters: JitTestCounters,
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
}

impl<D: DispatchCompiler<MC>, MC: MemoryConfig> Default for DispatchTarget<D, MC> {
    fn default() -> Self {
        Self {
            fun: AtomicUsize::new(Jitted::<D, MC>::run_entrypoint_interpreted as usize),
            remaining_calls: AtomicUsize::new(1000),
            #[cfg(test)]
            jit_counters: JitTestCounters::new(),
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
pub trait DispatchCompiler<MC: MemoryConfig>: Sized {
    /// In some implementations there are shared resources that we don't want to recreate for every
    /// new compiler. These are stored in the compiler context.
    type Context: Default;

    /// A new compiler is created, given a context.
    fn new(context: &Self::Context) -> Self;

    /// Whether compilation should be attempted for the instruction sequence.
    fn should_compile(&self, target: &DispatchTarget<Self, MC>) -> bool;

    /// Compile an instruction sequence, hot-swapping the `run_entrypoint` function contained in `target` in
    /// the process. This could be to an interpreted execution method, and/or jit-compiled
    /// function.
    ///
    /// NB - the hot-swapping of JIT-compiled entrypoints may occur at any time, and is not
    /// guaranteed to be contained within the call-time of this function. (This is true for
    /// outline jit, especially).
    fn compile(target: &JittedPage<Self, MC>, program_counter: Address) -> DispatchFn<Self, MC>;
}

/// JIT compiler for entrypoints that performs compilation inline, in the same thread as execution.
#[derive(Clone)]
pub struct InlineCompiler {
    jit: Rc<LazyCell<RefCell<JIT>>>,
}

impl Default for InlineCompiler {
    fn default() -> Self {
        Self {
            jit: Rc::new(LazyCell::new(|| {
                RefCell::new(JIT::new().expect("InlineCompiler should instantiate its `JIT`"))
            })),
        }
    }
}

/// This lets the router know that it can merge `InlineCompiler` ranges as long as they point to
/// the same underlying JIT instance.
impl RouterEq for InlineCompiler {
    fn router_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.jit, &other.jit)
    }
}

impl<MC: MemoryConfig> DispatchCompiler<MC> for InlineCompiler {
    type Context = ();

    fn new(_: &()) -> Self {
        Self::default()
    }

    #[inline]
    fn should_compile(&self, _target: &DispatchTarget<Self, MC>) -> bool {
        // every entrypoint must be compiled immediately for inline jit, as it's used for testing
        // jit compatibility with interpreted mode.
        true
    }

    fn compile(target: &JittedPage<Self, MC>, program_counter: Address) -> DispatchFn<Self, MC> {
        // The `InlineCompiler` is exclusively used in a single threaded context (compilation is
        // done in the same thread as execution). Therefore, this borrow should never panic as
        // there can be no other attempts to borrow concurrently.
        let mut jit = (**target.compiler.jit).borrow_mut();

        let offset = address_to_halfword_index(program_counter);

        let fun: DispatchFn<Self, MC> = match jit.compile_page(target, offset, program_counter) {
            Some(jitfn) => jitfn,
            None => Jitted::run_entrypoint_not_compiled,
        };

        target.entries[offset].dispatch.set(fun);

        fun
    }
}

/// Unsafe Rust escape hatches
mod internal_corro {
    /// A wrapper to make a value `Send`
    pub(super) struct SendWrapper<T> {
        /// Do not use directly! Use [`Self::as_mut`] instead.
        _no_please_no: T,
    }

    impl<T> SendWrapper<T> {
        /// Wrap a `T` to make it [`Send`]
        pub(super) fn new(t: T) -> Self {
            Self { _no_please_no: t }
        }

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

/// We want all outline compilers to share one background thread. So the context needed to create a
/// new outline compiler is the sender end of a channel pointing to the thread
pub struct OutlineCompilerContext<MC: MemoryConfig> {
    sender: Sender<CompilationRequest<OutlineCompiler<MC>, MC>>,
}

impl<MC: MemoryConfig + Send> Default for OutlineCompilerContext<MC> {
    /// Creating a new `OutlineCompilerContext` spawns the background thread.
    fn default() -> Self {
        let (sender, receiver) = mpsc::channel::<CompilationRequest<OutlineCompiler<MC>, MC>>();

        std::thread::spawn(move || {
            {
                while let Ok(msg) = receiver.recv() {
                    let mut jit_guard = msg
                        .page
                        .compiler
                        ._do_not_use_in_main_thread
                        .lock()
                        .expect("Only this thread locks the JIT");

                    // SAFETY: Only this thread touches this so it is safe to get a mutable
                    // reference to it.
                    let jit = unsafe { jit_guard.as_mut() };

                    if Arc::strong_count(&msg.page) == 1 {
                        // The main thread already dropped the page (we are the only reference)
                        // - possibly because the memory the page corresponds to is now writable.
                        // There's no need to handle this request.
                        continue;
                    }

                    let offset = address_to_halfword_index(msg.program_counter);

                    if let Some(jitfn) = jit.compile_page(&msg.page, offset, msg.program_counter) {
                        let dispatch = &msg.page.entries[offset].dispatch;

                        debug_assert_eq!(
                            dispatch.fun.load(Ordering::Acquire),
                            Jitted::<Self, MC>::run_entrypoint_not_compiled as usize,
                            "Unexpected function pointer in dispatch target"
                        );

                        dispatch.fun.store(jitfn as usize, Ordering::Release);
                    };
                }
            }
        });

        Self { sender }
    }
}

/// JIT compiler for entrypoints that performs compilation in the shared
/// background thread.
#[perfect_derive(Clone)]
pub struct OutlineCompiler<MC: MemoryConfig> {
    // We must not touch the jit from the execution thread. On each compilation request it is
    // passed through to the background thread as part of the [`PageEntry`]. The background thread
    // is allowed to lock the mutex and use the underlying JIT instance.
    _do_not_use_in_main_thread: Arc<LazyLock<Mutex<internal_corro::SendWrapper<JIT>>>>,
    sender: Sender<CompilationRequest<Self, MC>>,
}

/// The router should merge `OutlineCompiler` ranges if they point to the same underlying JIT
/// instance.
impl<MC: MemoryConfig> RouterEq for OutlineCompiler<MC> {
    fn router_eq(&self, other: &Self) -> bool {
        // The two `Arc`s are only 'used' here to check pointer equality, this cannot disturb the
        // other thread.
        Arc::ptr_eq(
            &self._do_not_use_in_main_thread,
            &other._do_not_use_in_main_thread,
        )
    }
}

impl<MC: MemoryConfig + Send> DispatchCompiler<MC> for OutlineCompiler<MC> {
    /// This contains the sender pointing to the shared background thread.
    type Context = OutlineCompilerContext<MC>;

    /// Instantiate a new outline compiler with a fresh JIT instance and a clone of the sender to
    /// the background thread.
    fn new(context: &Self::Context) -> Self {
        let jit: Arc<LazyLock<_>> = Arc::new(LazyLock::new(|| {
            let jit_internal = JIT::new().expect("OutlineCompiler should instantiate its `JIT`");
            Mutex::new(internal_corro::SendWrapper::new(jit_internal))
        }));

        Self {
            _do_not_use_in_main_thread: jit,
            sender: context.sender.clone(),
        }
    }

    fn should_compile(&self, target: &DispatchTarget<Self, MC>) -> bool {
        target.remaining_calls.fetch_sub(1, Ordering::SeqCst) == 1
    }

    fn compile(target: &JittedPage<Self, MC>, program_counter: Address) -> DispatchFn<Self, MC> {
        let fun = Jitted::run_entrypoint_not_compiled;

        let offset = address_to_halfword_index(program_counter);
        target.entries[offset].dispatch.set(fun);

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
        let _ = target.compiler.sender.send(request);

        fun
    }
}

/// Compilation request sent to the background JIT-thread of the [`OutlineCompiler`].
struct CompilationRequest<D: DispatchCompiler<MC>, MC: MemoryConfig> {
    /// Reference to the page containing the entrypoint to compile. This carries the compiler
    /// itself, which the background thread uses to access the JIT instance.
    page: Arc<super::state::PageEntry<Jitted<D, MC>, D>>,
    /// The program counter of the entrypoint we wish to compile.
    program_counter: Address,
}

#[cfg(test)]
pub(crate) mod jit_counters {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use crate::machine_state::memory::MemoryConfig;
    use crate::machine_state::page_cache::DispatchTarget;
    use crate::machine_state::page_cache::dispatch::DispatchCompiler;

    /// Test-only counters for JIT execution.
    #[derive(Debug, Default)]
    pub struct JitTestCounters {
        jit_calls: AtomicUsize,
        budget_check_passes: AtomicUsize,
        fallback_calls: AtomicUsize,
    }

    impl JitTestCounters {
        /// Create a new set of counters, all initialised to zero.
        pub fn new() -> Self {
            Self::default()
        }

        /// Create a new set of counters with specified initial values.
        pub fn with_values(
            jit_calls: usize,
            budget_check_passes: usize,
            fallback_calls: usize,
        ) -> Self {
            Self {
                jit_calls: AtomicUsize::new(jit_calls),
                budget_check_passes: AtomicUsize::new(budget_check_passes),
                fallback_calls: AtomicUsize::new(fallback_calls),
            }
        }

        /// Record a JIT call.
        pub fn record_jit_call(&self) {
            self.jit_calls.fetch_add(1, Ordering::Relaxed);
        }

        /// Record a passed budget check.
        pub fn record_budget_check_pass(&self) {
            self.budget_check_passes.fetch_add(1, Ordering::Relaxed);
        }

        /// Record a fallback to interpreted execution.
        pub fn record_fallback_to_interpreter(&self) {
            self.fallback_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl PartialEq for JitTestCounters {
        fn eq(&self, other: &Self) -> bool {
            self.jit_calls.load(Ordering::Relaxed) == other.jit_calls.load(Ordering::Relaxed)
                && self.budget_check_passes.load(Ordering::Relaxed)
                    == other.budget_check_passes.load(Ordering::Relaxed)
                && self.fallback_calls.load(Ordering::Relaxed)
                    == other.fallback_calls.load(Ordering::Relaxed)
        }
    }

    impl Eq for JitTestCounters {}

    impl Clone for JitTestCounters {
        fn clone(&self) -> Self {
            Self {
                jit_calls: AtomicUsize::new(self.jit_calls.load(Ordering::Relaxed)),
                budget_check_passes: AtomicUsize::new(
                    self.budget_check_passes.load(Ordering::Relaxed),
                ),
                fallback_calls: AtomicUsize::new(self.fallback_calls.load(Ordering::Relaxed)),
            }
        }
    }

    impl<D: DispatchCompiler<MC>, MC: MemoryConfig> DispatchTarget<D, MC> {
        /// Get the number of times this dispatch target has been called for verification in tests.
        pub(crate) fn num_jit_calls(&self) -> usize {
            self.jit_counters.jit_calls.load(Ordering::SeqCst)
        }

        /// Get a reference to the test-only JIT counters.
        pub(crate) fn jit_counters(&self) -> &JitTestCounters {
            &self.jit_counters
        }
    }
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::mode::Normal;

    use super::DispatchTarget;
    use super::InlineCompiler;
    use crate::jit::state_access::ExceptionCode;
    use crate::machine_state::MachineCoreState;
    use crate::machine_state::memory::Address;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::page_cache::dispatch::DispatchCompiler;
    use crate::machine_state::page_cache::jitted::Jitted;
    use crate::machine_state::page_cache::jitted::JittedPage;

    #[test]
    fn test_dispatch_debug_classification() {
        let dispatch = DispatchTarget::<InlineCompiler, M4K>::default();
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

        extern "C" fn compiled_dummy<D: DispatchCompiler<M4K>>(
            _: &JittedPage<D, M4K>,
            _: &mut MachineCoreState<M4K, Normal>,
            _: Address,
            _: usize,
            _: &mut ExceptionCode,
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
