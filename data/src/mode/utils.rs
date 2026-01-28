// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Utilities for modal components

use std::borrow::Borrow;
use std::ops::Deref;
use std::panic::resume_unwind;

use perfect_derive::perfect_derive;

use crate::components::atom::AtomMode;
use crate::components::atom::CloneAtomMode;
use crate::components::bytes::BytesMode;
use crate::components::bytes::CloneBytesMode;
use crate::components::data_space::CloneDataSpaceMode;
use crate::components::data_space::DataSpaceMode;

/// Source for a state component, either borrowed or owned
///
/// We use this type to store the source version of the underlying proof data. Merkle tree proofs
/// usually encode the initial state. Hence we need to figure out what parts of the initial state
/// need to go into the proof.
///
/// Normally the source is borrowed from state in [`Normal`] mode. In those cases, it is enough to
/// obtain a reference to the source.
///
/// However, there are cases when no source can be borrowed, because it is owned. During the
/// proof generation, we might create a new state. This can happen when a dynamic state
/// component holds states. E.g. it can grow and therefore create new states. During proof
/// generation we don't differentiate between modes, hence you can't create a state in [`Normal`]
/// mode and then borrow it.
///
/// Wherever newly created states are allowed, the upstream state which allows this to happen is in
/// charge of preventing those states from being included in the Merkle tree proof. In the example
/// above, that would be the growable state component. It needs to differentiate between states that
/// existed before the proof generation started, and those that were created during proof.
///
/// It is possible to override states when we allow state creation. Overriding isn't the same as
/// writing the underlying values from one state to the other as it would override the state
/// that is specific to the mode. This would be caught by producing an invalid Merkle tree proof in
/// our test suite. In such cases, the claimed initial state hash of the Merkle tree would not match
/// that of the actual initial state in [`Normal`] mode.
///
/// Another benefit of allowing states to be created from an owned source is that it simplifies our
/// test suite where we want to instantiate a state using the [`Prove`] mode without a previous
/// state in [`Normal`] mode. This simplifies the API for tests and allows us to write one test and
/// test it against 3 different backends.
///
/// It is possible to simplify this type by only allowing owned states. This has some downsides,
/// as it would require all [`Normal`] states to be cloned when wrapping them into [`Prove`]. This
/// could be costly and unnecessary if the states isn't even used.
///
/// [`Normal`]: crate::mode::Normal
/// [`Prove`]: crate::mode::Prove
#[perfect_derive(Debug, Clone)]
pub enum Source<'a, T, R: ?Sized = T> {
    Borrowed(&'a R),
    Owned(Box<T>),
}

impl<'a, T, R: ?Sized> Source<'a, T, R> {
    /// Construct a [`Source`] that is considered owned.
    pub fn owned(value: T) -> Self {
        Source::Owned(Box::new(value))
    }

    /// Construct a [`Source`] that is borrowed from somewhere else.
    pub fn borrowed(value: &'a R) -> Self {
        Source::Borrowed(value)
    }
}

impl<T: Borrow<R>, R: ?Sized> Deref for Source<'_, T, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        match self {
            Source::Borrowed(value) => value,
            Source::Owned(value) => value.as_ref().borrow(),
        }
    }
}

impl<T: Default, R: ?Sized> Default for Source<'_, T, R> {
    fn default() -> Self {
        Source::Owned(Default::default())
    }
}

/// Panic payload that is raised when a value isn't present when running in [`Verify`] mode
///
/// [`Verify`]: crate::mode::Verify
#[derive(Copy, Clone, Debug, Eq, PartialEq, derive_more::Display, thiserror::Error)]
pub struct NotFound;

/// Indicate that a value isn't present but should be when running in [`Verify`] mode.
///
/// # Safety
///
/// This function must only be called in the implementations of [`Verify`] mode. Calling this
/// function outside of that context is undefined behavior as there may not be any panic handlers
/// such as [`catch_not_found`] installed which can catch the panic and convert it into an
/// error.
///
/// [`Verify`]: crate::mode::Verify
pub unsafe fn not_found() -> ! {
    // We use [`resume_unwind`] over [`panic_any`] to avoid calling the panic hook.
    // XXX: This fails without a message when there is no matching [`catch_not_found`] wrapper.
    resume_unwind(Box::new(NotFound))
}

/// Run the given closure and catch calls to `not_found`.
///
/// If `not_found` is called, this function returns `Err(NotFound)`. Otherwise, it returns `Ok` with
/// the result of the closure.
///
/// [`Verify`]: crate::mode::Verify
pub fn catch_not_found<R, F: FnOnce() -> R + std::panic::UnwindSafe>(f: F) -> Result<R, NotFound> {
    match std::panic::catch_unwind(f) {
        Ok(res) => Ok(res),
        Err(err) => match err.downcast::<NotFound>() {
            Ok(not_found) => Err(*not_found),
            Err(other) => resume_unwind(other),
        },
    }
}

/// Result of catching either a `NotFound` or any other panic
pub enum CaughtNotFoundOrPanic {
    /// `not_found` was called
    NotFound(NotFound),

    /// A panic occurred
    Other(Box<dyn std::any::Any + Send>),
}

/// Like [`catch_not_found`] but also catches other panics.
pub fn catch_not_found_and_more<R, F: FnOnce() -> R + std::panic::UnwindSafe>(
    f: F,
) -> Result<R, CaughtNotFoundOrPanic> {
    match std::panic::catch_unwind(f) {
        Ok(res) => Ok(res),
        Err(err) => Err(match err.downcast::<NotFound>() {
            Ok(not_found) => CaughtNotFoundOrPanic::NotFound(*not_found),
            Err(other) => CaughtNotFoundOrPanic::Other(other),
        }),
    }
}

trait_set::trait_set! {
    /// Mode for all tests
    ///
    /// Each state component comes with a small set of mode-constraining traits. When a component
    /// is used in tests, it is best to mention those traits in this trait alias, so that they are
    /// available to all tests.
    pub trait TestMode =
        AtomMode
        + CloneAtomMode
        + DataSpaceMode
        + CloneDataSpaceMode
        + BytesMode
        + CloneBytesMode;
}

/// Generate a test against all modes.
#[macro_export]
macro_rules! mode_test {
    ($(#[$attr:meta])* $fun_name:ident, $ty_name:ident $(: $ty_bound:path)?, $expr:block) => {
        $(#[$attr])*
        #[test]
        fn $fun_name() {
            fn inner<$ty_name: $crate::mode::utils::TestMode $(+ $ty_bound)?>() {
                $expr
            }

            inner::<$crate::mode::Normal>();
            inner::<$crate::mode::Prove>();
            inner::<$crate::mode::Verify>();
        }
    };
}

/// Assert that the given expression evaluates to the expected result.
#[cfg(test)]
macro_rules! assert_eq_found {
    ( $expr:expr, $result:expr ) => {{
        let result = $crate::mode::utils::catch_not_found(|| $expr);
        assert_eq!(result, Ok($result));
    }};
}

#[cfg(test)]
pub(crate) use assert_eq_found;

/// Assert that the given expression evaluates to a [`NotFound`] error.
#[cfg(test)]
macro_rules! assert_not_found {
    ( $expr:expr ) => {{
        let result =
            $crate::mode::utils::catch_not_found(|| $expr).expect_err("computation should fail");
        assert_eq!(result, $crate::mode::utils::NotFound);
    }};
}

#[cfg(test)]
pub(crate) use assert_not_found;
