// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Wrapper for C-style error handling with out-parameters.
//!
//! Any fallible function will return an 'error code' - either
//! `1` for failure or `0` for success.
//!
//! Any returned values are written via 'out-pointers' - and should only be
//! loaded on success.

use std::mem::MaybeUninit;

use cranelift::frontend::FunctionBuilder;

use crate::jit::builder::typed::Pointer;
use crate::jit::builder::typed::Value;
use crate::traps::Exception;

/// Helper type for ensuring fallible operations are handled correctly.
///
/// The errno is constructed out of three pieces:
/// - whether or not a failure occurred
/// - if yes, the pointer to the exception in memory that has been written with the failure kind
/// - if no, a handler to load any state that was returned in `out-params` that is now safe to
///   access.
pub(crate) struct ErrnoImpl<T, F>
where
    F: FnOnce(&mut FunctionBuilder) -> T,
{
    /// Boolean value indicating whether an exception occurred
    pub(crate) is_exception: Value<bool>,

    /// Pointer to the exception in memory, if an exception occurred
    pub(crate) exception_ptr: Pointer<MaybeUninit<Exception>>,

    /// Retrieve the result in case of success
    pub(crate) on_ok: F,
}

impl<T, F> ErrnoImpl<T, F>
where
    F: FnOnce(&mut FunctionBuilder) -> T,
{
    /// Construct a new `Errno` that must be handled.
    pub(crate) fn new(
        is_exception: Value<bool>,
        exception_ptr: Pointer<MaybeUninit<Exception>>,
        on_ok: F,
    ) -> Self {
        Self {
            is_exception,
            exception_ptr,
            on_ok,
        }
    }
}
