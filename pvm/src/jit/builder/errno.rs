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

use cranelift::frontend::FunctionBuilder;

use crate::jit::builder::typed::Value;
use crate::jit::state_access::ExceptionCode;

/// Helper type for ensuring fallible operations are handled correctly.
///
/// The errno is constructed out of two pieces:
/// - an exception code indicating whether an exception occurred and which type
/// - a handler to load any state that was returned in `out-params` that is now safe to
///   access on success.
pub(crate) struct ErrnoImpl<T, F>
where
    F: FnOnce(&mut FunctionBuilder) -> T,
{
    /// Exception code, indicates whether an exception occurred and which
    pub(crate) code: Value<ExceptionCode>,

    /// Retrieve the result in case of success
    pub(crate) on_ok: F,
}

impl<T, F> ErrnoImpl<T, F>
where
    F: FnOnce(&mut FunctionBuilder) -> T,
{
    /// Construct a new `Errno` that must be handled.
    pub(crate) fn new(code: Value<ExceptionCode>, on_ok: F) -> Self {
        Self { code, on_ok }
    }
}
