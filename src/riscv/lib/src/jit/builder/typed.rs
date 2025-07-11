// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Strongly-typed wrapper around Cranelift IR values
//!
//! This module provides a [`Value`] type that wraps Cranelift's untyped IR values with
//! compile-time type information. This enables type-safe operations on IR values while maintaining
//! the flexibility of the underlying Cranelift infrastructure.
//!
//! The main type [`Value`] is a zero-cost abstraction that carries type information
//! in the type system without runtime overhead. It provides safe conversion functions
//! and operations that preserve type safety across IR transformations.

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use cranelift::codegen::ir::Value as CraneliftValue;

/// Strongly-typed IR value
#[derive(Debug)]
pub struct Value<T> {
    /// Underlying Cranelift value
    value: CraneliftValue,

    _pd: PhantomData<T>,
}

impl<T> Value<T> {
    /// Enhance a raw Canelift value with a type.
    ///
    /// # Safety
    ///
    /// The caller must ensure the type `T` is correct for the given Cranelift value.
    pub unsafe fn from_raw(value: CraneliftValue) -> Self {
        Value {
            value,
            _pd: PhantomData,
        }
    }

    /// Extract the raw Cranelift value from this typed value.
    pub fn to_value(self) -> CraneliftValue {
        self.value
    }

    /// Lift a unary operation on a Cranelift value to a typed value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the function `f` produces a Cranelift value that represents the
    /// type `T`.
    pub unsafe fn lift_unary(self, f: impl FnOnce(CraneliftValue) -> CraneliftValue) -> Value<T> {
        let raw = f(self.to_value());

        // SAFETY: `f` must produce a Cranelift value of type `T`.
        unsafe { Value::<T>::from_raw(raw) }
    }

    /// Lift a binary operation on two Cranelift values to typed values.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the function `f` produces a Cranelift value that represents the
    /// type `T`.
    pub unsafe fn lift_binary(
        self,
        f: impl FnOnce(CraneliftValue, CraneliftValue) -> CraneliftValue,
        rhs: Value<T>,
    ) -> Value<T> {
        let raw = f(self.to_value(), rhs.to_value());

        // SAFETY: `f` must produce a Cranelift value of type `T`.
        unsafe { Value::<T>::from_raw(raw) }
    }
}

impl<T> Value<NonNull<MaybeUninit<T>>> {
    /// Treat the pointee `T` as initialised.
    ///
    /// # Safety
    ///
    /// You must ensure that the pointee is indeed initialised before using it.
    pub unsafe fn assume_init(self) -> Value<NonNull<T>> {
        Value {
            value: self.value,
            _pd: PhantomData,
        }
    }
}

// The deriver macro imposes `T: Copy` on `Value<T>`. We don't want that, so we write our own impl.
impl<T> Copy for Value<T> {}

// See `impl Copy` for why we hand-write this impl.
impl<T> Clone for Value<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// See `impl Copy` for why we hand-write this impl.
impl<T> PartialEq for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

// See `impl Copy` for why we hand-write this impl.
impl<T> Eq for Value<T> {}

// See `impl Copy` for why we hand-write this impl.
impl<T> PartialOrd for Value<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// See `impl Copy` for why we hand-write this impl.
impl<T> Ord for Value<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

/// IR pointer to a value of type `T`
pub type Pointer<T> = Value<NonNull<T>>;
