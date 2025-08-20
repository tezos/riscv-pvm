// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! TODO

use crate::jit::builder::typed::Value;
use crate::state_context::StateContext;

/// TODO
pub trait JitStateContext: StateContext {
    /// TODO
    fn to_jit_value<T>(value: Self::Value<T>) -> Value<T>;
}
