// SPDX-FileCopyrightText: 2024-2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025-2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Abstraction over the execution method when dispatching from a [`CodePage`].
//!
//! [`CodePage`]: super::CodePage

use crate::machine_state::memory::Address;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::page_cache::dispatch::DispatchCompiler;
use crate::machine_state::page_cache::dispatch::DispatchFn;
use crate::machine_state::page_cache::entrypoint::Page;
use crate::machine_state::page_cache::router::RouterEq;

/// A compiler that does not actually do any compilation.
#[derive(Debug, Default, Clone)]
pub struct InterpretedCompiler;

/// Since [`InterpretedCompiler`] does no compilation at all, [`router_eq`] should always return
/// false, so that the ranges in the router are kept as small as possible. This helps reduce the
/// number of pages that could get unneccessarily dropped.
///
/// [`router_eq`]: crate::machine_state::page_cache::router::RouterEq::router_eq
impl RouterEq for InterpretedCompiler {
    fn router_eq(&self, _other: &Self) -> bool {
        false
    }
}

/// Dummy implementation of the `DispatchCompiler` trait.
impl<MC: MemoryConfig> DispatchCompiler<MC> for InterpretedCompiler {
    /// The `InterpretedCompiler` needs no context.
    type Context = ();

    /// Trivial constructor for an empty struct.
    fn new(_ctx: &()) -> Self {
        InterpretedCompiler
    }

    /// We never compile, instead simply return `None`.
    fn compile(
        _target: &Page<Self, MC>,
        _program_counter: Address,
    ) -> Option<DispatchFn<Self, MC>> {
        None
    }
}
