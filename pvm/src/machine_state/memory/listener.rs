// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Infrastructure that allows various non-memory parts of the RISC-V vm to be notified of any
//! permission updates that happen in main-memory.

use std::ops::RangeInclusive;

use super::Permissions;

/// A `MemoryGovernanceListener` is allowed to 'hook-into' any permission updates that may occur in
/// memory.
///
/// This allows other parts of the state (e.g. the *PageCache* or future *JIT Router* to ensure
/// they remain synchronised with memory).
pub trait MemoryGovernanceListener {
    /// Handle notification that a region of memory (set of pages) has had its permissions updated.
    fn handle_permissions_update(&mut self, pages: RangeInclusive<u64>, permissions: Permissions);
}

/// A memory governance listener that does nothing.
pub struct NoopMemoryGovernanceListener;

impl MemoryGovernanceListener for NoopMemoryGovernanceListener {
    fn handle_permissions_update(
        &mut self,
        _pages: RangeInclusive<u64>,
        _permissions: Permissions,
    ) {
    }
}

impl<MGL> MemoryGovernanceListener for &mut MGL
where
    MGL: MemoryGovernanceListener,
{
    fn handle_permissions_update(&mut self, pages: RangeInclusive<u64>, permissions: Permissions) {
        MGL::handle_permissions_update(*self, pages, permissions);
    }
}
