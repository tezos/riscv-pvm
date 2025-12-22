// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! A dummy version of the page cache for use in proof and verify modes.

use super::CodePage;
use super::MachineCoreState;
use super::PageCache;
use super::StepManyResult;
use super::memory;
use super::memory::Address;
use super::memory::MemoryConfig;
use super::memory::listener::MemoryGovernanceListener;
use crate::exceptions::Exception;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;

/// A page cache that does nothing. Used in proof and verify modes.
pub struct EmptyPageCache;

impl<MC: MemoryConfig, M: ManagerBase> PageCache<MC, M> for EmptyPageCache {
    fn new() -> Self {
        EmptyPageCache
    }

    fn get_code_page(&mut self, _addr: Address) -> Option<impl CodePage<'_, MC, M>>
    where
        M: ManagerRead,
    {
        Option::<NoCodePage>::None
    }

    fn populate_page(&mut self, _addr: Address, _core: &MachineCoreState<MC, M>)
    where
        M: ManagerRead + ManagerWrite,
    {
    }
}

impl MemoryGovernanceListener for EmptyPageCache {
    fn handle_permissions_update(
        &mut self,
        _range: std::ops::RangeInclusive<u64>,
        _perms: memory::Permissions,
    ) {
    }
}

enum NoCodePage {}

impl<'a, MC: MemoryConfig, M: ManagerBase> CodePage<'a, MC, M> for NoCodePage {
    fn run(
        &mut self,
        _core: &mut MachineCoreState<MC, M>,
        _instr_pc: Address,
        _max_steps: usize,
    ) -> StepManyResult<Exception> {
        match *self {}
    }
}
