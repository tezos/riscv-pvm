// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use super::ManagerBase;
use crate::state_backend::Ref;

/// Transformer of borrowed regions of manager `I`
pub trait FnManager<'a, I: ManagerBase> {
    /// Resulting manager type
    type Output: ManagerBase;

    /// Create a region of the output manager from a borrowed region of the input manager.
    fn map_region<E: 'static, const LEN: usize>(
        input: &'a I::Region<E, LEN>,
    ) -> <Self::Output as ManagerBase>::Region<E, LEN>;

    /// Create a dynamic region of the output manager from a borrowed dynamic region of the input
    /// manager.
    fn map_dyn_region(input: &'a I::DynRegion) -> <Self::Output as ManagerBase>::DynRegion;
}

/// Identity transformation for [`FnManager`]
pub enum FnManagerIdent {}

impl<'a, M: ManagerBase + 'a> FnManager<'a, M> for FnManagerIdent {
    type Output = Ref<'a, M>;

    fn map_region<E: 'static, const LEN: usize>(input: &M::Region<E, LEN>) -> &M::Region<E, LEN> {
        input
    }

    fn map_dyn_region(input: &M::DynRegion) -> &M::DynRegion {
        input
    }
}
