// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use super::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::Ref;

/// Transformation from a manager `I` to another manager
pub trait FnManager<I: ManagerBase> {
    /// Resulting manager type
    type Output: ManagerBase;

    /// Transform the region of manager `I` to one of manager `O`.
    fn map_region<E: 'static + Clone, const LEN: usize>(
        input: I::Region<E, LEN>,
    ) -> <Self::Output as ManagerBase>::Region<E, LEN>;

    /// Transform the dynamic region of manager `I` to one of manager `O`.
    fn map_dyn_region<const LEN: usize>(
        input: I::DynRegion<LEN>,
    ) -> <Self::Output as ManagerBase>::DynRegion<LEN>;
}

/// Identity transformation for [`FnManager`]
pub enum FnManagerIdent {}

impl<M: ManagerBase> FnManager<M> for FnManagerIdent {
    type Output = M;

    fn map_region<E: 'static + Clone, const LEN: usize>(
        input: M::Region<E, LEN>,
    ) -> M::Region<E, LEN> {
        input
    }

    fn map_dyn_region<const LEN: usize>(input: M::DynRegion<LEN>) -> M::DynRegion<LEN> {
        input
    }
}

/// Transformation from a manager `Ref<'_, M>` to `M` by cloning its regions.
pub enum FnManagerDerefClone {}

impl<M: ManagerClone> FnManager<Ref<'_, M>> for FnManagerDerefClone {
    type Output = M;

    fn map_region<E: 'static + Clone, const LEN: usize>(
        input: &M::Region<E, LEN>,
    ) -> M::Region<E, LEN> {
        <M as ManagerClone>::clone_region(input)
    }

    fn map_dyn_region<const LEN: usize>(input: &M::DynRegion<LEN>) -> M::DynRegion<LEN> {
        <M as ManagerClone>::clone_dyn_region(input)
    }
}
