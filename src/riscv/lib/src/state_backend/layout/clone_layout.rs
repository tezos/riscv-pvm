// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! This module defines the [`CloneLayout`] trait and its implementations.

use crate::state_backend::Array;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::state_backend::Cells;
use crate::state_backend::DynArray;
use crate::state_backend::DynCells;
use crate::state_backend::Layout;
use crate::state_backend::ManagerClone;
use crate::state_backend::Many;
use crate::state_backend::Ref;

/// [`Layout`] which can be cloned.
pub trait CloneLayout: Layout {
    /// Clone the allocated space of this layout behind a [`Ref`].
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M>;
}

impl<E: Clone + 'static> CloneLayout for Atom<E> {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let region = space.into_region();
        let region = M::clone_region(region);
        Cell::bind(region)
    }
}

impl<E: Clone + 'static, const LEN: usize> CloneLayout for Array<E, LEN> {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let region = space.into_region();
        let region = M::clone_region(region);
        Cells::bind(region)
    }
}

impl<A: CloneLayout, B: CloneLayout> CloneLayout for (A, B) {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let a = A::clone_allocated(space.0);
        let b = B::clone_allocated(space.1);
        (a, b)
    }
}

impl<A: CloneLayout, B: CloneLayout, C: CloneLayout> CloneLayout for (A, B, C) {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let a = A::clone_allocated(space.0);
        let b = B::clone_allocated(space.1);
        let c = C::clone_allocated(space.2);
        (a, b, c)
    }
}

impl<A: CloneLayout, B: CloneLayout, C: CloneLayout, D: CloneLayout> CloneLayout for (A, B, C, D) {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let a = A::clone_allocated(space.0);
        let b = B::clone_allocated(space.1);
        let c = C::clone_allocated(space.2);
        let d = D::clone_allocated(space.3);
        (a, b, c, d)
    }
}

impl<A: CloneLayout, B: CloneLayout, C: CloneLayout, D: CloneLayout, E: CloneLayout> CloneLayout
    for (A, B, C, D, E)
{
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let a = A::clone_allocated(space.0);
        let b = B::clone_allocated(space.1);
        let c = C::clone_allocated(space.2);
        let d = D::clone_allocated(space.3);
        let e = E::clone_allocated(space.4);
        (a, b, c, d, e)
    }
}

impl<const LEN: usize> CloneLayout for DynArray<LEN> {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        let region = space.region_ref();
        let region = M::clone_dyn_region(region);
        DynCells::bind(region)
    }
}

impl<L: CloneLayout, const LEN: usize> CloneLayout for [L; LEN] {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        space.map(L::clone_allocated)
    }
}

impl<L: CloneLayout, const LEN: usize> CloneLayout for Many<L, LEN> {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        space.into_iter().map(L::clone_allocated).collect()
    }
}

impl<L: CloneLayout> CloneLayout for Box<L> {
    fn clone_allocated<M: ManagerClone>(space: Self::Allocated<Ref<'_, M>>) -> Self::Allocated<M> {
        Box::new(L::clone_allocated(*space))
    }
}
