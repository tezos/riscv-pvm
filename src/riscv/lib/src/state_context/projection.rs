// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;

use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;

/// Helper for type equality for higher-kinded types
///
/// There is no first-class mechanism to express type equality for higher kinded types in Rust. For
/// example, a type expression `Foo<A>` can be equal to `Bar<B>` but we can't express `Foo` is equal
/// to `Bar`. Type constructors must be fully applied at the type level in Rust.
///
/// This poses a problem for [`Projection`] which wants to use subject and target types of shape
/// `Foo<MC, M>` for any `MC` and `M`. It is not possible to directly express that a projection has
/// `Foo` as target or subject, without applying `MC` and `M`.
///
/// [`TypeCons`] provides a work around. Thanks to generic associated types, we can express
/// properties like these: if `A == B` then for all `MC` and `M`,
/// `A::Instance<MC, M> == B::Instance<MC, M>`.
pub trait TypeCons {
    /// Fully apply the type constructor
    type Applied<MC: MemoryConfig, M: ManagerBase>;
}

/// Apply a type constructor `C` to memory config `MC` and manager `M`.
pub type ApplyCons<C, MC, M> = <C as TypeCons>::Applied<MC, M>;

/// Type constructor [`ManagerBase::Region`]
pub struct RegionCons<E, const LEN: usize>(PhantomData<E>);

impl<E: 'static, const LEN: usize> TypeCons for RegionCons<E, LEN> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = M::Region<E, LEN>;
}

/// Type constructor [`crate::state_backend::Cell`]
pub struct CellCons<E>(PhantomData<E>);

impl<E: 'static> TypeCons for CellCons<E> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = crate::state_backend::Cell<E, M>;
}

/// Type constructor [`crate::state_backend::Cells`]
pub struct CellsCons<E, const LEN: usize>(PhantomData<E>);

impl<E: 'static, const LEN: usize> TypeCons for CellsCons<E, LEN> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = crate::state_backend::Cells<E, LEN, M>;
}

/// Type constructor [`crate::machine_state::MachineCoreState`]
pub struct MachineCoreCons;

impl TypeCons for MachineCoreCons {
    type Applied<MC: MemoryConfig, M: ManagerBase> = crate::machine_state::MachineCoreState<MC, M>;
}

/// Projections give you access to a value of the target type within the value of a subject type.
pub trait Projection {
    /// Subject that contains the target value
    type Subject: TypeCons;

    /// Type of the target value
    type Target: TypeCons;

    /// Obtain a reference to the target value within the subject value.
    fn project<'a, MC: MemoryConfig, M: ManagerBase + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
    ) -> &'a ApplyCons<Self::Target, MC, M>;

    /// Obtain a mutable reference to the target value within the subject value.
    fn project_mut<'a, MC: MemoryConfig, M: ManagerBase + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
    ) -> &'a mut ApplyCons<Self::Target, MC, M>;

    /// Get the offset of the target value within the subject value. In other words, it is the
    /// offset to an address of the subject value that would give you the address of the target
    /// value.
    fn pointer_offset<MC: MemoryConfig, M: ManagerBase>() -> usize;
}

/// Implement a projection for a subject type to a target type given a field accessor path.
macro_rules! impl_projection {
    ($vis:vis $name:ident ( $subject:ty => $target:ty ) = $($field:ident).+) => {
        $vis enum $name {}

        impl $crate::state_context::projection::Projection for $name {
            type Subject = $subject;

            type Target = $target;

            #[inline]
            fn project<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerBase + 'a,
            >(
                state: &'a $crate::state_context::projection::ApplyCons<Self::Subject, MC, M>,
            ) -> &'a $crate::state_context::projection::ApplyCons<Self::Target, MC, M> {
                &state.$($field).+
            }

            #[inline]
            fn project_mut<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerBase + 'a,
            >(
                state: &'a mut $crate::state_context::projection::ApplyCons<Self::Subject, MC, M>,
            ) -> &'a mut $crate::state_context::projection::ApplyCons<Self::Target, MC, M> {
                &mut state.$($field).+
            }

            fn pointer_offset<
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerBase,
            >() -> usize {
                std::mem::offset_of!(
                    $crate::state_context::projection::ApplyCons<Self::Subject, MC, M>,
                    $($field).+
                )
            }
        }
    };
}

pub(crate) use impl_projection;

trait_set::trait_set! {
    pub trait MachineCoreProjection = Projection<Subject = MachineCoreCons>;
}
