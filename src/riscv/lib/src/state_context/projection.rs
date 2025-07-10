// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Field projection utilities for type-safe access to nested state components
//!
//! This module provides a projection system that allows safe access to nested components
//! within complex state structures. It uses type constructors to work around Rust's
//! limitations with higher-kinded types, enabling generic projections that work across
//! different memory configurations and state backend managers.

use std::marker::PhantomData;

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::Cell;
use crate::state_backend::Cells;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;

/// Helper for type equality for higher-kinded types
///
/// There is no first-class mechanism to express type equality for higher kinded types in Rust. For
/// example, a type expression `Foo<A>` can be equal to `Bar<B>` but we can't express `Foo` is equal
/// to `Bar`. Type constructors must be fully applied at the type level in Rust.
///
/// This poses a problem for [`Projection`] which wants to use a subject type of shape `Foo<MC, M>`
/// for any `MC` and `M`. It is not possible to directly express that a projection has `Foo` as
/// subject, without applying `MC` and `M`.
///
/// [`TypeCons`] provides a work around. Thanks to generic associated types, we can express
/// properties like these: if `A == B` then for all `MC` and `M`,
/// `A::Instance<MC, M> == B::Instance<MC, M>`.
pub trait TypeCons {
    /// Fully apply the type constructor
    type Applied<MC: MemoryConfig, M: ManagerBase>;
}

/// Apply a type constructor `TC` to memory config `MC` and manager `M`.
pub type ApplyCons<TC, MC, M> = <TC as TypeCons>::Applied<MC, M>;

/// Type constructor [`ManagerBase::Region`]
pub struct RegionCons<E, const LEN: usize>(PhantomData<E>);

impl<E: 'static, const LEN: usize> TypeCons for RegionCons<E, LEN> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = M::Region<E, LEN>;
}

/// Type constructor [`crate::state_backend::Cell`]
pub struct CellCons<E>(PhantomData<E>);

impl<E: 'static> TypeCons for CellCons<E> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = Cell<E, M>;
}

/// Type constructor [`crate::state_backend::Cells`]
pub struct CellsCons<E, const LEN: usize>(PhantomData<E>);

impl<E: 'static, const LEN: usize> TypeCons for CellsCons<E, LEN> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = Cells<E, LEN, M>;
}

/// Type constructor [`crate::machine_state::MachineCoreState`]
pub struct MachineCoreCons;

impl TypeCons for MachineCoreCons {
    type Applied<MC: MemoryConfig, M: ManagerBase> = MachineCoreState<MC, M>;
}

/// Projections give you access to a value of the target type within the value of a subject type.
pub trait Projection {
    /// Subject that contains the target value
    type Subject: TypeCons;

    /// Type of the target value
    type Target;

    /// Projection parameter
    ///
    /// For example, this could be an index when the projection is selecting an element from an
    /// array. In practise this can be any kind of information that is required to perform the
    /// projection.
    type Parameter: tuples::Tuple;

    /// Obtain a reference to the target value within the subject value.
    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> &'a Self::Target;

    /// Read the target value from the subject value.
    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> Self::Target
    where
        Self::Target: Copy;

    /// Obtain a mutable reference to the target value within the subject value.
    fn project_write<'a, MC: MemoryConfig, M: ManagerWrite + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
        value: Self::Target,
    );

    /// Get the offset of the target value within the subject value. In other words, it is the
    /// offset to an address of the subject value that would give you the address of the target
    /// value. This is exclusive to the [`crate::state_backend::owned_backend::Owned`] state
    /// backend.
    fn owned_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> i32;
}

/// Implement a projection by pre-composing a field access to an existing projection.
macro_rules! impl_projection {
    (
        $vis:vis projection $name:ident {
            subject = $subject:ty,
            target_projection = $target:ty,
            path = $($field:ident).+ $(,)?
        }
    ) => {
        $vis enum $name {}

        impl $crate::state_context::projection::Projection for $name {
            type Subject = $subject;

            type Target = <$target as $crate::state_context::projection::Projection>::Target;

            type Parameter = <$target as $crate::state_context::projection::Projection>::Parameter;

            #[inline]
            fn project_ref<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerRead + 'a,
            >(
                state: &'a $crate::state_context::projection::ApplyCons<Self::Subject, MC, M>,
                param: Self::Parameter,
            ) -> &'a Self::Target {
                <$target>::project_ref::<MC, M>(
                    &state.$($field).+,
                    param,
                )
            }

            #[inline]
            fn project_read<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerRead + 'a,
            >(
                state: &'a $crate::state_context::projection::ApplyCons<Self::Subject, MC, M>,
                param: Self::Parameter,
            ) -> Self::Target {
                <$target>::project_read::<MC, M>(
                    &state.$($field).+,
                    param,
                )
            }

            #[inline]
            fn project_write<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerWrite + 'a,
            >(
                state: &'a mut $crate::state_context::projection::ApplyCons<Self::Subject, MC, M>,
                param: Self::Parameter,
                value: Self::Target,
            ) {
                <$target>::project_write::<MC, M>(
                    &mut state.$($field).+,
                    param,
                    value,
                )
            }

            fn owned_pointer_offset<MC: $crate::machine_state::memory::MemoryConfig>(
                param: Self::Parameter
            ) -> i32 {
                let field_offset: i32 = std::mem::offset_of!(
                    $crate::state_context::projection::ApplyCons<
                        $subject,
                        MC,
                        $crate::state_backend::owned_backend::Owned
                    >,
                    $($field).+
                ).try_into().expect("Field offset exceeds i32 range");
                field_offset + <$target>::owned_pointer_offset::<MC>(param)
            }
        }
    };
}

pub(crate) use impl_projection;

trait_set::trait_set! {
    /// Alias for [`Projection`] with `MachineCoreCons` as subject type
    pub trait MachineCoreProjection = Projection<Subject = MachineCoreCons>;
}
