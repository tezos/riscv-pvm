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
use std::ops::Deref;
use std::ops::DerefMut;

use cranelift::codegen::ir;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift::prelude::MemFlags;
use cranelift::prelude::isa::TargetFrontendConfig;

use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::Cell;
use crate::state_backend::Cells;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerWrite;
use crate::state_backend::owned_backend::Owned;

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

/// Type constructor [`Box`]
pub struct BoxCons<T>(PhantomData<T>);

impl<T: TypeCons> TypeCons for BoxCons<T> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = Box<ApplyCons<T, MC, M>>;
}

/// Type constructor `[T; LEN]`
pub struct ArrayCons<T, const LEN: usize>(PhantomData<T>);

impl<T: TypeCons, const LEN: usize> TypeCons for ArrayCons<T, LEN> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = [ApplyCons<T, MC, M>; LEN];
}

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

/// Offset from a base pointer to a [projection's] subject, within the owned backend.
///
/// Additional offsets may be added to an existing one, to build up offsets within
/// layered projections. All such additions will panic on overflowing the `i32` range.
///
/// [projection's]: Projection
#[derive(Debug, Clone)]
pub enum ProjectionOffset {
    /// Target value is directly accessible from the base pointer
    ///
    /// Adding the offset to the base pointer will yield the address of the target value.
    Direct {
        /// Offset from the base in bytes
        offset: i32,
    },

    /// Target value is accessible via an indirection
    ///
    /// Adding the offset to the base pointer will yield the address of the next base pointer. The
    /// `inner` projection then needs to proceed with the new base pointer.
    Indirect {
        /// Offset from the base in bytes
        offset: i32,

        /// Inner projection to be applied after resolving the indirection
        inner: Box<ProjectionOffset>,
    },
}

impl ProjectionOffset {
    /// Create a new projection offset from the given offset.
    ///
    /// # Panics
    ///
    /// Panics if the offset overflows the `i32` range.
    pub fn direct(offset: usize) -> Self {
        let offset = offset
            .try_into()
            .expect("Projection offset overflows i32 range");
        Self::Direct { offset }
    }

    /// Resolve the projection offset to a base pointer and an offset relative to the base. Adding
    /// the offset to the base pointer will yield the address of the target value.
    pub fn build_base_and_offset(
        &self,
        target_config: &TargetFrontendConfig,
        builder: &mut FunctionBuilder,
        base: ir::Value,
    ) -> (ir::Value, Offset32) {
        match self {
            ProjectionOffset::Direct { offset } => {
                let offset = Offset32::new(*offset);
                (base, offset)
            }

            ProjectionOffset::Indirect { offset, inner } => {
                let new_base = builder.ins().load(
                    target_config.pointer_type(),
                    MemFlags::trusted(),
                    base,
                    *offset,
                );
                inner.build_base_and_offset(target_config, builder, new_base)
            }
        }
    }
}

impl std::ops::Add<usize> for ProjectionOffset {
    type Output = Self;

    fn add(self, offset: usize) -> Self {
        let offset: u32 = offset
            .try_into()
            .expect("Projection offset overflows u32 range");

        match self {
            ProjectionOffset::Direct { offset: raw_offset } => ProjectionOffset::Direct {
                offset: raw_offset
                    .checked_add_unsigned(offset)
                    .expect("Projection offset overflow"),
            },

            ProjectionOffset::Indirect {
                offset: base_offset,
                inner,
            } => ProjectionOffset::Indirect {
                offset: base_offset
                    .checked_add_unsigned(offset)
                    .expect("Projection offset overflow"),
                inner,
            },
        }
    }
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
    type Parameter;

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
    fn owned_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> ProjectionOffset;
}

/// A projection from [`Box`] to its inner type
pub struct BoxProj<P>(P);

impl<P: Projection> Projection for BoxProj<P> {
    type Subject = BoxCons<P::Subject>;

    type Target = P::Target;

    type Parameter = P::Parameter;

    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> &'a Self::Target {
        P::project_ref::<MC, M>(state.deref(), param)
    }

    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> Self::Target
    where
        Self::Target: Copy,
    {
        P::project_read::<MC, M>(state.deref(), param)
    }

    fn project_write<'a, MC: MemoryConfig, M: ManagerWrite + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
        value: Self::Target,
    ) {
        P::project_write::<MC, M>(state.deref_mut(), param, value);
    }

    fn owned_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> ProjectionOffset {
        let offset = P::owned_pointer_offset::<MC>(param);
        ProjectionOffset::Indirect {
            offset: 0,
            inner: Box::new(offset),
        }
    }
}

/// Parameter for an array projection
pub struct ArrayProjParam<T> {
    /// Index of the element to project
    pub index: usize,

    /// Parameter for the inner projection
    pub inner_param: T,
}

/// A projection from an array to one of its elements
pub struct ArrayProj<P, const LEN: usize>(P);

impl<P: Projection, const LEN: usize> Projection for ArrayProj<P, LEN> {
    type Subject = ArrayCons<P::Subject, LEN>;

    type Target = P::Target;

    type Parameter = ArrayProjParam<P::Parameter>;

    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> &'a Self::Target {
        let inner_state = &state[param.index];
        P::project_ref::<MC, M>(inner_state, param.inner_param)
    }

    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> Self::Target
    where
        Self::Target: Copy,
    {
        let inner_state = &state[param.index];
        P::project_read::<MC, M>(inner_state, param.inner_param)
    }

    fn project_write<'a, MC: MemoryConfig, M: ManagerWrite + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
        value: Self::Target,
    ) {
        let inner_state = &mut state[param.index];
        P::project_write::<MC, M>(inner_state, param.inner_param, value);
    }

    fn owned_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> ProjectionOffset {
        assert!(
            param.index < LEN,
            "Array index out of bounds: {} >= {}",
            param.index,
            LEN
        );

        let inner_offset = P::owned_pointer_offset::<MC>(param.inner_param);

        let elem_size = std::mem::size_of::<<P::Subject as TypeCons>::Applied<MC, Owned>>();
        let offset = param
            .index
            .checked_mul(elem_size)
            .expect("Array index overflow");

        inner_offset + offset
    }
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
            ) -> $crate::state_context::projection::ProjectionOffset {
                let field_offset = std::mem::offset_of!(
                    $crate::state_context::projection::ApplyCons<
                        $subject,
                        MC,
                        $crate::state_backend::owned_backend::Owned
                    >,
                    $($field).+
                );

                <$target>::owned_pointer_offset::<MC>(param) + field_offset
            }
        }
    };
}

pub(crate) use impl_projection;

trait_set::trait_set! {
    /// Alias for [`Projection`] with `MachineCoreCons` as subject type
    pub trait MachineCoreProjection = Projection<Subject = MachineCoreCons>;
}
