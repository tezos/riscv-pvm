// SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Generic state backends
//!
//! # Managers
//!
//! Different backends have different capabilities and they are described as `Manager<Capability>`.
//! Some of these capabilities are:
//! - [ManagerBase]
//! - [ManagerAlloc]
//! - [ManagerRead]
//! - [ManagerWrite]
//!
//! # Modes
//!
//! Modes are ZSTs implementing these traits.
//! The main difference between them is the top-level functionality it provides
//! and management of the underlying state memory.
//!
//! These modes can be:
//!
//! - [Normal]
//!   Mode which has the full state allocated in memory. It can execute one step
//!   or multiple steps at a time faster.
//! - [Verify]
//!   Mode capable of partially allocating a state and verify a given proof.
//!   Needs to be light on memory usage since it runs in the protocol.
//! - [Prove]
//!   Mode capable of generating a proof for running one step.
//!
//! [Normal]: octez_riscv_data::mode::Normal
//! [Verify]: octez_riscv_data::mode::Verify
//! [Prove]: octez_riscv_data::mode::Prove

mod elems;
pub mod normal_backend;
pub mod proof_backend;
pub(crate) mod proof_layout;
mod region;
pub mod verify_backend;

use std::marker::PhantomData;

use bincode::de::Decode;
use bincode::de::Decoder;
use bincode::enc::Encode;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
pub use elems::*;
pub use proof_layout::*;
pub use region::*;

use crate::machine_state::memory::MemoryConfig;
use crate::state_context::projection::ApplyCons;
use crate::state_context::projection::Projection;
use crate::state_context::projection::ProjectionOffset;
use crate::state_context::projection::RegionCons;

/// Manager of the state backend storage
pub trait ManagerBase: Sized {
    /// Region that has been allocated in the state storage
    type Region<E: 'static, const LEN: usize>;

    /// Dynamic region represents a fixed-sized byte vector that has been allocated in the state storage
    type DynRegion;

    /// The `ManagerRoot` is the ultimate manager type used to run things.
    ///
    /// It is primarily used to defer trait bounds to a later point when the root manager type is
    /// known.
    ///
    /// You might need to refer to a function `foo<M: ManagerWrite>(...)`. Unless you're running
    /// that function, the `M: ManagerWrite` bound is not needed. However, Rust does not let you
    /// express that directly. The trait bound would be immediately needed, therefore granting the
    /// function that refers to `foo` the same capabilities.
    ///
    /// The `ManagerRoot` is a utility that can be used to express that the function is only
    /// callable when `M` is the root manager type. We instantiate `foo<M::ManagerRoot>`, so that
    /// the trait bounds are imposed on the root manager type.
    ///
    /// This alone does not let us run `foo`. Fortunately, `ManagerWrite` requires that
    /// `ManagerRoot = Self`. This in contexts with `M: ManagerWrite`, we can actually run
    /// `foo<M::ManagerRoot>` as `foo<M>`.
    type ManagerRoot: ManagerBase<ManagerRoot = Self::ManagerRoot>;
}

/// Manager with allocation capabilities
///
/// Any `ManagerAlloc` inherently has read & write capabilities,
/// since the manager creates the values on the first allocation.
pub trait ManagerAlloc: ManagerRead + ManagerWrite {
    /// Allocate a region in the state storage.
    fn allocate_region<E, const LEN: usize>(init_value: [E; LEN]) -> Self::Region<E, LEN>;

    /// Allocate a dynamic region in the state storage.
    fn allocate_dyn_region(len: usize) -> Self::DynRegion;
}

/// Manager with read capabilities
pub trait ManagerRead: ManagerBase {
    /// Read an element in the region.
    fn region_read<E: Copy, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> E;

    /// Obtain a reference to an element in the region.
    fn region_ref<E: 'static, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> &E;

    /// Read all elements in the region.
    fn region_read_all<E: Copy, const LEN: usize>(region: &Self::Region<E, LEN>) -> Vec<E>;

    /// Read the length of the dynamic region in bytes.
    fn dyn_region_len(region: &Self::DynRegion) -> usize;

    /// Read an element in the region. `address` is in bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure the access is within bounds.
    ///
    /// ```text
    /// address + E:STORED <= region.len()
    /// ```
    unsafe fn dyn_region_read<E: Elem>(region: &Self::DynRegion, address: usize) -> E;

    /// Read elements from the region. `address` is in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the read would go out of bounds.
    fn dyn_region_read_all<E: Elem>(region: &Self::DynRegion, address: usize, values: &mut [E]) {
        if values.is_empty() {
            return;
        }

        assert!(
            values
                .len()
                .checked_mul(E::STORED_SIZE.get())
                .expect("Total length should not overflow")
                .checked_add(address)
                .expect("End address should not overflow")
                <= Self::dyn_region_len(region),
        );

        for (i, value) in values.iter_mut().enumerate() {
            // SAFETY: The assertion above ensures all reads are within bounds.
            unsafe {
                *value = Self::dyn_region_read::<E>(
                    region,
                    E::STORED_SIZE.get().wrapping_mul(i).wrapping_add(address),
                )
            };
        }
    }
}

/// Manager with write capabilities
pub trait ManagerWrite: ManagerBase<ManagerRoot = Self> {
    /// Update an element in the region.
    fn region_write<E: 'static, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    );

    /// Update all elements in the region.
    fn region_write_all<E: Copy, const LEN: usize>(region: &mut Self::Region<E, LEN>, value: &[E]);

    /// Update an element in the region. `address` is in bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure the access is within bounds.
    ///
    /// ```text
    /// address + E:STORED <= region.len()
    /// ```
    unsafe fn dyn_region_write<E: Elem>(region: &mut Self::DynRegion, address: usize, value: E);

    /// Update multiple elements in the region. `address` is in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the write would go out of bounds.
    fn dyn_region_write_all<E: Elem + Copy>(
        region: &mut Self::DynRegion,
        address: usize,
        values: &[E],
    ) where
        Self: ManagerRead,
    {
        if values.is_empty() {
            return;
        }

        assert!(
            values
                .len()
                .checked_mul(E::STORED_SIZE.get())
                .expect("Total length should not overflow")
                .checked_add(address)
                .expect("End address should not overflow")
                <= Self::dyn_region_len(region)
        );

        for (i, value) in values.iter().enumerate() {
            // SAFETY: The assertion above ensures all writes are within bounds.
            unsafe {
                Self::dyn_region_write::<E>(
                    region,
                    E::STORED_SIZE.get().wrapping_mul(i).wrapping_add(address),
                    *value,
                );
            }
        }
    }
}

/// Manager with the ability to serialise regions
pub trait ManagerSerialise: ManagerRead {
    /// Serialise the contents of the region.
    fn serialise_region<T: Encode, const LEN: usize, E: Encoder>(
        region: &Self::Region<T, LEN>,
        encoder: E,
    ) -> Result<(), EncodeError>;

    /// Serialise the contents of the dynamic region.
    fn serialise_dyn_region<E: Encoder>(
        region: &Self::DynRegion,
        encoder: E,
    ) -> Result<(), EncodeError>;
}

/// Manager with the ability to deserialise regions
pub trait ManagerDeserialise: ManagerBase {
    /// Deserialise a region.
    fn deserialise_region<T: Decode<D::Context>, const LEN: usize, D: Decoder>(
        decoder: D,
    ) -> Result<Self::Region<T, LEN>, DecodeError>;

    /// Deserialise the dynamic region.
    fn deserialise_dyn_region<D: Decoder>(decoder: D) -> Result<Self::DynRegion, DecodeError>;
}

/// Manager with the ability to clone regions
pub trait ManagerClone: ManagerBase {
    /// Clone the region.
    fn clone_region<E: Clone, const LEN: usize>(
        region: &Self::Region<E, LEN>,
    ) -> Self::Region<E, LEN>;

    /// Clone the dynamic region.
    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion;
}

/// Projection from [`ManagerBase::Region`] to the element type `E`
pub struct RegionProj<E, const LEN: usize>(PhantomData<E>);

impl<E: 'static, const LEN: usize> Projection for RegionProj<E, LEN> {
    type Subject = RegionCons<E, LEN>;

    type Target = E;

    // The parameter needs to be a tuple to allow better composition via the `tuples` crate.
    type Parameter = (usize,);

    #[inline]
    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> &'a Self::Target {
        M::region_ref(state, param.0)
    }

    #[inline]
    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + 'a>(
        state: &'a ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
    ) -> Self::Target
    where
        Self::Target: Copy,
    {
        M::region_read(state, param.0)
    }

    #[inline]
    fn project_write<'a, MC: MemoryConfig, M: ManagerWrite + 'a>(
        state: &'a mut ApplyCons<Self::Subject, MC, M>,
        param: Self::Parameter,
        value: Self::Target,
    ) {
        M::region_write(state, param.0, value);
    }

    fn normal_pointer_offset<MC: MemoryConfig>(param: Self::Parameter) -> ProjectionOffset {
        assert!(
            param.0 < LEN,
            "Region index out of bounds: {} >= {}",
            param.0,
            LEN
        );

        let offset = std::mem::size_of::<E>()
            .checked_mul(param.0)
            .expect("Region offset exceeds usize range");

        ProjectionOffset::direct(offset)
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use trait_set::trait_set;

    use super::ManagerAlloc;
    use super::ManagerClone;
    use super::ManagerRead;
    use super::ManagerWrite;
    use crate::machine_state::test_helpers::ManagerTestInit;

    /// Generate a test against all test backends.
    #[macro_export]
    macro_rules! backend_test {
        ( $(#[$m:meta])* $name:ident, $fac_name:ident, $expr:block ) => {
            $(#[$m])*
            #[test]
            fn $name() {
                use octez_riscv_data::mode::Normal;
                use octez_riscv_data::mode::Prove;
                use octez_riscv_data::mode::Verify;
                use $crate::state_backend::test_helpers::TestBackendFactory;

                fn inner<$fac_name: TestBackendFactory>() {
                    $expr
                }

                inner::<Normal>();
                inner::<Prove>();
                inner::<Verify>();
            }
        };
    }

    trait_set! {
        /// This lets you construct backends for any layout.
        ///
        /// Used for testing.
        pub trait TestBackendFactory = ManagerRead
            + ManagerWrite
            + ManagerClone
            + ManagerAlloc
            + ManagerTestInit
            ;
    }
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::serialisation::serialise;

    use super::*;
    use crate::state::NewState;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;

    #[test]
    fn test_example_normal() {
        struct Example<M: ManagerBase> {
            first: Cell<u64, M>,
            second: Cells<u32, 4, M>,
        }

        let first_value: u64 = rand::random();
        let second_value: [u32; 4] = rand::random();

        let mut instance: Example<Normal> = Example {
            first: Cell::new(),
            second: Cells::new(),
        };

        instance.first.write(first_value);
        assert_eq!(instance.first.read(), first_value);

        instance.second.write_all(&second_value);
        assert_eq!(instance.second.read_all(), second_value);

        let first_value_read =
            u64::from_le_bytes(serialise(instance.first).unwrap().try_into().unwrap());
        assert_eq!(first_value_read, first_value);

        let second_value_read = unsafe {
            let data = serialise(instance.second).unwrap();
            data.as_ptr().cast::<[u32; 4]>().read().map(u32::from_le)
        };
        assert_eq!(second_value_read, second_value);
    }
}
