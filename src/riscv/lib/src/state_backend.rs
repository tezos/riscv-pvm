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

use bincode::enc::Encoder;
use bincode::error::EncodeError;
pub use elems::*;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::serialisation::elem::Elem;
pub use proof_layout::*;
pub use region::*;

/// Manager of the state backend storage
pub trait ManagerBase: Mode + Sized {
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
    /// Allocate a dynamic region in the state storage.
    fn allocate_dyn_region(len: usize) -> Self::DynRegion;
}

/// Manager with read capabilities
pub trait ManagerRead: ManagerBase + AtomMode {
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
pub trait ManagerWrite: ManagerBase<ManagerRoot = Self> + AtomMode {
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
pub trait ManagerSerialise: ManagerRead + EncodeAtomMode {
    /// Serialise the contents of the dynamic region.
    fn serialise_dyn_region<E: Encoder>(
        region: &Self::DynRegion,
        encoder: E,
    ) -> Result<(), EncodeError>;
}

/// Manager with the ability to clone regions
pub trait ManagerClone: ManagerBase + CloneAtomMode {
    /// Clone the dynamic region.
    fn clone_dyn_region(region: &Self::DynRegion) -> Self::DynRegion;
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
    use std::num::NonZeroUsize;

    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use rand::RngCore;

    use super::*;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned::ProofTreeDeserialiser;

    /// Data structure whose [`Elem`] implementation only writes to part of the given space
    #[derive(Clone)]
    struct PartialWrite {
        a: u32,
        b: u64,
    }

    impl Elem for PartialWrite {
        const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(16).unwrap();

        unsafe fn write_unaligned(self, dest: *mut u8) {
            unsafe {
                dest.cast::<u32>().write(self.a);
                dest.add(8).cast::<u64>().write(self.b);
            }
        }

        unsafe fn read_unaligned(source: *const u8) -> Self {
            unsafe {
                let a = source.cast::<u32>().read();
                let b = source.add(8).cast::<u64>().read();
                PartialWrite { a, b }
            }
        }
    }

    #[test]
    fn test_partial_elem_impls() {
        const LEN: usize = 4096;

        let mut mem_normal = DynCells::new(LEN);

        // Randomise the contents
        let mut rand_buffer = [0u8; LEN];
        rand::rng().fill_bytes(&mut rand_buffer);
        mem_normal.write_all(0, &rand_buffer);

        // State in Prove mode needs to be derived from the Normal mode state
        let mem_prove_source = mem_normal.clone();
        let mut mem_prove = mem_prove_source.start_proof();

        // This is the value that we want to write in all modes
        let value = PartialWrite {
            a: 0xDEADBEEF,
            b: 0x1122334455667788,
        };

        unsafe {
            // Perform the writes in Normal and Prove mode to start out, Verify will follow
            mem_normal.write(0, value.clone());
            mem_prove.write(0, value.clone());
        }

        // The Verify mode needs a proof, so we generate it from the Prove mode
        let merkle_tree = MerkleTree::from_foldable(&mem_prove);
        let proof_tree = merkle_tree_to_merkle_proof(merkle_tree);
        let proof_deser = ProofTreeDeserialiser::from(ProofTree::Present(&proof_tree));
        let mut mem_verify = DynCells::from_proof(proof_deser).unwrap().into_result();

        unsafe {
            // Finally, also perform the write in Verify mode
            mem_verify.write(0, value);
        }

        // Normal and Prove mode should be identical
        let hash_normal = Hash::from_foldable(&mem_normal);
        let hash_prove = Hash::from_foldable(&mem_prove);
        assert_eq!(hash_normal, hash_prove);

        // Normal and Verify mode should be identical
        let hash_verify = PartialHash::from_foldable(Some(&proof_tree), &mem_verify)
            .to_hash()
            .unwrap();
        assert_eq!(hash_normal, hash_verify);
    }
}
