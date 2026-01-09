// SPDX-FileCopyrightText: 2023-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Generic state backends
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
pub mod proof_backend;
pub(crate) mod proof_layout;
mod region;
pub mod verify_backend;

pub use elems::*;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::data_space::CloneDataSpaceMode;
use octez_riscv_data::components::data_space::DataSpaceMode;
pub use proof_layout::*;
use trait_set::trait_set;

trait_set! {
    /// Manager with read capabilities
    pub trait ManagerRead = AtomMode + DataSpaceMode;

    /// Manager with write capabilities
    pub trait ManagerWrite = AtomMode + DataSpaceMode;

    /// Manager with the ability to clone regions
    pub trait ManagerClone = CloneAtomMode + CloneDataSpaceMode;
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use trait_set::trait_set;

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
            + ManagerTestInit
            ;
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use octez_riscv_data::components::data_space::DataSpace;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::serialisation::elem::Elem;
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

        let mut mem_normal = DataSpace::new(LEN);

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
        let mut mem_verify = DataSpace::from_proof(proof_deser).unwrap().into_result();

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
