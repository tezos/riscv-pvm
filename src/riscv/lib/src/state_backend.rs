// SPDX-FileCopyrightText: 2023-2026 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2026 Nomadic Labs <contact@nomadic-labs.com>
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
pub use proof_layout::*;

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use octez_riscv_data::components::atom::Atom;
    use octez_riscv_data::components::atom::AtomMode;
    use octez_riscv_data::components::bytes::Bytes;
    use octez_riscv_data::components::bytes::BytesMode;
    use octez_riscv_data::components::data_space::DataSpace;
    use octez_riscv_data::foldable::Fold;
    use octez_riscv_data::foldable::Foldable;
    use octez_riscv_data::foldable::NodeFold;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::HashFold;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::hash::PartialHashFold;
    use octez_riscv_data::merkle_proof::DeserialiserNode;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::merkle_tree::MerkleTreeFold;
    use octez_riscv_data::mode::Mode;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::mode::utils::catch_not_found;
    use octez_riscv_data::serialisation::elem::Elem;
    use proptest::collection::vec;
    use proptest::prelude::Just;
    use proptest::prelude::Strategy;
    use proptest::prelude::any;
    use proptest::prop_assert_eq;
    use proptest::prop_oneof;
    use proptest::proptest;
    use rand::RngCore;

    use super::*;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_compressed_merkle_tree;
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
        let proof_tree = merkle_tree_to_compressed_merkle_tree(merkle_tree).to_proof();
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

    #[ignore = "RV-895: Enable once dynamic state component creation is supported in Prove/Verify modes"]
    #[test]
    fn dyn_atom_creation_rv_895() {
        struct Foo<M: Mode> {
            bar: Atom<u64, M>,
        }

        fn operation<M: AtomMode>(foo: &mut Foo<M>) {
            // This would work:
            // foo.bar.write(foo.bar.read() * 2);

            // This does not:
            foo.bar = Atom::new(foo.bar.read() * 2);
        }

        impl Foo<Normal> {
            fn start_proof(&self) -> Foo<Prove<'_>> {
                Foo {
                    bar: self.bar.start_proof(),
                }
            }
        }

        impl Foldable<HashFold> for Foo<Prove<'_>> {
            fn fold(&self, builder: HashFold) -> Hash {
                let mut node = builder.into_node_fold();
                node.add(&self.bar);
                node.done()
            }
        }

        impl Foldable<MerkleTreeFold> for Foo<Prove<'_>> {
            fn fold(&self, builder: MerkleTreeFold) -> MerkleTree {
                let mut node = builder.into_node_fold();
                node.add(&self.bar);
                node.done()
            }
        }

        impl Foldable<PartialHashFold<'_>> for Foo<Verify> {
            fn fold(&self, builder: PartialHashFold) -> PartialHash {
                let mut node = builder.into_node_fold();
                node.add(&self.bar);
                node.done()
            }
        }

        impl FromProof for Foo<Verify> {
            fn from_proof<Proof: octez_riscv_data::merkle_proof::Deserialiser>(
                proof: Proof,
            ) -> octez_riscv_data::merkle_proof::SuspendedResult<Proof, Self> {
                let node = proof.into_node()?;

                let (node, bar) = node.next_branch()?;

                let this = Foo { bar };
                node.done(this)
            }
        }

        let foo_normal = Foo {
            bar: Atom::new(1337),
        };

        let mut foo_prove = foo_normal.start_proof();

        operation(&mut foo_prove);

        let merkle_tree = MerkleTree::from_foldable(&foo_prove);
        let expected_hash = Hash::from_foldable(&foo_prove);

        let merkle_proof = merkle_tree_to_compressed_merkle_tree(merkle_tree).to_proof();
        let proof_deser = ProofTreeDeserialiser::from(ProofTree::Present(&merkle_proof));

        let mut foo_verify = Foo::<Verify>::from_proof(proof_deser)
            .unwrap()
            .into_result();

        let final_hash = catch_not_found(move || {
            operation(&mut foo_verify);
            PartialHash::from_foldable(Some(&merkle_proof), &foo_verify)
                .to_hash()
                .unwrap()
        })
        .unwrap();

        assert_eq!(expected_hash, final_hash);
    }

    // Bytes behaves the same across different modes
    #[test]
    fn bytes_are_same_across_modes() {
        /// Operations to be issued against the Bytes state component
        #[derive(Debug, Clone)]
        enum Op {
            Read { offset: usize, size: usize },
            Write { offset: usize, data: Vec<u8> },
            Len,
            Resize { new_size: usize },
        }

        /// Results of operations issued against the Bytes state component
        #[derive(Debug, PartialEq, Eq)]
        enum OpResult {
            Read { read: usize, data: Vec<u8> },
            Wrote { wrote: usize },
            Len { len: usize },
            Void,
        }

        // Strategies for generating operations to be issued against the Bytes state component
        let op_strat = prop_oneof![
            (0usize..100, 0usize..50).prop_map(|(offset, size)| Op::Read { offset, size }),
            (0usize..100, vec(any::<u8>(), 0..50))
                .prop_map(|(offset, data)| Op::Write { offset, data }),
            (0usize..150).prop_map(|new_size| Op::Resize { new_size }),
            Just(Op::Len),
        ];

        /// Run the given sequence of operations against the Bytes state component
        fn run_ops<M: BytesMode>(bytes: &mut Bytes<M>, ops: &[Op]) -> Vec<OpResult> {
            ops.iter()
                .map(|op| match op {
                    Op::Read { offset, size } => {
                        let mut data = vec![0u8; *size];
                        let read = bytes.read(*offset, &mut data);
                        OpResult::Read { read, data }
                    }

                    Op::Write { offset, data } => {
                        let wrote = bytes.write(*offset, data);
                        OpResult::Wrote { wrote }
                    }

                    Op::Resize { new_size } => {
                        bytes.resize(*new_size);
                        OpResult::Void
                    }

                    Op::Len => OpResult::Len { len: bytes.len() },
                })
                .collect()
        }

        proptest!(|(ops in vec(op_strat, 1..20))| {
            let mut bytes_normal = Bytes::<Normal>::new();
            let results_normal = run_ops(&mut bytes_normal, &ops);
            let hash_normal = Hash::from_foldable(&bytes_normal);

            let mut bytes_prove = Bytes::<Prove>::new();
            let results_prove = run_ops(&mut bytes_prove, &ops);
            prop_assert_eq!(&results_normal, &results_prove);

            let hash_prove = Hash::from_foldable(&bytes_prove);
            prop_assert_eq!(hash_normal, hash_prove);

            let merkle_tree = MerkleTree::from_foldable(&bytes_prove);
            let merkle_proof = merkle_tree_to_compressed_merkle_tree(merkle_tree).to_proof();

            let mut bytes_verify = Bytes::<Verify>::new();
            let results_verify = run_ops(&mut bytes_verify, &ops);
            prop_assert_eq!(results_normal, results_verify);

            let hash_verify =
                PartialHash::from_foldable(Some(&merkle_proof), &bytes_verify)
                    .to_hash()
                    .unwrap();
            prop_assert_eq!(hash_normal, hash_verify);
        });
    }
}
