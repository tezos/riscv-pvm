// SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

#[cfg(test)]
mod tests {
    use octez_riscv_data::foldable::Fold;
    use octez_riscv_data::foldable::Foldable;
    use octez_riscv_data::foldable::NodeFold;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Normal;

    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::state::NewState;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;
    use crate::state_backend::ManagerBase;
    use crate::state_backend::ProofPart;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned;
    use crate::state_backend::verify_backend::handle_stepper_panics;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MyFoo(u64);

    impl ConstDefault for MyFoo {
        const DEFAULT: Self = MyFoo(42);
    }

    // Test that the Atom layout initialises the underlying Cell correctly.
    backend_test!(test_cell_init, F, {
        assert_eq!(Cell::<MyFoo, F>::new().read(), MyFoo::DEFAULT);
    });

    // Test that the Array layout initialises the underlying Cells correctly.
    backend_test!(test_cells_init, F, {
        assert_eq!(
            Cells::<MyFoo, 1337, F>::new().read_all(),
            [MyFoo::DEFAULT; 1337]
        );
    });

    #[test]
    fn test_struct_example() {
        struct Foo<M: ManagerBase> {
            bar: Cell<u64, M>,
            qux: Cells<u8, 64, M>,
        }

        impl<F: Fold, M: ManagerBase> Foldable<F> for Foo<M>
        where
            Cell<u64, M>: Foldable<F>,
            Cells<u8, 64, M>: Foldable<F>,
        {
            fn fold(&self, builder: F) -> <F as Fold>::Folded {
                let mut builder = builder.into_node_fold();
                builder.add(&self.bar);
                builder.add(&self.qux);
                builder.done()
            }
        }

        fn inner(bar: u64, qux: [u8; 64]) {
            let mut foo = Foo::<Normal> {
                bar: Cell::new(),
                qux: Cells::new(),
            };

            foo.bar.write(bar);
            foo.qux.write_all(&qux);

            // Obtain the state hash
            let hash = Hash::from_foldable(&foo);

            // Obtain the Merkle tree via the `Prove` mode
            let mut proof_foo = Foo {
                bar: foo.bar.start_proof(),
                qux: foo.qux.start_proof(),
            };

            let tree = MerkleTree::from_foldable(&proof_foo);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Modify the values so they appear in the proof
            proof_foo.bar.write(bar.wrapping_add(1));
            proof_foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));

            // Obtain the Merkle tree, again, to make sure the root hash has not changed
            let tree = MerkleTree::from_foldable(&proof_foo);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Produce a proof
            let proof = merkle_tree_to_merkle_proof(tree);
            let proof_hash = proof.root_hash();
            assert_eq!(hash, proof_hash);

            // Apply the same modification on the `Normal` state in order to obtain
            // the final state hash
            foo.bar.write(bar.wrapping_add(1));
            foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));
            let final_hash = Hash::from_foldable(&foo);

            // Verify the proof and check the final hash
            handle_stepper_panics(|| {
                let mut verify_foo = {
                    let (bar, qux) = deserialise_owned::deserialise(ProofPart::Present(&proof))
                        .unwrap()
                        .0;
                    Foo { bar, qux }
                };

                assert_eq!(bar, verify_foo.bar.read());
                assert_eq!(qux, verify_foo.qux.read_all().as_slice());

                // Apply the same modification to the state in `Verify` mode and check
                // that the final hash is correct
                verify_foo.bar.write(bar.wrapping_add(1));
                verify_foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));

                let verify_hash = PartialHash::from_foldable(Some(&proof), &verify_foo)
                    .to_hash()
                    .unwrap();
                assert_eq!(verify_hash, final_hash)
            })
            .unwrap();
        }

        proptest::proptest!(|(bar: u64, qux: [u8; 64])| {
            inner(bar, qux);
        });
    }
}
