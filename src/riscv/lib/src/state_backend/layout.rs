// SPDX-FileCopyrightText: 2023 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::marker::PhantomData;

/// Structural description of a state type
pub trait Layout {
    /// Representation of the allocated regions in the state backend
    type Allocated<M: super::ManagerBase>;
}

impl Layout for () {
    type Allocated<M: super::ManagerBase> = ();
}

impl<T: Layout> Layout for Box<T> {
    type Allocated<M: super::ManagerBase> = Box<T::Allocated<M>>;
}

/// `L::Allocated`
pub type AllocatedOf<L, M> = <L as Layout>::Allocated<M>;

/// Layout for a single value
#[repr(transparent)]
pub struct Atom<T> {
    _pd: PhantomData<T>,
}

impl<T: 'static> Layout for Atom<T> {
    type Allocated<M: super::ManagerBase> = super::Cell<T, M>;
}

/// Layout for a fixed number of values
#[repr(transparent)]
pub struct Array<T, const LEN: usize> {
    _pd: PhantomData<T>,
}

impl<T: 'static, const LEN: usize> Layout for Array<T, LEN> {
    type Allocated<M: super::ManagerBase> = super::Cells<T, LEN, M>;
}

/// Layout for a fixed number of bytes, readable as types implementing [`super::elems::Elem`].
pub struct DynArray {}

impl Layout for DynArray {
    type Allocated<M: super::ManagerBase> = super::DynCells<M>;
}

/// Usage: Provide a struct with each field holding a layout.
///
/// ```ignore
/// use octez_riscv::state_backend::*;
/// use octez_riscv::machine_state::csregisters::CSRRepr;
/// use octez_riscv::struct_layout;
///
/// struct_layout! {
///     pub struct ExampleLayout {
///         satp_ppn: Atom<CSRRepr>,
///         mode: Atom<u8>,
///         cached: Atom<bool>,
///     }
/// }
/// ```
#[macro_export]
macro_rules! struct_layout {
    (
        $(
            #[$attributes:meta]
        )*
        $vis:vis struct $layout_t:ident $(< $($param:ident),+ >)? {
            $($field_vis:vis $field_name:ident: $cell_repr:ty),+
            $(,)?
        }
    ) => {
        paste::paste! {
            #[derive(bincode::Encode, bincode::Decode, Debug, Clone, PartialEq, Eq)]
            $(
                #[$attributes]
            )*
            $vis struct [<$layout_t F>]<
                $(
                    [<$field_name:camel>]
                ),+
            > {
                $(
                    $field_vis $field_name: [<$field_name:camel>]
                ),+
            }

            impl <
                __F: octez_riscv_data::foldable::Fold,
                $(
                    [<$field_name:camel>]: octez_riscv_data::foldable::Foldable<__F>
                ),+
            > octez_riscv_data::foldable::Foldable<__F> for [<$layout_t F>]<
                $(
                    [<$field_name:camel>]
                ),+
            > {
                fn fold(&self, builder: __F) -> __F::Folded {
                    use octez_riscv_data::foldable::NodeFold;

                    let mut builder = builder.into_node_fold();

                    $(
                        builder.add(&self.$field_name);
                    )+

                    builder.done()
                }
            }

            $vis type $layout_t $(< $($param),+ >)? = [<$layout_t F>]<
                $(
                    $cell_repr
                ),+
            >;

            impl <
                $(
                    [<$field_name:camel>]: $crate::state_backend::Layout
                ),+
            > $crate::state_backend::Layout for [<$layout_t F>]<
                $(
                    [<$field_name:camel>]
                ),+
            > {
                type Allocated<M: $crate::state_backend::ManagerBase> = [<$layout_t F>]<
                    $(
                        <[<$field_name:camel>] as $crate::state_backend::Layout>::Allocated<M>
                    ),+
                >;

            }
        }
    };
}

impl<A, B> Layout for (A, B)
where
    A: Layout,
    B: Layout,
{
    type Allocated<M: super::ManagerBase> = (A::Allocated<M>, B::Allocated<M>);
}

impl<A, B, C> Layout for (A, B, C)
where
    A: Layout,
    B: Layout,
    C: Layout,
{
    type Allocated<M: super::ManagerBase> = (A::Allocated<M>, B::Allocated<M>, C::Allocated<M>);
}

impl<A, B, C, D> Layout for (A, B, C, D)
where
    A: Layout,
    B: Layout,
    C: Layout,
    D: Layout,
{
    type Allocated<M: super::ManagerBase> = (
        A::Allocated<M>,
        B::Allocated<M>,
        C::Allocated<M>,
        D::Allocated<M>,
    );
}

impl<A, B, C, D, E> Layout for (A, B, C, D, E)
where
    A: Layout,
    B: Layout,
    C: Layout,
    D: Layout,
    E: Layout,
{
    type Allocated<M: super::ManagerBase> = (
        A::Allocated<M>,
        B::Allocated<M>,
        C::Allocated<M>,
        D::Allocated<M>,
        E::Allocated<M>,
    );
}

impl<A, B, C, D, E, F> Layout for (A, B, C, D, E, F)
where
    A: Layout,
    B: Layout,
    C: Layout,
    D: Layout,
    E: Layout,
    F: Layout,
{
    type Allocated<M: super::ManagerBase> = (
        A::Allocated<M>,
        B::Allocated<M>,
        C::Allocated<M>,
        D::Allocated<M>,
        E::Allocated<M>,
        F::Allocated<M>,
    );
}

impl<T, const LEN: usize> Layout for [T; LEN]
where
    T: Layout,
{
    type Allocated<M: super::ManagerBase> = [T::Allocated<M>; LEN];
}

/// This [`Layout`] is identical to [`[T; LEN]`] but it allows you to choose a very high `LEN`.
pub struct Many<T: Layout, const LEN: usize>(PhantomData<[T; LEN]>);

impl<T, const LEN: usize> Layout for Many<T, LEN>
where
    T: Layout,
{
    type Allocated<M: super::ManagerBase> = Vec<T::Allocated<M>>;
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::Verify;

    use super::*;
    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::state::NewState;
    use crate::state_backend::Cell;
    use crate::state_backend::Cells;
    use crate::state_backend::ProofPart;
    use crate::state_backend::proof_backend::ProofWrapper;
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
    fn test_struct_layout() {
        struct_layout! {
            pub struct Foo {
                bar: Atom<u64>,
                qux: Array<u8, 64>,
            }
        }

        fn inner(bar: u64, qux: [u8; 64]) {
            let mut foo = AllocatedOf::<Foo, Normal> {
                bar: Cell::new(),
                qux: Cells::new(),
            };

            foo.bar.write(bar);
            foo.qux.write_all(&qux);

            // Obtain the state hash
            let refs = FooF {
                bar: &foo.bar,
                qux: &foo.qux,
            };
            let hash = Hash::from_foldable(&refs);

            // Obtain the Merkle tree via the `Prove` mode
            let mut proof_foo = FooF {
                bar: foo.bar.struct_ref::<ProofWrapper>(),
                qux: foo.qux.struct_ref::<ProofWrapper>(),
            };
            let proof_foo_refs = FooF {
                bar: &proof_foo.bar,
                qux: &proof_foo.qux,
            };

            let tree = MerkleTree::from_foldable(&proof_foo_refs);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Modify the values so they appear in the proof
            proof_foo.bar.write(bar.wrapping_add(1));
            proof_foo.qux.write_all(&qux.map(|x| x.wrapping_add(1)));

            // Obtain the Merkle tree, again, to make sure the root hash has not changed
            let proof_foo_refs = FooF {
                bar: &proof_foo.bar,
                qux: &proof_foo.qux,
            };

            let tree = MerkleTree::from_foldable(&proof_foo_refs);
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
            let refs = FooF {
                bar: &foo.bar,
                qux: &foo.qux,
            };
            let final_hash = Hash::from_foldable(&refs);

            // Verify the proof and check the final hash
            handle_stepper_panics(|| {
                let mut verify_foo = {
                    let (bar, qux) = deserialise_owned::deserialise(ProofPart::Present(&proof))
                        .unwrap()
                        .0;
                    AllocatedOf::<Foo, Verify> { bar, qux }
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
