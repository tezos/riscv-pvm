// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fmt::Debug;
use std::sync::Arc;

use bytes::Bytes;
use bytes::BytesMut;
use octez_riscv_data::hash::Hash;

use super::Key;
use super::node::MavlNode;
use super::node::delete;
use super::node::get;
use super::node::get_mut;
use super::node::set;
use super::node_resolver::MavlNodeResolver;
use super::node_wrapper::MavlNodeWrapper;

/// A key-value store tree with left and right nodes that supports traversal and value retrieval.
#[derive(Clone, Debug)]
pub struct Avl<Resolver: MavlNodeResolver> {
    root: Resolver::NodeWrapper,
}

impl<Resolver: MavlNodeResolver> Avl<Resolver> {
    pub fn new() -> Self {
        Self {
            root: Resolver::NodeWrapper::new(None),
        }
    }

    /// Delete the node in the tree with a given key.
    pub fn delete(&mut self, key: &Key, node_resolver: &Resolver) -> bool {
        delete(
            self.root
                .try_borrow_mut()
                .expect("Resolving the root should not fail"),
            key,
            node_resolver,
        )
    }

    /// The data stored in a node in the tree with a given key.
    pub fn get(&self, key: &Key, node_resolver: &Resolver) -> Option<&BytesMut> {
        get(
            self.root
                .try_borrow()
                .expect("Resolving the root should not fail"),
            key,
            node_resolver,
        )
    }

    /// A mutable reference to the data stored in a node in the tree with a given key.
    pub(super) fn get_mut(&mut self, key: &Key, node_resolver: &Resolver) -> Option<&mut BytesMut> {
        get_mut(
            self.root
                .try_borrow_mut()
                .expect("Resolving the root should not fail"),
            key,
            node_resolver,
        )
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub fn hash(&self, node_resolver: &Resolver) -> Hash {
        let encodable = self
            .root
            .try_borrow()
            .expect("Resolving the root should not fail")
            .as_ref()
            .map(|node| node.to_encode(node_resolver));
        Hash::hash_encodable(encodable).expect("Should be hashable")
    }

    /// Creates an in order iterator for the nodes in the tree
    pub(super) fn iter<'a>(&'a self, node_resolver: &'a Resolver) -> AvlIterator<'a, Resolver> {
        match self
            .root
            .try_borrow()
            .expect("Resolving the root should not fail")
        {
            None => AvlIterator {
                stack: vec![],
                current: &None,
                node_resolver,
            },
            Some(_) => AvlIterator {
                stack: vec![],
                current: self
                    .root
                    .try_borrow()
                    .expect("Resolving the root should not fail"),
                node_resolver,
            },
        }
    }

    /// The root node of the tree.
    #[cfg(test)]
    pub(super) fn root(&self, node_resolver: &Resolver) -> &Option<Arc<MavlNode<Resolver>>> {
        node_resolver.resolve(&self.root);
        self.root.try_borrow().unwrap()
    }

    /// A mutable reference to the root node of the tree.
    pub(super) fn root_mut(
        &mut self,
        node_resolver: &Resolver,
    ) -> &mut Option<Arc<MavlNode<Resolver>>> {
        node_resolver.resolve(&self.root);
        self.root
            .try_borrow_mut()
            .expect("Resolving the root should not fail")
    }

    /// Set the value of a node in the tree with a given key.
    pub fn set(&mut self, key: &Key, data: Bytes, node_resolver: &Resolver) {
        set(
            self.root
                .try_borrow_mut()
                .expect("Resolving the root should not fail"),
            key,
            data,
            node_resolver,
        );
    }
}

/// Used for iterating through the nodes
/// of the [`Avl`] tree in order.
pub(super) struct AvlIterator<'a, Resolver: MavlNodeResolver> {
    stack: Vec<&'a Arc<MavlNode<Resolver>>>,
    current: &'a Option<Arc<MavlNode<Resolver>>>,
    node_resolver: &'a Resolver,
}

impl<'a, Resolver: MavlNodeResolver> Iterator for AvlIterator<'a, Resolver> {
    type Item = &'a Arc<MavlNode<Resolver>>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.current {
            self.stack.push(node);
            self.current = node.left_ref(self.node_resolver);
        }

        let ret = self.stack.pop()?;
        self.current = ret.right_ref(self.node_resolver);
        Some(ret)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Bytes;
    use proptest::prelude::*;

    use super::*;
    use crate::merkle_layer::KEY_MAX_SIZE;
    use crate::merkle_layer::Key;
    use crate::merkle_layer::node_resolver::InMemoryMavlNodeResolver;

    #[derive(Debug, Clone)]
    enum Operation {
        Get(Key),
        Upsert(Key, Bytes),
        Delete(Key),
    }

    fn key_strategy() -> impl Strategy<Value = Key> {
        proptest::collection::vec(any::<u8>(), 1usize..=KEY_MAX_SIZE).prop_map(Key)
    }

    fn value_strategy() -> impl Strategy<Value = Bytes> {
        proptest::collection::vec(any::<u8>(), 1usize..=200usize).prop_map(Bytes::from)
    }

    fn operations_strategy(length: usize) -> impl Strategy<Value = Vec<Operation>> {
        let count = length.div_ceil(10);
        (
            proptest::collection::vec(key_strategy(), count),
            proptest::collection::vec(value_strategy(), count),
        )
            .prop_flat_map(move |(keys, values)| {
                proptest::collection::vec(
                    prop_oneof![
                        proptest::sample::select(keys.clone()).prop_map(Operation::Get),
                        (
                            proptest::sample::select(keys.clone()),
                            proptest::sample::select(values)
                        )
                            .prop_map(|(key, value)| Operation::Upsert(key, value)),
                        proptest::sample::select(keys).prop_map(Operation::Delete)
                    ],
                    length,
                )
            })
    }

    fn height_and_balance_factor_sanity_check_helper<Resolver: MavlNodeResolver>(
        node: Arc<MavlNode<Resolver>>,
        node_resolver: &Resolver,
    ) -> (bool, usize) {
        let (left_good, left_height) = match node.left_ref(node_resolver) {
            None => (true, 0),
            Some(left_node) => {
                height_and_balance_factor_sanity_check_helper(left_node.clone(), node_resolver)
            }
        };
        if !left_good {
            return (false, 0);
        }
        let (right_good, right_height) = match node.right_ref(node_resolver) {
            None => (true, 0),
            Some(right_node) => {
                height_and_balance_factor_sanity_check_helper(right_node.clone(), node_resolver)
            }
        };
        if !(right_good) {
            return (false, 0);
        }
        let balance_factor = (right_height as i64) - (left_height as i64);
        if balance_factor.abs() > 1 || node.balance_factor() != balance_factor {
            (false, 0)
        } else {
            (true, std::cmp::max(left_height, right_height) + 1)
        }
    }

    impl<Resolver: MavlNodeResolver> Avl<Resolver> {
        pub fn height_and_balance_factor_sanity_check(&self, node_resolver: &Resolver) -> bool {
            match self
                .root
                .try_borrow()
                .expect("Resolving the root should not fail")
            {
                None => true,
                Some(node) => {
                    let (ret, _) =
                        height_and_balance_factor_sanity_check_helper(node.clone(), node_resolver);
                    ret
                }
            }
        }
    }

    fn compare_tree_to_reference<Resolver: MavlNodeResolver>(
        tree: &Avl<Resolver>,
        reference: &BTreeMap<Key, Bytes>,
        node_resolver: &Resolver,
    ) {
        let tree_iter = tree.iter(node_resolver);
        let mut reference_iter = reference.iter();
        for node in tree_iter {
            if let Some((key, value)) = reference_iter.next() {
                assert_eq!(node.key(), key);
                assert_eq!(node.data(), value);
            } else {
                panic!("The reference implementation has less elements than the tree");
            }
        }
        assert_eq!(
            reference_iter.next(),
            None,
            "The reference implementation has more elements than the tree"
        );
    }

    proptest! {
        #[test]
        fn avl_driver_test(operations in (1usize..500usize).prop_flat_map(operations_strategy)) {
            let node_resolver = Arc::new(InMemoryMavlNodeResolver::default());
            let mut tree: Avl<InMemoryMavlNodeResolver> = Avl::new();
            let mut reference: BTreeMap<Key, Bytes> = BTreeMap::new();
            for operation in operations {
                match operation {
                    Operation::Get(key) => {
                        let tree_value = tree.get(&key, node_resolver.as_ref()).map(|b| b.clone().freeze());
                        assert_eq!(tree_value.as_ref(), reference.get(&key));
                        continue;
                    },
                    Operation::Upsert(key, value) => {
                        tree.set(&key, value.clone(), node_resolver.as_ref());
                        reference.insert(key, value);
                    }
                    Operation::Delete(key) => {
                        tree.delete(&key, node_resolver.as_ref());
                        reference.remove(&key);
                    }
                }
                compare_tree_to_reference(&tree, &reference, &node_resolver);
                assert!(tree.height_and_balance_factor_sanity_check(node_resolver.as_ref()));
            }
        }
    }
}
