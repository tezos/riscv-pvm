// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fmt::Debug;
use std::sync::Arc;

use bytes::Bytes;
use octez_riscv_data::serialisation::serialise_into;

use super::Key;
use super::node::MavlNode;
use super::node_operations::delete;
use super::node_operations::get;
use super::node_operations::set;
use crate::commit_operation::CommitOperationCollection;

/// A key-value store tree with left and right nodes that supports traversal and value retrieval.
#[derive(Clone, Default, Debug)]
pub(super) struct Avl {
    root: Option<Arc<MavlNode>>,
}

impl Avl {
    /// Delete the node in the tree with a given key.
    pub(super) fn delete(&mut self, key: &Key) -> bool {
        delete(&mut self.root, key)
    }

    /// The data stored in a node in the tree with a given key.
    pub(super) fn get(&self, key: &Key) -> Option<&Bytes> {
        get(&self.root, key)
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub fn hash(&mut self) -> blake3::Hash {
        let encodable = self.root.as_deref().map(|node| node.to_encode());
        let mut hasher = blake3::Hasher::new();
        serialise_into(encodable, &mut hasher)
            .expect("None of the `EncodeError`s can be triggered by this encoding");
        hasher.finalize()
    }

    /// The root node of the tree.
    #[cfg(test)]
    pub(super) fn root(&self) -> &Option<Arc<MavlNode>> {
        &self.root
    }

    /// A mutable reference to the root node of the tree.
    pub(super) fn root_mut(&mut self) -> &mut Option<Arc<MavlNode>> {
        &mut self.root
    }

    /// Set the value of a node in the tree with a given key.
    pub(super) fn set(
        &mut self,
        key: &Key,
        data: Bytes,
        commit_collection: &mut CommitOperationCollection,
    ) {
        set(&mut self.root, key, data, commit_collection);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Bytes;
    use proptest::prelude::*;

    use super::*;
    use crate::commit_operation::CommitOperationCollection;
    use crate::merkle_layer::KEY_MAX_SIZE;
    use crate::merkle_layer::Key;

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

    struct AvlIterator<'a> {
        stack: Vec<&'a Arc<MavlNode>>,
        current: &'a Option<Arc<MavlNode>>,
    }

    fn height_and_balance_factor_sanity_check_helper(node: Arc<MavlNode>) -> (bool, usize) {
        let (left_good, left_height) = match node.left_ref() {
            None => (true, 0),
            Some(left_node) => height_and_balance_factor_sanity_check_helper(left_node.clone()),
        };
        if !left_good {
            return (false, 0);
        }
        let (right_good, right_height) = match node.right_ref() {
            None => (true, 0),
            Some(right_node) => height_and_balance_factor_sanity_check_helper(right_node.clone()),
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

    impl Avl {
        fn iter(&self) -> AvlIterator {
            match &self.root {
                None => AvlIterator {
                    stack: vec![],
                    current: &None,
                },
                Some(_) => AvlIterator {
                    stack: vec![],
                    current: &self.root,
                },
            }
        }

        pub fn height_and_balance_factor_sanity_check(&self) -> bool {
            match &self.root {
                None => true,
                Some(node) => {
                    let (ret, _) = height_and_balance_factor_sanity_check_helper(node.clone());
                    ret
                }
            }
        }
    }

    impl<'a> Iterator for AvlIterator<'a> {
        type Item = &'a Arc<MavlNode>;

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(node) = self.current {
                self.stack.push(node);
                self.current = node.left_ref();
            }

            let ret = self.stack.pop()?;
            self.current = ret.right_ref();
            Some(ret)
        }
    }

    fn compare_tree_to_reference(tree: &Avl, reference: &BTreeMap<Key, Bytes>) {
        let tree_iter = tree.iter();
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
            let mut tree: Avl = Default::default();
            let mut commit_collection: CommitOperationCollection = Default::default();
            let mut reference: BTreeMap<Key, Bytes> = BTreeMap::new();
            for operation in operations {
                match operation {
                    Operation::Get(key) => {
                        let tree_value = tree.get(&key);
                        let reference_value = reference.get(&key);
                        assert_eq!(tree_value, reference_value);
                        continue;
                    },
                    Operation::Upsert(key, value) => {
                        tree.set(&key, value.clone(), &mut commit_collection);
                        reference.insert(key, value);
                    }
                    Operation::Delete(key) => {
                        tree.delete(&key);
                        reference.remove(&key);
                    }
                }
                compare_tree_to_reference(&tree, &reference);
                assert!(tree.height_and_balance_factor_sanity_check());
            }
        }
    }
}
