// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for an optional root node of a Merklisable AVL tree

use std::fmt::Debug;
use std::sync::Arc;

use octez_riscv_data::hash::Hash;

use super::node::Node;
use super::node::Value;
use super::node::delete;
use super::node::get;
use super::node::get_mut;
use super::node::set;
use super::node::write;
use crate::key::Key;

/// A key-value store tree with left and right nodes that supports traversal and value retrieval.
#[derive(Clone, Default, Debug)]
pub struct Tree {
    root: Option<Arc<Node>>,
}

impl Tree {
    /// Delete the node in the tree with a given key.
    pub fn delete(&mut self, key: &Key) -> bool {
        delete(&mut self.root, key)
    }

    /// The data stored in a node in the tree with a given key.
    pub fn get(&self, key: &Key) -> Option<&Value> {
        get(&self.root, key)
    }

    /// A mutable reference to the data stored in a node in the tree with a given key.
    pub fn get_mut(&mut self, key: &Key) -> Option<&mut Value> {
        get_mut(&mut self.root, key)
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub(crate) fn hash(&self) -> Hash {
        let encodable = self.root.as_deref().map(|node| node.to_encode());
        Hash::hash_encodable(encodable).expect("Should be hashable")
    }

    /// Creates an in order iterator for the nodes in the tree
    pub(crate) fn iter(&self) -> TreeIterator {
        match &self.root {
            None => TreeIterator {
                stack: vec![],
                current: &None,
            },
            Some(_) => TreeIterator {
                stack: vec![],
                current: &self.root,
            },
        }
    }

    /// The root node of the tree.
    #[cfg(test)]
    pub(crate) fn root(&self) -> &Option<Arc<Node>> {
        &self.root
    }

    /// A mutable reference to the root node of the tree.
    pub(crate) fn root_mut(&mut self) -> &mut Option<Arc<Node>> {
        &mut self.root
    }

    /// Set the value of a node in the tree with a given key.
    pub fn set(&mut self, key: &Key, data: &[u8]) {
        set(&mut self.root, key, data);
    }

    /// Writes the data to the node associated with a given [Key] with the given offset.
    pub(crate) fn write(&mut self, key: &Key, offset: usize, data: &[u8]) {
        write(&mut self.root, key, offset, data);
    }
}

/// Used for iterating through the nodes
/// of the [`Tree`] tree in order.
pub(crate) struct TreeIterator<'a> {
    stack: Vec<&'a Arc<Node>>,
    current: &'a Option<Arc<Node>>,
}

impl<'a> Iterator for TreeIterator<'a> {
    type Item = &'a Arc<Node>;

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Bytes;
    use proptest::prelude::*;

    use super::*;
    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;

    #[derive(Debug, Clone)]
    enum Operation {
        Get(Key),
        Upsert(Key, Bytes),
        Delete(Key),
    }

    fn key_strategy() -> impl Strategy<Value = Key> {
        proptest::collection::vec(any::<u8>(), 1usize..=KEY_MAX_SIZE)
            .prop_map(|bytes| Key::new(&bytes).expect("bytes are a valid key"))
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

    fn height_and_balance_factor_sanity_check_helper(node: Arc<Node>) -> (bool, usize) {
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

    impl Tree {
        pub(crate) fn height_and_balance_factor_sanity_check(&self) -> bool {
            match &self.root {
                None => true,
                Some(node) => {
                    let (ret, _) = height_and_balance_factor_sanity_check_helper(node.clone());
                    ret
                }
            }
        }
    }

    fn compare_tree_to_reference(tree: &Tree, reference: &BTreeMap<Key, Bytes>) {
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
            let mut tree: Tree = Default::default();
            let mut reference: BTreeMap<Key, Bytes> = BTreeMap::new();
            for operation in operations {
                match operation {
                    Operation::Get(key) => {
                        let tree_value = tree.get(&key);

                        // The values in the Options are comparable, but the Options themselves are
                        // not. So we need to awkwardly match on both Options to compare them.
                        match (tree_value, reference.get(&key)) {
                            (Some(tree_bytes), Some(reference_bytes)) => {
                                assert_eq!(tree_bytes, reference_bytes)
                            }
                            (None, None) => {}
                            (lhs, rhs) => panic!(
                                "Mismatch between tree (is_none() = {}) and reference (is_none() = {}) for Get operation",
                                lhs.is_none(),
                                rhs.is_none(),
                            ),
                        }

                        continue;
                    }
                    Operation::Upsert(key, value) => {
                        tree.set(&key, &value);
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
