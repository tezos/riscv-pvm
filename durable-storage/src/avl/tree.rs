// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for an optional root [`Node`] of a Merklisable AVL tree

use std::cmp::Ordering;

use octez_riscv_data::hash::Hash;
use perfect_derive::perfect_derive;

use super::node::Node;
use super::node::Value;
use crate::avl::resolver::Resolver;
use crate::errors::OperationalError;
#[cfg(test)]
use crate::key::KEY_MAX_SIZE;
use crate::key::Key;

/// A key-value store tree with left and right nodes that supports traversal and value retrieval.
#[perfect_derive(Clone, Default, Debug)]
#[derive(derive_more::From)]
pub struct Tree<Id>(Option<Id>);

impl<Id> Tree<Id> {
    /// Delete the [`Node`] in the [`Tree`] with a given key.
    ///
    /// Returns true if the [`Tree`] has shrunk in size.
    pub fn delete(
        &mut self,
        key: &Key,
        resolver: &mut impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError>
    where
        Id: Clone,
    {
        let old_balance_factor = self.balance_factor(resolver)?;
        let Some(node) = self.root_mut() else {
            // The key does not exist so nothing will happen.
            return Ok(false);
        };

        let resolved_node = resolver.resolve(node)?;
        match resolved_node.key().cmp(key) {
            Ordering::Equal => match (
                resolved_node.left_ref().root(),
                resolved_node.right_ref().root(),
            ) {
                (None, None) => {
                    self.take();
                    Ok(true)
                }
                (Some(left), None) => {
                    *node = left.clone();
                    Ok(true)
                }
                (None, Some(right)) => {
                    *node = right.clone();
                    Ok(true)
                }
                (Some(_), Some(_)) => {
                    let (new_node, shrank) = Node::replace_with_successor(node, resolver)?;
                    *node = new_node;
                    Ok(shrank)
                }
            },
            Ordering::Greater => {
                let node_mut = resolver.resolve_mut(node)?;
                let left_shrank = node_mut.left_mut().delete(key, resolver)?;
                *node_mut.balance_factor_mut() += if left_shrank { 1 } else { 0 };
                self.rebalance(resolver)?;
                Ok(old_balance_factor.abs() == 1 && self.balance_factor(resolver)? == 0)
            }
            Ordering::Less => {
                let node_mut = resolver.resolve_mut(node)?;
                let right_shrank = node_mut.right_mut().delete(key, resolver)?;
                *node_mut.balance_factor_mut() -= if right_shrank { 1 } else { 0 };
                self.rebalance(resolver)?;
                Ok(old_balance_factor.abs() == 1 && self.balance_factor(resolver)? == 0)
            }
        }
    }

    #[inline]
    /// Set the value of the [`Node`] with a given key.
    ///
    /// Returns true if the [`Tree`] has grown in size.
    pub fn set(
        &mut self,
        key: &Key,
        data: &[u8],
        resolver: &mut impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError>
    where
        Id: Clone + From<Node<Id>>,
    {
        self.upsert(key, 0, |old_data| old_data.set(data), resolver)
    }

    /// Returns the node [`struct@Hash`], potentially re-hashing uncached [`Node`]s.
    ///
    /// If the [`struct@Hash`] has been cached, the memo is returned. Otherwise, the
    /// [`struct@Hash`] is calculated and cached.
    pub(crate) fn hash(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<Hash, OperationalError> {
        let encodable = self
            .0
            .as_ref()
            .map(|id| {
                let node = resolver.resolve(id)?;
                let value = node.to_encode(resolver);
                Ok(value)
            })
            .transpose()?;
        Ok(Hash::hash_encodable(encodable).expect("Should be hashable"))
    }

    /// Creates an in-order iterator for the [`Node`]s in the [`Tree`]
    pub(crate) fn iter<'tree, 'res, Res: Resolver<Id, Node<Id>>>(
        &'tree self,
        resolver: &'res Res,
    ) -> TreeIterator<'tree, 'res, Id, Res> {
        TreeIterator {
            stack: vec![],
            current: self,
            resolver,
        }
    }

    /// Take the root [`Node`] out of this tree, leaving the [`Tree`] empty.
    pub(crate) const fn take(&mut self) -> Option<Id> {
        self.0.take()
    }

    #[inline]
    /// The difference in heights between any child branches in the [`Tree`].
    pub(super) fn balance_factor(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<i64, OperationalError> {
        let Some(node) = self.root() else {
            return Ok(0);
        };

        let resolved_node = resolver.resolve(node)?;
        let balance_factor = resolved_node.balance_factor();
        Ok(balance_factor)
    }

    #[inline]
    /// A reference to the root [`Node`].
    pub(super) fn root(&self) -> Option<&Id> {
        self.0.as_ref()
    }

    #[inline]
    /// A mutable reference to the root [`Node`].
    pub(super) fn root_mut(&mut self) -> Option<&mut Id> {
        self.0.as_mut()
    }

    /// Takes the occupied [`Tree`] with the minimum [`Key`] from this [`Tree`] and replaces it
    /// with an empty [`Tree`].
    ///
    /// Returns a tuple of:
    ///  - The occupied [`Tree`] with the minimum [`Key`].
    ///  - The minimum [`Tree`]'s right child, if it hasn't been moved to its new position.
    ///  - True if the [`Tree`] has shrunk in size.
    pub(super) fn take_min(
        &mut self,
        resolver: &mut impl Resolver<Id, Node<Id>>,
    ) -> Result<(Tree<Id>, Tree<Id>, bool), OperationalError>
    where
        Id: Clone,
    {
        let Some(node_arc) = self.root_mut() else {
            return Ok((None.into(), None.into(), false));
        };

        let node_mut = resolver.resolve_mut(node_arc)?;

        // Base case
        if node_mut.left_ref().root().is_none() {
            let right = node_mut.right_mut().take();
            Ok((self.take().into(), right.into(), true))
        // Recursive
        } else {
            Ok(Node::take_min(node_arc, resolver)?)
        }
    }

    /// Set or update the value of a [`Node`] in the [`Tree`] with a given key and given offset.
    ///
    /// `data` defines what data is upserted.
    ///
    /// Returns true if the [`Tree`] has grown in size.
    pub(crate) fn upsert(
        &mut self,
        key: &Key,
        offset: usize,
        data: impl FnOnce(&mut Value),
        resolver: &mut impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError>
    where
        Id: Clone + From<Node<Id>>,
    {
        let node = self.root_mut();
        let Some(node) = node else {
            // We can't create a new `Node` with a non-zero offset.
            //
            // This shouldn't happen: it's prevented by the `Database` API.
            assert_eq!(offset, 0);

            // TODO: RV-895: Dynamic creation of the `Bytes` (alias `Value`) state component may cause
            // problems with proof generation
            let mut new_data = Value::default();
            data(&mut new_data);

            // The key does not exist and a new `Node` shall be created.
            let new_node: Node<Id> = Node::new(key.clone(), new_data);
            let new_id = Id::from(new_node);
            self.0 = Some(new_id);

            return Ok(true);
        };
        let grew = resolver
            .resolve_mut(node)?
            .upsert(key, offset, data, resolver)?;
        if grew {
            self.rebalance(resolver)?;
            Ok(self.balance_factor(resolver)? != 0)
        } else {
            Ok(false)
        }
    }

    /// Writes the data to the [`Node`] in this [`Tree`] associated with a given [Key] with the
    /// given offset, overwriting existing data if the node already exists.
    ///
    /// Returns true if the [`Tree`] has grown in size.
    pub(crate) fn write(
        &mut self,
        key: &Key,
        offset: usize,
        data: &[u8],
        resolver: &mut impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError>
    where
        Id: Clone + From<Node<Id>>,
    {
        self.upsert(
            key,
            offset,
            |old_data| {
                // This shouldn't happen: it's prevented by the `Database` API.
                assert!(offset <= old_data.len());

                let Some(new_data_end) = offset.checked_add(data.len()) else {
                    // The asynchronous Merkle worker means that errors can't be returned to the
                    // `Database`.
                    panic!(
                        "Offset + data.len() overflows (`{offset:?}` + `{:?}`)",
                        data.len()
                    );
                };

                let final_len = std::cmp::max(old_data.len(), new_data_end);
                old_data.resize(final_len);
                old_data.write(offset, data);
            },
            resolver,
        )
    }

    /// Rebalance the [`Tree`] so that the difference in height between any child branches is in
    /// the range of -1..=1.
    ///
    /// The [`Tree`] must already have balance factor in the range of -2..=2, else it is an invalid
    /// AVL tree.
    fn rebalance(
        &mut self,
        resolver: &mut impl Resolver<Id, Node<Id>>,
    ) -> Result<(), OperationalError>
    where
        Id: Clone,
    {
        match self.root_mut() {
            Some(node) => Node::rebalance(node, resolver),
            None => Ok(()),
        }
    }
}

/// Used for iterating through the nodes of the [`Tree`] tree in order.
pub(crate) struct TreeIterator<'tree, 'res, Id, Res> {
    stack: Vec<&'tree Id>,
    current: &'tree Tree<Id>,
    resolver: &'res Res,
}

impl<'tree, 'res, Id, Res: Resolver<Id, Node<Id>>> Iterator for TreeIterator<'tree, 'res, Id, Res> {
    type Item = &'tree Id;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.current.root() {
            self.stack.push(node);

            let resolved_node = self.resolver.resolve(node).ok()?;
            self.current = resolved_node.left_ref();
        }

        let ret = self.stack.pop()?;
        let resolved_node = self.resolver.resolve(ret).ok()?;
        self.current = resolved_node.right_ref();

        Some(ret)
    }
}

#[cfg(test)]
impl<Id> Tree<Id> {
    #[inline]
    /// The data stored in a [`Node`] in the [`Tree`] with a given [`Key`].
    pub fn get(
        &self,
        key: &Key,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<Option<&Value>, OperationalError> {
        let Some(node) = self.root() else {
            return Ok(None);
        };
        Node::get(node, key, resolver)
    }

    /// Asserts that the [`Tree`] is a valid AVL tree
    pub(crate) fn check(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<(), OperationalError>
    where
        Id: std::fmt::Debug,
    {
        let inorder = self.is_inorder(resolver)?;
        let is_balanced = self.is_balanced(resolver)?;
        let has_correct_balance_factors = self.has_correct_balance_factors(resolver)?;
        if !inorder || !is_balanced || !has_correct_balance_factors {
            eprintln!("{self:?}");
        }
        assert!(inorder, "AVL tree isn't in order");
        assert!(is_balanced, "AVL tree isn't balanced");
        assert!(
            has_correct_balance_factors,
            "AVL tree balance factors are incorrect"
        );
        Ok(())
    }

    /// Returns true if the [`Tree`] is in-order.
    pub(crate) fn is_inorder(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError> {
        self.is_inorder_inner(
            &Key::new(&[u8::MIN]).expect("Size less than KEY_MAX_SIZE"),
            &Key::new(&[u8::MAX; KEY_MAX_SIZE]).expect("Size less than KEY_MAX_SIZE"),
            resolver,
        )
    }

    /// Returns true if the balance factors stored in any [`Node`]'s subtree are correct.
    pub(super) fn has_correct_balance_factors(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError>
    where
        Id: std::fmt::Debug,
    {
        match self.root() {
            None => Ok(true),
            Some(node) => resolver
                .resolve(node)
                .map(|res| res.has_correct_balance_factors(resolver))?,
        }
    }

    /// Returns the height of the [`Tree`].
    pub(super) fn height(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<u32, OperationalError> {
        match self.root() {
            None => Ok(0),
            Some(node) => resolver.resolve(node).map(|res| res.height(resolver))?,
        }
    }

    /// Returns true if the [`Tree`] is balanced.
    pub(super) fn is_balanced(
        &self,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError> {
        match self.root() {
            None => Ok(true),
            Some(node) => resolver
                .resolve(node)
                .map(|res| res.is_balanced(resolver))?,
        }
    }

    /// Returns true if the [`Tree`] is in-order and all values lie between the `min` and `max`.
    pub(super) fn is_inorder_inner(
        &self,
        min: &Key,
        max: &Key,
        resolver: &impl Resolver<Id, Node<Id>>,
    ) -> Result<bool, OperationalError> {
        match self.root() {
            None => Ok(true),
            Some(node) => resolver
                .resolve(node)
                .map(|res| res.is_inorder(min, max, resolver))?,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::prelude::*;

    use bytes::Bytes;
    use goldenfile::Mint;
    use proptest::prelude::*;

    use super::*;
    use crate::avl::resolver::ArcNodeId;
    use crate::avl::resolver::ArcResolver;
    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;

    const GOLDEN_DIR: &str = "tests/expected";

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

    fn compare_tree_to_reference(tree: &Tree<ArcNodeId>, reference: &BTreeMap<Key, Bytes>) {
        let resolver = ArcResolver;
        let tree_iter = tree.iter(&resolver);
        let mut reference_iter = reference.iter();
        for node in tree_iter {
            let node = resolver.resolve(node).unwrap();
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
            let mut tree: Tree<ArcNodeId> = Default::default();
            let mut reference: BTreeMap<Key, Bytes> = BTreeMap::new();
            let mut resolver = ArcResolver;
            for operation in operations {
                match operation {
                    Operation::Get(key) => {
                        let tree_value = tree.get(&key, &resolver)?;

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
                        tree.set(&key, &value, &mut resolver)?;
                        reference.insert(key, value);
                    }
                    Operation::Delete(key) => {
                        tree.delete(&key, &mut resolver)?;
                        reference.remove(&key);
                    }
                }
                compare_tree_to_reference(&tree, &reference);
                tree.check(&resolver)?;
            }
        }
    }

    #[test]
    fn test_hash_consistency() {
        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut resolver = ArcResolver;

        let data = ["42", "6 * 9", "1337", "31337"];

        // Create a collection of digests from a series of tree mutations
        let digests = {
            let mut digests: Vec<Hash> = [
                Key::new(&[42]),
                Key::new(&[6, 9]),
                Key::new(&[13, 37]),
                Key::new(&[31, 33, 7]),
            ]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"))
            .iter()
            .zip(data)
            .map(|(key, data)| -> Hash {
                let digest = tree
                    .hash(&resolver)
                    .expect("Resolving the tree should succeed.");
                tree.set(key, data.as_bytes(), &mut resolver)
                    .expect("Set should succeed");
                digest
            })
            .collect();

            digests.push(
                tree.hash(&resolver)
                    .expect("Resolving the tree should succeed."),
            );

            digests
        };

        let serialised = octez_riscv_data::serialisation::serialise(digests).unwrap();

        let mut mint = Mint::new(GOLDEN_DIR);
        let mut golden = mint.new_goldenfile("digests.out").unwrap();

        golden.write_all(&serialised).unwrap();
    }
}
