// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for a Merklisable node of an AVL tree

use std::cmp::Ordering;
use std::sync::OnceLock;

use bincode::Encode;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use perfect_derive::perfect_derive;

use super::tree::Tree;
use crate::avl::resolver::Resolver;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::key::Key;

/// Value stored in a node
pub type Value = Bytes<Normal>;

#[derive(Encode)]
/// A serialisable representation of [`Node`].
struct NodeHashRepresentation<'a, Value> {
    key: &'a Key,
    data: Value,
    // The bytes of the hash of an optional left child
    left: Option<&'a Hash>,
    // The bytes of the hash of an optional right child
    right: Option<&'a Hash>,
    balance_factor: i64,
}

/// A node that supports rebalancing and Merklisation.
#[perfect_derive(Clone, Default, Debug)]
pub struct Node<Id> {
    key: Key,
    data: Value,
    left: Tree<Id>,
    right: Tree<Id>,

    /// A cache for the hash of this node. This uses `OnceLock` so that updating the cache is a
    /// non-mutating operation.
    ///
    /// An uninitialised hash is a hash that has not been set or has been dirtied.
    hash: OnceLock<Hash>,

    /// The difference in heights between child branches (right - left).
    balance_factor: i64,
}

impl<Id> Node<Id> {
    /// Create a new leaf [`Node`] from the given key and data.
    pub(crate) fn new(key: Key, data: impl Into<Value>) -> Self {
        Node {
            key,
            data: data.into(),
            balance_factor: 0,
            ..Default::default()
        }
    }

    /// Converts the [`Node`] to an encoded, serialisable representation,
    /// [`NodeHashRepresentation`], potentially re-hashing uncached [`Node`]s.
    pub(crate) fn to_encode<'a>(&'a self, resolver: &impl Resolver<Id, Self>) -> impl Encode + 'a {
        // Recursively hashes any left child and its children. Stops when a hash was cached or a
        // node is blinded.
        let left = self.left_ref().root().map(|node| resolver.hash(node));

        // Recursively hashes any right child and its children. Stops when a hash was cached or a
        // node is blinded.
        let right = self.right_ref().root().map(|node| resolver.hash(node));

        NodeHashRepresentation {
            key: &self.key,
            data: &self.data,
            left,
            right,
            balance_factor: self.balance_factor,
        }
    }

    /// Returns the hash of this node.
    ///
    /// If the hash has been cached, the memo is returned. Otherwise, the hash is calculated and
    /// cached.
    pub(crate) fn hash(&self, resolver: &impl Resolver<Id, Self>) -> &Hash {
        self.hash.get_or_init(|| {
            let data = self.to_encode(resolver);
            Hash::hash_encodable(data).expect("The hashing should not fail")
        })
    }

    #[inline]
    /// The difference in heights between child branches.
    pub(super) fn balance_factor(&self) -> i64 {
        self.balance_factor
    }

    #[inline]
    /// A mutable reference to the difference in heights between child branches.
    pub(super) fn balance_factor_mut(&mut self) -> &mut i64 {
        &mut self.balance_factor
    }

    #[inline]
    /// The [`Key`] used for determining the [`Node`].
    pub(super) fn key(&self) -> &Key {
        &self.key
    }

    #[inline]
    /// A mutable reference to the left branch.
    pub(super) fn left_mut(&mut self) -> &mut Tree<Id> {
        self.invalidate_hash();
        &mut self.left
    }

    #[inline]
    /// An immutable reference to the left branch.
    pub(super) fn left_ref(&self) -> &Tree<Id> {
        &self.left
    }

    /// Rebalance the subtree of the [`Node`] so that the difference in height between child
    /// branches is in the range of -1..=1.
    ///
    /// The subtree of the [`Node`] must already have balance factor in the range of -2..=2, else
    /// it is an invalid AVL tree.
    pub(super) fn rebalance(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(), OperationalError>
    where
        Id: Clone,
    {
        let resolved_node = resolver.resolve(node)?;
        let balance_factor = resolved_node.balance_factor();
        match balance_factor {
            2 => {
                let right_balance = resolved_node.right_ref().balance_factor(resolver)?;

                match right_balance {
                    1 | 0 => Self::rotate_left(node, resolver)?,
                    -1 => Self::rotate_right_left(node, resolver)?,
                    _ => panic!(
                        "Rebalancing an invalid AVL tree. The balance factor of the right node is {right_balance:?}, but it should be in the range of -1..=1"
                    ),
                };
            }
            -1..=1 => (),
            -2 => {
                let left_balance = resolved_node.left_ref().balance_factor(resolver)?;

                match left_balance {
                    1 => Self::rotate_left_right(node, resolver)?,
                    -1 | 0 => Self::rotate_right(node, resolver)?,
                    _ => panic!(
                        "Rebalancing an invalid AVL tree. The balance factor of the left node is {left_balance:?}, but it should be in the range of -1..=1"
                    ),
                };
            }
            _ => panic!(
                "Rebalancing an invalid AVL tree. The balance factor is {:?}, but it should be in the range of -2..=2",
                resolved_node.balance_factor()
            ),
        };
        Ok(())
    }

    /// Remove the successor of the [`Node`] from its subtree and replace the original [`Node`]
    /// with the successor.
    ///
    /// Returns a tuple of:
    ///  - The [`Node`] at the root of the new subtree.
    ///  - `true` if the [`Tree`] has shrunk in size.
    pub(super) fn replace_with_successor(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(Id, bool), OperationalError>
    where
        Id: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;

        // If the right child has a left child, the successor is the min of the left child's subtree.
        let (mut successor, shrank) = if resolver
            .resolve(
                node_mut
                    .right_ref()
                    .root()
                    .expect("A node with a successor must have a right child"),
            )?
            .left_ref()
            .root()
            .is_some()
        {
            let right = node_mut.right_mut();
            let (mut min, _, shrank) = Tree::take_min(right, resolver)?;
            (
                min.take()
                    .expect("A node with a successor must have a right child"),
                shrank,
            )
        // If there is no left child of the right child, the successor is the right child.
        } else {
            let mut successor = node_mut
                .right_mut()
                .take()
                .expect("A node with a successor must have a right child");
            let successor_mut = resolver.resolve_mut(&mut successor)?;

            // Bump up the (optional) right child of the right child, causing the subtree to shrink.
            node_mut.right = successor_mut.right.take().into();
            (successor, true)
        };

        let successor_mut = resolver.resolve_mut(&mut successor)?;

        successor_mut.balance_factor = node_mut.balance_factor() - if shrank { 1 } else { 0 };
        successor_mut.left = std::mem::take(&mut node_mut.left);
        successor_mut.right = std::mem::take(&mut node_mut.right);

        Self::rebalance(&mut successor, resolver)?;

        let successor_balance_factor = resolver.resolve(&successor)?.balance_factor();
        let shrank = node_mut.balance_factor().abs() == 1 && successor_balance_factor == 0;

        Ok((successor, shrank))
    }

    #[inline]
    /// A mutable reference to the right branch.
    pub(super) fn right_mut(&mut self) -> &mut Tree<Id> {
        self.invalidate_hash();
        &mut self.right
    }

    #[inline]
    /// An immutable reference to the right branch.
    pub(super) fn right_ref(&self) -> &Tree<Id> {
        &self.right
    }

    /// Takes the occupied [`Tree`] with the minimum [`Key`] from this [`Node`]'s subtree and
    /// replaces it with an empty [`Tree`].
    ///
    /// Returns a tuple of:
    ///  - The occupied [`Tree`] with the minimum [`Key`].
    ///  - The minimum [`Tree`]'s right child, if it hasn't been moved to its new position.
    ///  - True if this [`Node`]'s subtree has shrunk in size.
    pub(super) fn take_min(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(Tree<Id>, Tree<Id>, bool), OperationalError>
    where
        Id: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;

        let old_node_bf = node_mut.balance_factor();
        let left = node_mut.left_mut();
        let (min, right, left_shrank) = Tree::take_min(left, resolver)?;

        if right.root().is_some() {
            *node_mut.left_mut() = right;
            node_mut.balance_factor += 1;
        } else if left_shrank {
            node_mut.balance_factor += 1;
        };

        Node::rebalance(node, resolver)?;
        Ok((
            min,
            None.into(),
            old_node_bf.abs() == 1 && resolver.resolve(node)?.balance_factor() == 0,
        ))
    }

    /// Set or update the value of a [`Node`] in this [`Node`]'s subtree with a given key and given
    /// offset.
    ///
    /// `data` defines what data is upserted.
    ///
    /// Returns true if this [`Node`]s subtree has grown in size.
    pub(super) fn upsert(
        &mut self,
        key: &Key,
        offset: usize,
        data: impl FnOnce(&mut Value) -> Result<(), Error>,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<bool, Error>
    where
        Id: Clone + From<Self>,
    {
        // SAFETY: The default recursion limit in Rust is 128
        // see: <https://doc.rust-lang.org/reference/attributes/limits.html#r-attributes.limits.recursion_limit.syntax>
        //
        // This function recurses once for every node it traverses, meaning that the number
        // of recursions are equal to or less than the height of the node.
        //
        // To hit this limit, the lower bound on the number of nodes in a valid AVL tree is:
        // fibonacci(height + 3) - 1
        // see: <https://www.cs.cornell.edu/courses/cs2112/2020fa/lectures/avl/>
        //
        // with height = 128:
        // fibonacci(128 + 3) - 1 > 1x10^27
        //
        // This would require:
        //  - more nodes than 64-bit systems can address.
        //  - more disk space than has ever been produced.
        //  - inserting 2 billion nodes every second since the dawn of the universe.
        match self.key.cmp(key) {
            // The key already exists and should be updated.
            Ordering::Equal => {
                data(&mut self.data)?;
                self.invalidate_hash();
                Ok(false)
            }
            Ordering::Greater => {
                let grew = self.left_mut().upsert(key, offset, data, resolver)?;
                if grew {
                    self.balance_factor -= 1;
                }
                Ok(grew)
            }
            Ordering::Less => {
                let grew = self.right_mut().upsert(key, offset, data, resolver)?;
                if grew {
                    self.balance_factor += 1;
                }
                Ok(grew)
            }
        }
    }

    /// Mark the hash of this node as dirty.
    fn invalidate_hash(&mut self) {
        self.hash = OnceLock::new();
    }

    /// Rotate this [`Node`]'s subtree left.
    ///
    /// For example:
    ///
    /// ```text
    ///     BEFORE          AFTER
    /// node                  A
    ///     \               /   \
    ///       A          node    C
    ///     /   \          \
    ///    B     C          B
    /// ```
    ///
    /// Assumes this [`Node`]'s balance factor is 2 and the right [`Node`]'s balance factor is +1
    /// or 0.
    fn rotate_left(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(), OperationalError>
    where
        Id: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;
        let mut right = node_mut
            .right_mut()
            .take()
            .expect("There should be a right node to rotate left");
        let right_mut = resolver.resolve_mut(&mut right)?;

        *node_mut.right_mut() = right_mut.left_mut().take().into();

        // new_node_bf = B.height() - node.left.height()
        // old_node_bf = A.height() - node.left.height()
        //             = (1 + std::cmp::max(C.height(), B.height()) - node.left.height()
        //
        // new_node_bf - old_node_bf = B.height() - 1 - std::cmp::max(C.height(), B.height())
        //
        // new_node_bf = old_node_bf + B.height() - 1 + std::cmp::min(-C.height(), -B.height())
        //             = old_node_bf - 1 + std::cmp::min(B.height() - C.height(),
        //                                               B.height() - B.height())
        //
        //             = old_node_bf - 1 + std::cmp::min(-A.balance_factor, 0)
        //
        // For inserting a node, this will always be zero, however deletion allows for rotation cases
        // where the balance factor of A is -1
        let new_node_bf = node_mut.balance_factor - 1 + std::cmp::min(-right_mut.balance_factor, 0);
        node_mut.balance_factor = new_node_bf;

        // new_A_bf = C.height() - node.height()
        //          = C.height - (1 + std::cmp::max(node.left.height(), A.height()))
        // old_A_bf = C.height() - B.height()
        //
        // new_A_bf - old_A_bf = - 1 - std::cmp::max(node.left.height(), A.height()) + B.height()
        //
        // new_A_bf = old_node_bf - 1 + std::cmp::min(-node.left.height(), -A.height()) + B.height()
        //          = old_node_bf - 1 + std::cmp::min(B.height() - node.left.height(),
        //                                            B.height() - B.height())
        //
        //          = old_node_bf - 1 + std::cmp::min(new_node_bf, 0)
        right_mut.balance_factor = right_mut.balance_factor - 1 + std::cmp::min(new_node_bf, 0);

        *right_mut.left_mut() = Some(node.clone()).into();
        *node = right;
        Ok(())
    }

    /// Rotate the left child of this [`Node`] left, then this [`Node`]'s subtree right.
    ///
    /// For example:
    ///
    /// ```text
    ///     BEFORE          AFTER
    ///          node         C
    ///         /           /   \
    ///       A            A    node
    ///     /  \          / \   /
    ///    B    C        B   D E
    ///        / \
    ///       D   E
    /// ```
    ///
    /// Assumes this [`Node`]'s balance factor is -2 and the left [Node]'s balance factor is +1.
    fn rotate_left_right(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(), OperationalError>
    where
        Id: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;

        let mut left = node_mut
            .left_mut()
            .take()
            .expect("Left child must exist for the right rotation of the node");
        let left_mut = resolver.resolve_mut(&mut left)?;

        let mut left_right = left_mut
            .right_mut()
            .take()
            .expect("Left's right child must exist for the left rotation of the left node");
        let left_right_mut = resolver.resolve_mut(&mut left_right)?;

        // From the `rotate_left` derivation, the first rotation does:
        //   new_A_bf_1 = old_A_bf - 1 + std::cmp::min(-A.right.balance_factor, 0)
        // As this function assumes old_A_bf is +1:
        //   new_A_bf_1 = std::cmp::min(-A.right.balance_factor, 0)
        // The second rotation doesn't mutate A's subtree, so the final balance factor is:
        left_mut.balance_factor = std::cmp::min(-left_right_mut.balance_factor, 0);

        // B's right child is between B and B, it's moved to node's left
        node_mut.left = left_right_mut.right.take().into();

        // B's left child is between A and B, it's moved to A's right
        left_mut.right = left_right_mut.left.take().into();

        // Set A
        left_right_mut.left = Some(left).into();

        // If B is 0 or 1, the new node balance factor will be 0
        // If B is -1, the new node balance factor will be 1
        node_mut.balance_factor = std::cmp::max(0, -left_right_mut.balance_factor);

        // Set node
        left_right_mut.right = Some(node.clone()).into();

        // The new root will always be balanced
        left_right_mut.balance_factor = 0;
        *node = left_right;
        Ok(())
    }

    /// Rotate this [`Node`]'s subtree right.
    ///
    /// For example:
    ///
    /// ```text
    ///     BEFORE          AFTER
    ///          node         A
    ///         /           /   \
    ///       A            B    node
    ///     /  \                /
    ///    B    C              C
    /// ```
    ///
    /// Assumes this [`Node`]'s balance factor is -2 and the left [`Node`]'s balance factor is -1
    /// or 0.
    fn rotate_right(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(), OperationalError>
    where
        Id: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;
        let mut left = node_mut
            .left_mut()
            .take()
            .expect("There should be a left node to rotate right");
        let left_mut = resolver.resolve_mut(&mut left)?;

        *node_mut.left_mut() = left_mut.right_mut().take().into();

        // new_node_bf = node.right.height() - C.height()
        // old_node_bf = node.right.height() - A.height()
        //             = node.right.height() - (1 + std::cmp::max(C.height(), B.height()))
        //
        // new_node_bf - old_node_bf = 1 + std::cmp::max(C.height(), B.height()) - C.height()
        //
        // new_node_bf = old_node_bf + 1 + std::cmp::max(C.height(), B.height()) - C.height()
        //             = old_node_bf + 1 + std::cmp::max(C.height() - C.height(),
        //                                               B.height() - C.height())
        //
        //             = old_node_bf + 1 + std::cmp::max(0, -A.balance_factor)
        let new_node_bf = node_mut.balance_factor + 1 + std::cmp::max(0, -left_mut.balance_factor);
        node_mut.balance_factor = new_node_bf;

        // new_A_bf = node.height() - B.height()
        //          = (1 + std::cmp::max(node.right.height(), C.height())) - B.height()
        // old_A_bf = C.height() - B.height()
        //
        // new_A_bf - old_A_bf = 1 + std::cmp::max(node.right.height(), C.height()) - C.height()
        //
        // new_A_bf = old_A_bf + 1 + std::cmp::max(node.right.height(), C.height()) - C.height()
        //          = old_A_bf + 1 + std::cmp::max(node.right.height() - C.height(),
        //                                         C.height()) - C.height()
        //
        //          = old_A_bf + 1 + std::cmp::max(new_node_bf, 0)
        //
        // For inserting a node, this will always be zero, however deletion allows for rotation cases
        // where the balance factor of A is 1
        left_mut.balance_factor = left_mut.balance_factor + 1 + std::cmp::max(new_node_bf, 0);

        *left_mut.right_mut() = Some(node.clone()).into();
        *node = left;
        Ok(())
    }

    /// Rotate the right child of this [`Node`] right, then this [`Node`]'s subtree left.
    ///
    /// For example:
    ///
    /// ```text
    ///     BEFORE          AFTER
    ///    node               B
    ///      \              /   \
    ///       A           node   A
    ///      / \            \   / \
    ///     B   C            D E   C
    ///    / \
    ///   D   E
    /// ```
    ///
    /// Assumes this [`Node`]'s balance factor is +2 and the left [Node]'s balance factor is -1.
    fn rotate_right_left(
        node: &mut Id,
        resolver: &mut impl Resolver<Id, Self>,
    ) -> Result<(), OperationalError>
    where
        Id: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;
        let mut right = node_mut
            .right_mut()
            .take()
            .expect("Right child must exist for the left rotation of the node");
        let right_mut = resolver.resolve_mut(&mut right)?;

        let mut right_left = right_mut
            .left_mut()
            .take()
            .expect("Right's left child must exist for the right rotation of the right node");
        let right_left_mut = resolver.resolve_mut(&mut right_left)?;

        // From the `rotate_right` derivation, the first rotation does:
        //   new_A_bf_1 = old_A_bf + 1 + std::cmp::max(0, -A.left.balance_factor)
        // As this function assumes old_A_bf is -1:
        //   new_A_bf_1 = std::cmp::max(0, -A.left.balance_factor)
        // The second rotation doesn't mutate A's subtree, so the final balance factor is:
        right_mut.balance_factor = std::cmp::max(0, -right_left_mut.balance_factor);

        // B's left child is between node and B, it's moved to node's right
        node_mut.right = right_left_mut.left.take().into();

        // B's right child is between B and A, it's moved to A's left
        right_mut.left = right_left_mut.right.take().into();

        // Set A
        right_left_mut.right = Some(right).into();

        // If B is 0 or -1, the new node balance factor will be 0
        // If B is 1, the new node balance factor will be -1
        node_mut.balance_factor = -std::cmp::max(0, right_left_mut.balance_factor);

        // Set node
        right_left_mut.left = Some(node.clone()).into();

        // The new root will always be balanced
        right_left_mut.balance_factor = 0;
        *node = right_left;
        Ok(())
    }
}

#[cfg(test)]
impl<Id> Node<Id> {
    #[inline]
    /// The data stored in the [`Node`].
    pub(crate) fn data(&self) -> &Value {
        &self.data
    }

    /// The data stored in a [`Node`] within the subtree of this [`Node`] with a given [`Key`] .
    pub(super) fn get<'a>(
        mut node: &'a Id,
        key: &Key,
        resolver: &impl Resolver<Id, Self>,
    ) -> Result<Option<&'a Value>, OperationalError> {
        loop {
            let resolved_node = resolver.resolve(node)?;
            match resolved_node.key().cmp(key) {
                Ordering::Equal => return Ok(Some(resolved_node.data())),
                Ordering::Greater => {
                    let Some(left) = resolved_node.left_ref().root() else {
                        return Ok(None);
                    };
                    node = left;
                }
                Ordering::Less => {
                    let Some(right) = resolved_node.right_ref().root() else {
                        return Ok(None);
                    };
                    node = right;
                }
            }
        }
    }

    /// Returns true if the balance factors stored in the [`Node`]'s subtree are correct.
    pub(super) fn has_correct_balance_factors(
        &self,
        resolver: &impl Resolver<Id, Self>,
    ) -> Result<bool, OperationalError>
    where
        Id: std::fmt::Debug,
    {
        let left_height = self.left_ref().height(resolver)?;
        let right_height = self.right_ref().height(resolver)?;
        let calculated_balance_factor = right_height as i64 - left_height as i64;
        if self.balance_factor() != calculated_balance_factor {
            eprintln!(
                "Node has balance_factor {:?}, should be {calculated_balance_factor:?}\nnode: {self:?}",
                self.balance_factor()
            );
            return Ok(false);
        }
        let res = self.left_ref().has_correct_balance_factors(resolver)?
            && self.right_ref().has_correct_balance_factors(resolver)?;
        Ok(res)
    }

    /// Returns the height of this [`Node`]'s subtree.
    pub(super) fn height(
        &self,
        resolver: &impl Resolver<Id, Self>,
    ) -> Result<u32, OperationalError> {
        let left_height = self.left_ref().height(resolver)?;
        let right_height = self.right_ref().height(resolver)?;
        Ok(1 + std::cmp::max(left_height, right_height))
    }

    /// Returns true if this [`Node`]'s subtree is balanced.
    pub(super) fn is_balanced(
        &self,
        resolver: &impl Resolver<Id, Self>,
    ) -> Result<bool, OperationalError> {
        let balance_factor = self.balance_factor();
        if balance_factor.abs() > 1 {
            eprintln!("Balance factor not in -1..=1: {balance_factor:?}");
            return Ok(false);
        }
        let res =
            self.left_ref().is_balanced(resolver)? && self.right_ref().is_balanced(resolver)?;
        Ok(res)
    }

    /// Returns true if this [`Node`]'s subtree is in-order.
    pub(super) fn is_inorder(
        &self,
        min: &Key,
        max: &Key,
        resolver: &impl Resolver<Id, Self>,
    ) -> Result<bool, OperationalError> {
        if self.key() < min || self.key() > max {
            return Ok(false);
        }
        let res = self
            .left_ref()
            .is_inorder_inner(min, self.key(), resolver)?
            && self
                .right_ref()
                .is_inorder_inner(self.key(), max, resolver)?;
        Ok(res)
    }
}
