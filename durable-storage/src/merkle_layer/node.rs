// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::OnceLock;

use bincode::BorrowDecode;
use bincode::Encode;
use bytes::Bytes;
use bytes::BytesMut;
use octez_riscv_data::hash::Hash;

use super::node_resolver::MavlNodeResolver;
use super::node_wrapper::MavlNodeWrapper;
use crate::key::Key;

/// A node that supports rebalancing and Merklisation.
#[derive(Clone, Default, Debug)]
pub(crate) struct MavlNode<Resolver: MavlNodeResolver> {
    key: Key,
    data: BytesMut,
    left: Resolver::NodeWrapper,
    right: Resolver::NodeWrapper,

    /// A cache for the hash of this node. This uses `OnceLock` so that updating the cache is a
    /// non-mutating operation.
    ///
    /// An uninitialised hash is a hash that has not been set or has been dirtied.
    hash: OnceLock<Hash>,

    /// The difference in heights between child branches (right - left).
    balance_factor: i64,
}

#[derive(Encode, BorrowDecode)]
/// A serialisable representation of [`MavlNode`].
pub(crate) struct MavlNodeHashRepresentation<'a> {
    pub(crate) key: &'a [u8],
    data: &'a [u8],
    // The bytes of the hash of an optional left child
    pub(crate) left: Option<&'a [u8]>,
    // The bytes of the hash of an optional right child
    pub(crate) right: Option<&'a [u8]>,
    balance_factor: i64,
}

impl<Resolver: MavlNodeResolver> MavlNode<Resolver> {
    /// The difference in heights between child branches.
    #[cfg(test)]
    pub(super) fn balance_factor(&self) -> i64 {
        self.balance_factor
    }

    /// The data stored in the node.
    pub(super) fn data(&self) -> &BytesMut {
        &self.data
    }

    /// The key used for determining the node.
    pub(super) fn key(&self) -> &Key {
        &self.key
    }

    /// A mutable reference to the left branch.
    pub(super) fn left_mut(&mut self, node_resolver: &Resolver) -> &mut Option<Arc<Self>> {
        self.resolve_children(node_resolver);
        self.invalidate_hash();
        self.left
            .try_borrow_mut()
            .expect("The left child is already resolved")
    }

    /// An immutable reference to the left branch.
    pub(super) fn left_ref(&self, node_resolver: &Resolver) -> &Option<Arc<Self>> {
        self.resolve_children(node_resolver);
        self.left
            .try_borrow()
            .expect("The left child is already resolved")
    }

    /// Create a new leaf node from the given key and data.
    pub(super) fn new(key: Key, data: Bytes) -> Self {
        MavlNode {
            key,
            data: data.into(),
            left: Resolver::NodeWrapper::new(None),
            right: Resolver::NodeWrapper::new(None),
            hash: OnceLock::new(),
            balance_factor: 0,
        }
    }

    /// A mutable reference to the right branch.
    pub(super) fn right_mut(&mut self, node_resolver: &Resolver) -> &mut Option<Arc<Self>> {
        self.resolve_children(node_resolver);
        self.invalidate_hash();
        self.right
            .try_borrow_mut()
            .expect("The right child is already resolved")
    }

    /// An immutable reference to the right branch.
    pub(super) fn right_ref(&self, node_resolver: &Resolver) -> &Option<Arc<Self>> {
        self.resolve_children(node_resolver);
        self.right
            .try_borrow()
            .expect("The right child is already resolved")
    }

    /// Converts the node to an encoded, serialisable representation, potentially re-hashing
    /// uncached nodes.
    pub(super) fn to_encode(&self, node_resolver: &Resolver) -> MavlNodeHashRepresentation<'_> {
        MavlNodeHashRepresentation {
            key: &self.key.as_slice(),
            data: &self.data,

            // Recursively hashes any left child and its children
            left: self
                .left_ref(node_resolver)
                .as_ref()
                .map(|left_ref| hash(left_ref, node_resolver))
                .map(|h| h.as_ref()),

            // Recursively hashes any right child and its children
            right: self
                .right_ref(node_resolver)
                .as_ref()
                .map(|right_ref| hash(right_ref, node_resolver))
                .map(|h| h.as_ref()),

            balance_factor: self.balance_factor,
        }
    }

    /// Mark the hash of this node as dirty.
    fn invalidate_hash(&mut self) {
        self.hash = OnceLock::new();
    }

    pub(super) fn decode(
        hash_representation: MavlNodeHashRepresentation,
        commited_hash: Hash,
        key: Key,
        left: Resolver::NodeWrapper,
        right: Resolver::NodeWrapper,
    ) -> Self {
        Self {
            key,
            data: BytesMut::from(hash_representation.data),
            left,
            right,
            hash: OnceLock::from(commited_hash),
            balance_factor: hash_representation.balance_factor,
        }
    }

    fn resolve_children(&self, node_resolver: &Resolver) {
        node_resolver.resolve(&self.left);
        node_resolver.resolve(&self.right);
    }
}

/// Delete the value of the node with a given key. If the key does not exist, do nothing.
///
/// Returns true if the subtree has shrank in size.
pub(super) fn delete<Resolver: MavlNodeResolver>(
    root: &mut Option<Arc<MavlNode<Resolver>>>,
    key: &Key,
    node_resolver: &Resolver,
) -> bool {
    let Some(node) = root else {
        // The key does not exist so nothing will happen.
        return false;
    };
    match node.key.cmp(key) {
        Ordering::Equal => match (node.left_ref(node_resolver), node.right_ref(node_resolver)) {
            (None, None) => {
                *root = None;
                true
            }
            (Some(left), None) => {
                *node = left.clone();
                true
            }
            (None, Some(right)) => {
                *node = right.clone();
                true
            }
            (Some(_), Some(_)) => {
                let (new_node, shrank) = replace_with_successor(node, node_resolver);
                *node = new_node;
                shrank
            }
        },
        Ordering::Greater => {
            let node_mut = Arc::make_mut(node);
            let old_balance_factor = node_mut.balance_factor;

            let left_shrank = delete(node_mut.left_mut(node_resolver), key, node_resolver);

            node_mut.balance_factor += if left_shrank { 1 } else { 0 };
            *node = rebalance(node, node_resolver);
            old_balance_factor.abs() == 1 && node.balance_factor == 0
        }
        Ordering::Less => {
            let node_mut = Arc::make_mut(node);
            let old_balance_factor = node_mut.balance_factor;

            let right_shrank = delete(node_mut.right_mut(node_resolver), key, node_resolver);

            node_mut.balance_factor -= if right_shrank { 1 } else { 0 };
            *node = rebalance(node, node_resolver);
            old_balance_factor.abs() == 1 && node.balance_factor == 0
        }
    }
}

/// The data stored in a node in the tree with a given key.
pub(super) fn get<'a, Resolver: MavlNodeResolver>(
    root: &'a Option<Arc<MavlNode<Resolver>>>,
    key: &Key,
    node_resolver: &Resolver,
) -> Option<&'a BytesMut> {
    let mut node = root.as_deref()?;
    loop {
        match node.key().cmp(key) {
            Ordering::Equal => return Some(node.data()),
            Ordering::Greater => node = node.left_ref(node_resolver).as_deref()?,
            Ordering::Less => node = node.right_ref(node_resolver).as_deref()?,
        }
    }
}

/// A mutable reference to the data stored in a node in the tree with a given key.
pub(super) fn get_mut<'a, Resolver: MavlNodeResolver>(
    root: &'a mut Option<Arc<MavlNode<Resolver>>>,
    key: &Key,
    node_resolver: &Resolver,
) -> Option<&'a mut BytesMut> {
    let node = root.as_mut()?;
    let node = Arc::make_mut(node);
    match node.key().cmp(key) {
        Ordering::Equal => {
            node.invalidate_hash();
            Some(&mut node.data)
        }
        Ordering::Greater => get_mut(node.left_mut(node_resolver), key, node_resolver),
        Ordering::Less => get_mut(node.right_mut(node_resolver), key, node_resolver),
    }
}

/// Returns the hash of this node, including recursively hashing any child nodes.
///
/// If the hash has been cached, the memo is returned. Otherwise, the hash is calculated and
/// cached.
pub(super) fn hash<'a, Resolver: MavlNodeResolver>(
    node: &'a Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> &'a Hash {
    node.hash.get_or_init(|| {
        Hash::hash_encodable(node.to_encode(node_resolver)).expect("The hashing should not fail")
    })
}

/// Rebalance the node so that the difference in height between child branches is in the range
/// of -1..=1.
///
/// The node must already have balance factor in the range of -2..=2, or it is an invalid AVL
/// node.
///
/// Returns the rebalanced subtree.
#[must_use]
fn rebalance<Resolver: MavlNodeResolver>(
    node: &mut Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> Arc<MavlNode<Resolver>> {
    match node.balance_factor {
        2 => {
            let right_balance = node
                .right_ref(node_resolver)
                .as_ref()
                .map_or(0, |r| r.balance_factor);

            match right_balance {
                1 | 0 => rotate_left(node, node_resolver),
                -1 => rotate_right_left(node, node_resolver),
                _ => panic!(
                    "Rebalancing an invalid AVL tree. The balance factor of the right node is {right_balance:?}, but it should be in the range of -1..=1"
                ),
            }
        }
        -1..=1 => node.clone(),
        -2 => {
            let left_balance = node
                .left_ref(node_resolver)
                .as_ref()
                .map_or(0, |l| l.balance_factor);

            match left_balance {
                1 => rotate_left_right(node, node_resolver),
                -1 | 0 => rotate_right(node, node_resolver),
                _ => panic!(
                    "Rebalancing an invalid AVL tree. The balance factor of the left node is {left_balance:?}, but it should be in the range of -1..=1"
                ),
            }
        }
        _ => panic!(
            "Rebalancing an invalid AVL tree. The balance factor is {:?}, but it should be in the range of -2..=2",
            node.balance_factor
        ),
    }
}

/// Rotate this node left.
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
/// Assumes the balance factor is 2 and the right node's balance factor is -1 or 0.
///
/// Returns the rotated subtree.
#[must_use]
fn rotate_left<Resolver: MavlNodeResolver>(
    node: &mut Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> Arc<MavlNode<Resolver>> {
    let node_mut = Arc::make_mut(node);
    let mut right = node_mut
        .right_mut(node_resolver)
        .take()
        .expect("There should be a right node to rotate left");
    let right_mut = Arc::make_mut(&mut right);

    *node_mut.right_mut(node_resolver) = right_mut.left_mut(node_resolver).take();

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

    *right_mut.left_mut(node_resolver) = Some(node.clone());
    right
}

/// Rotate this node right.
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
/// Assumes the balance factor is -2 and the left node's balance factor is -1 or 0.
///
/// Returns the rotated subtree.
#[must_use]
fn rotate_right<Resolver: MavlNodeResolver>(
    node: &mut Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> Arc<MavlNode<Resolver>> {
    let node_mut = Arc::make_mut(node);
    let mut left = node_mut
        .left_mut(node_resolver)
        .take()
        .expect("There should be a left node to rotate right");
    let left_mut = Arc::make_mut(&mut left);

    *node_mut.left_mut(node_resolver) = left_mut.right_mut(node_resolver).take();

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

    *left_mut.right_mut(node_resolver) = Some(node.clone());
    left
}

/// Rotate the left child of this node left, then this node right.
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
/// Assumes the balance factor is -2 and the left node's balance factor is +1.
///
/// Returns the rotated subtree.
#[must_use]
fn rotate_left_right<Resolver: MavlNodeResolver>(
    node: &mut Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> Arc<MavlNode<Resolver>> {
    let node_mut = Arc::make_mut(node);

    let mut left = node_mut
        .left_mut(node_resolver)
        .take()
        .expect("Left child must exist for the right rotation of the node");
    let left_mut = Arc::make_mut(&mut left);

    let mut left_right = left_mut
        .right_mut(node_resolver)
        .take()
        .expect("Left's right child must exist for the left rotation of the left node");

    // From the `rotate_left` derivation, the first rotation does:
    //   new_A_bf_1 = old_A_bf - 1 + std::cmp::min(-A.right.balance_factor, 0)
    // As this function assumes old_A_bf is +1:
    //   new_A_bf_1 = std::cmp::min(-A.right.balance_factor, 0)
    // The second rotation doesn't mutate A's subtree, so the final balance factor is:
    left_mut.balance_factor = std::cmp::min(-left_right.balance_factor, 0);

    let left_right_mut = Arc::make_mut(&mut left_right);

    // B's right child is between B and B, it's moved to node's left
    *node_mut.left_mut(node_resolver) = left_right_mut.right_mut(node_resolver).take();

    // B's left child is between A and B, it's moved to A's right
    *left_mut.right_mut(node_resolver) = left_right_mut.left_mut(node_resolver).take();

    // Set A
    *left_right_mut.left_mut(node_resolver) = Some(left);

    // If B is 0 or 1, the new node balance factor will be 0
    // If B is -1, the new node balance factor will be 1
    node_mut.balance_factor = std::cmp::max(0, -left_right_mut.balance_factor);

    // Set node
    *left_right_mut.right_mut(node_resolver) = Some(node.clone());

    // The new root will always be balanced
    left_right_mut.balance_factor = 0;
    left_right
}

/// Rotate the right child of this node right, then this node left.
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
/// Assumes the balance factor is +2 and the left node's balance factor is -1.
///
/// Returns the rotated subtree.
#[must_use]
fn rotate_right_left<Resolver: MavlNodeResolver>(
    node: &mut Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> Arc<MavlNode<Resolver>> {
    let node_mut = Arc::make_mut(node);

    let mut right = node_mut
        .right_mut(node_resolver)
        .take()
        .expect("Right child must exist for the left rotation of the node");
    let right_mut = Arc::make_mut(&mut right);

    let mut right_left = right_mut
        .left_mut(node_resolver)
        .take()
        .expect("Right's left child must exist for the right rotation of the right node");

    // From the `rotate_right` derivation, the first rotation does:
    //   new_A_bf_1 = old_A_bf + 1 + std::cmp::max(0, -A.left.balance_factor)
    // As this function assumes old_A_bf is -1:
    //   new_A_bf_1 = std::cmp::max(0, -A.left.balance_factor)
    // The second rotation doesn't mutate A's subtree, so the final balance factor is:
    right_mut.balance_factor = std::cmp::max(0, -right_left.balance_factor);

    let right_left_mut = Arc::make_mut(&mut right_left);

    // B's left child is between node and B, it's moved to node's right
    *node_mut.right_mut(node_resolver) = right_left_mut.left_mut(node_resolver).take();

    // B's right child is between B and A, it's moved to A's left
    *right_mut.left_mut(node_resolver) = right_left_mut.right_mut(node_resolver).take();

    // Set A
    *right_left_mut.right_mut(node_resolver) = Some(right);

    // If B is 0 or -1, the new node balance factor will be 0
    // If B is 1, the new node balance factor will be -1
    node_mut.balance_factor = -std::cmp::max(0, right_left_mut.balance_factor);

    // Set node
    *right_left_mut.left_mut(node_resolver) = Some(node.clone());

    // The new root will always be balanced
    right_left_mut.balance_factor = 0;
    right_left
}

/// Remove the successor of the node from its subtree and replace the original node with it.
///
/// Returns:
///  - The new subtree.
///  - True if the subtree has shrank in size.
#[must_use]
fn replace_with_successor<Resolver: MavlNodeResolver>(
    node: &mut Arc<MavlNode<Resolver>>,
    node_resolver: &Resolver,
) -> (Arc<MavlNode<Resolver>>, bool) {
    let node_balance_factor = node.balance_factor;
    let node_mut = Arc::make_mut(node);
    let node_bf = node_mut.balance_factor;

    // If the right child has a left child, the successor is the min of the left child's subtree.
    let (mut successor, shrank) = if node_mut
        .right_ref(node_resolver)
        .as_ref()
        .expect("A node with a successor must have a right child")
        .left_ref(node_resolver)
        .is_some()
    {
        let right = node_mut.right_mut(node_resolver);
        let (min, _, shrank) = take_min(right, node_resolver);
        (
            min.expect("A node with a successor must have a right child"),
            shrank,
        )
    // If there is no left child of the right child, the successor is the right child.
    } else {
        let mut successor = node_mut
            .right_mut(node_resolver)
            .take()
            .expect("A node with a successor must have a right child");
        let successor_mut = Arc::make_mut(&mut successor);

        // Bump up the (optional) right child of the right child, causing the subtree to shrink.
        *node_mut.right_mut(node_resolver) = successor_mut.right_mut(node_resolver).take();
        (successor, true)
    };

    let successor_mut = Arc::make_mut(&mut successor);

    successor_mut.balance_factor = node_bf - if shrank { 1 } else { 0 };
    successor_mut.left = node.left.clone();
    successor_mut.right = node.right.clone();

    successor = rebalance(&mut successor, node_resolver);

    let shrank = node_balance_factor.abs() == 1 && successor.balance_factor == 0;
    (successor, shrank)
}

/// Set the value of the node with a given key.
///
/// Returns true if the subtree has grown in size.
pub(super) fn set<Resolver: MavlNodeResolver>(
    root: &mut Option<Arc<MavlNode<Resolver>>>,
    key: &Key,
    data: Bytes,
    node_resolver: &Resolver,
) -> bool {
    let Some(node) = root else {
        // The key does not exist and a new node shall be created.
        *root = Some(Arc::new(MavlNode::new(key.clone(), data)));
        return true;
    };
    // SAFETY: The default recursion limit in Rust is 128
    // see: <https://doc.rust-lang.org/reference/attributes/limits.html#r-attributes.limits.recursion_limit.syntax>
    //
    // This function recurses once for every node it traverses, meaning that the number
    // of recursions are equal to or less than the height of the node.
    //
    // The lower bound on the number of nodes in a valid AVL tree is:
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
    match node.key.cmp(key) {
        // The key already exists and should be updated.
        Ordering::Equal => {
            let node = Arc::make_mut(node);
            node.data = data.into();
            node.invalidate_hash();
            false
        }
        Ordering::Greater => {
            let node_mut = Arc::make_mut(node);
            let grew = set(node_mut.left_mut(node_resolver), key, data, node_resolver);
            if grew {
                node_mut.balance_factor -= 1;
                *node = rebalance(node, node_resolver);
                node.balance_factor != 0
            } else {
                false
            }
        }
        Ordering::Less => {
            let node_mut = Arc::make_mut(node);
            let grew = set(node_mut.right_mut(node_resolver), key, data, node_resolver);
            if grew {
                node_mut.balance_factor += 1;
                *node = rebalance(node, node_resolver);
                node.balance_factor != 0
            } else {
                false
            }
        }
    }
}

type TakeMinReturnType<Resolver> = (
    Option<Arc<MavlNode<Resolver>>>,
    Option<Arc<MavlNode<Resolver>>>,
    bool,
);

/// Remove the minimum node from this subtree and return it.
///
/// Returns:
///  - The minimum node.
///  - The minimum node's right child, if it hasn't been moved to its new position.
///  - True if the subtree has shrank in size.
#[must_use]
fn take_min<Resolver: MavlNodeResolver>(
    node: &mut Option<Arc<MavlNode<Resolver>>>,
    node_resolver: &Resolver,
) -> TakeMinReturnType<Resolver> {
    // Shouldn't occur if this function is used sensibly, but there is no danger to defending
    // against this.
    let Some(node_arc) = node else {
        return (None, None, false);
    };

    let node_mut = Arc::make_mut(node_arc);

    // Base case
    if node_mut.left_ref(node_resolver).is_none() {
        let mut min = node.take().expect("Already checked");
        let min_mut = Arc::make_mut(&mut min);

        let right = min_mut.right_mut(node_resolver).take();

        (Some(min), right, true)
    // Recursive
    } else {
        let old_node_bf = node_mut.balance_factor;
        let left = node_mut.left_mut(node_resolver);
        let (min, right, left_shrank) = take_min(left, node_resolver);

        if right.is_some() {
            *node_mut.left_mut(node_resolver) = right;
            node_mut.balance_factor += 1;
        } else if left_shrank {
            node_mut.balance_factor += 1;
        };

        *node_arc = rebalance(node_arc, node_resolver);
        (
            min,
            None,
            old_node_bf.abs() == 1 && node_arc.balance_factor == 0,
        )
    }
}
