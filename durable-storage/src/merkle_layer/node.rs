// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;

use bytes::Bytes;

use super::Key;

/// A node that supports rebalancing and Merklisation.
#[derive(Clone, Default, Debug)]
pub(super) struct MavlNode {
    key: Key,
    data: Bytes,
    left: Option<Arc<Self>>,
    right: Option<Arc<Self>>,

    /// A [None] hash is a hash that has not been set or has been dirtied.
    hash: Option<blake3::Hash>,

    /// The difference in heights between child branches (right - left).
    balance_factor: i64,
}

impl MavlNode {
    /// The difference in heights between child branches.
    #[cfg(test)]
    pub(super) fn balance_factor(&self) -> i64 {
        self.balance_factor
    }

    /// The data stored in the node.
    pub(super) fn data(&self) -> &Bytes {
        &self.data
    }

    /// The key used for determining the node.
    pub(super) fn key(&self) -> &Key {
        &self.key
    }

    /// A mutable reference to the left branch.
    pub(super) fn left_mut(&mut self) -> &mut Option<Arc<Self>> {
        self.invalidate_hash();
        &mut self.left
    }

    /// An immutable reference to the left branch.
    pub(super) fn left_ref(&self) -> &Option<Arc<Self>> {
        &self.left
    }

    /// Create a new leaf node from the given key and data.
    pub(super) fn new(key: Key, data: Bytes) -> Self {
        MavlNode {
            key,
            data,
            balance_factor: 0,
            ..Default::default()
        }
    }

    /// A mutable reference to the right branch.
    pub(super) fn right_mut(&mut self) -> &mut Option<Arc<Self>> {
        self.invalidate_hash();
        &mut self.right
    }

    /// An immutable reference to the right branch.
    pub(super) fn right_ref(&self) -> &Option<Arc<Self>> {
        &self.right
    }

    /// Mark the hash of this node as dirty.
    fn invalidate_hash(&mut self) {
        self.hash = None;
    }
}

/// The data stored in a node in the tree with a given key.
pub(super) fn get<'a>(root: &'a Option<Arc<MavlNode>>, key: &Key) -> Option<&'a Bytes> {
    let mut node = root.as_deref()?;
    loop {
        match node.key().cmp(key) {
            Ordering::Equal => return Some(node.data()),
            Ordering::Greater => node = node.left_ref().as_deref()?,
            Ordering::Less => node = node.right_ref().as_deref()?,
        }
    }
}

/// Rebalance the node so that the difference in height between child branches is in the range
/// of -1..=1.
///
/// The node must already have balance factor in the range of -2..=2, or it is an invalid AVL
/// node.
fn rebalance(node: &mut Arc<MavlNode>) -> Arc<MavlNode> {
    match node.balance_factor {
        2 => {
            let right_balance = node.right.as_ref().map_or(0, |r| r.balance_factor);

            match right_balance {
                1 | 0 => rotate_left(node),
                -1 => rotate_right_left(node),
                _ => panic!(
                    "Rebalancing an invalid AVL tree. The balance factor of the right node is {right_balance:?}, but it should be in the range of -1..=1"
                ),
            }
        }
        -1..=1 => node.clone(),
        -2 => {
            let left_balance = node.left.as_ref().map_or(0, |l| l.balance_factor);

            match left_balance {
                1 => rotate_left_right(node),
                -1 | 0 => rotate_right(node),
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
fn rotate_left(node: &mut Arc<MavlNode>) -> Arc<MavlNode> {
    let node_mut = Arc::make_mut(node);
    let mut right = node_mut
        .right_mut()
        .take()
        .expect("There should be a right node to rotate left");
    let right_mut = Arc::make_mut(&mut right);

    *node_mut.right_mut() = right_mut.left_mut().take();

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

    *right_mut.left_mut() = Some(node.clone());
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
fn rotate_right(node: &mut Arc<MavlNode>) -> Arc<MavlNode> {
    let node_mut = Arc::make_mut(node);
    let mut left = node_mut
        .left_mut()
        .take()
        .expect("There should be a left node to rotate right");
    let left_mut = Arc::make_mut(&mut left);

    *node_mut.left_mut() = left_mut.right_mut().take();

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

    *left_mut.right_mut() = Some(node.clone());
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
fn rotate_left_right(node: &mut Arc<MavlNode>) -> Arc<MavlNode> {
    let node_mut = Arc::make_mut(node);

    let mut left = node_mut
        .left_mut()
        .take()
        .expect("Left child must exist for the right rotation of the node");
    let left_mut = Arc::make_mut(&mut left);

    let mut left_right = left_mut
        .right_mut()
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
    node_mut.left = left_right_mut.right.take();

    // B's left child is between A and B, it's moved to A's right
    left_mut.right = left_right_mut.left.take();

    // Set A
    left_right_mut.left = Some(left);

    // If B is 0 or 1, the new node balance factor will be 0
    // If B is -1, the new node balance factor will be 1
    node_mut.balance_factor = std::cmp::max(0, -left_right_mut.balance_factor);

    // Set node
    left_right_mut.right = Some(node.clone());

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
fn rotate_right_left(node: &mut Arc<MavlNode>) -> Arc<MavlNode> {
    let node_mut = Arc::make_mut(node);

    let mut right = node_mut
        .right_mut()
        .take()
        .expect("Right child must exist for the left rotation of the node");
    let right_mut = Arc::make_mut(&mut right);

    let mut right_left = right_mut
        .left_mut()
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
    node_mut.right = right_left_mut.left.take();

    // B's right child is between B and A, it's moved to A's left
    right_mut.left = right_left_mut.right.take();

    // Set A
    right_left_mut.right = Some(right);

    // If B is 0 or -1, the new node balance factor will be 0
    // If B is 1, the new node balance factor will be -1
    node_mut.balance_factor = -std::cmp::max(0, right_left_mut.balance_factor);

    // Set node
    right_left_mut.left = Some(node.clone());

    // The new root will always be balanced
    right_left_mut.balance_factor = 0;
    right_left
}

/// Set the value of the node with a given key.
///
/// Returns true if the subtree has grown in size.
pub(super) fn set(root: &mut Option<Arc<MavlNode>>, key: &Key, data: Bytes) -> bool {
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
            node.data = data;
            false
        }
        Ordering::Greater => {
            let node_mut = Arc::make_mut(node);
            let grew = set(&mut node_mut.left, key, data);
            if grew {
                node_mut.balance_factor -= 1;
                *node = rebalance(node);
                node.balance_factor != 0
            } else {
                false
            }
        }
        Ordering::Less => {
            let node_mut = Arc::make_mut(node);
            let grew = set(&mut node_mut.right, key, data);
            if grew {
                node_mut.balance_factor += 1;
                *node = rebalance(node);
                node.balance_factor != 0
            } else {
                false
            }
        }
    }
}
