use super::Key;
use crate::commit_operation::CommitOperationCollection;
use crate::merkle_layer::node::MavlNode;
use bytes::Bytes;
use std::cmp::Ordering;
use std::sync::Arc;

/// Delete the value of the node with a given key. If the key does not exist, do nothing.
///
/// Returns true if the subtree has shrank in size.
pub(super) fn delete(root: &mut Option<Arc<MavlNode>>, key: &Key) -> bool {
    let Some(node) = root else {
        // The key does not exist so nothing will happen.
        return false;
    };
    match node.key.cmp(key) {
        Ordering::Equal => {
            let node_mut = Arc::make_mut(node);
            match (
                node_mut.left_ref().is_some(),
                node_mut.right_ref().is_some(),
            ) {
                (false, false) => {
                    *root = None;
                    true
                }
                (true, false) => {
                    let left = node_mut.left_mut().as_mut().expect("Checked is_some()");
                    *node_mut = Arc::make_mut(left).clone();
                    true
                }
                (false, true) => {
                    let right = node_mut.right_mut().as_mut().expect("Checked is_some()");
                    *node_mut = Arc::make_mut(right).clone();
                    true
                }
                (true, true) => {
                    let (new_node, shrank) = replace_with_successor(node);
                    *node = new_node;
                    shrank
                }
            }
        }
        Ordering::Greater => {
            let node_mut = Arc::make_mut(node);
            let old_balance_factor = node_mut.balance_factor;

            let left_shrank = delete(node_mut.left_mut(), key);

            node_mut.balance_factor += if left_shrank { 1 } else { 0 };
            *node = rebalance(node);
            old_balance_factor.abs() == 1 && node.balance_factor == 0
        }
        Ordering::Less => {
            let node_mut = Arc::make_mut(node);
            let old_balance_factor = node_mut.balance_factor;

            let right_shrank = delete(node_mut.right_mut(), key);

            node_mut.balance_factor -= if right_shrank { 1 } else { 0 };
            *node = rebalance(node);
            old_balance_factor.abs() == 1 && node.balance_factor == 0
        }
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
///
/// Returns the rebalanced subtree.
#[must_use]
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
///
/// Returns the rotated subtree.
#[must_use]
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
///
/// Returns the rotated subtree.
#[must_use]
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
///
/// Returns the rotated subtree.
#[must_use]
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
///
/// Returns the rotated subtree.
#[must_use]
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

/// Remove the successor of the node from its subtree and replace the original node with it.
///
/// Returns:
///  - The new subtree.
///  - True if the subtree has shrank in size.
#[must_use]
fn replace_with_successor(node: &mut Arc<MavlNode>) -> (Arc<MavlNode>, bool) {
    let node_balance_factor = node.balance_factor;
    let node_mut = Arc::make_mut(node);
    let node_bf = node_mut.balance_factor;

    // If the right child has a left child, the successor is the min of the left child's subtree.
    let (mut successor, shrank) = if node_mut
        .right_ref()
        .as_ref()
        .expect("A node with a successor must have a right child")
        .left_ref()
        .is_some()
    {
        let right = node_mut.right_mut();
        let (min, _, shrank) = take_min(right);
        (
            min.expect("A node with a successor must have a right child"),
            shrank,
        )
    // If there is no left child of the right child, the successor is the right child.
    } else {
        let mut successor = node_mut
            .right_mut()
            .take()
            .expect("A node with a successor must have a right child");
        let successor_mut = Arc::make_mut(&mut successor);

        // Bump up the (optional) right child of the right child, causing the subtree to shrink.
        node_mut.right = successor_mut.right.take();
        (successor, true)
    };

    let successor_mut = Arc::make_mut(&mut successor);

    successor_mut.balance_factor = node_bf - if shrank { 1 } else { 0 };
    successor_mut.left = node.left.clone();
    successor_mut.right = node.right.clone();

    successor = rebalance(&mut successor);

    let shrank = node_balance_factor.abs() == 1 && successor.balance_factor == 0;
    (successor, shrank)
}

/// Set the value of the node with a given key.
///
/// Returns true if the subtree has grown in size.
pub(super) fn set(
    root: &mut Option<Arc<MavlNode>>,
    key: &Key,
    data: Bytes,
    commit_collection: &mut CommitOperationCollection,
) -> bool {
    let Some(node) = root else {
        // The key does not exist and a new node shall be created.
        *root = Some(Arc::new(MavlNode::new(key.clone(), data)));
        commit_collection.add_new_node_to_commit(root.as_ref().expect("This node is not None"));
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
            node.invalidate_hash();
            false
        }
        Ordering::Greater => {
            let node_mut = Arc::make_mut(node);
            let grew = set(node_mut.left_mut(), key, data, commit_collection);
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
            let grew = set(node_mut.right_mut(), key, data, commit_collection);
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

/// Remove the minimum node from this subtree and return it.
///
/// Returns:
///  - The minimum node.
///  - The minimum node's right child, if it hasn't been moved to its new position.
///  - True if the subtree has shrank in size.
#[must_use]
fn take_min(
    node: &mut Option<Arc<MavlNode>>,
) -> (Option<Arc<MavlNode>>, Option<Arc<MavlNode>>, bool) {
    // Shouldn't occur if this function is used sensibly, but there is no danger to defending
    // against this.
    let Some(node_arc) = node else {
        return (None, None, false);
    };

    let node_mut = Arc::make_mut(node_arc);

    // Base case
    if node_mut.left_ref().is_none() {
        let mut min = node.take().expect("Already checked");
        let min_mut = Arc::make_mut(&mut min);

        let right = min_mut.right_mut().take();

        (Some(min), right, true)
    // Recursive
    } else {
        let old_node_bf = node_mut.balance_factor;
        let left = node_mut.left_mut();
        let (min, right, left_shrank) = take_min(left);

        if right.is_some() {
            *node_mut.left_mut() = right;
            node_mut.balance_factor += 1;
        } else if left_shrank {
            node_mut.balance_factor += 1;
        };

        *node_arc = rebalance(node_arc);
        (
            min,
            None,
            old_node_bf.abs() == 1 && node_arc.balance_factor == 0,
        )
    }
}
