// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for a Merklisable node of an AVL tree

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::ops::Deref;
use std::sync::OnceLock;

use bincode::Decode;
use bincode::Encode;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::components::bytes::BytesMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use perfect_derive::perfect_derive;

use super::resolver::LazyDataId;
use super::resolver::LazyTreeId;
use super::resolver::ProveNode;
use super::resolver::TreeResolver;
use super::tree::Tree;
use crate::avl::resolver::AvlResolver;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::rkyv_codec::rkyv_deserialise;
use crate::rkyv_codec::rkyv_serialise;
use crate::storage::KeyValueStore;
use crate::storage::Loadable;
use crate::storage::Storable;
use crate::storage::StoreOptions;

/// Metadata of a [`Node`] needed for accesses.
///
/// `Meta` is serialised through two codecs: bincode (as a Merkle leaf, via the
/// shared `data` crate's fold machinery, which keeps the Merkle hashes and
/// proofs byte-identical) and rkyv (as part of the on-disk [`StoredNode`] blob).
#[derive(
    Clone, Default, Debug, Encode, Decode, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub(crate) struct Meta {
    key: Key,

    /// The difference in heights between child branches (right - left).
    balance_factor: i64,
}

/// This type is a compact serialised form of a [`Node`] with metadata and child subtree hashes.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct StoredNode {
    meta: Meta,
    data: Hash,
    left: Hash,
    right: Hash,
}

/// A node that supports rebalancing and Merklisation.
#[perfect_derive(Clone, Default, Debug)]
pub struct Node<TreeId, DataId, M: Mode> {
    meta: Atom<Meta, M>,
    data: DataId,
    left: TreeId,
    right: TreeId,

    /// A cache for the hash of this node. This uses `OnceLock` so that updating the cache is a
    /// non-mutating operation.
    ///
    /// An uninitialised hash is a hash that has not been set or has been dirtied.
    hash: OnceLock<Hash>,
}

impl Node<LazyTreeId, LazyDataId, Normal> {
    /// Converts the [`Node`] to prove mode.
    pub fn into_proof(self) -> ProveNode {
        Node {
            meta: self.meta.into_proof(),
            data: self.data.into_proof(),
            left: self.left.into_proof(),
            right: self.right.into_proof(),
            hash: self.hash.clone(),
        }
    }

    /// Resolve the data field on a lazy node.
    pub(crate) fn resolve_data(
        &self,
        resolver: &impl AvlResolver<super::resolver::LazyNodeId, LazyDataId, LazyTreeId, Normal>,
    ) -> Result<&Bytes<Normal>, OperationalError> {
        resolver.resolve_bytes(&self.data, &self.meta.key)
    }
}

impl<TreeId, DataId, M: Mode> Node<TreeId, DataId, M> {
    /// Mark the hash of this node as dirty.
    fn invalidate_hash(&mut self) {
        self.hash = OnceLock::new();
    }

    /// Direct access to the left child identifier.
    pub(crate) fn left_id(&self) -> &TreeId {
        &self.left
    }

    /// A mutable reference to the left branch.
    #[inline]
    pub(super) fn left_mut<NodeId>(
        &mut self,
        resolver: &mut impl TreeResolver<NodeId, TreeId>,
    ) -> Result<&mut Tree<NodeId>, OperationalError> {
        self.invalidate_hash();
        resolver.resolve_mut(&mut self.left)
    }

    /// An immutable reference to the left branch.
    #[inline]
    pub(super) fn left_ref<NodeId>(
        &self,
        resolver: &impl TreeResolver<NodeId, TreeId>,
    ) -> Result<&Tree<NodeId>, OperationalError> {
        resolver.resolve(&self.left)
    }

    /// Direct access to the right child identifier.
    pub(crate) fn right_id(&self) -> &TreeId {
        &self.right
    }

    /// A mutable reference to the right branch.
    #[inline]
    pub(super) fn right_mut<NodeId>(
        &mut self,
        resolver: &mut impl TreeResolver<NodeId, TreeId>,
    ) -> Result<&mut Tree<NodeId>, OperationalError> {
        self.invalidate_hash();
        resolver.resolve_mut(&mut self.right)
    }

    /// An immutable reference to the right branch.
    #[inline]
    pub(super) fn right_ref<NodeId>(
        &self,
        resolver: &impl TreeResolver<NodeId, TreeId>,
    ) -> Result<&Tree<NodeId>, OperationalError> {
        resolver.resolve(&self.right)
    }
}

impl<TreeId, DataId, M: AtomMode> Node<TreeId, DataId, M> {
    /// Find the id of the [`Node`] within the subtree of this [`Node`] with a given [`Key`].
    pub(super) fn get<'a, NodeId>(
        mut node: &'a NodeId,
        key: &Key,
        resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<Option<&'a NodeId>, OperationalError>
    where
        NodeId: 'a,
        DataId: 'a,
        TreeId: 'a,
        M: 'a,
    {
        loop {
            let resolved_node = resolver.resolve(node)?;
            match resolved_node.key().cmp(key) {
                Ordering::Equal => return Ok(Some(node)),
                Ordering::Greater => {
                    let Some(left) = resolved_node.left_ref(resolver)?.root() else {
                        return Ok(None);
                    };
                    node = left;
                }
                Ordering::Less => {
                    let Some(right) = resolved_node.right_ref(resolver)?.root() else {
                        return Ok(None);
                    };
                    node = right;
                }
            }
        }
    }

    /// Create a new leaf [`Node`] from the given key and data.
    pub(crate) fn new(key: Key, data: impl Into<DataId>) -> Self
    where
        TreeId: Default,
    {
        Node {
            meta: Atom::new(Meta {
                key,
                balance_factor: 0,
            }),
            data: data.into(),
            left: TreeId::default(),
            right: TreeId::default(),
            hash: OnceLock::new(),
        }
    }

    /// Construct a [`Node`] from its branches in a proof, advancing the deserialiser context.
    pub(super) fn from_branches<D: DeserialiserNode>(
        ctx: D,
    ) -> Result<(D, Self), <D::Parent as Deserialiser>::Error>
    where
        Atom<Meta, M>: FromProof,
        DataId: FromProof,
        TreeId: FromProof,
    {
        let (ctx, meta) = ctx.next_branch::<Atom<Meta, M>>()?;
        let (ctx, data) = ctx.next_branch::<DataId>()?;
        let (ctx, left) = ctx.next_branch::<TreeId>()?;
        let (ctx, right) = ctx.next_branch::<TreeId>()?;

        let node = Node {
            meta,
            data,
            left,
            right,
            hash: Default::default(),
        };

        Ok((ctx, node))
    }

    /// Returns the hash of this node.
    ///
    /// If the hash has been cached, the memo is returned. Otherwise, the hash is calculated and
    /// cached.
    pub(crate) fn hash(&self) -> &Hash
    where
        Atom<Meta, M>: Foldable<HashFold>,
        DataId: Foldable<HashFold>,
        TreeId: Foldable<HashFold>,
    {
        self.hash.get_or_init(|| Hash::from_foldable(self))
    }

    /// The metadata of this [`Node`].
    #[inline]
    pub(crate) fn meta(&self) -> &Atom<Meta, M> {
        &self.meta
    }

    /// The difference in heights between child branches.
    #[inline]
    pub(crate) fn balance_factor(&self) -> i64 {
        self.meta.balance_factor
    }

    /// A mutable reference to the difference in heights between child branches.
    #[inline]
    pub(super) fn balance_factor_mut(&mut self) -> &mut i64 {
        &mut self.meta.balance_factor
    }

    /// The [`Key`] used for determining the [`Node`].
    #[inline]
    pub(crate) fn key(&self) -> &Key {
        &self.meta.key
    }

    /// Retrieve the value associated with this node.
    #[inline]
    pub(crate) fn data(&self) -> &DataId {
        &self.data
    }

    /// Rebalance the subtree of the [`Node`] so that the difference in height between child
    /// branches is in the range of -1..=1.
    ///
    /// The subtree of the [`Node`] must already have balance factor in the range of -2..=2, else
    /// it is an invalid AVL tree.
    pub(super) fn rebalance<NodeId: Clone>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(), OperationalError> {
        let resolved_node = resolver.resolve(node)?;
        let balance_factor = resolved_node.balance_factor();
        match balance_factor {
            2 => {
                let right_balance = resolved_node
                    .right_ref(resolver)?
                    .balance_factor(resolver)?;

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
                let left_balance = resolved_node.left_ref(resolver)?.balance_factor(resolver)?;

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
    pub(super) fn replace_with_successor<NodeId>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(NodeId, bool), OperationalError>
    where
        NodeId: Clone,
        TreeId: Default,
    {
        let node_mut = resolver.resolve_mut(node)?;

        // If the right child has a left child, the successor is the min of the left child's subtree.
        let (mut successor, shrank) = if resolver
            .resolve(
                node_mut
                    .right_ref(resolver)?
                    .root()
                    .expect("A node with a successor must have a right child"),
            )?
            .left_ref(resolver)?
            .root()
            .is_some()
        {
            let right = node_mut.right_mut(resolver)?;
            let (mut min, _, shrank) = Tree::take_min(right, resolver)?;
            (
                min.take()
                    .expect("A node with a successor must have a right child"),
                shrank,
            )
        // If there is no left child of the right child, the successor is the right child.
        } else {
            let mut successor = node_mut
                .right_mut(resolver)?
                .take()
                .expect("A node with a successor must have a right child");
            let successor_mut = resolver.resolve_mut(&mut successor)?;

            // Bump up the (optional) right child of the right child, causing the subtree to shrink.
            let target_right = node_mut.right_mut(resolver)?;
            *target_right = Tree::from(successor_mut.right_mut(resolver)?.take());

            (successor, true)
        };

        let successor_mut = resolver.resolve_mut(&mut successor)?;

        successor_mut.meta.balance_factor = node_mut.balance_factor() - if shrank { 1 } else { 0 };
        successor_mut.left = std::mem::take(&mut node_mut.left);
        successor_mut.right = std::mem::take(&mut node_mut.right);

        Self::rebalance(&mut successor, resolver)?;

        let successor_balance_factor = resolver.resolve(&successor)?.balance_factor();
        let shrank = node_mut.balance_factor().abs() == 1 && successor_balance_factor == 0;

        Ok((successor, shrank))
    }

    /// Takes the occupied [`Tree`] with the minimum [`Key`] from this [`Node`]'s subtree and
    /// replaces it with an empty [`Tree`].
    ///
    /// Returns a tuple of:
    ///  - The occupied [`Tree`] with the minimum [`Key`].
    ///  - The minimum [`Tree`]'s right child, if it hasn't been moved to its new position.
    ///  - True if this [`Node`]'s subtree has shrunk in size.
    pub(super) fn take_min<NodeId>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(Tree<NodeId>, Tree<NodeId>, bool), OperationalError>
    where
        NodeId: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;

        let old_node_bf = node_mut.balance_factor();
        let left = node_mut.left_mut(resolver)?;
        let (min, right, left_shrank) = Tree::take_min(left, resolver)?;

        if right.root().is_some() {
            let target_node_left = node_mut.left_mut(resolver)?;
            *target_node_left = right;
            node_mut.meta.balance_factor += 1;
        } else if left_shrank {
            node_mut.meta.balance_factor += 1;
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
    pub(super) fn upsert<NodeId>(
        &mut self,
        key: &Key,
        offset: usize,
        data: impl FnOnce(&mut Bytes<M>) -> Result<(), Error>,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, Error>
    where
        TreeId: Default,
        DataId: From<Bytes<M>>,
        NodeId: Clone + From<Node<TreeId, DataId, M>>,
        M: BytesMode,
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
        match self.meta.key.cmp(key) {
            // The key already exists and should be updated.
            Ordering::Equal => {
                let bytes = resolver.resolve_mut_bytes(&mut self.data, &self.meta.key)?;
                data(bytes)?;
                self.invalidate_hash();
                Ok(false)
            }
            Ordering::Greater => {
                let grew = self
                    .left_mut(resolver)?
                    .upsert(key, offset, data, resolver)?;
                if grew {
                    self.meta.balance_factor -= 1;
                }
                Ok(grew)
            }
            Ordering::Less => {
                let grew = self
                    .right_mut(resolver)?
                    .upsert(key, offset, data, resolver)?;
                if grew {
                    self.meta.balance_factor += 1;
                }
                Ok(grew)
            }
        }
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
    fn rotate_left<NodeId>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(), OperationalError>
    where
        NodeId: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;
        let mut right = node_mut
            .right_mut(resolver)?
            .take()
            .expect("There should be a right node to rotate left");
        let right_mut = resolver.resolve_mut(&mut right)?;

        let target_tree = node_mut.right_mut(resolver)?;
        *target_tree = Tree::from(right_mut.left_mut(resolver)?.take());

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
        let new_node_bf =
            node_mut.meta.balance_factor - 1 + std::cmp::min(-right_mut.meta.balance_factor, 0);
        node_mut.meta.balance_factor = new_node_bf;

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
        right_mut.meta.balance_factor =
            right_mut.meta.balance_factor - 1 + std::cmp::min(new_node_bf, 0);

        let target_right_left = right_mut.left_mut(resolver)?;
        *target_right_left = Tree::from(Some(node.clone()));

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
    fn rotate_left_right<NodeId>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(), OperationalError>
    where
        NodeId: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;

        let mut left = node_mut
            .left_mut(resolver)?
            .take()
            .expect("Left child must exist for the right rotation of the node");
        let left_mut = resolver.resolve_mut(&mut left)?;

        let mut left_right = left_mut
            .right_mut(resolver)?
            .take()
            .expect("Left's right child must exist for the left rotation of the left node");
        let left_right_mut = resolver.resolve_mut(&mut left_right)?;

        // From the `rotate_left` derivation, the first rotation does:
        //   new_A_bf_1 = old_A_bf - 1 + std::cmp::min(-A.right.balance_factor, 0)
        // As this function assumes old_A_bf is +1:
        //   new_A_bf_1 = std::cmp::min(-A.right.balance_factor, 0)
        // The second rotation doesn't mutate A's subtree, so the final balance factor is:
        left_mut.meta.balance_factor = std::cmp::min(-left_right_mut.meta.balance_factor, 0);

        // B's right child is between B and B, it's moved to node's left
        let target_node_left = node_mut.left_mut(resolver)?;
        *target_node_left = Tree::from(left_right_mut.right_mut(resolver)?.take());

        // B's left child is between A and B, it's moved to A's right
        let target_node_right = left_mut.right_mut(resolver)?;
        *target_node_right = Tree::from(left_right_mut.left_mut(resolver)?.take());

        // Set A
        let target_left_right_left = left_right_mut.left_mut(resolver)?;
        *target_left_right_left = Tree::from(Some(left));

        // If B is 0 or 1, the new node balance factor will be 0
        // If B is -1, the new node balance factor will be 1
        node_mut.meta.balance_factor = std::cmp::max(0, -left_right_mut.meta.balance_factor);

        // Set node
        let target_left_right_right = left_right_mut.right_mut(resolver)?;
        *target_left_right_right = Tree::from(Some(node.clone()));

        // The new root will always be balanced
        left_right_mut.meta.balance_factor = 0;
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
    fn rotate_right<NodeId>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(), OperationalError>
    where
        NodeId: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;
        let mut left = node_mut
            .left_mut(resolver)?
            .take()
            .expect("There should be a left node to rotate right");
        let left_mut = resolver.resolve_mut(&mut left)?;

        let target_node_left = node_mut.left_mut(resolver)?;
        *target_node_left = Tree::from(left_mut.right_mut(resolver)?.take());

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
        let new_node_bf =
            node_mut.meta.balance_factor + 1 + std::cmp::max(0, -left_mut.meta.balance_factor);
        node_mut.meta.balance_factor = new_node_bf;

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
        left_mut.meta.balance_factor =
            left_mut.meta.balance_factor + 1 + std::cmp::max(new_node_bf, 0);

        let target_left_right = left_mut.right_mut(resolver)?;
        *target_left_right = Tree::from(Some(node.clone()));

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
    fn rotate_right_left<NodeId>(
        node: &mut NodeId,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(), OperationalError>
    where
        NodeId: Clone,
    {
        let node_mut = resolver.resolve_mut(node)?;
        let mut right = node_mut
            .right_mut(resolver)?
            .take()
            .expect("Right child must exist for the left rotation of the node");
        let right_mut = resolver.resolve_mut(&mut right)?;

        let mut right_left = right_mut
            .left_mut(resolver)?
            .take()
            .expect("Right's left child must exist for the right rotation of the right node");
        let right_left_mut = resolver.resolve_mut(&mut right_left)?;

        // From the `rotate_right` derivation, the first rotation does:
        //   new_A_bf_1 = old_A_bf + 1 + std::cmp::max(0, -A.left.balance_factor)
        // As this function assumes old_A_bf is -1:
        //   new_A_bf_1 = std::cmp::max(0, -A.left.balance_factor)
        // The second rotation doesn't mutate A's subtree, so the final balance factor is:
        right_mut.meta.balance_factor = std::cmp::max(0, -right_left_mut.meta.balance_factor);

        // B's left child is between node and B, it's moved to node's right
        let target_node_right = node_mut.right_mut(resolver)?;
        *target_node_right = Tree::from(resolver.resolve_mut(&mut right_left_mut.left)?.take());

        // B's right child is between B and A, it's moved to A's left
        let target_right_left = right_mut.left_mut(resolver)?;
        *target_right_left = Tree::from(resolver.resolve_mut(&mut right_left_mut.right)?.take());

        // Set A
        let target_right_left_right = right_left_mut.right_mut(resolver)?;
        *target_right_left_right = Tree::from(Some(right));

        // If B is 0 or -1, the new node balance factor will be 0
        // If B is 1, the new node balance factor will be -1
        node_mut.meta.balance_factor = -std::cmp::max(0, right_left_mut.meta.balance_factor);

        // Set node
        let target_right_left_left = right_left_mut.left_mut(resolver)?;
        *target_right_left_left = Tree::from(Some(node.clone()));

        // The new root will always be balanced
        right_left_mut.meta.balance_factor = 0;
        *node = right_left;
        Ok(())
    }
}

#[cfg(test)]
impl<TreeId, DataId, M: AtomMode> Node<TreeId, DataId, M> {
    /// Returns true if the balance factors stored in the [`Node`]'s subtree are correct.
    pub(super) fn has_correct_balance_factors<NodeId>(
        &self,
        resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, OperationalError>
    where
        NodeId: std::fmt::Debug,
        TreeId: std::fmt::Debug,
        DataId: std::fmt::Debug,
        Atom<Meta, M>: std::fmt::Debug,
    {
        let left_height = self.left_ref(resolver)?.height(resolver)?;
        let right_height = self.right_ref(resolver)?.height(resolver)?;
        let calculated_balance_factor = right_height as i64 - left_height as i64;
        if self.balance_factor() != calculated_balance_factor {
            eprintln!(
                "Node has balance_factor {:?}, should be {calculated_balance_factor:?}\nnode: {self:?}",
                self.balance_factor()
            );
            return Ok(false);
        }

        let left_correct = self
            .left_ref(resolver)?
            .has_correct_balance_factors(resolver)?;

        let right_correct = self
            .right_ref(resolver)?
            .has_correct_balance_factors(resolver)?;

        Ok(left_correct && right_correct)
    }

    /// Returns the height of this [`Node`]'s subtree.
    pub(super) fn height<NodeId>(
        &self,
        resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<u32, OperationalError> {
        let left_height = self.left_ref(resolver)?.height(resolver)?;
        let right_height = self.right_ref(resolver)?.height(resolver)?;
        Ok(1 + std::cmp::max(left_height, right_height))
    }

    /// Returns true if this [`Node`]'s subtree is balanced.
    pub(super) fn is_balanced<NodeId>(
        &self,
        resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, OperationalError> {
        let balance_factor = self.balance_factor();
        if balance_factor.abs() > 1 {
            eprintln!("Balance factor not in -1..=1: {balance_factor:?}");
            return Ok(false);
        }

        let left_balanced = self.left_ref(resolver)?.is_balanced(resolver)?;
        let right_balanced = self.right_ref(resolver)?.is_balanced(resolver)?;

        Ok(left_balanced && right_balanced)
    }

    /// Returns true if this [`Node`]'s subtree is in-order.
    pub(super) fn is_inorder<NodeId>(
        &self,
        min: &Key,
        max: &Key,
        resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, OperationalError> {
        if self.key() < min || self.key() > max {
            return Ok(false);
        }

        let left_in_order = self
            .left_ref(resolver)?
            .is_inorder_inner(min, self.key(), resolver)?;

        let right_in_order =
            self.right_ref(resolver)?
                .is_inorder_inner(self.key(), max, resolver)?;

        Ok(left_in_order && right_in_order)
    }
}

impl<TreeId: Storable, DataId> Storable for Node<TreeId, DataId, Normal>
where
    DataId: Foldable<HashFold> + Borrow<[u8]>,
{
    fn store(
        &self,
        store: &impl KeyValueStore,
        options: &StoreOptions,
    ) -> Result<(), OperationalError> {
        // The stored representation is more compact. We don't include the `data` field, as that
        // should be written to the KV store separately.
        let repr = StoredNode {
            meta: self.meta.deref().clone(),
            data: Hash::from_foldable(&self.data),
            left: Hash::from_foldable(&self.left),
            right: Hash::from_foldable(&self.right),
        };

        let &id = self.hash();
        let bytes = rkyv_serialise(&repr)?;
        store.blob_set(id, bytes)?;

        // Are we in charge of writing the value data to the KV store?
        if options.node_data() {
            let key: &[u8] = self.meta.key.as_ref();
            let value: &[u8] = self.data.borrow();
            store.set(key, value)?;
        }

        if options.deep() {
            self.left.store(store, options)?;
            self.right.store(store, options)?;
        }

        Ok(())
    }
}

impl<TreeId: Loadable, DataId: DataLoadable> Loadable for Node<TreeId, DataId, Normal> {
    fn load(id: Hash, store: &impl KeyValueStore) -> Result<Self, OperationalError> {
        let StoredNode {
            meta,
            left,
            right,
            data,
        } = {
            let bytes =
                store
                    .blob_get(id)
                    .map_err(|error| OperationalError::CommitDataMissing {
                        root: id,
                        source: Box::new(error),
                    })?;
            rkyv_deserialise(bytes.as_ref())?
        };

        let meta = Atom::new(meta);

        // The stored representation does not include the `data` field, so we need to load it
        // separately from the KV store.
        let data = DataId::load(data, &meta.key, store)?;

        let left = TreeId::load(left, store)?;
        let right = TreeId::load(right, store)?;

        Ok(Self {
            meta,
            data,
            left,
            right,
            hash: OnceLock::new(),
        })
    }
}

impl<F, TreeId, DataId, M> Foldable<F> for Node<TreeId, DataId, M>
where
    F: Fold,
    M: Mode,
    Atom<Meta, M>: Foldable<F>,
    DataId: Foldable<F>,
    TreeId: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        let mut node = builder.into_node_fold();
        node.add(&self.meta);
        node.add(&self.data);
        node.add(&self.left);
        node.add(&self.right);
        node.done()
    }
}

/// Helper trait for node data that can be reconstructed
/// from a [`KeyValueStore`] by content hash and key.
trait DataLoadable: Sized {
    /// Load data identified by `id` and `key` from `store`.
    fn load(id: Hash, key: &Key, store: &impl KeyValueStore) -> Result<Self, OperationalError>;
}

impl DataLoadable for Bytes<Normal> {
    fn load(_id: Hash, key: &Key, store: &impl KeyValueStore) -> Result<Self, OperationalError> {
        let data = {
            let bytes =
                store
                    .get(key.as_ref())
                    .map_err(|error| OperationalError::CommitValueMissing {
                        key: key.clone(),
                        source: Box::new(error),
                    })?;
            Bytes::from(bytes.as_ref())
        };

        Ok(data)
    }
}

impl DataLoadable for LazyDataId {
    fn load(id: Hash, key: &Key, store: &impl KeyValueStore) -> Result<Self, OperationalError> {
        // TODO (RV-998): go all in on laziness
        let data = match store.get(key.as_ref()) {
            Ok(bytes) => {
                let bytes = Bytes::from(bytes.as_ref());
                LazyDataId::from(bytes)
            }
            // On deletion, the persistence layer has already deleted the value from the
            // KV-store. Therefore, loading fails with `KeyNotFound`. To ensure deletion
            // still succeeds, we wrap the value. This is okay as the node will be deleted
            // immediately.
            //
            // Attempting to do anything with the data (except hashing) will fail.
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound)) => LazyDataId::from(id),
            Err(error) => {
                return Err(OperationalError::CommitValueMissing {
                    key: key.clone(),
                    source: Box::new(error),
                });
            }
        };

        Ok(data)
    }
}
