// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for an optional root [`Node`] of a Merklisable AVL tree

use std::cmp::Ordering;
use std::sync::LazyLock;

use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::components::bytes::BytesMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserError;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::ProofError;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::utils::not_found;
use perfect_derive::perfect_derive;

use super::node::Node;
use super::resolver::ProveNodeId;
use super::resolver::VerifyNodeId;
use crate::avl::resolver::AvlResolver;
use crate::avl::resolver::LazyNodeId;
use crate::avl::resolver::NodeResolver;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::storage::KeyValueStore;
use crate::storage::Loadable;
use crate::storage::Storable;
use crate::storage::StoreOptions;

/// Hash of the empty AVL tree (`Tree::<NodeId>(None)`).
/// The empty-tree hash is independent of the `NodeId` type parameter.
static EMPTY_TREE_HASH: LazyLock<Hash> =
    LazyLock::new(|| Hash::from_foldable(&Tree::<LazyNodeId>::default()));

/// A key-value store tree with left and right nodes that supports traversal and value retrieval.
#[perfect_derive(Clone, Default, Debug)]
#[derive(derive_more::From)]
pub struct Tree<NodeId>(Option<NodeId>);

impl Tree<LazyNodeId> {
    /// Converts the [`Tree`] to [`Prove`] mode.
    ///
    /// [`Prove`]: octez_riscv_data::mode::Prove
    pub(crate) fn into_proof(self) -> Tree<ProveNodeId> {
        Tree(self.0.map(|id| id.into_proof()))
    }
}

impl<NodeId: FromProof> Tree<NodeId> {
    /// Parse a tree from a proof deserialiser node.
    pub(super) fn from_branches<D: DeserialiserNode>(
        ctx: D,
    ) -> Result<(D, Partial<Self>), <D::Parent as Deserialiser>::Error> {
        match ctx.presence() {
            Partial::Absent => Ok((ctx, Partial::Absent)),
            // TODO: RV-895: The proof should include the empty tree rather
            // than blinding it.
            Partial::Blinded(hash) if hash == *EMPTY_TREE_HASH => {
                Ok((ctx, Partial::Present(Tree::default())))
            }
            Partial::Blinded(hash) => Ok((ctx, Partial::Blinded(hash))),
            Partial::Present(()) => {
                // TODO (RV-996): leverage <bool> as a union tag to allow future versioning
                //                of the database/merkle layer in a backwards-compatible way
                let (ctx, present) = ctx.next_branch_with(|proof| proof.into_leaf::<bool>())?;
                match present {
                    Partial::Present(true) => {
                        let (ctx, node_id) = ctx.next_branch()?;
                        Ok((ctx, Partial::Present(Tree::from(Some(node_id)))))
                    }
                    Partial::Present(false) => Ok((ctx, Partial::Present(Tree::default()))),
                    // SAFETY: called only in `Verify` mode
                    Partial::Blinded(_) | Partial::Absent => unsafe { not_found() },
                }
            }
        }
    }
}

impl<NodeId> Tree<NodeId> {
    /// Delete the [`Node`] in the [`Tree`] with a given key.
    ///
    /// Returns true if the [`Tree`] has shrunk in size.
    pub fn delete<TreeId, DataId, M: AtomMode>(
        &mut self,
        key: &Key,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, OperationalError>
    where
        NodeId: Clone,
        TreeId: Default,
    {
        let old_balance_factor = self.balance_factor(resolver)?;
        let Some(node) = self.root_mut() else {
            // The key does not exist so nothing will happen.
            return Ok(false);
        };

        let resolved_node = resolver.resolve(node)?;
        match resolved_node.key().cmp(key) {
            Ordering::Equal => {
                resolver.track_deleted_node(node);
                match (
                    resolved_node.left_ref(resolver)?.root(),
                    resolved_node.right_ref(resolver)?.root(),
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
                }
            }
            Ordering::Greater => {
                let node_mut = resolver.resolve_mut(node)?;
                let left_shrank = node_mut.left_mut(resolver)?.delete(key, resolver)?;
                *node_mut.balance_factor_mut() += if left_shrank { 1 } else { 0 };
                self.rebalance(resolver)?;
                Ok(old_balance_factor.abs() == 1 && self.balance_factor(resolver)? == 0)
            }
            Ordering::Less => {
                let node_mut = resolver.resolve_mut(node)?;
                let right_shrank = node_mut.right_mut(resolver)?.delete(key, resolver)?;
                *node_mut.balance_factor_mut() -= if right_shrank { 1 } else { 0 };
                self.rebalance(resolver)?;
                Ok(old_balance_factor.abs() == 1 && self.balance_factor(resolver)? == 0)
            }
        }
    }

    #[inline]
    /// Find the id of the [`Node`] in the [`Tree`] with a given [`Key`].
    pub(crate) fn get<'a, TreeId: 'a, DataId: 'a, M: AtomMode + 'a>(
        &'a self,
        key: &Key,
        resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<Option<&'a NodeId>, OperationalError> {
        let Some(node) = self.root() else {
            return Ok(None);
        };
        Node::get(node, key, resolver)
    }

    #[inline]
    /// Set the value of the [`Node`] with a given key.
    ///
    /// Returns true if the [`Tree`] has grown in size.
    pub fn set<TreeId, DataId, M: BytesMode + AtomMode>(
        &mut self,
        key: &Key,
        data: &[u8],
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, OperationalError>
    where
        NodeId: Clone + From<Node<TreeId, DataId, M>>,
        DataId: From<Bytes<M>>,
        TreeId: Default,
    {
        let result = self.upsert(
            key,
            0,
            |old_data| {
                old_data.set(data);
                Ok(())
            },
            resolver,
        );

        result.map_err(|e| match e {
            crate::errors::Error::Operational(e) => e,
            crate::errors::Error::InvalidArgument(_) => {
                unreachable!("`set` at offset 0 cannot produce InvalidArgumentError")
            }
        })
    }

    /// Returns the hash of this tree.
    pub(crate) fn hash(&self) -> Hash
    where
        NodeId: Foldable<HashFold>,
    {
        Hash::from_foldable(self)
    }

    /// The hash of a non-empty [`Tree`] whose root [`Node`] hashes to `node_hash`.
    ///
    /// A present tree folds as `H(present_flag, node_hash)`, so this reuses the
    /// [`Foldable<HashFold>`] implementation (via a throwaway `Tree<Hash>`) to stay in
    /// lock-step with it. This is the storage key under which the node body is persisted.
    pub(crate) fn present_hash(node_hash: Hash) -> Hash {
        Tree::<Hash>::from(Some(node_hash)).hash()
    }

    /// Take the root [`Node`] out of this tree, leaving the [`Tree`] empty.
    pub(crate) const fn take(&mut self) -> Option<NodeId> {
        self.0.take()
    }

    #[inline]
    /// The difference in heights between any child branches in the [`Tree`].
    pub(super) fn balance_factor<TreeId, DataId, M: AtomMode>(
        &self,
        resolver: &impl NodeResolver<NodeId, DataId, TreeId, M>,
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
    pub(crate) fn root(&self) -> Option<&NodeId> {
        self.0.as_ref()
    }

    #[inline]
    /// A mutable reference to the root [`Node`].
    pub(super) fn root_mut(&mut self) -> Option<&mut NodeId> {
        self.0.as_mut()
    }

    /// Takes the occupied [`Tree`] with the minimum [`Key`] from this [`Tree`] and replaces it
    /// with an empty [`Tree`].
    ///
    /// Returns a tuple of:
    ///  - The occupied [`Tree`] with the minimum [`Key`].
    ///  - The minimum [`Tree`]'s right child, if it hasn't been moved to its new position.
    ///  - True if the [`Tree`] has shrunk in size.
    pub(super) fn take_min<TreeId, DataId, M: AtomMode>(
        &mut self,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(Tree<NodeId>, Tree<NodeId>, bool), OperationalError>
    where
        NodeId: Clone,
    {
        let Some(node_arc) = self.root_mut() else {
            return Ok((None.into(), None.into(), false));
        };

        let node_mut = resolver.resolve_mut(node_arc)?;

        // Base case
        if node_mut.left_ref(resolver)?.root().is_none() {
            let right = node_mut.right_mut(resolver)?.take();
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
    pub(crate) fn upsert<TreeId, DataId, M: BytesMode + AtomMode>(
        &mut self,
        key: &Key,
        offset: usize,
        data: impl FnOnce(&mut Bytes<M>) -> Result<(), Error>,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, Error>
    where
        NodeId: Clone + From<Node<TreeId, DataId, M>>,
        DataId: From<Bytes<M>>,
        TreeId: Default,
    {
        let node = self.root_mut();
        let Some(node) = node else {
            // The key does not exist and a new `Node` will be created unless the call to `data` fails.
            //
            // TODO: RV-895: Dynamic creation of the `Bytes<M>` state component may cause
            // problems with proof generation
            let mut new_data = Bytes::<M>::default();
            data(&mut new_data)?;

            let new_node: Node<TreeId, DataId, M> = Node::new(key.clone(), new_data);
            let new_id = NodeId::from(new_node);
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
    pub(crate) fn write<TreeId, DataId, M: BytesMode + AtomMode>(
        &mut self,
        key: &Key,
        offset: usize,
        data: &[u8],
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<bool, Error>
    where
        NodeId: Clone + From<Node<TreeId, DataId, M>>,
        DataId: From<Bytes<M>>,
        TreeId: Default,
    {
        self.upsert(
            key,
            offset,
            |old_data| {
                if offset > old_data.len() {
                    return Err(InvalidArgumentError::OffsetTooLarge)?;
                }

                let Some(new_data_end) = offset.checked_add(data.len()) else {
                    return Err(InvalidArgumentError::OffsetTooLarge)?;
                };

                let final_len = std::cmp::max(old_data.len(), new_data_end);
                old_data.resize(final_len);
                old_data.write(offset, data);
                Ok(())
            },
            resolver,
        )
    }

    /// Rebalance the [`Tree`] so that the difference in height between any child branches is in
    /// the range of -1..=1.
    ///
    /// The [`Tree`] must already have balance factor in the range of -2..=2, else it is an invalid
    /// AVL tree.
    fn rebalance<TreeId, DataId, M: AtomMode>(
        &mut self,
        resolver: &mut impl AvlResolver<NodeId, DataId, TreeId, M>,
    ) -> Result<(), OperationalError>
    where
        NodeId: Clone,
    {
        match self.root_mut() {
            Some(node) => Node::rebalance(node, resolver),
            None => Ok(()),
        }
    }
}

impl<NodeId: Foldable<HashFold>> Foldable<HashFold> for Tree<NodeId> {
    fn fold(&self, builder: HashFold) -> <HashFold as Fold>::Folded {
        let mut node = builder.into_node_fold();

        let present = self.0.is_some();
        node.add(&Hash::hash_encodable(present).expect("Hashing a bool should never fail"));

        if let Some(inner) = self.0.as_ref() {
            node.add(inner);
        }

        node.done()
    }
}

impl Foldable<PartialHashFold> for Tree<VerifyNodeId> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        // SAFETY: Extra care is required in folding for `Verify` in the MAVL tree, due to
        // consequences from restructuring due to AVL balancing. This fold correctness relies
        // on:
        // (a) the `bool_leaf` always folding to `PartialHash::Present(...)` so its `prev_hash` is
        // never substituted from the popped proof child.
        // (b) the inner `VerifyNodeId` overrides the proof with its own captured sub-proof, before
        // delegating to `Node::fold`. Otherwise, using `previous` could result in incorrect
        // substitution.
        let mut node = builder.into_node_fold();

        let present = self.0.is_some();
        let bool_hash = Hash::hash_encodable(present).expect("Hashing a bool should never fail");
        node.add(&PartialHash::Present(bool_hash));

        if let Some(inner) = self.0.as_ref() {
            node.add(inner);
        }

        node.done()
    }
}

impl FromProof for Tree<VerifyNodeId> {
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let ctx = proof.into_node()?;
        let (ctx, tree) = Tree::from_branches(ctx)?;
        // The top-level fold for a Normal-mode `Tree<LazyNodeId>` always emits a node fold,
        // so a well-formed proof's root deserialises as `Present`. `Blinded`/`Absent` here
        // would mean the entire state has been hidden, which would make the `Tree` unusable
        // for verification.
        let Partial::Present(tree) = tree else {
            return Err(Proof::Error::custom(ProofError::Custom(
                "malformed proof - deserialising MAVL tree without being present.".into(),
            )));
        };
        ctx.done(tree)
    }
}

impl<NodeId: Storable> Storable for Tree<NodeId> {
    fn store(
        &self,
        store: &impl KeyValueStore,
        options: &StoreOptions,
    ) -> Result<(), OperationalError> {
        // We don't store empty trees. All leaf nodes contain two empty trees. Adding
        // redundant writes to all leaves is not desirable. The empty tree can be recovered
        // during loading, as the hash of the empty tree is known.
        //
        // A non-empty tree's root node persists its body *directly under this tree's hash*
        // (see the `Storable` impl for `Node`). Storing the node by the parent tree hash
        // removes the intermediate tree->node pointer that previously had to be read before
        // the node itself, saving a lookup per level when resolving.
        match &self.0 {
            None => Ok(()),
            Some(node) => node.store(store, options),
        }
    }
}

impl<NodeId: Loadable> Loadable for Tree<NodeId> {
    fn load(id: Hash, store: &impl KeyValueStore) -> Result<Self, OperationalError> {
        // Empty trees are not stored. We can short-circuit here, if we detect the requested hash
        // corresponds to the hash of the empty tree.
        if id == *EMPTY_TREE_HASH {
            return Ok(Self(None));
        }

        // The root node's body is stored directly under this tree's hash. Hand the tree hash
        // to the node loader: for `LazyNodeId` this reads the body in a single lookup and
        // materialises the children lazily; for `ArcNodeId` it eagerly loads the whole subtree.
        NodeId::load(id, store).map(|node| Self(Some(node)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::prelude::*;
    use std::marker::PhantomData;

    use goldenfile::Mint;
    use octez_riscv_data::components::atom::Atom;
    use octez_riscv_data::mode::Normal;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    use super::*;
    use crate::avl::resolver::ArcNodeId;
    use crate::avl::resolver::ArcResolver;
    use crate::avl::resolver::ArcTreeId;
    use crate::avl::resolver::DataResolver;
    use crate::avl::resolver::Resolver;
    use crate::key::KEY_MAX_SIZE;
    use crate::key::Key;

    impl<NodeId> Tree<NodeId> {
        /// Asserts that the [`Tree`] is a valid AVL tree
        pub(crate) fn check<TreeId, DataId, M: AtomMode>(
            &self,
            resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
        ) -> Result<(), OperationalError>
        where
            NodeId: std::fmt::Debug,
            TreeId: std::fmt::Debug,
            DataId: std::fmt::Debug,
            Atom<i64, M>: std::fmt::Debug,
            Atom<Key, M>: std::fmt::Debug,
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
        pub(crate) fn is_inorder<TreeId, DataId, M: AtomMode>(
            &self,
            resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
        ) -> Result<bool, OperationalError> {
            self.is_inorder_inner(
                &Key::new(&[u8::MIN]).expect("Size less than KEY_MAX_SIZE"),
                &Key::new(&[u8::MAX; KEY_MAX_SIZE]).expect("Size less than KEY_MAX_SIZE"),
                resolver,
            )
        }

        /// Returns true if the balance factors stored in any [`Node`]'s subtree are correct.
        pub(crate) fn has_correct_balance_factors<TreeId, DataId, M: AtomMode>(
            &self,
            resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
        ) -> Result<bool, OperationalError>
        where
            NodeId: std::fmt::Debug,
            TreeId: std::fmt::Debug,
            DataId: std::fmt::Debug,
            Atom<i64, M>: std::fmt::Debug,
            Atom<Key, M>: std::fmt::Debug,
        {
            match self.root() {
                None => Ok(true),
                Some(node) => resolver
                    .resolve(node)
                    .map(|res| res.has_correct_balance_factors(resolver))?,
            }
        }

        /// Returns the height of the [`Tree`].
        pub(crate) fn height<TreeId, DataId, M: AtomMode>(
            &self,
            resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
        ) -> Result<u32, OperationalError> {
            match self.root() {
                None => Ok(0),
                Some(node) => resolver.resolve(node).map(|res| res.height(resolver))?,
            }
        }

        /// Returns true if the [`Tree`] is balanced.
        pub(crate) fn is_balanced<TreeId, DataId, M: AtomMode>(
            &self,
            resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
        ) -> Result<bool, OperationalError> {
            match self.root() {
                None => Ok(true),
                Some(node) => resolver
                    .resolve(node)
                    .map(|res| res.is_balanced(resolver))?,
            }
        }

        /// Returns true if the [`Tree`] is in-order and all values lie between the `min` and `max`.
        pub(crate) fn is_inorder_inner<TreeId, DataId, M: AtomMode>(
            &self,
            min: &Key,
            max: &Key,
            resolver: &impl AvlResolver<NodeId, DataId, TreeId, M>,
        ) -> Result<bool, OperationalError> {
            match self.root() {
                None => Ok(true),
                Some(node) => resolver
                    .resolve(node)
                    .map(|res| res.is_inorder(min, max, resolver))?,
            }
        }

        /// Creates an in-order iterator for the [`Node`]s in the [`Tree`].
        ///
        /// Each call to [`Iterator::next`] first descends as far left as possible from the current
        /// subtree, pushing the visited node ids onto an explicit stack. Once it reaches an empty
        /// subtree, it pops the next node from the stack, yields that node, and continues from that
        /// node's right subtree on the next call.
        ///
        /// The iterator yields an error if resolving any intermediate node or subtree fails.
        pub(crate) fn iter<
            'tree,
            'res,
            TreeId,
            DataId,
            M: octez_riscv_data::mode::Mode,
            Res: AvlResolver<NodeId, DataId, TreeId, M>,
        >(
            &'tree self,
            resolver: &'res Res,
        ) -> TreeIterator<'tree, 'res, NodeId, DataId, TreeId, M, Res> {
            TreeIterator {
                stack: vec![],
                current: self,
                resolver,
                _marker: std::marker::PhantomData,
            }
        }
    }

    /// Used for iterating through the nodes of the [`Tree`] in-order (left-to-right).
    ///
    /// `current` tracks the subtree that still needs to be explored, while `stack` stores the path of
    /// ancestor node ids whose left subtrees have already been explored. This lets the iterator do an
    /// in-order traversal without recursion.
    ///
    /// Resolution failures are surfaced as iterator items of type `Err`.
    pub(crate) struct TreeIterator<'tree, 'res, NodeId, DataId, TreeId, M, Resolver> {
        stack: Vec<&'tree NodeId>,
        current: &'tree Tree<NodeId>,
        resolver: &'res Resolver,
        #[expect(
            clippy::type_complexity,
            reason = "Moving into type would hide underlying use"
        )]
        _marker: PhantomData<fn() -> (TreeId, DataId, M)>,
    }

    impl<
        'tree,
        'res,
        NodeId,
        DataId: 'tree,
        TreeId: 'tree,
        M: octez_riscv_data::mode::Mode + 'tree,
        Resolver: AvlResolver<NodeId, DataId, TreeId, M>,
    > TreeIterator<'tree, 'res, NodeId, DataId, TreeId, M, Resolver>
    {
        /// Helper to descend to the leftmost node in the current subtree, pushing nodes onto the stack.
        fn advance_to_leftmost_in_subtree(
            &mut self,
            mut node_id: &'tree NodeId,
        ) -> Result<(), OperationalError> {
            loop {
                self.stack.push(node_id);
                let resolved_node = self.resolver.resolve(node_id)?;
                let left = resolved_node.left_ref(self.resolver)?;
                match left.root() {
                    Some(left_id) => node_id = left_id,
                    None => {
                        self.current = left;
                        break;
                    }
                }
            }
            Ok(())
        }

        /// Helper to pop the next node from the stack and prepare to traverse its right subtree.
        fn pop_and_prepare_right_subtree(
            &mut self,
        ) -> Result<Option<&'tree Node<TreeId, DataId, M>>, OperationalError> {
            let node_id = match self.stack.pop() {
                Some(id) => id,
                None => return Ok(None),
            };
            let resolved_node = self.resolver.resolve(node_id)?;
            let right = resolved_node.right_ref(self.resolver)?;
            self.current = right;
            Ok(Some(resolved_node))
        }
    }

    impl<
        'tree,
        'res,
        NodeId,
        DataId: 'tree,
        TreeId: 'tree,
        M: octez_riscv_data::mode::Mode + 'tree,
        Resolver: AvlResolver<NodeId, DataId, TreeId, M>,
    > Iterator for TreeIterator<'tree, 'res, NodeId, DataId, TreeId, M, Resolver>
    {
        type Item = Result<&'tree Node<TreeId, DataId, M>, OperationalError>;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(root_id) = self.current.root() {
                if let Err(err) = self.advance_to_leftmost_in_subtree(root_id) {
                    return Some(Err(err));
                }
            }

            match self.pop_and_prepare_right_subtree() {
                Ok(Some(node)) => Some(Ok(node)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        }
    }

    const GOLDEN_DIR: &str = "tests/expected";

    #[derive(Debug, Clone)]
    enum Operation {
        Get(Key),
        Upsert(Key, bytes::Bytes),
        Delete(Key),
    }

    fn key_strategy() -> impl Strategy<Value = Key> {
        proptest::collection::vec(any::<u8>(), 1usize..=KEY_MAX_SIZE)
            .prop_map(|bytes| Key::new(&bytes).expect("bytes are a valid key"))
    }

    fn value_strategy() -> impl Strategy<Value = bytes::Bytes> {
        proptest::collection::vec(any::<u8>(), 1usize..=200usize).prop_map(bytes::Bytes::from)
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

    fn compare_tree_to_reference(tree: &Tree<ArcNodeId>, reference: &BTreeMap<Key, bytes::Bytes>) {
        let resolver = ArcResolver;
        let tree_iter = tree.iter(&resolver);
        let mut reference_iter = reference.iter();
        for node in tree_iter {
            let node = node.expect("Tree iterator should yield nodes successfully");
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

    fn build_tree(keys: &[Key]) -> Result<Tree<ArcNodeId>, OperationalError> {
        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut resolver = ArcResolver;

        for key in keys {
            tree.set(key, key.as_ref(), &mut resolver)?;
        }

        Ok(tree)
    }

    fn iterated_keys(tree: &Tree<ArcNodeId>) -> Result<Vec<Key>, OperationalError> {
        let resolver = ArcResolver;
        tree.iter(&resolver)
            .map(|node| node.map(|node| node.key().clone()))
            .collect()
    }

    #[derive(Debug)]
    struct FailOnKeyResolver {
        target_failure_key: Key,
    }

    impl FailOnKeyResolver {
        fn new(target_failure_key: Key) -> Self {
            Self { target_failure_key }
        }
    }

    impl Resolver<ArcNodeId, Node<ArcTreeId, Bytes<Normal>, Normal>> for FailOnKeyResolver {
        fn resolve<'a>(
            &self,
            id: &'a ArcNodeId,
        ) -> Result<&'a Node<ArcTreeId, Bytes<Normal>, Normal>, OperationalError> {
            let node = ArcResolver.resolve(id)?;

            if node.key() == &self.target_failure_key {
                return Err(OperationalError::ResolverInvariantViolated);
            }

            Ok(node)
        }

        fn resolve_mut<'a>(
            &mut self,
            id: &'a mut ArcNodeId,
        ) -> Result<&'a mut Node<ArcTreeId, Bytes<Normal>, Normal>, OperationalError> {
            let node = ArcResolver.resolve_mut(id)?;

            if node.key() == &self.target_failure_key {
                return Err(OperationalError::ResolverInvariantViolated);
            }

            Ok(node)
        }
    }

    impl Resolver<ArcTreeId, Tree<ArcNodeId>> for FailOnKeyResolver {
        fn resolve<'a>(&self, id: &'a ArcTreeId) -> Result<&'a Tree<ArcNodeId>, OperationalError> {
            ArcResolver.resolve(id)
        }

        fn resolve_mut<'a>(
            &mut self,
            id: &'a mut ArcTreeId,
        ) -> Result<&'a mut Tree<ArcNodeId>, OperationalError> {
            ArcResolver.resolve_mut(id)
        }
    }

    impl DataResolver<Bytes<Normal>, Normal> for FailOnKeyResolver {
        fn resolve_bytes<'a>(
            &self,
            id: &'a Bytes<Normal>,
            _key: &Key,
        ) -> Result<&'a Bytes<Normal>, OperationalError> {
            // resolution would fail on the node by the key first
            Ok(id)
        }

        fn resolve_mut_bytes<'a>(
            &self,
            id: &'a mut Bytes<Normal>,
            _key: &Key,
        ) -> Result<&'a mut Bytes<Normal>, OperationalError> {
            // resolution would fail on the node by the key first
            Ok(id)
        }
    }

    fn assert_iterator_failure_on_key(
        tree: &Tree<ArcNodeId>,
        target_failure_key: &Key,
        expected_prefix: &[Key],
    ) -> Result<(), TestCaseError> {
        let resolver = FailOnKeyResolver::new(target_failure_key.clone());
        let mut iter = tree.iter(&resolver);
        let mut observed_prefix = Vec::new();

        loop {
            match iter.next() {
                Some(Ok(node)) => observed_prefix.push(node.key().clone()),
                Some(Err(OperationalError::ResolverInvariantViolated)) => break,
                Some(Err(err)) => {
                    return Err(TestCaseError::fail(format!(
                        "iterator returned an unexpected error: {err:?}"
                    )));
                }
                None => {
                    return Err(TestCaseError::fail(
                        "iterator unexpectedly completed without surfacing the resolver failure"
                            .to_owned(),
                    ));
                }
            }
        }

        prop_assert_eq!(observed_prefix, expected_prefix);

        // check that the iterator continues to surface the same failure on subsequent calls,
        // without yielding any more valid nodes.
        prop_assert!(matches!(
            iter.next(),
            Some(Err(OperationalError::ResolverInvariantViolated))
        ));

        Ok(())
    }

    fn four_distinct_keys_strategy() -> impl Strategy<Value = [Key; 4]> {
        proptest::collection::btree_set(any::<u32>(), 4).prop_map(|values| {
            values
                .into_iter()
                .map(|value| {
                    Key::new(&value.to_be_bytes())
                        .expect("u32 keys are always shorter than KEY_MAX_SIZE")
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("the strategy always produces exactly four distinct keys")
        })
    }

    #[test]
    fn test_get_error_surfacing_vs_missing_key() {
        let root = Key::new(&[2]).expect("The key should be valid.");
        let left = Key::new(&[1]).expect("The key should be valid.");
        let missing = Key::new(&[0]).expect("The key should be valid.");
        let no_failure_key = Key::new(&[255]).expect("The key should be valid.");

        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut setup_resolver = ArcResolver;
        tree.set(&root, b"root", &mut setup_resolver)
            .expect("Setting the root should succeed.");
        // The second insert descends from the existing root and populates the root's left subtree;
        // it does not replace the tree root.
        tree.set(&left, b"left", &mut setup_resolver)
            .expect("Setting the left child should succeed.");

        let ok_resolver = FailOnKeyResolver::new(no_failure_key);
        assert!(
            matches!(tree.get(&missing, &ok_resolver), Ok(None)),
            "Missing key lookup should be distinguishable as Ok(None)."
        );

        let failing_resolver = FailOnKeyResolver::new(left);
        assert!(
            matches!(
                tree.get(&missing, &failing_resolver),
                Err(OperationalError::ResolverInvariantViolated)
            ),
            "Resolver failures should be propagated as Error."
        );
    }

    proptest! {
        #[test]
        fn test_iterator_error_surfacing(
            ordered_keys in four_distinct_keys_strategy(),
            fail_on_left in any::<bool>(),
        ) {
            let [a, b, c, d] = ordered_keys;

            // Test a two-node and an effectively rebalanced three-node tree.
            let (two_node_keys, two_node_expected, three_node_keys, three_node_expected, target_failure_key, two_node_failure_prefix) = if fail_on_left {
                (
                    vec![c.clone(), b.clone()],
                    vec![b.clone(), c.clone()],
                    vec![c.clone(), b.clone(), a.clone()],
                    vec![a.clone(), b.clone(), c.clone()],
                    b.clone(),
                    vec![],
                )
            } else {
                (
                    vec![b.clone(), c.clone()],
                    vec![b.clone(), c.clone()],
                    vec![b.clone(), c.clone(), d.clone()],
                    vec![b.clone(), c.clone(), d.clone()],
                    c.clone(),
                    vec![b.clone()],
                )
            };

            let two_node_tree = build_tree(&two_node_keys)?;
            let three_node_tree = build_tree(&three_node_keys)?;
            let resolver = ArcResolver;

            prop_assert_eq!(iterated_keys(&two_node_tree)?, two_node_expected);
            prop_assert_eq!(
                iterated_keys(&three_node_tree)?,
                three_node_expected
            );

            two_node_tree.check(&resolver)?;
            three_node_tree.check(&resolver)?;

            assert_iterator_failure_on_key(&two_node_tree, &target_failure_key, &two_node_failure_prefix)?;
            // in the right-side failure case with three nodes, the failure key is the root after rebalancing,
            // so the iterator will fail immediately without yielding any nodes.
            assert_iterator_failure_on_key(&three_node_tree, &target_failure_key, &[])?;
        }

        #[test]
        fn avl_driver_test(operations in (1usize..500usize).prop_flat_map(operations_strategy)) {
            let mut tree: Tree<ArcNodeId> = Default::default();
            let mut reference: BTreeMap<Key, bytes::Bytes> = BTreeMap::new();
            let mut resolver = ArcResolver;
            for operation in operations {
                match operation {
                    Operation::Get(key) => {
                        let tree_value = match tree.get(&key, &resolver)? {
                            Some(node_id) => Some(resolver.resolve(node_id)?.data()),
                            None => None,
                        };

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
                let digest = tree.hash();
                tree.set(key, data.as_bytes(), &mut resolver)
                    .expect("Set should succeed");
                digest
            })
            .collect();

            digests.push(tree.hash());

            digests
        };

        let serialised = octez_riscv_data::serialisation::serialise(digests).unwrap();

        let mut mint = Mint::new(GOLDEN_DIR);
        let mut golden = mint.new_goldenfile("digests.out").unwrap();

        golden.write_all(&serialised).unwrap();
    }

    #[test]
    fn test_write_non_existent_key_non_zero_offset() {
        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut resolver = ArcResolver;

        let new = Key::new(&[0]).expect("Size less than KEY_MAX_SIZE");
        let result = tree.write(&new, 1, b"offset too large", &mut resolver);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument(InvalidArgumentError::OffsetTooLarge))
        ));
        assert!(
            tree.root().is_none(),
            "A failed write must not insert a node into an empty tree."
        );

        let existing = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        tree.set(&existing, b"zero offset", &mut resolver)
            .expect("Setting an existing key should succeed.");
        let new = Key::new(&[2]).expect("Size less than KEY_MAX_SIZE");
        let result = tree.write(&new, 1, b"nonzero offset", &mut resolver);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument(InvalidArgumentError::OffsetTooLarge))
        ));

        let node_id = tree
            .get(&existing, &resolver)
            .expect("Resolver failure not expected.")
            .expect("Pre-existing key should still be present.");
        let value = resolver
            .resolve(node_id)
            .expect("Resolver failure not expected.")
            .data();
        let mut buf = [0u8; 11];
        value.read(0, &mut buf);
        assert_eq!(&buf, b"zero offset");
        assert!(matches!(tree.get(&new, &resolver), Ok(None)));

        tree.write(&existing, 1, b"o", &mut resolver)
            .expect("Writing to an existing key with a non-zero offset should succeed.");
        let node_id = tree
            .get(&existing, &resolver)
            .expect("Resolver failure not expected.")
            .expect("Pre-existing key should still be present.");
        let value = resolver
            .resolve(node_id)
            .expect("Resolver failure not expected.")
            .data();
        let mut buf = [0u8; 11];
        value.read(0, &mut buf);
        assert_eq!(&buf, b"zoro offset");
    }
}
