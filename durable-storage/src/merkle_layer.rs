// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! A Merkleised key-value store layer.
//!
//! [`MerkleLayer`] wraps a [`KeyValueStore`] (KV) and duplicates all stored data in an AVL
//! [`Tree`]. When [`MerkleLayer::commit`] is called, the tree is serialised and stored in the KV
//! and the root hash of the tree is used to identify that commitment of the layer as a
//! [`CommitId`]. The inverse operation, [`MerkleLayer::checkout`], takes a [`CommitId`] and
//! restores the tree root as a blinded node that is loaded from the KV on demand.
//!
//! `M` is an implementation of the PVM's operational [`Mode`].
//!
//! [`MerkleLayer::try_clone_with`] enables forking snapshots. Clones share the underlying tree
//! cheaply via an `Arc` and diverge upon mutation, using copy-on-write (CoW) semantics.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::Arc;

use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold;
use octez_riscv_data::merkle_proof::proof_tree::MinimumPresence;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::serialise;
use perfect_derive::perfect_derive;

use crate::avl::resolver::LazyNodeId;
use crate::avl::resolver::LazyResolver;
use crate::avl::resolver::ProveNodeId;
use crate::avl::resolver::ProveResolver;
use crate::avl::resolver::Resolver;
use crate::avl::resolver::VerifyNodeId;
use crate::avl::resolver::VerifyResolver;
use crate::avl::tree::Tree;
use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::storage::KeyValueStore;
use crate::storage::Loadable;
use crate::storage::PersistentKeyValueStore;
use crate::storage::Storable;
use crate::storage::StoreOptions;

/// A layer for transforming data into a Merkle-ised representation before commitment to a
/// [`PersistentKeyValueStore`].
#[perfect_derive(Debug)]
pub struct MerkleLayer<KV, M: Mode> {
    inner: M::Select<MerkleLayerTemplate<KV>>,
}

impl<KV> MerkleLayer<KV, Normal> {
    /// Create a new, empty Merkle layer that will commit to the provided persistence layer.
    pub fn new(persistence: Arc<KV>) -> Self
    where
        KV: KeyValueStore,
    {
        MerkleLayer {
            inner: NormalImpl::new(persistence),
        }
    }

    /// Load the Merkle layer from the given key-value store.
    pub fn checkout(persistence: Arc<KV>, root: CommitId) -> Result<Self, Error>
    where
        KV: KeyValueStore,
    {
        Ok(MerkleLayer {
            inner: NormalImpl::checkout(persistence, root)?,
        })
    }

    /// Generates a commitment for the [MerkleLayer].
    pub fn commit(&mut self, options: &StoreOptions) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        self.inner.commit(options)
    }

    /// Snapshot the current tree and enter prove mode.
    ///
    /// The returned layer holds two trees: an immutable `initial_tree` (a cheap `Clone` of the
    /// Normal-mode tree) and a `working_tree` derived from it via [`Tree::into_proof`]. Mutations
    /// during the proof step run against the working tree; the initial tree drives the
    /// [`MerkleProofFold`] that generates the proof.
    ///
    /// [`MerkleProofFold`]: octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "non-test callers wired in follow-up PVM integration (RV-957)"
        )
    )]
    pub fn start_proof(&self) -> MerkleLayer<KV, Prove<'_>> {
        let initial_tree = self.inner.tree.clone();
        let working_tree = initial_tree.clone().into_proof();
        let resolver = ProveResolver::start(LazyResolver::new(self.inner.persistence.clone()));

        MerkleLayer {
            inner: ProveImpl {
                initial_tree,
                working_tree,
                resolver,
            },
        }
    }
}

impl<KV, M: MerkleLayerMode> MerkleLayer<KV, M> {
    /// Clone the Merkle layer. The new layer will commit to the provided persistence layer.
    pub fn try_clone_with(&self, persistence: Arc<KV>) -> Self
    where
        KV: KeyValueStore,
    {
        M::try_clone_with(self, persistence)
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub fn hash(&self) -> Hash {
        M::hash(self)
    }

    /// Delete the data associated with a given [Key].
    pub fn delete(&mut self, key: &Key) -> Result<(), OperationalError>
    where
        KV: KeyValueStore,
    {
        M::delete(self, key)
    }

    /// Sets the data associated with a given [Key].
    pub fn set(&mut self, key: &Key, data: &[u8]) -> Result<(), OperationalError>
    where
        KV: KeyValueStore,
    {
        M::set(self, key, data)
    }

    /// Writes the data to the node associated with a given [Key] with the given offset.
    pub fn write(&mut self, key: &Key, offset: usize, data: &[u8]) -> Result<(), Error>
    where
        KV: KeyValueStore,
    {
        M::write(self, key, offset, data)
    }
}

/// Modes that implements this trait support Merkle layer operations
pub trait MerkleLayerMode: Mode {
    /// See [`MerkleLayer::try_clone_with`]
    fn try_clone_with<KV: KeyValueStore>(
        this: &MerkleLayer<KV, Self>,
        persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self>;

    /// See [`MerkleLayer::hash`]
    fn hash<KV>(this: &MerkleLayer<KV, Self>) -> Hash;

    /// See [`MerkleLayer::delete`]
    fn delete<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
    ) -> Result<(), OperationalError>;

    /// See [`MerkleLayer::set`]
    fn set<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        data: &[u8],
    ) -> Result<(), OperationalError>;

    /// See [`MerkleLayer::write`]
    fn write<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        offset: usize,
        data: &[u8],
    ) -> Result<(), Error>;
}

impl MerkleLayerMode for Normal {
    fn try_clone_with<KV: KeyValueStore>(
        this: &MerkleLayer<KV, Self>,
        persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self> {
        MerkleLayer {
            inner: this.inner.try_clone_with(persistence),
        }
    }

    fn hash<KV>(this: &MerkleLayer<KV, Self>) -> Hash {
        this.inner.hash()
    }

    fn delete<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
    ) -> Result<(), OperationalError> {
        this.inner.delete(key)
    }

    fn set<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        data: &[u8],
    ) -> Result<(), OperationalError> {
        this.inner.set(key, data)
    }

    fn write<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        offset: usize,
        data: &[u8],
    ) -> Result<(), Error> {
        this.inner.write(key, offset, data)
    }
}

impl MerkleLayerMode for Prove<'_> {
    fn try_clone_with<KV: KeyValueStore>(
        _this: &MerkleLayer<KV, Self>,
        _persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self> {
        unimplemented!("Prove-mode MerkleLayer does not support cloning")
    }

    fn hash<KV>(this: &MerkleLayer<KV, Self>) -> Hash {
        this.inner.working_tree.hash()
    }

    fn delete<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
    ) -> Result<(), OperationalError> {
        this.inner
            .working_tree
            .delete(key, &mut this.inner.resolver)?;
        Ok(())
    }

    fn set<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        data: &[u8],
    ) -> Result<(), OperationalError> {
        this.inner
            .working_tree
            .set(key, data, &mut this.inner.resolver)?;
        Ok(())
    }

    fn write<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        offset: usize,
        data: &[u8],
    ) -> Result<(), Error> {
        this.inner
            .working_tree
            .write(key, offset, data, &mut this.inner.resolver)?;
        Ok(())
    }
}

impl MerkleLayerMode for Verify {
    fn try_clone_with<KV: KeyValueStore>(
        this: &MerkleLayer<KV, Self>,
        _persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self> {
        MerkleLayer {
            inner: VerifyImpl {
                tree: this.inner.tree.clone(),
                resolver: VerifyResolver,
            },
        }
    }

    fn hash<KV>(_this: &MerkleLayer<KV, Self>) -> Hash {
        unimplemented!("Blocked by RV-961")
    }

    fn delete<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
    ) -> Result<(), OperationalError> {
        this.inner.tree.delete(key, &mut this.inner.resolver)?;
        Ok(())
    }

    fn set<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        data: &[u8],
    ) -> Result<(), OperationalError> {
        this.inner.tree.set(key, data, &mut this.inner.resolver)?;
        Ok(())
    }

    fn write<KV: KeyValueStore>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        offset: usize,
        data: &[u8],
    ) -> Result<(), Error> {
        this.inner
            .tree
            .write(key, offset, data, &mut this.inner.resolver)?;
        Ok(())
    }
}

struct MerkleLayerTemplate<KV>(PhantomData<KV>, Infallible);

impl<KV> Modal for MerkleLayerTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = ProveImpl<KV>;

    type Verify = VerifyImpl;
}

#[derive(Debug)]
struct NormalImpl<KV> {
    tree: Tree<LazyNodeId>,
    persistence: Arc<KV>,
    resolver: LazyResolver<KV>,
}

impl<KV> NormalImpl<KV> {
    /// Create a new, empty Merkle layer that will commit to the provided persistence layer.
    fn new(persistence: Arc<KV>) -> Self {
        NormalImpl {
            tree: Tree::default(),
            persistence: persistence.clone(),
            resolver: LazyResolver::new(persistence),
        }
    }

    /// Clone the Merkle layer. The new layer will commit to the provided persistence layer.
    fn try_clone_with(&self, persistence: Arc<KV>) -> Self {
        Self {
            tree: self.tree.clone(),
            persistence: persistence.clone(),
            resolver: LazyResolver::new(persistence),
        }
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    fn hash(&self) -> Hash {
        Hash::from_foldable(&self.tree)
    }

    /// Delete the data associated with a given [Key].
    fn delete(&mut self, key: &Key) -> Result<(), OperationalError>
    where
        KV: KeyValueStore,
    {
        self.tree.delete(key, &mut self.resolver)?;
        Ok(())
    }

    /// Sets the data associated with a given [Key].
    fn set(&mut self, key: &Key, data: &[u8]) -> Result<(), OperationalError>
    where
        KV: KeyValueStore,
    {
        self.tree.set(key, data, &mut self.resolver)?;
        Ok(())
    }

    /// Writes the data to the node associated with a given [Key] with the given offset.
    fn write(&mut self, key: &Key, offset: usize, data: &[u8]) -> Result<(), Error>
    where
        KV: KeyValueStore,
    {
        self.tree.write(key, offset, data, &mut self.resolver)?;
        Ok(())
    }

    /// Load the Merkle layer from the given key-value store with lazy node loading.
    fn checkout(persistence: Arc<KV>, root: CommitId) -> Result<Self, Error>
    where
        KV: KeyValueStore,
    {
        let resolver = LazyResolver::new(persistence.clone());
        let tree = Tree::load(*root.as_hash(), persistence.as_ref())?;

        Ok(Self {
            tree,
            persistence,
            resolver,
        })
    }

    /// Generates a commitment for the [MerkleLayer].
    fn commit(&mut self, options: &StoreOptions) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        self.tree.store(self.persistence.as_ref(), options)?;
        Ok(CommitId::from(self.hash()))
    }
}

#[derive(Debug)]
struct VerifyImpl {
    tree: Tree<VerifyNodeId>,
    resolver: VerifyResolver,
}

/// Prove-mode backing state for a [`MerkleLayer`].
///
/// A proof is generated against the **initial** state that the step read from, not against the
/// post-step state: AVL rotations on `set`/`delete`/`write` rewrite subtree structure, so folding
/// the working tree would produce a proof whose shape does not match the initial root. Prove mode
/// therefore holds both trees:
///
/// * `initial_tree` — an immutable snapshot of the Normal-mode tree, captured at `start_proof`
///   time. This is what the [`MerkleProofFold`] implementation on [`MerkleLayer`] walks.
/// * `working_tree` — a Prove-mode projection that the step mutates. Its root hash is the
///   final-state hash. Its per-node [`Cell<bool>`] read flags are the source of truth for
///   deciding which fields of an initial node were actually read.
/// * `resolver` — a [`ProveResolver`] wrapping a [`LazyResolver`]. Its access set tells the fold
///   which initial-tree nodes can be blinded, and its `deleted_nodes` map preserves the
///   prove-mode projection of any node that was unlinked from the working tree during the step.
///
/// The `'normal` lifetime carries the borrow from the originating Normal-mode state, matching
/// the lifetime on [`Prove`].
///
/// [`MerkleProofFold`]: octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold
/// [`Cell<bool>`]: std::cell::Cell
#[derive(Debug)]
struct ProveImpl<KV> {
    initial_tree: Tree<LazyNodeId>,
    working_tree: Tree<ProveNodeId>,
    resolver: ProveResolver<LazyResolver<KV>>,
}

// ---------------------------------------------------------------------------
// Fold implementations for prove-mode MerkleLayer
// ---------------------------------------------------------------------------

impl<KV: KeyValueStore> Foldable<HashFold> for MerkleLayer<KV, Prove<'_>> {
    fn fold(&self, _builder: HashFold) -> <HashFold as Fold>::Folded {
        self.inner.working_tree.hash()
    }
}

impl<KV: KeyValueStore> Foldable<MerkleProofFold> for MerkleLayer<KV, Prove<'_>> {
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        // The root tree is always the entry point — it's never a `ProveTreeId` child, so it
        // won't appear in `accessed_items.trees`. Fold it unconditionally; child subtrees
        // are checked against the access set inside `InitialTreeFold`.
        let wrapper = InitialTreeFold {
            tree: &self.inner.initial_tree,
            prove_impl: &self.inner,
            is_root: true,
        };
        wrapper.fold(builder)
    }
}

/// Wrapper that folds an initial-tree [`Tree<LazyNodeId>`] into a [`MerkleProofFold`].
///
/// Drives the tree-level structure (occupied-bool + optional node) off the initial tree, then
/// delegates node-level folding to [`InitialNodeFold`].
struct InitialTreeFold<'a, KV> {
    tree: &'a Tree<LazyNodeId>,
    prove_impl: &'a ProveImpl<KV>,
    /// The root tree is always folded; child subtrees are checked against the access set.
    is_root: bool,
}

impl<KV: KeyValueStore> Foldable<MerkleProofFold> for InitialTreeFold<'_, KV> {
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        let tree_hash = self.tree.hash();

        // The root tree is always folded. Child subtrees are checked against the access set.
        if !self.is_root && !self.prove_impl.resolver.was_tree_accessed(&tree_hash) {
            return builder.into_blind(tree_hash);
        }

        let mut node_fold = builder.into_node_fold();

        // Bool leaf: true if the initial tree is occupied.
        let present = self.tree.root().is_some();
        let bool_data = serialise(present).expect("Serialising a bool should not fail");
        let bool_leaf = MerkleProofFold::new_leaf(MinimumPresence::Present, bool_data);
        node_fold.add(&bool_leaf);

        if let Some(lazy_node_id) = self.tree.root() {
            let child = InitialNodeFold {
                node_id: lazy_node_id,
                prove_impl: self.prove_impl,
            };
            node_fold.add(&child);
        }

        node_fold.done()
    }
}

/// Wrapper that folds an initial-node [`LazyNodeId`] into a [`MerkleProofFold`].
///
/// If the node was not accessed during the step, it is blinded.  Otherwise, per-field presence
/// (meta, data) is taken from the prove-mode `Node`'s read flags, and the children are folded
/// recursively off the **initial tree**'s structure (not the working tree's).
struct InitialNodeFold<'a, KV> {
    node_id: &'a LazyNodeId,
    prove_impl: &'a ProveImpl<KV>,
}

impl<KV: KeyValueStore> Foldable<MerkleProofFold> for InitialNodeFold<'_, KV> {
    fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
        let hash = Hash::from_foldable(self.node_id);

        if !self.prove_impl.resolver.was_node_accessed(&hash) {
            return builder.into_blind(hash);
        }

        // Resolve the initial node to get its key and children.
        let initial_node = self
            .prove_impl
            .resolver
            .inner()
            .resolve(self.node_id)
            .expect("Accessed node should be cached in LazyResolver");

        // Find the prove-mode projection carrying the read flags.
        //
        // 1. Check `deleted_nodes` first (covers delete and delete-reinsert-same-key).
        //    For delete-then-reinsert-same-key: the deleted node's flags are correct because
        //    this fold walks the *initial* tree — the initial node was read/deleted during the
        //    step, so its flags reflect that access. The newly inserted node at the same key
        //    is a different node in the working tree and is irrelevant to the initial fold.
        // 2. Otherwise search the working tree by key.
        let deleted = self.prove_impl.resolver.deleted_node(&hash);
        let prove_node = if let Some(deleted_id) = &deleted {
            deleted_id
                .cached_node()
                .expect("Deleted ProveNodeId should have a cached prove-mode node")
        } else {
            let key = initial_node.key();
            self.prove_impl
                .working_tree
                .find_node(key, &self.prove_impl.resolver)
                .expect("Working-tree lookup should not fail for an accessed node")
                .expect("Accessed node should still be present in the working tree")
                .cached_node()
                .expect("Working-tree ProveNodeId should have a cached prove-mode node")
        };

        // Fold meta + data from the prove-mode node (they carry per-field read flags).
        let mut node_fold = builder.into_node_fold();
        node_fold.add(prove_node.meta_atom());
        node_fold.add(prove_node.data_bytes());

        // Fold left + right from the **initial** tree's children.
        let left = InitialTreeFold {
            tree: self
                .prove_impl
                .resolver
                .inner()
                .resolve(initial_node.left_id())
                .expect("Accessed subtree should be cached in LazyResolver"),
            prove_impl: self.prove_impl,
            is_root: false,
        };
        node_fold.add(&left);

        let right = InitialTreeFold {
            tree: self
                .prove_impl
                .resolver
                .inner()
                .resolve(initial_node.right_id())
                .expect("Accessed subtree should be cached in LazyResolver"),
            prove_impl: self.prove_impl,
            is_root: false,
        };
        node_fold.add(&right);

        node_fold.done()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use octez_riscv_data::components::bytes::Bytes;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::mode::utils::catch_not_found;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::MerkleLayer;
    use super::ProveImpl;
    use crate::avl::node::Node;
    use crate::avl::resolver::ArcTreeId;
    use crate::avl::resolver::LazyNodeId;
    use crate::avl::resolver::LazyResolver;
    use crate::avl::resolver::LazyTreeId;
    use crate::avl::resolver::ProveResolver;
    use crate::avl::resolver::VerifyNodeId;
    use crate::avl::resolver::VerifyResolver;
    use crate::avl::resolver::VerifyTreeId;
    use crate::avl::tree::Tree;
    use crate::errors::OperationalError;
    use crate::key::Key;
    use crate::merkle_layer::VerifyImpl;
    use crate::storage::setup_repo;
    use crate::storage::KeyValueStore;
    use crate::storage::TestKeyValueStore;
    use crate::storage::TestRepo;

    impl<KV: KeyValueStore> MerkleLayer<KV, Normal> {
        fn tree(&self) -> &Tree<LazyNodeId> {
            &self.inner.tree
        }

        /// Clear all data from the [MerkleLayer].
        fn clear(&mut self) {
            self.inner.tree.take();
        }

        /// Returns an immutable reference to the data stored for a given [Key].
        pub fn get(&mut self, key: &Key) -> Result<Option<&Bytes<Normal>>, OperationalError> {
            self.inner.tree.get(key, &self.inner.resolver)
        }
    }

    impl<KV: KeyValueStore> MerkleLayer<KV, Prove<'_>> {
        fn get(&self, key: &Key) -> Result<Option<&Bytes<Prove<'static>>>, OperationalError> {
            self.inner.working_tree.get(key, &self.inner.resolver)
        }
    }

    impl<KV> MerkleLayer<KV, Verify> {
        fn get(&self, key: &Key) -> Result<Option<&Bytes<Verify>>, OperationalError> {
            self.inner.tree.get(key, &self.inner.resolver)
        }

        /// Construct a Verify-mode MerkleLayer from a deserialised tree.
        pub(crate) fn from_verify_tree(tree: Tree<VerifyNodeId>) -> Self {
            MerkleLayer {
                inner: VerifyImpl {
                    tree,
                    resolver: VerifyResolver,
                },
            }
        }
    }

    fn new_merkle_layer(repo: TestRepo) -> MerkleLayer<TestKeyValueStore, Normal> {
        let persistence_layer = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();

        MerkleLayer::new(persistence_layer)
    }

    #[test]
    fn test_mavl_cow() {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = [vec![0; 0], vec![13; 5], vec![42; 129]];

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);

        for i in 0..keys.len() {
            ml.set(&keys[i], &data[i])
                .expect("setting node should succeed");
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
        }

        let mut ml2 = ml.try_clone_with(ml.inner.persistence.clone());
        let original_hash = ml.hash();
        assert_eq!(original_hash, ml2.hash());

        let cow_data = "🐮<(moo!)";
        ml2.set(&keys[0], cow_data.as_bytes())
            .expect("setting node should succeed");
        assert_ne!(original_hash, ml2.hash());
        assert_eq!(original_hash, ml.hash());

        let old_node1: Node<LazyTreeId, Normal> =
            Node::new(keys[0].clone(), bytes::Bytes::copy_from_slice(&data[0]));
        let new_node1: Node<LazyTreeId, Normal> = Node::new(keys[0].clone(), cow_data.as_bytes());

        let node2: Node<LazyTreeId, Normal> =
            Node::new(keys[1].clone(), bytes::Bytes::copy_from_slice(&data[1]));
        let node3: Node<LazyTreeId, Normal> =
            Node::new(keys[2].clone(), bytes::Bytes::copy_from_slice(&data[2]));

        assert_eq!(
            &old_node1.data(),
            &ml.get(&keys[0])
                .expect("The node should be retrieved successfully. Merkle layer: {ml:?}")
                .expect("The data should exist.")
        );
        assert_eq!(
            &new_node1.data(),
            &ml2.get(&keys[0])
                .expect("The node should be retrieved successfully. Merkle layer: {ml2:?}")
                .expect("The data should exist.")
        );

        assert_eq!(
            &node2.data(),
            &ml.get(&keys[1])
                .expect("The node should be retrieved successfully. Merkle layer: {ml:?}")
                .expect("The data should exist.")
        );
        assert_eq!(
            &node2.data(),
            &ml2.get(&keys[1])
                .expect("The node should be retrieved successfully. Merkle layer: {ml2:?}")
                .expect("The data should exist.")
        );

        assert_eq!(
            &node3.data(),
            &ml.get(&keys[2])
                .expect("The node should be retrieved successfully. Merkle layer: {ml:?}")
                .expect("The data should exist.")
        );
        assert_eq!(
            &node3.data(),
            &ml2.get(&keys[2])
                .expect("The node should be retrieved successfully. Merkle layer: {ml2:?}")
                .expect("The data should exist.")
        );

        ml.tree()
            .check(&ml.inner.resolver)
            .expect("the tree should be retrieved successfully.");
        ml2.tree()
            .check(&ml.inner.resolver)
            .expect("the tree should be retrieved successfully.");
    }

    proptest! {
        #[test]
        fn test_mavl_cow_prop(keys1 in prop::collection::vec(any::<[u8; 2]>(), 0..500), keys2 in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data1 = bytes::Bytes::from("property");
            let data2 = bytes::Bytes::from("cow");

            let (_keepalive, repo) = setup_repo();
            let mut ml = new_merkle_layer(repo);

            // Set all the keys in the tree
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data1).expect("setting node should succeed");
            }

            // Create a cheap copy
            let original_hash = ml.hash();
            let mut ml2 = ml.try_clone_with(ml.inner.persistence.clone());
            prop_assert_eq!(original_hash, ml2.hash());

            // Delete all the keys in the copy
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml2.delete(&key).expect("deleting node should succeed.");
                prop_assert_eq!(ml2.get(&key).expect(""), None);
            }

            // Set new keys in the copy
            for bytes in &keys2 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml2.set(&key, &data2).expect("setting node should succeed");
            }

            if keys1.is_empty() && keys2.is_empty() {
                prop_assert_eq!(original_hash, ml2.hash());
            } else {
                prop_assert_ne!(original_hash, ml2.hash());
            }
            prop_assert_eq!(original_hash, ml.hash());

            // Check both trees are still correct
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully").expect("The data should exist."), &data1);
            }
            for bytes in &keys2 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml2.get(&key).expect("The node should be retrieved successfully").expect("The data should exist."), &data2);
            }

            ml.tree().check(&ml.inner.resolver).expect("the tree should be retrieved successfully.");
            ml2.tree().check(&ml2.inner.resolver).expect("the tree should be retrieved successfully.");
        }
    }

    #[test]
    fn test_mavl_create() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("create");
        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        let empty_hash = ml.hash();
        ml.set(&key, &data).expect("setting node should succeed");
        assert_ne!(empty_hash, ml.hash());

        let node: Node<LazyTreeId, Normal> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());
        ml.tree()
            .check(&ml.inner.resolver)
            .expect("the tree should be retrieved successfully.");
    }

    #[test]
    fn test_mavl_create_existing() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("old");
        let data2 = bytes::Bytes::from("new");
        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        ml.set(&key, &data).expect("setting node should succeed");
        let old_hash = ml.hash();

        let node: Node<LazyTreeId, Normal> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());

        ml.set(&key, &data2).expect("setting node should succeed");
        assert_ne!(old_hash, ml.hash());
        assert!(
            ml.tree()
                .is_inorder(&ml.inner.resolver)
                .expect("The tree should be retrieved successfully."),
            "AVL isn't in order: {ml:?}"
        );
        let node: Node<LazyTreeId, Normal> = Node::new(key.clone(), data2);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());
        ml.tree()
            .check(&ml.inner.resolver)
            .expect("the tree should be retrieved successfully.");
    }

    #[test]
    fn test_mavl_create_heterogenous_key() {
        let keys = [
            Key::new(&[255, 0]),
            Key::new(&[0]),
            Key::new(&[0, 0]),
            Key::new(&[0, 0, 0]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = [
            bytes::Bytes::from("255, 0"),
            bytes::Bytes::from("0"),
            bytes::Bytes::from("0, 0"),
            bytes::Bytes::from("0, 0, 0"),
        ];

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);

        for (key, data) in keys.iter().zip(data.iter()) {
            let old_hash = ml.hash();
            ml.set(key, data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully")
                    .expect("The data should exist."),
                data
            );
        }
    }

    #[test]
    fn test_mavl_create_imbalanced() {
        let keys = [
            Key::new(&[6]),
            Key::new(&[5]),
            Key::new(&[4]),
            Key::new(&[3]),
            Key::new(&[2]),
            Key::new(&[1]),
            Key::new(&[0]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        let empty_hash = ml.hash();

        // Left imbalance
        let data = bytes::Bytes::from("imbalanced left");
        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
        }

        ml.clear();
        assert_eq!(empty_hash, ml.hash());

        let keys = {
            let mut keys = keys;
            keys.sort();
            keys
        };

        // Right imbalance
        let data = bytes::Bytes::from("imbalanced right");
        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
        }
    }

    #[test]
    fn test_mavl_create_left_right() {
        let keys = [Key::new(&[2]), Key::new(&[0]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = bytes::Bytes::from("left_right");

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);

        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully")
                    .expect("The data should exist."),
                &data
            );
        }
    }

    #[test]
    fn test_mavl_create_right_left() {
        let keys = [Key::new(&[0]), Key::new(&[2]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = bytes::Bytes::from("right_left");

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);

        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully")
                    .expect("The data should exist."),
                &data
            );
        }
    }

    #[test]
    fn test_mavl_create_right_left_nonzero_node_bf() {
        let keys = [
            Key::new(&[0]),
            Key::new(&[9]),
            Key::new(&[6]),
            Key::new(&[8]),
            Key::new(&[7]),
            Key::new(&[1]),
            Key::new(&[5]),
            Key::new(&[4]),
            Key::new(&[2]),
            Key::new(&[3]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = bytes::Bytes::from("right_left");

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);

        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully")
                    .expect("The data should exist."),
                &data
            );
        }
    }

    #[test]
    fn test_mavl_create_left_right_nonzero_node_bf() {
        let keys = [
            Key::new(&[10]),
            Key::new(&[7]),
            Key::new(&[9]),
            Key::new(&[8]),
            Key::new(&[0]),
            Key::new(&[6]),
            Key::new(&[5]),
            Key::new(&[4]),
            Key::new(&[1]),
            Key::new(&[2]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = bytes::Bytes::from("right_left");

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);

        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash());
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully")
                    .expect("The data should exist."),
                &data
            );
        }
    }

    proptest! {
        #[test]
        fn test_mavl_create_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = bytes::Bytes::from("property");
            let (_keepalive, repo) = setup_repo();
            let mut ml = new_merkle_layer(repo);
            let old_hash = ml.hash();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data).expect("setting node should succeed");
            }

            if !keys.is_empty() {
                assert_ne!(old_hash, ml.hash());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully").expect("The data should exist."), &data);
            }

            ml.tree().check(&ml.inner.resolver).expect("the tree should be retrieved successfully.");
        }
    }

    #[test]
    fn test_mavl_delete() {
        let key = Key::new(&[1]).expect("Sizes less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("delete");
        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        let empty_hash = ml.hash();
        ml.set(&key, &data).expect("setting node should succeed");
        let full_hash = ml.hash();

        ml.delete(&key).expect("deleting node should succeed.");
        assert_ne!(full_hash, ml.hash());
        assert_eq!(empty_hash, ml.hash());

        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully.");

        assert_eq!(get_node, None);
        ml.tree()
            .check(&ml.inner.resolver)
            .expect("the tree should be retrieved successfully.");
    }

    proptest! {
        #[test]
        fn test_mavl_delete_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = bytes::Bytes::from("delete_prop");
            let (_keepalive, repo) = setup_repo();
            let mut ml = new_merkle_layer(repo);
            let empty_hash = ml.hash();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data).expect("setting node should succeed");
            }

            if !keys.is_empty() {
                prop_assert_ne!(empty_hash, ml.hash());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.delete(&key).expect("delete should succeed.");
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully."), None);
            }

            prop_assert_eq!(empty_hash, ml.hash());

            ml.tree().check(&ml.inner.resolver).expect("the tree should be retrieved successfully.");
        }
    }

    fn test_mavl_delete_keys(keys: &[Key]) {
        let data = bytes::Bytes::from("delete");

        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        let empty_hash = ml.hash();

        for key in keys.iter() {
            ml.set(key, &data).expect("setting node should succeed");
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully")
                    .expect("The data should exist."),
                &data,
            );
        }

        if !keys.is_empty() {
            assert_ne!(empty_hash, ml.hash());
        }

        for key in keys.iter() {
            ml.delete(key).expect("deleting node should succeed.");
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            ml.delete(key).expect("deleting node should succeed.");
            assert_eq!(ml.get(key).expect("The data should exist."), None);
        }

        assert_eq!(empty_hash, ml.hash());
    }

    // Requires replacing a node with its successor while rebalancing a node on the return path.
    //
    //      BEFORE        AFTER
    //         2x           3
    //       /   \        /   \
    //      0     4      1     5
    //       \   / \         /  \
    //        1 3   5       4    6
    //               \
    //                6
    #[test]
    fn test_mavl_delete_rebalance_needed() {
        let keys = [
            Key::new(&[2]),
            Key::new(&[0]),
            Key::new(&[3]),
            Key::new(&[4]),
            Key::new(&[5]),
            Key::new(&[1]),
            Key::new(&[6]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        test_mavl_delete_keys(&keys);
    }

    // Requires replacing a deleted node with a successor that is its right child.
    //
    //      BEFORE       AFTER
    //        1x           2
    //       / \          / \
    //      0   2        0   3
    //           \
    //            3
    #[test]
    fn test_mavl_delete_right_successor() {
        let keys = [
            Key::new(&[1]),
            Key::new(&[2]),
            Key::new(&[0]),
            Key::new(&[3]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        test_mavl_delete_keys(&keys);
    }

    // Requires replacing a deleted node with a successor that is its right child and has a right
    // child of its own.
    //
    //      BEFORE       AFTER
    //        4x           5
    //       / \          / \
    //      1   5        1   6
    //     /     \      /
    //    0       6    0
    #[test]
    fn test_mavl_delete_successor_right_child() {
        let keys = [
            Key::new(&[4]),
            Key::new(&[5]),
            Key::new(&[1]),
            Key::new(&[6]),
            Key::new(&[0]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        test_mavl_delete_keys(&keys);
    }

    // Requires replacing a deleted node with a successor that isn't its right child.
    //
    //      BEFORE    AFTER
    //        1x        2
    //       / \       / \
    //      0   3     0   3
    //         /
    //        2
    #[test]
    fn test_mavl_delete_take_min() {
        let keys = [
            Key::new(&[1]),
            Key::new(&[3]),
            Key::new(&[0]),
            Key::new(&[2]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        test_mavl_delete_keys(&keys);
    }

    // Requires replacing a deleted node with a successor that isn't its right child and isn't the
    // right child's left child.
    //
    //      BEFORE         AFTER
    //         2x            3
    //       /    \       /    \
    //      0      5     0      5
    //       \    / \     \    / \
    //        1  4   6     1  4   6
    //          /
    //         3
    #[test]
    fn test_mavl_delete_take_min_recursive() {
        let keys = [
            Key::new(&[2]),
            Key::new(&[4]),
            Key::new(&[0]),
            Key::new(&[6]),
            Key::new(&[5]),
            Key::new(&[1]),
            Key::new(&[3]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        test_mavl_delete_keys(&keys);
    }

    // Requires rebalancing a node where the balance factor is -2 and the left child's balance
    // factor is 0:
    //      BEFORE      DELETED     ROTATED
    //        4x           5            1
    //       / \          /           /  \
    //      1   5        1           3    5
    //     / \          / \         /
    //    0   3        0   3       0
    #[test]
    fn test_mavl_delete_zero_double_rotation_balance_factor() {
        let keys = [
            Key::new(&[4]),
            Key::new(&[0]),
            Key::new(&[5]),
            Key::new(&[1]),
            Key::new(&[3]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        test_mavl_delete_keys(&keys);
    }

    #[test]
    fn test_mavl_write_new_value() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("write_new_value");
        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        let old_hash = ml.hash();
        ml.write(&key, 0, &data).expect("write should succeed.");

        let node: Node<ArcTreeId, Normal> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());
        assert_ne!(old_hash, ml.hash());
        ml.tree()
            .check(&ml.inner.resolver)
            .expect("The tree should be retrieved successfully.");
    }

    #[test]
    fn test_mavl_write_no_truncation() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("a long value");
        let data2 = bytes::Bytes::from("good");
        let (_keepalive, repo) = setup_repo();
        let mut ml = new_merkle_layer(repo);
        ml.set(&key, &data).expect("setting node should succeed");
        let old_hash = ml.hash();

        let data_len = data.len();
        let node: Node<LazyTreeId, Normal> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());

        ml.write(&key, 2, &data2).expect("write should succeed.");
        assert_ne!(old_hash, ml.hash());
        assert!(
            ml.inner
                .tree
                .is_inorder(&ml.inner.resolver)
                .expect("The tree should be retrieved successfully."),
            "AVL isn't in order: {ml:?}"
        );
        let node: Node<LazyTreeId, Normal> =
            Node::new(key.clone(), bytes::Bytes::from("a good value"));
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(get_node.len(), data_len);
        assert_eq!(&get_node, &node.data());
        ml.tree()
            .check(&ml.inner.resolver)
            .expect("the tree should be retrieved successfully.");
    }

    proptest! {
        #[test]
        fn test_mavl_write_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..10)) {
            let data = bytes::Bytes::from(vec![0; 500]);
            let alternating = bytes::Bytes::from([1, 0]
                .iter()
                .cycle()
                .take(500)
                .cloned()
                .collect::<Vec<_>>());

            let (_keepalive, repo) = setup_repo();
            let mut ml = new_merkle_layer(repo);
            let old_hash = ml.hash();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data).expect("setting node should succeed");
                for offset in 0..250 {
                    ml.write(&key, offset * 2, &[1]).expect("write should succeed.");
                }
            }

            if !keys.is_empty() {
                assert_ne!(old_hash, ml.hash());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key)
                                .expect("The node should be retrieved successfully")
                                .expect("The data should exist."),
                                &alternating);
            }

            ml.tree().check(&ml.inner.resolver).expect("the tree should be retrieved successfully.");
        }
    }

    #[cfg(feature = "rocksdb")]
    #[derive(Debug, Clone)]
    enum GeneratedOperation {
        // Key, Value
        Set([u8; 2], Vec<u8>),
        // Key, Value, offset hint (used to generate a valid offset for existing values)
        Write([u8; 2], Vec<u8>, u8),
        // Key
        Delete([u8; 2]),
    }

    #[cfg(feature = "rocksdb")]
    fn generated_operations_strategy(
        length: usize,
    ) -> impl Strategy<Value = Vec<GeneratedOperation>> {
        let count = length.div_ceil(10);

        (
            prop::collection::vec(any::<[u8; 2]>(), count),
            prop::collection::vec(prop::collection::vec(any::<u8>(), 0..64), count),
        )
            .prop_flat_map(move |(keys, values)| {
                prop::collection::vec(
                    prop_oneof![
                        (
                            proptest::sample::select(keys.clone()),
                            proptest::sample::select(values.clone()),
                        )
                            .prop_map(|(key, value)| GeneratedOperation::Set(key, value)),
                        (
                            proptest::sample::select(keys.clone()),
                            proptest::sample::select(values),
                            any::<u8>(),
                        )
                            .prop_map(|(key, value, offset_hint)| {
                                GeneratedOperation::Write(key, value, offset_hint)
                            }),
                        proptest::sample::select(keys).prop_map(GeneratedOperation::Delete),
                    ],
                    length,
                )
            })
    }

    #[cfg(feature = "rocksdb")]
    proptest! {
        #[test]
        fn test_merkle_layer_checkout_lazy_from_commit(
            operations in (1usize..100usize).prop_flat_map(generated_operations_strategy)
        ) {
            use std::collections::BTreeMap;
            use std::collections::BTreeSet;
            use octez_riscv_test_utils::TestableTmpdir;
            use crate::repo::DirectoryManager;


            let tmpdir = TestableTmpdir::new();
            let repo = DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");
            let persistence = TestKeyValueStore::new(&repo)
                .expect("Creating a persistence layer should succeed");
            let persistence = std::sync::Arc::new(persistence);

            let mut merkle_layer = MerkleLayer::new(persistence.clone());
            let mut reference: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
            let mut touched: BTreeSet<Key> = BTreeSet::new();

            for operation in operations {
                match operation {
                    GeneratedOperation::Set(bytes, value) => {
                        let key = Key::new(&bytes).expect("Size less than KEY_MAX_SIZE");
                        merkle_layer
                            .set(&key, &value)
                            .expect("Set operation should succeed");
                        reference.insert(key.clone(), value);
                        touched.insert(key);
                    }
                    GeneratedOperation::Write(bytes, value, offset_hint) => {
                        let key = Key::new(&bytes).expect("Size less than KEY_MAX_SIZE");
                        let offset = match reference.get(&key) {
                            Some(existing) if !existing.is_empty() => {
                                1 + (offset_hint as usize % existing.len())
                            }
                            _ => 0,
                        };

                        merkle_layer
                            .write(&key, offset, &value)
                            .expect("Write operation should succeed");

                        let entry = reference.entry(key.clone()).or_default();
                        let new_len = offset
                            .checked_add(value.len())
                            .expect("Generated offset and value length should not overflow");
                        if entry.len() < new_len {
                            entry.resize(new_len, 0);
                        }
                        entry[offset..new_len].copy_from_slice(&value);
                        touched.insert(key);
                    }
                    GeneratedOperation::Delete(bytes) => {
                        let key = Key::new(&bytes).expect("Sizes less than KEY_MAX_SIZE");
                        merkle_layer
                            .delete(&key)
                            .expect("Delete operation should succeed");
                        reference.remove(&key);
                        touched.insert(key);
                    }
                }
            }

            let expected_hash = merkle_layer.hash();
            let commit_opts = crate::storage::StoreOptions::default().with_deep().with_node_data();
            let commit_id = merkle_layer.commit(&commit_opts).expect("Commit operation should succeed");

            let mut lazy_loaded = MerkleLayer::checkout(persistence, commit_id)
                .expect("Lazy checkout should succeed");
            let loaded_hash = lazy_loaded.hash();

            prop_assert_eq!(loaded_hash, expected_hash);

            for key in touched {
                let expected = reference.get(&key);
                let loaded = lazy_loaded
                    .get(&key)
                    .expect("Lookup in lazy-loaded tree should succeed");

                match (expected, loaded) {
                    (Some(expected), Some(loaded)) => {
                        let mut loaded_bytes = vec![0; loaded.len()];
                        let bytes_read = loaded.read(0, &mut loaded_bytes);
                        prop_assert_eq!(bytes_read, loaded_bytes.len());
                        prop_assert_eq!(loaded_bytes.as_slice(), expected.as_slice());
                    }
                    (None, None) => {}
                    (Some(_), None) => panic!("Expected an existing key in lazy-loaded tree: {key:?}"),
                    (None, Some(_)) => panic!("Expected a missing key in lazy-loaded tree: {key:?}"),
                }
            }
        }
    }

    /// - Add some data to the Merkle layer.
    /// - Commit the data to relevant column family
    /// - Check whether the data is persisted.
    /// - Check whether the hash contained in the commit id
    ///   is the same as the root hash
    #[cfg(feature = "rocksdb")]
    #[test]
    fn test_merkle_layer_commit_persists_nodes() {
        use crate::storage::Loadable;
        use crate::storage::Storable;
        use crate::storage::StoreOptions;

        let (_keepalive, repo) = setup_repo();
        let mut merkle_layer = new_merkle_layer(repo);

        let keys = [
            Key::new(&[12]).unwrap(),
            Key::new(&[1]).unwrap(),
            Key::new(&[72]).unwrap(),
            Key::new(&[3]).unwrap(),
            Key::new(&[4]).unwrap(),
            Key::new(&[17]).unwrap(),
            Key::new(&[8]).unwrap(),
        ];

        let data = [
            bytes::Bytes::from_static(b"aasd"),
            bytes::Bytes::from_static(b"aksdja"),
            bytes::Bytes::from_static(b"agfgd"),
            bytes::Bytes::from_static(b"45gfgdf"),
            bytes::Bytes::from_static(b"sfdsdfsd"),
            bytes::Bytes::from_static(b"asdfsfd"),
            bytes::Bytes::from_static(b"asdfsdf"),
        ];

        for (key, data_elem) in keys.iter().zip(data.iter()) {
            merkle_layer
                .set(key, data_elem)
                .expect("setting node should succeed");
        }

        let commit_opts = StoreOptions::default().with_deep().with_node_data();
        let commit_id = merkle_layer
            .commit(&commit_opts)
            .expect("The commit operation should not fail");

        for node in merkle_layer.inner.tree.iter(&merkle_layer.inner.resolver) {
            let node: &Node<LazyTreeId, Normal> =
                node.expect("The node should be retrieved successfully");

            node.store(
                merkle_layer.inner.persistence.as_ref(),
                &StoreOptions::default().with_shallow().with_node_data(),
            )
            .expect("Storing node should succeed");

            let loaded_node: Node<LazyTreeId, Normal> =
                Node::load(*node.hash(), merkle_layer.inner.persistence.as_ref())
                    .expect("Loading node should succeed");

            assert_eq!(node.hash(), loaded_node.hash());
            assert_eq!(node.balance_factor(), loaded_node.balance_factor());
            assert_eq!(node.key(), loaded_node.key());
            assert_eq!(node.value(), loaded_node.value());
        }

        let root_hash = merkle_layer.hash();
        assert_eq!(*commit_id.as_hash(), root_hash);
    }

    #[test]
    fn test_prove_delete() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let (_keepalive, repo) = setup_repo();
        let persistence: Arc<TestKeyValueStore> = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<TestKeyValueStore, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                initial_tree: Tree::default(),
                working_tree: Tree::default(),
                resolver: ProveResolver::start(LazyResolver::new(persistence)),
            },
        };
        ml.delete(&key)
            .expect("deleting a key that doesn't exist should succeed");

        ml.set(&key, b"delete")
            .expect("setting node should succeed");
        ml.delete(&key).expect("delete should succeed");

        let got = ml
            .get(&key)
            .expect("The node should be retrieved successfully.");
        assert!(got.is_none(), "data should not exist after deletion");
    }

    #[test]
    fn test_prove_multiple_keys() {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));
        let data: [&[u8]; 3] = [b"too cold", b"too hot", b"just right"];

        let (_keepalive, repo) = setup_repo();
        let persistence: Arc<TestKeyValueStore> = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<TestKeyValueStore, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                initial_tree: Tree::default(),
                working_tree: Tree::default(),
                resolver: ProveResolver::start(LazyResolver::new(persistence)),
            },
        };
        for (key, datum) in keys.iter().zip(data.iter()) {
            ml.set(key, datum).expect("setting node should succeed");
        }

        for (key, datum) in keys.iter().zip(data.iter()) {
            let got = ml
                .get(key)
                .expect("The node should be retrieved successfully.")
                .expect("data should exist");
            assert_eq!(got, datum);
        }
    }

    #[test]
    #[ignore = "This is not currently supported."]
    fn test_prove_try_clone_with_cow() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let (_keepalive, repo) = setup_repo();
        let persistence: Arc<TestKeyValueStore> = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<TestKeyValueStore, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                initial_tree: Tree::default(),
                working_tree: Tree::default(),
                resolver: ProveResolver::start(LazyResolver::new(persistence)),
            },
        };
        let cow_data = "🐮<(prove a moo!)";
        ml.set(&key, cow_data.as_bytes())
            .expect("setting node should succeed");

        let (_keepalive, repo) = setup_repo();
        let kv: Arc<TestKeyValueStore> = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();

        let mut ml2 = ml.try_clone_with(kv);
        let cow_data2 = "🐮<(mooify a moo!)";
        ml2.set(&key, cow_data2.as_bytes())
            .expect("setting node should succeed");

        // Original should be unchanged.
        let got_original = ml
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("data should exist");
        assert_eq!(got_original, &cow_data.as_bytes().to_vec());

        // Clone should have the new value.
        let got_clone = ml2
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("data should exist");
        assert_eq!(got_clone, &cow_data2.as_bytes().to_vec());
    }

    #[test]
    fn test_prove_write_partial() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let (_keepalive, repo) = setup_repo();
        let persistence: Arc<TestKeyValueStore> = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<TestKeyValueStore, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                initial_tree: Tree::default(),
                working_tree: Tree::default(),
                resolver: ProveResolver::start(LazyResolver::new(persistence)),
            },
        };
        ml.set(&key, b"partial")
            .expect("setting node should succeed");
        ml.write(&key, 4, b"ying").expect("write should succeed");

        let got = ml
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");

        assert_eq!(got, b"partying");
    }

    #[test]
    fn test_prove_verify_round_trip() {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));

        // `Normal` mode
        let (_keepalive, repo) = setup_repo();
        let mut normal_ml = new_merkle_layer(repo);
        normal_ml
            .set(&keys[0], &[])
            .expect("setting node should succeed");
        normal_ml
            .set(&keys[1], b"prove to verify")
            .expect("setting node should succeed");

        // Causes the tree to rotate
        normal_ml
            .set(&keys[2], &[])
            .expect("setting node should succeed");

        // `Prove` mode
        let initial_tree = normal_ml.inner.tree.clone();
        let working_tree = initial_tree.clone().into_proof();
        let prove_ml: MerkleLayer<TestKeyValueStore, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                initial_tree,
                working_tree,
                resolver: ProveResolver::start(LazyResolver::new(
                    normal_ml.inner.persistence.clone(),
                )),
            },
        };

        // Read to mark it as present in the proof
        let node = prove_ml
            .get(&keys[1])
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");
        assert_eq!(node, b"prove to verify");

        // Verify mode
        let proof = MerkleProof::from_foldable(&prove_ml);
        let verify_tree_id = VerifyTreeId::from_proof(ProofTree::Present(&proof))
            .expect("The proof should be deserialisable")
            .into_result();

        let tree = match verify_tree_id {
            VerifyTreeId::Present(tree) => tree,
            _ => panic!("Should be present"),
        };

        let verify_ml: MerkleLayer<TestKeyValueStore, Verify> = MerkleLayer::from_verify_tree(tree);

        let node = verify_ml
            .get(&keys[1])
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");
        assert_eq!(node, b"prove to verify");
    }

    #[test]
    fn test_verify_delete() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut ml: MerkleLayer<TestKeyValueStore, Verify> =
            MerkleLayer::from_verify_tree(Tree::default());
        ml.delete(&key)
            .expect("deleting a key that doesn't exist should succeed");

        ml.set(&key, b"delete")
            .expect("setting node should succeed");
        ml.delete(&key).expect("delete should succeed");

        let got = ml
            .get(&key)
            .expect("The node should be retrieved successfully.");
        assert!(got.is_none(), "data should not exist after deletion");
    }

    #[test]
    fn test_verify_multiple_keys() {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));
        let data: [&[u8]; 3] = [b"too cold", b"too hot", b"just right"];

        let mut ml: MerkleLayer<TestKeyValueStore, Verify> =
            MerkleLayer::from_verify_tree(Tree::default());
        for (key, datum) in keys.iter().zip(data.iter()) {
            ml.set(key, datum).expect("setting node should succeed");
        }

        for (key, datum) in keys.iter().zip(data.iter()) {
            let got = ml
                .get(key)
                .expect("The node should be retrieved successfully.")
                .expect("data should exist");
            assert_eq!(got, datum);
        }
    }

    #[test]
    #[ignore = "This is not currently supported."]
    fn test_verify_try_clone_with_cow() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut ml: MerkleLayer<TestKeyValueStore, Verify> =
            MerkleLayer::from_verify_tree(Tree::default());
        let cow_data = "🐮<(verify a moo!)";
        ml.set(&key, cow_data.as_bytes())
            .expect("setting node should succeed");

        let (_keepalive, repo) = setup_repo();
        let kv: Arc<TestKeyValueStore> = TestKeyValueStore::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();

        let mut ml2 = ml.try_clone_with(kv);
        let cow_data2 = "🐮<(mooify a moo!)";
        ml2.set(&key, cow_data2.as_bytes())
            .expect("setting node should succeed");

        // Original should be unchanged.
        let got_original = ml
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("data should exist");
        assert_eq!(got_original, &cow_data.as_bytes().to_vec());

        // Clone should have the new value.
        let got_clone = ml2
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("data should exist");
        assert_eq!(got_clone, &cow_data2.as_bytes().to_vec());
    }

    #[test]
    fn test_verify_write_partial() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut ml: MerkleLayer<TestKeyValueStore, Verify> =
            MerkleLayer::from_verify_tree(Tree::default());
        ml.set(&key, b"partial")
            .expect("setting node should succeed");
        ml.write(&key, 4, b"ying").expect("write should succeed");

        let got = ml
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");

        assert_eq!(got, b"partying");
    }

    #[test]
    fn test_prove_hash() {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));

        let (_keepalive, repo) = setup_repo();
        let mut normal_ml = new_merkle_layer(repo);
        normal_ml
            .set(&keys[0], &[])
            .expect("setting node should succeed");
        normal_ml
            .set(&keys[1], &[])
            .expect("setting node should succeed");
        normal_ml
            .set(&keys[2], &[])
            .expect("setting node should succeed");

        let normal_hash = normal_ml.hash();

        let initial_tree = normal_ml.inner.tree.clone();
        let working_tree = initial_tree.clone().into_proof();
        let prove_ml: MerkleLayer<TestKeyValueStore, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                initial_tree,
                working_tree,
                resolver: ProveResolver::start(LazyResolver::new(
                    normal_ml.inner.persistence.clone(),
                )),
            },
        };

        assert_eq!(normal_hash, prove_ml.hash());
    }

    // -----------------------------------------------------------------------
    // Prove → Verify round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prove_verify_round_trip_write() {
        let key1 = Key::new(&[1]).expect("key should be valid");
        let key2 = Key::new(&[2]).expect("key should be valid");
        let key3 = Key::new(&[3]).expect("key should be valid");

        // ---- Normal: build a three-node tree ----
        let (_keepalive, repo) = setup_repo();
        let mut normal_ml = new_merkle_layer(repo);
        normal_ml.set(&key1, b"alpha").expect("set should succeed");
        normal_ml.set(&key2, b"beta").expect("set should succeed");
        normal_ml.set(&key3, b"gamma").expect("set should succeed");

        let initial_hash = normal_ml.hash();

        // ---- Prove: start proof, read key2, write key2 ----
        let mut prove_ml = normal_ml.start_proof();

        // Read key2 (triggers access tracking for meta + data along the search path).
        let data = prove_ml
            .get(&key2)
            .expect("get should succeed")
            .expect("key2 should exist");
        let mut buf = vec![0u8; data.len()];
        data.read(0, &mut buf);
        assert_eq!(&buf, b"beta");

        // Overwrite key2 in place (no structural change).
        prove_ml
            .write(&key2, 0, b"BETA")
            .expect("write should succeed");

        // Expected final-state hash from prove-mode working tree.
        let expected_hash = prove_ml.hash();
        assert_ne!(initial_hash, expected_hash, "write should change the hash");

        // ---- Generate proof ----
        let merkle_proof = MerkleProof::from_foldable(&prove_ml);

        // ---- Verify: deserialize proof, replay identical ops ----
        let proof_deser = ProofTree::Present(&merkle_proof);
        let verify_tree_id = VerifyTreeId::from_proof(proof_deser)
            .expect("proof deserialization should succeed")
            .into_result();

        let VerifyTreeId::Present(mut verify_tree) = verify_tree_id else {
            panic!("expected Present tree from proof, got {verify_tree_id:#?}");
        };
        let mut verify_resolver = VerifyResolver;

        let final_hash = catch_not_found(move || {
            // Read key2.
            let data = verify_tree
                .get(&key2, &verify_resolver)
                .expect("get should succeed")
                .expect("key2 should be present in proof");
            let mut buf = vec![0u8; data.len()];
            data.read(0, &mut buf);
            assert_eq!(&buf, b"beta", "verify should see initial data");

            // Overwrite key2.
            verify_tree
                .write(&key2, 0, b"BETA", &mut verify_resolver)
                .expect("write should succeed");

            // Compute verify hash. For MAVL trees, blinded nodes already carry their
            // hash, so we don't need the proof tree for hash recovery.
            let verify_tree_id = VerifyTreeId::Present(verify_tree);
            PartialHash::from_foldable(None, &verify_tree_id)
                .to_hash()
                .expect("verify hash should be computable")
        })
        .expect("verify operations should not trigger not_found");

        assert_eq!(
            expected_hash, final_hash,
            "prove and verify hashes should match after identical get + write operations"
        );
    }

    // -----------------------------------------------------------------------
    // Property-based prove → verify round-trip tests
    // -----------------------------------------------------------------------

    /// Operations executed during a prove step and replayed during verification.
    #[derive(Debug, Clone)]
    enum StepOp {
        /// Set (insert or overwrite) a key.
        Set(Key, Vec<u8>),
        /// Delete a key (may be absent — that is a no-op on both sides).
        Delete(Key),
        /// In-place write at offset 0. Only applied to keys that are known to exist; skipped
        /// otherwise. This avoids creating new nodes (which would be a `set`, not a `write`).
        Write(Key, Vec<u8>),
    }

    /// Execute a [`StepOp`] on a prove-mode [`MerkleLayer`].
    fn apply_step_op_prove<KV: KeyValueStore>(ml: &mut MerkleLayer<KV, Prove<'_>>, op: &StepOp) {
        match op {
            StepOp::Set(key, data) => {
                ml.set(key, data).expect("prove set should succeed");
            }
            StepOp::Delete(key) => {
                ml.delete(key).expect("prove delete should succeed");
            }
            StepOp::Write(key, data) => {
                if ml.get(key).expect("get should succeed").is_some() {
                    ml.write(key, 0, data).expect("prove write should succeed");
                }
            }
        }
    }

    /// Apply a [`StepOp`] to a Normal-mode [`MerkleLayer`].
    fn apply_step_op_normal<KV: KeyValueStore>(ml: &mut MerkleLayer<KV, Normal>, op: &StepOp) {
        match op {
            StepOp::Set(key, data) => {
                ml.set(key, data).expect("normal set should succeed");
            }
            StepOp::Delete(key) => {
                ml.delete(key).expect("normal delete should succeed");
            }
            StepOp::Write(key, data) => {
                if ml.get(key).expect("get should succeed").is_some() {
                    ml.write(key, 0, data).expect("normal write should succeed");
                }
            }
        }
    }

    /// Core assertion: prove-mode proof and hash are consistent with Normal mode.
    ///
    /// Checks two properties:
    /// 1. The proof's initial root hash matches the initial Normal-mode hash (proof encodes initial
    ///    state correctly).
    /// 2. The prove-mode final hash matches Normal-mode after identical operations (prove-mode
    ///    mutations are faithful).
    fn assert_prove_mode_correct(setup_keys: &[[u8; 2]], step_ops: &[StepOp]) {
        // ---- Normal: build initial tree ----
        let (_keepalive, repo) = setup_repo();
        let mut normal_ml = new_merkle_layer(repo);

        for bytes in setup_keys {
            let key = Key::new(bytes).expect("key should be valid");
            normal_ml.set(&key, bytes).expect("set should succeed");
        }

        let initial_hash = normal_ml.hash();

        // ---- Prove: start proof and execute step operations ----
        let mut prove_ml = normal_ml.start_proof();
        for op in step_ops {
            apply_step_op_prove(&mut prove_ml, op);
        }
        let prove_final_hash = prove_ml.hash();

        // ---- Generate proof ----
        let merkle_proof = MerkleProof::from_foldable(&prove_ml);

        // Property 1: proof's root hash == initial Normal-mode hash.
        assert_eq!(
            initial_hash,
            merkle_proof.root_hash(),
            "proof root hash must match initial Normal-mode hash"
        );

        // ---- Normal: replay same operations for reference hash ----
        for op in step_ops {
            apply_step_op_normal(&mut normal_ml, op);
        }
        let normal_final_hash = normal_ml.hash();

        // Property 2: prove-mode final hash == Normal-mode final hash.
        assert_eq!(
            normal_final_hash, prove_final_hash,
            "prove-mode final hash must match Normal-mode final hash"
        );
    }

    proptest! {
        /// Property: for any initial tree and sequence of set/delete/write operations during a
        /// prove step, the generated Merkle proof is sufficient for a verifier to replay the
        /// same operations and arrive at the same final-state hash.
        #[test]
        fn test_prove_verify_round_trip_prop(
            setup_keys in prop::collection::vec(any::<[u8; 2]>(), 1..30),
            seed in 0..100usize,
        ) {
            let keys: Vec<Key> = setup_keys
                .iter()
                .map(|b| Key::new(b).expect("key should be valid"))
                .collect();

            let mut step_ops = Vec::new();
            let ops_count = 1 + (seed % 20);
            for i in 0..ops_count {
                let key = keys[i % keys.len()].clone();
                let data = vec![(i as u8).wrapping_mul(37); 1 + (i % 16)];
                match (i + seed) % 3 {
                    0 => step_ops.push(StepOp::Set(key, data)),
                    1 => step_ops.push(StepOp::Delete(key)),
                    _ => step_ops.push(StepOp::Write(key, data)),
                }
            }

            assert_prove_mode_correct(&setup_keys, &step_ops);
        }

        /// Property: prove-verify round trip with only write (no structural change).
        #[test]
        fn test_prove_verify_writes_only_prop(
            setup_keys in prop::collection::vec(any::<[u8; 2]>(), 1..30),
        ) {
            let keys: Vec<Key> = setup_keys
                .iter()
                .map(|b| Key::new(b).expect("key should be valid"))
                .collect();

            let step_ops: Vec<StepOp> = keys
                .iter()
                .enumerate()
                .map(|(i, key)| StepOp::Write(key.clone(), vec![0xAA; 1 + (i % 8)]))
                .collect();

            assert_prove_mode_correct(&setup_keys, &step_ops);
        }

        /// Property: prove-verify round trip with only deletes (structural changes).
        #[test]
        fn test_prove_verify_deletes_only_prop(
            setup_keys in prop::collection::vec(any::<[u8; 2]>(), 1..30),
            delete_indices in prop::collection::vec(any::<usize>(), 1..10),
        ) {
            let keys: Vec<Key> = setup_keys
                .iter()
                .map(|b| Key::new(b).expect("key should be valid"))
                .collect();

            let step_ops: Vec<StepOp> = delete_indices
                .iter()
                .map(|&i| StepOp::Delete(keys[i % keys.len()].clone()))
                .collect();

            assert_prove_mode_correct(&setup_keys, &step_ops);
        }

        /// Property: prove-verify round trip with sets on new keys (insertions + rotations).
        #[test]
        fn test_prove_verify_insertions_prop(
            setup_keys in prop::collection::vec(any::<[u8; 2]>(), 1..20),
            new_keys in prop::collection::vec(any::<[u8; 2]>(), 1..10),
        ) {
            let step_ops: Vec<StepOp> = new_keys
                .iter()
                .enumerate()
                .map(|(i, bytes)| {
                    let key = Key::new(bytes).expect("key should be valid");
                    StepOp::Set(key, vec![0xBB; 1 + (i % 8)])
                })
                .collect();

            assert_prove_mode_correct(&setup_keys, &step_ops);
        }
    }
}
