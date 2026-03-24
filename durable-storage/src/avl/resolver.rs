// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Resolution strategies for identifiers of [`Tree`] and [`Node`] objects.
//!
//! This module supports two resolver modes:
//! - [`ArcResolver`] for eagerly loaded values, where node and tree identifiers directly contain
//!   in-memory data.
//! - [`LazyResolver`] for hash-backed values, where identifiers can start as hashes and materialise
//!   values from storage on first access.
//!
//! # Lazy loading strategy
//! `LazyResolver` works with [`LazyId`] wrappers. A `LazyId` keeps a hash (`id`) and/or a loaded
//! value (`inner`) and transitions from "hash-only" to "value-cached" when `resolve` or
//! `resolve_mut` is called. This avoids loading the full tree upfront while preserving stable hash
//! computation.
//!
//! # ArcResolver vs LazyResolver
//! Use [`ArcResolver`] when values are already present and can be shared directly via [`Arc`]. Use
//! [`LazyResolver`] when values are persisted in a [`KeyValueStore`] and should be fetched on
//! demand.
//!
//! [`Tree`]: crate::avl::tree::Tree
//! [`Node`]: crate::avl::node::Node

use std::sync::Arc;
use std::sync::OnceLock;

use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::serialisation::deserialise;
use trait_set::trait_set;

use super::node::Node;
use super::tree::Tree;
use crate::avl::node::NodeHashRepresentation;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::storage::KeyValueStore;

/// Trait for resolving identifiers to values.
pub trait Resolver<Id, Value> {
    /// Resolve an identifier to a value.
    fn resolve<'a>(&self, id: &'a Id) -> Result<&'a Value, OperationalError>;

    /// Resolve an identifier to a mutable value.
    fn resolve_mut<'a>(&mut self, id: &'a mut Id) -> Result<&'a mut Value, OperationalError>;
}

trait_set! {
    /// Specialised [`Resolver`] for MAVL nodes
    pub trait NodeResolver<NodeId, TreeId, M: Mode> = Resolver<NodeId, Node<TreeId, M>>;

    /// Specialised [`Resolver`] for MAVL trees
    pub trait TreeResolver<NodeId, TreeId> = Resolver<TreeId, Tree<NodeId>>;

    /// Specialised [`Resolver`] for MAVL nodes and trees
    pub trait AvlResolver<NodeId, TreeId, M: Mode> = NodeResolver<NodeId, TreeId, M> + TreeResolver<NodeId, TreeId>;
}

/// Trait for types that can be used as a resolver identifier
pub trait ResolverId {
    /// Returns identity in the form of a [`Hash`].
    fn hash(&self) -> Hash;
}

/// Identifier for a node that is always present.
#[derive(Debug, Clone, derive_more::From)]
#[from(Node<ArcTreeId, Normal>)]
pub struct ArcNodeId(Arc<Node<ArcTreeId, Normal>>);

impl ResolverId for ArcNodeId {
    fn hash(&self) -> Hash {
        *self.0.hash()
    }
}
impl Foldable<HashFold> for ArcNodeId {
    fn fold(&self, _builder: HashFold) -> <HashFold as Fold>::Folded {
        *self.0.hash()
    }
}

/// ID for a tree that is always present
#[derive(Debug, Clone, derive_more::From, Default)]
pub struct ArcTreeId(Tree<ArcNodeId>);

impl ResolverId for ArcTreeId {
    fn hash(&self) -> Hash {
        self.0.hash()
    }
}
impl Foldable<HashFold> for ArcTreeId {
    fn fold(&self, _builder: HashFold) -> <HashFold as Fold>::Folded {
        self.0.hash()
    }
}

/// Eager resolver that serves identifiers backed by in-memory [`Arc`] values.
///
/// This resolver never touches persistent storage. It is useful for trees already in memory
/// and for tests.
#[derive(Clone, Debug)]
pub struct ArcResolver;

impl Resolver<ArcNodeId, Node<ArcTreeId, Normal>> for ArcResolver {
    fn resolve<'a>(
        &self,
        id: &'a ArcNodeId,
    ) -> Result<&'a Node<ArcTreeId, Normal>, OperationalError> {
        Ok(id.0.as_ref())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut ArcNodeId,
    ) -> Result<&'a mut Node<ArcTreeId, Normal>, OperationalError> {
        Ok(Arc::make_mut(&mut id.0))
    }
}

impl Resolver<ArcTreeId, Tree<ArcNodeId>> for ArcResolver {
    fn resolve<'a>(&self, id: &'a ArcTreeId) -> Result<&'a Tree<ArcNodeId>, OperationalError> {
        Ok(&id.0)
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut ArcTreeId,
    ) -> Result<&'a mut Tree<ArcNodeId>, OperationalError> {
        Ok(&mut id.0)
    }
}

/// Identifier wrapper used by lazy resolution.
///
/// A `LazyId` is either:
/// - unresolved, where `id` contains a hash and `inner` is empty, or
/// - resolved, where `inner` contains the loaded value and `id` may be absent.
///
/// This representation lets hashes move through the AVL structure without forcing immediate loads,
/// while caching loaded values for subsequent accesses.
#[derive(Default, Debug, Clone)]
pub struct LazyId<Id, Value> {
    inner: OnceLock<Value>,
    id: Option<Id>,
}

impl<Id, Value> LazyId<Id, Value> {
    /// Create an identifier from an already loaded value.
    pub fn new(value: Value) -> Self {
        Self {
            inner: OnceLock::from(value),
            id: None,
        }
    }

    /// Return the identifier if available.
    fn id(&self) -> Option<&Id> {
        self.id.as_ref()
    }
}

impl<Value> From<Hash> for LazyId<Hash, Value> {
    fn from(hash: Hash) -> Self {
        Self {
            inner: OnceLock::new(),
            id: Some(hash),
        }
    }
}

impl<F: Fold, Id: Foldable<F>, Value: Foldable<F>> Foldable<F> for LazyId<Id, Value> {
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        if let Some(value) = self.inner.get() {
            return value.fold(builder);
        }

        self.id.as_ref().expect("TODO").fold(builder)
    }
}

/// Identifier for an AVL node.
#[derive(Debug, Clone)]
pub struct LazyNodeId(LazyId<Hash, Arc<Node<LazyTreeId, Normal>>>);

impl From<Node<LazyTreeId, Normal>> for LazyNodeId {
    fn from(value: Node<LazyTreeId, Normal>) -> Self {
        let value = Arc::new(value);
        Self(LazyId::new(value))
    }
}

impl From<Hash> for LazyNodeId {
    fn from(hash: Hash) -> LazyNodeId {
        LazyNodeId(hash.into())
    }
}

impl ResolverId for LazyNodeId {
    fn hash(&self) -> Hash {
        if let Some(node) = self.0.inner.get() {
            return *node.hash();
        }

        self.0
            .id()
            .cloned()
            .expect("ID should be present when node is absent")
    }
}
impl Foldable<HashFold> for LazyNodeId {
    fn fold(&self, builder: HashFold) -> <HashFold as Fold>::Folded {
        self.0.fold(builder)
    }
}

/// Identifier for an AVL tree.
#[derive(Debug, Clone)]
pub struct LazyTreeId(LazyId<Hash, Tree<LazyNodeId>>);

impl From<Hash> for LazyTreeId {
    fn from(hash: Hash) -> Self {
        LazyTreeId(hash.into())
    }
}

impl Default for LazyTreeId {
    fn default() -> Self {
        Self(LazyId {
            inner: OnceLock::from(Tree::default()),
            id: None,
        })
    }
}

impl ResolverId for LazyTreeId {
    fn hash(&self) -> Hash {
        if let Some(tree) = self.0.inner.get() {
            return tree.hash();
        }

        self.0
            .id()
            .cloned()
            .expect("ID should be present when tree is absent")
    }
}
impl Foldable<HashFold> for LazyTreeId {
    fn fold(&self, builder: HashFold) -> <HashFold as Fold>::Folded {
        self.0.fold(builder)
    }
}

/// Resolver that lazily loads AVL nodes and trees from a [`KeyValueStore`].
///
/// In contrast to [`ArcResolver`], this resolver can start from hash-only identifiers and defer
/// storage reads until an identifier is resolved. Loaded values are cached in their corresponding
/// [`LazyId`] so repeated resolutions avoid extra storage lookups.
#[derive(Clone, Debug)]
pub struct LazyResolver<KV> {
    persistence_layer: Arc<KV>,
}

impl<KV> LazyResolver<KV> {
    /// Create a resolver backed by a persistence layer.
    ///
    /// The provided store is shared via [`Arc`], which allows cloned resolvers to resolve against
    /// the same underlying storage.
    pub fn new(persistence_layer: Arc<KV>) -> Self {
        Self { persistence_layer }
    }
}

impl<KV: KeyValueStore> LazyResolver<KV> {
    /// Load and decode a node by its content hash.
    ///
    /// This performs a `blob_get` lookup and deserialises the returned bytes into a node representation.
    fn load_node(&self, hash: Hash) -> Result<Arc<Node<LazyTreeId, Normal>>, OperationalError> {
        let bytes = self
            .persistence_layer
            .blob_get(hash)
            .map_err(|error| error.into_resolver_op_error(hash))?;
        let noderepr =
            deserialise::<NodeHashRepresentation<Bytes<Normal>, Key, Hash>>(bytes.as_ref())?;
        Ok(Arc::new(Node::from(noderepr)))
    }

    /// Load and decode a tree root reference by its content hash.
    ///
    /// The serialised payload is expected to be an optional root node hash, which is then wrapped
    /// in a lazy node identifier.
    fn load_tree(&self, hash: Hash) -> Result<Tree<LazyNodeId>, OperationalError> {
        if hash == crate::merkle_layer::empty_tree_hash() {
            return Ok(Tree::default());
        }

        let bytes = self
            .persistence_layer
            .blob_get(hash)
            .map_err(|error| error.into_resolver_op_error(hash))?;
        let tree_repr = deserialise::<Option<Hash>>(bytes.as_ref())?.map(LazyNodeId::from);
        Ok(Tree::from(tree_repr))
    }
}

impl<KV: KeyValueStore> Resolver<LazyNodeId, Node<LazyTreeId, Normal>> for LazyResolver<KV> {
    fn resolve<'a>(
        &self,
        id: &'a LazyNodeId,
    ) -> Result<&'a Node<LazyTreeId, Normal>, OperationalError> {
        if let Some(value) = id.0.inner.get() {
            return Ok(value);
        }
        let &hash =
            id.0.id()
                .ok_or(OperationalError::ResolverInvariantViolated)?;
        let node = self.load_node(hash)?;
        let _ = id.0.inner.set(node);
        Ok(id.0.inner.wait().as_ref())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut LazyNodeId,
    ) -> Result<&'a mut Node<LazyTreeId, Normal>, OperationalError> {
        if let Some(value) = id.0.inner.get_mut() {
            let temp = value as *mut Arc<_>;
            // This unsafe workaround is required because the rust borrow-checker
            // is unable to identify the `value` mutable reference being dropped straight
            // away if the condition is false.
            //
            // SAFETY: This is a valid active &mut Arc<_> reference with no other
            // references to the same Arc being used after this return.
            return Ok(Arc::make_mut(unsafe { &mut *temp }));
        };

        let hash =
            id.0.id()
                .ok_or(OperationalError::ResolverInvariantViolated)?;
        let _ = id.0.inner.set(self.load_node(*hash)?);

        id.0.id = None;
        id.0.inner
            .get_mut()
            .ok_or(OperationalError::ResolverInvariantViolated)
            .map(Arc::make_mut)
    }
}

impl<KV: KeyValueStore> Resolver<LazyTreeId, Tree<LazyNodeId>> for LazyResolver<KV> {
    fn resolve<'a>(&self, id: &'a LazyTreeId) -> Result<&'a Tree<LazyNodeId>, OperationalError> {
        if let Some(value) = id.0.inner.get() {
            return Ok(value);
        }
        let &hash =
            id.0.id()
                .ok_or(OperationalError::ResolverInvariantViolated)?;
        let tree = self.load_tree(hash)?;
        let _ = id.0.inner.set(tree);
        Ok(id.0.inner.wait())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut LazyTreeId,
    ) -> Result<&'a mut Tree<LazyNodeId>, OperationalError> {
        if id.0.inner.get().is_none() {
            let hash =
                id.0.id()
                    .ok_or(OperationalError::ResolverInvariantViolated)?;
            let _ = id.0.inner.set(self.load_tree(*hash)?);
        }

        id.0.id = None;
        id.0.inner
            .get_mut()
            .ok_or(OperationalError::ResolverInvariantViolated)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::OnceLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use octez_riscv_data::foldable::Foldable;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::HashFold;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::serialisation;

    use super::ArcNodeId;
    use super::ArcResolver;
    use super::LazyId;
    use super::LazyNodeId;
    use super::LazyResolver;
    use super::LazyTreeId;
    use super::Resolver;
    use crate::avl::resolver::AvlResolver;
    use crate::avl::resolver::ResolverId;
    use crate::avl::tree::Tree;
    use crate::errors::Error;
    use crate::errors::InvalidArgumentError;
    use crate::errors::OperationalError;
    use crate::key::Key;
    use crate::storage::KeyValueStore;
    use crate::storage::in_memory::InMemoryKeyValueStore;
    use crate::storage::in_memory::InMemoryRepo;

    /// A wrapper around an in-memory key-value store that counts the number of `blob_get` calls.
    #[derive(Debug, Default)]
    struct CountingKeyValueStore {
        inner: InMemoryKeyValueStore,
        blob_get_calls: AtomicUsize,
    }

    impl CountingKeyValueStore {
        fn blob_get_calls(&self) -> usize {
            self.blob_get_calls.load(Ordering::SeqCst)
        }
    }

    impl KeyValueStore for CountingKeyValueStore {
        type Repo = InMemoryRepo;

        fn new(_repo: &Self::Repo) -> Result<Self, OperationalError> {
            Ok(Self::default())
        }

        fn try_clone(&self, _repo: &Self::Repo) -> Result<Self, OperationalError> {
            Ok(Self {
                inner: self.inner.try_clone()?,
                blob_get_calls: AtomicUsize::new(self.blob_get_calls()),
            })
        }

        fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
            self.blob_get_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.blob_get(key)
        }

        fn blob_set(
            &self,
            key: impl AsRef<[u8]>,
            data: impl AsRef<[u8]>,
        ) -> Result<(), OperationalError> {
            self.inner.blob_set(key, data)
        }

        fn blob_delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError> {
            self.inner.blob_delete(key)
        }

        fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
            self.inner.get(key)
        }

        fn set(
            &self,
            key: impl AsRef<[u8]>,
            value: impl AsRef<[u8]>,
        ) -> Result<(), OperationalError> {
            self.inner.set(key, value)
        }

        fn write(
            &self,
            key: impl AsRef<[u8]>,
            offset: usize,
            value: impl AsRef<[u8]>,
        ) -> Result<(), Error> {
            self.inner.write(key, offset, value)
        }

        fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError> {
            self.inner.delete(key)
        }
    }

    fn persist_tree<NodeId, TreeId, Res, KV>(
        tree: &Tree<NodeId>,
        resolver: &Res,
        persistence_layer: &KV,
    ) where
        NodeId: ResolverId + Foldable<HashFold>,
        TreeId: ResolverId,
        KV: KeyValueStore,
        Res: AvlResolver<NodeId, TreeId, Normal>,
    {
        // LazyTreeId resolves by loading a serialised optional root hash.
        let tree_id = tree.hash();
        let tree_repr: Option<Hash> = tree.root().map(ResolverId::hash);
        let tree_bytes =
            serialisation::serialise(tree_repr).expect("tree serialisation should succeed");
        persistence_layer
            .blob_set(tree_id, tree_bytes)
            .expect("persisting trees should succeed");

        let Some(node_id) = tree.root() else {
            return;
        };

        let node = resolver
            .resolve(node_id)
            .expect("resolving nodes should succeed");
        let encoded = node.to_encode();
        let node_bytes =
            serialisation::serialise(encoded).expect("node serialisation should succeed");
        persistence_layer
            .blob_set(node_id.hash(), node_bytes)
            .expect("persisting nodes should succeed");

        persist_tree(
            node.left_ref(resolver)
                .expect("left subtree should resolve"),
            resolver,
            persistence_layer,
        );
        persist_tree(
            node.right_ref(resolver)
                .expect("right subtree should resolve"),
            resolver,
            persistence_layer,
        );
    }

    #[test]
    fn lazy_resolver_loads_values_only_when_accessed() {
        let root_key = Key::new(&[2]).expect("key should be valid");
        let left_key = Key::new(&[1]).expect("key should be valid");

        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut eager_resolver = ArcResolver;
        tree.set(&root_key, b"root", &mut eager_resolver)
            .expect("set should succeed");
        tree.set(&left_key, b"left", &mut eager_resolver)
            .expect("set should succeed");

        let tree_hash = tree.hash();
        let root_hash = tree.root().expect("tree should have a root node").hash();

        let persistence_layer = Arc::new(CountingKeyValueStore::default());
        persist_tree(&tree, &eager_resolver, persistence_layer.as_ref());

        let lazy_resolver = LazyResolver::new(persistence_layer.clone());
        let lazy_tree: LazyTreeId = LazyTreeId::from(tree_hash);

        assert_eq!(persistence_layer.blob_get_calls(), 0);
        assert_eq!(lazy_tree.hash(), tree_hash);
        assert_eq!(
            persistence_layer.blob_get_calls(),
            0,
            "hash-only operations should not trigger loads"
        );

        let loaded_tree = lazy_resolver
            .resolve(&lazy_tree)
            .expect("resolving tree should succeed");
        assert_eq!(
            persistence_layer.blob_get_calls(),
            1,
            "resolving tree should load only the tree payload"
        );

        let lazy_root = loaded_tree.root().expect("tree should have a root");
        assert!(lazy_root.0.inner.get().is_none());
        assert_eq!(lazy_root.hash(), root_hash);
        assert_eq!(
            persistence_layer.blob_get_calls(),
            1,
            "resolving tree should not eagerly load the root node payload"
        );

        let _ = lazy_resolver
            .resolve(lazy_root)
            .expect("resolving root node should succeed");
        assert_eq!(
            persistence_layer.blob_get_calls(),
            2,
            "node payload should be loaded only when node is accessed"
        );
    }

    #[test]
    fn lazy_resolver_returns_invariant_error_when_hash_is_missing() {
        let persistence_layer = Arc::new(InMemoryKeyValueStore::default());
        let mut lazy_resolver = LazyResolver::new(persistence_layer);

        let node_without_hash = LazyNodeId(LazyId {
            inner: OnceLock::new(),
            id: None,
        });
        let mut node_without_hash_mut = node_without_hash.clone();

        assert!(matches!(
            lazy_resolver.resolve(&node_without_hash),
            Err(OperationalError::ResolverInvariantViolated)
        ));
        assert!(matches!(
            lazy_resolver.resolve_mut(&mut node_without_hash_mut),
            Err(OperationalError::ResolverInvariantViolated)
        ));

        let tree_without_hash = LazyTreeId(LazyId {
            inner: OnceLock::new(),
            id: None,
        });
        let mut tree_without_hash_mut = tree_without_hash.clone();

        assert!(matches!(
            lazy_resolver.resolve(&tree_without_hash),
            Err(OperationalError::ResolverInvariantViolated)
        ));
        assert!(matches!(
            lazy_resolver.resolve_mut(&mut tree_without_hash_mut),
            Err(OperationalError::ResolverInvariantViolated)
        ));
    }

    #[test]
    fn lazy_resolver_maps_missing_cas_entries_to_lookup_error() {
        let missing_hash = Hash::hash_bytes(b"missing");
        let persistence_layer = Arc::new(InMemoryKeyValueStore::default());
        let lazy_resolver = LazyResolver::new(persistence_layer);

        let node_id = LazyNodeId::from(missing_hash);
        assert!(matches!(
            lazy_resolver.resolve(&node_id),
            Err(OperationalError::ResolverCasLookup {
                hash,
                error: InvalidArgumentError::KeyNotFound
            }) if hash == missing_hash
        ));

        let tree_id = LazyTreeId::from(missing_hash);
        assert!(matches!(
            lazy_resolver.resolve(&tree_id),
            Err(OperationalError::ResolverCasLookup {
                hash,
                error: InvalidArgumentError::KeyNotFound
            }) if hash == missing_hash
        ));
    }

    #[test]
    fn lazy_resolver_caches_values_after_first_load() {
        let root_key = Key::new(&[1]).expect("key should be valid");

        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut eager_resolver = ArcResolver;
        tree.set(&root_key, b"root", &mut eager_resolver)
            .expect("set should succeed");

        let root_hash = tree.root().expect("tree should have a root node").hash();

        let persistence_layer = Arc::new(CountingKeyValueStore::default());
        persist_tree(&tree, &eager_resolver, persistence_layer.as_ref());

        let mut node_id: LazyNodeId = LazyNodeId::from(root_hash);
        let mut lazy_resolver = LazyResolver::new(persistence_layer.clone());

        let _ = lazy_resolver
            .resolve(&node_id)
            .expect("first resolve should succeed");
        assert_eq!(persistence_layer.blob_get_calls(), 1);

        let _ = lazy_resolver
            .resolve(&node_id)
            .expect("second resolve should use cache");
        assert_eq!(persistence_layer.blob_get_calls(), 1);

        let _ = lazy_resolver
            .resolve_mut(&mut node_id)
            .expect("resolve_mut should use cached value");
        assert_eq!(persistence_layer.blob_get_calls(), 1);
    }

    #[test]
    fn lazy_resolver_supports_mutation_through_resolve_mut() {
        let root_key = Key::new(&[1]).expect("key should be valid");

        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut eager_resolver = ArcResolver;
        tree.set(&root_key, b"root", &mut eager_resolver)
            .expect("set should succeed");

        let root_hash = tree.root().expect("tree should have a root node").hash();

        let persistence_layer = Arc::new(CountingKeyValueStore::default());
        persist_tree(&tree, &eager_resolver, persistence_layer.as_ref());

        let mut lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let mut lazy_resolver = LazyResolver::new(persistence_layer.clone());

        {
            let root_id = lazy_tree
                .root_mut()
                .expect("lazy tree should have a root node");

            let node = lazy_resolver
                .resolve_mut(root_id)
                .expect("resolve_mut should load node");
            assert_eq!(persistence_layer.blob_get_calls(), 1);

            node.upsert(
                &root_key,
                0,
                |value| {
                    value.set(b"root-mutated");
                    Ok(())
                },
                &mut lazy_resolver,
            )
            .expect("mutating the resolved node should succeed");
            assert_eq!(persistence_layer.blob_get_calls(), 1);
        }

        let root_id = lazy_tree
            .root()
            .expect("lazy tree should still have a root");
        let node = lazy_resolver
            .resolve(root_id)
            .expect("resolving mutated node should succeed");

        let mut data = vec![0; node.data().len()];
        node.data().read(0, &mut data);
        assert_eq!(data.as_slice(), b"root-mutated");

        let persisted_tree_hash = lazy_tree.hash();
        persist_tree(&lazy_tree, &lazy_resolver, persistence_layer.as_ref());
        assert_eq!(
            persistence_layer.blob_get_calls(),
            1,
            "Re-persisting the tree should not load empty child subtrees from storage"
        );

        let reloaded_tree_id = LazyTreeId::from(persisted_tree_hash);

        let reloaded_tree = lazy_resolver
            .resolve(&reloaded_tree_id)
            .expect("resolving persisted mutated tree should succeed");
        assert_eq!(persistence_layer.blob_get_calls(), 2);

        let reloaded_root = reloaded_tree
            .root()
            .expect("persisted mutated tree should have a root node");

        let reloaded_node = lazy_resolver
            .resolve(reloaded_root)
            .expect("resolving persisted mutated root node should succeed");
        assert_eq!(persistence_layer.blob_get_calls(), 3);

        let mut reloaded_data = vec![0; reloaded_node.data().len()];
        reloaded_node.data().read(0, &mut reloaded_data);
        assert_eq!(reloaded_data.as_slice(), b"root-mutated");
    }

    #[test]
    fn lazy_resolver_hash_changes_after_mutating_loaded_child() {
        let root_key = Key::new(&[2]).expect("key should be valid");
        let left_key = Key::new(&[1]).expect("key should be valid");

        let mut original_tree: Tree<ArcNodeId> = Default::default();
        let mut original_resolver = ArcResolver;
        original_tree
            .set(&root_key, b"root", &mut original_resolver)
            .expect("set should succeed");
        original_tree
            .set(&left_key, b"left", &mut original_resolver)
            .expect("set should succeed");

        let initial_tree_hash: Hash = original_tree.hash();

        let persisted_root_hash = original_tree
            .root()
            .expect("tree should have a root node")
            .hash();

        let persistence_layer = Arc::new(InMemoryKeyValueStore::default());
        persist_tree(
            &original_tree,
            &original_resolver,
            persistence_layer.as_ref(),
        );

        let mut lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(persisted_root_hash)).into();
        let mut lazy_resolver = LazyResolver::new(persistence_layer);

        lazy_tree
            .set(&left_key, b"left-mutated", &mut lazy_resolver)
            .expect("set should succeed");
        let hash_after_mutation = lazy_tree.hash();

        let mut expected_tree = original_tree.clone();
        let mut expected_resolver = ArcResolver;
        expected_tree
            .set(&left_key, b"left-mutated", &mut expected_resolver)
            .expect("set should succeed");
        let expected_hash = expected_tree.hash();

        assert_ne!(
            hash_after_mutation, initial_tree_hash,
            "tree hash should change when mutating an existing child node"
        );
        assert_eq!(
            hash_after_mutation, expected_hash,
            "lazy resolver tree hash should match eager resolver after same mutation"
        );
    }
}
