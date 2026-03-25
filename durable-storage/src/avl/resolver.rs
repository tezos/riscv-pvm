// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Resolution strategies for identifiers of [`Tree`] and [`Node`] objects.
//!
//! This module supports three resolver modes:
//! - [`ArcResolver`] for eagerly loaded values, where node and tree identifiers directly contain
//!   in-memory data.
//! - [`LazyResolver`] for hash-backed values, where identifiers can start as hashes and materialise
//!   values from storage on first access.
//! - [`ProveResolver`] to project lazy identifiers into prove-mode nodes and trees on
//!   demand without changing the underlying storage format.
//!
//! # Lazy loading strategy
//! `LazyResolver` works with [`LazyId`] wrappers. A `LazyId` keeps a hash (`id`) and/or a loaded
//! value (`inner`). Immutable resolution populates `inner` while keeping the hash available for
//! later lookups. Mutable resolution clears the stored hash once the loaded value becomes the
//! source of truth. This avoids loading the full tree upfront while preserving stable hash
//! computation for unchanged identifiers.
//!
//! # ArcResolver vs LazyResolver
//! Use [`ArcResolver`] when values are already present and can be shared directly via [`Arc`]. Use
//! [`LazyResolver`] when values are persisted in a [`KeyValueStore`] and should be fetched on
//! demand.
//!
//! [`Tree`]: crate::avl::tree::Tree
//! [`Node`]: crate::avl::node::Node

use std::rc::Rc;
use std::sync::Arc;
use std::sync::OnceLock;

use octez_riscv_data::components::bytes::Bytes;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
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

/// Identifier for a node that is always present.
#[derive(Debug, Clone, derive_more::From)]
#[from(Node<ArcTreeId, Normal>)]
pub struct ArcNodeId(Arc<Node<ArcTreeId, Normal>>);

impl Foldable<HashFold> for ArcNodeId {
    fn fold(&self, _builder: HashFold) -> <HashFold as Fold>::Folded {
        *self.0.hash()
    }
}

/// ID for a tree that is always present
#[derive(Debug, Clone, derive_more::From, Default)]
pub struct ArcTreeId(Tree<ArcNodeId>);

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
/// A [`LazyId`] may be in one of three states:
/// - hash-only, where `id` contains a hash and `inner` is empty,
/// - cached, where both `id` and `inner` are populated after immutable resolution, or
/// - owned, where `inner` is populated and `id` has been cleared after mutable resolution or
///   construction from an in-memory value.
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

/// Identifier for an AVL node.
#[derive(Debug, Clone)]
pub struct LazyNodeId(LazyId<Hash, Arc<Node<LazyTreeId, Normal>>>);

impl LazyNodeId {
    /// Wrap this lazy node identifier in a prove-mode identifier.
    ///
    /// The returned [`ProveNodeId`] keeps the original lazy identifier for hash lookups and
    /// storage-backed resolution, while leaving its prove-mode cache empty until a
    /// [`ProveResolver`] materialises the node.
    pub fn into_proof(self) -> ProveNodeId {
        ProveNodeId {
            node: self.clone(),
            inner: OnceLock::new(),
        }
    }
}

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

impl Foldable<HashFold> for LazyNodeId {
    fn fold(&self, _builder: HashFold) -> <HashFold as Fold>::Folded {
        if let Some(node) = self.0.inner.get() {
            return *node.hash();
        }

        self.0
            .id()
            .cloned()
            .expect("ID should be present when node is absent")
    }
}

/// Identifier for an AVL tree.
#[derive(Debug, Clone)]
pub struct LazyTreeId(LazyId<Hash, Tree<LazyNodeId>>);

impl LazyTreeId {
    /// Wrap this lazy tree identifier in a prove-mode identifier.
    ///
    /// The returned [`ProveTreeId`] keeps the original lazy identifier for hash lookups and
    /// storage-backed resolution, while deferring construction of the prove-mode tree until a
    /// [`ProveResolver`] resolves it.
    pub fn into_proof(self) -> ProveTreeId {
        ProveTreeId {
            tree: self.clone(),
            inner: OnceLock::new(),
        }
    }
}

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

impl Foldable<HashFold> for LazyTreeId {
    fn fold(&self, _builder: HashFold) -> <HashFold as Fold>::Folded {
        if let Some(tree) = self.0.inner.get() {
            return tree.hash();
        }

        self.0
            .id()
            .cloned()
            .expect("ID should be present when tree is absent")
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
    /// in a lazy node identifier. The canonical empty-tree hash resolves directly to default
    /// [`Tree`] without hitting storage.
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
            // SAFETY: This is a valid active `&mut Arc<_>` reference with no other
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

/// Identifier for a node resolved in [`Prove`] mode.
///
/// This wrapper keeps the original [`LazyNodeId`] to allow delegating hash computation and
/// storage access to a lazy resolver. Once resolved, it caches the prove-mode projection of the
/// node in `inner` so repeated accesses do not rebuild it.
#[derive(Clone)]
pub struct ProveNodeId {
    inner: OnceLock<Rc<Node<ProveTreeId, Prove<'static>>>>,
    node: LazyNodeId,
}

impl Foldable<HashFold> for ProveNodeId {
    fn fold(&self, builder: HashFold) -> <HashFold as Fold>::Folded {
        match self.inner.get() {
            Some(inner) => *inner.hash(),
            None => self.node.fold(builder),
        }
    }
}

/// Identifier for a tree resolved in [`Prove`] mode.
///
/// Like [`ProveNodeId`], this wrapper keeps the original lazy identifier and fills `inner` on the
/// first prove-mode resolution. The cached tree then serves subsequent reads without reprojecting
/// the lazy tree.
#[derive(Clone)]
pub struct ProveTreeId {
    inner: OnceLock<Tree<ProveNodeId>>,
    tree: LazyTreeId,
}

impl Foldable<HashFold> for ProveTreeId {
    fn fold(&self, builder: HashFold) -> <HashFold as Fold>::Folded {
        match self.inner.get() {
            Some(inner) => inner.hash(),
            None => self.tree.fold(builder),
        }
    }
}

/// Adapter that projects lazy AVL identifiers into prove-mode values on demand.
///
/// [`ProveResolver`] wraps another resolver for [`LazyNodeId`] and [`LazyTreeId`]. It preserves the
/// lazy resolver's hash behaviour, but caches prove-mode nodes and trees inside [`ProveNodeId`]
/// and [`ProveTreeId`] once they are resolved.
pub struct ProveResolver<R>(R);

impl<R: Resolver<LazyNodeId, Node<LazyTreeId, Normal>> + Resolver<LazyTreeId, Tree<LazyNodeId>>>
    Resolver<ProveNodeId, Node<ProveTreeId, Prove<'static>>> for ProveResolver<R>
{
    fn resolve<'b>(
        &self,
        id: &'b ProveNodeId,
    ) -> Result<&'b Node<ProveTreeId, Prove<'static>>, OperationalError> {
        if let Some(inner) = id.inner.get() {
            return Ok(inner);
        }

        let resolved: &Node<LazyTreeId, Normal> = self.0.resolve(&id.node)?;
        let result_node: Node<ProveTreeId, Prove<'static>> = resolved.clone().into_proof();
        id.inner
            .set(Rc::new(result_node))
            .map_err(|_| OperationalError::ResolverInvariantViolated)?;

        Ok(id.inner.wait())
    }

    fn resolve_mut<'b>(
        &mut self,
        id: &'b mut ProveNodeId,
    ) -> Result<&'b mut Node<ProveTreeId, Prove<'static>>, OperationalError> {
        {
            // SAFETY: Rust doesn't understand that the reference on `id.inner` is dropped on return.
            let inner_mut = unsafe { &mut *(&mut id.inner as *mut OnceLock<_>) };
            if let Some(inner) = inner_mut.get_mut() {
                let inner = Rc::make_mut(inner);
                return Ok(inner);
            }
        }

        let resolved: &Node<LazyTreeId, Normal> = self.0.resolve(&id.node)?;
        let result_node: Node<ProveTreeId, Prove<'static>> = resolved.clone().into_proof();
        id.inner
            .set(Rc::new(result_node))
            .map_err(|_| OperationalError::ResolverInvariantViolated)?;

        Ok(Rc::make_mut(
            id.inner.get_mut().expect("inner was just set"),
        ))
    }
}

impl<R: Resolver<LazyTreeId, Tree<LazyNodeId>>> Resolver<ProveTreeId, Tree<ProveNodeId>>
    for ProveResolver<R>
{
    fn resolve<'b>(&self, id: &'b ProveTreeId) -> Result<&'b Tree<ProveNodeId>, OperationalError> {
        if let Some(inner) = id.inner.get() {
            return Ok(inner);
        }

        let resolved: &Tree<LazyNodeId> = self.0.resolve(&id.tree)?;
        let result_tree: Tree<ProveNodeId> = resolved.clone().into_proof();
        id.inner
            .set(result_tree)
            .map_err(|_| OperationalError::ResolverInvariantViolated)?;

        Ok(id.inner.wait())
    }

    fn resolve_mut<'b>(
        &mut self,
        id: &'b mut ProveTreeId,
    ) -> Result<&'b mut Tree<ProveNodeId>, OperationalError> {
        {
            // SAFETY: Rust doesn't understand that the reference on `id.inner` is dropped on return.
            let inner_mut = unsafe { &mut *(&mut id.inner as *mut OnceLock<_>) };
            if let Some(inner) = inner_mut.get_mut() {
                return Ok(inner);
            }
        }

        let resolved: &Tree<LazyNodeId> = self.0.resolve(&id.tree)?;
        let result_tree: Tree<ProveNodeId> = resolved.clone().into_proof();
        id.inner
            .set(result_tree)
            .map_err(|_| OperationalError::ResolverInvariantViolated)?;

        Ok(id.inner.get_mut().expect("inner was just set"))
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
    use super::ProveNodeId;
    use super::ProveResolver;
    use super::Resolver;
    use crate::avl::resolver::AvlResolver;
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
        NodeId: Foldable<HashFold>,
        TreeId: Foldable<HashFold>,
        KV: KeyValueStore,
        Res: AvlResolver<NodeId, TreeId, Normal>,
    {
        // LazyTreeId resolves by loading a serialised optional root hash.
        let tree_hash = tree.hash();
        let tree_repr: Option<Hash> = tree.root().map(Hash::from_foldable);
        let tree_bytes =
            serialisation::serialise(tree_repr).expect("tree serialisation should succeed");

        persistence_layer
            .blob_set(tree_hash, tree_bytes)
            .expect("persisting trees should succeed");

        let Some(node_id) = tree.root() else {
            return;
        };

        let node = resolver
            .resolve(node_id)
            .expect("resolving nodes should succeed");

        let node_hash = Hash::from_foldable(node_id);
        let node_repr = node.to_encode();
        let node_bytes =
            serialisation::serialise(node_repr).expect("node serialisation should succeed");

        persistence_layer
            .blob_set(node_hash, node_bytes)
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
        let root_hash = Hash::from_foldable(tree.root().expect("tree should have a root node"));

        let persistence_layer = Arc::new(CountingKeyValueStore::default());
        persist_tree(&tree, &eager_resolver, persistence_layer.as_ref());

        let lazy_resolver = LazyResolver::new(persistence_layer.clone());
        let lazy_tree: LazyTreeId = LazyTreeId::from(tree_hash);

        assert_eq!(persistence_layer.blob_get_calls(), 0);
        assert_eq!(Hash::from_foldable(&lazy_tree), tree_hash);
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
        assert_eq!(Hash::from_foldable(&lazy_root), root_hash);
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

        let root_hash = Hash::from_foldable(tree.root().expect("tree should have a root node"));

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

        let root_hash = Hash::from_foldable(tree.root().expect("tree should have a root node"));

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

        let persisted_root_hash =
            Hash::from_foldable(original_tree.root().expect("tree should have a root node"));

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

    /// Helper: build a two-node tree (root key=2, left key=1), persist it,
    /// and return the root node hash, the eager tree, and the persistence
    /// layer for counting calls.
    fn setup_prove_fixture() -> (Hash, Tree<ArcNodeId>, Arc<CountingKeyValueStore>) {
        let root_key = Key::new(&[2]).expect("key should be valid");
        let left_key = Key::new(&[1]).expect("key should be valid");

        let mut tree: Tree<ArcNodeId> = Default::default();
        let mut resolver = ArcResolver;
        tree.set(&root_key, b"root", &mut resolver)
            .expect("set should succeed");
        tree.set(&left_key, b"left", &mut resolver)
            .expect("set should succeed");

        let root_hash = Hash::from_foldable(tree.root().expect("tree should have a root node"));

        let persistence_layer = Arc::new(CountingKeyValueStore::default());
        persist_tree(&tree, &resolver, persistence_layer.as_ref());

        (root_hash, tree, persistence_layer)
    }

    #[test]
    fn prove_resolver_resolves_node() {
        let (root_hash, _, persistence_layer) = setup_prove_fixture();
        let lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let lazy_resolver = LazyResolver::new(persistence_layer);

        let prove_resolver = ProveResolver(lazy_resolver);
        let lazy_node_id = lazy_tree.root().expect("tree should have a root");

        let prove_id = lazy_node_id.clone().into_proof();

        prove_resolver
            .resolve(&prove_id)
            .expect("should resolve to a prove node");
    }

    #[test]
    fn prove_resolver_hash_matches_lazy_resolver() {
        let (root_hash, _, persistence_layer) = setup_prove_fixture();
        let lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let lazy_resolver = LazyResolver::new(persistence_layer);
        let lazy_root = lazy_tree.root().expect("tree should have a root");

        let expected_hash = Hash::from_foldable(lazy_root);

        let prove_resolver = ProveResolver(lazy_resolver);
        let prove_id = lazy_root.clone().into_proof();

        // Hash before resolve (delegates to lazy path).
        let hash_before = Hash::from_foldable(&prove_id);
        assert_eq!(
            hash_before, expected_hash,
            "prove hash before resolve should match lazy hash"
        );

        // Resolve, then hash again (now computed from prove-mode inner).
        prove_resolver
            .resolve(&prove_id)
            .expect("resolve should succeed");

        let hash_after = Hash::from_foldable(&prove_id);
        assert_eq!(
            hash_after, expected_hash,
            "prove hash after resolve should still match lazy hash"
        );
    }

    #[test]
    fn prove_resolver_recursive_resolution() {
        let (root_hash, _, persistence_layer) = setup_prove_fixture();
        let lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let lazy_resolver = LazyResolver::new(persistence_layer);

        let prove_resolver = ProveResolver(lazy_resolver);
        let prove_root_id = lazy_tree
            .root()
            .expect("tree should have a root")
            .clone()
            .into_proof();

        // Resolve root node.
        let prove_node = prove_resolver
            .resolve(&prove_root_id)
            .expect("should resolve root prove node");

        // Resolve its left subtree.
        let left_tree: &Tree<ProveNodeId> = prove_node
            .left_ref(&prove_resolver)
            .expect("left subtree should resolve");
        let left_root_arc = left_tree.root().expect("left subtree should have a root");

        // Resolve the left child node.
        prove_resolver
            .resolve(left_root_arc)
            .expect("should resolve left prove node");

        // Resolve the right subtree — should be empty for a two-node tree
        // with keys [1, 2] where 2 is root and 1 is left.
        let right_tree: &Tree<ProveNodeId> = prove_node
            .right_ref(&prove_resolver)
            .expect("right subtree should resolve");
        assert!(right_tree.root().is_none(), "right subtree should be empty");
    }

    #[test]
    fn prove_resolver_caches_resolved_values() {
        let (root_hash, _, persistence_layer) = setup_prove_fixture();
        let lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let lazy_resolver = LazyResolver::new(persistence_layer.clone());

        let prove_resolver = ProveResolver(lazy_resolver);
        let prove_id = lazy_tree
            .root()
            .expect("tree should have a root")
            .clone()
            .into_proof();

        prove_resolver
            .resolve(&prove_id)
            .expect("first resolve should succeed");
        let calls_after_first = persistence_layer.blob_get_calls();

        prove_resolver
            .resolve(&prove_id)
            .expect("second resolve should use cache");
        assert_eq!(
            persistence_layer.blob_get_calls(),
            calls_after_first,
            "second resolve should not trigger additional storage loads"
        );
    }

    #[test]
    fn prove_resolver_resolve_mut_node() {
        let (root_hash, _, persistence_layer) = setup_prove_fixture();
        let lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let lazy_resolver = LazyResolver::new(persistence_layer);

        let mut prove_resolver = ProveResolver(lazy_resolver);
        let mut prove_id = lazy_tree
            .root()
            .expect("tree should have a root")
            .clone()
            .into_proof();

        let node = prove_resolver
            .resolve_mut(&mut prove_id)
            .expect("resolve_mut should succeed");

        // Verify we can read the key from the mutably resolved node.
        let root_key = Key::new(&[2]).expect("key should be valid");
        assert_eq!(node.key(), &root_key);
    }

    #[test]
    fn prove_resolver_tree_hash_matches_lazy_resolver() {
        let (root_hash, _, persistence_layer) = setup_prove_fixture();
        let lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(root_hash)).into();
        let lazy_resolver = LazyResolver::new(persistence_layer);

        let lazy_root = lazy_tree.root().expect("tree should have a root");

        // Get the lazy left child node hash via the lazy resolver.
        let node = lazy_resolver
            .resolve(lazy_root)
            .expect("resolve should succeed");
        let lazy_left_tree = node
            .left_ref(&lazy_resolver)
            .expect("left subtree should resolve");
        let lazy_left_id = lazy_left_tree
            .root()
            .expect("left subtree should have a root");
        let lazy_left_node_hash = Hash::from_foldable(lazy_left_id);

        // Now get the same hash via the prove resolver path.
        let prove_resolver = ProveResolver(lazy_resolver);
        let prove_root_id = lazy_root.clone().into_proof();
        let prove_node = prove_resolver
            .resolve(&prove_root_id)
            .expect("resolve should succeed");

        let left_prove_tree = prove_node
            .left_ref(&prove_resolver)
            .expect("left subtree should resolve");
        let left_prove_node_arc = left_prove_tree
            .root()
            .expect("left subtree should have a root");

        // Hash via the node resolver.
        let prove_left_node_hash = Hash::from_foldable(left_prove_node_arc);

        assert_eq!(
            prove_left_node_hash, lazy_left_node_hash,
            "prove tree child node hash should match lazy resolver hash"
        );
    }

    #[test]
    fn prove_resolver_empty_tree() {
        let persistence_layer = Arc::new(CountingKeyValueStore::default());
        let lazy_resolver = LazyResolver::new(persistence_layer);

        let empty_lazy_tree: LazyTreeId = LazyTreeId::default();
        let prove_tree_id = empty_lazy_tree.clone().into_proof();

        let prove_resolver = ProveResolver(lazy_resolver);
        let tree = prove_resolver
            .resolve(&prove_tree_id)
            .expect("resolving empty prove tree should succeed");

        assert!(tree.root().is_none(), "empty tree should have no root");
    }
}
