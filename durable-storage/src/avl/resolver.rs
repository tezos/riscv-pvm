// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for resolving data from identifiers of [`Tree`] and [`Node`] objects.
//!
//! [`Tree`]: crate::avl::tree::Tree
//! [`Node`]: crate::avl::node::Node

use std::sync::Arc;
use std::sync::OnceLock;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::serialisation::deserialise;
use trait_set::trait_set;

use super::node::Node;
use super::tree::Tree;
use crate::avl::node::NodeHashRepresentation;
use crate::avl::node::Value;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::storage::KeyValueStore;

/// Trait for resolving identifiers to values.
pub trait Resolver<Id, Value> {
    /// Retrieve the hash that belongs to an identifier.
    ///
    /// Depending on the implementation, this may compute or fetch the hash without resolving the
    /// full value.
    fn get_hash(&self, id: &Id) -> Hash;

    /// Resolve an identifier to a value.
    fn resolve<'a>(&self, id: &'a Id) -> Result<&'a Value, OperationalError>;

    /// Resolve an identifier to a mutable value.
    fn resolve_mut<'a>(&mut self, id: &'a mut Id) -> Result<&'a mut Value, OperationalError>;
}

trait_set! {
    /// Specialised [`Resolver`] for MAVL nodes
    pub trait NodeResolver<NodeId, TreeId> = Resolver<NodeId, Node<TreeId>>;

    /// Specialised [`Resolver`] for MAVL trees
    pub trait TreeResolver<NodeId, TreeId> = Resolver<TreeId, Tree<NodeId>>;

    /// Specialised [`Resolver`] for MAVL nodes and trees
    pub trait AvlResolver<NodeId, TreeId> = NodeResolver<NodeId, TreeId> + TreeResolver<NodeId, TreeId>;
}

/// Identifier for a node that is always present.
#[derive(Debug, Clone, derive_more::From)]
#[from(Node<ArcTreeId>)]
pub struct ArcNodeId(Arc<Node<ArcTreeId>>);

/// ID for a tree that is always present
#[derive(Debug, Clone, derive_more::From, Default)]
pub struct ArcTreeId(Tree<ArcNodeId>);

/// Provide values identified by an [`Arc`].
#[derive(Clone, Debug)]
pub struct ArcResolver;

impl Resolver<ArcNodeId, Node<ArcTreeId>> for ArcResolver {
    fn get_hash(&self, id: &ArcNodeId) -> Hash {
        *id.0.hash(self)
    }

    fn resolve<'a>(&self, id: &'a ArcNodeId) -> Result<&'a Node<ArcTreeId>, OperationalError> {
        Ok(id.0.as_ref())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut ArcNodeId,
    ) -> Result<&'a mut Node<ArcTreeId>, OperationalError> {
        Ok(Arc::make_mut(&mut id.0))
    }
}

impl Resolver<ArcTreeId, Tree<ArcNodeId>> for ArcResolver {
    fn get_hash(&self, id: &ArcTreeId) -> Hash {
        id.0.hash(self)
    }

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

/// Identifier that can be resolved lazily and cached after first load.
///
/// We assume the invariant that if the identifier is not available,
/// the value must be available, and vice versa.
#[derive(Default, Debug, Clone)]
pub struct LazyId<Id, Value> {
    inner: OnceLock<Value>,
    id: Option<Id>,
}

impl<Value> From<Hash> for LazyId<Hash, Value> {
    fn from(hash: Hash) -> Self {
        Self {
            inner: OnceLock::new(),
            id: Some(hash),
        }
    }
}

impl<Id, Value> LazyId<Id, Value> {
    /// Create an identifier from an already loaded value.
    pub fn new(value: Value) -> Self {
        let id = Self {
            inner: OnceLock::from(value),
            id: None,
        };
        id
    }

    /// Return the identifier if available.
    fn id(&self) -> Option<&Id> {
        self.id.as_ref()
    }
}

/// Identifier for an AVL node.
#[derive(Debug, Clone)]
pub struct LazyNodeId(LazyId<Hash, Arc<Node<LazyTreeId>>>);

impl From<Node<LazyTreeId>> for LazyNodeId {
    fn from(value: Node<LazyTreeId>) -> Self {
        let value = Arc::new(value);
        Self(LazyId::new(value))
    }
}

impl From<Hash> for LazyNodeId {
    fn from(hash: Hash) -> LazyNodeId {
        LazyNodeId(hash.into())
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

/// Resolver that lazily loads AVL nodes from a key-value store.
#[derive(Clone, Debug)]
pub struct LazyResolver<KV> {
    persistence_layer: Arc<KV>,
}

impl<KV> LazyResolver<KV> {
    /// Create a resolver backed by the given persistence layer.
    pub fn new(persistence_layer: Arc<KV>) -> Self {
        Self { persistence_layer }
    }
}

impl<KV: KeyValueStore> LazyResolver<KV> {
    fn load_node(&self, hash: Hash) -> Result<Arc<Node<LazyTreeId>>, OperationalError> {
        let bytes = self
            .persistence_layer
            .blob_get(hash)
            .map_err(|_| OperationalError::Resolver)?;
        let noderepr = deserialise::<NodeHashRepresentation<Value, Key, Hash>>(bytes.as_ref())?;
        Ok(Arc::new(Node::from(noderepr)))
    }

    fn load_tree(&self, hash: Hash) -> Result<Tree<LazyNodeId>, OperationalError> {
        let bytes = self
            .persistence_layer
            .blob_get(hash)
            .map_err(|_| OperationalError::Resolver)?;
        let tree_repr = deserialise::<Option<Hash>>(bytes.as_ref())?.map(LazyNodeId::from);
        Ok(Tree::from(tree_repr))
    }
}

impl<KV: KeyValueStore> Resolver<LazyNodeId, Node<LazyTreeId>> for LazyResolver<KV> {
    fn get_hash<'a>(&self, id: &'a LazyNodeId) -> Hash {
        match id.0.inner.get() {
            Some(value) => *value.hash(self),
            None => *id.0.id().expect("Hash must be present if value is not."),
        }
    }

    fn resolve<'a>(&self, id: &'a LazyNodeId) -> Result<&'a Node<LazyTreeId>, OperationalError> {
        if let Some(value) = id.0.inner.get() {
            return Ok(value);
        }
        let &hash = id.0.id().ok_or(OperationalError::Resolver)?;
        let node = self.load_node(hash)?;
        let _ = id.0.inner.set(node);
        Ok(id.0.inner.wait().as_ref())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut LazyNodeId,
    ) -> Result<&'a mut Node<LazyTreeId>, OperationalError> {
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

        let hash = id.0.id().ok_or(OperationalError::Resolver)?;
        let _ = id.0.inner.set(self.load_node(*hash)?);

        id.0.id = None;
        id.0.inner
            .get_mut()
            .ok_or(OperationalError::Resolver)
            .map(Arc::make_mut)
    }
}

impl<KV: KeyValueStore> Resolver<LazyTreeId, Tree<LazyNodeId>> for LazyResolver<KV> {
    fn get_hash<'a>(&self, id: &'a LazyTreeId) -> Hash {
        match id.0.inner.get() {
            Some(value) => Hash::from(*value.hash(self)),
            None => *id.0.id().expect("Hash must be present if value is not."),
        }
    }

    fn resolve<'a>(&self, id: &'a LazyTreeId) -> Result<&'a Tree<LazyNodeId>, OperationalError> {
        if let Some(value) = id.0.inner.get() {
            return Ok(value);
        }
        let &hash = id.0.id().ok_or(OperationalError::Resolver)?;
        let tree = self.load_tree(hash)?;
        let _ = id.0.inner.set(tree);
        Ok(id.0.inner.wait())
    }

    fn resolve_mut<'a>(
        &mut self,
        id: &'a mut LazyTreeId,
    ) -> Result<&'a mut Tree<LazyNodeId>, OperationalError> {
        if id.0.inner.get().is_none() {
            let hash = id.0.id().ok_or(OperationalError::Resolver)?;
            let _ = id.0.inner.set(self.load_tree(*hash)?);
        }

        id.0.id = None;
        id.0.inner.get_mut().ok_or(OperationalError::Resolver)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::HashedData;
    use octez_riscv_data::serialisation;

    use super::ArcNodeId;
    use super::ArcResolver;
    use super::LazyNodeId;
    use super::LazyResolver;
    use super::Resolver;
    use crate::avl::tree::Tree;
    use crate::key::Key;
    use crate::storage::KeyValueStore;
    use crate::storage::in_memory::InMemoryKeyValueStore;

    fn persist_tree(tree: &Tree<ArcNodeId>, persistence_layer: &InMemoryKeyValueStore) {
        fn persist_subtree(
            tree: &Tree<ArcNodeId>,
            resolver: &ArcResolver,
            persistence_layer: &InMemoryKeyValueStore,
        ) {
            // LazyTreeId resolves by loading a serialised optional root hash.
            let tree_repr: Option<Hash> = tree.root().map(|node_id| resolver.get_hash(node_id));
            let tree_bytes =
                serialisation::serialise(tree_repr).expect("tree serialisation should succeed");
            persistence_layer
                .blob_set(HashedData::from_data(tree_bytes))
                .expect("persisting trees should succeed");

            let Some(node_id) = tree.root() else {
                return;
            };

            let node = resolver
                .resolve(node_id)
                .expect("resolving nodes should succeed");
            let encoded = node.to_encode(resolver);
            let node_bytes =
                serialisation::serialise(encoded).expect("node serialisation should succeed");
            persistence_layer
                .blob_set(HashedData::from_data(node_bytes))
                .expect("persisting nodes should succeed");

            persist_subtree(
                node.left_ref(resolver)
                    .expect("left subtree should resolve"),
                resolver,
                persistence_layer,
            );
            persist_subtree(
                node.right_ref(resolver)
                    .expect("right subtree should resolve"),
                resolver,
                persistence_layer,
            );
        }

        let resolver = ArcResolver;
        persist_subtree(tree, &resolver, persistence_layer);
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

        let initial_tree_hash: Hash = original_tree.hash(&original_resolver);

        let persisted_root_hash = Hash::from(
            original_tree
                .root()
                .map(|root_id| *original_resolver.get_hash(root_id))
                .expect("tree should have a root node"),
        );

        let persistence_layer = Arc::new(InMemoryKeyValueStore::default());
        persist_tree(&original_tree, persistence_layer.as_ref());

        let mut lazy_tree: Tree<LazyNodeId> = Some(LazyNodeId::from(persisted_root_hash)).into();
        let mut lazy_resolver = LazyResolver::new(persistence_layer);

        lazy_tree
            .set(&left_key, b"left-mutated", &mut lazy_resolver)
            .expect("set should succeed");
        let hash_after_mutation = lazy_tree.hash(&lazy_resolver);

        let mut expected_tree = original_tree.clone();
        let mut expected_resolver = ArcResolver;
        expected_tree
            .set(&left_key, b"left-mutated", &mut expected_resolver)
            .expect("set should succeed");
        let expected_hash = expected_tree.hash(&expected_resolver);

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
