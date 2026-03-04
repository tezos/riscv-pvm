// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! A Merkleised key-value store layer.
//!
//! [`MerkleLayer`] wraps a [`KeyValueStore`] (KV) and duplicates all stored data in an AVL
//! [`Tree`]. When [`MerkleLayer::commit`] is called, the tree is serialised and stored in the KV
//! and the root hash of the tree is used to identify that commitment of the layer as a
//! [`CommitId`]. The inverse operation, [`MerkleLayer::checkout`], takes a [`CommitId`] and
//! reconstructs the tree from the KV.
//!
//! `M` is an implementation of the PVM's operational [`Mode`].
//!
//! [`MerkleLayer::try_clone_with`] enables forking snapshots. Clones share the underlying tree
//! cheaply via an `Arc` and diverge upon mutation, using copy-on-write (CoW) semantics.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::Arc;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashedData;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::serialisation;
use perfect_derive::perfect_derive;

use crate::avl::resolver::ArcNodeId;
use crate::avl::resolver::ArcResolver;
use crate::avl::resolver::Resolver;
use crate::avl::tree::Tree;
use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::key::Key;
use crate::storage::KeyValueStore;
use crate::storage::PersistentKeyValueStore;

/// A layer for transforming data into a Merkle-ised representation before commitment to a
/// [`PersistentKeyValueStore`].
#[perfect_derive(Debug)]
pub struct MerkleLayer<KV, M: Mode> {
    inner: M::Select<MerkleLayerTemplate<KV>>,
}

impl<KV> MerkleLayer<KV, Normal> {
    /// Create a new, empty Merkle layer that will commit to the provided persistence layer.
    pub fn new(persistence: Arc<KV>) -> Self {
        MerkleLayer {
            inner: NormalImpl::new(persistence),
        }
    }

    /// Load the Merkle layer from the given key-value store.
    pub fn checkout(persistence: Arc<KV>, root: CommitId) -> Result<Self, OperationalError>
    where
        KV: KeyValueStore,
    {
        Ok(MerkleLayer {
            inner: NormalImpl::checkout(persistence, root)?,
        })
    }

    /// Generates a commitment for the [MerkleLayer].
    pub fn commit(&mut self) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        self.inner.commit()
    }
}

impl<KV, M: MerkleLayerMode> MerkleLayer<KV, M> {
    /// Clone the Merkle layer. The new layer will commit to the provided persistence layer.
    pub fn try_clone_with(&self, persistence: Arc<KV>) -> Self {
        M::try_clone_with(self, persistence)
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub fn hash(&mut self) -> Result<Hash, OperationalError> {
        M::hash(self)
    }

    /// Delete the data associated with a given [Key].
    pub fn delete(&mut self, key: &Key) -> Result<(), OperationalError> {
        M::delete(self, key)
    }

    /// Sets the data associated with a given [Key].
    pub fn set(&mut self, key: &Key, data: &[u8]) -> Result<(), OperationalError> {
        M::set(self, key, data)
    }

    /// Writes the data to the node associated with a given [Key] with the given offset.
    pub fn write(&mut self, key: &Key, offset: usize, data: &[u8]) -> Result<(), Error> {
        M::write(self, key, offset, data)
    }
}

/// Modes that implements this trait support Merkle layer operations
pub trait MerkleLayerMode: Mode {
    /// See [`MerkleLayer::try_clone_with`]
    fn try_clone_with<KV>(
        this: &MerkleLayer<KV, Self>,
        persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self>;

    /// See [`MerkleLayer::hash`]
    fn hash<KV>(this: &mut MerkleLayer<KV, Self>) -> Result<Hash, OperationalError>;

    /// See [`MerkleLayer::delete`]
    fn delete<KV>(this: &mut MerkleLayer<KV, Self>, key: &Key) -> Result<(), OperationalError>;

    /// See [`MerkleLayer::set`]
    fn set<KV>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        data: &[u8],
    ) -> Result<(), OperationalError>;

    /// See [`MerkleLayer::write`]
    fn write<KV>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        offset: usize,
        data: &[u8],
    ) -> Result<(), Error>;
}

impl MerkleLayerMode for Normal {
    fn try_clone_with<KV>(
        this: &MerkleLayer<KV, Self>,
        persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self> {
        MerkleLayer {
            inner: this.inner.try_clone_with(persistence),
        }
    }

    fn hash<KV>(this: &mut MerkleLayer<KV, Self>) -> Result<Hash, OperationalError> {
        this.inner.hash()
    }

    fn delete<KV>(this: &mut MerkleLayer<KV, Self>, key: &Key) -> Result<(), OperationalError> {
        this.inner.delete(key)
    }

    fn set<KV>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        data: &[u8],
    ) -> Result<(), OperationalError> {
        this.inner.set(key, data)
    }

    fn write<KV>(
        this: &mut MerkleLayer<KV, Self>,
        key: &Key,
        offset: usize,
        data: &[u8],
    ) -> Result<(), Error> {
        this.inner.write(key, offset, data)
    }
}

struct MerkleLayerTemplate<KV>(PhantomData<KV>, Infallible);

impl<KV> Modal for MerkleLayerTemplate<KV> {
    type Normal = NormalImpl<KV>;

    type Prove<'normal> = Infallible;

    type Verify = Infallible;
}

#[derive(Debug)]
struct NormalImpl<KV> {
    tree: Tree<ArcNodeId>,
    persistence: Arc<KV>,
    resolver: ArcResolver,
}

impl<KV> NormalImpl<KV> {
    /// Create a new, empty Merkle layer that will commit to the provided persistence layer.
    fn new(persistence: Arc<KV>) -> Self {
        NormalImpl {
            tree: Tree::default(),
            persistence,
            resolver: ArcResolver,
        }
    }

    /// Clone the Merkle layer. The new layer will commit to the provided persistence layer.
    fn try_clone_with(&self, persistence: Arc<KV>) -> Self {
        Self {
            tree: self.tree.clone(),
            persistence,
            resolver: ArcResolver,
        }
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    fn hash(&mut self) -> Result<Hash, OperationalError> {
        self.tree.hash(&self.resolver)
    }

    /// Delete the data associated with a given [Key].
    fn delete(&mut self, key: &Key) -> Result<(), OperationalError> {
        self.tree.delete(key, &mut self.resolver)?;
        Ok(())
    }

    /// Sets the data associated with a given [Key].
    fn set(&mut self, key: &Key, data: &[u8]) -> Result<(), OperationalError> {
        self.tree.set(key, data, &mut self.resolver)?;
        Ok(())
    }

    /// Writes the data to the node associated with a given [Key] with the given offset.
    fn write(&mut self, key: &Key, offset: usize, data: &[u8]) -> Result<(), Error> {
        self.tree.write(key, offset, data, &mut self.resolver)?;
        Ok(())
    }

    /// Load the Merkle layer from the given key-value store.
    fn checkout(_persistence: Arc<KV>, _root: CommitId) -> Result<Self, OperationalError> {
        todo!("RV-862: implement checkout")
    }

    /// Generates a commitment for the [MerkleLayer].
    fn commit(&mut self) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        // Note that although we're doing in order
        // iteration of the nodes the hashes are
        // calculated during the encoding of the node
        // if necessary.
        for node in self.tree.iter(&self.resolver) {
            let node = self.resolver.resolve(node)?;
            let encoded = node.to_encode(&self.resolver);
            let value = serialisation::serialise(encoded)
                .expect("Serialisation of node data should not fail");
            let blob = HashedData::from_data(value);
            self.persistence.blob_set(blob)?;
        }

        Ok(CommitId::from(self.hash()?))
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_test_utils::TestableTmpdir;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::MerkleLayer;
    use crate::avl::node::Node;
    use crate::avl::node::Value;
    use crate::avl::resolver::ArcNodeId;
    use crate::avl::tree::Tree;
    use crate::errors::OperationalError;
    use crate::key::Key;
    use crate::repo::DirectoryManager;
    use crate::storage::KeyValueStore;
    use crate::storage::TestKeyValueStore;

    impl<KV> MerkleLayer<KV, Normal> {
        fn tree(&self) -> &Tree<ArcNodeId> {
            &self.inner.tree
        }

        /// Clear all data from the [MerkleLayer].
        fn clear(&mut self) {
            self.inner.tree.take();
        }

        /// Returns an immutable reference to the data stored for a given [Key].
        pub fn get(&mut self, key: &Key) -> Result<Option<&Value>, OperationalError> {
            self.inner.tree.get(key, &self.inner.resolver)
        }
    }

    fn new_merkle_layer() -> MerkleLayer<TestKeyValueStore, Normal> {
        let tmpdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");

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

        let mut ml = new_merkle_layer();

        for i in 0..keys.len() {
            ml.set(&keys[i], &data[i])
                .expect("setting node should succeed");
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
        }

        let mut ml2 = ml.try_clone_with(ml.inner.persistence.clone());
        let original_hash = ml.hash().expect("hash operation should succeed.");
        assert_eq!(
            original_hash,
            ml2.hash().expect("hash operation should succeed.")
        );

        let cow_data = "🐮<(moo!)";
        ml2.set(&keys[0], cow_data.as_bytes())
            .expect("setting node should succeed");
        assert_ne!(
            original_hash,
            ml2.hash().expect("hash operation should succeed.")
        );
        assert_eq!(
            original_hash,
            ml.hash().expect("hash operation should succeed.")
        );

        let old_node1: Node<ArcNodeId> =
            Node::new(keys[0].clone(), Bytes::copy_from_slice(&data[0]));
        let new_node1: Node<ArcNodeId> = Node::new(keys[0].clone(), cow_data.as_bytes());

        let node2: Node<ArcNodeId> = Node::new(keys[1].clone(), Bytes::copy_from_slice(&data[1]));
        let node3: Node<ArcNodeId> = Node::new(keys[2].clone(), Bytes::copy_from_slice(&data[2]));

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
            let data1 = Bytes::from("property");
            let data2 = Bytes::from("cow");
            let mut ml = new_merkle_layer();

            // Set all the keys in the tree
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data1).expect("setting node should succeed");
            }

            // Create a cheap copy
            let original_hash = ml.hash().expect("hash operation should succeed.");
            let mut ml2 = ml.try_clone_with(ml.inner.persistence.clone());
            prop_assert_eq!(original_hash, ml2.hash().expect("hash operation should succeed."));

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
                prop_assert_eq!(original_hash, ml2.hash().expect("hash operation should succeed."));
            } else {
                prop_assert_ne!(original_hash, ml2.hash().expect("hash operation should succeed."));
            }
            prop_assert_eq!(original_hash, ml.hash().expect("hash operation should succeed."));

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
        let data = Bytes::from("create");
        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash().expect("hash operation should succeed.");
        ml.set(&key, &data).expect("setting node should succeed");
        assert_ne!(
            empty_hash,
            ml.hash().expect("hash operation should succeed.")
        );

        let node: Node<ArcNodeId> = Node::new(key.clone(), data);
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
        let data = Bytes::from("old");
        let data2 = Bytes::from("new");
        let mut ml = new_merkle_layer();
        ml.set(&key, &data).expect("setting node should succeed");
        let old_hash = ml.hash().expect("hash operation should succeed.");

        let node: Node<ArcNodeId> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());

        ml.set(&key, &data2).expect("setting node should succeed");
        assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
        assert!(
            ml.inner
                .tree
                .is_inorder(&ml.inner.resolver)
                .expect("The tree should be retrieved successfully."),
            "AVL isn't in order: {ml:?}"
        );
        let node: Node<ArcNodeId> = Node::new(key.clone(), data2);
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
            Bytes::from("255, 0"),
            Bytes::from("0"),
            Bytes::from("0, 0"),
            Bytes::from("0, 0, 0"),
        ];

        let mut ml = new_merkle_layer();

        for (key, data) in keys.iter().zip(data.iter()) {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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

        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash().expect("hash operation should succeed.");

        // Left imbalance
        let data = Bytes::from("imbalanced left");
        for key in keys.iter() {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
        }

        ml.clear();
        assert_eq!(
            empty_hash,
            ml.hash().expect("hash operation should succeed.")
        );

        let keys = {
            let mut keys = keys;
            keys.sort();
            keys
        };

        // Right imbalance
        let data = Bytes::from("imbalanced right");
        for key in keys.iter() {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
        }
    }

    #[test]
    fn test_mavl_create_left_right() {
        let keys = [Key::new(&[2]), Key::new(&[0]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = Bytes::from("left_right");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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

        let data = Bytes::from("right_left");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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

        let data = Bytes::from("right_left");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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

        let data = Bytes::from("right_left");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            let old_hash = ml.hash().expect("hash operation should succeed.");
            ml.set(key, &data).expect("setting node should succeed");
            assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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
            let data = Bytes::from("property");
            let mut ml = new_merkle_layer();
            let old_hash = ml.hash().expect("hash operation should succeed.");

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data).expect("setting node should succeed");
            }

            if !keys.is_empty() {
                assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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
        let data = Bytes::from("delete");
        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash().expect("hash operation should succeed.");
        ml.set(&key, &data).expect("setting node should succeed");
        let full_hash = ml.hash().expect("hash operation should succeed.");

        ml.delete(&key).expect("deleting node should succeed.");
        assert_ne!(
            full_hash,
            ml.hash().expect("hash operation should succeed.")
        );
        assert_eq!(
            empty_hash,
            ml.hash().expect("hash operation should succeed.")
        );

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
            let data = Bytes::from("delete_prop");
            let mut ml = new_merkle_layer();
            let empty_hash = ml.hash().expect("hash operation should succeed.");

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data).expect("setting node should succeed");
            }

            if !keys.is_empty() {
                prop_assert_ne!(empty_hash, ml.hash().expect("hash operation should succeed."));
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.delete(&key).expect("delete should succeed.");
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully."), None);
            }

            prop_assert_eq!(empty_hash, ml.hash().expect("hash operation should succeed."));

            ml.tree().check(&ml.inner.resolver).expect("the tree should be retrieved successfully.");
        }
    }

    fn test_mavl_delete_keys(keys: &[Key]) {
        let data = Bytes::from("delete");

        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash().expect("hash operation should succeed.");

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
            assert_ne!(
                empty_hash,
                ml.hash().expect("hash operation should succeed.")
            );
        }

        for key in keys.iter() {
            ml.delete(key).expect("deleting node should succeed.");
            ml.tree()
                .check(&ml.inner.resolver)
                .expect("the tree should be retrieved successfully.");
            ml.delete(key).expect("deleting node should succeed.");
            assert_eq!(ml.get(key).expect("The data should exist."), None);
        }

        assert_eq!(
            empty_hash,
            ml.hash().expect("hash operation should succeed.")
        );
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
        let data = Bytes::from("write_new_value");
        let mut ml = new_merkle_layer();
        let old_hash = ml.hash().expect("hash operation should succeed.");
        ml.write(&key, 0, &data).expect("write should succeed.");

        let node: Node<ArcNodeId> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());
        assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
        ml.tree()
            .check(&ml.inner.resolver)
            .expect("The tree should be retrieved successfully.");
    }

    #[test]
    fn test_mavl_write_no_truncation() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("a long value");
        let data2 = Bytes::from("good");
        let mut ml = new_merkle_layer();
        ml.set(&key, &data).expect("setting node should succeed");
        let old_hash = ml.hash().expect("hash operation should succeed.");

        let data_len = data.len();
        let node: Node<ArcNodeId> = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully")
            .expect("The data should exist.");

        assert_eq!(&get_node, &node.data());

        ml.write(&key, 2, &data2).expect("write should succeed.");
        assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
        assert!(
            ml.inner
                .tree
                .is_inorder(&ml.inner.resolver)
                .expect("The tree should be retrieved successfully."),
            "AVL isn't in order: {ml:?}"
        );
        let node: Node<ArcNodeId> = Node::new(key.clone(), Bytes::from("a good value"));
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
            let data = Bytes::from(vec![0; 500]);
            let alternating = Bytes::from([1, 0]
                .iter()
                .cycle()
                .take(500)
                .cloned()
                .collect::<Vec<_>>());

            let mut ml = new_merkle_layer();
            let old_hash = ml.hash().expect("hash operation should succeed.");

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data).expect("setting node should succeed");
                for offset in 0..250 {
                    ml.write(&key, offset * 2, &[1]).expect("write should succeed.");
                }
            }

            if !keys.is_empty() {
                assert_ne!(old_hash, ml.hash().expect("hash operation should succeed."));
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

    /// - Add some data to the Merkle layer.
    /// - Commit the data to relevant column family
    /// - Check whether the data is persisted.
    /// - Check whether the hash contained in the commit id
    ///   is the same as the root hash
    #[cfg(feature = "rocksdb")]
    #[test]
    fn test_merkle_layer_commit_persists_nodes() {
        use crate::avl::resolver::Resolver;

        let mut merkle_layer = new_merkle_layer();

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
            Bytes::from_static(b"aasd"),
            Bytes::from_static(b"aksdja"),
            Bytes::from_static(b"agfgd"),
            Bytes::from_static(b"45gfgdf"),
            Bytes::from_static(b"sfdsdfsd"),
            Bytes::from_static(b"asdfsfd"),
            Bytes::from_static(b"asdfsdf"),
        ];

        for (key, data_elem) in keys.iter().zip(data.iter()) {
            merkle_layer
                .set(key, data_elem)
                .expect("setting node should succeed");
        }

        let commit_id = merkle_layer
            .commit()
            .expect("The commit operation should not fail");

        for node in merkle_layer.inner.tree.iter(&merkle_layer.inner.resolver) {
            let node = merkle_layer.inner.resolver.resolve(node).unwrap();
            let encoded = node.to_encode(&merkle_layer.inner.resolver);
            let serialised = octez_riscv_data::serialisation::serialise(encoded)
                .expect("We should be able to serialise the node");
            let node_hash = *node.hash(&merkle_layer.inner.resolver);
            let blob = merkle_layer
                .inner
                .persistence
                .blob_get(node_hash)
                .expect("The blob with the given key should be present");
            assert_eq!(serialised, blob.as_ref());
        }

        let root_hash = merkle_layer
            .inner
            .tree
            .hash(&merkle_layer.inner.resolver)
            .expect("Resolving the node should succeed.");
        assert_eq!(*commit_id.as_hash(), root_hash);
    }
}
