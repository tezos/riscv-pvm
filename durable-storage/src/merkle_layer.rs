// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
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

use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use crate::avl::resolver::LazyNodeId;
use crate::avl::resolver::LazyResolver;
use crate::avl::resolver::ProveNodeId;
use crate::avl::resolver::ProveResolver;
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

impl MerkleLayerMode for Prove<'static> {
    fn try_clone_with<KV: KeyValueStore>(
        this: &MerkleLayer<KV, Self>,
        persistence: Arc<KV>,
    ) -> MerkleLayer<KV, Self> {
        MerkleLayer {
            inner: ProveImpl {
                tree: this.inner.tree.clone(),
                resolver: ProveResolver::start(LazyResolver::new(persistence)),
            },
        }
    }

    fn hash<KV>(this: &MerkleLayer<KV, Self>) -> Hash {
        this.inner.tree.hash()
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
struct ProveImpl<KV> {
    tree: Tree<ProveNodeId>,
    resolver: ProveResolver<LazyResolver<KV>>,
}

#[derive(Debug)]
struct VerifyImpl {
    tree: Tree<VerifyNodeId>,
    resolver: VerifyResolver,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use octez_riscv_data::components::bytes::Bytes;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::mode::utils::catch_not_found;
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
    use crate::storage::KeyValueStore;
    use crate::storage::PersistentKeyValueStore;
    use crate::storage::kv_test;

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

    impl<KV: KeyValueStore> MerkleLayer<KV, Prove<'static>> {
        fn get(&self, key: &Key) -> Result<Option<&Bytes<Prove<'static>>>, OperationalError> {
            self.inner.tree.get(key, &self.inner.resolver)
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

    fn new_merkle_layer<KV: KeyValueStore>(repo: &KV::Repo) -> MerkleLayer<KV, Normal> {
        let persistence_layer = KV::new(repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        MerkleLayer::new(persistence_layer)
    }

    kv_test!(test_mavl_cow, KV, {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = [vec![0; 0], vec![13; 5], vec![42; 129]];

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);

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
    });

    kv_test!(test_mavl_cow_prop, KV, {
        proptest!(|(keys1 in prop::collection::vec(any::<[u8; 2]>(), 0..500), keys2 in prop::collection::vec(any::<[u8; 2]>(), 0..500))| {
            let data1 = bytes::Bytes::from("property");
            let data2 = bytes::Bytes::from("cow");

            let (_keepalive, repo) = KV::setup_repo();
            let mut ml = new_merkle_layer::<KV>(&repo);

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
        });
    });

    kv_test!(test_mavl_create, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("create");
        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);
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
    });

    kv_test!(test_mavl_create_existing, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("old");
        let data2 = bytes::Bytes::from("new");
        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);
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
    });

    kv_test!(test_mavl_create_heterogenous_key, KV, {
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

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);

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
    });

    kv_test!(test_mavl_create_imbalanced, KV, {
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

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);
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
    });

    kv_test!(test_mavl_create_left_right, KV, {
        let keys = [Key::new(&[2]), Key::new(&[0]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = bytes::Bytes::from("left_right");

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);

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
    });

    kv_test!(test_mavl_create_right_left, KV, {
        let keys = [Key::new(&[0]), Key::new(&[2]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = bytes::Bytes::from("right_left");

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);

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
    });

    kv_test!(test_mavl_create_right_left_nonzero_node_bf, KV, {
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

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);

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
    });

    kv_test!(test_mavl_create_left_right_nonzero_node_bf, KV, {
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

        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);

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
    });

    kv_test!(test_mavl_create_prop, KV, {
        proptest!(|(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500))| {
            let data = bytes::Bytes::from("property");
            let (_keepalive, repo) = KV::setup_repo();
            let mut ml = new_merkle_layer::<KV>(&repo);
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
        });
    });

    kv_test!(test_mavl_delete, KV, {
        let key = Key::new(&[1]).expect("Sizes less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("delete");
        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);
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
    });

    kv_test!(test_mavl_delete_prop, KV, {
        proptest!(|(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500))| {
            let data = bytes::Bytes::from("delete_prop");
            let (_keepalive, repo) = KV::setup_repo();
            let mut ml = new_merkle_layer::<KV>(&repo);
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
        });
    });

    fn test_mavl_delete_keys<KV: KeyValueStore>(repo: &KV::Repo, keys: &[Key]) {
        let data = bytes::Bytes::from("delete");

        let mut ml = new_merkle_layer::<KV>(repo);
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
    kv_test!(test_mavl_delete_rebalance_needed, KV, {
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
        let (_keepalive, repo) = KV::setup_repo();
        test_mavl_delete_keys::<KV>(&repo, &keys);
    });

    // Requires replacing a deleted node with a successor that is its right child.
    //
    //      BEFORE       AFTER
    //        1x           2
    //       / \          / \
    //      0   2        0   3
    //           \
    //            3
    kv_test!(test_mavl_delete_right_successor, KV, {
        let keys = [
            Key::new(&[1]),
            Key::new(&[2]),
            Key::new(&[0]),
            Key::new(&[3]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        let (_keepalive, repo) = KV::setup_repo();
        test_mavl_delete_keys::<KV>(&repo, &keys);
    });

    // Requires replacing a deleted node with a successor that is its right child and has a right
    // child of its own.
    //
    //      BEFORE       AFTER
    //        4x           5
    //       / \          / \
    //      1   5        1   6
    //     /     \      /
    //    0       6    0
    kv_test!(test_mavl_delete_successor_right_child, KV, {
        let keys = [
            Key::new(&[4]),
            Key::new(&[5]),
            Key::new(&[1]),
            Key::new(&[6]),
            Key::new(&[0]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        let (_keepalive, repo) = KV::setup_repo();
        test_mavl_delete_keys::<KV>(&repo, &keys);
    });

    // Requires replacing a deleted node with a successor that isn't its right child.
    //
    //      BEFORE    AFTER
    //        1x        2
    //       / \       / \
    //      0   3     0   3
    //         /
    //        2
    kv_test!(test_mavl_delete_take_min, KV, {
        let keys = [
            Key::new(&[1]),
            Key::new(&[3]),
            Key::new(&[0]),
            Key::new(&[2]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        let (_keepalive, repo) = KV::setup_repo();
        test_mavl_delete_keys::<KV>(&repo, &keys);
    });

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
    kv_test!(test_mavl_delete_take_min_recursive, KV, {
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
        let (_keepalive, repo) = KV::setup_repo();
        test_mavl_delete_keys::<KV>(&repo, &keys);
    });

    // Requires rebalancing a node where the balance factor is -2 and the left child's balance
    // factor is 0:
    //      BEFORE      DELETED     ROTATED
    //        4x           5            1
    //       / \          /           /  \
    //      1   5        1           3    5
    //     / \          / \         /
    //    0   3        0   3       0
    kv_test!(test_mavl_delete_zero_double_rotation_balance_factor, KV, {
        let keys = [
            Key::new(&[4]),
            Key::new(&[0]),
            Key::new(&[5]),
            Key::new(&[1]),
            Key::new(&[3]),
        ]
        .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));
        let (_keepalive, repo) = KV::setup_repo();
        test_mavl_delete_keys::<KV>(&repo, &keys);
    });

    kv_test!(test_mavl_write_new_value, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("write_new_value");
        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);
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
    });

    kv_test!(test_mavl_write_no_truncation, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = bytes::Bytes::from("a long value");
        let data2 = bytes::Bytes::from("good");
        let (_keepalive, repo) = KV::setup_repo();
        let mut ml = new_merkle_layer::<KV>(&repo);
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
    });

    kv_test!(test_mavl_write_prop, KV, {
        proptest!(|(keys in prop::collection::vec(any::<[u8; 2]>(), 0..10))| {
            let data = bytes::Bytes::from(vec![0; 500]);
            let alternating = bytes::Bytes::from([1, 0]
                .iter()
                .cycle()
                .take(500)
                .cloned()
                .collect::<Vec<_>>());

            let (_keepalive, repo) = KV::setup_repo();
            let mut ml = new_merkle_layer::<KV>(&repo);
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
        });
    });

    #[derive(Debug, Clone)]
    enum GeneratedOperation {
        // Key, Value
        Set([u8; 2], Vec<u8>),
        // Key, Value, offset hint (used to generate a valid offset for existing values)
        Write([u8; 2], Vec<u8>, u8),
        // Key
        Delete([u8; 2]),
    }

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

    kv_test!(test_merkle_layer_checkout_lazy_from_commit, KV: PersistentKeyValueStore, {
        proptest!(|(operations in (1usize..100usize).prop_flat_map(generated_operations_strategy))| {
            use std::collections::BTreeMap;
            use std::collections::BTreeSet;

            let (_keepalive, repo) = KV::setup_repo();
            let persistence: Arc<KV> = KV::new(&repo)
                .expect("Creating a persistence layer should succeed")
                .into();

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
        });
    });

    // - Add some data to the Merkle layer.
    // - Commit the data to relevant column family
    // - Check whether the data is persisted.
    // - Check whether the hash contained in the commit id
    //   is the same as the root hash
    kv_test!(test_merkle_layer_commit_persists_nodes, KV: PersistentKeyValueStore, {
        use crate::storage::Loadable;
        use crate::storage::Storable;
        use crate::storage::StoreOptions;

        let (_keepalive, repo) = KV::setup_repo();
        let mut merkle_layer = new_merkle_layer::<KV>(&repo);

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
            assert_eq!(node.data(), loaded_node.data());
        }

        let root_hash = merkle_layer.hash();
        assert_eq!(*commit_id.as_hash(), root_hash);
    });

    kv_test!(test_prove_delete, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let (_keepalive, repo) = KV::setup_repo();
        let persistence: Arc<KV> = KV::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: Tree::default(),
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
    });

    kv_test!(test_prove_multiple_keys, KV, {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));
        let data: [&[u8]; 3] = [b"too cold", b"too hot", b"just right"];

        let (_keepalive, repo) = KV::setup_repo();
        let persistence: Arc<KV> = KV::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: Tree::default(),
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
    });

    kv_test!(test_prove_try_clone_with_cow, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let (_keepalive, repo) = KV::setup_repo();
        let persistence: Arc<KV> = KV::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: Tree::default(),
                resolver: ProveResolver::start(LazyResolver::new(persistence)),
            },
        };
        let cow_data = "🐮<(prove a moo!)";
        ml.set(&key, cow_data.as_bytes())
            .expect("setting node should succeed");

        let (_keepalive, repo) = KV::setup_repo();
        let kv: Arc<KV> = KV::new(&repo)
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
    });

    kv_test!(test_prove_write_partial, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let (_keepalive, repo) = KV::setup_repo();
        let persistence: Arc<KV> = KV::new(&repo)
            .expect("Creating a persistence layer should succeed")
            .into();
        let mut ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: Tree::default(),
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
    });

    kv_test!(test_prove_verify_round_trip, KV, {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));

        // `Normal` mode
        let (_keepalive, repo) = KV::setup_repo();
        let mut normal_ml = new_merkle_layer::<KV>(&repo);
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
        let resolver = LazyResolver::new(normal_ml.inner.persistence.clone());
        let resolver = ProveResolver::start(resolver);
        let prove_tree = normal_ml.inner.tree.into_proof(&resolver);

        let prove_ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: prove_tree,
                resolver,
            },
        };

        // Read to mark it as present in the proof
        let node = prove_ml
            .get(&keys[1])
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");
        assert_eq!(node, b"prove to verify");

        // Verify mode
        let proof = MerkleProof::from_foldable(&prove_ml.inner.tree);
        let verify_tree_id = VerifyTreeId::from_proof(ProofTree::Present(&proof))
            .expect("The proof should be deserialisable")
            .into_result();

        let tree = match verify_tree_id {
            VerifyTreeId::Present(tree) => tree,
            _ => panic!("Should be present"),
        };

        let verify_ml: MerkleLayer<KV, Verify> = MerkleLayer::from_verify_tree(tree);

        let node = verify_ml
            .get(&keys[1])
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");
        assert_eq!(node, b"prove to verify");
    });

    kv_test!(test_verify_delete, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut ml: MerkleLayer<KV, Verify> = MerkleLayer::from_verify_tree(Tree::default());
        ml.delete(&key)
            .expect("deleting a key that doesn't exist should succeed");

        ml.set(&key, b"delete")
            .expect("setting node should succeed");
        ml.delete(&key).expect("delete should succeed");

        let got = ml
            .get(&key)
            .expect("The node should be retrieved successfully.");
        assert!(got.is_none(), "data should not exist after deletion");
    });

    kv_test!(test_verify_multiple_keys, KV, {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));
        let data: [&[u8]; 3] = [b"too cold", b"too hot", b"just right"];

        let mut ml: MerkleLayer<KV, Verify> = MerkleLayer::from_verify_tree(Tree::default());
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
    });

    kv_test!(test_verify_try_clone_with_cow, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut ml: MerkleLayer<KV, Verify> = MerkleLayer::from_verify_tree(Tree::default());
        let cow_data = "🐮<(verify a moo!)";
        ml.set(&key, cow_data.as_bytes())
            .expect("setting node should succeed");

        let (_keepalive, repo) = KV::setup_repo();
        let kv: Arc<KV> = KV::new(&repo)
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
    });

    kv_test!(test_verify_write_partial, KV, {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");

        let mut ml: MerkleLayer<KV, Verify> = MerkleLayer::from_verify_tree(Tree::default());
        ml.set(&key, b"partial")
            .expect("setting node should succeed");
        ml.write(&key, 4, b"ying").expect("write should succeed");

        let got = ml
            .get(&key)
            .expect("The node should be retrieved successfully.")
            .expect("The data should exist");

        assert_eq!(got, b"partying");
    });

    kv_test!(test_prove_hash, KV, {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Size less than KEY_MAX_SIZE"));

        let (_keepalive, repo) = KV::setup_repo();
        let mut normal_ml = new_merkle_layer::<KV>(&repo);
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

        let resolver = LazyResolver::new(normal_ml.inner.persistence.clone());
        let resolver = ProveResolver::start(resolver);
        let prove_tree = normal_ml.inner.tree.into_proof(&resolver);

        let prove_ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: prove_tree,
                resolver,
            },
        };

        assert_eq!(normal_hash, prove_ml.hash());
    });

    kv_test!(test_prove_verify_round_trip_write, KV, {
        let keys = [Key::new(&[1]), Key::new(&[2]), Key::new(&[3])]
            .map(|r| r.expect("key should be valid"));

        // ---- Normal: build a three-node tree ----
        let (_keepalive, repo) = KV::setup_repo();
        let mut normal_ml = new_merkle_layer::<KV>(&repo);
        normal_ml
            .set(&keys[0], b"alpha")
            .expect("set should succeed");
        normal_ml
            .set(&keys[1], b"beta")
            .expect("set should succeed");
        normal_ml
            .set(&keys[2], b"gamma")
            .expect("set should succeed");

        let initial_hash = normal_ml.hash();

        // ---- Prove: read key then overwrite it ----
        let resolver = LazyResolver::new(normal_ml.inner.persistence.clone());
        let resolver = ProveResolver::start(resolver);
        let prove_tree = normal_ml.inner.tree.into_proof(&resolver);

        let mut prove_ml: MerkleLayer<KV, Prove<'static>> = MerkleLayer {
            inner: ProveImpl {
                tree: prove_tree,
                resolver,
            },
        };

        let data = prove_ml
            .get(&keys[1])
            .expect("get should succeed")
            .expect("key should exist");
        assert_eq!(data, b"beta");

        prove_ml
            .write(&keys[1], 0, b"BETA")
            .expect("write should succeed");

        let expected_hash = prove_ml.hash();
        assert_ne!(initial_hash, expected_hash, "write should change the hash");

        // ---- Generate proof ----
        let merkle_proof = MerkleProof::from_foldable(&prove_ml.inner.tree);

        // ---- Verify: deserialize proof, replay identical ops ----
        let verify_tree_id = VerifyTreeId::from_proof(ProofTree::Present(&merkle_proof))
            .expect("proof deserialization should succeed")
            .into_result();

        let VerifyTreeId::Present(verify_tree) = verify_tree_id else {
            panic!("expected Present tree from proof");
        };

        let mut verify_ml: MerkleLayer<KV, Verify> = MerkleLayer::from_verify_tree(verify_tree);

        let final_hash = catch_not_found(move || {
            let data = verify_ml
                .get(&keys[1])
                .expect("get should succeed")
                .expect("key should be present in proof");
            assert_eq!(data, b"beta", "verify should see initial data");

            verify_ml
                .write(&keys[1], 0, b"BETA")
                .expect("write should succeed");

            let verify_tree_id = VerifyTreeId::Present(verify_ml.inner.tree);
            PartialHash::from_foldable(Some(merkle_proof), &verify_tree_id)
                .to_hash()
                .expect("verify hash should be computable")
        })
        .expect("verify operations should not trigger not_found");

        assert_eq!(
            expected_hash, final_hash,
            "prove and verify hashes should match after identical get + write operations"
        );
    });
}
