// SPDX-FileCopyrightText: 2025-2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::sync::Arc;

use octez_riscv_data::hash::Hash;

use crate::avl::node::Value;
use crate::avl::tree::Tree;
use crate::commit::CommitId;
use crate::key::Key;
use crate::persistence_layer::HashedData;
use crate::persistence_layer::PersistenceLayer;
use crate::persistence_layer::PersistenceLayerError;

/// Errors for fallible [MerkleLayer] operations.
#[derive(Debug, thiserror::Error)]
pub enum MerkleLayerError {
    #[error("Some error happened while trying to encode the node {0}")]
    EncodeError(#[from] bincode::error::EncodeError),

    #[error("Some error happened during the interaction with the persistence layer {0}")]
    PersistenceLayerError(#[from] PersistenceLayerError),
}

/// A layer for transforming data into a Merkelised representation before commitment to the [PersistenceLayer].
#[derive(Clone, Debug)]
pub struct MerkleLayer {
    tree: Tree,
    persistence: Arc<PersistenceLayer>,
}

/// A layer for transforming data into a Merkelised representation before commitment to the [PersistenceLayer].
impl MerkleLayer {
    /// Create a new, empty Merkle layer that will commit to the provided persistence layer.
    pub fn new(persistence: Arc<PersistenceLayer>) -> Self {
        MerkleLayer {
            tree: Tree::default(),
            persistence,
        }
    }

    /// Persist the data stored in the [MerkleLayer] to durable storage via the [PersistenceLayer].
    pub fn checkout(
        _persistence: Arc<PersistenceLayer>,
        _root: Hash,
    ) -> Result<Self, MerkleLayerError> {
        todo!()
    }

    /// Clone the Merkle layer. The new layer will commit to the provided persistence layer.
    pub fn clone_with(
        &self,
        _persistence: Arc<PersistenceLayer>,
    ) -> Result<Self, MerkleLayerError> {
        Ok(self.clone())
    }

    /// Clear all data from the [MerkleLayer].
    #[cfg_attr(
        not(any(test, feature = "bench")),
        expect(dead_code, reason = "Not pub in `Database`")
    )]
    pub fn clear(&mut self) {
        self.tree.take();
    }

    /// Generates a commitment for the [MerkleLayer].
    pub fn commit(&self) -> Result<CommitId, MerkleLayerError> {
        // Note that although we're doing in order
        // iteration of the nodes the hashes are
        // calculated during the encoding of the node
        // if necessary.
        for node in self.tree.iter() {
            let value = octez_riscv_data::serialisation::serialise(node.to_encode())?;
            let blob = HashedData::from_value(value.as_slice());
            self.persistence.blob_set(&blob)?;
        }

        Ok(CommitId::from(self.hash()))
    }

    /// Delete the data associated with a given [Key].
    pub fn delete(&mut self, key: &Key) {
        self.tree.delete(key);
    }

    /// Returns an immutable reference to the data stored for a given [Key].
    #[cfg_attr(
        not(any(test, feature = "bench")),
        expect(dead_code, reason = "Not pub in `Database`")
    )]
    pub fn get(&self, key: &Key) -> Option<&Value> {
        self.tree.get(key)
    }

    /// Returns a mutable reference to the data stored for a given [Key].
    #[cfg_attr(
        not(any(test, feature = "bench")),
        expect(dead_code, reason = "Not pub in `Database`")
    )]
    pub fn get_mut(&mut self, key: &Key) -> Option<&mut Value> {
        self.tree.get_mut(key)
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub fn hash(&self) -> Hash {
        self.tree.hash()
    }

    /// Sets the data associated with a given [Key].
    pub fn set(&mut self, key: &Key, data: &[u8]) {
        self.tree.set(key, data);
    }

    /// Writes the data to the node associated with a given [Key] with the given offset.
    pub fn write(&mut self, key: &Key, offset: usize, data: &[u8]) {
        self.tree.write(key, offset, data);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Bytes;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::MerkleLayer;
    use crate::avl::Node;
    use crate::avl::Tree;
    use crate::avl::hash;
    use crate::key::Key;
    use crate::persistence_layer::PersistenceLayer;
    use crate::persistence_layer::utils::TestableTmpdir;
    use crate::repo::DirectoryManager;

    impl MerkleLayer {
        fn tree(&self) -> &Tree {
            &self.tree
        }
    }

    fn new_merkle_layer() -> MerkleLayer {
        let tmpdir = TestableTmpdir::new();

        let repo =
            DirectoryManager::new(tmpdir.path()).expect("Failed to create directory manager");

        let persistence_layer = PersistenceLayer::new(&repo)
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
            ml.set(&keys[i], &data[i]);
            ml.tree().check();
        }

        let mut ml2 = ml.clone();
        let original_hash = ml.hash();
        assert_eq!(original_hash, ml2.hash());

        let cow_data = "🐮<(moo!)";
        ml2.set(&keys[0], cow_data.as_bytes());
        assert_ne!(original_hash, ml2.hash());
        assert_eq!(original_hash, ml.hash());

        let old_node1 = Node::new(keys[0].clone(), Bytes::copy_from_slice(&data[0]));
        let new_node1 = Node::new(keys[0].clone(), cow_data.as_bytes());

        let node2 = Node::new(keys[1].clone(), Bytes::copy_from_slice(&data[1]));
        let node3 = Node::new(keys[2].clone(), Bytes::copy_from_slice(&data[2]));

        assert_eq!(
            &old_node1.data(),
            &ml.get(&keys[0])
                .expect("The node should be retrieved successfully. Merkle layer: {ml:?}")
        );
        assert_eq!(
            &new_node1.data(),
            &ml2.get(&keys[0])
                .expect("The node should be retrieved successfully. Merkle layer: {ml2:?}")
        );

        assert_eq!(
            &node2.data(),
            &ml.get(&keys[1])
                .expect("The node should be retrieved successfully. Merkle layer: {ml:?}")
        );
        assert_eq!(
            &node2.data(),
            &ml2.get(&keys[1])
                .expect("The node should be retrieved successfully. Merkle layer: {ml2:?}")
        );

        assert_eq!(
            &node3.data(),
            &ml.get(&keys[2])
                .expect("The node should be retrieved successfully. Merkle layer: {ml:?}")
        );
        assert_eq!(
            &node3.data(),
            &ml2.get(&keys[2])
                .expect("The node should be retrieved successfully. Merkle layer: {ml2:?}")
        );

        ml.tree().check();
        ml2.tree().check();
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
                ml.set(&key, &data1);
            }

            // Create a cheap copy
            let original_hash = ml.hash();
            let mut ml2 = ml.clone();
            prop_assert_eq!(original_hash, ml2.hash());

            // Delete all the keys in the copy
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml2.delete(&key);
                prop_assert_eq!(ml2.get(&key), None);
            }

            // Set new keys in the copy
            for bytes in &keys2 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml2.set(&key, &data2);
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
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully"), &data1);
            }
            for bytes in &keys2 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml2.get(&key).expect("The node should be retrieved successfully"), &data2);
            }

            ml.tree().check();
            ml2.tree().check();
        }
    }

    #[test]
    fn test_mavl_create() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("create");
        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash();
        ml.set(&key, &data);
        assert_ne!(empty_hash, ml.hash());

        let node = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());
        ml.tree().check();
    }

    #[test]
    fn test_mavl_create_existing() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("old");
        let data2 = Bytes::from("new");
        let mut ml = new_merkle_layer();
        ml.set(&key, &data);
        let old_hash = ml.hash();

        let node = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());

        ml.set(&key, &data2);
        assert_ne!(old_hash, ml.hash());
        assert!(ml.tree.is_inorder(), "AVL isn't in order: {ml:?}");
        let node = Node::new(key.clone(), data2);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());
        ml.tree().check();
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
            let old_hash = ml.hash();
            ml.set(key, data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully"),
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
        let empty_hash = ml.hash();

        // Left imbalance
        let data = Bytes::from("imbalanced left");
        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
        }

        ml.clear();
        assert_eq!(empty_hash, ml.hash());

        let keys = {
            let mut keys = keys;
            keys.sort();
            keys
        };

        // Right imbalance
        let data = Bytes::from("imbalanced right");
        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
        }
    }

    #[test]
    fn test_mavl_create_left_right() {
        let keys = [Key::new(&[2]), Key::new(&[0]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = Bytes::from("left_right");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            let old_hash = ml.hash();
            ml.set(key, &data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully"),
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
            let old_hash = ml.hash();
            ml.set(key, &data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully"),
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
            let old_hash = ml.hash();
            ml.set(key, &data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully"),
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
            let old_hash = ml.hash();
            ml.set(key, &data);
            assert_ne!(old_hash, ml.hash());
            ml.tree().check();
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully"),
                &data
            );
        }
    }

    proptest! {
        #[test]
        fn test_mavl_create_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = Bytes::from("property");
            let mut ml = new_merkle_layer();
            let old_hash = ml.hash();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data);
            }

            if !keys.is_empty() {
                assert_ne!(old_hash, ml.hash());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully"), &data);
            }

            ml.tree().check();
        }
    }

    #[test]
    fn test_mavl_delete() {
        let key = Key::new(&[1]).expect("Sizes less than KEY_MAX_SIZE");
        let data = Bytes::from("delete");
        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash();
        ml.set(&key, &data);
        let full_hash = ml.hash();
        assert_ne!(empty_hash, full_hash);

        ml.delete(&key);
        assert_ne!(full_hash, ml.hash());
        assert_eq!(empty_hash, ml.hash());

        let get_node = ml.get(&key);

        assert_eq!(get_node, None);
        ml.tree().check();
    }

    proptest! {
        #[test]
        fn test_mavl_delete_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = Bytes::from("delete_prop");
            let mut ml = new_merkle_layer();
            let empty_hash = ml.hash();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data);
            }

            if !keys.is_empty() {
                prop_assert_ne!(empty_hash, ml.hash());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.delete(&key);
                prop_assert_eq!(ml.get(&key), None);
            }

            prop_assert_eq!(empty_hash, ml.hash());

            ml.tree().check();
        }
    }

    fn test_mavl_delete_keys(keys: &[Key]) {
        let data = Bytes::from("delete");

        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash();

        for key in keys.iter() {
            ml.set(key, &data);
            ml.tree().check();
            assert_eq!(
                ml.get(key)
                    .expect("The node should be retrieved successfully"),
                &data,
            );
        }

        if !keys.is_empty() {
            assert_ne!(empty_hash, ml.hash());
        }

        for key in keys.iter() {
            ml.delete(key);
            ml.tree().check();
            ml.delete(key);
            assert_eq!(ml.get(key), None);
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
    fn test_mavl_get_mut() {
        let key = Key::new(&[1]).expect("Sizes less than KEY_MAX_SIZE");
        let data = Bytes::from("get_mut");
        let mut ml = new_merkle_layer();
        let empty_hash = ml.hash();
        ml.set(&key, &data.clone());
        let full_hash = ml.hash();
        assert_ne!(empty_hash, full_hash);

        let data2 = Bytes::from("mutated with larger data");

        let data_mut = ml.get_mut(&key).expect("The operation should succeed");
        data_mut.set(&data2);

        assert_ne!(empty_hash, ml.hash());
        assert_ne!(full_hash, ml.hash());

        let before_node = Node::new(key.clone(), data);
        let after_node = Node::new(key.clone(), data2);

        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_ne!(&get_node, &before_node.data());
        assert_eq!(&get_node, &after_node.data());

        ml.tree().check();
    }

    #[test]
    fn test_mavl_get_mut_cow() {
        let key = Key::new(&[1]).expect("Sizes less than KEY_MAX_SIZE");
        let data = Bytes::from("get_mut_cow");

        let mut ml = new_merkle_layer();

        let empty_hash = ml.hash();
        ml.set(&key, &data);
        let before_hash = ml.hash();
        assert_ne!(empty_hash, before_hash);

        let ml2 = ml.clone();
        assert_eq!(before_hash, ml2.hash());

        let data2 = Bytes::from("mutated");

        let data_mut = ml.get_mut(&key).expect("The operation should succeed");
        data_mut.set(&data2);

        assert_ne!(empty_hash, ml.hash());
        assert_ne!(before_hash, ml.hash());
        assert_ne!(ml2.hash(), ml.hash());

        let before_node = Node::new(key.clone(), data);

        let get_node = ml2
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &before_node.data());

        ml.tree().check();
    }

    proptest! {
        #[test]
        fn test_mavl_get_mut_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = Bytes::from("get_mut_prop");
            let mut ml = new_merkle_layer();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data);
            }

            let data2 = Bytes::from("mutated");
            let mut seen = HashSet::new();
            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");

                let before_hash = ml.hash();
                let data_mut = ml.get_mut(&key).expect("The operation should succeed");
                data_mut.set(&data2);

                let before_node = Node::new(key.clone(), data.clone());
                let after_node = Node::new(key.clone(), data2.clone());

                let get_node = ml
                    .get(&key)
                    .expect("The node should be retrieved successfully");

                assert_ne!(&get_node, &before_node.data());
                assert_eq!(&get_node, &after_node.data());

                prop_assert_eq!(before_hash == ml.hash(), seen.contains(&key));
                seen.insert(key);
            }

            ml.tree().check();
        }
    }

    #[test]
    fn test_mavl_write_new_value() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("write_new_value");
        let mut ml = new_merkle_layer();
        let old_hash = ml.hash();
        ml.write(&key, 0, &data);

        let node = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());
        assert_ne!(old_hash, ml.hash());
        ml.tree().check();
    }

    #[test]
    fn test_mavl_write_no_truncation() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("a long value");
        let data2 = Bytes::from("good");
        let mut ml = new_merkle_layer();
        ml.set(&key, &data);
        let old_hash = ml.hash();

        let data_len = data.len();
        let node = Node::new(key.clone(), data);
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());

        ml.write(&key, 2, &data2);
        assert_ne!(old_hash, ml.hash());
        assert!(ml.tree.is_inorder(), "AVL isn't in order: {ml:?}");
        let node = Node::new(key.clone(), Bytes::from("a good value"));
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(get_node.len(), data_len);
        assert_eq!(&get_node, &node.data());
        ml.tree().check();
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
            let old_hash = ml.hash();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, &data);
                for offset in 0..250 {
                    ml.write(&key, offset * 2, &[1]);
                }
            }

            if !keys.is_empty() {
                assert_ne!(old_hash, ml.hash());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key).expect("The node should be retrieved successfully"), &alternating);
            }

            ml.tree().check();
        }
    }

    /// - Add some data to the Merkle layer.
    /// - Commit the data to relevant column family
    /// - Check whether the data is persisted.
    /// - Check whether the hash contained in the commit id
    ///   is the same as the root hash
    #[test]
    fn test_merkle_layer_commit_persists_nodes() {
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
            merkle_layer.set(key, data_elem);
        }

        let commit_id = merkle_layer
            .commit()
            .expect("The commit operation should not fail");

        for node in merkle_layer.tree.iter() {
            let serialised = octez_riscv_data::serialisation::serialise(node.to_encode())
                .expect("We should be able to serialise the node");
            let node_hash = hash(node);
            let blob = merkle_layer
                .persistence
                .blob_get(node_hash)
                .expect("The blob with the given key should be present");
            assert_eq!(serialised, blob.as_ref());
        }

        let root_hash = merkle_layer.tree.hash();
        assert_eq!(*commit_id.as_hash(), root_hash);
    }
}
