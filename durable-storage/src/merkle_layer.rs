// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::sync::Arc;

use bytes::Bytes;

mod node;
mod tree;

use tree::Avl;

use crate::persistence_layer::PersistenceLayer;

/// An identifier generated for a given commit.
#[derive(Debug, PartialEq, Eq)]
pub struct CommitId;

/// A unique key used to store, retrieve and mutate data in durable storage.
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Key(Vec<u8>);

impl Key {
    pub fn new(bytes: &[u8]) -> Result<Self, MerkleLayerError> {
        if bytes.len() > KEY_MAX_SIZE {
            return Err(MerkleLayerError::KeyTooLong);
        }
        Ok(Key(bytes.to_vec()))
    }
}

/// Maximum size of a key in bytes
pub(crate) const KEY_MAX_SIZE: usize = 256;

/// Errors for fallible [MerkleLayer] operations.
#[derive(Debug, thiserror::Error)]
pub enum MerkleLayerError {
    /// The maximum size of a key is [`KEY_MAX_SIZE`].
    #[error("The provided key is too long")]
    KeyTooLong,

    /// An error occurred when trying to make changes in the Merkle layer persist on disk.
    #[expect(dead_code, reason = "Not yet implemented")]
    #[error("An error occurred in the persistence layer")]
    PersistenceError,
}

/// A layer for transforming data into a Merkelised representation before commitment to the [PersistenceLayer].
#[derive(Clone, Debug)]
pub struct MerkleLayer {
    tree: Avl,
    #[expect(dead_code, reason = "To be used in RV-825")]
    persistence: Arc<PersistenceLayer>,
}

/// A layer for transforming data into a Merkelised representation before commitment to the [PersistenceLayer].
impl MerkleLayer {
    /// Create a new, empty Merkle layer that will commit to the provided persistence layer.
    pub fn new(persistence: Arc<PersistenceLayer>) -> Self {
        MerkleLayer {
            tree: Avl::default(),
            persistence,
        }
    }

    /// Persist the data stored in the [MerkleLayer] to durable storage via the [PersistenceLayer].
    pub fn checkout(
        _persistence: Arc<PersistenceLayer>,
        _root: blake3::Hash,
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
    pub fn clear(&mut self) {
        self.tree.root_mut().take();
    }

    /// Generates a commitment for the [MerkleLayer].
    pub fn commit(&self) -> Result<CommitId, MerkleLayerError> {
        todo!()
    }

    /// Delete the data associated with a given [Key].
    pub fn delete(&mut self, key: &Key) {
        self.tree.delete(key);
    }

    /// Returns an immutable reference to the data stored for a given [Key].
    pub fn get(&self, key: &Key) -> Option<&Bytes> {
        self.tree.get(key)
    }

    /// Returns a mutable reference to the data stored for a given [Key].
    #[expect(dead_code, reason = "Not yet implemented")]
    pub fn get_mut(&mut self, _key: &Key) -> Option<&mut Bytes> {
        todo!()
    }

    /// Returns the root hash, potentially re-hashing uncached nodes.
    pub fn hash(&mut self) -> blake3::Hash {
        todo!()
    }

    /// Sets the data associated with a given [Key].
    pub fn set(&mut self, key: &Key, data: Bytes) {
        self.tree.set(key, data)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use bytes::Bytes;
    use proptest::prelude::*;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::KEY_MAX_SIZE;
    use super::Key;
    use super::MerkleLayer;
    use super::node::MavlNode;
    use super::tree::Avl;
    use crate::persistence_layer::PersistenceLayer;
    use crate::repo::DirectoryManager;

    impl Avl {
        fn check(&self, line: u32) {
            let inorder = self.is_inorder();
            let is_balanced = self.is_balanced();
            let has_correct_balance_factors = self.has_correct_balance_factors();
            if !inorder || !is_balanced || !has_correct_balance_factors {
                eprintln!("{self:?}");
                eprintln!("Line {line:?}");
            }
            assert!(inorder, "AVL tree isn't in order");
            assert!(is_balanced, "AVL tree isn't balanced");
            assert!(
                has_correct_balance_factors,
                "AVL tree balance factors are incorrect"
            );
        }

        fn has_correct_balance_factors(&self) -> bool {
            has_correct_balance_factors(self.root())
        }

        fn is_balanced(&self) -> bool {
            is_balanced(self.root())
        }

        fn is_inorder(&self) -> bool {
            is_inorder(
                self.root(),
                &Key::new(&[u8::MIN]).expect("Size less than KEY_MAX_SIZE"),
                &Key::new(&[u8::MAX; KEY_MAX_SIZE]).expect("Size less than KEY_MAX_SIZE"),
            )
        }
    }

    impl MavlNode {
        pub(super) fn height(&self) -> u32 {
            let left_height = self.left_ref().as_ref().map_or(0, |l| l.height());
            let right_height = self.right_ref().as_ref().map_or(0, |r| r.height());
            1 + std::cmp::max(left_height, right_height)
        }
    }

    pub(super) fn has_correct_balance_factors(node: &Option<Arc<MavlNode>>) -> bool {
        if let Some(node) = node.as_ref() {
            let left_height = node.left_ref().as_ref().map_or(0, |l| l.height());
            let right_height = node.right_ref().as_ref().map_or(0, |r| r.height());
            let calculated_balance_factor = right_height as i64 - left_height as i64;
            if node.balance_factor() != calculated_balance_factor {
                eprintln!(
                    "Node has balance_factor {:?}, should be {calculated_balance_factor:?}\nnode: {node:?}",
                    node.balance_factor()
                );
                return false;
            }
            return has_correct_balance_factors(node.left_ref())
                && has_correct_balance_factors(node.right_ref());
        }
        true
    }

    pub(super) fn is_balanced(node: &Option<Arc<MavlNode>>) -> bool {
        if let Some(node) = node {
            let balance_factor = node.balance_factor();
            if balance_factor.abs() > 1 {
                eprintln!("Balance factor not in -1..=1: {balance_factor:?}");
                return false;
            }
            return is_balanced(node.left_ref()) && is_balanced(node.right_ref());
        }
        true
    }

    pub(super) fn is_inorder(node: &Option<Arc<MavlNode>>, min: &Key, max: &Key) -> bool {
        if let Some(node) = node.as_ref() {
            if node.key() < min || node.key() > max {
                return false;
            }
            return is_inorder(node.left_ref(), min, node.key())
                && is_inorder(node.right_ref(), node.key(), max);
        }
        true
    }

    fn new_merkle_layer() -> MerkleLayer {
        let directory_manager =
            DirectoryManager::new(Path::new("/tmp")).expect("Should be a valid directory");
        let persistence_layer =
            PersistenceLayer::new(&directory_manager).expect("Should be a valid directory");
        MerkleLayer::new(persistence_layer.into())
    }

    #[test]
    fn test_mavl_cow() {
        let keys = [Key::new(&[0]), Key::new(&[1]), Key::new(&[2])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = [vec![0; 0], vec![13; 5], vec![42; 129]];

        let mut ml = new_merkle_layer();

        for i in 0..keys.len() {
            ml.set(&keys[i], data[i].clone().into());
            ml.tree.check(line!());
        }

        let mut ml2 = ml.clone();

        let cow_data = "🐮<(moo!)";
        ml2.set(&keys[0], cow_data.into());

        let old_node1 = MavlNode::new(keys[0].clone(), data[0].clone().into());
        let new_node1 = MavlNode::new(keys[0].clone(), cow_data.into());

        let node2 = MavlNode::new(keys[1].clone(), data[1].clone().into());
        let node3 = MavlNode::new(keys[2].clone(), data[2].clone().into());

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

        ml.tree.check(line!());
        ml2.tree.check(line!());
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
                ml.set(&key, data1.clone());
            }

            // Create a cheap copy
            let mut ml2 = ml.clone();

            // Delete all the keys in the copy
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml2.delete(&key);
                prop_assert_eq!(ml2.get(&key), None);
            }

            // Set new keys in the copy
            for bytes in &keys2 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml2.set(&key, data2.clone());
            }

            // Check both trees are still correct
            for bytes in &keys1 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key), Some(&data1));
            }
            for bytes in &keys2 {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml2.get(&key), Some(&data2));
            }

            ml.tree.check(line!());
            ml2.tree.check(line!());
        }
    }

    #[test]
    fn test_mavl_create() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("create");
        let mut ml = new_merkle_layer();
        ml.set(&key, data.clone());

        let node = MavlNode::new(key.clone(), data.clone());
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());
        ml.tree.check(line!());
    }

    #[test]
    fn test_mavl_create_existing() {
        let key = Key::new(&[1]).expect("Size less than KEY_MAX_SIZE");
        let data = Bytes::from("old");
        let data2 = Bytes::from("new");
        let mut ml = new_merkle_layer();
        ml.set(&key, data.clone());

        let node = MavlNode::new(key.clone(), data.clone());
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());

        ml.set(&key, data2.clone());
        assert!(ml.tree.is_inorder(), "AVL isn't in order: {ml:?}");
        let node = MavlNode::new(key.clone(), data2.clone());
        let get_node = ml
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(&get_node, &node.data());
        ml.tree.check(line!());
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
            ml.set(key, data.clone());
            ml.tree.check(line!());
            assert_eq!(ml.get(key), Some(data));
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

        // Left imbalance
        let data = Bytes::from("imbalanced left");
        for key in keys.iter() {
            ml.set(key, data.clone());
            ml.tree.check(line!());
        }

        ml.clear();

        let keys = {
            let mut keys = keys;
            keys.sort();
            keys
        };

        // Right imbalance
        let data = Bytes::from("imbalanced right");
        for key in keys.iter() {
            ml.set(key, data.clone());
            ml.tree.check(line!());
        }
    }

    #[test]
    fn test_mavl_create_left_right() {
        let keys = [Key::new(&[2]), Key::new(&[0]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = Bytes::from("left_right");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            ml.set(key, data.clone());
            ml.tree.check(line!());
            assert_eq!(ml.get(key), Some(&data));
        }
    }

    #[test]
    fn test_mavl_create_right_left() {
        let keys = [Key::new(&[0]), Key::new(&[2]), Key::new(&[1])]
            .map(|r| r.expect("Sizes less than KEY_MAX_SIZE"));

        let data = Bytes::from("right_left");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            ml.set(key, data.clone());
            ml.tree.check(line!());
            assert_eq!(ml.get(key), Some(&data));
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
            ml.set(key, data.clone());
            ml.tree.check(line!());
            assert_eq!(ml.get(key), Some(&data));
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
            ml.set(key, data.clone());
            ml.tree.check(line!());
            assert_eq!(ml.get(key), Some(&data));
        }
    }

    proptest! {
        #[test]
        fn test_mavl_create_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = Bytes::from("property");
            let mut ml = new_merkle_layer();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, data.clone());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                prop_assert_eq!(ml.get(&key), Some(&data));
            }

            ml.tree.check(line!());
        }
    }

    #[test]
    fn test_mavl_delete() {
        let key = Key::new(&[1]).expect("Sizes less than KEY_MAX_SIZE");
        let data = Bytes::from("delete");
        let mut ml = new_merkle_layer();
        ml.set(&key, data.clone());

        ml.delete(&key);

        let get_node = ml.get(&key);

        assert_eq!(get_node, None);
        ml.tree.check(line!());
    }

    proptest! {
        #[test]
        fn test_mavl_delete_prop(keys in prop::collection::vec(any::<[u8; 2]>(), 0..500)) {
            let data = Bytes::from("delete_prop");
            let mut ml = new_merkle_layer();

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.set(&key, data.clone());
            }

            for bytes in &keys {
                let key = Key::new(bytes).expect("Sizes less than KEY_MAX_SIZE");
                ml.delete(&key);
                prop_assert_eq!(ml.get(&key), None);
            }

            ml.tree.check(line!());
        }
    }

    fn test_mavl_delete_keys(keys: &[Key]) {
        let data = Bytes::from("delete");

        let mut ml = new_merkle_layer();

        for key in keys.iter() {
            ml.set(key, data.clone());
            ml.tree.check(line!());
            assert_eq!(ml.get(key), Some(data.clone()).as_ref());
        }

        for key in keys.iter() {
            ml.delete(key);
            ml.tree.check(line!());
            ml.delete(key);
            assert_eq!(ml.get(key), None);
        }
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
}
