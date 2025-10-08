// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

pub mod node;
pub mod tree;

use std::cmp::Ordering;

use tree::BinaryTree;
use tree::MerkleBinaryTree;

/// An identifier generated for a given commit.
pub struct CommitId;

/// A unique key used to store, retrieve and mutate data in durable storage.
#[derive(Clone, Debug, Default)]
pub struct Key([u8; KEY_LENGTH]);

impl Eq for Key {}

impl Ord for Key {
    fn cmp(&self, other: &Key) -> Ordering {
        for (l, r) in self.0.iter().zip(other.0.iter()) {
            let comparison = l.cmp(r);
            if comparison != Ordering::Equal {
                return comparison;
            }
        }

        Ordering::Equal
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Key) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Key) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const KEY_LENGTH: usize = 32;

/// Errors for fallible [MerkleLayer] operations.
pub enum MerkleLayerError {}

/// A layer for transforming data into a Merkelised representation before commitment to the [PersistenceLayer].
pub trait MerkleLayer: MerkleLayerStable + MerkleLayerInvalidating {}

/// [MerkleLayer] operations which invalidate the root hash.
pub trait MerkleLayerInvalidating {
    /// Clear all data from the [MerkleLayer].
    fn clear(&mut self);

    /// Delete the data associated with a given [Key].
    fn delete(&mut self, key: &Key);

    /// Returns a mutable reference to the data stored for a given [Key].
    fn get_mut(&mut self, key: &Key) -> Option<&mut Vec<u8>>;

    /// Returns the root hash, potentially re-hashing uncached nodes.
    fn hash(&mut self) -> blake3::Hash;

    /// Sets the data associated with a given [Key].
    fn set(&mut self, key: &Key, data: Vec<u8>);
}

/// [MerkleLayer] operations where the root hash remains unchanged.
pub trait MerkleLayerStable: Clone + Sized {
    /// Persist the data stored in the [MerkleLayer] to durable storage via the [PersistenceLayer].
    fn checkout(
        persistence: PersistenceLayer,
        root: blake3::Hash,
    ) -> Result<Self, MerkleLayerError>;

    /// Generates a commitment for the [MerkleLayer].
    fn commit(&self) -> Result<CommitId, MerkleLayerError>;

    /// Creates an empty [MerkleLayer].
    fn empty(persistence: PersistenceLayer) -> Self;

    /// Returns an immutable reference to the data stored for a given [Key].
    fn get(&self, key: &Key) -> Option<&Vec<u8>>;
}

impl<T: BinaryTree + Clone> MerkleLayerStable for T {
    fn empty(_persistence: PersistenceLayer) -> Self {
        Self::default()
    }

    fn checkout(
        _persistence: PersistenceLayer,
        _root: blake3::Hash,
    ) -> Result<Self, MerkleLayerError> {
        todo!()
    }

    fn commit(&self) -> Result<CommitId, MerkleLayerError> {
        todo!()
    }

    fn get(&self, key: &Key) -> Option<&Vec<u8>> {
        T::get(self, key)
    }
}

impl<T: MerkleBinaryTree> MerkleLayerInvalidating for T {
    fn clear(&mut self) {
        self.root_mut().take();
    }

    fn get_mut(&mut self, _key: &Key) -> Option<&mut Vec<u8>> {
        todo!()
    }

    fn hash(&mut self) -> blake3::Hash {
        T::hash(self)
    }

    fn set(&mut self, key: &Key, data: Vec<u8>) {
        T::set(self, key, data)
    }

    fn delete(&mut self, key: &Key) {
        T::delete(self, key)
    }
}

const NONE_HASH: blake3::Hash = blake3::Hash::from_bytes([0; 32]);

/// A stand-in for the in-development layer for persisting data to durable storage.
pub struct PersistenceLayer;

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::KEY_LENGTH;
    use super::Key;
    use super::MerkleLayerInvalidating;
    use super::MerkleLayerStable;
    use super::PersistenceLayer;
    use super::node::MavlNode;
    use super::node::NodeData;
    use super::tree::Avl;

    #[cfg(test)]
    trait ValidateTree<Node> {
        fn is_inorder(&self) -> bool;

        fn is_inorder_inner(node: &Option<Box<Node>>, min: &Key, max: &Key) -> bool;
    }

    #[cfg(test)]
    impl<T, Node> ValidateTree<Node> for T
    where
        T: super::tree::BinaryTree<Node = Node>,
        Node: super::node::BinaryTreeNode + NodeData,
    {
        fn is_inorder(&self) -> bool {
            Self::is_inorder_inner(
                self.root(),
                &Key([u8::MIN; KEY_LENGTH]),
                &Key([u8::MAX; KEY_LENGTH]),
            )
        }

        fn is_inorder_inner(node: &Option<Box<Node>>, min: &Key, max: &Key) -> bool {
            if let Some(node) = node.as_ref() {
                if node.key() < min || node.key() > max {
                    return false;
                }
                return Self::is_inorder_inner(node.left_ref(), min, node.key())
                    && Self::is_inorder_inner(node.right_ref(), node.key(), max);
            }
            true
        }
    }

    #[test]
    fn test_key_comparison() {
        let mut key1: Key = Key([0; KEY_LENGTH]);
        let mut key2: Key = Key([0; KEY_LENGTH]);

        assert_eq!(key1, key2);
        assert_eq!(key1.cmp(&key2), Ordering::Equal);
        key1.0[1] = 1;
        assert_eq!(key1.cmp(&key2), Ordering::Greater);
        key2.0[0] = 1;
        assert_eq!(key1.cmp(&key2), Ordering::Less);
    }

    #[test]
    fn test_mavl_create() {
        let key: Key = Key([1; KEY_LENGTH]);
        let data = vec![0; 8];
        let mut avl = Avl::<MavlNode>::empty(PersistenceLayer {});
        avl.set(&key, data.clone());

        let node: MavlNode = NodeData::new(key.clone(), data.clone());
        let get_node = avl
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(get_node, NodeData::data(&node));
    }

    #[test]
    fn test_mavl_create_existing() {
        let key: Key = Key([1; KEY_LENGTH]);
        let data = vec![0; 8];
        let mut avl = Avl::<MavlNode>::empty(PersistenceLayer{});
        avl.set(&key, data.clone());

        let node: MavlNode = NodeData::new(key.clone(), data.clone());
        let get_node = avl
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(get_node, NodeData::data(&node));

        let hash1 = avl.hash();

        avl.set(&key, data.clone());
        assert!(avl.is_inorder(), "AVL isn't in order: {avl:?}");
        let node: MavlNode = NodeData::new(key.clone(), data.clone());
        let get_node = avl
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(get_node, NodeData::data(&node));
        let hash2 = avl.hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_mavl_delete() {
        let key: Key = Key([1; KEY_LENGTH]);
        let data = vec![0; 8];
        let mut avl = Avl::<MavlNode>::empty(PersistenceLayer{});

        avl.set(&key, data.clone());

        let node: MavlNode = NodeData::new(key.clone(), data.clone());
        let get_node = avl
            .get(&key)
            .expect("The node should be retrieved successfully");

        assert_eq!(get_node, NodeData::data(&node));
        avl.delete(&key);
        assert_eq!(avl.get(&key), None);
    }

    #[test]
    fn test_mavl_delete_invalid_key() {
        let key = Key([1; KEY_LENGTH]);

        let invalid_key = Key([255; KEY_LENGTH]);

        let data = vec![];

        let mut avl = Avl::<MavlNode>::empty(PersistenceLayer{});

        avl.set(&key, data.clone());
        assert!(avl.is_inorder(), "AVL tree isn't in order: {avl:?}");

        let hash1 = avl.hash();

        avl.delete(&invalid_key.clone());
        assert!(avl.is_inorder(), "AVL tree isn't in order: {avl:?}");

        let hash2 = avl.hash();
        assert!(hash1 == hash2);
    }

    #[test]
    fn test_mavl_invalidate_hash() {
        let keys = [
            Key([3; KEY_LENGTH]),
            Key([1; KEY_LENGTH]),
            Key([2; KEY_LENGTH]),
            Key([4; KEY_LENGTH]),
        ];

        let data = vec![];

        let mut avl = Avl::<MavlNode>::default();

        for key in keys.iter() {
            avl.set(key, data.clone());
            assert!(avl.is_inorder(), "avl isn't in order: {avl:?}");
        }

        let hash1 = avl.hash();
        avl.delete(&keys[3].clone());
        assert!(avl.is_inorder(), "avl isn't in order: {avl:?}");
        let hash2 = avl.hash();
        assert!(hash1 != hash2);
    }
}
