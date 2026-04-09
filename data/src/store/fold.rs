// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of `BlobStoreFold` that allows foldable components to be stored as Merkle trees.

use std::sync::Arc;

use derive_more::From;

use super::BlobStore;
use super::BlobStoreError;
use crate::foldable::Fold;
use crate::foldable::FoldLeaf;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::hash::Hash;
use crate::hash::HashedData;

/// A builder type for saving the Merkle tree structure of any `Foldable` type into any `BlobStore`.
#[derive(From)]
pub struct BlobStoreFold<BS> {
    store: Arc<BS>,
}

/// The node builder type corresponding to `BlobStoreFold`. Tracks the concatenated hashes as a
/// byte string, before finally hashing them when the node is `done`.
///
/// This also has to track the error status of the fold because the `NodeFold` trait defines the
/// method `add` as infallible. In our case, `add` may cause an error because any access of the
/// `BlobStore` may do so. We store that error in the `error` field and subsequently short-circuit
/// on any further calls to `add`, returning the error when `done` is called.
pub struct BlobStoreNodeFold<BS: BlobStore> {
    store: Arc<BS>,
    bytes: Vec<u8>,
    error: Option<BlobStoreError>,
}

impl<BS: BlobStore> Fold for BlobStoreFold<BS> {
    type Folded = Result<Hash, BlobStoreError>;

    type NodeFold = BlobStoreNodeFold<BS>;

    fn into_node_fold(self) -> BlobStoreNodeFold<BS> {
        BlobStoreNodeFold {
            store: self.store,
            bytes: vec![],
            error: None,
        }
    }
}

impl<BS: BlobStore> FoldLeaf for BlobStoreFold<BS> {
    fn fold_leaf_raw(self, bytes: &[u8]) -> Result<Hash, BlobStoreError> {
        let hashed = HashedData::from_data(bytes);
        self.store.blob_set(&hashed)?;
        Ok(hashed.hash())
    }
}

impl<BS: BlobStore> NodeFold for BlobStoreNodeFold<BS> {
    type Parent = BlobStoreFold<BS>;

    fn add<T: Foldable<BlobStoreFold<BS>>>(&mut self, child: &T) {
        if self.error.is_some() {
            return;
        }
        match child.fold(BlobStoreFold {
            store: Arc::clone(&self.store),
        }) {
            Ok(hash) => {
                self.bytes.extend_from_slice(hash.as_ref());
            }
            Err(e) => {
                self.error = Some(e);
            }
        };
    }

    fn done(self) -> Result<Hash, BlobStoreError> {
        if let Some(e) = self.error {
            Err(e)
        } else {
            let hashed = HashedData::from_data(self.bytes);
            self.store.blob_set(&hashed)?;
            Ok(hashed.hash())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::BlobStoreFold;
    use crate::components::atom::Atom;
    use crate::components::bytes::Bytes;
    use crate::components::data_space::DataSpace;
    use crate::components::vector::Vector;
    use crate::foldable::Foldable;
    use crate::hash::Hash;
    use crate::hash::HashedData;
    use crate::mode::Normal;
    use crate::store::BlobStore;
    use crate::store::BlobStoreError;
    use crate::store::InMemoryBlobStore;

    #[derive(Debug, thiserror::Error)]
    enum TestError {
        #[error("Test error")]
        TestError,
    }

    impl From<TestError> for BlobStoreError {
        fn from(e: TestError) -> Self {
            Self::Custom(Box::new(e))
        }
    }

    struct ErroringBlobStore {
        inner: InMemoryBlobStore,
        error_hash: Hash,
    }

    impl ErroringBlobStore {
        fn from_hash(hash: Hash) -> Self {
            ErroringBlobStore {
                inner: InMemoryBlobStore::new(),
                error_hash: hash,
            }
        }
    }

    impl BlobStore for ErroringBlobStore {
        fn blob_get(&self, key: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
            Ok(self.inner.blob_get(key).unwrap())
        }

        fn blob_set<Data: AsRef<[u8]>>(
            &self,
            blob: &HashedData<Data>,
        ) -> Result<(), BlobStoreError> {
            if blob.hash() == self.error_hash {
                Err(TestError::TestError)?
            } else {
                self.inner.blob_set(blob).unwrap();
                Ok(())
            }
        }

        fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError> {
            self.inner.blob_delete(key).unwrap();
            Ok(())
        }
    }

    type A<T> = Atom<T, Normal>;

    fn a<T>(t: T) -> A<T> {
        Atom::new(t)
    }

    type Data = (
        [(A<u8>, A<u64>); 4],
        Bytes<Normal>,
        (Vector<A<bool>, Normal>, DataSpace<Normal>, (A<u8>, A<bool>)),
    );

    #[test]
    fn fold_in_memory() {
        let data: Data = (
            [(a(1), a(2)), (a(3), a(4)), (a(7), a(8)), (a(9), a(2379))],
            Bytes::new(10000),
            (
                Vector::new(vec![a(false); 37]),
                DataSpace::new(20000),
                (a(79), a(false)),
            ),
        );

        let store = Arc::new(InMemoryBlobStore::new());
        let folded = data
            .fold(BlobStoreFold {
                store: Arc::clone(&store),
            })
            .unwrap();

        // Hash agreement between `BlobStoreFold` and `HashFold`
        let root_hash = Hash::from_foldable(&data);
        assert_eq!(folded, root_hash);

        // a few more hashes we can check are in the store
        let hash1 = Hash::hash_encodable(false).unwrap();
        let page_encoding = {
            let mut arr = [0u8; 4104];
            // the first four bytes encode the length, which is 4096, i.e. 256 * 16
            arr[1] = 16;
            arr
        };
        let hash2 = Hash::hash_encodable(page_encoding).unwrap();
        let hash3 = Hash::from_foldable(&DataSpace::<Normal>::new(20000));

        // 'false' is encoded as [0]
        assert_eq!(store.blob_get(hash1).unwrap().as_ref(), [0]);

        // a full page of zeroes
        assert_eq!(store.blob_get(hash2).unwrap().as_ref(), page_encoding);

        // the node for a `DataSpace` has two children so hash concatenation is 64 bytes
        assert_eq!(store.blob_get(hash3).unwrap().as_ref().len(), 64);

        // the root node has three children so hash concatenation is 96 bytes
        assert_eq!(store.blob_get(root_hash).unwrap().as_ref().len(), 96);
    }

    #[test]
    fn fold_error_from_blob_store() {
        let data: (A<u8>, A<u32>, A<u16>) = (a(9), a(2379), a(10));
        let hash1 = Hash::hash_encodable(9u8).unwrap();
        let hash2 = Hash::hash_encodable(2379u32).unwrap();
        let hash3 = Hash::hash_encodable(10u16).unwrap();

        let error_store = Arc::new(ErroringBlobStore::from_hash(hash2));
        let folded = data.fold(BlobStoreFold {
            store: error_store.clone(),
        });

        // fold passes through errors from the blob-store
        assert!(folded.is_err());

        // the first child hash has been stored
        assert_eq!(error_store.blob_get(hash1).unwrap().as_ref(), [9]);

        // the final child hash has not been stored (the error caused the fold to short-circuit)
        assert!(error_store.inner.blob_get(hash3).is_err())
    }
}
