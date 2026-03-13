// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of `BlobStoreUnfold` that allows unfoldable components that are stored as Merkle
//! trees to be extracted from any `BlobStore`.

use std::collections::VecDeque;
use std::sync::Arc;

use bincode::Decode;

use super::BlobStore;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::hash::Hash;
use crate::serialisation::deserialise_checked;

/// A source for unfolding that connects to any `BlobStore` and extracts the Merkle tree with
/// root-hash `hash`.
pub struct BlobStoreUnfold<BS> {
    store: Arc<BS>,
    hash: Hash,
}

// TODO TZX-86: cfg annotation will be removed as part of wiring up the new PVM commit and checkout
#[cfg(test)]
impl<BS> BlobStoreUnfold<BS> {
    fn new(store: Arc<BS>, hash: Hash) -> Self {
        Self { store, hash }
    }
}

/// The node unfolder corresponding to `BlobStoreUnfold`, has a single `store` but a vector of
/// `hashes` representing the children of the node.
pub struct BlobStoreNodeUnfold<BS> {
    store: Arc<BS>,
    hashes: VecDeque<Hash>,
}

/// Source-specific unfold error type for `BlobStoreUnfold`.
#[derive(Debug, thiserror::Error)]
pub enum BlobStoreUnfoldError {
    #[error("Node with {remainder} leftover bytes forming an incomplete hash")]
    InvalidNode { remainder: usize },
}

impl<BS: BlobStore> Unfold for BlobStoreUnfold<BS> {
    type NodeUnfold = BlobStoreNodeUnfold<BS>;

    fn into_node(self) -> Result<BlobStoreNodeUnfold<BS>, UnfoldError> {
        let bytes = self
            .store
            .blob_get(self.hash)
            .map_err(|e| UnfoldError::OfSource(Box::new(e)))?;

        let (chunks, remainder) = bytes.as_ref().as_chunks::<{ Hash::DIGEST_SIZE }>();

        if !remainder.is_empty() {
            return Err(UnfoldError::OfSource(Box::new(
                BlobStoreUnfoldError::InvalidNode {
                    remainder: remainder.len(),
                },
            )));
        }

        Ok(BlobStoreNodeUnfold {
            store: self.store.clone(),
            hashes: chunks
                .iter()
                .cloned()
                .map(Hash::from)
                .collect::<VecDeque<_>>(),
        })
    }

    fn into_leaf<T: Decode<()>>(self) -> Result<T, UnfoldError> {
        let bytes = self
            .store
            .blob_get(self.hash)
            .map_err(|e| UnfoldError::OfSource(Box::new(e)))?;
        let value = deserialise_checked::<T>(bytes.as_ref())?;

        Ok(value)
    }

    fn into_leaf_raw<const LEN: usize>(self) -> Result<Box<[u8; LEN]>, UnfoldError> {
        let bytes = self
            .store
            .blob_get(self.hash)
            .map_err(|e| UnfoldError::OfSource(Box::new(e)))?;

        let boxed_array = Box::new(<[u8; LEN]>::try_from(bytes.as_ref()).map_err(|_| {
            UnfoldError::UnexpectedLeafSize {
                expected: LEN,
                got: bytes.as_ref().len(),
            }
        })?);

        Ok(boxed_array)
    }
}

impl<BS: BlobStore> NodeUnfold for BlobStoreNodeUnfold<BS> {
    type Parent = BlobStoreUnfold<BS>;

    fn next_branch_with<T>(
        &mut self,
        unfolder: impl FnOnce(BlobStoreUnfold<BS>) -> Result<T, UnfoldError>,
    ) -> Result<T, UnfoldError> {
        let Some(hash) = self.hashes.pop_front() else {
            return Err(UnfoldError::TooFewChildren);
        };

        unfolder(BlobStoreUnfold {
            store: self.store.clone(),
            hash,
        })
    }

    fn done<T>(self, value: T) -> Result<T, UnfoldError> {
        if self.hashes.is_empty() {
            Ok(value)
        } else {
            Err(UnfoldError::TooManyChildren(self.hashes.len()))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bincode::error::DecodeError;

    use super::BlobStoreUnfold;
    use super::BlobStoreUnfoldError;
    use crate::components::atom::Atom;
    use crate::components::bytes::Bytes;
    use crate::foldable::Foldable;
    use crate::foldable::UnfoldError;
    use crate::foldable::Unfoldable;
    use crate::mode::Normal;
    use crate::store::InMemoryBlobStore;
    use crate::store::fold::BlobStoreFold;

    type Data = (Bytes<Normal>, Bytes<Normal>, (Bytes<Normal>, Bytes<Normal>));

    #[test]
    fn fold_unfold() {
        let data: Data = (
            Bytes::new(7985),
            Bytes::new(3720),
            (Bytes::new(38), Bytes::new(13002)),
        );
        let store = Arc::new(InMemoryBlobStore::new());
        let root_hash = data.fold(BlobStoreFold::from(Arc::clone(&store))).unwrap();

        let unfolded = Data::unfold(BlobStoreUnfold::new(store, root_hash)).unwrap();

        assert_eq!(data, unfolded);
    }

    #[test]
    fn fold_unfold_error() {
        let store = Arc::new(InMemoryBlobStore::new());
        let builder = || BlobStoreFold::from(Arc::clone(&store));

        let atom = || Atom::new(37u32);
        let atom_hash = atom().fold(builder()).unwrap();

        let big_atom = Atom::<[u8; 43], Normal>::new([98u8; 43]);
        let big_atom_hash = big_atom.fold(builder()).unwrap();

        let five: [Atom<u32, Normal>; 5] = [atom(), atom(), atom(), atom(), atom()];
        let five_hash = five.fold(builder()).unwrap();

        let source = |hash| BlobStoreUnfold::new(Arc::clone(&store), hash);

        // Expects a node with two children, finds five.
        let unfold_result = <[Atom<u32, Normal>; 2]>::unfold(source(five_hash));
        assert!(matches!(
            unfold_result,
            Err(UnfoldError::TooManyChildren(3))
        ));

        // Expects a node with seven children, finds five.
        let unfold_result = <[Atom<u32, Normal>; 7]>::unfold(source(five_hash));
        assert!(matches!(unfold_result, Err(UnfoldError::TooFewChildren)));

        // Expects a node, finds a leaf with 43 bytes instead.
        let unfold_result = <[Atom<u32, Normal>; 3]>::unfold(source(big_atom_hash));
        assert!(
            matches!(unfold_result, Err(UnfoldError::OfSource(boxed_err))
            if boxed_err.is::<BlobStoreUnfoldError>()
            && matches!(
                boxed_err.downcast_ref::<BlobStoreUnfoldError>(),
                Some(BlobStoreUnfoldError::InvalidNode { remainder: 11 })))
        );

        // Expects a leaf, finds a node instead.
        let unfold_result = Atom::<u8, Normal>::unfold(source(five_hash));
        assert!(matches!(
            unfold_result,
            Err(UnfoldError::Deserialise(DecodeError::OtherString(string)))
            if string == "Slice was length 160, expected 1"));

        // Expects a leaf with 13 bytes, finds one with 4 bytes.
        let unfold_result = Atom::<[u8; 13], Normal>::unfold(source(atom_hash));
        println!("{unfold_result:?}");
        assert!(matches!(
            unfold_result,
            Err(UnfoldError::Deserialise(DecodeError::UnexpectedEnd {
                additional: 9
            }))
        ));

        // Expects a leaf with 1 byte, finds one with 4 bytes.
        let unfold_result = Atom::<u8, Normal>::unfold(source(atom_hash));
        assert!(matches!(
            unfold_result,
            Err(UnfoldError::Deserialise(DecodeError::OtherString(string)))
            if string == "Slice was length 4, expected 1"));
    }
}
