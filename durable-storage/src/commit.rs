// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of commit identifiers for the durable storage.
//! A [`CommitId`] uniquely identifies a prior, immutable state.

use bincode::Decode;
use bincode::Encode;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::hash::Hash;

use crate::registry::REGISTRY_ARITY;

/// [`CommitId`]'s are used to generate commits & to checkout specific commits
/// from a `DirectoryManager`.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Encode, Decode)]
pub struct CommitId(Hash);

impl CommitId {
    pub fn as_hash(&self) -> &Hash {
        &self.0
    }

    /// Returns the hex encoded commit id.
    pub fn hex_encode(&self) -> String {
        hex::encode(self.0)
    }

    /// Compute the Merkle root hash from the list of database commit IDs.
    pub(super) fn compute_root_hash(db_hashes: &[Self]) -> Self {
        if db_hashes.is_empty() {
            let empty_hash: Hash = Hash::hash_bytes(&[]);
            return CommitId::from(empty_hash);
        }

        let get_hash = |idx: usize| db_hashes[idx].as_hash();
        let tree = IndexableSeqAsTree::new(db_hashes.len(), REGISTRY_ARITY, &get_hash);
        let hash = Hash::from_foldable(&tree);
        CommitId::from(hash)
    }
}

impl From<Hash> for CommitId {
    fn from(hash: Hash) -> Self {
        Self(hash)
    }
}
