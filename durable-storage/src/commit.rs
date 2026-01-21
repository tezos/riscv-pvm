// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of commit identifiers for the durable storage.
//! A [`CommitId`] uniquely identifies a prior, immutable state.

use octez_riscv_data::hash::Hash;

/// [`CommitId`]'s are used to generate commits & to checkout specific commits
/// from a `DirectoryManager`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommitId(Hash);

impl CommitId {
    pub fn as_hash(&self) -> &Hash {
        &self.0
    }

    /// Returns the hex encoded commit id.
    pub fn hex_encode(&self) -> String {
        hex::encode(self.0)
    }
}

impl From<Hash> for CommitId {
    fn from(hash: Hash) -> Self {
        Self(hash)
    }
}
