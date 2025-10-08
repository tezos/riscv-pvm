// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

/// An identifier generated for a given commit.
pub struct CommitId;

/// A unique key used to store, retrieve and mutate data in durable storage.
pub struct Key;

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

/// A stand-in for the in-development layer for persisting data to durable storage.
pub struct PersistenceLayer;
