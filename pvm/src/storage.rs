// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2024 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

mod chunked_io;

use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use bincode::Decode;
use bincode::Encode;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashedData;
use octez_riscv_data::serialisation;
use octez_riscv_data::store::BlobStore;
use octez_riscv_data::store::BlobStoreError;
use thiserror::Error;

const CHUNK_SIZE: usize = 4096;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Serialization error: {0}")]
    CommitSerializationError(#[from] EncodeError),

    #[error("Deserialization error: {0}")]
    CommitDeserializationError(#[from] DecodeError),

    #[error("Invalid repo")]
    InvalidRepo,

    #[error("Committed chunk {0} not found")]
    ChunkNotFound(String),

    #[error("Blob store error")]
    BlobStore(#[from] BlobStoreError),
}

/// A subtrait for `BlobStore` to provide extra functionality required by the PVM storage to export
/// PVM snapshots.
pub trait PersistentBlobStore: BlobStore {
    /// Initialise a store. Either create a new directory if `path` does not exist or initialise in
    /// an existing directory.
    ///
    /// Should throw [`StorageError::InvalidRepo`] if `path` is a file.
    fn init_from_path(path: impl AsRef<Path>) -> Result<Self, StorageError>
    where
        Self: Sized;

    /// Copy a specific blob across to a different store. While this should be functionally
    /// equivalent to using `blob_get` followed by `blob_set` to copy the blob across, it could in
    /// many impls be more efficient (especially for large blobs) by using `std::fs::copy` or
    /// equivalent instead.
    fn export_blob(&self, other: &mut Self, hash: &Hash) -> Result<(), StorageError> {
        let blob = self.blob_get(*hash)?;
        other.blob_set(&HashedData::from_data(blob.as_ref()))?;

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Store {
    path: Box<Path>,
}

impl Store {
    fn file_name_of_hash(hash: &Hash) -> String {
        hex::encode(hash)
    }

    fn path_of_hash(&self, hash: &Hash) -> PathBuf {
        self.path.join(Self::file_name_of_hash(hash))
    }

    fn write_data_if_new(&self, file_name: PathBuf, data: &[u8]) -> Result<(), StorageError> {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(file_name)
        {
            Ok(mut f) => f.write_all(data)?,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => return Err(StorageError::IoError(e)),
        }
        Ok(())
    }

    /// Store data and return its hash. The data is written to disk only if
    /// previously unseen.
    pub fn store(&self, data: &[u8]) -> Result<Hash, StorageError> {
        let hash = Hash::hash_bytes(data);
        let file_name = self.path_of_hash(&hash);
        self.write_data_if_new(file_name, data)?;
        Ok(hash)
    }
}

impl PersistentBlobStore for Store {
    fn init_from_path(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            std::fs::create_dir(&path)?;
        } else if path.metadata()?.is_file() {
            return Err(StorageError::InvalidRepo);
        }

        Ok(Store {
            path: path.into_boxed_path(),
        })
    }

    fn export_blob(&self, other: &mut Self, hash: &Hash) -> Result<(), StorageError> {
        let source_path = self.path_of_hash(hash);
        let target_path = other.path_of_hash(hash);
        std::fs::copy(source_path, target_path)?;
        Ok(())
    }
}

impl BlobStore for Store {
    fn blob_get(&self, hash: Hash) -> Result<impl AsRef<[u8]>, BlobStoreError> {
        let file_name = self.path_of_hash(&hash);
        std::fs::read(file_name).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                BlobStoreError::NotFound(hash)
            } else {
                BlobStoreError::Custom(Box::new(e))
            }
        })
    }

    fn blob_set<Data: AsRef<[u8]>>(&self, blob: &HashedData<Data>) -> Result<(), BlobStoreError> {
        let file_name = self.path_of_hash(&blob.hash());
        self.write_data_if_new(file_name, blob.data())
            .map_err(|e| BlobStoreError::Custom(Box::new(e)))?;
        Ok(())
    }

    fn blob_delete(&self, key: Hash) -> Result<(), BlobStoreError> {
        let file_name = self.path_of_hash(&key);
        match std::fs::remove_file(file_name) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(BlobStoreError::Custom(Box::new(e))),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Repo<BS> {
    backend: BS,
}

impl<BS: PersistentBlobStore> Repo<BS> {
    /// Load or create new repo at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        Ok(Repo {
            backend: BS::init_from_path(path)?,
        })
    }

    /// A snapshot is a new repo to which only `id` has been committed. This method exports an `id`
    /// assuming that it represents a Merkle tree of uniform depth one (the result of using
    /// `commit_serialised`, which chunks the serialisation).
    pub fn export_snapshot_chunked(
        &self,
        id: &Hash,
        path: impl AsRef<Path>,
    ) -> Result<(), StorageError> {
        // Only export a snapshot to a new or empty directory
        let path = path.as_ref();
        if !path.exists() || path.read_dir()?.next().is_none() {
            std::fs::create_dir_all(path)?;
        } else {
            return Err(StorageError::InvalidRepo);
        };
        let mut other = BS::init_from_path(path)?;
        let bytes = self.backend.blob_get(*id)?;
        let commit: Vec<Hash> = serialisation::deserialise(bytes.as_ref())?;
        for chunk in commit {
            self.backend.export_blob(&mut other, &chunk)?;
        }
        self.backend.export_blob(&mut other, id)?;
        Ok(())
    }

    /// A snapshot is a new repo to which only `id` has been committed. This method exports an `id`
    /// which represents any Merkle tree structure.
    ///
    /// Currently unimplemented.
    ///
    /// TODO (TZX-121)
    pub fn export_snapshot_folded(
        &self,
        _id: &Hash,
        _path: impl AsRef<Path>,
    ) -> Result<(), StorageError> {
        todo!()
    }
}

impl<BS: BlobStore> Repo<BS> {
    pub fn new(store: BS) -> Self {
        Repo { backend: store }
    }

    pub fn close(self) {}

    /// Create a new commit for `bytes` and return the commit id.
    pub fn commit(&self, bytes: &[u8]) -> Result<Hash, StorageError> {
        let mut commit = Vec::with_capacity(bytes.len().div_ceil(CHUNK_SIZE) * Hash::DIGEST_SIZE);

        for chunk in bytes.chunks(CHUNK_SIZE) {
            let hashed_chunk = HashedData::from_data(chunk);
            self.backend.blob_set(&hashed_chunk)?;
            commit.push(hashed_chunk.hash());
        }

        // A commit contains the list of all chunks needed to reconstruct `data`.
        let commit_bytes = serialisation::serialise(&commit)?;
        let hashed_commit_bytes = HashedData::from_data(commit_bytes);
        self.backend.blob_set(&hashed_commit_bytes)?;

        Ok(hashed_commit_bytes.hash())
    }

    /// Commit something serialisable and return the commit ID.
    pub fn commit_serialised(&self, subject: &impl Encode) -> Result<Hash, StorageError> {
        let chunk_hashes = {
            let mut writer = chunked_io::ChunkWriter::new(&self.backend);
            serialisation::serialise_into(subject, &mut writer)?;
            writer.finalise()?
        };

        // A commit contains the list of all chunks needed to reconstruct the underlying data.
        let commit_bytes = serialisation::serialise(&chunk_hashes)?;
        let hashed_commit_bytes = HashedData::from_data(commit_bytes);
        self.backend.blob_set(&hashed_commit_bytes)?;

        Ok(hashed_commit_bytes.hash())
    }

    /// Checkout the bytes committed under `id`, if the commit exists.
    pub fn checkout(&self, id: &Hash) -> Result<Vec<u8>, StorageError> {
        let bytes = self.backend.blob_get(*id)?;

        let commit: Vec<Hash> = serialisation::deserialise(bytes.as_ref())?;
        let mut bytes = Vec::new();

        for hash in commit {
            let chunk = self.backend.blob_get(hash).map_err(|e| {
                if let BlobStoreError::NotFound(hash) = e {
                    StorageError::ChunkNotFound(hash.to_string())
                } else {
                    StorageError::BlobStore(e)
                }
            })?;
            bytes.extend_from_slice(chunk.as_ref());
        }
        Ok(bytes)
    }

    /// Checkout something deserialisable from the store.
    pub fn checkout_serialised<S: Decode<()>>(&self, id: &Hash) -> Result<S, StorageError> {
        let mut reader = chunked_io::ChunkedReader::new(&self.backend, id)?;
        Ok(serialisation::deserialise_from(&mut reader)?)
    }
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::HashedData;
    use octez_riscv_data::store::BlobStore;
    use octez_riscv_data::store::BlobStoreError;

    use super::PersistentBlobStore;
    use super::Store;

    #[test]
    fn blob_store_test() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = Store::init_from_path(tmp_dir.path()).unwrap();

        let data1: &[u8] = &[3, 4, 5, 6, 8];
        let data2: &[u8] = b"Hi";

        let hash1 = Hash::hash_bytes(data1);
        let hash2 = Hash::hash_bytes(data2);

        store.blob_set(&HashedData::from_data(data1)).unwrap();
        store.blob_set(&HashedData::from_data(data2)).unwrap();

        assert_eq!(store.blob_get(hash1).unwrap().as_ref(), &[3, 4, 5, 6, 8]);
        assert_eq!(store.blob_get(hash2).unwrap().as_ref(), &[72, 105]);

        store.blob_delete(hash1).unwrap();

        match store.blob_get(hash1) {
            Err(BlobStoreError::NotFound(hash)) => {
                assert_eq!(hash, hash1)
            }
            _ => panic!("Expected `NotFound` error"),
        };

        // Both no-ops
        store.blob_set(&HashedData::from_data(data2)).unwrap();
        store.blob_delete(hash1).unwrap();

        assert_eq!(store.blob_get(hash2).unwrap().as_ref(), &[72, 105]);
    }
}
