// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! In-memory storage backend [`WriteableKeyValueStore`]-compatible with the Persistence layer

use std::collections::HashMap;
#[cfg(test_utils)]
use std::io::Read;
#[cfg(test_utils)]
use std::io::Write;
use std::sync::RwLock;

use bytes::Bytes;
use bytes::BytesMut;

use super::ReadableKeyValueStore;
use super::StoreId;
use super::WriteableKeyValueStore;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::merkle_worker::MerkleWorker;

/// Repository used by [`InMemoryKeyValueStore`].
///
/// Will never write to disk.
#[derive(Debug, Default, Clone)]
pub struct InMemoryRepo {
    #[cfg(test_utils)]
    commits: std::sync::Arc<RwLock<HashMap<crate::commit::CommitId, InMemorySnapshot>>>,

    #[cfg(test_utils)]
    registry_commits: std::sync::Arc<RwLock<HashMap<crate::commit::CommitId, Vec<u8>>>>,
}

#[cfg(test_utils)]
impl InMemoryRepo {
    /// Remove the snapshot stored for `id` if it exists
    pub fn remove_commit(&self, id: &crate::commit::CommitId) -> Result<(), OperationalError> {
        self.commits
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?
            .remove(id);
        Ok(())
    }

    /// Remove the registry manifest stored for `id` if it exists
    #[cfg(rocksdb_test_utils)]
    pub(crate) fn remove_registry_commit(
        &self,
        id: &crate::commit::CommitId,
    ) -> Result<(), OperationalError> {
        self.registry_commits
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?
            .remove(id);
        Ok(())
    }
}

/// In-memory key-value store
#[derive(Debug, Default)]
pub struct InMemoryKeyValueStore {
    /// Holds blobs.
    blobs: RwLock<HashMap<Bytes, Bytes>>,

    /// Holds the underlying key-value pairs
    values: RwLock<HashMap<Bytes, BytesMut>>,

    /// Distinguishes this store from every other, including copies of it.
    store_id: StoreId,
}

impl InMemoryKeyValueStore {
    pub fn try_clone(&self) -> Result<Self, OperationalError> {
        let blobs = self
            .blobs
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?
            .clone();

        let values = self
            .values
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?
            .clone();

        Ok(Self {
            blobs: RwLock::new(blobs),
            values: RwLock::new(values),
            // A copy is its own store: it holds what the original held at this moment, and nothing
            // written to the original afterwards.
            store_id: StoreId::next(),
        })
    }
}

impl ReadableKeyValueStore for InMemoryKeyValueStore {
    type Repo = InMemoryRepo;

    type Merkle = MerkleWorker<Self>;
    fn store_id(&self) -> StoreId {
        self.store_id
    }

    fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        let blob_store = self
            .blobs
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?;

        let data = blob_store
            .get(key.as_ref())
            .ok_or(InvalidArgumentError::KeyNotFound)?;

        Ok(data.clone())
    }

    fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error> {
        let store = self
            .values
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?;

        let value = store
            .get(key.as_ref())
            .ok_or(InvalidArgumentError::KeyNotFound)?;

        Ok(value.clone())
    }
}

impl WriteableKeyValueStore for InMemoryKeyValueStore {
    fn new(_repo: &Self::Repo) -> Result<Self, OperationalError> {
        Ok(Self::default())
    }

    fn try_clone(&self, _repo: &Self::Repo) -> Result<Self, OperationalError> {
        self.try_clone()
    }

    fn blob_set(
        &self,
        key: impl AsRef<[u8]>,
        data: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError> {
        let mut blob_store = self
            .blobs
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?;

        blob_store.insert(
            Bytes::copy_from_slice(key.as_ref()),
            Bytes::copy_from_slice(data.as_ref()),
        );

        Ok(())
    }

    fn blob_delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        let mut blob_store = self
            .blobs
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?;

        blob_store.remove(key.as_ref());

        Ok(())
    }

    fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        let mut store = self
            .values
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?;

        let key = Bytes::copy_from_slice(key.as_ref());
        let value = BytesMut::from(Bytes::copy_from_slice(value.as_ref()));
        store.insert(key, value);

        Ok(())
    }

    fn write(
        &self,
        key: impl AsRef<[u8]>,
        offset: usize,
        value: impl AsRef<[u8]>,
    ) -> Result<(), Error> {
        let mut store = self
            .values
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?;

        let entry = if offset == 0 {
            let key = Bytes::copy_from_slice(key.as_ref());
            store.entry(key).or_default()
        } else {
            store
                .get_mut(key.as_ref())
                .ok_or(InvalidArgumentError::OffsetTooLarge)?
        };

        // Trying to write past the existing value is not allowed.
        if offset > entry.len() {
            return Err(InvalidArgumentError::OffsetTooLarge)?;
        }

        let value = value.as_ref();

        // Figure out which portion overlaps and consume that one first.
        let overlap = value.len().min(entry.len() - offset);
        entry[offset..][..overlap].copy_from_slice(&value[..overlap]);

        // The prefix of that value has been consumed.
        let value = &value[overlap..];

        // Nothing left? Ok, we're done.
        if value.is_empty() {
            return Ok(());
        }

        // Otherwise append the rest to the entry.
        entry.extend_from_slice(value);

        Ok(())
    }

    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError> {
        let mut store = self
            .values
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?;

        store.remove(key.as_ref());

        Ok(())
    }
}

/// Test-only snapshot of an [`InMemoryKeyValueStore`]
#[cfg(test_utils)]
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct InMemorySnapshot {
    blobs: HashMap<Bytes, Bytes>,
    values: HashMap<Bytes, Bytes>,
}

/// File name used within a commit directory by [`InMemoryKeyValueStore`].
#[cfg(test_utils)]
const STORE_FILE: &str = "in_memory_snapshot.bin";

#[cfg(test_utils)]
impl super::PersistentKeyValueStore for InMemoryKeyValueStore {
    fn commit_to_path(&self, path: &std::path::Path) -> Result<(), OperationalError> {
        let blobs = self
            .blobs
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?
            .clone();
        let values = self
            .values
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?
            .iter()
            .map(|(k, v)| (k.clone(), Bytes::copy_from_slice(v)))
            .collect();
        let snapshot = InMemorySnapshot { blobs, values };

        std::fs::create_dir_all(path).map_err(|error| OperationalError::DirCreationFailed {
            path: path.to_path_buf(),
            error,
        })?;
        let file = std::fs::File::create(path.join(STORE_FILE))
            .map_err(|error| OperationalError::FileWriteFailed { error })?;
        let writer = rkyv::ser::writer::IoWriter::new(std::io::BufWriter::new(file));

        rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(&snapshot, writer)
            .map_err(|error| OperationalError::FileWriteFailed {
                error: std::io::Error::other(error),
            })?
            .into_inner()
            .flush()
            .map_err(|error| OperationalError::FileWriteFailed { error })
    }

    fn commit(
        &self,
        repo: &InMemoryRepo,
        id: &crate::commit::CommitId,
    ) -> Result<(), OperationalError> {
        let blobs = self
            .blobs
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?
            .clone();
        let values = self
            .values
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?
            .iter()
            .map(|(k, v)| (k.clone(), Bytes::copy_from_slice(v)))
            .collect();
        repo.commits
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?
            .insert(*id, InMemorySnapshot { blobs, values });
        Ok(())
    }

    fn checkout_from_path(
        source_path: &std::path::Path,
        // The in-memory store keeps no working copy on disk
        _working_path: tempfile::TempDir,
    ) -> Result<Self, OperationalError> {
        let store_file = source_path.join(STORE_FILE);
        if !store_file.exists() {
            return Err(OperationalError::CommitNotFound);
        }

        let file = std::fs::File::open(&store_file)
            .map_err(|error| OperationalError::FileReadFailed { error })?;
        let mut reader = std::io::BufReader::new(file);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|error| OperationalError::FileReadFailed { error })?;

        // Fully deserialise the snapshot
        let snapshot =
            rkyv::from_bytes::<InMemorySnapshot, rkyv::rancor::Error>(&bytes).map_err(|error| {
                OperationalError::FileReadFailed {
                    error: std::io::Error::other(error),
                }
            })?;

        Ok(Self {
            blobs: RwLock::new(snapshot.blobs),
            values: RwLock::new(
                snapshot
                    .values
                    .into_iter()
                    .map(|(k, v)| (k, BytesMut::from(v.as_ref())))
                    .collect(),
            ),
            store_id: StoreId::next(),
        })
    }

    fn checkout(
        repo: &InMemoryRepo,
        id: &crate::commit::CommitId,
    ) -> Result<Self, OperationalError> {
        let commits = repo
            .commits
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?;
        let snapshot = commits.get(id).ok_or(OperationalError::CommitNotFound)?;
        Ok(Self {
            blobs: RwLock::new(snapshot.blobs.clone()),
            values: RwLock::new(
                snapshot
                    .values
                    .iter()
                    .map(|(k, v)| (k.clone(), BytesMut::from(v.as_ref())))
                    .collect(),
            ),
            store_id: StoreId::next(),
        })
    }
}

#[cfg(all(test, test_utils))]
mod tests {
    use octez_riscv_test_utils::TestableTmpdir;

    use super::*;
    use crate::storage::PersistentKeyValueStore;

    // Test for the commit to and checkout from path implementations for
    // `InMemoryKeyValueStore`, which are themselves only used in tests
    #[test]
    fn test_commit_to_path_checkout_roundtrip() {
        let store = InMemoryKeyValueStore::default();

        store
            .blob_set(b"blob-key", b"blob-data")
            .expect("Should be able to set a blob");
        store
            .set(b"/key/a", b"value-a")
            .expect("Should be able to set a value");
        store
            .set(b"/key/b", b"value-b")
            .expect("Should be able to set another value");
        store
            .write(b"/key/b", 5, b"b-amended")
            .expect("Should be able to write at an offset");
        store
            .set(b"/key/c", b"value-c")
            .expect("Should be able to set a third value");
        store
            .delete(b"/key/c")
            .expect("Should be able to delete a value");

        let commit_dir = TestableTmpdir::new();
        store
            .commit_to_path(commit_dir.path())
            .expect("Should be able to commit to a path");

        let working_path =
            tempfile::TempDir::new().expect("Should be able to create a working dir");
        let restored = InMemoryKeyValueStore::checkout_from_path(commit_dir.path(), working_path)
            .expect("Should be able to checkout from a path");

        assert_eq!(
            *store.blobs.read().expect("Lock should not be poisoned"),
            *restored.blobs.read().expect("Lock should not be poisoned"),
        );
        assert_eq!(
            *store.values.read().expect("Lock should not be poisoned"),
            *restored.values.read().expect("Lock should not be poisoned"),
        );
    }
}

#[cfg(test_utils)]
impl crate::repo::RegistryRepo for InMemoryRepo {
    fn read_registry_commit(
        &self,
        id: &crate::commit::CommitId,
    ) -> Result<Vec<u8>, OperationalError> {
        let commits = self
            .registry_commits
            .read()
            .map_err(|_| OperationalError::LockPoisoned)?;
        commits
            .get(id)
            .cloned()
            .ok_or(OperationalError::CommitNotFound)
    }

    fn write_registry_commit(
        &self,
        id: &crate::commit::CommitId,
        bytes: &[u8],
    ) -> Result<(), OperationalError> {
        let mut commits = self
            .registry_commits
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?;
        commits.insert(*id, bytes.to_vec());
        Ok(())
    }
}
