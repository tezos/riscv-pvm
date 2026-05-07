// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! In-memory storage backend [`KeyValueStore`]-compatible with the Persistence layer

use std::collections::HashMap;
use std::sync::RwLock;

use bytes::Bytes;
use bytes::BytesMut;

use super::KeyValueStore;
use crate::errors::Error;
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;

/// Repository used by [`InMemoryKeyValueStore`].
///
/// Will never write to disk.
#[derive(Debug, Default, Clone)]
pub struct InMemoryRepo {
    #[cfg(any(test, feature = "unstable-test-utils"))]
    commits: std::sync::Arc<RwLock<HashMap<crate::commit::CommitId, InMemorySnapshot>>>,

    #[cfg(any(test, feature = "unstable-test-utils"))]
    registry_commits: std::sync::Arc<RwLock<HashMap<crate::commit::CommitId, Vec<u8>>>>,
}

/// In-memory key-value store
#[derive(Debug, Default)]
pub struct InMemoryKeyValueStore {
    /// Holds blobs.
    blobs: RwLock<HashMap<Bytes, Bytes>>,

    /// Holds the underlying key-value pairs
    values: RwLock<HashMap<Bytes, BytesMut>>,
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
        })
    }
}

impl KeyValueStore for InMemoryKeyValueStore {
    type Repo = InMemoryRepo;

    fn new(_repo: &Self::Repo) -> Result<Self, OperationalError> {
        Ok(Self::default())
    }

    fn try_clone(&self, _repo: &Self::Repo) -> Result<Self, OperationalError> {
        self.try_clone()
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

/// Test-only snapshot repository for [`InMemoryRepo`]
#[cfg(any(test, feature = "unstable-test-utils"))]
#[derive(Debug)]
struct InMemorySnapshot {
    blobs: HashMap<Bytes, Bytes>,
    values: HashMap<Bytes, BytesMut>,
}

#[cfg(any(test, feature = "unstable-test-utils"))]
impl super::PersistentKeyValueStore for InMemoryKeyValueStore {
    fn commit_to_path(&self, _path: &std::path::Path) -> Result<(), OperationalError> {
        unimplemented!("In-memory store cannot commit to disk")
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
            .clone();
        repo.commits
            .write()
            .map_err(|_| OperationalError::LockPoisoned)?
            .insert(*id, InMemorySnapshot { blobs, values });
        Ok(())
    }

    fn checkout_from_path(
        _source_path: &std::path::Path,
        _working_path: tempfile::TempDir,
    ) -> Result<Self, OperationalError> {
        unimplemented!("In-memory store cannot check out from disk")
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
            values: RwLock::new(snapshot.values.clone()),
        })
    }
}

#[cfg(any(test, feature = "unstable-test-utils"))]
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
