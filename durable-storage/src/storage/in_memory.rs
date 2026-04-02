// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
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
#[derive(Debug, Copy, Clone)]
pub struct InMemoryRepo;

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
