// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::str::FromStr;

use crate::sbi_durable;

pub const MAX_STORE_V2_KEY_SIZE: usize = 256;
const MAX_VALUE_SIZE: usize = 4096;

#[derive(Debug)]
pub enum KeyError {
    KeyTooLarge,
}

#[repr(transparent)]
pub struct Key([u8]);

impl Key {
    const fn check_bytes(bytes: &[u8]) -> Result<(), KeyError> {
        if bytes.len() > MAX_STORE_V2_KEY_SIZE {
            return Err(KeyError::KeyTooLarge);
        }

        Ok(())
    }

    pub fn from_bytes(key: &[u8]) -> Result<&Self, KeyError> {
        Self::check_bytes(key)?;
        Ok(unsafe { std::mem::transmute::<&[u8], &Key>(key) })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl core::fmt::Debug for Key {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Key({self})")
    }
}

impl core::fmt::Display for Key {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match core::str::from_utf8(&self.0) {
            Ok(s) => f.write_str(s),
            Err(_) => write!(f, "Key({:02x?})", &self.0),
        }
    }
}

#[derive(Debug)]
pub enum NameError {}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Name(String);

impl core::fmt::Display for Name {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Name {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Name {
    type Err = NameError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Name(s.to_owned()))
    }
}

#[derive(Debug)]
pub enum KeySpaceError {
    Host(sbi_durable::Error),
    ValueTooLarge,
}

impl core::fmt::Display for KeySpaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Host(error) => write!(f, "keyspace host error: {error}"),
            Self::ValueTooLarge => f.write_str("keyspace value exceeds 4 KiB"),
        }
    }
}

impl From<sbi_durable::Error> for KeySpaceError {
    fn from(value: sbi_durable::Error) -> Self {
        Self::Host(value)
    }
}

#[allow(dead_code)]
pub trait KeySpace {
    type Error;

    fn get(&self, key: &Key) -> Result<Option<Vec<u8>>, Self::Error>;

    fn read(
        &self,
        key: &Key,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<Option<usize>, Self::Error>;

    fn set(&mut self, key: &Key, value: impl AsRef<[u8]>) -> Result<(), Self::Error>;

    fn write(
        &mut self,
        key: &Key,
        offset: usize,
        data: impl AsRef<[u8]>,
    ) -> Result<usize, Self::Error>;

    fn value_length(&self, key: &Key) -> Result<Option<usize>, Self::Error>;

    fn contains(&self, key: &Key) -> Result<bool, Self::Error>;

    fn delete(&mut self, key: &Key) -> Result<bool, Self::Error>;

    fn clear(&mut self) -> Result<(), Self::Error>;

    fn copy_from(&mut self, other: &Self) -> Result<(), Self::Error>;

    fn move_from(&mut self, other: &mut Self) -> Result<(), Self::Error>;

    fn hash(&self) -> Result<Vec<u8>, Self::Error>;
}

#[derive(Debug)]
pub enum KeySpaceLoaderError {
    Host(sbi_durable::Error),
    NameTooLarge,
}

impl core::fmt::Display for KeySpaceLoaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Host(error) => write!(f, "keyspace loader host error: {error}"),
            Self::NameTooLarge => f.write_str("keyspace name exceeds durable key size limit"),
        }
    }
}

impl From<sbi_durable::Error> for KeySpaceLoaderError {
    fn from(value: sbi_durable::Error) -> Self {
        Self::Host(value)
    }
}

pub trait KeySpaceLoader {
    type KeySpace: KeySpace;

    fn load_or_create(&mut self, name: Name) -> Result<&mut Self::KeySpace, KeySpaceLoaderError>;
}

#[derive(Debug, Clone)]
pub struct DurableKeySpace {
    index: usize,
}

impl DurableKeySpace {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

impl KeySpace for DurableKeySpace {
    type Error = KeySpaceError;

    fn get(&self, key: &Key) -> Result<Option<Vec<u8>>, Self::Error> {
        let Some(length) = self.value_length(key)? else {
            return Ok(None);
        };

        if length > MAX_VALUE_SIZE {
            return Err(KeySpaceError::ValueTooLarge);
        }

        let mut buffer = vec![0u8; length];
        let read = sbi_durable::database_read(self.index, key.as_bytes(), 0, &mut buffer)?;
        buffer.truncate(read);
        Ok(Some(buffer))
    }

    fn read(
        &self,
        key: &Key,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<Option<usize>, Self::Error> {
        if !self.contains(key)? {
            return Ok(None);
        }

        if buffer.len() > MAX_VALUE_SIZE {
            return Err(KeySpaceError::ValueTooLarge);
        }

        let read = sbi_durable::database_read(self.index, key.as_bytes(), offset, buffer)?;
        Ok(Some(read))
    }

    fn set(&mut self, key: &Key, value: impl AsRef<[u8]>) -> Result<(), Self::Error> {
        let value = value.as_ref();
        if value.len() > MAX_VALUE_SIZE {
            return Err(KeySpaceError::ValueTooLarge);
        }

        sbi_durable::database_set(self.index, key.as_bytes(), value)?;
        Ok(())
    }

    fn write(
        &mut self,
        key: &Key,
        offset: usize,
        data: impl AsRef<[u8]>,
    ) -> Result<usize, Self::Error> {
        let data = data.as_ref();
        if data.len() > MAX_VALUE_SIZE {
            return Err(KeySpaceError::ValueTooLarge);
        }

        Ok(sbi_durable::database_write(
            self.index,
            key.as_bytes(),
            offset,
            data,
        )?)
    }

    fn value_length(&self, key: &Key) -> Result<Option<usize>, Self::Error> {
        if !self.contains(key)? {
            return Ok(None);
        }

        Ok(Some(sbi_durable::database_value_length(
            self.index,
            key.as_bytes(),
        )?))
    }

    fn contains(&self, key: &Key) -> Result<bool, Self::Error> {
        Ok(sbi_durable::database_exists(self.index, key.as_bytes())?)
    }

    fn delete(&mut self, key: &Key) -> Result<bool, Self::Error> {
        Ok(sbi_durable::database_delete(self.index, key.as_bytes())?)
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        sbi_durable::registry_clear_database(self.index)?;
        Ok(())
    }

    fn copy_from(&mut self, other: &Self) -> Result<(), Self::Error> {
        sbi_durable::registry_copy_database(other.index, self.index)?;
        Ok(())
    }

    fn move_from(&mut self, other: &mut Self) -> Result<(), Self::Error> {
        sbi_durable::registry_move_database(other.index, self.index)?;
        Ok(())
    }

    fn hash(&self) -> Result<Vec<u8>, Self::Error> {
        Ok(sbi_durable::database_hash(self.index)?.to_vec())
    }
}

#[derive(Debug, Default)]
pub struct DurableKeySpaceLoader {
    keyspaces: BTreeMap<Name, DurableKeySpace>,
}

const KEYSPACE_INDEX_DATABASE: usize = 0;
const KEYSPACE_INDEX_PREFIX: &[u8] = b"/keyspaces/";

fn keyspace_index_key(name: &Name) -> Result<Vec<u8>, KeySpaceLoaderError> {
    let mut key = Vec::with_capacity(KEYSPACE_INDEX_PREFIX.len() + name.as_str().len());
    key.extend_from_slice(KEYSPACE_INDEX_PREFIX);
    key.extend_from_slice(name.as_str().as_bytes());

    if key.len() > MAX_STORE_V2_KEY_SIZE {
        return Err(KeySpaceLoaderError::NameTooLarge);
    }

    Ok(key)
}

fn decode_database_index(bytes: &[u8]) -> usize {
    let raw: [u8; 8] = bytes
        .try_into()
        .expect("persisted keyspace index must be encoded as u64");
    u64::from_le_bytes(raw) as usize
}

impl KeySpaceLoader for DurableKeySpaceLoader {
    type KeySpace = DurableKeySpace;

    fn load_or_create(&mut self, name: Name) -> Result<&mut Self::KeySpace, KeySpaceLoaderError> {
        if self.keyspaces.contains_key(&name) {
            return Ok(self
                .keyspaces
                .get_mut(&name)
                .expect("cached keyspace should remain present"));
        }

        let mut registry_len = sbi_durable::registry_len()?;
        if registry_len == 0 {
            sbi_durable::registry_resize_tick(1)?;
            registry_len = 1;
        }

        let mapping_key = keyspace_index_key(&name)?;
        let mapping_key_ref =
            Key::from_bytes(&mapping_key).expect("loader mapping key should be valid");

        let index =
            if sbi_durable::database_exists(KEYSPACE_INDEX_DATABASE, mapping_key_ref.as_bytes())? {
                let mapping_len = sbi_durable::database_value_length(
                    KEYSPACE_INDEX_DATABASE,
                    mapping_key_ref.as_bytes(),
                )?;
                let mut buffer = vec![0u8; mapping_len];
                let bytes_read = sbi_durable::database_read(
                    KEYSPACE_INDEX_DATABASE,
                    mapping_key_ref.as_bytes(),
                    0,
                    &mut buffer,
                )?;
                buffer.truncate(bytes_read);
                decode_database_index(&buffer)
            } else {
                let new_index = registry_len;
                sbi_durable::registry_resize_tick(new_index + 1)?;
                sbi_durable::database_set(
                    KEYSPACE_INDEX_DATABASE,
                    mapping_key_ref.as_bytes(),
                    &(new_index as u64).to_le_bytes(),
                )?;
                new_index
            };

        self.keyspaces
            .insert(name.clone(), DurableKeySpace::new(index));
        Ok(self
            .keyspaces
            .get_mut(&name)
            .expect("keyspace just inserted"))
    }
}
