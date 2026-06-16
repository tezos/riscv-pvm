// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! In-memory reference model for the long-running [`Database`] tests
//!
//! [`Database`]: crate::database::Database

use std::collections::HashMap;
use std::collections::VecDeque;

use bytes::Bytes;
use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

use crate::key::Key;
use crate::test_helpers::database::DatabaseOperation;
use crate::test_helpers::database::DatabaseReferenceModel;

/// Maximum number of keys retained in the hot / recently-deleted pools.
const POOL_CAP: usize = 64;

/// Snapshot of the model's key pools, used as an input for operation
/// generation strategies
#[derive(Clone, Debug, Default)]
pub struct KeyPools {
    /// All present keys, sorted.
    pub existing: Vec<Key>,
    /// Recently written or read keys.
    pub hot: Vec<Key>,
    /// Recently deleted keys.
    pub deleted: Vec<Key>,
}

/// Reference model which tracks the key/value store and extra state used
/// to guide the operation generation strategy. The model of a failing epoch
/// can be persisted alongside the durable storage commit and reloaded on replay.
#[serde_with::serde_as]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LongTestModel {
    #[serde_as(as = "HashMap<_, serde_with::hex::Hex>")]
    pub(crate) data: HashMap<Key, Bytes>,
    hot: VecDeque<Key>,
    recently_deleted: VecDeque<Key>,
}

impl LongTestModel {
    /// A snapshot of the key pools used to inform operation generation.
    /// The existing keys are sorted in order to make the operations strategy
    /// reproducible.
    pub fn pools(&self) -> KeyPools {
        let mut existing: Vec<Key> = self.data.keys().cloned().collect();
        existing.sort();
        KeyPools {
            existing,
            hot: self.hot.iter().cloned().collect(),
            deleted: self.recently_deleted.iter().cloned().collect(),
        }
    }

    fn touch_hot(&mut self, key: &Key) {
        // When key becomes hot, remove from recently deleted set
        self.recently_deleted.retain(|k| k != key);

        if !self.hot.contains(key) {
            self.hot.push_back(key.clone());
            while self.hot.len() > POOL_CAP {
                self.hot.pop_front();
            }
        }
    }

    fn mark_deleted(&mut self, key: &Key) {
        // When key is deleted, remove from hot set
        self.hot.retain(|k| k != key);

        if !self.recently_deleted.contains(key) {
            self.recently_deleted.push_back(key.clone());
            while self.recently_deleted.len() > POOL_CAP {
                self.recently_deleted.pop_front();
            }
        }
    }
}

impl DatabaseReferenceModel for LongTestModel {
    fn data(&self) -> &HashMap<Key, Bytes> {
        &self.data
    }

    fn apply(&mut self, operation: &DatabaseOperation) {
        match operation {
            DatabaseOperation::Set(key, data) => {
                if data.len() <= MAX_FILE_CHUNK_SIZE {
                    self.data.insert(key.clone(), data.clone());
                    self.touch_hot(key);
                }
            }
            DatabaseOperation::Write(key, offset, data) => {
                if let Some(new_value) = self.write_outcome(key, *offset, data) {
                    self.data.insert(key.clone(), new_value);
                    self.touch_hot(key);
                }
            }
            DatabaseOperation::Read(key, _, _) => {
                if self.data.contains_key(key) {
                    self.touch_hot(key);
                }
            }
            DatabaseOperation::Delete(key) => {
                if self.data.remove(key).is_some() {
                    self.mark_deleted(key);
                }
            }
            DatabaseOperation::Exists(_)
            | DatabaseOperation::ValueLength(_)
            | DatabaseOperation::Hash
            | DatabaseOperation::Commit
            | DatabaseOperation::Checkout
            | DatabaseOperation::CommitCheckoutRoundtrip => {}
        }
    }
}
