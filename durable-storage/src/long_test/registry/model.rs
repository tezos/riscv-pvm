// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! In-memory reference model for the long-running [`Registry`] test.
//!
//! The registry model is a vector of database [`LongTestModel`]s plus the
//! number of leading `permanent` databases which may never be moved, cleared,
//! or copied into (see [`super::strategy`]).
//!
//! [`Registry`]: crate::registry::Registry

use crate::long_test::database::model::KeyPools;
use crate::long_test::database::model::LongTestModel;
use crate::test_helpers::database::DatabaseReferenceModel;
use crate::test_helpers::registry::RegistryOperation;

/// Reference model tracking each database's key/value store and the pools used
/// to guide operation generation. The model of a failing epoch can be persisted
/// alongside the durable storage commit and reloaded on replay.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RegistryLongTestModel {
    /// One reference model per database, indexed as in the registry.
    pub(crate) databases: Vec<LongTestModel>,
    /// The number of leading databases which cannot be moved, cleared, or
    /// copied into.
    pub(crate) permanent: usize,
}

impl RegistryLongTestModel {
    /// A model for a registry of `permanent` empty databases.
    pub fn new(permanent: usize) -> Self {
        RegistryLongTestModel {
            databases: vec![LongTestModel::default(); permanent],
            permanent,
        }
    }

    /// The number of databases currently modelled.
    pub(crate) fn len(&self) -> usize {
        self.databases.len()
    }

    /// The number of leading, permanent databases.
    pub fn permanent(&self) -> usize {
        self.permanent
    }

    /// A per-database snapshot of the key pools used to inform operation
    /// generation.
    pub fn pools(&self) -> Vec<KeyPools> {
        self.databases.iter().map(LongTestModel::pools).collect()
    }

    /// Total number of key/value pairs across all databases.
    pub(crate) fn total_entries(&self) -> usize {
        self.databases.iter().map(|db| db.data.len()).sum()
    }

    /// Update the model to reflect a successfully applied `operation`.
    pub fn apply(&mut self, operation: &RegistryOperation) {
        match operation {
            RegistryOperation::Database(index, op) => self.databases[*index].apply(op),
            RegistryOperation::GrowRegistry => {
                // Growing appends a fresh, empty database.
                self.databases.push(LongTestModel::default());
            }
            RegistryOperation::ShrinkRegistry => {
                if self.databases.len() > self.permanent {
                    self.databases.pop();
                }
            }
            RegistryOperation::CopyDatabase(src, dst) => {
                if src != dst {
                    self.databases[*dst] = self.databases[*src].clone();
                }
            }
            RegistryOperation::MoveDatabase(src, dst) => {
                if src != dst {
                    self.databases[*dst] = std::mem::take(&mut self.databases[*src]);
                }
            }
            RegistryOperation::ClearDatabase(index) => {
                self.databases[*index] = LongTestModel::default();
            }
            RegistryOperation::CommitCheckoutRoundtrip => {}
        }
    }
}
