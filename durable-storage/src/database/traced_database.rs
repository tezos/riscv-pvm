// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

#![cfg(any(test, rocksdb_test_utils))]

//! [`Database`] wrapper which can record execution traces.
//!
//! Available to unit tests and, under the `unstable-test-utils` feature, to the
//! long-running test binary (`src/bin/long_test.rs`).

use std::cell::RefCell;
#[cfg(test)]
use std::collections::HashMap;

use bytes::Bytes;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::ProvableExt;
use octez_riscv_data::mode::Prove;
#[cfg(test)]
use octez_riscv_data::mode::Verify;
use tokio::runtime::Handle;

use crate::commit::CommitId;
use crate::database::Database;
use crate::database::DatabaseMode;
#[cfg(test)]
use crate::database::VerifyImpl;
use crate::errors::Error;
#[cfg(test)]
use crate::errors::InvalidArgumentError;
use crate::errors::OperationalError;
use crate::key::Key;
#[cfg(test)]
use crate::merkle_layer::new_verify_layer;
use crate::merkle_worker::BackgroundKeyValueStore;
use crate::merkle_worker::BackgroundPersistentKeyValueStore;
#[cfg(test)]
use crate::storage::KeyValueStore;
use crate::storage::PersistentKeyValueStore;
#[cfg(test)]
use crate::storage::TestKeyValueStoreSetup;

/// A sequence of recorded [`Database`] operations
pub(crate) type Trace = Vec<TraceEntry>;

/// The trace of a [`Database`] operation
#[serde_with::serde_as]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) enum TraceEntry {
    Set {
        key: Key,
        #[serde_as(as = "serde_with::hex::Hex")]
        data: Bytes,
        result: Result<(), String>,
    },
    Write {
        key: Key,
        offset: usize,
        #[serde_as(as = "serde_with::hex::Hex")]
        data: Bytes,
        result: Result<usize, String>,
    },
    Delete {
        key: Key,
        result: Result<(), String>,
    },
    Read {
        key: Key,
        offset: usize,
        #[serde_as(as = "Result<serde_with::hex::Hex, serde_with::Same>")]
        result: Result<Vec<u8>, String>,
    },
    ReadBytes {
        key: Key,
        offset: usize,
        max_bytes: usize,
        #[serde_as(as = "Result<serde_with::hex::Hex, serde_with::Same>")]
        result: Result<Vec<u8>, String>,
    },
    Exists {
        key: Key,
        result: Result<bool, String>,
    },
    ValueLength {
        key: Key,
        result: Result<usize, String>,
    },
    Hash {
        result: Result<Hash, String>,
    },
    Commit {
        result: Result<CommitId, String>,
    },
}

/// A [`Database`] wrapper which can record execution traces
pub(crate) struct TracedDatabase<KV, M: Mode = Normal> {
    inner: Database<KV, M>,
    trace: RefCell<Vec<TraceEntry>>,
}

impl<KV: BackgroundKeyValueStore> TracedDatabase<KV, Normal> {
    /// Equivalent to [`Database::try_new`] which also records a [`TraceEntry`].
    #[cfg(test)]
    pub(crate) fn try_new(handle: &Handle, repo: &KV::Repo) -> Result<Self, OperationalError> {
        Ok(TracedDatabase::from(Database::try_new(handle, repo)?))
    }

    /// Equivalent to [`Database::commit`] which also records a [`TraceEntry`].
    pub(crate) fn commit(&self, repo: &KV::Repo) -> Result<CommitId, OperationalError>
    where
        KV: PersistentKeyValueStore,
    {
        let result = self.inner.commit(repo);
        self.trace.borrow_mut().push(TraceEntry::Commit {
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::checkout`] which also records a [`TraceEntry`].
    pub(crate) fn checkout(
        handle: &Handle,
        repo: &KV::Repo,
        commit_id: CommitId,
    ) -> Result<Self, Error>
    where
        KV: BackgroundPersistentKeyValueStore,
    {
        Ok(TracedDatabase::from(Database::checkout(
            handle, repo, commit_id,
        )?))
    }

    /// Access the inner database directly, bypassing tracing.
    ///
    /// Used for test setup operations that need to bypass API-layer validation
    /// (e.g. setting values larger than MAX_FILE_CHUNK_SIZE).
    pub(crate) fn inner_mut(&mut self) -> &mut Database<KV, Normal> {
        &mut self.inner
    }

    /// Immutable access to the inner database, bypassing tracing.
    #[cfg(any(test, rocksdb_test_utils))]
    pub(crate) fn inner(&self) -> &Database<KV, Normal> {
        &self.inner
    }
}

#[cfg(test)]
impl<KV> TracedDatabase<KV, Verify>
where
    KV: KeyValueStore + TestKeyValueStoreSetup,
{
    pub(crate) fn new_verify(repo: &KV::Repo) -> Self {
        TracedDatabase::from(Database {
            inner: VerifyImpl {
                merkle: new_verify_layer::<KV>(repo),
            },
        })
    }
}

impl<KV: BackgroundKeyValueStore, M: DatabaseMode> TracedDatabase<KV, M> {
    /// Equivalent to [`Database::set`] which also records a [`TraceEntry`].
    pub(crate) fn set(&mut self, key: Key, data: Bytes) -> Result<(), Error> {
        let result = self.inner.set(key.clone(), data.clone());
        self.trace.borrow_mut().push(TraceEntry::Set {
            key,
            data,
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::write`] which also records a [`TraceEntry`].
    pub(crate) fn write(&mut self, key: Key, offset: usize, data: Bytes) -> Result<usize, Error> {
        let result = self.inner.write(key.clone(), offset, data.clone());
        self.trace.borrow_mut().push(TraceEntry::Write {
            key,
            offset,
            data,
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::delete`] which also records a [`TraceEntry`].
    pub(crate) fn delete(&mut self, key: Key) -> Result<(), OperationalError> {
        let result = self.inner.delete(key.clone());
        self.trace.borrow_mut().push(TraceEntry::Delete {
            key,
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::read`] which also records a [`TraceEntry`].
    pub(crate) fn read(&self, key: &Key, offset: usize, output: &mut [u8]) -> Result<usize, Error> {
        let result = self.inner.read(key, offset, &mut *output);
        let trace_res = match &result {
            Ok(n) => Ok(output[..*n].to_vec()),
            Err(e) => Err(format!("{e:?}")),
        };
        self.trace.borrow_mut().push(TraceEntry::Read {
            key: key.clone(),
            offset,
            result: trace_res,
        });
        result
    }

    /// Equivalent to [`Database::read_bytes`] which also records a [`TraceEntry`].
    #[cfg(test)]
    pub(crate) fn read_bytes(
        &self,
        key: &Key,
        offset: usize,
        max_bytes: usize,
    ) -> Result<Vec<u8>, Error> {
        let result = self
            .inner
            .read_bytes(key, offset, max_bytes)
            .map(|slice| slice.as_ref().to_vec());
        self.trace.borrow_mut().push(TraceEntry::ReadBytes {
            key: key.clone(),
            offset,
            max_bytes,
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::exists`] which also records a [`TraceEntry`].
    pub(crate) fn exists(&self, key: &Key) -> Result<bool, Error> {
        let result = self.inner.exists(key);
        self.trace.borrow_mut().push(TraceEntry::Exists {
            key: key.clone(),
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::value_length`] which also records a [`TraceEntry`].
    pub(crate) fn value_length(&self, key: &Key) -> Result<usize, Error> {
        let result = self.inner.value_length(key);
        self.trace.borrow_mut().push(TraceEntry::ValueLength {
            key: key.clone(),
            result: trace_result(&result),
        });
        result
    }

    /// Equivalent to [`Database::hash`] which also records a [`TraceEntry`].
    pub(crate) fn hash(&self) -> Result<Hash, OperationalError> {
        let result = self.inner.hash();
        self.trace.borrow_mut().push(TraceEntry::Hash {
            result: trace_result(&result),
        });
        result
    }

    /// Get the recorded trace.
    pub(crate) fn into_trace(self) -> Trace {
        self.trace.into_inner()
    }

    /// Get both the inner database and the recorded trace.
    pub(crate) fn into_parts(self) -> (Database<KV, M>, Trace) {
        (self.inner, self.trace.into_inner())
    }

    /// Insert entries into the database and return the inserted key pairs
    #[cfg(test)]
    pub(crate) fn insert_entries(
        &mut self,
        entries: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> HashMap<Key, Bytes> {
        let mut expected = HashMap::new();
        for (key, value) in entries {
            let key = Key::new(&key).expect("Size less than KEY_MAX_SIZE");
            let value = Bytes::copy_from_slice(&value);
            self.set(key.clone(), value.clone())
                .expect("Writing should succeed");
            expected.insert(key, value);
        }
        expected
    }

    /// Assert that a database contains the expected value for a given key.
    #[cfg(test)]
    pub(crate) fn assert_database_value(&self, key: &Key, expected: &[u8]) {
        self.inner.assert_database_value(key, expected);
    }

    /// Assert that a database does not contain the given key.
    #[cfg(test)]
    pub(crate) fn assert_traced_database_missing(&self, key: &Key) {
        assert!(matches!(
            self.read_bytes(key, 0, 0),
            Err(Error::InvalidArgument(InvalidArgumentError::KeyNotFound))
        ));
    }
}

impl<KV, M: Mode> From<Database<KV, M>> for TracedDatabase<KV, M> {
    fn from(inner: Database<KV, M>) -> Self {
        Self {
            inner,
            trace: RefCell::new(Vec::new()),
        }
    }
}

impl<KV, M: Mode, F: Fold> Foldable<F> for TracedDatabase<KV, M>
where
    Database<KV, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        self.inner.fold(builder)
    }
}

impl<'normal, KV: BackgroundKeyValueStore> ProvableExt<'normal, 'static, OperationalError>
    for TracedDatabase<KV, Normal>
{
    type Prover = TracedDatabase<KV, Prove<'static>>;

    fn try_start_proof(&'normal self) -> Result<Self::Prover, OperationalError> {
        let prove_db = self.inner.try_start_proof()?;
        let prover = TracedDatabase::from(prove_db);

        Ok(prover)
    }
}

fn trace_result<T: Clone, E: std::fmt::Debug>(result: &Result<T, E>) -> Result<T, String> {
    match result {
        Ok(v) => Ok(v.clone()),
        Err(e) => Err(format!("{e:?}")),
    }
}
