// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Storage backends

pub mod in_memory;

use std::path::Path;
use std::sync::Arc;

use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use tempfile::TempDir;

use crate::commit::CommitId;
use crate::errors::Error;
use crate::errors::OperationalError;
use crate::merkle_worker::CommittedRoot;
use crate::merkle_worker::MerkleHandle;
use crate::merkle_worker::MerkleWorker;

/// Identifies one key-value store, so that data can be known to be in *this* store rather than
/// merely in some store.
///
/// A tree is shared between stores by cloning - [`crate::registry::Registry::copy_database`] hands
/// the destination a copy of the source's store and the same in-memory nodes. Whether a node still
/// needs storing therefore cannot be a single flag on the node: the same node owes a copy to each
/// store, and one store satisfying its obligation says nothing about the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreId(u64);

impl StoreId {
    /// The id no store has, meaning "not in any store".
    pub const NONE: Self = Self(0);

    /// Allocate an id no other store holds.
    pub fn next() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    /// The raw value, for recording compactly.
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for StoreId {
    fn from(raw: u64) -> Self {
        Self(raw)
    }
}

/// A freshly allocated id, not [`StoreId::NONE`].
///
/// Reached through `#[derive(Default)]` on stores that hold a [`StoreId`], such as
/// [`in_memory::InMemoryKeyValueStore`] - a defaulted store is empty and genuinely its own, so a
/// fresh identity is the right answer rather than [`StoreId::NONE`]. It is also the safe direction:
/// a store that wrongly believed it shared an identity would skip storing data it does not have,
/// where one with an identity of its own merely stores something again.
impl Default for StoreId {
    fn default() -> Self {
        Self::next()
    }
}

/// Types that implement this trait can serve reads from a key-value store.
///
/// This is the half of the store interface that requires no ability to modify the stored data.
/// Note that [`ReadableKeyValueStore::Repo`] lives here rather than on
/// [`WriteableKeyValueStore`]: a store which cannot be written to still needs to know the
/// repository its data is read from.
pub trait ReadableKeyValueStore: Sized {
    /// Type of repository required to initialise a key value store.
    type Repo;

    /// The identity of the store that Merkle node bodies are written to.
    ///
    /// Recorded on a node so that committing can tell it is already stored and skip writing it
    /// again. It identifies wherever [`ReadableKeyValueStore::node_get`] reads from, which is not
    /// necessarily this instance: a backend may keep nodes in a store shared by the whole
    /// repository, in which case every database of that repository reports the same identity and a
    /// node written through one is recognised by all of them. See [`StoreId`].
    fn store_id(&self) -> StoreId;

    /// How a Normal-mode database over this store tracks its Merkle root.
    ///
    /// Writeable stores are pinned to a live tree in a [`MerkleWorker`] by
    /// [`WriteableKeyValueStore`]; read-only stores are pinned to a [`CommittedRoot`] by
    /// [`ReadOnlyKeyValueStore`], and so have no tree and no worker thread at all.
    type Merkle: MerkleHandle<Self>;
    /// Retrieves the Merkle node body associated with the given hash.
    fn node_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error>;

    /// Retrieves the data associated with the given blob key.
    ///
    /// Blobs belong to whoever is using this store as one; the Merkle layer does not keep its nodes
    /// here. See [`ReadableKeyValueStore::node_get`].
    fn blob_get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error>;

    /// Retrieves a value associated with the given key.
    fn get(&self, key: impl AsRef<[u8]>) -> Result<impl AsRef<[u8]>, Error>;
}

/// Types that implement this trait can be used as the underlying key-value store
///
/// Writes change the root hash, so a database over such a store needs a live Merkle tree to track
/// it: [`ReadableKeyValueStore::Merkle`] is pinned to a [`MerkleWorker`] here.
pub trait WriteableKeyValueStore: ReadableKeyValueStore<Merkle = MerkleWorker<Self>> {
    /// Create a new instance of the key-value store.
    ///
    /// The backend may make use of the repo provided, for persistence - if required.
    fn new(repo: &Self::Repo) -> Result<Self, OperationalError>;

    /// Attempt to make a copy of the key-value store.
    fn try_clone(&self, repo: &Self::Repo) -> Result<Self, OperationalError>;

    /// Register a Merkle node body under its hash.
    fn node_set(
        &self,
        key: impl AsRef<[u8]>,
        data: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError>;

    /// Deletes the Merkle node body associated with the given hash.
    fn node_delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError>;

    /// Record that the node stored under `parent` refers to the one stored under `child`.
    ///
    /// Written as each node is stored, so that liveness can later be decided from a node upwards
    /// rather than by traversing every retained root downwards. Nodes are immutable, so an edge
    /// once true stays true until the parent itself is collected.
    fn edge_set(
        &self,
        child: impl AsRef<[u8]>,
        parent: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError>;

    /// Record the node under `key` as belonging to the commit at `seq`.
    ///
    /// What makes collection proportional to the garbage rather than to the store: a round scans
    /// only the nodes whose commit it is dropping, and never looks at one a retained commit wrote.
    fn node_written_at(
        &self,
        seq: crate::journal::Seq,
        key: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError>;

    /// Register data under a blob key.
    fn blob_set(
        &self,
        key: impl AsRef<[u8]>,
        data: impl AsRef<[u8]>,
    ) -> Result<(), OperationalError>;

    /// Deletes a value associated with the given blob key.
    fn blob_delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError>;

    /// Sets a value for the given key.
    fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), OperationalError>;

    /// Writes a value for the given key with a given offset.
    fn write(
        &self,
        key: impl AsRef<[u8]>,
        offset: usize,
        value: impl AsRef<[u8]>,
    ) -> Result<(), Error>;

    /// Deletes a value associated with the given key.
    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), OperationalError>;
}

/// Types that implement this trait can be used as a persistent key-value store
pub trait PersistentKeyValueStore: WriteableKeyValueStore + Sized {
    /// Commits the current state of the store to the given path.
    ///
    /// Implementations will treat the path as a directory.
    fn commit_to_path(&self, path: &Path) -> Result<(), OperationalError>;

    /// Write the current key-value state to the repository.
    fn commit(&self, repo: &Self::Repo, id: &CommitId) -> Result<(), OperationalError>;

    /// The position the commit about to be made will be recorded at.
    ///
    /// Asked of the store rather than the repository so that a backend which records nothing can
    /// say so without every caller having to know which kind of repository it has. Read before the
    /// nodes are written, since each is recorded as belonging to that commit.
    fn next_commit_seq(&self, repo: &Self::Repo) -> Result<crate::journal::Seq, OperationalError>;

    /// Checkout the state from `source_path` but leave it untouched. Modifications to the
    /// resulting state will be reflected in `working_path`.
    ///
    /// The repository is needed as well as the path because a commit directory need not hold
    /// everything the state is made of: a backend keeping Merkle node bodies in a store shared by
    /// the repository reaches them through `repo`, not through `source_path`.
    fn checkout_from_path(
        repo: &Self::Repo,
        source_path: &Path,
        working_path: TempDir,
    ) -> Result<Self, OperationalError>;

    /// Retrieve a key-value state from the repository.
    fn checkout(repo: &Self::Repo, id: &CommitId) -> Result<Self, OperationalError>;
}

/// An in-place, read-only view of a commit.
///
/// Stores implementing this trait should read the committed state where it lies. This is safe
/// as no mutable operations may be performed.
///
/// The root hash of such a state cannot change, and is already known - it is the commit id - so
/// [`ReadableKeyValueStore::Merkle`] is pinned to a [`CommittedRoot`] here. A database over a
/// read-only store therefore holds no Merkle tree, spawns no worker, and needs no async runtime.
pub trait ReadOnlyKeyValueStore: ReadableKeyValueStore<Merkle = CommittedRoot> {
    /// The store a read-only view can be upgraded into.
    type Writeable: PersistentKeyValueStore<Repo = Self::Repo>;

    /// Read the state of the given commit in the repository, without making a working copy.
    fn checkout_read_only(repo: &Self::Repo, id: &CommitId) -> Result<Self, OperationalError>;

    /// Read the state committed at `source_path`, without making a working copy.
    ///
    /// Takes the repository for the same reason as [`PersistentKeyValueStore::checkout_from_path`].
    fn checkout_read_only_from_path(
        repo: &Self::Repo,
        source_path: &Path,
    ) -> Result<Self, OperationalError>;

    /// Copy the committed state into a fresh working state, which can then be modified.
    fn to_writeable(&self, repo: &Self::Repo) -> Result<Self::Writeable, OperationalError>;
}

#[cfg(test)]
cfg_if::cfg_if! {
    if #[cfg(rocksdb)] {
        /// Key-value store backend used when the `rocksdb` feature is enabled.
        pub(crate) type TestKeyValueStore = crate::persistence_layer::PersistenceLayer;

        /// Repository type required to initialise [`TestKeyValueStore`].
        pub(crate) type TestRepo = <TestKeyValueStore as ReadableKeyValueStore>::Repo;

        /// Create a test repository for [`TestKeyValueStore`].
        ///
        /// Returns `(keepalive, repo)`:
        /// - `keepalive` is a temporary directory handle that must stay in scope for the lifetime
        ///   of `repo`.
        /// - `repo` is the backend repository value to pass into
        ///   [`WriteableKeyValueStore::new`] / [`WriteableKeyValueStore::try_clone`].
        ///
        /// TODO RV-942: Refactor the function to avoid the need for `keepalive` return value.
        pub(crate) fn setup_repo() -> (octez_riscv_test_utils::TestableTmpdir, TestRepo) {
            use crate::repo::DirectoryManager;

            let tmpdir = octez_riscv_test_utils::TestableTmpdir::new();
            let dir_manager = DirectoryManager::new(tmpdir.path()).expect("creating manager should succeed.");

            (tmpdir, dir_manager)
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(test)] {
        /// Type of storage backend
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Backend {
            /// In-memory backend (`InMemoryKeyValueStore`)
            InMemory,
            /// Persistent backend (`PersistenceLayer`)
            Persistent,
        }

        // TODO: This trait currently duplicates the functionality of the `TestKeyValueStore` mechanism defined above.
        // Tests which use the `kv_test!` macro will be refactored to use `TestKeyValueStoreSetup`, which can supersede
        // `WriteableKeyValueStore` once all tests have been rewritten.
        pub(crate) trait TestKeyValueStoreSetup: WriteableKeyValueStore + std::fmt::Debug {
            /// Temporary directory handle that must stay in scope for the lifetime of the repo
            type Keepalive;

            /// Allows tests to match on which backend is being used
            const BACKEND: Backend;

            /// Create a test repository
            fn setup_repo() -> (Self::Keepalive, Self::Repo);
        }

        impl TestKeyValueStoreSetup for in_memory::InMemoryKeyValueStore {
            type Keepalive = ();

            const BACKEND: Backend = Backend::InMemory;

            fn setup_repo() -> ((), in_memory::InMemoryRepo) {
                ((), in_memory::InMemoryRepo::default())
            }
        }

        #[cfg(rocksdb)]
        impl TestKeyValueStoreSetup for crate::persistence_layer::PersistenceLayer {
            type Keepalive = octez_riscv_test_utils::TestableTmpdir;

            const BACKEND: Backend = Backend::Persistent;

            fn setup_repo() -> (Self::Keepalive, Self::Repo) {
                use crate::repo::DirectoryManager;

                let tmpdir = octez_riscv_test_utils::TestableTmpdir::new();
                let dir_manager =
                    DirectoryManager::new(tmpdir.path()).expect("creating manager should succeed.");

                (tmpdir, dir_manager)
            }
        }

        /// Macro which runs a test against every available KV backend and compares traces.
        ///
        /// It can be used in 3 ways, one for unit tests, one for property-based tests with no
        /// setup before the proptest loop, and one for property-based tests with setup:
        ///
        /// 1. `kv_test!(name, KV: Bound, { body })`:
        ///    Expands to a single `#[test]` whose body runs once per available backend.
        ///    When the `rocksdb` flag is off the test only runs with the in-memory backend.
        ///    When it is on, the test runs twice, once with the persistence layer and once
        ///    with the in-memory backend.
        ///
        /// 2. `kv_test!(name, KV: Bound, [args in strategies], { body })`:
        ///    Expands to a single `#[test]` running one `proptest!` over the given strategies.
        ///    Each iteration runs the body once per available backend with the same
        ///    proptest-generated inputs.
        ///
        /// 3. `kv_test!(name, KV: Bound, <setup>, [args in strategies], { body })`:
        ///    Same as 2, with `<setup>` which runs once per backend before the proptest loop.
        ///    It must be one of:
        ///    - `setup |repo| = { ... }` for tests that only need a repo. The block
        ///      must return `(KV::Keepalive, KV::Repo)`. Inside the test body only
        ///      `repo: &KV::Repo` is exposed.
        ///    - `setup_runtime |handle, repo| = { ... }` for tests that need a Tokio
        ///      runtime. The block must return `(Runtime, Handle, KV::Keepalive, KV::Repo)`.
        ///      Inside the test body, `handle: &Handle` and `repo: &KV::Repo` are exposed.
        ///
        /// Note: For PBTs, in order to run the test body twice, the generated inputs are cloned.
        /// For this to be valid, cloning must create a deep copy. This means that proptests over
        /// pointers should not be run with this macro.
        ///
        /// In both cases, the "return" value of the body is taken to be a trace. The traces
        /// produced by the 2 runs / iterations are then compared.
        macro_rules! kv_test {
            // Property-based test variant with no setup.
            // Forwards to the shared `@prop_test` arm with an empty setup.
            ($(#[$attr:meta])* $fun_name:ident, $ty_name:ident $(: $ty_bound:path)?,
             [$($arg:ident in $strat:expr),* $(,)?],
             $body:block) => {
                $crate::storage::kv_test!(@prop_test
                    $(#[$attr])* $fun_name, $ty_name $(: $ty_bound)?,
                    ret_ty = (),
                    setup_values = _unused,
                    setup = {},
                    args = [$($arg in $strat),*],
                    body = $body
                );
            };

            // Property-based test variant with `setup`.
            // Forwards to the shared `@prop_test` arm with the setup tuple shape.
            ($(#[$attr:meta])* $fun_name:ident, $ty_name:ident $(: $ty_bound:path)?,
             setup |$repo:ident| = $setup:block,
             [$($arg:ident in $strat:expr),* $(,)?],
             $body:block) => {
                $crate::storage::kv_test!(@prop_test
                    $(#[$attr])* $fun_name, $ty_name $(: $ty_bound)?,
                    ret_ty = (
                        <$ty_name as $crate::storage::TestKeyValueStoreSetup>::Keepalive,
                        <$ty_name as $crate::storage::ReadableKeyValueStore>::Repo,
                    ),
                    setup_values = (_keepalive, $repo),
                    setup = $setup,
                    args = [$($arg in $strat),*],
                    body = $body
                );
            };

            // Property-based test variant with `setup_runtime`.
            // Forwards to the shared `@prop_test` arm with the setup tuple shape.
            ($(#[$attr:meta])* $fun_name:ident, $ty_name:ident $(: $ty_bound:path)?,
             setup_runtime |$handle:ident, $repo:ident| = $setup:block,
             [$($arg:ident in $strat:expr),* $(,)?],
             $body:block) => {
                $crate::storage::kv_test!(@prop_test
                    $(#[$attr])* $fun_name, $ty_name $(: $ty_bound)?,
                    ret_ty = (
                        ::tokio::runtime::Runtime,
                        ::tokio::runtime::Handle,
                        <$ty_name as $crate::storage::TestKeyValueStoreSetup>::Keepalive,
                        <$ty_name as $crate::storage::ReadableKeyValueStore>::Repo,
                    ),
                    setup_values = (_runtime, $handle, _keepalive, $repo),
                    setup = $setup,
                    args = [$($arg in $strat),*],
                    body = $body
                );
            };

            // Internal arm for property-based tests
            (@prop_test
             $(#[$attr:meta])* $fun_name:ident, $ty_name:ident $(: $ty_bound:path)?,
             ret_ty = $ret_ty:ty,
             setup_values = $setup_values:tt,
             setup = $setup:block,
             args = [$($arg:ident in $strat:expr),* $(,)?],
             body = $body:block) => {
                $(#[$attr])*
                #[test]
                fn $fun_name() {
                    fn _kv_test_setup<
                        $ty_name: $crate::storage::TestKeyValueStoreSetup
                                $(+ $ty_bound)?,
                    >() -> $ret_ty
                    where
                        <$ty_name as $crate::storage::ReadableKeyValueStore>::Repo:
                            $crate::repo::RegistryRepo,
                    {
                        $setup
                    }

                    #[cfg(rocksdb)]
                    let rocksdb_setup =
                        _kv_test_setup::<$crate::persistence_layer::PersistenceLayer>();

                    let in_memory_setup =
                        _kv_test_setup::<$crate::storage::in_memory::InMemoryKeyValueStore>();

                    ::proptest::proptest!(|($($arg in $strat),*)| {
                        #[cfg(rocksdb)]
                        let rocksdb_trace = {
                            type $ty_name = $crate::persistence_layer::PersistenceLayer;
                            let $setup_values = &rocksdb_setup;
                            eprintln!("Running test with persistence layer");
                            $(let $arg = ::std::clone::Clone::clone(&$arg);)*
                            $body
                        };

                        let _in_memory_trace = {
                            type $ty_name = $crate::storage::in_memory::InMemoryKeyValueStore;
                            let $setup_values = &in_memory_setup;
                            eprintln!("Running test with in-memory backend");
                            $body
                        };

                        #[cfg(rocksdb)]
                        ::proptest::prop_assert_eq!(
                            rocksdb_trace, _in_memory_trace,
                            "trace mismatch"
                        );
                    });
                }
            };

            // Unit test variant
            ($(#[$attr:meta])* $fun_name:ident, $ty_name:ident $(: $ty_bound:path)?, $body:block) => {
                $(#[$attr])*
                #[test]
                fn $fun_name() {
                    fn _bound_check<
                        $ty_name: $crate::storage::TestKeyValueStoreSetup
                                $(+ $ty_bound)?,
                    >()
                    where
                        <$ty_name as $crate::storage::ReadableKeyValueStore>::Repo:
                            $crate::repo::RegistryRepo,
                    {}

                    #[cfg(rocksdb)]
                    let rocksdb_trace = {
                        type $ty_name = $crate::persistence_layer::PersistenceLayer;
                        _bound_check::<$ty_name>();
                        eprintln!("Running test with persistence layer");
                        $body
                    };

                    let _in_memory_trace = {
                        type $ty_name = $crate::storage::in_memory::InMemoryKeyValueStore;
                        _bound_check::<$ty_name>();
                        eprintln!("Running test with in-memory backend");
                        $body
                    };

                    #[cfg(rocksdb)]
                    assert_eq!(
                        rocksdb_trace, _in_memory_trace,
                        "trace mismatch"
                    );
                }
            };
        }

        pub(crate) use kv_test;
    }
}

/// Options for storing MAVL values in a [`WriteableKeyValueStore`]
#[derive(Debug, Clone, Default)]
pub struct StoreOptions {
    /// Persist the key-value pairs from MAVL nodes
    node_data: bool,

    /// Which commit the nodes being stored belong to.
    ///
    /// Recorded against each node so that collection can find, without looking at the rest of the
    /// store, the nodes whose commit has since been dropped. Defaults to the very first position,
    /// which is the conservative answer: a node recorded there is examined by any collection rather
    /// than being skipped by one.
    written_at: crate::journal::Seq,
}

impl StoreOptions {
    /// Persists the key-value data of nodes.
    ///
    /// Turning this option on ensures the nodes are persisted completely. When using the
    /// [`crate::merkle_layer::MerkleLayer`] in isolation, this is necessary as there is no other
    /// component that will be writing the key-value data to the store.
    pub fn with_node_data(self) -> Self {
        Self {
            node_data: true,
            ..self
        }
    }

    /// Do not persist key-value data of nodes.
    ///
    /// This lets you avoid writing key-value data to the store when another component does this
    /// already. This is, for example, the case in the [`crate::database::Database`] component which
    /// mutates the store directly ahead of commitments. At commitment time, you only need to
    /// persist the remaining tree and node structures.
    pub fn without_node_data(self) -> Self {
        Self {
            node_data: false,
            ..self
        }
    }

    /// Returns whether node key-value data should be persisted.
    pub fn node_data(&self) -> bool {
        self.node_data
    }

    /// Record the nodes being stored as belonging to the commit at `seq`.
    pub fn written_at(self, seq: crate::journal::Seq) -> Self {
        Self {
            written_at: seq,
            ..self
        }
    }

    /// Which commit the nodes being stored belong to.
    pub fn commit_seq(&self) -> crate::journal::Seq {
        self.written_at
    }
}

/// This trait marks values that can be persisted into a [`WriteableKeyValueStore`].
pub trait Storable: Foldable<HashFold> {
    /// Persist this value into `store` according to `options`.
    fn store(
        &self,
        store: &impl WriteableKeyValueStore,
        options: &StoreOptions,
    ) -> Result<(), OperationalError>;
}

impl<T: Storable> Storable for Arc<T> {
    fn store(
        &self,
        store: &impl WriteableKeyValueStore,
        options: &StoreOptions,
    ) -> Result<(), OperationalError> {
        T::store(self, store, options)
    }
}

/// This trait marks values that can be reconstructed from a [`ReadableKeyValueStore`] by content hash.
pub trait Loadable: Sized {
    /// Load a value identified by `id` from `store`.
    fn load(id: Hash, store: &impl ReadableKeyValueStore) -> Result<Self, OperationalError>;
}

impl<T: Loadable> Loadable for Arc<T> {
    fn load(id: Hash, store: &impl ReadableKeyValueStore) -> Result<Self, OperationalError> {
        T::load(id, store).map(Arc::new)
    }
}
