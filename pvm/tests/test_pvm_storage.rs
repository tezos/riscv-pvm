// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

#![cfg(test)]

use std::fs::File;

use octez_riscv::machine_state::memory::M1M;
use octez_riscv::machine_state::page_cache::EmptyPageCache;
use octez_riscv::pvm::Pvm;
use octez_riscv::pvm::durable_storage::DurableStorageDummy;
use octez_riscv::pvm::node_pvm::NodePvm;
use octez_riscv::pvm::node_pvm::PvmStorage;
use octez_riscv::storage::Repo;
use octez_riscv::storage::StorageError;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::store::BlobStoreError;
use octez_riscv_test_utils::TestableTmpdir;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

macro_rules! repo_test {
    (
        $ty_name:ident,
        $(#[$attr:meta])*
        fn $fn_name:ident() $body:block
    ) => {
        paste::paste! {
            $(#[$attr])*
            #[test]
            fn $fn_name() {
                fn inner<$ty_name: octez_riscv::storage::PersistentBlobStore>() $body
                inner::<octez_riscv::storage::Store>();
            }

            $(#[$attr])*
            #[cfg(feature = "rocksdb")]
            #[test]
            fn [<$fn_name _rocksdb>]() {
                fn inner<$ty_name: octez_riscv::storage::PersistentBlobStore>() $body
                inner::<octez_riscv::storage::rocksdb_store::RocksDBStore>();
            }
        }
    }
}

repo_test!(
    TestStore,
    fn test_repo() {
        let mut runner = TestRunner::default();

        let tmp_dir = TestableTmpdir::new();
        let mut test_data = Vec::new();

        // Create a new repo, commit 5 times and check that all 5 commits can
        // be checked out
        let repo = Repo::<TestStore>::load(tmp_dir.path()).unwrap();
        for _ in 0..5 {
            let data = prop::collection::vec(any::<u8>(), 0..100)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let commit_id = repo.commit(&data).unwrap();
            test_data.push((commit_id, data));
        }
        for (commit_id, bytes) in test_data.iter() {
            let checked_out_data = repo.checkout(commit_id).unwrap();
            assert_eq!(checked_out_data, *bytes);
        }
        repo.close();

        // Reload the repo and check that all previous commits can be checked out
        let repo = Repo::<TestStore>::load(tmp_dir.path()).unwrap();
        for (commit_id, bytes) in test_data.iter() {
            let checked_out_data = repo.checkout(commit_id).unwrap();
            assert_eq!(checked_out_data, *bytes);
        }

        // Make 5 additional commits and check that all 10 commits can be checked out
        for _ in 0..5 {
            let data = prop::collection::vec(any::<u8>(), 0..100)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let commit_id = repo.commit(&data).unwrap();
            test_data.push((commit_id, data));
        }
        for (commit_id, bytes) in test_data.iter() {
            let checked_out_data = repo.checkout(commit_id).unwrap();
            assert_eq!(checked_out_data, *bytes);
        }

        // Check that an unknown commit returns a `NotFound` error
        let unknown_hash: Hash = [0u8; Hash::DIGEST_SIZE].into();
        assert!(matches!(
            repo.checkout(&unknown_hash),
            Err(StorageError::BlobStore(BlobStoreError::NotFound(_)))
        ));

        // Check that exporting a snapshot creates a new repo which contains
        // the requested commit
        let snapshot_dir = TestableTmpdir::new();
        let (export_hash, export_data) = &test_data[0];
        repo.export_snapshot_chunked(*export_hash, snapshot_dir.path())
            .unwrap();
        let snapshot_repo = Repo::<TestStore>::load(snapshot_dir.path()).unwrap();
        let checked_out_data = snapshot_repo.checkout(export_hash).unwrap();
        assert_eq!(checked_out_data, *export_data);
    }
);

repo_test!(
    TestStore,
    fn test_repo_serialised() {
        let mut runner = TestRunner::default();

        let tmp_dir = TestableTmpdir::new();
        let mut test_data = Vec::new();

        // Create a new repo, commit 5 times and check that all 5 commits can
        // be checked out
        let repo = Repo::<TestStore>::load(tmp_dir.path()).unwrap();
        for _ in 0..5 {
            let data = prop::collection::vec(any::<u8>(), 0..100)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let commit_id = repo.commit_serialised(&data).unwrap();
            test_data.push((commit_id, data));
        }
        for (commit_id, bytes) in test_data.iter() {
            let checked_out_data: Vec<u8> = repo.checkout_serialised(commit_id).unwrap();
            assert_eq!(checked_out_data, *bytes);
        }

        repo.close();

        // Reload the repo and check that all previous commits can be checked out
        let repo = Repo::<TestStore>::load(tmp_dir.path()).unwrap();

        for (commit_id, bytes) in test_data.iter() {
            let checked_out_data: Vec<u8> = repo.checkout_serialised(commit_id).unwrap();
            assert_eq!(checked_out_data, *bytes);
        }

        // Make 5 additional commits and check that all 10 commits can be checked out
        for _ in 0..5 {
            let data = prop::collection::vec(any::<u8>(), 0..100)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let commit_id = repo.commit_serialised(&data).unwrap();
            test_data.push((commit_id, data));
        }
        for (commit_id, bytes) in test_data.iter() {
            let checked_out_data: Vec<u8> = repo.checkout_serialised(commit_id).unwrap();
            assert_eq!(checked_out_data, *bytes);
        }

        // Check that an unknown commit returns a `NotFound` error
        let unknown_hash: Hash = [0u8; Hash::DIGEST_SIZE].into();
        assert!(matches!(
            repo.checkout_serialised::<Vec<u8>>(&unknown_hash),
            Err(StorageError::BlobStore(BlobStoreError::NotFound(_)))
        ));

        // Check that exporting a snapshot creates a new repo which contains
        // the requested commit
        let snapshot_dir = TestableTmpdir::new();
        let (export_hash, export_data) = &test_data[0];
        repo.export_snapshot_chunked(*export_hash, snapshot_dir.path())
            .unwrap();
        let snapshot_repo = Repo::<TestStore>::load(snapshot_dir.path()).unwrap();
        let checked_out_data: Vec<u8> = snapshot_repo.checkout_serialised(export_hash).unwrap();
        assert_eq!(checked_out_data, *export_data);
    }
);

type TestPvm = Pvm<M1M, EmptyPageCache, DurableStorageDummy, Normal>;

repo_test!(
    TestStore,
    fn test_repo_folding() {
        let tmp_dir = TestableTmpdir::new();
        let repo = Repo::<TestStore>::load(tmp_dir.path()).unwrap();

        let empty_pvm = TestPvm::default();
        let hash = repo.commit_folded(&empty_pvm).unwrap();

        repo.close();

        let repo = Repo::<TestStore>::load(tmp_dir.path()).unwrap();
        let checked_out = repo.checkout_folded::<TestPvm>(&hash).unwrap();

        // We use `assert!` to avoid very verbose test failures
        assert!(empty_pvm == checked_out);

        let snapshot_dir = TestableTmpdir::new();
        repo.export_snapshot_folded::<TestPvm>(hash, snapshot_dir.path())
            .unwrap();

        let snapshot_repo = Repo::<TestStore>::load(snapshot_dir.path()).unwrap();
        let checked_out_from_snapshot = snapshot_repo.checkout_folded::<TestPvm>(&hash).unwrap();

        assert!(empty_pvm == checked_out_from_snapshot);
    }
);

// Mirrors `src/lib_riscv/pvm/test/test_storage.ml`
repo_test!(
    TestStore,
    fn test_pvm_storage() {
        let tmp_dir = TestableTmpdir::new();
        let empty = NodePvm::empty();
        let mut repo = PvmStorage::<TestStore>::load(tmp_dir.path()).unwrap();
        let id = repo.commit(&empty).unwrap();
        let checked_out_empty = repo.checkout(&id).unwrap();
        assert_eq!(empty, checked_out_empty);
        let id2 = repo.commit(&empty).unwrap();
        assert_eq!(id, id2);
        repo.close()
    }
);

repo_test!(
    TestStore,
    fn test_invalid_repo() {
        // Error if we try to initialise a repo with a path that is a file, not a directory.
        let tmp_dir = TestableTmpdir::new();
        let tmp_file_path = tmp_dir.path().join("blah");
        let _tmp_file = File::create(tmp_file_path.as_path()).unwrap();
        let load_result = Repo::<TestStore>::load(tmp_file_path.as_path());

        println!("{:?}", load_result.as_ref().err());
        assert!(matches!(
            load_result,
            Err(StorageError::InvalidRepo | StorageError::RocksDBError(_))
        ));

        // Error if we try to export a snapshot to non-empty directory.
        let tmp_dir_2 = TestableTmpdir::new();
        let repo = Repo::<TestStore>::load(tmp_dir_2.path()).unwrap();

        let data = vec![];
        let id = repo.commit(&data).unwrap();
        let export_result = repo.export_snapshot_chunked(id, tmp_dir.path());

        assert!(matches!(export_result, Err(StorageError::InvalidRepo)));

        // Error if we try to export a snapshot to a path that is a file, not a directory.
        let export_result = repo.export_snapshot_chunked(id, tmp_file_path);
        assert!(matches!(export_result, Err(StorageError::InvalidRepo)));
    }
);
