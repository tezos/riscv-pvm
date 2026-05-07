#![cfg(test)]

use bytes::Bytes;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::test_helpers::Operation;
use octez_riscv_durable_storage::test_helpers::make_operations;
use octez_riscv_durable_storage::test_helpers::operations_strategy;
use octez_riscv_durable_storage::test_helpers::run_operations;
use proptest::proptest;

cfg_if::cfg_if! {
    if #[cfg(feature = "rocksdb")] {
        use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
        use octez_riscv_durable_storage::repo::DirectoryManager;
        use octez_riscv_test_utils::TestableTmpdir;
        type TestKv = PersistenceLayer;

        const PROPTEST_CASES: u32 = 128;

        fn setup_repo() -> (TestableTmpdir, DirectoryManager) {
            let tmpdir = TestableTmpdir::new();
            let base_dir = tmpdir.path().join("registry");
            let repo = DirectoryManager::new(&base_dir).expect("Failed to create manager");
            (tmpdir, repo)
        }
    } else {
        use octez_riscv_durable_storage::storage::in_memory::InMemoryKeyValueStore;
        use octez_riscv_durable_storage::storage::in_memory::InMemoryRepo;
        type TestKv = InMemoryKeyValueStore;

        const PROPTEST_CASES: u32 = 256;

        fn setup_repo() -> ((), InMemoryRepo) {
            ((), InMemoryRepo::default())
        }
    }
}

#[test]
fn test_durable_storage_manual() {
    let operations = vec![
        Operation::GrowRegistry,
        Operation::Set(Key::new(&[0]).unwrap(), Bytes::copy_from_slice(&[0; 10])),
        Operation::Exists(Key::new(&[0]).unwrap()),
        Operation::Write(Key::new(&[0]).unwrap(), 5, Bytes::copy_from_slice(&[0; 4])),
        Operation::GrowRegistry,
        Operation::Set(Key::new(&[1]).unwrap(), Bytes::copy_from_slice(&[0; 10])),
        Operation::Commit,
        Operation::Checkout,
        Operation::ShrinkRegistry,
        Operation::Set(Key::new(&[2]).unwrap(), Bytes::copy_from_slice(&[0; 10])),
        Operation::Exists(Key::new(&[1]).unwrap()),
        Operation::Delete(Key::new(&[1]).unwrap()),
        Operation::Exists(Key::new(&[1]).unwrap()),
        Operation::ShrinkRegistry,
        Operation::ShrinkRegistry,
    ];

    let (_keepalive, repo) = setup_repo();
    run_operations::<TestKv>(repo, operations);
}

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(PROPTEST_CASES))]
    #[test]
    fn test_durable_storage_prop((keys, values, ops) in operations_strategy(1usize..100)) {
        let (_keepalive, repo) = setup_repo();
        let operations = make_operations(keys, values, ops);
        run_operations::<TestKv>(repo, operations);
    }
}
