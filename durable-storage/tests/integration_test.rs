#![cfg(test)]

use std::collections::HashMap;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use octez_riscv_durable_storage::key::KEY_MAX_SIZE;
use octez_riscv_durable_storage::key::Key;
use octez_riscv_durable_storage::registry::Registry;
use octez_riscv_durable_storage::repo::DirectoryManager;
use octez_riscv_test_utils::TestableTmpdir;
use proptest::prelude::*;
use proptest::proptest;
use proptest::sample::Index;

cfg_if::cfg_if! {
    if #[cfg(feature = "rocksdb")] {
        use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
        type TestKv = PersistenceLayer;
    } else {
        use octez_riscv_durable_storage::storage::in_memory::InMemoryKeyValueStore;
        type TestKv = InMemoryKeyValueStore;
    }
}

#[derive(Debug, Clone)]
enum Operation {
    // Database operations
    Set(Key, Bytes),
    Write(Key, usize, Bytes),
    Read(Key, usize, usize),
    Delete(Key),
    Exists(Key),
    ValueLength(Key),
    Hash,
    Commit,

    // Registry operations
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase,
    MoveDatabase,
    ClearDatabase,
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
        Operation::ShrinkRegistry,
        Operation::Set(Key::new(&[2]).unwrap(), Bytes::copy_from_slice(&[0; 10])),
        Operation::Exists(Key::new(&[1]).unwrap()),
        Operation::Delete(Key::new(&[1]).unwrap()),
        Operation::Exists(Key::new(&[1]).unwrap()),
        Operation::ShrinkRegistry,
        Operation::ShrinkRegistry,
    ];

    test_durable_storage_inner(operations)
}

const VALUE_MAX_SIZE: usize = 10_000;

#[derive(Debug, Clone)]
enum OperationView {
    // Database operations
    Set(Index, Index),
    Write(Index, usize, Index),
    Read(Index, usize, usize),
    Delete(Index),
    Exists(Index),
    ValueLength(Index),
    Hash,
    Commit,

    // Registry operations
    GrowRegistry,
    ShrinkRegistry,
    CopyDatabase,
    MoveDatabase,
    ClearDatabase,
}

proptest! {
    #[test]
    fn test_durable_storage_prop((keys, values, ops) in operations_strategy(1usize..100)) {
        // Auto formatting doesn't work within the proptest macro, so this passes off to a
        // standalone function.
        test_durable_storage(keys, values, ops)
    }
}

fn test_durable_storage(keys: Vec<Key>, values: Vec<Bytes>, ops: Vec<OperationView>) {
    // `OperationView` uses an index to choose keys and values from a generated set so that
    // proptest's shrinking algorithm can find a minimal-failing test with fewer iterations.
    let operations = ops
        .into_iter()
        .map(|op| match op {
            OperationView::Set(k_idx, v_idx) => Operation::Set(
                keys[k_idx.index(keys.len())].clone(),
                values[v_idx.index(values.len())].clone(),
            ),
            OperationView::Write(k_idx, offset, v_idx) => Operation::Write(
                keys[k_idx.index(keys.len())].clone(),
                offset,
                values[v_idx.index(values.len())].clone(),
            ),
            OperationView::Read(k_idx, offset, len) => {
                Operation::Read(keys[k_idx.index(keys.len())].clone(), offset, len)
            }
            OperationView::Delete(idx) => Operation::Delete(keys[idx.index(keys.len())].clone()),
            OperationView::Exists(idx) => Operation::Exists(keys[idx.index(keys.len())].clone()),
            OperationView::ValueLength(idx) => {
                Operation::ValueLength(keys[idx.index(keys.len())].clone())
            }
            OperationView::Hash => Operation::Hash,
            OperationView::Commit => Operation::Commit,
            OperationView::GrowRegistry => Operation::GrowRegistry,
            OperationView::ShrinkRegistry => Operation::ShrinkRegistry,
            OperationView::CopyDatabase => Operation::CopyDatabase,
            OperationView::MoveDatabase => Operation::MoveDatabase,
            OperationView::ClearDatabase => Operation::ClearDatabase,
        })
        .collect();
    test_durable_storage_inner(operations)
}

fn key_strategy() -> impl Strategy<Value = Key> {
    proptest::collection::vec(any::<u8>(), 1usize..=KEY_MAX_SIZE)
        .prop_map(|bytes| Key::new(&bytes).expect("The size is less than KEY_MAX_SIZE"))
}

fn value_strategy() -> impl Strategy<Value = Bytes> {
    proptest::collection::vec(any::<u8>(), 1usize..=VALUE_MAX_SIZE).prop_map(Bytes::from)
}

fn operations_strategy(
    length: impl Strategy<Value = usize>,
) -> impl Strategy<Value = (Vec<Key>, Vec<Bytes>, Vec<OperationView>)> {
    length.prop_flat_map(|length| {
        let count = length.div_ceil(10);
        let set = (any::<Index>(), any::<Index>()).prop_map(|(k, v)| OperationView::Set(k, v));

        let read = (
            any::<Index>(),
            prop_oneof![
                2 => Just(0),
                1 => 1..=VALUE_MAX_SIZE,
            ],
            0..=VALUE_MAX_SIZE,
        )
            .prop_map(|(k, off, len)| OperationView::Read(k, off, len));

        // Writes biased towards valid offsets
        let write_valid = (
            any::<Index>(),
            prop_oneof![
                2 => Just(0),
                1 => 1..=VALUE_MAX_SIZE,
            ],
            any::<Index>(),
        )
            .prop_map(|(k, off, v)| OperationView::Write(k, off, v));

        // Writes biased towards out-of-bounds offsets
        let write_invalid = (any::<Index>(), VALUE_MAX_SIZE..=usize::MAX, any::<Index>())
            .prop_map(|(k, off, v)| OperationView::Write(k, off, v));

        (
            proptest::collection::vec(key_strategy(), count),
            proptest::collection::vec(value_strategy(), count),
            // The chosen frequencies emulate real workloads
            proptest::collection::vec(
                prop_oneof![
                    // Database operations
                    20 => set,
                    20 => read,
                    3 => write_valid,
                    2 => write_invalid,

                    10 => any::<Index>().prop_map(OperationView::Delete),
                    10 => any::<Index>().prop_map(OperationView::Exists),
                    5 => any::<Index>().prop_map(OperationView::ValueLength),
                    10 => Just(OperationView::Hash),
                    3 => Just(OperationView::Commit),

                    // Registry operations
                    3 => Just(OperationView::GrowRegistry),
                    2 => Just(OperationView::ShrinkRegistry),
                    2 => Just(OperationView::CopyDatabase),
                    1 => Just(OperationView::MoveDatabase),
                    2 => Just(OperationView::ClearDatabase),
                ],
                length,
            ),
        )
    })
}

#[derive(Clone, Debug, Default)]
struct GoldenData {
    seen: HashMap<Key, Bytes>,
    last: Option<(Hash, HashMap<Key, Bytes>)>,
    ambiguous_hash: bool,
}

fn grow_registry(registry: &mut Registry<TestKv, Normal>, golden: &mut Vec<GoldenData>) {
    let new = registry.len();

    registry
        .resize(new.saturating_add(1))
        .expect("Resizing the registry should succeed");

    if let Some(previous) = new.checked_sub(1) {
        registry.copy_database(previous, new).unwrap();
    }

    if golden.is_empty() {
        golden.resize(1, Default::default());
    } else {
        golden.push(golden[golden.len() - 1].clone());
    }
}

// Get the index of the current database, or grow the registry until it has a valid index
fn get_index(registry: &mut Registry<TestKv, Normal>, golden: &mut Vec<GoldenData>) -> usize {
    if let Some(index) = registry.len().checked_sub(1) {
        index
    } else {
        grow_registry(registry, golden);
        get_index(registry, golden)
    }
}

fn update_value(value: &mut Bytes, offset: usize, bytes: Bytes) {
    let mut new_value: Vec<u8> = value.clone().into();
    let overwrite_len = std::cmp::min(bytes.len(), new_value.len().saturating_sub(offset));
    if overwrite_len > 0 {
        new_value[offset..offset + overwrite_len].copy_from_slice(&bytes[..overwrite_len]);
    }
    if bytes.len() > overwrite_len {
        new_value.extend_from_slice(&bytes[overwrite_len..]);
    }
    *value = Bytes::copy_from_slice(&new_value);
}

fn test_durable_storage_inner(operations: Vec<Operation>) {
    let tmpdir = TestableTmpdir::new();
    let base_dir = tmpdir.path().join("registry");
    let repo = DirectoryManager::new(&base_dir).expect("Failed to create manager");
    let mut registry: Registry<TestKv, Normal> =
        Registry::new(repo).expect("Creating the registry should succeed");

    let mut golden: Vec<GoldenData> = vec![];

    for operation in operations {
        match operation {
            // Database operations
            Operation::Set(key, bytes) => {
                let index = get_index(&mut registry, &mut golden);

                let data = Bytes::copy_from_slice(&bytes);

                registry
                    .database_mut(index)
                    .expect("The index is in bounds")
                    .set(key.clone(), data)
                    .expect("Setting should succeed");

                golden[index].seen.insert(key, bytes.clone());
            }
            Operation::Write(key, offset, bytes) => {
                let data = Bytes::copy_from_slice(&bytes);
                let index = get_index(&mut registry, &mut golden);

                let result = registry
                    .database_mut(index)
                    .expect("The index is in bounds")
                    .write(key.clone(), offset, data);

                let should_succeed = if let Some(map_value) = golden[index].seen.get_mut(&key) {
                    if offset > map_value.len() || offset.checked_add(bytes.len()).is_none() {
                        false
                    } else {
                        update_value(map_value, offset, bytes);
                        true
                    }
                } else if offset > 0 {
                    false
                } else {
                    golden[index].seen.insert(key, bytes);
                    true
                };

                if should_succeed {
                    assert!(
                        result.is_ok(),
                        "Write should have succeeded but failed: {:?}",
                        result.err()
                    );
                } else {
                    assert!(result.is_err(), "Write should have failed but succeeded");
                }
            }
            Operation::Read(key, offset, len) => {
                let index = get_index(&mut registry, &mut golden);
                let mut database_value = vec![0; len];

                let mut cursor = 0;
                let mut result = registry
                    .database(index)
                    .expect("The index is in bounds")
                    .read(&key, offset + cursor, &mut database_value[cursor..]);

                while let Ok(read) = result {
                    if read == 0 {
                        break;
                    }
                    cursor += read;
                    result = registry
                        .database(index)
                        .expect("The index is in bounds")
                        .read(&key, offset + cursor, &mut database_value[cursor..])
                }

                if let Some(map_value) = golden[index].seen.get(&key) {
                    if offset > map_value.len() {
                        assert!(result.is_err());
                    } else {
                        let expected_len = std::cmp::min(len, map_value.len() - offset);
                        assert!(cursor >= expected_len);
                        assert_eq!(
                            &database_value[..expected_len],
                            &map_value[offset..offset + expected_len]
                        );
                    }
                } else {
                    assert!(result.is_err());
                }
            }
            Operation::Delete(key) => {
                let index = get_index(&mut registry, &mut golden);

                // The hash of the `Database` can differ even if the key-value pairs stored are
                // the same, because deletion and reinsertion can cause the shape of the AVL
                // tree to change.
                let deleted = golden[index].seen.remove(&key).is_some();
                if deleted {
                    golden[index].ambiguous_hash = true;
                }

                registry
                    .database_mut(index)
                    .expect("The index is in bounds")
                    .delete(key)
                    .expect("Deleting should succeed");
            }
            Operation::Exists(key) => {
                let index = get_index(&mut registry, &mut golden);

                let in_database = registry
                    .database(index)
                    .expect("The index is in bounds")
                    .exists(&key)
                    .expect("Writing should succeed");
                let in_map = golden[index].seen.contains_key(&key);

                assert_eq!(in_database, in_map);
            }
            Operation::ValueLength(key) => {
                let index = get_index(&mut registry, &mut golden);

                let database_length = registry
                    .database(index)
                    .expect("The index is in bounds")
                    .value_length(&key);

                let map_value = golden[index].seen.get(&key);

                match (database_length, map_value) {
                    (Ok(database_length), Some(map_value)) => {
                        assert_eq!(database_length, map_value.len())
                    }
                    (Err(_), None) => (),
                    _ => panic!("The value exists in one map but not the other"),
                }
            }
            Operation::Hash => {
                let index = get_index(&mut registry, &mut golden);

                let new_digest = registry
                    .database(index)
                    .expect("The index is in bounds")
                    .hash()
                    .expect("Hash should succeed");

                if let (Some((old_digest, old_map)), false) =
                    (&golden[index].last, &golden[index].ambiguous_hash)
                {
                    assert_eq!(new_digest == *old_digest, golden[index].seen == *old_map);
                }

                golden[index].last = Some((new_digest, golden[index].seen.clone()));
            }
            Operation::Commit => {
                #[cfg(feature = "rocksdb")]
                registry.commit().expect("Committing should succeed");
            }

            // Registry operations
            Operation::GrowRegistry => grow_registry(&mut registry, &mut golden),
            Operation::ShrinkRegistry => {
                // Make sure there's a database to drop
                if registry.is_empty() {
                    grow_registry(&mut registry, &mut golden);
                };

                let new_size = registry.len().saturating_sub(1);
                registry
                    .resize(new_size)
                    .expect("Resizing the registry should succeed");

                golden.truncate(new_size);
            }
            Operation::ClearDatabase => {
                let index = get_index(&mut registry, &mut golden);

                registry
                    .clear_database(index)
                    .expect("Clearing the database should be successful");

                golden[index].seen.clear();
                golden[index].ambiguous_hash = false;
                golden[index].last = None;
            }
            Operation::CopyDatabase => {
                let (src, dst) = {
                    while registry.len() < 2 {
                        grow_registry(&mut registry, &mut golden);
                    }
                    (registry.len() - 2, registry.len() - 1)
                };

                registry
                    .copy_database(src, dst)
                    .expect("Copying the database should be successful");

                golden[dst] = golden[src].clone();
            }
            Operation::MoveDatabase => {
                let (src, dst) = {
                    while registry.len() < 2 {
                        grow_registry(&mut registry, &mut golden);
                    }
                    (registry.len() - 2, registry.len() - 1)
                };

                registry
                    .move_database(src, dst)
                    .expect("Moving the database should be successful");

                let empty = Default::default();
                let new_dst = std::mem::replace(&mut golden[src], empty);
                golden[dst] = new_dst;
            }
        }
    }
}
