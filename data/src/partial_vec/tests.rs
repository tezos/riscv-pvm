// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`PartialVec`]

use std::collections::BTreeMap;

use proptest::arbitrary::any;
use proptest::collection::vec;
use proptest::prelude::Strategy;
use proptest::proptest;

use crate::partial_vec::PartialVec;
use crate::partial_vec::RangeEntry;

/// Strategy to generate random byte vectors of limited length
fn data_vec() -> impl Strategy<Value = Vec<u8>> {
    vec(any::<u8>(), 0..1024)
}

proptest! {
    /// Defining a range anywhere should never fail.
    #[test]
    fn insert_randomly(init_data in vec((any::<usize>(), data_vec()), ..1024)) {
        let mut vec = PartialVec::empty();

        for (index, data) in init_data {
            vec.define(index, data);
        }
    }

    /// It should be able to extend a partially defined vector by defining more at the end.
    #[test]
    fn insert_end(init_len: usize, data in vec(data_vec(), ..1024)){
        let mut added_so_far = init_len;
        let mut vec = PartialVec::empty();

        for chunk in data {
            let chunk_len = chunk.len();
            vec.define(added_so_far, chunk.clone());

            let values = vec.continuous_defined_range(added_so_far..added_so_far + chunk_len).unwrap();
            let values = values.into_iter().flatten().copied().collect::<Vec<_>>();
            assert_eq!(values, chunk);

            added_so_far += chunk_len;
        }
    }

    /// `get` returns the correct value for any index in defined ranges.
    #[test]
    fn get_returns_defined_data(
        entries in vec((..100usize, data_vec()), ..16),
        query_indices in vec(..200usize, ..32)
    ) {
        let mut vec = PartialVec::empty();
        let mut reference = BTreeMap::new();

        // Build the PartialVec and a reference map (later writes overwrite earlier ones)
        for (offset, data) in entries {
            for (i, &val) in data.iter().enumerate() {
                reference.insert(offset.saturating_add(i), val);
            }

            vec.define(offset, data);
        }

        // Getting should match the reference for all queried indices
        for query_idx in query_indices {
            assert_eq!(vec.get(query_idx), reference.get(&query_idx));
            assert_eq!(vec.get_mut(query_idx), reference.get_mut(&query_idx));
        }
    }

    /// `get_mut` allows modifying values and subsequent `get` reflects the change.
    #[test]
    fn get_mut_modifies_data(
        offset in 0usize..1000,
        data in vec(any::<u8>(), 1..64),
        local_idx in any::<proptest::sample::Index>(),
        new_value in any::<u8>()
    ) {
        let mut vec = PartialVec::empty();
        vec.define(offset, data.clone());

        let local_idx = local_idx.index(data.len());
        let idx = offset + local_idx;

        // Modify via get_mut
        if let Some(elem) = vec.get_mut(idx) {
            *elem = new_value;
        }

        // Verify the modification
        assert_eq!(vec.get(idx), Some(&new_value));
    }

    /// `get` and `get_mut` return `None` for undefined indices.
    #[test]
    fn get_returns_none_for_undefined_indices(
        entries in vec((0usize..200, vec(any::<u8>(), 0..64)), 0..32),
        query_indices in vec(0usize..300, 1..64)
    ) {
        let mut vec = PartialVec::empty();
        let mut reference = BTreeMap::new();

        // Build the PartialVec and a reference map (later writes overwrite earlier ones)
        for (offset, data) in entries {
            for (i, &val) in data.iter().enumerate() {
                reference.insert(offset + i, val);
            }

            vec.define(offset, data);
        }

        let undefined_indices = query_indices
            .into_iter()
            .filter(|idx| !reference.contains_key(idx))
            .collect::<Vec<_>>();

        // Ensure the test run exercises the undefined-index path.
        proptest::prop_assume!(!undefined_indices.is_empty());

        for idx in undefined_indices {
            assert_eq!(vec.get(idx), None);
            assert_eq!(vec.get_mut(idx), None);
        }
    }
}

/// Ensure defining continuous ranges can be recovered.
#[test]
fn get_continuous() {
    let mut vec = PartialVec::empty();

    vec.define(10, Vec::from_iter(0..10));
    vec.define(20, Vec::from_iter(10..20));

    assert!(vec.is_any_defined(10..20));
    assert!(vec.is_all_defined(10..20));

    let values = vec
        .continuous_defined_range(10..30)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, Vec::from_iter(0..20));
}

/// Ensure defining ranges with gaps behaves correctly and is reflected in queries.
#[test]
fn get_with_gaps() {
    let mut vec = PartialVec::empty();

    // Don't define index 3
    vec.define(1, vec!["a", "b"]);
    vec.define(4, vec!["c", "d"]);

    assert_eq!(vec.continuous_defined_range(1..6), None);

    assert!(vec.is_any_defined(1..6));
    assert!(!vec.is_all_defined(1..6));

    // This vector tracks which indices are defined in the range.
    // We use `None` as an initial value to detect whether we covered the entire range by setting
    // each entry to `Some(_)`. E.g. if there are `None`s in the vector, we made a mistake.
    let mut defined = vec![None; 5];

    // Iterate through the retrieved range and populate the `defined` vector.
    vec.range(1..6).fold(defined.as_mut_slice(), |acc, item| {
        match item {
            RangeEntry::Undefined { length, .. } => {
                acc[..length].fill(Some(false));
            }

            RangeEntry::Defined { data, .. } => {
                acc[..data.len()].fill(Some(true));
            }
        }

        &mut acc[item.width()..]
    });

    assert_eq!(defined, vec![
        Some(true),
        Some(true),
        Some(false),
        Some(true),
        Some(true)
    ]);
}

/// Check if `is_all_defined` works in various scenarios where there are no gaps
#[test]
fn is_all_defined_basic() {
    let mut vec = PartialVec::empty();

    vec.define(10, Vec::from_iter(0u8..20));

    // Fully within the defined range
    assert!(vec.is_all_defined(10..30));
    assert!(vec.is_all_defined(15..25));
    assert!(vec.is_all_defined(10..20));

    // Overlapping with undefined at the start
    assert!(!vec.is_all_defined(5..15));

    // Overlapping with undefined at the end
    assert!(!vec.is_all_defined(25..35));

    // Entirely in undefined region
    assert!(!vec.is_all_defined(0..5));
    assert!(!vec.is_all_defined(100..200));

    // Empty range is vacuously continuously defined
    assert!(vec.is_all_defined(0..0));
    assert!(vec.is_all_defined(50..50));
}

/// Check if `is_all_defined` works in various scenarios where there are gaps.
#[test]
fn is_all_defined_with_gaps() {
    let mut vec = PartialVec::empty();

    // Create two defined regions with a gap
    vec.define(10, Vec::from_iter(0u8..10)); // 10..20
    vec.define(30, Vec::from_iter(0u8..10)); // 30..40

    // Each region individually
    assert!(vec.is_all_defined(10..20));
    assert!(vec.is_all_defined(30..40));

    // Spanning the gap
    assert!(!vec.is_all_defined(10..40));
    assert!(!vec.is_all_defined(15..35));

    // Just the gap
    assert!(!vec.is_all_defined(20..30));
}

/// Check if `is_all_defined` works when there are adjacent defined regions.
#[test]
fn is_all_defined_adjacent_regions() {
    let mut vec = PartialVec::empty();

    // Create adjacent defined regions (no gap)
    vec.define(10, Vec::from_iter(0u8..10)); // 10..20
    vec.define(20, Vec::from_iter(0u8..10)); // 20..30

    // Each region individually
    assert!(vec.is_all_defined(10..20));
    assert!(vec.is_all_defined(20..30));

    // Spanning both regions
    assert!(vec.is_all_defined(10..30));
    assert!(vec.is_all_defined(15..25));
}

/// Check that `is_any_defined` works in various scenarios without gaps.
#[test]
fn is_any_defined_basic() {
    let mut vec = PartialVec::empty();

    vec.define(10, Vec::from_iter(0u8..10)); // 10..20

    // Fully within the defined range
    assert!(vec.is_any_defined(10..20));
    assert!(vec.is_any_defined(12..18));

    // Overlapping with defined at the start
    assert!(vec.is_any_defined(5..15));

    // Overlapping with defined at the end
    assert!(vec.is_any_defined(15..25));

    // Entirely in undefined region
    assert!(!vec.is_any_defined(0..5));
    assert!(!vec.is_any_defined(100..200));

    // Empty range has no defined entries
    assert!(!vec.is_any_defined(0..0));
    assert!(!vec.is_any_defined(15..15));
}

/// Check that `is_any_defined` works in various scenarios with gaps.
#[test]
fn is_any_defined_with_gaps() {
    let mut vec = PartialVec::empty();

    // Create two defined regions with a gap
    vec.define(10, Vec::from_iter(0u8..10)); // 10..20
    vec.define(30, Vec::from_iter(0u8..10)); // 30..40

    // Each region individually
    assert!(vec.is_any_defined(10..20));
    assert!(vec.is_any_defined(30..40));

    // Spanning both regions and the gap
    assert!(vec.is_any_defined(10..40));
    assert!(vec.is_any_defined(15..35));

    // Just the gap (no defined data)
    assert!(!vec.is_any_defined(20..30));
    assert!(!vec.is_any_defined(22..28));
}

/// Check that truncation works correctly on a fully defined vector.
#[test]
fn truncate_defined_vector() {
    let mut vec = PartialVec::from(vec![1, 2, 3, 4, 5]);

    vec.truncate(3);

    let values = vec
        .continuous_defined_range(0..3)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 2, 3]);
}

/// Check that truncating to zero on a defined vector works correctly.
#[test]
fn truncate_to_zero_defined() {
    let mut vec = PartialVec::from(vec![1, 2, 3, 4, 5]);

    vec.truncate(0);

    // After truncating to 0, the vector should have no defined data
    assert!(!vec.is_any_defined(0..5));
}

/// Check that truncating an undefined vector works correctly.
#[test]
fn truncate_undefined_vector() {
    let mut vec: PartialVec<u8> = PartialVec::empty();
    vec.truncate(5);

    // After truncating, the undefined region should be 5 elements
    assert!(!vec.is_any_defined(0..5));

    // Check that the vector reports length 5
    let entries: Vec<_> = vec.range(0..5).collect();
    assert_eq!(entries.len(), 1);
    match entries[0] {
        RangeEntry::Undefined { length } => assert_eq!(length, 5),
        _ => panic!("Expected undefined entry"),
    }
}

/// Check that truncating a vector with undefined followed by defined data works correctly.
#[test]
fn truncate_with_undefined_prefix() {
    let mut vec: PartialVec<u8> = PartialVec::empty();
    vec.define(5, vec![1, 2, 3]);

    vec.truncate(3);

    // After truncation, we should have only 3 undefined elements
    let entries: Vec<_> = vec.range(0..3).collect();
    assert_eq!(entries.len(), 1);
    match &entries[0] {
        RangeEntry::Undefined { length } => assert_eq!(*length, 3),
        _ => panic!("Expected undefined entry"),
    }
}

/// Check that in-place replacement within an existing defined range works correctly.
///
/// This tests the case where new data fits entirely within an existing defined entry,
/// replacing elements without extending the range.
#[test]
fn define_inplace_replacement() {
    let mut vec = PartialVec::from(vec![1, 2, 3, 4, 5]);

    // Replace elements at indices 1 and 2 (values 2, 3) with 10, 20
    vec.define(1, vec![10, 20]);

    let values = vec
        .continuous_defined_range(0..5)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 10, 20, 4, 5]);
}

/// Check that in-place replacement at the start of a defined range works.
#[test]
fn define_inplace_replacement_at_start() {
    let mut vec = PartialVec::from(vec![1, 2, 3, 4, 5]);

    // Replace first two elements
    vec.define(0, vec![10, 20]);

    let values = vec
        .continuous_defined_range(0..5)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![10, 20, 3, 4, 5]);
}

/// Check that defining data spanning between two existing entries works.
#[test]
fn define_data_between_entries() {
    let mut vec: PartialVec<u8> = PartialVec::empty();

    // Now create a structure with defined regions on both sides
    vec.define(0, vec![1, 2, 3]); // Left side: indices 0-2
    vec.define(7, vec![7, 8, 9]); // Right side: indices 7-9

    // Now define data that spans across the middle, overlapping both regions
    // This should trigger the cross-boundary insertion code path
    vec.define(2, vec![20, 30, 40, 50, 60]);

    // Check that we can read back the defined portions as expected
    let values = vec
        .continuous_defined_range(0..10)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 2, 20, 30, 40, 50, 60, 7, 8, 9]);
}

/// Check truncation when keep_length is smaller than the extent of a single defined entry.
#[test]
fn truncate_single_entry() {
    let mut vec: PartialVec<u8> = PartialVec::empty();
    vec.define(3, vec![1, 2, 3, 4, 5]);

    // Truncate to 5 elements - this should keep the only entry (2 defined)
    vec.truncate(5);

    // Verify that indices 0-2 are undefined
    assert!(!vec.is_any_defined(0..3));

    // Verify that indices 3-4 are defined
    let values = vec
        .continuous_defined_range(3..5)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 2]);

    // Verify that elements beyond 5 are gone
    assert!(!vec.is_any_defined(5..8));
}

/// Check that `defined_range` returns only defined entries with correct offsets.
#[test]
fn defined_range_basic() {
    let mut vec: PartialVec<u8> = PartialVec::empty();

    vec.define(5, vec![1, 2, 3]); // indices 5-7
    vec.define(12, vec![4, 5, 6]); // indices 12-14

    // Query a range that includes both defined regions and gaps
    let entries: Vec<_> = vec.defined_range(0..20).collect();

    assert_eq!(entries.len(), 2);

    // First defined region starts at offset 5 in the queried range
    assert_eq!(entries[0].0, 5);
    assert_eq!(entries[0].1, &[1, 2, 3]);

    // Second defined region starts at offset 12 in the queried range
    assert_eq!(entries[1].0, 12);
    assert_eq!(entries[1].1, &[4, 5, 6]);
}

/// Check that `defined_range` works with partial overlaps.
#[test]
fn defined_range_partial_overlap() {
    let mut vec: PartialVec<u8> = PartialVec::empty();

    vec.define(5, vec![1, 2, 3, 4, 5]); // indices 5-9

    // Query a range that partially overlaps the defined region
    let entries: Vec<_> = vec.defined_range(7..15).collect();

    assert_eq!(entries.len(), 1);
    // The defined data starts at offset 0 in the queried range (7..15)
    // and contains elements from index 7 and 8 of the original vec
    assert_eq!(entries[0].0, 0);
    assert_eq!(entries[0].1, &[3, 4, 5]);
}

/// Check that `defined_range` returns no entries when there is no defined data.
#[test]
fn defined_range_no_defined_data() {
    let vec: PartialVec<u8> = PartialVec::empty();
    let entries: Vec<_> = vec.defined_range(0..20).collect();
    assert!(entries.is_empty());
}

/// Check that `is_all_undefined` returns true for a fully undefined vector.
#[test]
fn is_undefined_empty() {
    let vec: PartialVec<u8> = PartialVec::empty();
    assert!(vec.is_all_undefined());
}

/// Check that `is_all_undefined` returns true after truncating all data.
#[test]
fn is_undefined_after_truncate() {
    let mut vec = PartialVec::from(vec![1u8, 2, 3, 4, 5]);
    assert!(!vec.is_all_undefined());

    vec.truncate(0);
    assert!(vec.is_all_undefined());
}

/// Check that `is_all_undefined` returns false when there is defined data.
#[test]
fn is_undefined_with_defined_data() {
    let mut vec: PartialVec<u8> = PartialVec::empty();
    vec.define(50, vec![1, 2, 3]);
    assert!(!vec.is_all_undefined());
}

/// Check that `is_all_undefined` returns false for a fully defined vector.
#[test]
fn is_undefined_fully_defined() {
    let vec = PartialVec::from(vec![1u8, 2, 3, 4, 5]);
    assert!(!vec.is_all_undefined());
}

/// Check that `RangeEntry::is_empty` works correctly.
#[test]
fn range_entry_is_empty() {
    // Create a vector with defined and undefined regions
    let mut vec: PartialVec<u8> = PartialVec::empty();
    vec.define(3, vec![1, 2, 3]);

    // Get entries from the range and check is_empty on them
    let entries: Vec<_> = vec.range(0..10).collect();

    // We should have: undefined (0..3), defined (3..6), undefined (6..10)
    assert_eq!(entries.len(), 3);

    // All entries should be non-empty since they have length > 0
    for entry in &entries {
        assert!(!entry.is_empty());
    }

    // Test with an empty range - should produce no entries
    let empty_entries: Vec<_> = vec.range(0..0).collect();
    assert!(empty_entries.is_empty());
}

/// Check that defining a large number of small ranges does not cause stack overflow.
#[test]
fn stack_depth() {
    let mut vec: PartialVec<u8> = PartialVec::empty();

    for i in 0..40000 {
        vec.define(i * 2 + 1, vec![0]);
    }
}

/// Check defining data that starts before an existing entry and overlaps it.
///
/// This tests the `insert_entry` code path where `new_entry.start < existing_entry.start`,
/// which inserts a prefix of the new data before the existing entry.
#[test]
fn define_prefix_before_existing() {
    let mut vec: PartialVec<u8> = PartialVec::empty();

    // Create a defined region at indices 5-9
    vec.define(5, vec![5, 6, 7, 8, 9]);

    // Define data starting at index 2 that overlaps the existing region
    // This should insert [2, 3, 4] before the existing entry
    vec.define(2, vec![2, 3, 4, 50, 60]);

    // Indices 2-4 should be [2, 3, 4] (newly inserted prefix)
    // Indices 5-9 should be [50, 60, 7, 8, 9] (partially overwritten)
    let values = vec
        .continuous_defined_range(2..10)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![2, 3, 4, 50, 60, 7, 8, 9]);
}

/// Check that defining empty data is a no-op.
#[test]
fn define_empty_data() {
    let mut vec = PartialVec::from(vec![1u8, 2, 3, 4, 5]);

    // Define empty data at various positions
    vec.define(0, vec![]);
    vec.define(2, vec![]);
    vec.define(10, vec![]);

    // Original data should be unchanged
    let values = vec
        .continuous_defined_range(0..5)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}
