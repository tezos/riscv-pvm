// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`PartialVec`]

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
    fn insert_randomly(init_len: usize, init_data in vec((any::<usize>(), data_vec()), ..1024)) {
        let mut vec = PartialVec::undefined(init_len);

        for (index, data) in init_data {
            vec.define(index, data);
        }
    }

    /// It should be able to extend a partially defined vector by defining more at the end.
    #[test]
    fn insert_end(init_len: usize, data in vec(data_vec(), ..1024)){
        let mut added_so_far = init_len;
        let mut vec = PartialVec::undefined(init_len);

        for chunk in data {
            let chunk_len = chunk.len();
            vec.define(added_so_far, chunk.clone());

            let values = vec.continuous_defined_range(added_so_far..added_so_far + chunk_len).unwrap();
            let values = values.into_iter().flatten().copied().collect::<Vec<_>>();
            assert_eq!(values, chunk);

            added_so_far += chunk_len;
        }
    }
}

/// Ensure defining continuous ranges can be recovered.
#[test]
fn get_continuous() {
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::undefined(1024);

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
    let mut vec = PartialVec::defined(vec![1, 2, 3, 4, 5]);

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
    let mut vec = PartialVec::defined(vec![1, 2, 3, 4, 5]);

    vec.truncate(0);

    // After truncating to 0, the vector should have no defined data
    assert!(!vec.is_any_defined(0..5));
}

/// Check that truncating an undefined vector works correctly.
#[test]
fn truncate_undefined_vector() {
    let mut vec: PartialVec<u8> = PartialVec::undefined(10);
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
fn truncate_concatenated_with_undefined_left() {
    let mut vec: PartialVec<u8> = PartialVec::undefined(5);
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
    let mut vec = PartialVec::defined(vec![1, 2, 3, 4, 5]);

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
    let mut vec = PartialVec::defined(vec![1, 2, 3, 4, 5]);

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

/// Check that defining data spanning both left and right nodes works correctly.
///
/// This tests the cross-boundary insertion case where the inserted range overlaps
/// with both left and right children of a Concatenated node.
#[test]
fn define_spanning_left_and_right() {
    let mut vec: PartialVec<u8> = PartialVec::undefined(10);

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

/// Check that defining data across a Defined+Undefined boundary triggers the optimization.
///
/// This tests the special case in define() where the left child is Defined
/// and the right child is Undefined, allowing in-place extension.
#[test]
fn define_across_defined_undefined_boundary() {
    // Create structure: Defined[1,2,3] ++ Undefined(5)
    // by starting with undefined and defining at offset 0
    let mut vec: PartialVec<u8> = PartialVec::undefined(8);
    vec.define(0, vec![1, 2, 3]);

    // Now we have a Concatenated node: Defined[1,2,3] (left) ++ Undefined(5) (right)
    // Define data starting within the defined region and extending into undefined.
    // This should trigger the optimization path at lines 217-226.
    vec.define(1, vec![10, 20, 30, 40]);

    // Check that we can read back the defined portions
    // The result should be [1, 10, 20, 30, 40] at indices 0-4
    let values = vec
        .continuous_defined_range(0..5)
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 10, 20, 30, 40]);
}

/// Check truncation when keep_length is greater than left node width.
///
/// This tests the case where truncation needs to continue into the right child
/// of a Concatenated node.
#[test]
fn truncate_into_right_child() {
    // Create a Concatenated structure by starting with undefined and defining data.
    // This creates: Undefined(3) ++ Defined[1,2,3,4,5]
    let mut vec: PartialVec<u8> = PartialVec::undefined(3);
    vec.define(3, vec![1, 2, 3, 4, 5]);

    // Truncate to 5 elements - this should keep the left node (3 undefined)
    // and continue into the right node (keeping 2 defined elements)
    vec.truncate(5);

    // Verify that indices 0-2 are undefined
    assert!(!vec.is_all_defined(0..3));

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
    let mut vec: PartialVec<u8> = PartialVec::undefined(20);

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
    let mut vec: PartialVec<u8> = PartialVec::undefined(20);

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
    let vec: PartialVec<u8> = PartialVec::undefined(20);
    let entries: Vec<_> = vec.defined_range(0..20).collect();
    assert!(entries.is_empty());
}

/// Check that `is_all_undefined` returns true for a fully undefined vector.
#[test]
fn is_undefined_empty() {
    let vec: PartialVec<u8> = PartialVec::undefined(100);
    assert!(vec.is_all_undefined());
}

/// Check that `is_all_undefined` returns true for a zero-length vector.
#[test]
fn is_undefined_zero_length() {
    let vec: PartialVec<u8> = PartialVec::undefined(0);
    assert!(vec.is_all_undefined());
}

/// Check that `is_all_undefined` returns false when there is defined data.
#[test]
fn is_undefined_with_defined_data() {
    let mut vec: PartialVec<u8> = PartialVec::undefined(100);
    vec.define(50, vec![1, 2, 3]);
    assert!(!vec.is_all_undefined());
}

/// Check that `is_all_undefined` returns false for a fully defined vector.
#[test]
fn is_undefined_fully_defined() {
    let vec = PartialVec::defined(vec![1u8, 2, 3, 4, 5]);
    assert!(!vec.is_all_undefined());
}

/// Check that `RangeEntry::is_empty` works correctly.
#[test]
fn range_entry_is_empty() {
    // Create a vector with defined and undefined regions
    let mut vec: PartialVec<u8> = PartialVec::undefined(10);
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
