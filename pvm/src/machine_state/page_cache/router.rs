// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The [`Router`] tracks ranges of pages which are executable and non-writable.
//!
//! Each range is associated with a compiler, which will usually be a [`JIT`] instance or a means
//! of sending compilation requests to a [`JIT`] instance. The [`Router`] is responsible for
//! instantiating new compilers and extending their range when it is safe to do so.
//!
//! When we run in 'interpreted mode' the compiler will just be an empty struct. In this case, the
//! [`Router`] is really no more than a range set, but it still has a vital role to play: because
//! it keeps track of which page ranges are executable, it allows the page cache to more
//! efficiently invalidate just the relevant pages when a large section of memory is marked as
//! writable.
//!
//! The [`Router`] has three jobs:
//!
//! - To give a reference to the compiler for a given page index. This is required when the page
//!   cache populates a new page in the range: the compiler reference is stored alongside the page
//!   for very fast lookup as it is executed.
//!
//! - To instantiate new ranges when they have become executable. This involves instantiating new
//!   compilers or extending existing ones where possible.
//!
//! - To invalidate ranges when they have become writable. When a page is marked as writable the
//!   compiler associated with that page becomes invalid for the whole of its range. The [`Router`]
//!   will give the page cache an iterator of newly-invalid ranges which it can delete.
//!
//! [`JIT`]: crate::jit::JIT

use std::ops::RangeInclusive;

use perfect_derive::perfect_derive;

/// This trait allows the compiler type to specify its own logic for merging ranges.
pub trait RouterEq {
    /// Should specify when `self` and `other` are identical, in the sense that we can merge their
    /// page ranges when possible
    fn router_eq(&self, other: &Self) -> bool;
}

/// Internal utility type. An item in the [`Router`]'s map.
#[perfect_derive(Debug, Default, Clone)]
struct RouterTarget<T>(T);

/// This `PartialEq` implementation copies the [`RouterEq`] implementation for `T` itself.
impl<T: RouterEq> PartialEq for RouterTarget<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.router_eq(&other.0)
    }
}

/// The core router type: it tracks ranges covered by each compiler, handles extending them when
/// their [`RouterEq`] impl says to do so (see [`add_range`]) and removes ranges when any part of
/// them becomes invalid (see [`drain_overlapping`]).
///
/// [`add_range`]: Router::add_range
/// [`drain_overlapping`]: Router::drain_overlapping
#[perfect_derive(Default, Debug, Clone)]
pub struct Router<T: Clone + RouterEq> {
    internal: rangemap::RangeInclusiveMap<u64, RouterTarget<T>>,
}

impl<T: Clone + RouterEq> Router<T> {
    /// Inserts a new range without removing any existing ones it overlaps with. This involves
    /// creating new entries for the gaps and merging those into existing entries when the
    /// [`RouterEq`] implementation for `T` tells us to.
    ///
    /// We pass an explicit `constructor` method to handle the actual creation of new `T` values.
    pub fn add_range(&mut self, new: RangeInclusive<u64>, constructor: impl Fn() -> T) {
        // We extend by one so that the `overlapping` iterator will give us touching entries as
        // well as overlapping ones.
        let extended = dilate_by_one(&new);
        let overlapping_or_touching = self
            .internal
            .overlapping(&extended)
            .map(|(r, t)| RouterEntry {
                range: r.clone(),
                target: Some(t.clone()),
            })
            .collect::<Vec<_>>();

        let entries_count = overlapping_or_touching.len();

        // 'hot path' case with no touching entries is a single insert with no pre-existing target
        if entries_count == 0 {
            self.insert_entry(RouterEntry::from_range(new), &constructor);
            return;
        }

        // NB: `rangemap::RangeInclusiveMap` uses `PartialEq` to merge the ranges of 'equal'
        // targets. This means that when we call insert entry below, ranges that touch will be
        // merged iff their targets are `router_eq` to one another.

        // gap at start, if it exists
        if overlapping_or_touching[0].range.start() > new.start() {
            self.insert_entry(
                overlapping_or_touching[0].start_gap(*new.start()),
                &constructor,
            );
        }

        for i in 0..(entries_count - 1) {
            let lhs = &overlapping_or_touching[i];
            let rhs = &overlapping_or_touching[i + 1];

            // each middle gap, if it exists
            if !lhs.is_touching(rhs) {
                self.insert_entry(lhs.gap(rhs), &constructor);
            }
        }

        // gap at end, if it exists
        if overlapping_or_touching[entries_count - 1].range.end() < new.end() {
            self.insert_entry(
                overlapping_or_touching[entries_count - 1].end_gap(*new.end()),
                &constructor,
            );
        }
    }

    /// Acquire a reference to a target `T`.
    pub fn get(&self, idx: &u64) -> Option<&T> {
        self.internal.get(idx).map(|t| &t.0)
    }

    /// When any part of a range is invalidated we should remove the whole entry. This removes any
    /// overlapping ranges and returns them as an iterator which the page cache can use to clean
    /// up the invalidated pages.
    pub fn drain_overlapping(
        &mut self,
        range: RangeInclusive<u64>,
    ) -> impl Iterator<Item = RangeInclusive<u64>> {
        let ranges = self
            .internal
            .overlapping(&range)
            .map(|(r, _)| r.clone())
            .collect::<Vec<_>>();
        let range_count = ranges.len();
        if range_count != 0 {
            let total_range = *ranges[0].start()..=*ranges[range_count - 1].end();
            self.internal.remove(total_range);
        }
        ranges.into_iter()
    }

    /// Utility method that inserts an entry, either with the given target or with a new
    /// (created using `constructor`) target if `target == None`.
    fn insert_entry(&mut self, entry: RouterEntry<T>, constructor: impl Fn() -> T) {
        self.internal.insert(
            entry.range,
            entry.target.unwrap_or_else(|| RouterTarget(constructor())),
        );
    }
}

/// A utility type representing an entry in a [`Router`] that handles the merging logic.
struct RouterEntry<T> {
    range: RangeInclusive<u64>,

    /// If `Some`, represents an existing entry or a new entry with a specified target. If `None`,
    /// represents a new entry for which we will create a new `default` target.
    target: Option<RouterTarget<T>>,
}

impl<T: Clone> RouterEntry<T> {
    fn from_range(range: RangeInclusive<u64>) -> Self {
        Self {
            range,
            target: None,
        }
    }

    fn is_longer_or_eq(&self, other: &Self) -> bool {
        self.range.end() - self.range.start() >= other.range.end() - other.range.start()
    }

    /// Two entries are 'touching' if there is no overlap but also no gap between them.
    ///
    /// Assumptions:
    ///
    /// - `self` is before `other`.
    fn is_touching(&self, other: &Self) -> bool {
        self.range.end().saturating_add(1) == *other.range.start()
    }

    /// Create a `RouterEntry` for the gap between `self` and `other`. If either can be extended
    /// we clone its target to the new entry; if both are safe to extend we choose the longer one,
    /// defaulting to the left hand side (`self`).
    ///
    /// Assumptions:
    ///
    /// - `self` is before `other`,
    /// - they are neither touching nor overlapping.
    fn gap(&self, other: &Self) -> Self {
        let target = match (self.target.is_some(), other.target.is_some()) {
            (false, false) => None,
            (true, false) => self.target.clone(),
            (false, true) => other.target.clone(),
            _ if self.is_longer_or_eq(other) => self.target.clone(),
            _ => other.target.clone(),
        };
        Self {
            range: self.range.end().saturating_add(1)..=other.range.start().saturating_sub(1),
            target,
        }
    }

    /// Create a new entry for the gap between `gap_start` and `self`. If `self` can be extended we
    /// clone its target to the new entry.
    ///
    /// Assumptions:
    ///
    /// - `gap_start` is before `self`.
    fn start_gap(&self, gap_start: u64) -> Self {
        Self {
            range: gap_start..=self.range.start().saturating_sub(1),
            target: self.target.clone(),
        }
    }

    /// Create a new entry for the gap between `self` and `gap_end`. If `self` can be extended we
    /// clone its target to the new entry.
    ///
    /// Assumptions:
    ///
    /// - `self` is before `gap_end`.
    fn end_gap(&self, gap_end: u64) -> Self {
        Self {
            range: self.range.end().saturating_add(1)..=gap_end,
            target: self.target.clone(),
        }
    }
}

/// Utility function to make a range bigger by one on both ends.
fn dilate_by_one(range: &RangeInclusive<u64>) -> RangeInclusive<u64> {
    range.start().saturating_sub(1)..=range.end().saturating_add(1)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::ops::RangeInclusive;
    use std::sync::Arc;

    use proptest::arbitrary::any;
    use proptest::collection::vec;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use proptest::strategy::Just;
    use proptest::strategy::Strategy;

    use super::Router;
    use super::RouterEq;
    use super::dilate_by_one;

    // We test the router with `Arc<RefCell<String>>` as a 'toy model' for the compiler.
    type TestTarget = Arc<RefCell<String>>;

    impl RouterEq for TestTarget {
        fn router_eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(self, other)
        }
    }

    type TestRouter = Router<TestTarget>;

    impl TestRouter {
        fn as_vec(&self) -> Vec<(RangeInclusive<u64>, String)> {
            self.internal
                .iter()
                .map(|(r, t)| (r.clone(), t.0.borrow().clone()))
                .collect::<Vec<_>>()
        }

        // Appends `s` to the `String` targeted by the index `idx`.
        //
        // Panics if the index isn't in an existing range.
        fn append(&self, idx: &u64, s: &str) {
            self.get(idx)
                .unwrap_or_else(|| panic!("Index {idx} should be in a range here"))
                .borrow_mut()
                .push_str(s);
        }
    }

    #[derive(Clone, Default)]
    struct AlwaysMergeTarget;

    impl RouterEq for AlwaysMergeTarget {
        fn router_eq(&self, _other: &Self) -> bool {
            true
        }
    }

    #[derive(Clone, Default)]
    struct NeverMergeTarget;

    impl RouterEq for NeverMergeTarget {
        fn router_eq(&self, _other: &Self) -> bool {
            false
        }
    }

    impl<T: Clone + RouterEq> Router<T> {
        fn ranges(&self) -> Vec<RangeInclusive<u64>> {
            self.internal
                .iter()
                .map(|(r, _)| r.clone())
                .collect::<Vec<_>>()
        }

        fn add_range_default(&mut self, range: RangeInclusive<u64>)
        where
            T: Default,
        {
            self.add_range(range, T::default);
        }
    }

    #[test]
    fn test_dilate_by_one() {
        assert_eq!(0..=5, dilate_by_one(&(0..=4)));
        assert_eq!(0..=5, dilate_by_one(&(1..=4)));
        assert_eq!(0..=u64::MAX, dilate_by_one(&(1..=(u64::MAX - 1))));
        assert_eq!(0..=u64::MAX, dilate_by_one(&(1..=u64::MAX)));
    }

    #[test]
    fn test_always_merge() {
        let mut r = Router::<AlwaysMergeTarget>::default();
        r.add_range_default(0..=1);
        r.add_range_default(3..=3);
        r.add_range_default(7..=9);
        r.add_range_default(11..=14);
        assert_eq!(r.ranges(), vec![0..=1, 3..=3, 7..=9, 11..=14]);

        // This final add will cause all the ranges to be merged.
        r.add_range_default(2..=10);
        assert_eq!(r.ranges(), vec![0..=14]);
    }

    #[test]
    fn test_never_merge() {
        let mut r = Router::<NeverMergeTarget>::default();
        r.add_range_default(0..=1);
        r.add_range_default(3..=3);
        r.add_range_default(7..=9);
        r.add_range_default(11..=14);
        assert_eq!(r.ranges(), vec![0..=1, 3..=3, 7..=9, 11..=14]);

        // This final add will not cause any ranges to be merged.
        r.add_range_default(2..=10);
        assert_eq!(
            r.ranges(),
            vec![0..=1, 2..=2, 3..=3, 4..=6, 7..=9, 10..=10, 11..=14]
        );
    }

    proptest! {
        // In these three tests:
        //
        // - `a` is the length of the shorter range minus 1
        // - `b` is the length of the gap plus 1
        // - `c` is the difference in length between the two ranges
        #[test]
        fn test_merge_left_is_longer(a in 0..1000u64, b in 2..1000u64, c in 1..1000u64) {
            let mut r = TestRouter::default();
            r.add_range_default(0..=(a + c));
            r.add_range_default((a + b + c)..=(2 * a + b + c));
            r.add_range_default((a + c + 1)..=(a + b + c - 1));
            prop_assert_eq!(r.ranges(), vec![0..=(a + b + c - 1), (a + b + c)..=(2 * a + b + c)]);
        }

        #[test]
        fn test_merge_right_is_longer(a in 0..1000u64, b in 2..1000u64, c in 1..1000u64) {
            let mut r = TestRouter::default();
            r.add_range_default(0..=a);
            r.add_range_default((a + b)..=(2 * a + b + c));
            r.add_range_default((a + 1)..=(a + b - 1));
            prop_assert_eq!(r.ranges(), vec![0..=a, (a + 1)..=(2 * a + b + c)]);
        }

        #[test]
        fn test_merge_equal_length(a in 0..1000u64, b in 2..1000u64) {
            let mut r = TestRouter::default();
            r.add_range_default(0..=a);
            r.add_range_default((a + b)..=(2 * a + b));
            r.add_range_default((a + 1)..=(a + b - 1));
            prop_assert_eq!(r.ranges(), vec![0..=(a + b - 1), (a + b)..=(2 * a + b)]);
        }
    }

    #[test]
    fn test_add_range_merge_multiple() {
        let mut r = TestRouter::default();

        r.add_range_default(1..=3);
        r.add_range_default(5..=6);
        r.add_range_default(8..=8);
        r.add_range_default(10..=10);

        r.add_range_default(0..=11);

        assert_eq!(r.ranges(), vec![0..=4, 5..=7, 8..=9, 10..=11]);
    }

    #[test]
    fn test_get() {
        let mut r = TestRouter::default();

        r.add_range_default(2..=5);
        r.append(&3, "hello");

        r.add_range_default(7..=7);
        r.append(&7, "world");

        r.add_range_default(0..=9);
        r.append(&5, "a");
        r.append(&8, "b");

        assert_eq!(
            r.get(&0).map(|r| r.borrow().clone()),
            Some("helloa".to_string())
        );
        assert_eq!(
            r.get(&9).map(|r| r.borrow().clone()),
            Some("worldb".to_string())
        );
        assert_eq!(r.get(&10).map(|r| r.borrow().clone()), None);

        assert_eq!(
            r.as_vec(),
            vec![(0..=6, "helloa".to_string()), (7..=9, "worldb".to_string()),]
        );
    }

    proptest! {
        #[test]
        fn test_drain_overlapping(
            ranges in vec(any::<RangeInclusive<u64>>(), 0..100),
            (to_drain, idx) in any::<RangeInclusive<u64>>()
                .prop_flat_map(|r| (Just(r.clone()), r)),
        ) {
            let mut r = TestRouter::default();

            for range in ranges {
                // these ranges may collide, that doesn't matter
                r.add_range_default(range);
            }

            let before = r.ranges();

            let drained = r.drain_overlapping(to_drain.clone()).collect::<Vec<_>>();

            let after = r.ranges();

            let again = r.drain_overlapping(to_drain.clone()).collect::<Vec<_>>();

            // `drain_overlapping` is idempotent; it returns nothing the second time
            prop_assert_eq!(after.clone(), r.ranges());
            prop_assert_eq!(again, vec![]);

            prop_assert_eq!(None, r.get(to_drain.start()));
            prop_assert_eq!(None, r.get(to_drain.end()));
            prop_assert_eq!(None, r.get(&idx));

            let after = after.iter().collect::<HashSet<_>>();
            let drained = drained.iter().collect::<HashSet<_>>();
            let before = before.iter().collect::<HashSet<_>>();

            // `after` and `drained` are disjoint
            prop_assert!(after.is_disjoint(&drained));

            // as a set, `before` is the union of `after` and `drained`
            let union = after.union(&drained).cloned();
            prop_assert_eq!(before, union.collect::<HashSet<_>>());
        }
    }
}
