// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Like [`Vec`] but not continuously defined
//!
//! # Structure
//!
//! A [`PartialVec`] is a flat list of defined entries, where each entry has a start index and
//! contiguous data. Gaps between entries represent undefined regions. This allows efficient
//! representation of sparse data without allocating memory for gaps.
//!
//! ```text
//! Logical view:    [a][b][c][ ? ][ ? ][ ? ][ ? ][d][e][f]
//!                   0  1  2   3    4    5    6   7  8  9
//!                  |_______|___________________|_________|
//!                   defined      undefined       defined
//!
//! Internal structure (sorted by start index):
//!
//!   entries: [
//!     DefinedEntry { start: 0, data: [a, b, c] },
//!     DefinedEntry { start: 7, data: [d, e, f] },
//!   ]
//! ```
//!
//! Unlike a `BTreeMap<usize, T>`, this structure is optimized for range queries making it
//! memory-efficient for sparse but clustered data.

use std::ops::Range;

use perfect_derive::perfect_derive;

/// Entry in the partial vector that represents a defined data range
#[perfect_derive(Debug, Clone, Default)]
struct DefinedEntry<T> {
    /// Absolute position where this entry starts
    start: usize,

    /// Data for this entry
    data: Vec<T>,
}

impl<T> DefinedEntry<T> {
    /// Get the exclusive end index of this defined entry.
    const fn end(&self) -> usize {
        self.start + self.data.len()
    }
}

/// Partial vector that may not be defined for certain ranges
///
/// This data type can be seen as an alternative to `BTreeMap<usize, T>` which provides a more
/// memory-efficient representation for when there are many adjacent keys and the primary query
/// mechanism is using ranges.
#[derive(Debug, Clone)]
pub struct PartialVec<T> {
    entries: Vec<DefinedEntry<T>>,
}

impl<T> PartialVec<T> {
    /// Create an empty partial vector with no defined entries.
    pub fn empty() -> Self {
        Self { entries: vec![] }
    }

    /// Insert a new defined entry at the specified index, if possible.
    ///
    /// This function requires that the new entry does not overlap with any entries before `at`.
    ///
    /// If no insertion is possible, this function will have no effect.
    fn insert_entry(&mut self, at: usize, new_entry: &mut DefinedEntry<T>) {
        if at >= self.entries.len() {
            let entry = std::mem::take(new_entry);
            self.entries.push(entry);
            return;
        }

        let existing_entry = &mut self.entries[at];

        // New:      ####
        // Existing:     ####
        // Insert new entry before existing entry.
        if new_entry.end() <= existing_entry.start {
            let entry = std::mem::take(new_entry);
            self.entries.insert(at, entry);
            return;
        }

        // New:      ####
        // Existing:   ####
        // Insert prefix of new before existing entry.
        if new_entry.start < existing_entry.start {
            let keep = new_entry
                .data
                .len()
                .min(existing_entry.start - new_entry.start);

            let prefix_entry = DefinedEntry {
                start: new_entry.start,
                data: new_entry.data.drain(..keep).collect(),
            };

            new_entry.start += keep;

            self.entries.insert(at, prefix_entry);
        }

        // We compute this here to avoid borrowing issues in the below blocks.
        let max_extent = self.max_extend(at);

        // We're borrowing it again, to avoid lifetime issues in the above blocks.
        let existing_entry = &mut self.entries[at];

        // New:          ####
        // Existing:   ####
        // Update overlap between new and existing entry.
        // Any overhang of the new entry will either be handled below, or by the caller in a new
        // iteration.
        if new_entry.start < existing_entry.end() {
            let overlap = new_entry
                .data
                .len()
                .min(existing_entry.end() - new_entry.start);
            let gap = existing_entry
                .data
                .len()
                .min(new_entry.start - existing_entry.start);

            existing_entry
                .data
                .splice(gap..gap + overlap, new_entry.data.drain(..overlap));

            new_entry.start += overlap;
        }

        // New:          ####
        // Existing: ####
        if new_entry.start == existing_entry.end() {
            let extension = new_entry.end().min(max_extent) - new_entry.start;
            existing_entry
                .data
                .extend(new_entry.data.drain(..extension));
            new_entry.start += extension;
        }

        // Other cases don't apply to the entry at index `at`. The caller should handle them in the
        // next iteration.
    }

    /// Get a hypothetical range end for the entry at index `idx`. The entry could grow up to this,
    /// without overlapping the next entry.
    fn max_extend(&self, idx: usize) -> usize {
        // If `idx` is the last entry, the maximum extent is unbounded.
        if idx + 1 >= self.entries.len() {
            return usize::MAX;
        }

        self.entries[idx + 1].start
    }

    /// Locate the entry index that contains the provided vector index.
    fn find_entry_idx(&self, idx: usize) -> Option<usize> {
        self.entries
            .binary_search_by(|entry| {
                if entry.end() <= idx {
                    std::cmp::Ordering::Less
                } else if entry.start > idx {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()
    }

    /// Retrieve a reference to the element at the given index, if defined.
    pub fn get(&self, idx: usize) -> Option<&T> {
        let entry_idx = self.find_entry_idx(idx)?;
        let entry = &self.entries[entry_idx];
        // Entry was found - therefore `entry.start <= idx`
        let local_idx = idx - entry.start;
        entry.data.get(local_idx)
    }

    /// Retrieve a mutable reference to the element at the given index, if defined.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        let entry_idx = self.find_entry_idx(idx)?;
        let entry = &mut self.entries[entry_idx];
        // Entry was found - therefore `entry.start <= idx`
        let local_idx = idx - entry.start;
        entry.data.get_mut(local_idx)
    }

    /// Define a range within the partial vector.
    ///
    /// Existing data is overwritten if it overlaps the newly defined range.
    ///
    /// TODO: RV-898: Improve this method to consume the `new_data` vector from the right. This can
    /// be more performant as it avoid potentially-excessive internal copying.
    pub fn define(&mut self, offset: usize, new_data: Vec<T>) {
        if new_data.is_empty() {
            return;
        }

        let mut new_entry = DefinedEntry {
            start: offset,
            data: new_data,
        };

        // If you partition the list of entries by whether they end before the new entry starts,
        // the first entry that does not satisfy this is the first one that might overlap with
        // the new entry.
        //
        // Example:
        //
        // Entries by predicate: yyyyyyynnnnnn
        //                              ↑
        //                              Partition point
        let mut start_idx = self
            .entries
            .partition_point(|entry| entry.end() <= new_entry.start);

        while !new_entry.data.is_empty() {
            // Insert or merge the new entry at the current index.
            // This will update `new_entry` accordingly for the next iteration.
            self.insert_entry(start_idx, &mut new_entry);
            start_idx += 1;
        }
    }

    /// Mark everything beyond `keep_length` as undefined.
    pub fn truncate(&mut self, keep_length: usize) {
        // Remove entries that start at or after the "keep length" marker.
        self.entries.retain(|entry| entry.start < keep_length);

        // Any entry that extends beyond the "keep length" marker needs to be truncated.
        if let Some(last) = self.entries.last_mut()
            && last.end() > keep_length
        {
            last.data.truncate(keep_length - last.start);
        }
    }

    /// Fetch the entries of the partial vector that make up the provided range.
    pub fn range(&self, mut range: Range<usize>) -> impl Iterator<Item = RangeEntry<'_, T>> {
        // Find the first entry that might overlap with our range.
        let mut idx = self
            .entries
            .partition_point(|entry| entry.end() <= range.start);

        std::iter::from_fn(move || {
            if range.is_empty() {
                return None;
            }

            if idx >= self.entries.len() {
                // No more entries, but we still have range to cover.
                // The vector is implicitly undefined beyond the last defined entry.
                let length = range.len();

                range.start += length;

                return Some(RangeEntry::Undefined { length });
            }

            let entry = &self.entries[idx];

            if range.start < entry.start {
                // The query range starts before this entry. This means there is a gap of undefined
                // data.
                // Since we started with the first entry that might overlap, this gap is guaranteed
                // to be undefined.
                let gap = entry.start - range.start;

                // There is an edge case where the gap is larger than the remaining range.
                let length = range.len().min(gap);

                range.start += length;

                return Some(RangeEntry::Undefined { length });
            }

            // The query range starts at or after this entry's start.
            let offset = range.start - entry.start;

            // It's possible that the range goes beyond this entry. So we need to clip the length
            // of the data entry we're about to yield to not extend beyond the entry.
            let length = range.len().min(entry.data.len() - offset);

            range.start += length;
            idx += 1;

            Some(RangeEntry::Defined {
                data: &entry.data[offset..][..length],
            })
        })
    }

    /// Retrieve chunks that make up a continuous range.
    ///
    /// If there are gaps in the range, `None` is returned.
    pub fn continuous_defined_range(&self, range: Range<usize>) -> Option<Vec<&[T]>> {
        self.range(range)
            .map(|chunk| match chunk {
                RangeEntry::Undefined { .. } => None,
                RangeEntry::Defined { data, .. } => Some(data),
            })
            .collect::<Option<Vec<_>>>()
    }

    /// Retrieve defined entries in the given range.
    ///
    /// The returned iterator yields tuples of `(offset, defined_data)` where `offset` is the offset
    /// within the queried range.
    pub fn defined_range(&self, range: Range<usize>) -> impl Iterator<Item = (usize, &[T])> {
        let mut start = 0;
        self.range(range).filter_map(move |chunk| match chunk {
            RangeEntry::Undefined { length } => {
                start += length;
                None
            }

            RangeEntry::Defined { data } => {
                let item = (start, data);
                start += data.len();
                Some(item)
            }
        })
    }

    /// Check if the range is continuously defined.
    ///
    /// In other words, `self.continuous_range(range)` would succeed (return `Some`).
    pub fn is_all_defined(&self, range: Range<usize>) -> bool {
        self.range(range)
            .all(|chunk| matches!(chunk, RangeEntry::Defined { .. }))
    }

    /// Check if the range is at least partially defined.
    ///
    /// In other words, `self.range(range)` contains at least one defined entry.
    pub fn is_any_defined(&self, range: Range<usize>) -> bool {
        self.range(range)
            .any(|chunk| matches!(chunk, RangeEntry::Defined { .. }))
    }

    /// Is nothing in the partial vector defined?
    pub fn is_all_undefined(&self) -> bool {
        self.entries.iter().all(|entry| entry.data.is_empty())
    }
}

impl<T> Default for PartialVec<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> From<Vec<T>> for PartialVec<T> {
    fn from(data: Vec<T>) -> Self {
        if data.is_empty() {
            return Self::empty();
        }

        let entry = DefinedEntry { start: 0, data };

        Self {
            entries: vec![entry],
        }
    }
}

/// Entry within a partial vector
#[derive(Debug, Clone)]
pub enum RangeEntry<'a, T> {
    /// Undefined entry of a certain length
    Undefined { length: usize },

    /// Defined entry with data
    Defined { data: &'a [T] },
}

impl<'a, T> RangeEntry<'a, T> {
    /// Check if the entry is empty.
    pub fn is_empty(&self) -> bool {
        self.width() == 0
    }

    /// Get the number of elements that this entry represents, regardless of whether they are
    /// defined or not.
    pub fn width(&self) -> usize {
        match self {
            RangeEntry::Undefined { length, .. } => *length,
            RangeEntry::Defined { data, .. } => data.len(),
        }
    }
}

#[cfg(test)]
mod tests;
