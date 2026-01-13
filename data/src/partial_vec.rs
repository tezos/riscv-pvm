// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Like [`Vec`] but not continuously defined
//!
//! # Structure
//!
//! A [`PartialVec`] is a binary tree where leaves are either defined data or undefined gaps.
//! This allows efficient representation of sparse data without allocating memory for gaps.
//!
//! ```text
//! Logical view:    [a][b][c][ ? ][ ? ][ ? ][ ? ][d][e][f]
//!                   0  1  2   3    4    5    6   7  8  9
//!                  |_______|___________________|_________|
//!                   defined      undefined       defined
//!
//! Tree structure:
//!
//!                        Concatenated
//!                         (len: 10)
//!                        /          \
//!                Concatenated       Defined
//!                 (len: 7)          [d,e,f]
//!                /        \
//!           Defined     Undefined
//!           [a,b,c]      (len: 4)
//! ```
//!
//! Unlike a `BTreeMap<usize, T>`, this structure is optimized for range queries making it
//! memory-efficient for sparse but clustered data.

use std::ops::Range;

/// Partial vector that may not be defined for certain ranges
///
/// This data type can be seen as an alternative to `BTreeMap<usize, T>` which provides a more
/// memory-efficient representation for when there are many adjacent keys and the primary query
/// mechanism is using ranges.
#[derive(Debug, Clone)]
pub enum PartialVec<T> {
    /// Undefined vector of a certain width
    Undefined {
        /// How many elements are undefined
        length: usize,
    },

    /// Defined vector of a certain width
    Defined {
        /// Elements that define the range
        data: Vec<T>,
    },

    /// Concatenation of two partial vectors
    Concatenated {
        /// Combined width of both vectors
        ///
        /// This is not the number of defined items.
        node_width: usize,

        /// First partial vector
        left: Box<Self>,

        /// Second partial vector
        right: Box<Self>,
    },
}

impl<T> PartialVec<T> {
    /// Create a vector of defined data.
    pub fn defined(data: Vec<T>) -> Self {
        Self::Defined { data }
    }

    /// Create a vector of undefined data with the provided length.
    pub fn undefined(length: usize) -> Self {
        Self::Undefined { length }
    }

    /// Retrieve the width of the partial vector.
    ///
    /// This is not the number of defined elements, but the total number of elements represented by
    /// the partial vector - defined or undefined.
    /// In other words, `PartialVec::undefined(10).width() == 10`.
    fn width(&self) -> usize {
        match self {
            Self::Undefined { length } => *length,
            Self::Defined { data } => data.len(),
            Self::Concatenated { node_width, .. } => *node_width,
        }
    }

    /// Define a range within the partial vector.
    ///
    /// Existing data is overwritten if it overlaps the newly defined range.
    pub fn define(&mut self, offset: usize, new_data: Vec<T>) {
        let mut work = vec![(self, offset, new_data)];

        while let Some((target, offset, mut new_data)) = work.pop() {
            // If there is nothing left to insert, we might as well stop.
            if new_data.is_empty() {
                break;
            }

            match target {
                Self::Concatenated {
                    left,
                    right,
                    node_width,
                } => {
                    let new_data_end = offset + new_data.len();

                    // Inserting may extend the range represented by the node.
                    *node_width = std::cmp::max(*node_width, new_data_end);

                    // When the offset is larger than what the left node stores, then we can safely
                    // delegate all the work to the right node.
                    // This case also handles when we want to define something beyond the current
                    // node's width. In that scenario, we descend into the right node.
                    if offset >= left.width() {
                        let new_offset = offset - left.width();
                        work.push((right, new_offset, new_data));
                        continue;
                    }

                    // When the inserted range fits entirely within the left node, we can focus on
                    // the left node only.
                    if new_data_end <= left.width() {
                        work.push((left, offset, new_data));
                        continue;
                    }

                    // At this point we know that the inserted range overlaps with both the left and
                    // right nodes. So we divide the insertion vector accordingly.
                    let left_overlap_tail = left.width() - offset;
                    let new_right_data = new_data.split_off(left_overlap_tail);

                    work.push((left, offset, new_data));
                    work.push((right, 0, new_right_data));
                }

                Self::Undefined { length } => {
                    let new_data_end = offset + new_data.len();

                    let undefined_prefix_len = offset;
                    let undefined_suffix_len = length.saturating_sub(new_data_end);

                    // This is the "center" node. We prepend and append undefined nodes as needed.
                    // We don't need to worry about merging adjacent nodes, as this node is either
                    // the root or the right-most node in a concatenation. The `Concatenated` case
                    // deals with this for us.
                    let mut new_data_node = Self::defined(new_data);

                    // If the insertion does not happen at the beginning, then we keep a portion of
                    // the undefined range. The length of the existing undefined range is not
                    // considered, as we are only interested in filling the gap.
                    if undefined_prefix_len > 0 {
                        let prefix = Self::undefined(undefined_prefix_len);
                        new_data_node = Self::Concatenated {
                            node_width: prefix.width() + new_data_node.width(),
                            left: Box::new(prefix),
                            right: Box::new(new_data_node),
                        };
                    }

                    // If the tail of the inserted range does not extend beyond the existing
                    // undefined range, then the portion of the undefined range that covers that bit
                    // needs to be kept.
                    if undefined_suffix_len > 0 {
                        let suffix = Self::undefined(undefined_suffix_len);
                        new_data_node = Self::Concatenated {
                            node_width: new_data_node.width() + suffix.width(),
                            left: Box::new(new_data_node),
                            right: Box::new(suffix),
                        };
                    }

                    *target = new_data_node;
                }

                Self::Defined { data } => {
                    // When inserting beyond what is currently present, there needs to be an
                    // undefined gap between the existing range and the new range. This makes it a
                    // slightly special case which is best handled separately.
                    if offset > data.len() {
                        let gap = offset - data.len();

                        let new_data_node = Self::defined(new_data);
                        let new_node = Self::Concatenated {
                            node_width: gap + new_data_node.width(),
                            left: Box::new(Self::undefined(gap)),
                            right: Box::new(new_data_node),
                        };

                        // Taking the data lets us reuse the allocation. `Default for Vec` (like
                        // `Vec::new`) does not allocate any memory, so this is rather efficient.
                        // The alternative is a bit of unsafe code which doesn't seem worth the
                        // trade-off right now.
                        let data = std::mem::take(data);

                        *target = Self::Concatenated {
                            node_width: data.len() + new_node.width(),
                            left: Box::new(Self::defined(data)),
                            right: Box::new(new_node),
                        };

                        continue;
                    }

                    let new_data_end = offset + new_data.len();

                    // When we start inserting within the defined range, but the new data extends
                    // beyond the existing range, we can truncate and append in place.
                    // We can do the extension because there are no adjacent entries to worry about.
                    // The `Concatenated` case handles that.
                    if new_data_end >= data.len() {
                        // Truncation leaves the capacity unchanged.
                        data.truncate(offset);

                        // Using `append` to avoid the iterator intermediary.
                        data.append(&mut new_data);

                        continue;
                    }

                    // At this point we know that the newly inserted data fits entirely within the
                    // existing defined entry.
                    for (dst, src) in data[offset..new_data_end].iter_mut().zip(new_data) {
                        // We replace the existing data this way to not require `T: Copy`.
                        *dst = src;
                    }
                }
            }
        }
    }

    /// Mark everything beyond `keep_length` as undefined.
    pub fn truncate(&mut self, mut keep_length: usize) {
        // The borrow checker is unfortunately a little in the way in this method. Please see the
        // `PartialVec::Concatenated` case below for more information.
        let mut target: *mut Self = self;

        loop {
            // SAFETY: Dereferencing `target` is safe because we have not aliased the pointer.
            match unsafe { &mut *target } {
                Self::Concatenated {
                    left,
                    right,
                    node_width,
                } => {
                    *node_width = keep_length.min(*node_width);

                    // In case the left node will be fully kept, we can just move on to the right
                    // node.
                    if keep_length > left.width() {
                        target = right.as_mut();
                        keep_length -= left.width();

                        continue;
                    }

                    // We only need to keep the left portion.
                    let new_target = std::mem::take(left.as_mut());

                    // SAFETY: `left` and `right` are no longer used at this point. So technically
                    // `target` is an exclusive reference. The borrow checker is not able to detect
                    // this though, so we must resort to unsafe code.
                    // The old value is dropped as it is returned from the replace call.
                    unsafe { target.replace(new_target) };
                }

                Self::Undefined { length } => {
                    *length = keep_length.min(*length);
                    break;
                }

                Self::Defined { data } => {
                    data.truncate(keep_length);
                    break;
                }
            }
        }
    }

    /// Fetch the entries of the partial vector that make up the provided range.
    pub fn range(&self, mut range: Range<usize>) -> impl Iterator<Item = RangeEntry<T>> {
        let mut work = vec![self];

        std::iter::from_fn(move || {
            while !range.is_empty() {
                let Some(node) = work.pop() else {
                    // If we ran out of things to do, but the range is not empty yet, then the partial
                    // vector is implicitly undefined for the rest of the queried range.
                    let entry = RangeEntry::Undefined {
                        length: range.len(),
                    };

                    // We must drain the range to prevent further iterations.
                    range = 0..0;

                    return Some(entry);
                };

                if node.width() <= range.start {
                    // This branch happens in two cases:
                    // - top-level undefined, defined or concatenation node before the query range
                    // - when we split up a concatenation of two partial vectors, but the left
                    //   vector is before the query range

                    range.start -= node.width();
                    range.end -= node.width();

                    continue;
                }

                match node {
                    Self::Concatenated { left, right, .. } => {
                        // These nodes need to be traversed next. Otherwise we'll violate the
                        // left-to-right traversal order which is assumed in this function.
                        // After: left -> right -> rest...
                        work.push(right.as_ref());
                        work.push(left.as_ref());
                    }

                    Self::Undefined { length } => {
                        // The overlap needs to be capped by the query range as the overlap could
                        // otherwise be larger than the query range.
                        let overlap = range.len().min(length - range.start);

                        range.end = range.len() - overlap;
                        range.start = 0;

                        return Some(RangeEntry::Undefined { length: overlap });
                    }

                    Self::Defined { data } => {
                        let chunk = &data[range.start..range.end.min(data.len())];

                        range.end = range.len() - chunk.len();
                        range.start = 0;

                        return Some(RangeEntry::Defined { data: chunk });
                    }
                }
            }

            None
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
        self.range(0..self.width())
            .all(|chunk| matches!(chunk, RangeEntry::Undefined { .. }))
    }
}

impl<T> Default for PartialVec<T> {
    fn default() -> Self {
        Self::undefined(0)
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
