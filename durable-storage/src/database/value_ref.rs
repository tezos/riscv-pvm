// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::ops::Index;
use std::ops::Range;

use octez_riscv_data::mode::Normal;

/// Borrowed view of a value that can be sub-sliced by range.
pub trait ValueRef: Index<Range<usize>, Output = [u8]> {
    /// Returns the length in bytes of the underlying value.
    fn len(&self) -> usize;

    /// Returns `true` if the underlying value has length zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Adapter that promotes an [`AsRef<[u8]>`] value into a [`ValueRef`].
pub(crate) struct AsRefValueRef<T>(pub(crate) T);

impl<T: AsRef<[u8]>> Index<Range<usize>> for AsRefValueRef<T> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &[u8] {
        &self.0.as_ref()[range]
    }
}

impl<T: AsRef<[u8]>> ValueRef for AsRefValueRef<T> {
    fn len(&self) -> usize {
        self.0.as_ref().len()
    }
}

impl ValueRef for octez_riscv_data::components::bytes::Bytes<Normal> {
    fn len(&self) -> usize {
        self.len()
    }
}
