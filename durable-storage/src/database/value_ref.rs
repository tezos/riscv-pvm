// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use octez_riscv_data::mode::Normal;

/// Borrowed view of a value that supports buffered reads.
pub trait ValueRef {
    /// Returns the length in bytes of the underlying value.
    fn len(&self) -> usize;

    /// Returns `true` if the underlying value has length zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read up to `buf.len()` bytes starting at `offset` into `buf`. Returns the number of bytes
    /// actually read, which may be less than `buf.len()` if the read runs past the end.
    fn read(&self, offset: usize, buf: &mut [u8]) -> usize;
}

/// Adapter that promotes an [`AsRef<[u8]>`] value into a [`ValueRef`].
pub(crate) struct AsRefValueRef<T>(pub(crate) T);

impl<T: AsRef<[u8]>> ValueRef for AsRefValueRef<T> {
    fn len(&self) -> usize {
        self.0.as_ref().len()
    }

    fn read(&self, offset: usize, buf: &mut [u8]) -> usize {
        let src = self.0.as_ref();
        if offset >= src.len() {
            return 0;
        }
        let len = buf.len().min(src.len() - offset);
        buf[..len].copy_from_slice(&src[offset..offset + len]);
        len
    }
}

impl ValueRef for octez_riscv_data::components::bytes::Bytes<Normal> {
    fn len(&self) -> usize {
        self.len()
    }

    fn read(&self, offset: usize, buf: &mut [u8]) -> usize {
        self.read(offset, buf)
    }
}
