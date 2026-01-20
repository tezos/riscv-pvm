// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a byte array
//!
//! See [`Bytes`] for more details.

use std::ops::Range;

use bincode::Encode;
use perfect_derive::perfect_derive;

use crate::clone::CloneState;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;

/// Byte array state component
#[perfect_derive(Debug)]
pub struct Bytes<M: Mode> {
    bytes: M::Select<BytesTemplate>,
}

impl<M: BytesMode> Bytes<M> {
    /// Create an empty byte array.
    pub fn new() -> Self {
        M::new()
    }

    /// Read from the byte array.
    ///
    /// `start` is the starting index to read from. All data will be read into `buffer`. The
    /// maximum number of bytes read is the length of `buffer`.
    ///
    /// Returns the number of bytes actually read. This could be less than the length of `buffer`
    /// if the read goes or starts out of bounds.
    pub fn read(&self, start: usize, buffer: &mut [u8]) -> usize {
        M::read(self, start, buffer)
    }

    /// Write to the byte array.
    ///
    /// `start` is the starting index to write to. Data will be written from `buffer`. The maximum
    /// number of bytes written is the length of `buffer`.
    ///
    /// Returns the number of bytes actually written. This could be less than the length of `buffer`
    /// if the write goes or starts out of bounds.
    pub fn write(&mut self, start: usize, buffer: &[u8]) -> usize {
        M::write(self, start, buffer)
    }

    /// Append data to the end of the byte array.
    ///
    /// Returns the number of bytes actually appended. This could be less than the length of
    /// `buffer` if the byte array length would overflow [`usize`].
    pub fn append(&mut self, buffer: &[u8]) -> usize {
        let current_len = self.len();
        let new_len = buffer.len().saturating_add(current_len);
        self.resize(new_len);

        let bytes_to_write = buffer.len().min(new_len - current_len);
        self.write(current_len, &buffer[..bytes_to_write])
    }

    /// Overwrite the entire contents of the byte array with the given data.
    pub fn set(&mut self, buffer: &[u8]) {
        self.resize(buffer.len());
        self.write(0, buffer);
    }

    /// Is the length of the byte array zero?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of bytes the byte array holds.
    pub fn len(&self) -> usize {
        M::len(self)
    }

    /// Change the length of the byte array.
    ///
    /// If the new length is greater than the current length, the new bytes will be zeroed. If the
    /// new length is less than the current length, the byte array will be truncated.
    pub fn resize(&mut self, new_len: usize) {
        M::resize(self, new_len);
    }
}

impl<M: BytesMode> Default for Bytes<M> {
    fn default() -> Self {
        M::new()
    }
}

impl<M: CloneBytesMode> CloneState for Bytes<M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl<M: CloneBytesMode> Clone for Bytes<M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

impl<M: EncodeBytesMode> Encode for Bytes<M> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        M::encode(self, encoder)
    }
}

impl<T: AsRef<[u8]>, M: BytesMode> PartialEq<T> for Bytes<M> {
    fn eq(&self, other: &T) -> bool {
        let other = other.as_ref();
        let len = self.len();

        if len != other.len() {
            return false;
        }

        let mut chunk_lhs = vec![0u8; 4096];
        for start in (0..len).step_by(chunk_lhs.len()) {
            let read = self.read(start, &mut chunk_lhs);

            if chunk_lhs[..read] != other[start..][..read] {
                return false;
            }
        }

        true
    }
}

impl<M: BytesMode, N: BytesMode> PartialEq<Bytes<N>> for Bytes<M> {
    fn eq(&self, other: &Bytes<N>) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        let mut chunk_lhs = vec![0u8; 4096];
        let mut chunk_rhs = chunk_lhs.clone();

        for offset in (0..len).step_by(chunk_lhs.len()) {
            let read = self.read(offset, &mut chunk_lhs);

            if read != other.read(offset, &mut chunk_rhs) {
                return false;
            }

            if chunk_lhs[..read] != chunk_rhs[..read] {
                return false;
            }
        }

        true
    }
}

impl<M: BytesMode> Eq for Bytes<M> {}

impl From<bytes::Bytes> for Bytes<Normal> {
    fn from(bytes: bytes::Bytes) -> Self {
        let bytes = bytes::BytesMut::from(bytes);
        Bytes { bytes }
    }
}

impl From<&[u8]> for Bytes<Normal> {
    fn from(slice: &[u8]) -> Self {
        let bytes = bytes::BytesMut::from(slice);
        Bytes { bytes }
    }
}

/// Modal template for the [`Bytes`] component
///
/// This is used to select the appropriate implementation for the mode.
enum BytesTemplate {}

impl Modal for BytesTemplate {
    type Normal = bytes::BytesMut;

    type Prove<'normal> = ProveImpl;

    type Verify = VerifyImpl;
}

/// Mode types that implement this trait support common operations on the [`Bytes`] component.
///
/// See [`Bytes`] for a more convenient interface.
pub trait BytesMode: Mode {
    /// See [`Bytes::new`].
    fn new() -> Bytes<Self>;

    /// See [`Bytes::read`].
    fn read(this: &Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize;

    /// See [`Bytes::write`].
    fn write(this: &mut Bytes<Self>, start: usize, buffer: &[u8]) -> usize;

    /// See [`Bytes::len`].
    fn len(this: &Bytes<Self>) -> usize;

    /// See [`Bytes::resize`].
    fn resize(this: &mut Bytes<Self>, new_len: usize);
}

impl BytesMode for Normal {
    fn new() -> Bytes<Self> {
        Bytes {
            bytes: bytes::BytesMut::new(),
        }
    }

    fn read(this: &Bytes<Self>, start: usize, buffer: &mut [u8]) -> usize {
        // If the read starts out of bounds, there is nothing to read.
        if start >= this.bytes.len() {
            return 0;
        }

        let range = clamp_range(this.bytes.len(), start, buffer.len());
        let len = range.len();

        buffer[..len].copy_from_slice(&this.bytes[range]);

        len
    }

    fn write(this: &mut Bytes<Self>, start: usize, buffer: &[u8]) -> usize {
        // We can't write if we start out of bounds.
        if start >= this.bytes.len() {
            return 0;
        }

        let range = clamp_range(this.bytes.len(), start, buffer.len());
        let len = range.len();

        this.bytes[range].copy_from_slice(&buffer[..len]);

        len
    }

    fn len(this: &Bytes<Self>) -> usize {
        this.bytes.len()
    }

    fn resize(this: &mut Bytes<Self>, new_len: usize) {
        this.bytes.resize(new_len, 0x0);
    }
}

/// Mode types that implement this trait support cloning of the [`Bytes`] component
pub trait CloneBytesMode: Mode {
    /// Clones the given [`Bytes`] component.
    ///
    /// This clones the entire component, not just the internal value. Consider this when cloning
    /// components in [`crate::mode::Prove`] mode.
    fn clone(this: &Bytes<Self>) -> Bytes<Self>;
}

impl CloneBytesMode for Normal {
    fn clone(this: &Bytes<Self>) -> Bytes<Self> {
        Bytes {
            bytes: this.bytes.clone(),
        }
    }
}

/// Mode types that implement this trait support encoding of the [`Bytes`] component
pub trait EncodeBytesMode: Mode {
    /// Encodes the [`Bytes`] component as a byte vector.
    fn encode<E: bincode::enc::Encoder>(
        bytes: &Bytes<Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError>;
}

impl EncodeBytesMode for Normal {
    fn encode<E: bincode::enc::Encoder>(
        bytes: &Bytes<Self>,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bytes.bytes.encode(encoder)
    }
}

/// [`crate::mode::Prove`] mode implementation for the [`Bytes`] component
struct ProveImpl {}

/// [`crate::mode::Verify`] mode implementation for the [`Bytes`] component
struct VerifyImpl {}

/// Construct a range `start..start+len` then clamp it so it doesn't extend beyond `total_len`.
fn clamp_range(total_len: usize, start: usize, len: usize) -> Range<usize> {
    let end = start.saturating_add(len).min(total_len);
    start..end
}
