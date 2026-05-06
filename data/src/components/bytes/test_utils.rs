// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Utilities for testing `Bytes` component, which need to be accessible from the benchmarking code
//! as well as within test modules.

#![cfg(any(test, feature = "unstable-test-utils"))]

use proptest::collection::vec;
use proptest::prelude::Just;
use proptest::prelude::Strategy;
use proptest::prelude::any;
use proptest::prop_oneof;

use crate::components::bytes::Bytes;
use crate::components::bytes::BytesMode;

/// Operations to be issued against an immutable Bytes state component
#[derive(Debug, Clone)]
pub enum BytesOp {
    Read { offset: usize, size: usize },
    Len,
}

impl BytesOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub fn any(length: usize) -> impl Strategy<Value = Self> + Clone {
        prop_oneof![
            (0..length, 0usize..50).prop_map(|(offset, size)| Self::Read { offset, size }),
            Just(Self::Len),
        ]
    }

    /// Run an operation against an immutable Bytes state component.
    pub fn run<M: BytesMode>(&self, bytes: &Bytes<M>) -> BytesOpResult {
        match self {
            Self::Read { offset, size } => {
                let mut data = vec![0u8; *size];
                let read = bytes.read(*offset, &mut data);
                BytesOpResult::Read { read, data }
            }

            Self::Len => BytesOpResult::Len { len: bytes.len() },
        }
    }
}

/// Operations to be issued against a mutable Bytes state component
#[derive(Debug, Clone)]
pub enum BytesMutOp {
    Write { offset: usize, data: Vec<u8> },
    Resize { new_size: usize },
    Immutable { op: BytesOp },
}

impl BytesMutOp {
    /// Strategy for generating operations to be issued against the Bytes state component
    pub fn any(length: usize) -> impl Strategy<Value = Self> + Clone {
        prop_oneof![
            (0..length, vec(any::<u8>(), 0..50))
                .prop_map(|(offset, data)| Self::Write { offset, data }),
            (0..length).prop_map(|new_size| Self::Resize { new_size }),
            BytesOp::any(length).prop_map(|op| Self::Immutable { op }),
        ]
    }

    /// Run the operation against the Bytes state component.
    pub fn run<M: BytesMode>(&self, bytes: &mut Bytes<M>) -> BytesOpResult {
        match self {
            Self::Write { offset, data } => {
                let wrote = bytes.write(*offset, data);
                BytesOpResult::Wrote { wrote }
            }

            Self::Resize { new_size } => {
                bytes.resize(*new_size);
                BytesOpResult::Void
            }

            Self::Immutable { op } => op.run(bytes),
        }
    }
}

/// Results of operations issued against the Bytes state component
#[derive(Debug, PartialEq, Eq)]
pub enum BytesOpResult {
    Read { read: usize, data: Vec<u8> },
    Wrote { wrote: usize },
    Len { len: usize },
    Void,
}
