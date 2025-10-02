// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroU64;

/// Non-zero length equivalent in size to a `u64`.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct NonZeroLength {
    inner: NonZeroU64,
}

impl NonZeroLength {
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            None => None,
            Some(inner) => Some(Self { inner }),
        }
    }

    pub const fn get(self) -> u64 {
        self.inner.get()
    }

    pub const fn wrap(v: NonZeroU64) -> Self {
        Self { inner: v }
    }

    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        match self.inner.checked_mul(rhs.inner) {
            None => None,
            Some(inner) => Some(Self { inner }),
        }
    }

    pub const fn as_usize(self) -> usize {
        self.inner.get() as usize
    }
}
