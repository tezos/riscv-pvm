// SPDX-FileCopyrightText: 2023-2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::num::NonZeroUsize;

use octez_riscv_data::serialisation::elem::Elem;

use crate::machine_state::memory::PAGE_SIZE;

/// Types that are less than one page wide
pub trait NarrowlySized: Elem {
    /// Size of the type
    const NARROW_SIZE: NonZeroUsize = {
        if Self::STORED_SIZE.get() >= PAGE_SIZE.get() as usize {
            panic!("Type is too wide");
        }

        Self::STORED_SIZE
    };
}

impl<T: Elem> NarrowlySized for T {}

/// Types that can be copied and contain no non-static references
pub trait StaticCopy: Copy + 'static {}

impl<T: Copy + 'static> StaticCopy for T {}
