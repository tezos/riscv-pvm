// SPDX-FileCopyrightText: 2025-2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

/// Create a boxed array from a function.
pub fn boxed_from_fn<T, const LEN: usize>(mut f: impl FnMut() -> T) -> Box<[T; LEN]> {
    let mut entries = Vec::with_capacity(LEN);
    entries.resize_with(LEN, &mut f);
    entries
        .into_boxed_slice()
        .try_into()
        .map_err(|_| unreachable!("Converting vec into boxed slice of same length always succeeds"))
        .unwrap()
}
