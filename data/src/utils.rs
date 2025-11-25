// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Module for utilities that are useful across this crate

/// Compute the next power greater than or equal to `value`. Similar to [`usize::next_power_of_two`].
///
/// # Panics
///
/// Panics if `power` is less than 2.
pub(crate) const fn next_power_of(value: usize, power: usize) -> usize {
    // 0 -> 0 * power -> 0
    // 1 -> 1 * power -> power
    if value < 2 {
        return value * power;
    }

    let exp = 1 + (value - 1).ilog(power);
    power.pow(exp)
}
