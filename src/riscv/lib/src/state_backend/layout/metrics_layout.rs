// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

pub trait MetricsLayout {
    fn init(&mut self, path_from_root: Vec<u8>);
}
