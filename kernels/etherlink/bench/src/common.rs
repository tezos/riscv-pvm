// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::time::Duration;

use serde::Deserialize;

pub(crate) const EXPECTED_LEVELS: usize = 1;

#[derive(Deserialize, Debug, PartialEq)]
pub(crate) struct LogLine {
    pub(crate) elapsed: Duration,
    pub(crate) message: String,
}
