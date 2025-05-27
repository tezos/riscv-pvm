// SPDX-FileCopyrightText: 2023-2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use crate::default::ConstDefault;
use crate::machine_state::csregisters::Privilege;

/// Modes the hardware state can be in when running code
#[derive(
    Debug,
    PartialEq,
    PartialOrd,
    Eq,
    Copy,
    Clone,
    strum::EnumIter,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Mode {
    User,
    Supervisor,
    Machine,
}

impl Mode {
    /// Obtain the corresponding [`Privilege`] for [`Mode`].
    pub fn privilege(&self) -> Privilege {
        match self {
            Mode::User => Privilege::Unprivileged,
            Mode::Supervisor => Privilege::Supervisor,
            Mode::Machine => Privilege::Machine,
        }
    }
}

impl ConstDefault for Mode {
    const DEFAULT: Self = Self::Machine;
}

impl From<Mode> for u8 {
    #[inline]
    fn from(value: Mode) -> Self {
        value as u8
    }
}
