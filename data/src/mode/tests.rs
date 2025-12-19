// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use crate::components::atom::AtomMode;

trait_set::trait_set! {
    /// Mode for all tests
    pub(crate) trait TestMode = AtomMode;
}

/// Generate a test against all test backends.
macro_rules! backend_test {
    ( $(#[$m:meta])* $name:ident, $fac_name:ident, $expr:block ) => {
        $(#[$m])*
        #[test]
        fn $name() {
            fn inner<$fac_name: $crate::mode::tests::TestMode>() {
                $expr
            }

            inner::<$crate::mode::Normal>();
            inner::<$crate::mode::Prove>();
            inner::<$crate::mode::Verify>();
        }
    };
}

pub(crate) use backend_test;
