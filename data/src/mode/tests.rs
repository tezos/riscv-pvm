// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use crate::components::atom::AtomMode;

trait_set::trait_set! {
    /// Mode for all tests
    ///
    /// Each state component comes with a small set of mode-constraining traits. When a component
    /// is used in tests, it is best to mention those traits in this trait alias, so that they are
    /// available to all tests.
    pub(crate) trait TestMode = AtomMode;
}

/// Generate a test against all test backends.
macro_rules! backend_test {
    ($(#[$attr:meta])* $fun_name:ident, $ty_name:ident, $expr:block) => {
        $(#[$attr])*
        #[test]
        fn $fun_name() {
            fn inner<$ty_name: $crate::mode::tests::TestMode>() {
                $expr
            }

            inner::<$crate::mode::Normal>();
            inner::<$crate::mode::Prove>();
            inner::<$crate::mode::Verify>();
        }
    };
}

pub(crate) use backend_test;
