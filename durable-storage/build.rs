// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Emits cfg aliases for compound test/feature flags.
//!
//! - `test_utils`: alias for `feature = "unstable-test-utils"`. Thanks to the
//!   self dev-dependency in `Cargo.toml`, the feature is always enabled in
//!   test builds, so it is also equivalent to `any(test, feature = "unstable-test-utils")`.
//!
//! - `rocksdb`: alias for `feature = "rocksdb"`.
//!
//! - `rocksdb_test_utils`: alias for
//!   `all(feature = "unstable-test-utils", feature = "rocksdb")`.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(test_utils)");
    println!("cargo::rustc-check-cfg=cfg(rocksdb)");
    println!("cargo::rustc-check-cfg=cfg(rocksdb_test_utils)");

    let test_utils = std::env::var_os("CARGO_FEATURE_UNSTABLE_TEST_UTILS").is_some();
    let rocksdb = std::env::var_os("CARGO_FEATURE_ROCKSDB").is_some();

    if test_utils {
        println!("cargo::rustc-cfg=test_utils");
    }

    if rocksdb {
        println!("cargo::rustc-cfg=rocksdb");
    }

    if test_utils && rocksdb {
        println!("cargo::rustc-cfg=rocksdb_test_utils");
    }
}
