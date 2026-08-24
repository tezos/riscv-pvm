// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`NodeKey`]

use octez_riscv_data::mode_test;
use proptest::prelude::*;

use super::NodeKey;
use super::NodeKeyMode;
use crate::key::Key;

mode_test!(new_equal, F: NodeKeyMode, {
    proptest!(|(key: Key)| {
        let node_key = NodeKey::<F>::new(key.clone());

        prop_assert!(node_key.eq(&key), "A NodeKey is equal to the key the node was created from");
    });
});

mode_test!(new_not_equal, F: NodeKeyMode, {
    proptest!(|(
        (lhs, rhs) in (any::<Key>(), any::<Key>())
        .prop_filter("Testing with non-equal keys", |(lhs, rhs)| lhs != rhs)
    )| {
        let lhs = NodeKey::<F>::new(lhs);

        prop_assert!(lhs.ne(&rhs), "A NodeKey is not equal to a key different to the one the node was created from");
    });
});

mode_test!(new_cmp, F: NodeKeyMode, {
    proptest!(|(
        (lhs, rhs) in (any::<Key>(), any::<Key>())
    )| {
        let key_cmp = lhs.cmp(&rhs);

        let lhs = NodeKey::<F>::new(lhs);

        prop_assert_eq!(lhs.cmp(&rhs), key_cmp, "NodeKey::cmp behaves identically to Key::cmp");
    });
});
