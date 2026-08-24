// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`NodeKey`]

use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::utils::catch_not_found;
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

/// [`NodeKey`] behaves the same across different modes for eq
#[test]
fn node_key_eq_verifies_correctly() {
    proptest!(|(lhs: Key, rhs: Key)| {
        let key_normal = NodeKey::new(lhs);

        let eq_normal = key_normal.eq(&rhs);
        let hash_normal = Hash::from_foldable(&key_normal);

        // Prove mode
        let key_prove = key_normal.start_proof();

        let eq_prove = key_prove.eq(&rhs);
        let hash_prove = Hash::from_foldable(&key_prove);

        prop_assert_eq!(eq_normal, eq_prove);
        prop_assert_eq!(hash_normal, hash_prove);

        let merkle_proof = MerkleProof::from_foldable(&key_prove);

        // Verify mode
        let key_verify = NodeKey::from_proof(ProofTree::present(&merkle_proof))
            .expect("from_proof should succeed")
            .into_result();

        let eq_verify = key_verify.eq(&rhs);
        let hash_verify = PartialHash::from_foldable(Some(merkle_proof), &key_verify)
            .to_hash()
            .unwrap();

        prop_assert_eq!(eq_normal, eq_verify);
        prop_assert_eq!(hash_normal, hash_verify);
    });
}

/// [`NodeKey`] behaves the same across different modes for cmp
#[test]
fn node_key_cmp_verifies_correctly() {
    proptest!(|(lhs: Key, rhs: Key)| {
        let key_normal = NodeKey::new(lhs);

        let cmp_normal = key_normal.cmp(&rhs);
        let hash_normal = Hash::from_foldable(&key_normal);

        // Prove mode
        let key_prove = key_normal.start_proof();

        let cmp_prove = key_prove.cmp(&rhs);
        let hash_prove = Hash::from_foldable(&key_prove);

        prop_assert_eq!(cmp_normal, cmp_prove);
        prop_assert_eq!(hash_normal, hash_prove);

        let merkle_proof = MerkleProof::from_foldable(&key_prove);

        // Verify mode
        let key_verify = NodeKey::from_proof(ProofTree::present(&merkle_proof))
            .expect("from_proof should succeed")
            .into_result();

        let cmp_verify = key_verify.cmp(&rhs);
        let hash_verify = PartialHash::from_foldable(Some(merkle_proof), &key_verify)
            .to_hash()
            .unwrap();

        prop_assert_eq!(cmp_normal, cmp_verify);
        prop_assert_eq!(hash_normal, hash_verify);
    });
}

#[test]
fn verify_absent() {
    let key = Key::new(&[1; 42]).unwrap();

    let node_key = NodeKey::from_proof(ProofTree::absent())
        .expect("Should create absent NodeKey")
        .into_result();

    let value = catch_not_found(|| node_key.eq(&key)).ok();
    assert_eq!(value, None);

    let cmp_value = catch_not_found(|| node_key.cmp(&key)).ok();
    assert_eq!(cmp_value, None);
}

/// Test pinning NodeKey initial behaviour onto the same behaviour
/// as Atom<Key>
#[test]
fn verify_blinded() {
    let key = Key::new(&[1; 42]).unwrap();
    let hash = Hash::hash_encodable(&key).unwrap();

    let proof = MerkleProof::leaf_blind(hash);

    let node_key = NodeKey::from_proof(ProofTree::present(&proof))
        .expect("Should create absent NodeKey")
        .into_result();

    let eq_value = catch_not_found(|| node_key.eq(&key)).ok();
    assert_eq!(eq_value, None);

    let cmp_value = catch_not_found(|| node_key.cmp(&key)).ok();
    assert_eq!(cmp_value, None);
}
