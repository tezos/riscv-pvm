// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for [`NodeKey`]

use std::cmp::Ordering;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::utils::catch_not_found;
use octez_riscv_data::mode_test;
use octez_riscv_data::serialisation::serialise;
use proptest::prelude::*;

use super::NodeKey;
use super::NodeKeyMode;
use super::PAGE_SIZE;
use crate::key::KEY_MAX_SIZE;
use crate::key::Key;

/// Serialised size of a blinded proof leaf: a tag byte plus the hash it stands for.
const BLIND_LEAF_SIZE: usize = 1 + Hash::DIGEST_SIZE;

/// Serialised size of a proof leaf holding a full page: a tag byte, the page's one-byte
/// length prefix, and its bytes.
const PAGE_LEAF_SIZE: usize = 1 + 1 + PAGE_SIZE;

/// Serialised size of a key's length leaf: a tag byte and the length.
const LENGTH_LEAF_SIZE: usize = 1 + 1;

/// Serialised size of a proof for an ordering of two [`KEY_MAX_SIZE`] keys that differ in a
/// full page. What the ordering costs is set by the page tree's depth, which is the same
/// wherever the keys diverge, so this is what any ordering of two such keys costs - bar the
/// last page of a [`KEY_MAX_SIZE`] key, which is short of a full page and so can only be
/// cheaper.
const CMP_PROOF_SIZE: usize = 1
    + LENGTH_LEAF_SIZE
    // The page tree's root, the subtree spanning the pages the comparison does not reach,
    // and the revealed page beside the blinded page it shares a parent with.
    + 1
    + BLIND_LEAF_SIZE
    + (1 + PAGE_LEAF_SIZE + BLIND_LEAF_SIZE);

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

/// A blinded key still answers equality, in both directions, by hashing the key it is
/// compared against.
#[test]
fn verify_blinded_eq() {
    proptest!(|(key: Key, other: Key)| {
        let node_key = blinded_node_key(&key);

        prop_assert_eq!(catch_not_found(|| node_key.eq(&key)).ok(), Some(true));
        prop_assert_eq!(
            catch_not_found(|| node_key.eq(&other)).ok(),
            Some(key == other)
        );
    });
}

/// A blinded key can settle the [`Ordering::Equal`] case of a comparison, but not the
/// orderings that need the key's pages.
#[test]
fn verify_blinded_cmp() {
    proptest!(|(key: Key, other: Key)| {
        let node_key = blinded_node_key(&key);

        prop_assert_eq!(
            catch_not_found(|| node_key.cmp(&key)).ok(),
            Some(Ordering::Equal)
        );
        prop_assert_eq!(
            catch_not_found(|| node_key.cmp(&other)).ok(),
            (key == other).then_some(Ordering::Equal)
        );
    });
}

/// A key that was only ever compared for equality is blinded whole in the proof, and
/// remains usable for equality on the verifying side.
#[test]
fn eq_blinds_the_key_in_the_proof() {
    let key = Key::new(&[1; KEY_MAX_SIZE]).unwrap();
    let other = Key::new(&[2; KEY_MAX_SIZE]).unwrap();

    let (merkle_proof, key_verify) = prove(&key, |node_key| {
        assert!(!node_key.eq(&other));
    });

    // Neither the key nor its pages are in the proof - only the hash it folds to.
    assert_eq!(merkle_proof, blinded_proof(&key));

    assert_eq!(catch_not_found(|| key_verify.eq(&other)).ok(), Some(false));
    assert_eq!(catch_not_found(|| key_verify.eq(&key)).ok(), Some(true));
}

/// An exact match is settled by the key's hash, so it blinds the key whole too.
#[test]
fn equal_cmp_blinds_the_key_in_the_proof() {
    let key = Key::new(&[1; KEY_MAX_SIZE]).unwrap();

    let (merkle_proof, key_verify) = prove(&key, |node_key| {
        assert_eq!(node_key.cmp(&key), Ordering::Equal);
    });

    assert_eq!(merkle_proof, blinded_proof(&key));
    assert_eq!(
        catch_not_found(|| key_verify.cmp(&key)).ok(),
        Some(Ordering::Equal)
    );
}

/// Two keys that differ in their first page are ordered by that page alone: the rest of
/// the key never enters the comparison, so its subtree is blinded away.
#[test]
fn cmp_reveals_only_the_diverging_page() {
    let key = Key::new(&[1; KEY_MAX_SIZE]).unwrap();
    let other = Key::new(&[2; KEY_MAX_SIZE]).unwrap();

    let (merkle_proof, key_verify) = prove(&key, |node_key| {
        assert_eq!(node_key.cmp(&other), Ordering::Less);
    });

    assert_eq!(proof_size(&merkle_proof), CMP_PROOF_SIZE);
    assert_eq!(
        catch_not_found(|| key_verify.cmp(&other)).ok(),
        Some(Ordering::Less)
    );
}

/// Diverging in the last page means every page before it has to be proven equal - but they
/// are proven together, against the subtrees spanning them, which costs the same as the
/// single subtree an early divergence leaves untouched. The last page is therefore never the
/// dearest page to diverge in.
#[test]
fn cmp_diverging_late_proves_the_earlier_pages_as_subtrees() {
    let pages = KEY_MAX_SIZE.div_ceil(PAGE_SIZE);
    let last = divergence_proof_size(pages - 1);

    for page in 0..pages - 1 {
        assert!(
            last <= divergence_proof_size(page),
            "diverging in the last page should cost no more than diverging in page {page}"
        );
    }
}

/// A key that is a prefix of the key it is compared against agrees on every page they
/// share, so their lengths decide the ordering and no page is revealed.
#[test]
fn cmp_of_a_prefix_is_decided_by_length() {
    let key = Key::new(&[1; 2 * PAGE_SIZE]).unwrap();
    let other = Key::new(&[1; PAGE_SIZE]).unwrap();

    let (merkle_proof, key_verify) = prove(&key, |node_key| {
        assert_eq!(node_key.cmp(&other), Ordering::Greater);
    });

    // Only the length and a hash per page: none of the key's bytes are revealed.
    assert_eq!(
        proof_size(&merkle_proof),
        1 + LENGTH_LEAF_SIZE + 1 + 2 * BLIND_LEAF_SIZE
    );
    assert_eq!(
        catch_not_found(|| key_verify.cmp(&other)).ok(),
        Some(Ordering::Greater)
    );
}

/// The comparison a proof was generated for is the only one it has to order. A key that
/// diverges from this one in a page the proof blinded away can still be told apart from
/// it - the blinded subtree's hash settles that much - but ordering the two needs the page
/// itself, which is a missing-state failure rather than a wrong answer.
#[test]
fn cmp_against_an_unproven_key_is_not_found() {
    let key = Key::new(&[1; KEY_MAX_SIZE]).unwrap();
    let other = Key::new(&[2; KEY_MAX_SIZE]).unwrap();

    let mut bytes = [1; KEY_MAX_SIZE];
    bytes[KEY_MAX_SIZE - 1] = 3;
    let unproven = Key::new(&bytes).unwrap();

    let (_, key_verify) = prove(&key, |node_key| {
        assert_eq!(node_key.cmp(&other), Ordering::Less);
    });

    assert_eq!(catch_not_found(|| key_verify.cmp(&unproven)).ok(), None);
    assert_eq!(
        catch_not_found(|| key_verify.eq(&unproven)).ok(),
        Some(false)
    );
}

/// Keys sharing a prefix are what page-wise comparison exists for: arbitrary keys almost
/// always differ in their first page, which never exercises proving the pages before the
/// divergence equal.
#[test]
fn cmp_of_keys_sharing_a_prefix_verifies_from_the_blinded_pages() {
    proptest!(|(
        shared in prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE),
        lhs_tail in prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE),
        rhs_tail in prop::collection::vec(any::<u8>(), 0..=KEY_MAX_SIZE),
    )| {
        let extend = |tail: &[u8]| {
            let mut bytes = shared.clone();
            bytes.extend_from_slice(tail);
            bytes.truncate(KEY_MAX_SIZE);

            Key::new(&bytes).expect("Bytes within KEY_MAX_SIZE are a valid key")
        };

        let key = extend(&lhs_tail);
        let other = extend(&rhs_tail);
        let expected = key.cmp(&other);

        let (_, key_verify) = prove(&key, |node_key| {
            assert_eq!(node_key.cmp(&other), expected);
        });

        prop_assert_eq!(
            catch_not_found(|| key_verify.cmp(&other)).ok(),
            Some(expected)
        );
    });
}

/// An equality check is settled against the hash of the key as a whole - but an ordering
/// in the same proof puts the key's tree there instead, and that hash with it. The
/// equality then has to be answered from the tree, against a key whose pages run out
/// before the blinded chunk spanning the rest does.
#[test]
fn eq_still_answers_once_an_ordering_has_opened_the_key() {
    let key = Key::new(&[1; KEY_MAX_SIZE]).unwrap();

    // Shorter than `key`, and a whole number of pages, so it shares every page it has.
    let shorter = Key::new(&[1; 3 * PAGE_SIZE]).unwrap();

    let mut bytes = [1; KEY_MAX_SIZE];
    bytes[0] = 2;
    let ordered = Key::new(&bytes).unwrap();

    let (_, key_verify) = prove(&key, |node_key| {
        assert!(!node_key.eq(&shorter));
        assert_eq!(node_key.cmp(&ordered), Ordering::Less);
    });

    assert_eq!(
        catch_not_found(|| key_verify.eq(&shorter)).ok(),
        Some(false)
    );
    assert_eq!(
        catch_not_found(|| key_verify.cmp(&ordered)).ok(),
        Some(Ordering::Less)
    );
}

/// Generate a proof for the comparisons `compare` makes against `key`, and reconstruct
/// the key from it in [`Verify`] mode.
///
/// [`Verify`]: octez_riscv_data::mode::Verify
fn prove(
    key: &Key,
    compare: impl FnOnce(&NodeKey<Prove<'_>>),
) -> (MerkleProof, NodeKey<octez_riscv_data::mode::Verify>) {
    let key_normal = NodeKey::new(key.clone());
    let key_prove = key_normal.start_proof();

    compare(&key_prove);

    let merkle_proof = MerkleProof::from_foldable(&key_prove);

    // Whatever the proof carries, it must still hash to the key it was generated from.
    let key_verify = NodeKey::from_proof(ProofTree::present(&merkle_proof))
        .expect("from_proof should succeed")
        .into_result();
    assert_eq!(
        PartialHash::from_foldable(Some(merkle_proof.clone()), &key_verify)
            .to_hash()
            .expect("the proof should determine the key's hash"),
        Hash::from_foldable(&key_normal)
    );

    (merkle_proof, key_verify)
}

/// Prove an ordering of two [`KEY_MAX_SIZE`] keys that first differ in `page`, check that
/// the verifier orders them the same way, and give back the size of the proof it did so
/// against.
fn divergence_proof_size(page: usize) -> usize {
    let mut bytes = [1; KEY_MAX_SIZE];
    let key = Key::new(&bytes).unwrap();

    bytes[page * PAGE_SIZE] = 2;
    let other = Key::new(&bytes).unwrap();

    let (merkle_proof, key_verify) = prove(&key, |node_key| {
        assert_eq!(node_key.cmp(&other), Ordering::Less);
    });
    assert_eq!(
        catch_not_found(|| key_verify.cmp(&other)).ok(),
        Some(Ordering::Less)
    );

    proof_size(&merkle_proof)
}

/// The serialised size of a proof.
fn proof_size(proof: &MerkleProof) -> usize {
    serialise(proof)
        .expect("Serialising the proof should succeed")
        .len()
}

/// The proof of a key that is blinded whole: the single hash the key folds to.
fn blinded_proof(key: &Key) -> MerkleProof {
    MerkleProof::leaf_blind(Hash::from_foldable(&NodeKey::<Normal>::new(key.clone())))
}

/// Build a [`NodeKey`] in [`Verify`] mode whose key is blinded whole.
///
/// [`Verify`]: octez_riscv_data::mode::Verify
fn blinded_node_key(key: &Key) -> NodeKey<octez_riscv_data::mode::Verify> {
    let proof = blinded_proof(key);

    NodeKey::from_proof(ProofTree::present(&proof))
        .expect("Should create blinded NodeKey")
        .into_result()
}
