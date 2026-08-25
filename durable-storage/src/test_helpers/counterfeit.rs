// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Utilities for proof safety testing that allow generation of proofs for NDS instances that
//! cannot be created in the normal way, because they are too large, or may even be invalid. We
//! call such databases 'counterfeits'.

#![cfg(test)]

use std::sync::Arc;

use bytes::Bytes;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::Normal;
use tokio::runtime::Handle;

use crate::avl::node::Node;
use crate::avl::resolver::LazyDataId;
use crate::avl::resolver::LazyNodeId;
use crate::avl::resolver::LazyResolver;
use crate::avl::resolver::LazyTreeId;
use crate::avl::tree::Tree;
use crate::database::Database;
use crate::key::Key;
use crate::merkle_layer::MerkleLayer;
use crate::merkle_worker::BackgroundWriteableKeyValueStore;
use crate::merkle_worker::MerkleWorker;

/// To build a valid-seeming counterfeit we need the sequence of nodes (built from the deepest
/// layer upwards) to have `Key`s that appear in a sensible order.
///
/// This utility function can be given any slice of nodes where a node is represented by its `Key`
/// and some other data. It will sort the slice around a given 'pivot' `key` so that the keys are
/// monotonically (in both directions) further from the pivot. This slice can then be passed to
/// `single_key_counterfeit` and it will produce a counterfeit with keys that respect the tree
/// shape.
///
/// For every index in the slice, the reordered node will be below or above the pivot key depending
/// on the node at that index before sorting. This means that randomly generating sequences of keys
/// will randomly generate different tree shapes, while ensuring a uniform distribution of keys.
///
/// This approach has some nice properties: it means that if the randomly generated pivot key is
/// very small, the resulting tree shape will be left-leaning in a natural way. (In particular, the
/// probability of generating a completely left or right-leaning tree is not negligible. If we
/// chose to go left or right at each node independently, that would be extremely unlikely for
/// trees of any depth.)
fn sort_nodes<A: Clone>(key: &Key, nodes: &mut [(Key, A)]) {
    let mut below = vec![];
    let mut above = vec![];
    let mut is_below = vec![];

    for (k, h) in nodes.iter() {
        if *k < *key {
            below.push((k.clone(), h.clone()));
            is_below.push(true);
        } else {
            above.push((k.clone(), h.clone()));
            is_below.push(false);
        }
    }

    below.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
    above.sort_by(|(k1, _), (k2, _)| k2.cmp(k1));

    for i in 0..nodes.len() {
        if is_below[i] {
            nodes[i] = below.pop().expect("");
        } else {
            nodes[i] = above.pop().expect("");
        }
    }
}

/// Create a normal mode database that is in fact a counterfeit: it contains many nodes that are
/// impossible to resolve because they do not really exist. The purpose of this counterfeit is to
/// contain a specific `key` at a specific depth (given by the length of the `nodes` slice). This
/// counterfeit will then provide just enough real structure for a single-node database operation
/// (such as read or write) to be performed on that node.
fn single_key_counterfeit<KV: BackgroundWriteableKeyValueStore>(
    key: &Key,
    data: &Bytes,
    nodes: &mut [(Key, (Hash, Vec<u8>))],
    repo: &KV::Repo,
    handle: &Handle,
) -> Database<KV, Normal> {
    let persistent = KV::new(repo).expect("can create store from repo");
    let persistent = Arc::new(persistent);
    persistent.set(key, data).expect("can set key in store");
    let mut resolver = LazyResolver::new(persistent.clone());

    let mut tree = Tree::default();

    tree.set(key, data, &mut resolver)
        .expect("can set key in tree");

    sort_nodes(key, nodes);

    for (other_key, (hash, v)) in nodes {
        let mut data = octez_riscv_data::components::bytes::Bytes::new(v.len());
        data.write(0, v);
        let (left, right) = if key > other_key {
            (LazyTreeId::from(*hash), LazyTreeId::from(tree))
        } else {
            (LazyTreeId::from(tree), LazyTreeId::from(*hash))
        };

        let new_node = Node::from_raw(0, other_key.clone(), LazyDataId::from(data), left, right);
        let new_id = LazyNodeId::from(new_node);
        tree = Tree::from_raw(new_id);
    }

    let layer = MerkleLayer::new_raw(tree, persistent.clone(), resolver);

    let merkle = MerkleWorker::from_layer(handle, layer);

    Database::new_raw(persistent, merkle)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use octez_riscv_data::codec;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_proof::FromProof;
    use octez_riscv_data::merkle_proof::proof::Proof;
    use octez_riscv_data::merkle_proof::proof::deserialise_proof;
    use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
    use octez_riscv_data::merkle_proof::proof_tree::ProofTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::ProvableExt;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::serialisation::serialise;
    use proptest::collection;
    use proptest::prelude::*;

    use crate::database::Database;
    use crate::key::Key;
    use crate::storage::in_memory::InMemoryKeyValueStore;
    use crate::storage::in_memory::InMemoryRepo;
    use crate::test_helpers::proof_size::BALANCE_FACTOR_LEAF;
    use crate::test_helpers::proof_size::BLIND_LEAF;
    use crate::test_helpers::proof_size::LEN_LEAF;
    use crate::test_helpers::proof_size::TAG_BYTES;
    use crate::test_helpers::proof_size::TREE_WRAP;
    use crate::test_helpers::proof_size::avl_path_node;
    use crate::test_helpers::proof_size::key_leaf;

    /// Every key in these tests is [`COUNTERFEIT_KEY_LEN`] bytes, to make the size of the
    /// generated proofs predictable.
    fn key_strategy() -> impl Strategy<Value = Key> {
        collection::vec(any::<u8>(), COUNTERFEIT_KEY_LEN)
            .prop_map(|bs| Key::new(&bs).expect("Can convert the generated bytes to a key"))
    }

    /// The data vector generated for each node is of random length, to confirm that it is not
    /// included in the proof. (It is required in the counterfeit because the prove-mode resolver
    /// accesses it.)
    fn node_strategy() -> impl Strategy<Value = (Key, (Hash, Vec<u8>))> {
        (
            key_strategy(),
            any::<[u8; Hash::DIGEST_SIZE]>(),
            collection::vec(any::<u8>(), 0..100),
        )
            .prop_map(|(k, bs, data)| (k, (Hash::from(bs), data)))
    }

    /// Do the two vectors represent the same set, perhaps just in a different order?
    fn assert_same_set<A: std::fmt::Debug + PartialEq + Ord>(a: Vec<A>, b: Vec<A>) {
        let mut a = a;
        let mut b = b;
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    /// Every key generated by [`key_strategy`] is this many bytes.
    const COUNTERFEIT_KEY_LEN: usize = 32;

    /// The node holding the key under test, at the bottom of the search path.
    ///
    /// Its encoding coincides with [`avl_path_node`] — tree wrapper, node tag, balance factor
    /// and key leaves, and two blinded subtrees — but the two blinds are different things. A
    /// node *on* the path blinds its untouched value and the sibling the path skips; this node
    /// blinds both of its children, which are empty. `Node::get` returns as soon as the keys
    /// compare equal, so neither child is ever resolved and both are emitted as hashes rather
    /// than as present-but-empty subtrees.
    const COUNTERFEIT_TARGET_NODE: usize = TREE_WRAP
        + TAG_BYTES
        + BALANCE_FACTOR_LEAF
        + key_leaf(COUNTERFEIT_KEY_LEN)
        + 2 * BLIND_LEAF;

    /// The target node's value, opened by the read rather than blinded.
    ///
    /// This is `proof_size::value_open` for an empty value: no pages are touched, so the page
    /// tree contributes nothing and the sum falls back on its [`BLIND_LEAF`] floor.
    const EMPTY_VALUE_OPEN: usize = TAG_BYTES + LEN_LEAF + BLIND_LEAF;

    /// The length of the part of the proof which is present regardless of the depth of the tree:
    /// the target node, its opened value, and the final state hash that the serialised [`Proof`]
    /// is prefixed with.
    const ZERO_DEPTH_PROOF_LENGTH: usize =
        COUNTERFEIT_TARGET_NODE + EMPTY_VALUE_OPEN + Hash::DIGEST_SIZE;

    /// The extra length of proof added for each extra layer of depth.
    ///
    /// Each layer of the counterfeit contributes exactly one accessed node on the search path:
    /// its key is read to steer the traversal, its value is never touched, and one of its two
    /// children is a hash that is never resolved. That is precisely what [`avl_path_node`]
    /// charges for, so this tracks any change to the node encoding automatically.
    const PROOF_LENGTH_PER_LAYER: usize = avl_path_node(COUNTERFEIT_KEY_LEN);

    proptest! {
        #[test]
        fn test_sort_nodes(nodes in collection::vec(node_strategy(), 0..500), key in key_strategy()) {
            let sorted_nodes = {
                let mut working = nodes.clone();
                super::sort_nodes(&key, &mut working[..]);
                working
            };

            let mut top_bound = key.clone();
            let mut bottom_bound = key.clone();
            for (k, _) in &sorted_nodes {
                if *k >= key {
                    assert!(*k >= top_bound);
                    top_bound = k.clone();
                } else {
                    assert!(*k <= bottom_bound);
                    bottom_bound = k.clone();
                }
            }

            assert_same_set(nodes, sorted_nodes);
        }

        /// Constructs proofs for tiny read operations on keys at arbitrary depths (up to depth
        /// 500) using counterfeit databases. Checks that the proofs can be verified (without the
        /// stack overflowing) and that the size of the proofs grows by a fixed number of bytes per
        /// layer.
        #[test]
        fn test_size_by_depth_of_read_proofs(
            nodes in proptest::collection::vec(node_strategy(), 0..500),
            key in key_strategy(),
        ) {
            let depth = nodes.len();
            let mut nodes = nodes.clone();

            let repo = InMemoryRepo::default();
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_stack_size(10 * 1024 * 1024)
                .build()
                .expect("building runtime should succeed");
            let handle = runtime.handle();

            let db: Database<InMemoryKeyValueStore, Normal> = super::single_key_counterfeit(
                &key,
                &Bytes::from(""),
                &mut nodes[..],
                &repo,
                handle,
            );

            let mut dst = [0u8; 4];

            {
                let mut buf = &mut dst[..];
                db.read(&key, 0, &mut buf).expect("can read key from db");
                assert_eq!(dst, [0u8; 4]);
            }

            let root_hash = Hash::from_foldable(&db);

            let prove_db = db.try_start_proof().expect("can start proof");

            {
                let mut buf = &mut dst[..];
                prove_db.read(&key, 0, &mut buf).expect("can read key from db");
                assert_eq!(dst, [0; 4]);
            }

            let merkle_proof = MerkleProof::from_foldable(&prove_db);
            let final_state_hash = Hash::from_foldable(&prove_db);
            let proof = Proof::new(merkle_proof, final_state_hash);
            let proof_bytes = serialise(&proof).expect("can serialise proof");

            let (reconstructed, _) =
                deserialise_proof::<codec::Bincode, Database<InMemoryKeyValueStore, Verify>, _>(
                    proof_bytes.clone().into_iter(),
                )
                .expect("can deserialise proof");

            let verify_db = Database::<InMemoryKeyValueStore, Verify>::from_proof(ProofTree::present(
                reconstructed.tree(),
            ))
                .expect("can construct verify mode db")
                .into_result();

            let verify_root_hash =
                PartialHash::from_foldable(Some(reconstructed.tree().clone()), &verify_db)
                    .to_hash()
                    .expect("can hash the verify mode db");
            assert_eq!(root_hash, verify_root_hash);

            {
                let mut buf = &mut dst[..];
                verify_db.read(&key, 0, &mut buf).expect("can read key from verify mode db");
                assert_eq!(dst, [0; 4]);
            }

            assert_eq!(proof_bytes.len(), ZERO_DEPTH_PROOF_LENGTH + depth * PROOF_LENGTH_PER_LAYER);
        }
    }
}
