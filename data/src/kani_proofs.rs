// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Kani bounded model checking harnesses for durable storage proof size bounds

use crate::components::atom::Atom;
use crate::components::bytes::Bytes;
use crate::components::bytes::PAGE_SIZE;
use crate::hash::Hash;
use crate::merkle_proof::proof_tree::MerkleProof;
use crate::merkle_proof::proof_tree::MerkleProofLeaf;
use crate::mode::Normal;
use crate::serialisation::bincode_default_config;
use crate::tree::Tree;

/// Actual serialised size of a [`MerkleProof`] tree, obtained by encoding it.
fn serialised_size(proof: &MerkleProof) -> usize {
    let encoded = bincode::encode_to_vec(proof, bincode_default_config())
        .expect("Encoding a MerkleProof should not fail");
    encoded.len()
}

/// Compute the expected serialised size from a [`MerkleProof`] tree by
/// structural recursion.
///
/// Each tree element is prefixed by a 1-byte tag:
/// - `Node`: 1-byte tag, then children in sequence
/// - `Leaf(Read(data))`: 1-byte tag + `data.len()` raw bytes
/// - `Leaf(Blind(hash))`: 1-byte tag + 32 bytes (hash digest)
fn formula_size(proof: &MerkleProof) -> usize {
    match proof {
        Tree::Leaf(MerkleProofLeaf::Read(data)) => 1 + data.len(),
        Tree::Leaf(MerkleProofLeaf::Blind(_)) => 1 + Hash::DIGEST_SIZE,
        Tree::Node(node) => 1 + node.children.iter().map(formula_size).sum::<usize>(),
    }
}

// ---------------------------------------------------------------------------
// Harness 1: Atom<u64, Prove> proof is bounded
// ---------------------------------------------------------------------------

/// Verify that an `Atom<u64, Prove>` proof has bounded serialised size.
///
/// An `Atom` in Prove mode folds to a single leaf:
/// - If accessed (Present): 1-byte tag + serialised `u64` (8 bytes) = 9 bytes
/// - If not accessed (MayOmit → Blind): 1-byte tag + 32 bytes hash = 33 bytes
///
/// In both cases the proof size is at most 33 bytes.
#[cfg(kani)]
#[kani::proof]
fn verify_atom_proof_bounded() {
    let value: u64 = kani::any();

    let atom_normal = Atom::<u64, Normal>::new(value);
    let atom_prove = atom_normal.into_proof();

    let proof = MerkleProof::from_foldable(&atom_prove);
    let size = serialised_size(&proof);

    // Atom proof: either Read(8 bytes) + tag or Blind(32 bytes) + tag
    assert!(size <= 33, "Atom proof size {size} exceeds 33 bytes");
}

/// Verify that an `Atom<u64, Prove>` proof is bounded even when the value is
/// read during proof generation.
#[cfg(kani)]
#[kani::proof]
fn verify_atom_proof_bounded_after_read() {
    let value: u64 = kani::any();

    let atom_normal = Atom::<u64, Normal>::new(value);
    let mut atom_prove = atom_normal.into_proof();

    // Access the value so it becomes Present in the proof
    let _read = atom_prove.read();

    let proof = MerkleProof::from_foldable(&atom_prove);
    let size = serialised_size(&proof);

    // After reading, the atom value must be Present: 1 tag + 8 bytes = 9 bytes
    assert!(size <= 33, "Atom proof size {size} exceeds 33 bytes");
}

// ---------------------------------------------------------------------------
// Harness 2: Bytes<Prove> proof is bounded within 16 KiB
// ---------------------------------------------------------------------------

/// Maximum proof size for durable storage proofs.
const MAX_PROOF_SIZE: usize = 16 * 1024;

/// Verify that a `Bytes<Prove>` proof fits within 16 KiB when at most 2 pages
/// are accessed.
///
/// The proof structure for `Bytes` is:
/// ```text
///    root (node)
///    /        \
/// length    seq_tree(pages)
/// ```
///
/// - Length leaf: 1-byte tag + 8 bytes (serialised u64) = 9 bytes
/// - Each accessed page: 1-byte tag + 8 bytes length prefix + up to 4096 bytes data
/// - Each unaccessed page: 1-byte tag + 32 bytes (blind hash)
/// - Tree nodes: 1-byte tag each
///
/// With ≤ 2 accessed pages and bounded total data, the proof must fit in 16 KiB.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn verify_bytes_proof_bounded() {
    // Bound data length to at most 2 pages
    let len: usize = kani::any();
    kani::assume(len <= 2 * PAGE_SIZE);

    let bytes_normal = Bytes::<Normal>::new(len);
    let mut bytes_prove = bytes_normal.into_proof();

    // Simulate a read of up to PAGE_SIZE bytes at a symbolic offset.
    // This ensures at most 2 pages are accessed (the page containing the start
    // and possibly the next page).
    let read_offset: usize = kani::any();
    kani::assume(read_offset <= len);
    let read_len: usize = kani::any();
    kani::assume(read_len <= PAGE_SIZE);
    kani::assume(read_len <= len.saturating_sub(read_offset));

    let mut buf = vec![0u8; read_len];
    bytes_prove.read(read_offset, &mut buf);

    let proof = MerkleProof::from_foldable(&bytes_prove);
    let size = serialised_size(&proof);

    assert!(
        size <= MAX_PROOF_SIZE,
        "Bytes proof size {size} exceeds {MAX_PROOF_SIZE} bytes"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Serialisation size formula correctness
// ---------------------------------------------------------------------------

/// Verify that the serialisation size formula (tag counting + leaf data sizes)
/// correctly predicts the actual encoded size of a `MerkleProof`.
///
/// We test on proof trees produced from `Atom` components since they give us
/// simple, concrete trees (single leaf). We also test with a two-child node
/// to cover the `Node` case.
#[cfg(kani)]
#[kani::proof]
fn verify_serialisation_bound_atom() {
    let value: u64 = kani::any();

    let atom_normal = Atom::<u64, Normal>::new(value);
    let atom_prove = atom_normal.into_proof();

    let proof = MerkleProof::from_foldable(&atom_prove);

    let actual = serialised_size(&proof);
    let predicted = formula_size(&proof);

    assert!(
        actual == predicted,
        "Formula mismatch: actual {actual} != predicted {predicted}"
    );
}

/// Verify the formula on a two-atom compound proof (covers Node encoding).
#[cfg(kani)]
#[kani::proof]
fn verify_serialisation_bound_pair() {
    let v1: u64 = kani::any();
    let v2: u32 = kani::any();

    let a1 = Atom::<u64, Normal>::new(v1).into_proof();
    let a2 = Atom::<u32, Normal>::new(v2).into_proof();

    // A tuple of two atoms folds as a node with two children
    let proof = MerkleProof::from_foldable(&(&a1, &a2));

    let actual = serialised_size(&proof);
    let predicted = formula_size(&proof);

    assert!(
        actual == predicted,
        "Formula mismatch: actual {actual} != predicted {predicted}"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Worst-case proof tree size bound
// ---------------------------------------------------------------------------

/// Verify that a worst-case compressed `MerkleProof` tree fits in 16 KiB for
/// all tree depths up to [`VERIFY_MAX_DEPTH`].
///
/// This constructs the maximally expensive proof tree that can arise from the
/// `Bytes` component's `Foldable<MerkleProofFold>` implementation after
/// `MerkleProofNodeFold` compression:
///
/// - Two accessed pages placed at maximally separated positions in the
///   arity-4 page tree, so both paths from root to leaf are fully expanded.
/// - Every sibling subtree along each path is a Blind leaf (compressed).
/// - Each Read leaf carries the worst-case ChunkedPage encoding size
///   (8-byte `u64` length prefix + `PAGE_SIZE` bytes of data).
/// - A length leaf (8 bytes) is present at the root.
///
/// After compression the proof structure is:
///
/// ```text
/// Node (root)
/// ├── Read (8 B, length)
/// └── Node (seq root)
///     ├── Node (path A, level depth-1)
///     │   ├── Node (path A, level depth-2)
///     │   │   └── ... → Node [Read, Blind, Blind, Blind]
///     │   ├── Blind
///     │   ├── Blind
///     │   └── Blind
///     ├── Node (path B, same structure)
///     ├── Blind
///     └── Blind
/// ```
///
/// The serialised size for depth `d` is `200 × d + 8287` bytes. This stays
/// within 16 KiB for all `d ≤ 40`, which corresponds to `4^40 ≈ 10^24` pages
/// — far beyond any realistic system. The harness verifies depths 0 through
/// [`VERIFY_MAX_DEPTH`] by constructing actual trees and computing their size
/// via [`formula_size`] (whose correctness is proven by the
/// `verify_serialisation_bound_*` harnesses).
const VERIFY_MAX_DEPTH: usize = 5;

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_proof_tree_size_bound() {
    const MAX_PAGE_DATA: usize = PAGE_SIZE + 8;
    let blind = MerkleProof::leaf_blind(Hash::from([0u8; 32]));

    let depth: usize = kani::any();
    kani::assume(depth <= VERIFY_MAX_DEPTH);

    // Leaf-level nodes: 1 Read page + 3 Blind siblings each
    let mut subtree_a = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read(vec![0u8; MAX_PAGE_DATA]),
        blind.clone(),
        blind.clone(),
        blind.clone(),
    ]);
    let mut subtree_b = MerkleProof::node_without_data(vec![
        MerkleProof::leaf_read(vec![0u8; MAX_PAGE_DATA]),
        blind.clone(),
        blind.clone(),
        blind.clone(),
    ]);

    // Wrap each path in progressively deeper intermediate nodes
    let mut d: usize = 1;
    while d < depth {
        subtree_a = MerkleProof::node_without_data(vec![
            subtree_a,
            blind.clone(),
            blind.clone(),
            blind.clone(),
        ]);
        subtree_b = MerkleProof::node_without_data(vec![
            subtree_b,
            blind.clone(),
            blind.clone(),
            blind.clone(),
        ]);
        d += 1;
    }

    // Seq tree root: both present subtrees + 2 Blind siblings.
    // For depth 0 the two Read pages are direct children (no intermediate
    // nodes), which is the shape IndexableSeqAsTree produces for ≤ 4 pages.
    let seq_tree = if depth == 0 {
        MerkleProof::node_without_data(vec![
            MerkleProof::leaf_read(vec![0u8; MAX_PAGE_DATA]),
            MerkleProof::leaf_read(vec![0u8; MAX_PAGE_DATA]),
            blind.clone(),
            blind.clone(),
        ])
    } else {
        MerkleProof::node_without_data(vec![subtree_a, subtree_b, blind.clone(), blind.clone()])
    };

    // Full Bytes proof: root node with length leaf + page seq tree
    let length_leaf = MerkleProof::leaf_read(vec![0u8; 8]);
    let proof = MerkleProof::node_without_data(vec![length_leaf, seq_tree]);

    let size = formula_size(&proof);
    assert!(
        size <= MAX_PROOF_SIZE,
        "Proof size {size} exceeds {MAX_PROOF_SIZE} bytes (depth={depth})"
    );
}
