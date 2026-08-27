// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Model for upper bound on Regsitry and Database proof size
//!
//! Every Database proof is checked against the worst case model for the current
//! number of keys in tree shape and key sizes: the deepest AVL shape (a
//! Fibonacci-minimal tree), every key at [`KEY_MAX_SIZE`], and the most
//! expensive rebalancing per operation type (one rotation for an insert,
//! `floor((depth - 1) / 2)` for a delete). Operation value sizes are real:
//! the lengths and the byte ranges an operation touches are taken from
//! the [`DatabaseReferenceModel`].
//!
//! The bounds mirror the proof encoding: pre-order tree with a one-byte tag per
//! node or leaf, a `Blind` leaf carrying a hash and a `Read` leaf carrying raw
//! bytes (see the `Encode` impl in `octez_riscv_data::merkle_proof::proof_tree`).
//!
//! Registry proofs are checked against [`MAX_REGISTRY_PROOF_SIZE`], whereas database
//! proofs are checked against [`DATABASE_MAX_PROOF_SIZE`] - which takes into
//! account that databases are wrapped in the registry component too.
//!
//! Compression makes [`BLIND_LEAF`] an upper bound rather than an exact cost for
//! every subtree the model treats as blinded. `MerkleProof::blind` leaves a leaf
//! shorter than [`Hash::DIGEST_SIZE`] inlined as a `Read` leaf instead of
//! replacing it by its hash, and a subtree that was resolved but holds no data
//! (an empty AVL child) is emitted as a present node rather than being blinded.
//! Both encodings are shorter than a tag plus a digest, so the model stays
//! conservative when it charges [`BLIND_LEAF`] for them.

use std::collections::BTreeSet;
use std::ops::Range;

use octez_riscv_data::components::bytes::NODE_ARITY as BYTES_NODE_ARITY;
use octez_riscv_data::components::bytes::PAGE_SIZE;
use octez_riscv_data::components::vector::NODE_ARITY as VECTOR_NODE_ARITY;
use octez_riscv_data::foldable::seq_tree::tree_depth;
use octez_riscv_data::hash::Hash;

use crate::key::KEY_MAX_SIZE;
use crate::test_helpers::database::DatabaseOperation;
use crate::test_helpers::database::DatabaseReferenceModel;
use crate::test_helpers::registry::RegistryOperation;

/// Ceiling on the serialised size of a whole NDS proof.
///
/// We limit it to half of the max size supported by L1 - as the WASM component
/// will also be included (which itself also fits into 16KiB).
const MAX_PROOF_SIZE: usize = 16 * 1024;

/// Serialised size of a proof tree tag. Tags are written as one byte each by the
/// `Encode` impl of `octez_riscv_data::merkle_proof::proof_tree::MerkleProof`.
pub(crate) const TAG_BYTES: usize = 1;

/// A `Blind` leaf: tag plus [`Hash::DIGEST_SIZE`] bytes. Also used as the charge
/// for any subtree the model considers blindable, which the encoder may in fact
/// emit in a shorter form (see the note in the module documentation).
pub(crate) const BLIND_LEAF: usize = TAG_BYTES + Hash::DIGEST_SIZE;

/// An accessed AVL tree wrapper: proof node tag plus a `Read` leaf holding the
/// one-byte present flag (see `fold_resolved_tree` in `crate::merkle_layer`).
pub(crate) const TREE_WRAP: usize = TAG_BYTES + TAG_BYTES + size_of::<bool>();

/// A `Read` leaf holding a `u64` length (fixed-int bincode encoding).
pub(crate) const LEN_LEAF: usize = TAG_BYTES + size_of::<u64>();

/// A `Read` leaf holding a node's `i8` balance factor (fixed-int bincode
/// encoding).
pub(crate) const BALANCE_FACTOR_LEAF: usize = TAG_BYTES + size_of::<i8>();

/// A `Read` leaf holding a node's [`Key`]: a one-byte length
/// prefix plus the key bytes (see the `Encode` impl in `crate::key`).
///
/// [`Key`]: crate::key::Key
pub(crate) const fn key_leaf(key_len: usize) -> usize {
    assert!(key_len <= KEY_MAX_SIZE);
    TAG_BYTES + 1 + key_len
}

/// An accessed node on the search path: tree wrapper, node tag, balance factor
/// and key leaves, blinded data and one blinded sibling subtree. The balance
/// factor is shorter than a hash so it is always included directly. The key is
/// also always included in full, never blinded: generating the proof reads
/// every accessed node's key. The subtree the path continues into is charged
/// by its own level; the terminal node's second child is charged
/// via [`terminal_blind`].
pub(crate) const fn avl_path_node(key_len: usize) -> usize {
    TREE_WRAP + TAG_BYTES + BALANCE_FACTOR_LEAF + key_leaf(key_len) + 2 * BLIND_LEAF
}

/// An accessed node off the search path (touched by a rebalancing rotation):
/// like [`avl_path_node`], but both children may be blinded.
///
/// However, `key` is additionally blinded, alongside the data and the two children.
fn avl_extra_node() -> usize {
    TREE_WRAP + TAG_BYTES + BALANCE_FACTOR_LEAF + 4 * BLIND_LEAF
}

/// The blinded second child of the last node on a path.
const fn terminal_blind(path_len: usize) -> usize {
    if path_len == 0 { 0 } else { BLIND_LEAF }
}

/// Worst-case depth of an AVL tree holding `entries` nodes: the largest height
/// whose minimal node count `N(h) = N(h - 1) + N(h - 2) + 1` still fits. The
/// counts are computed in `u128`, which cannot overflow before exceeding any
/// `usize` entry count.
fn max_avl_depth(entries: usize) -> usize {
    let entries = entries as u128;
    let (mut depth, mut prev, mut min_nodes) = (0, 0u128, 0u128);
    loop {
        let next = min_nodes + prev + 1;
        if next > entries {
            return depth;
        }
        (depth, prev, min_nodes) = (depth + 1, min_nodes, next);
    }
}

/// Cost of a worst-case search path in a tree of the given depth, every key
/// charged at [`KEY_MAX_SIZE`].
fn avl_search_path(depth: usize) -> usize {
    depth * avl_path_node(KEY_MAX_SIZE) + terminal_blind(depth)
}

/// Bound on an opened value subtree: the `Bytes` component folds as a node of
/// a `u64` length leaf and a page tree of arity [`BYTES_NODE_ARITY`] with
/// [`PAGE_SIZE`]-byte page leaves (each a tag, a `u64` chunk length and the
/// page's content). `byte_ranges` are the value's byte ranges read or written
/// by the operation; every touched page is charged its content plus one full
/// branch of blinded siblings per tree layer. The branch charge deliberately
/// over-counts: touched pages can share their upper layers in the real proof
/// and siblings that are themselves touched are not blinded.
fn value_open(value_len: usize, byte_ranges: &[Range<usize>]) -> usize {
    let pages = value_len.div_ceil(PAGE_SIZE);
    let depth = tree_depth(pages, BYTES_NODE_ARITY) as usize;
    let touched = touched_pages(value_len, byte_ranges);

    let page_leaves: usize = touched
        .iter()
        .map(|page| {
            let content = PAGE_SIZE.min(value_len - page * PAGE_SIZE);
            TAG_BYTES + size_of::<u64>() + content
        })
        .sum();
    let layers = depth * touched.len() * (TAG_BYTES + (BYTES_NODE_ARITY - 1) * BLIND_LEAF);

    TAG_BYTES + LEN_LEAF + (page_leaves + layers).max(BLIND_LEAF)
}

/// The pages of a value of `value_len` bytes touched by the given byte ranges,
/// after clamping each range to the value's length.
fn touched_pages(value_len: usize, byte_ranges: &[Range<usize>]) -> BTreeSet<usize> {
    let mut pages = BTreeSet::new();
    for range in byte_ranges {
        let start = range.start.min(value_len);
        let end = range.end.min(value_len);
        if start < end {
            pages.extend(start / PAGE_SIZE..=(end - 1) / PAGE_SIZE);
        }
    }
    pages
}

/// The single-byte range read by `record_resize_boundary_dependency` (see
/// `octez_riscv_data::components::bytes`) when a value is resized from
/// `prev_len` to `new_len`.
fn resize_boundary(prev_len: usize, new_len: usize) -> Range<usize> {
    let boundary = prev_len.min(new_len);
    if boundary == 0 || prev_len == new_len {
        0..0
    } else {
        boundary - 1..boundary
    }
}

/// The value byte ranges an operation reads or writes; empty ranges are
/// ignored by [`value_open`].
fn value_byte_ranges(op: &DatabaseOperation, value_len: usize) -> [Range<usize>; 2] {
    match op {
        DatabaseOperation::Read(_, offset, len) => [*offset..offset.saturating_add(*len), 0..0],
        DatabaseOperation::Write(_, offset, data) => {
            let end = offset.saturating_add(data.len());
            [*offset..end, resize_boundary(value_len, end.max(value_len))]
        }
        // TODO TZX-197: once the prover stops including pages fully covered by
        // writes, `Set` needs no pre-value content (the resize boundary byte is
        // always inside the overwritten prefix); drop both ranges here.
        DatabaseOperation::Set(_, data) => [0..data.len(), resize_boundary(value_len, data.len())],
        _ => [0..0, 0..0],
    }
}

/// One layer of a registry slot path: the node tag plus the blinded siblings of
/// the child the path descends into.
const REGISTRY_LAYER: usize = TAG_BYTES + (VECTOR_NODE_ARITY - 1) * BLIND_LEAF;

/// The envelope a full proof wraps around the proofs of the db's it
/// touches. This includes the registry envelope: registry's `u64` length leaf,
/// and, per touched slot, one [`REGISTRY_LAYER`] per layer of the tree.
const fn registry_envelope(tree_depth: usize, touched_dbs: usize) -> usize {
    let path_cost = tree_depth * REGISTRY_LAYER;

    TAG_BYTES + LEN_LEAF + path_cost * touched_dbs
}

/// Largest registry the nds host functions admit: `registry_resize` refuses to grow
/// the registry beyond `Int32.max_int` slots.
#[cfg(rocksdb_test_utils)]
const MAX_REGISTRY_LEN: usize = i32::MAX as usize;

/// Depth bound for the slot tree of the largest admissible registry: the tree is
/// [`VECTOR_NODE_ARITY`]-ary, so `tree_depth` never exceeds the logarithm of
/// [`MAX_REGISTRY_LEN`] in that base.
#[cfg(rocksdb_test_utils)]
const MAX_REGISTRY_DEPTH: usize = MAX_REGISTRY_LEN
    .next_power_of_two() // required as `ilog2` rounds down
    .ilog2()
    .div_ceil(VECTOR_NODE_ARITY.ilog2()) as usize;

/// Ceiling on the serialised size of a whole registry proof: the shape that
/// ships for any operation on a database held in a registry slot. Proofs above
/// it are reported by [`assert_registry_proof_size`].
const MAX_REGISTRY_PROOF_SIZE: usize = MAX_PROOF_SIZE - Hash::DIGEST_SIZE;

/// Ceiling on the serialised size of a standalone database proof.
///
/// A database proof within this ceiling still fits [`MAX_REGISTRY_PROOF_SIZE`]
/// once wrapped, whatever the length of the registry holding it.
///
/// Proofs above it are reported by [`assert_database_proof_size`].
#[cfg(rocksdb_test_utils)]
const DATABASE_MAX_PROOF_SIZE: usize =
    MAX_REGISTRY_PROOF_SIZE - registry_envelope(MAX_REGISTRY_DEPTH, 1);

/// Print a warning, or panic when `fail_on_warning` is set.
fn report_over_maximum(fail_on_warning: bool, message: String) {
    if fail_on_warning {
        panic!("{message}");
    }
    eprintln!("warning: {message}");
}

/// Assert an actual serialised proof size is within the modelled bound. Sizes
/// above `maximum`, named `ceiling` in the report, are reported as warnings, or
/// as test failures when `fail_on_warning` is set.
///
/// `bound` is the bound of _just_ the initial state proof of the component under test.
fn assert_proof_size(
    op: &impl std::fmt::Debug,
    actual: usize,
    bound: usize,
    ceiling: &str,
    maximum: usize,
    fail_on_warning: bool,
) {
    // Generated proofs additionally contain the final state hash.
    let bound = bound + Hash::DIGEST_SIZE;

    assert!(
        actual <= bound,
        "proof size {actual} exceeds the modelled bound {bound} for {op:?}"
    );

    if actual > maximum {
        report_over_maximum(
            fail_on_warning,
            format!("proof size {actual} exceeds {ceiling} {maximum} for {op:?}"),
        );
    }
}

/// Assert a database proof is within its modelled bound and
/// [`DATABASE_MAX_PROOF_SIZE`], leaving room for the registry envelope.
#[cfg(rocksdb_test_utils)]
pub(crate) fn assert_database_proof_size(
    op: &impl std::fmt::Debug,
    actual: usize,
    bound: usize,
    fail_on_warning: bool,
) {
    assert_proof_size(
        op,
        actual,
        bound,
        "DATABASE_MAX_PROOF_SIZE",
        DATABASE_MAX_PROOF_SIZE,
        fail_on_warning,
    );
}

/// Assert a registry proof — envelope included — is within its modelled bound
/// and [`MAX_REGISTRY_PROOF_SIZE`].
pub(crate) fn assert_registry_proof_size(
    op: &impl std::fmt::Debug,
    actual: usize,
    bound: usize,
    fail_on_warning: bool,
) {
    assert_proof_size(
        op,
        actual,
        bound,
        "MAX_REGISTRY_PROOF_SIZE",
        MAX_REGISTRY_PROOF_SIZE,
        fail_on_warning,
    );
}

/// Bound on the serialised proof size of a database operation: the worst-case
/// tree shape and key sizes admitted by the pre-operation model's entry count,
/// with the model's real value sizes. `None` if the operation is not a provable
/// step.
pub(crate) fn database_operation_proof_size_bound(
    model: &impl DatabaseReferenceModel,
    op: &DatabaseOperation,
) -> Option<usize> {
    let key = match op {
        DatabaseOperation::Commit
        | DatabaseOperation::Checkout
        | DatabaseOperation::CommitCheckoutRoundtrip => return None,
        DatabaseOperation::Hash => return Some(BLIND_LEAF),
        DatabaseOperation::Set(key, _)
        | DatabaseOperation::Write(key, _, _)
        | DatabaseOperation::Read(key, _, _)
        | DatabaseOperation::Delete(key)
        | DatabaseOperation::Exists(key)
        | DatabaseOperation::ValueLength(key) => key,
    };

    let depth = max_avl_depth(model.data().len());
    let exists = model.data().contains_key(key);
    let mut cost = 0;

    match op {
        DatabaseOperation::Set(..) | DatabaseOperation::Write(..) if !exists => {
            // A fresh insert rebalances at most once: one rotation touches
            // up to two nodes off the search path.
            cost += avl_search_path(depth) + 2 * avl_extra_node();
        }
        DatabaseOperation::Delete(_) if exists => {
            // A delete also walks the in-order-successor path, but the
            // successor is a descendant of the deleted node, so the union
            // of both paths is a single root-to-leaf path. On the way back
            // up, a rotation fires only at path nodes whose strictly
            // shorter subtree the path descended into, and such a step
            // descends two height levels while any other descends one: a
            // path of `p` nodes with `s` rotations satisfies
            // `p + s <= depth`. A rotation (up to two extra nodes) costs
            // more than a path node, so the worst case takes the maximal
            // `s = (depth - 1) / 2` with `p = depth - s`.
            let rotations = (depth - 1) / 2;
            cost += avl_search_path(depth - rotations);
            cost += rotations * 2 * avl_extra_node();
        }
        _ => cost += avl_search_path(depth),
    }

    // Only operations that open the value subtree pay for it. `Delete`
    // drops the node with its data child blinded (already charged by the
    // path node) and `Exists` checks node presence alone, so neither
    // reads the value or its length.
    let opens_value = matches!(
        op,
        DatabaseOperation::Set(..)
            | DatabaseOperation::Write(..)
            | DatabaseOperation::Read(..)
            | DatabaseOperation::ValueLength(_)
    );

    if exists && opens_value {
        // The path node's blinded-data charge is deliberately kept even
        // though the data child is opened and charged by `value_open`.
        let value_len = model.data()[key].len();
        cost += value_open(value_len, &value_byte_ranges(op, value_len));
    }

    // An untouched database is compressed to a single blinded leaf.
    Some(cost.max(BLIND_LEAF))
}

/// Bound on the serialised proof size of a registry operation, computed over the
/// pre-operation reference models (one per registry slot). `None` if the
/// operation is not a provable step.
///
/// The proof folds as the [`registry_envelope`] over the slot tree of arity
/// [`VECTOR_NODE_ARITY`], wrapping the bound of the database operation it
/// carries, if any.
pub(crate) fn registry_operation_proof_size_bound<M: DatabaseReferenceModel>(
    databases: &[M],
    op: &RegistryOperation,
) -> Option<usize> {
    let depth = tree_depth(databases.len(), VECTOR_NODE_ARITY) as usize;

    let (touched_slots, inner) = match op {
        RegistryOperation::CommitCheckoutRoundtrip => return None,
        RegistryOperation::Database(index, db_op) => {
            let model = databases.get(*index).expect("The index is in bounds");
            (1, database_operation_proof_size_bound(model, db_op)?)
        }
        RegistryOperation::GrowRegistry
        | RegistryOperation::ShrinkRegistry
        | RegistryOperation::ClearDatabase(_) => (1, BLIND_LEAF),
        RegistryOperation::CopyDatabase(..) | RegistryOperation::MoveDatabase(..) => {
            (2, 2 * BLIND_LEAF)
        }
    };

    Some(registry_envelope(depth, touched_slots) + inner)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use tezos_smart_rollup_constants::core::MAX_FILE_CHUNK_SIZE;

    use super::*;
    use crate::key::Key;

    #[test]
    #[expect(
        clippy::single_range_in_vec_init,
        reason = "the arrays are slices of byte ranges, not range initialisers"
    )]
    fn touched_pages_track_page_size() {
        let len = 4 * PAGE_SIZE + PAGE_SIZE / 2;

        assert_eq!(touched_pages(len, &[0..1]), BTreeSet::from([0]));
        assert_eq!(
            touched_pages(len, &[PAGE_SIZE - 1..PAGE_SIZE + 1]),
            BTreeSet::from([0, 1])
        );
        // Ranges are clamped to the value length.
        assert_eq!(
            touched_pages(len, &[len - 1..len + PAGE_SIZE]),
            BTreeSet::from([4])
        );
        // Empty and out-of-bounds ranges touch nothing.
        assert!(touched_pages(len, &[PAGE_SIZE..PAGE_SIZE, len..len + 1]).is_empty());
    }

    // The page counts behind the data cost of chunk-bounded operations,
    // stated in terms of [`PAGE_SIZE`] and [`MAX_FILE_CHUNK_SIZE`] so they
    // hold whatever values those parameters take: a `Write`'s data range may
    // span `ceil(MAX_FILE_CHUNK_SIZE / PAGE_SIZE) + 1` pages when unaligned
    // (its resize boundary byte never adds a page beyond the last old byte's),
    // while a `Set` starts at offset zero and so always touches one page
    // fewer.
    #[test]
    fn operation_page_counts_respect_chunk_and_page_parameters() {
        let key = Key::new(b"key").expect("valid key");
        let value_len = MAX_FILE_CHUNK_SIZE + 4 * PAGE_SIZE + PAGE_SIZE / 2;
        let chunk_pages = MAX_FILE_CHUNK_SIZE.div_ceil(PAGE_SIZE);

        let pages_for = |op: &DatabaseOperation| {
            touched_pages(value_len, &value_byte_ranges(op, value_len)).len()
        };

        let max_write = [
            0,
            1,
            PAGE_SIZE / 2,
            PAGE_SIZE - 1,
            PAGE_SIZE,
            2 * PAGE_SIZE - 1,
            value_len - 1,
            value_len,
        ]
        .into_iter()
        .map(|offset| {
            let chunk = Bytes::from(vec![0u8; MAX_FILE_CHUNK_SIZE]);
            let write = pages_for(&DatabaseOperation::Write(key.clone(), offset, chunk));
            let read = pages_for(&DatabaseOperation::Read(
                key.clone(),
                offset,
                MAX_FILE_CHUNK_SIZE,
            ));
            assert!(read <= write, "a read touches no more pages than a write");
            write
        })
        .max()
        .expect("the offset list is not empty");

        let max_set = [1, MAX_FILE_CHUNK_SIZE]
            .into_iter()
            .map(|len| {
                pages_for(&DatabaseOperation::Set(
                    key.clone(),
                    Bytes::from(vec![0u8; len]),
                ))
            })
            .max()
            .expect("the length list is not empty");

        assert_eq!(max_write, chunk_pages + 1);
        assert_eq!(max_set, chunk_pages);
    }

    #[test]
    fn max_registry_depth_bounds_every_registry_length() {
        for len in [0, 1, 2, 3, VECTOR_NODE_ARITY, 1 << 20, MAX_REGISTRY_LEN] {
            let depth = tree_depth(len, VECTOR_NODE_ARITY) as usize;
            assert!(
                depth <= MAX_REGISTRY_DEPTH,
                "a registry of {len} slots is {depth} layers deep, \
                 beyond MAX_REGISTRY_DEPTH {MAX_REGISTRY_DEPTH}"
            );
        }
    }

    /// A database proof at [`DATABASE_MAX_PROOF_SIZE`] must still fit
    /// [`MAX_REGISTRY_PROOF_SIZE`] once the registry envelope is added, for every
    /// registry length.
    #[test]
    fn database_ceiling_leaves_room_for_the_registry_envelope() {
        for len in [1, 2, 3, VECTOR_NODE_ARITY, 1 << 20, MAX_REGISTRY_LEN] {
            let depth = tree_depth(len, VECTOR_NODE_ARITY) as usize;
            let wrapped = DATABASE_MAX_PROOF_SIZE + registry_envelope(depth, 1);
            assert!(
                wrapped <= MAX_REGISTRY_PROOF_SIZE,
                "a maximal database proof in a registry of {len} slots serialises \
                 to {wrapped} bytes, beyond MAX_REGISTRY_PROOF_SIZE {MAX_REGISTRY_PROOF_SIZE}"
            );
        }
    }

    #[test]
    fn max_avl_depth_matches_minimal_tree_sizes() {
        let expected = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (6, 3),
            (7, 4),
            (11, 4),
            (12, 5),
            (u64::MAX as usize, 91),
        ];
        for (entries, depth) in expected {
            assert_eq!(max_avl_depth(entries), depth, "for {entries} entries");
        }
    }
}
