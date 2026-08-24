// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Helpers for sequences that need to fold and unfold like trees

use std::ops::Range;

use crate::codec::LeafCodec;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::tree::Tree;

/// Driver for `Foldable` that lets you turn an indexable sequence into a tree-like structure where
/// the leaves are the items of the sequence
pub struct IndexableSeqAsTree<'a, L, G> {
    /// Total length of the sequence (i.e. not just the number of items in the current chunk)
    total_len: usize,

    /// Depth of the current chunk
    current_depth: u32,

    /// Index where the current chunk starts
    current_start: usize,

    /// Maximum number of children per node
    arity: usize,

    /// Item generator (e.g. retrieval function for items)
    generator: &'a G,

    _leaf: std::marker::PhantomData<L>,
}

impl<'a, L, G> IndexableSeqAsTree<'a, L, G> {
    /// Construct the [`Foldable`] driver for an indexable sequence.
    ///
    /// `len` is the length of the sequence. `arity` is the maximum number of children per node.
    /// `generator` is a function that, given an index, returns the corresponding item.
    ///
    /// It's worth clarifying the slightly unintuitive behaviour of this in the two special cases
    /// of length one and length zero: for length zero, we construct a tree with a single empty
    /// node, whereas for length one we don't need to construct _any nodes at all_---we just fold
    /// the single element. To explain this with lisp-like notation (with `arity` = 2):
    ///
    /// ```ignore
    /// [] --> ()
    /// [A] --> A
    /// [A, B] --> (A B)
    /// [A, B, C] --> ((A B) (C))
    /// ```
    pub fn new(len: usize, arity: usize, generator: &'a G) -> Self {
        // This is the tree depth needed to cover `len` items with nodes of `arity` children each.
        // We will gradually traverse down to depth 1, where the leaves are placed.
        let depth = tree_depth(len, arity);

        Self {
            total_len: len,
            current_depth: depth,
            current_start: 0,
            arity,
            generator,
            _leaf: std::marker::PhantomData,
        }
    }
}

impl<'a, L, G, F> Foldable<F> for IndexableSeqAsTree<'a, L, G>
where
    L: Foldable<F>,
    F: Fold,
    G: Fn(usize) -> L,
{
    fn fold(&self, builder: F) -> F::Folded {
        // For compatibility with the previous Merklisation scheme, we treat single-item sequences
        // as just a leaf.
        if self.total_len == 1 {
            return (self.generator)(self.current_start).fold(builder);
        }

        let mut builder = builder.into_node_fold();

        // Time to add leaves.
        if self.current_depth <= 1 {
            for idx in self.current_start..self.current_start + self.arity {
                if idx >= self.total_len {
                    break;
                }

                let item = (self.generator)(idx);
                builder.add(&item);
            }

            return builder.done();
        }

        let next_chunk_len = self.arity.pow(self.current_depth - 1);

        for child_no in 0..self.arity {
            let next_start = self.current_start + child_no * next_chunk_len;

            if next_start >= self.total_len {
                break;
            }

            builder.add(&IndexableSeqAsTree {
                total_len: self.total_len,
                current_depth: self.current_depth - 1,
                current_start: next_start,
                arity: self.arity,
                generator: self.generator,
                _leaf: std::marker::PhantomData,
            });
        }

        builder.done()
    }
}
/// Driver for folding a sequence as a tree in verify mode, skipping subtrees that hold no state.
///
/// [`IndexableSeqAsTree`] visits every leaf, which is what you want when the length is known to be
/// sound. In verify mode it is not: the length is recovered from the proof, and nothing bounds what
/// it may claim, so folding leaf by leaf would let a proof decide how much work the verifier does.
/// This driver asks `has_state` before descending, and folds a subtree with nothing underneath it
/// straight to the hash the reference proof already carries for it.
///
/// The two agree by construction. A subtree with nothing defined below it folds every leaf to
/// [`PartialHash::Previous`], and an all-`Previous` node reports whatever the proof said about it -
/// which is what [`PartialHashFold::skip_unchanged_subtree`] returns. Where that equivalence does
/// not hold, `skip_unchanged_subtree` declines and the descent happens as usual.
pub struct PrunedSeqAsTree<'a, Item, Generator, HasState> {
    /// Total length of the sequence (i.e. not just the number of items in the current chunk)
    total_len: usize,

    /// Depth of the current chunk
    current_depth: u32,

    /// Index where the current chunk starts
    current_start: usize,

    /// Maximum number of children per node
    arity: usize,

    /// Item generator (e.g. retrieval function for items)
    generator: &'a Generator,

    /// Reports whether any item in the given index range holds state
    has_state: &'a HasState,

    _item: std::marker::PhantomData<Item>,
}

impl<'a, Item, Generator, HasState> PrunedSeqAsTree<'a, Item, Generator, HasState> {
    /// Construct the driver for an indexable sequence whose items may be absent.
    ///
    /// `has_state` is given a range of item indices and reports whether any item in it is defined.
    /// It must not under-report: claiming a populated range is empty would fold that state away.
    pub fn new(
        len: usize,
        arity: usize,
        generator: &'a Generator,
        has_state: &'a HasState,
    ) -> Self {
        Self {
            total_len: len,
            current_depth: tree_depth(len, arity),
            current_start: 0,
            arity,
            generator,
            has_state,
            _item: std::marker::PhantomData,
        }
    }

    /// Range of item indices covered by the node currently being folded.
    fn covered_range(&self) -> Range<usize> {
        let width = self
            .arity
            .checked_pow(self.current_depth)
            .unwrap_or(usize::MAX);
        let end = self.current_start.saturating_add(width).min(self.total_len);

        self.current_start..end
    }
}

impl<'a, Codec, Item, Generator, HasState> Foldable<PartialHashFold<Codec>>
    for PrunedSeqAsTree<'a, Item, Generator, HasState>
where
    Codec: LeafCodec,
    Item: Foldable<PartialHashFold<Codec>>,
    Generator: Fn(usize) -> Item,
    HasState: Fn(Range<usize>) -> bool,
{
    fn fold(&self, builder: PartialHashFold<Codec>) -> PartialHash {
        // As in `IndexableSeqAsTree`, a single-item sequence is folded as a bare leaf.
        if self.total_len == 1 {
            return (self.generator)(self.current_start).fold(builder);
        }

        let mut builder = builder;

        let covered = self.covered_range();

        // An empty sequence folds to the hash of an empty node. That is a value in its own right
        // rather than something to defer to the reference proof for, so leave it to the descent -
        // and it costs nothing anyway, having no children to walk.
        if !covered.is_empty() && !(self.has_state)(covered) {
            match builder.skip_unchanged_subtree() {
                Ok(hash) => return hash,
                Err(unchanged) => builder = unchanged,
            }
        }

        let mut builder = builder.into_node_fold();

        // Time to add leaves.
        if self.current_depth <= 1 {
            for idx in self.current_start..self.current_start + self.arity {
                if idx >= self.total_len {
                    break;
                }

                let item = (self.generator)(idx);
                builder.add(&item);
            }

            return builder.done();
        }

        let next_chunk_len = self.arity.pow(self.current_depth - 1);

        for child_no in 0..self.arity {
            let next_start = self.current_start + child_no * next_chunk_len;

            if next_start >= self.total_len {
                break;
            }

            builder.add(&PrunedSeqAsTree {
                total_len: self.total_len,
                current_depth: self.current_depth - 1,
                current_start: next_start,
                arity: self.arity,
                generator: self.generator,
                has_state: self.has_state,
                _item: std::marker::PhantomData,
            });
        }

        builder.done()
    }
}

fn descend_helper<U, LeafHandler>(
    source: U,
    total_len: usize,
    current_depth: u32,
    current_start: usize,
    arity: usize,
    for_leaf: &mut LeafHandler,
) -> Result<(), UnfoldError>
where
    U: Unfold,
    LeafHandler: FnMut(usize, U) -> Result<(), UnfoldError>,
{
    if total_len == 1 {
        return for_leaf(current_start, source);
    }

    let mut source = source.into_node()?;

    if current_depth <= 1 {
        for idx in current_start..current_start + arity {
            if idx >= total_len {
                break;
            }

            source.next_branch_with(|ctx| for_leaf(idx, ctx))?;
        }

        return source.done(());
    }

    let next_chunk_len = arity.pow(current_depth - 1);

    for child_no in 0..arity {
        let next_start = current_start + child_no * next_chunk_len;

        if next_start >= total_len {
            break;
        }

        source.next_branch_with(|ctx| {
            descend_helper(
                ctx,
                total_len,
                current_depth - 1,
                next_start,
                arity,
                for_leaf,
            )
        })?
    }

    source.done(())
}

/// Use to unfold the components that use `IndexableSeqAsTree` for their fold implementations.
/// Traverses the tree structure calling `for_leaf` on each of the leaves. That function may do
/// further unfolding of the leaf if necessary, which is why it may be fallible.
///
/// To read about the tree structure used, see the documentation for `IndexableSeqAsTree`.
pub fn descend_tree<U, LeafHandler>(
    source: U,
    arity: usize,
    length: usize,
    for_leaf: &mut LeafHandler,
) -> Result<(), UnfoldError>
where
    U: Unfold,
    LeafHandler: FnMut(usize, U) -> Result<(), UnfoldError>,
{
    let depth = tree_depth(length, arity);

    descend_helper(source, length, depth, 0, arity, for_leaf)
}

/// Helper structure that can be used to adjust the depth of a proof tree for growable sequences
///
/// Use this when folding a `IndexableSeqAsTree` subtree that has changed in depth.
///
/// ```ignore
/// builder.add(&DepthAdjustedSeqAsTree {
///    inner: IndexableSeqAsTree::new(new_len, ...),
///    original_depth: old_depth,
///    current_depth: new_depth,
/// });
/// ```
pub struct DepthAdjustedSeqAsTree<T> {
    pub inner: T,
    pub original_depth: u32,
    pub current_depth: u32,
}

impl<C: LeafCodec, T: Foldable<PartialHashFold<C>>> Foldable<PartialHashFold<C>>
    for DepthAdjustedSeqAsTree<T>
{
    fn fold(&self, mut builder: PartialHashFold<C>) -> PartialHash {
        // If the original depth is larger than the current depth, then we need to scope the proof
        // that underlies the `PartialHash::Blinded` to not exceed that depth. We can do that by
        // picking the first child of any node until we reach the original depth - thereby
        // discarding the remaining child trees.
        if self.original_depth > self.current_depth {
            builder = builder.map_reference_proof(|mut proof| {
                for _ in self.current_depth..self.original_depth {
                    proof = match proof {
                        Tree::Node(mut node) => {
                            if node.children.is_empty() {
                                return None;
                            }

                            node.children.remove(0)
                        }

                        Tree::Leaf(_) => return None,
                    };
                }

                Some(proof)
            });
        }

        // If the original depth is smaller than the current depth, then we need to add dummy layers
        // onto the proof internal to the `PartialHashFold`.
        if self.original_depth < self.current_depth {
            builder = builder.map_reference_proof(|mut proof| {
                for _ in self.original_depth..self.current_depth {
                    proof = Tree::node_without_data(vec![proof]);
                }

                Some(proof)
            });
        }

        self.inner.fold(builder)
    }
}

/// Compute the depth of a Merkle tree that encodes a sequence of the given length with the given
/// arity.
pub fn tree_depth(length: usize, arity: usize) -> u32 {
    // `IndexableSeqAsTree` has a special-case layout where a single-element sequence is a
    // bare leaf and all other lengths are wrapped in at least one node. We encode that in
    // the adjusted depth by adding one level for all non-singleton lengths.
    length
        .saturating_sub(1)
        .checked_ilog(arity)
        .unwrap_or(0)
        .saturating_add(u32::from(length != 1))
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use proptest::prop_assert_eq;

    use crate::foldable::Foldable;
    use crate::foldable::Unfold;
    use crate::foldable::seq_tree::DepthAdjustedSeqAsTree;
    use crate::foldable::seq_tree::IndexableSeqAsTree;
    use crate::foldable::seq_tree::PrunedSeqAsTree;
    use crate::foldable::seq_tree::descend_tree;
    use crate::foldable::seq_tree::tree_depth;
    use crate::foldable::tests::TestFolder;
    use crate::foldable::tests::TestTree;
    use crate::hash::Hash;
    use crate::hash::PartialHash;
    use crate::merkle_proof::proof_tree::MerkleProof;
    use crate::serialisation::serialise;

    /// Build a Merkle tree with the given arity from the provided leaves.
    ///
    /// This function emulates the Merkle tree layout which was previously used for sequences.
    fn build_custom_merkle_tree(arity: usize, mut nodes: Vec<TestTree>) -> TestTree {
        assert!(!nodes.is_empty(), "Cannot build a tree with no leaves");

        let mut next_level = Vec::with_capacity(nodes.len().div_ceil(arity));

        while nodes.len() > 1 {
            for chunk in nodes.chunks(arity) {
                next_level.push(TestTree::Node(chunk.to_vec()));
            }

            std::mem::swap(&mut nodes, &mut next_level);
            next_level.truncate(0);
        }

        nodes.pop().unwrap_or_else(|| unreachable!())
    }

    fn generator(i: usize) -> TestTree {
        let bytes: Vec<u8> = serialise(i).unwrap();
        TestTree::Leaf(bytes)
    }

    /// This test ensures that the Merkle tree layout produced by [`IndexableSeqAsTree`] is
    /// consistent with the previous Merkle tree layout used for sequences.
    #[test]
    fn consistency_with_previous_merkle_tree_layout() {
        proptest::proptest!(|(arity in 2usize..=32, max_len in 1..=1024usize)| {
            let driver = IndexableSeqAsTree::new(max_len, arity, &generator);
            let tree = driver.fold(TestFolder);

            let custom_tree =
                build_custom_merkle_tree(arity, (0..max_len).map(generator).collect());

            assert_eq!(
                tree, custom_tree,
                "arity = {arity}, max_len = {max_len}\nleft = {tree:#?}\nright = {custom_tree:#?}"
            );
        });
    }

    /// Test length zero case.
    #[test]
    fn len_zero() {
        let driver = IndexableSeqAsTree::new(0, 2, &generator);
        let tree = driver.fold(TestFolder);

        assert_eq!(tree, TestTree::Node(vec![]));
    }

    /// Test length one case.
    #[test]
    fn len_one() {
        let driver = IndexableSeqAsTree::new(1, 2, &generator);
        let tree = driver.fold(TestFolder);

        assert_eq!(tree, generator(0));
    }

    /// Test that unfolding (using `descend_tree`) is the inverse to folding.
    #[test]
    fn fold_unfold() {
        proptest::proptest!(|(arity in 2usize..=32, max_len in 1..=1024usize)| {
            let driver = IndexableSeqAsTree::new(max_len, arity, &generator);
            let tree = driver.fold(TestFolder);

            let mut unfolded: Vec<Option<usize>> = vec![None; max_len];

            let mut for_leaf = |ix: usize, ctx: TestTree| {
                let value: usize = ctx.into_leaf()?;
                unfolded[ix] = Some(value);
                Ok(())
            };

            descend_tree(tree.clone(), arity, max_len, &mut for_leaf).unwrap();

            for (i, x) in unfolded.iter().enumerate() {
                assert_eq!(*x, Some(i));
            }
        });
    }

    /// Build a reference proof shaped like the sequence tree.
    ///
    /// `blinds` replaces a subtree with a blind, and `truncs` drops a node's last child so the
    /// proof carries fewer children than the state expects. Both are consumed in depth-first order,
    /// letting proptest explore proofs no honest prover would emit - which is the point, since a
    /// hostile proof is exactly what the verifier has to survive.
    fn build_reference_proof(
        total_len: usize,
        arity: usize,
        depth: u32,
        start: usize,
        blinds: &mut impl Iterator<Item = bool>,
        truncs: &mut impl Iterator<Item = bool>,
    ) -> MerkleProof {
        if total_len == 1 {
            return MerkleProof::leaf_read(vec![start as u8]);
        }

        if blinds.next().unwrap_or(false) {
            return MerkleProof::leaf_blind(Hash::hash_bytes(&[start as u8, depth as u8]));
        }

        let mut children = Vec::new();

        if depth <= 1 {
            for idx in start..start + arity {
                if idx >= total_len {
                    break;
                }

                children.push(MerkleProof::leaf_read(vec![idx as u8]));
            }
        } else {
            let chunk = arity.pow(depth - 1);

            for child_no in 0..arity {
                let next_start = start + child_no * chunk;
                if next_start >= total_len {
                    break;
                }

                children.push(build_reference_proof(
                    total_len,
                    arity,
                    depth - 1,
                    next_start,
                    blinds,
                    truncs,
                ));
            }
        }

        if truncs.next().unwrap_or(false) && children.len() > 1 {
            children.pop();
        }

        MerkleProof::node_without_data(children)
    }

    /// [`PrunedSeqAsTree`] decides the state hash, and acceptance is pinned by that hash, so it has
    /// to agree with a full descent on every input - not merely on the ones an honest prover emits.
    ///
    /// Among the shapes this covers is state set at a single index inside an otherwise undefined
    /// range: the path down to that index holds state at every level and is descended, while its
    /// siblings are folded from whatever the proof carries for them. It also covers a proof that
    /// blinds a subtree the state has written to, and one that supplies fewer children than the
    /// state expects - the latter has to come out `InvalidProof`, which is why the shortcut refuses
    /// to answer for a present node from its root hash alone.
    #[test]
    fn pruned_fold_agrees_with_full_descent() {
        proptest::proptest!(|(
            arity in 2usize..=4,
            len in 1usize..=30usize,
            defined in proptest::collection::vec(proptest::bool::ANY, 30),
            blinds in proptest::collection::vec(proptest::bool::ANY, 80),
            truncs in proptest::collection::vec(proptest::bool::ANY, 80),
        )| {
            let is_defined = |idx: usize| defined.get(idx).copied().unwrap_or(false);

            // Mirrors what the real generators produce: a defined item hashes to itself, an
            // undefined one defers to the proof exactly as `Partial::Absent` does.
            let generator = |idx: usize| {
                if is_defined(idx) {
                    PartialHash::Present(Hash::hash_bytes(&[idx as u8, 0xab]))
                } else {
                    PartialHash::Previous
                }
            };
            let has_state = |range: Range<usize>| range.into_iter().any(is_defined);

            let mut blind_iter = blinds.iter().copied();
            let mut trunc_iter = truncs.iter().copied();
            let proof = build_reference_proof(
                len,
                arity,
                tree_depth(len, arity),
                0,
                &mut blind_iter,
                &mut trunc_iter,
            );

            let full = PartialHash::from_foldable(
                Some(proof.clone()),
                &IndexableSeqAsTree::new(len, arity, &generator),
            );
            let pruned = PartialHash::from_foldable(
                Some(proof),
                &PrunedSeqAsTree::new(len, arity, &generator, &has_state),
            );

            prop_assert_eq!(full, pruned);
        });
    }

    /// Resizing across a power-of-arity boundary changes the tree's depth, and
    /// [`DepthAdjustedSeqAsTree`] re-scopes the reference proof to compensate - narrowing it by
    /// taking first children, or padding it with dummy layers. That is the one place where the proof
    /// and the state tree are deliberately not the same shape, so the shortcut has to agree with a
    /// full descent there too.
    ///
    /// Lengths are drawn at `arity^k` and one either side, so the adjustment is exercised in both
    /// directions, including the registry's case of a large claimed size resized by one across the
    /// boundary.
    #[test]
    fn pruned_fold_agrees_with_full_descent_across_a_depth_adjustment() {
        proptest::proptest!(|(
            arity in 2usize..=4,
            exp in 1u32..=4,
            orig_delta in -1i64..=1,
            len_delta in -1i64..=1,
            defined in proptest::collection::vec(proptest::bool::ANY, 300),
            blinds in proptest::collection::vec(proptest::bool::ANY, 200),
            truncs in proptest::collection::vec(proptest::bool::ANY, 200),
        )| {
            let boundary = arity.pow(exp) as i64;
            let original_len = (boundary + orig_delta).max(0) as usize;
            let len = (boundary + len_delta).max(0) as usize;

            let is_defined = |idx: usize| defined.get(idx).copied().unwrap_or(false);
            let generator = |idx: usize| {
                if is_defined(idx) {
                    PartialHash::Present(Hash::hash_bytes(&[idx as u8, 0xab]))
                } else {
                    PartialHash::Previous
                }
            };
            let has_state = |range: Range<usize>| range.into_iter().any(is_defined);

            // The proof describes the sequence as it was, so it is shaped at the original length.
            let proof_len = original_len.max(1);
            let mut blind_iter = blinds.iter().copied();
            let mut trunc_iter = truncs.iter().copied();
            let proof = build_reference_proof(
                proof_len,
                arity,
                tree_depth(proof_len, arity),
                0,
                &mut blind_iter,
                &mut trunc_iter,
            );

            let original_depth = tree_depth(original_len, arity);
            let current_depth = tree_depth(len, arity);

            let full = PartialHash::from_foldable(
                Some(proof.clone()),
                &DepthAdjustedSeqAsTree {
                    inner: IndexableSeqAsTree::new(len, arity, &generator),
                    original_depth,
                    current_depth,
                },
            );
            let pruned = PartialHash::from_foldable(
                Some(proof),
                &DepthAdjustedSeqAsTree {
                    inner: PrunedSeqAsTree::new(len, arity, &generator, &has_state),
                    original_depth,
                    current_depth,
                },
            );

            prop_assert_eq!(full, pruned);
        });
    }

    /// Short sequences, explicitly.
    ///
    /// An empty sequence folds to the hash of an empty node - a value in its own right rather than
    /// something to defer to the reference proof for - and a single-item one folds as a bare leaf.
    /// Skipping either would answer `Previous`, or a blind's hash, and both are wrong.
    ///
    /// Proptest found this case originally and its seeds are checked in, but those are RNG seeds
    /// rather than recorded inputs: they stop covering this the moment the strategy that consumed
    /// them changes, and nothing says so. Hence stating the case outright.
    #[test]
    fn pruned_fold_agrees_with_full_descent_on_short_sequences() {
        let generator = |_idx: usize| PartialHash::Previous;
        let has_state = |_range: Range<usize>| false;

        let proofs = [
            (
                "blinded",
                MerkleProof::leaf_blind(Hash::hash_bytes(b"blinded")),
            ),
            (
                "node",
                MerkleProof::node_without_data(vec![MerkleProof::leaf_read(vec![0u8])]),
            ),
            ("read leaf", MerkleProof::leaf_read(vec![1u8])),
        ];

        for arity in [2usize, 4, 32] {
            for len in [0usize, 1, 2] {
                for (name, proof) in &proofs {
                    let full = PartialHash::from_foldable(
                        Some(proof.clone()),
                        &IndexableSeqAsTree::new(len, arity, &generator),
                    );
                    let pruned = PartialHash::from_foldable(
                        Some(proof.clone()),
                        &PrunedSeqAsTree::new(len, arity, &generator, &has_state),
                    );

                    assert_eq!(full, pruned, "arity {arity}, len {len}, {name} proof");
                }
            }
        }
    }
}
