// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Helpers for sequences that need to fold and unfold like trees


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
    use crate::foldable::Foldable;
    use crate::foldable::Unfold;
    use crate::foldable::seq_tree::IndexableSeqAsTree;
    use crate::foldable::seq_tree::descend_tree;
    use crate::foldable::tests::TestFolder;
    use crate::foldable::tests::TestTree;
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
}
