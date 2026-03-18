// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Helpers for sequences that need to fold and unfold like trees

use derive_more::From;

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::Unfoldable;

/// Driver for `Foldable` that lets you turn an indexable sequence into a tree-like structure where
/// the leaves are the items of the sequence
pub struct IndexableSeqAsTree<'a, L, G> {
    /// Total length of the sequence (i.e. not just the number of items in the current chunk)
    total_len: usize,

    /// Number of items in the current chunk
    current_len: usize,

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
        Self {
            total_len: len,
            current_len: len,
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

        if self.current_len <= self.arity {
            for idx in self.current_start..self.current_start + self.current_len {
                let item = (self.generator)(idx);
                builder.add(&item);
            }

            return builder.done();
        }

        let next_chunk_len = self.arity.pow(
            self.current_len
                .saturating_sub(1)
                .checked_ilog(self.arity)
                .unwrap_or(0),
        );
        let chunks = self.current_len.div_ceil(next_chunk_len);

        for chunk_idx in 0..chunks {
            let chunk_offset = chunk_idx * next_chunk_len;
            let next_start = self.current_start + chunk_offset;
            let next_len = next_chunk_len.min(self.current_len - chunk_offset);

            builder.add(&IndexableSeqAsTree {
                total_len: self.total_len,
                current_len: next_len,
                current_start: next_start,
                arity: self.arity,
                generator: self.generator,
                _leaf: std::marker::PhantomData,
            });
        }

        builder.done()
    }
}

/// Helper struct for unfolding sequences in a state component.
#[derive(From)]
pub struct Many<T, const ARITY: usize, const LEN: usize>(Box<[T; LEN]>);

impl<T, const ARITY: usize, const LEN: usize> Many<T, ARITY, LEN> {
    /// Turn this into the underlying boxed array.
    pub fn into_boxed_array(self) -> Box<[T; LEN]> {
        self.0
    }
}

impl<T: Unfoldable, const ARITY: usize, const LEN: usize> Unfoldable for Many<T, ARITY, LEN> {
    fn unfold<U: Unfold>(source: U) -> Result<Self, U::Error> {
        let mut leaves = Vec::with_capacity(LEN);

        descend_tree(source, ARITY, LEN, &mut |_, source| {
            let leaf = T::unfold(source)?;
            leaves.push(leaf);
            Ok(())
        })?;

        let Ok(boxed_array): Result<Box<[T; LEN]>, _> = leaves.into_boxed_slice().try_into() else {
            unreachable!("Unexpected number of leaves collected")
        };

        Ok(Many::from(boxed_array))
    }
}

fn descend_helper<U, LeafHandler>(
    source: U,
    total_len: usize,
    current_start: usize,
    current_len: usize,
    arity: usize,
    for_leaf: &mut LeafHandler,
) -> Result<(), U::Error>
where
    U: Unfold,
    LeafHandler: FnMut(usize, U) -> Result<(), U::Error>,
{
    if total_len == 1 {
        return for_leaf(current_start, source);
    }

    let mut source = source.into_node()?;

    if current_len <= arity {
        for idx in current_start..current_start + current_len {
            source.next_branch_with(|ctx| for_leaf(idx, ctx))?;
        }

        return source.done(());
    }

    let next_chunk_len = arity.pow(
        current_len
            .saturating_sub(1)
            .checked_ilog(arity)
            .unwrap_or(0),
    );
    let chunks = current_len.div_ceil(next_chunk_len);

    for chunk_idx in 0..chunks {
        let chunk_offset = chunk_idx * next_chunk_len;
        let next_start = current_start + chunk_offset;
        let next_len = next_chunk_len.min(current_len - chunk_offset);

        source.next_branch_with(|ctx| {
            descend_helper(ctx, total_len, next_start, next_len, arity, for_leaf)
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
) -> Result<(), U::Error>
where
    U: Unfold,
    LeafHandler: FnMut(usize, U) -> Result<(), U::Error>,
{
    descend_helper(source, length, 0, length, arity, for_leaf)
}

#[cfg(test)]
mod tests {
    use crate::foldable::Foldable;
    use crate::foldable::Unfold;
    use crate::foldable::Unfoldable;
    use crate::foldable::seq_tree::IndexableSeqAsTree;
    use crate::foldable::seq_tree::Many;
    use crate::foldable::seq_tree::descend_tree;
    use crate::foldable::tests::TestFolder;
    use crate::foldable::tests::TestTree;
    use crate::serialisation::serialise;

    fn generator(i: usize) -> TestTree {
        let bytes: Vec<u8> = serialise(i).unwrap();
        TestTree::Leaf(bytes)
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

    /// Test the `Many` utility struct for unfolding.
    #[test]
    fn many_unfold() {
        type Data = Many<((u8, u8), u8, u8), 5, 71>;

        let mut data = Vec::with_capacity(71);
        for i in 0u8..71 {
            data.push(((i, i + 2), 200 - i, 2 * i));
        }

        let generator = |i: usize| data[i].fold(TestFolder);

        let driver = IndexableSeqAsTree::new(71, 5, &generator);
        let tree = driver.fold(TestFolder);

        let unfolded = Data::unfold(tree.clone()).unwrap();
        assert_eq!(&unfolded.into_boxed_array()[..], &data);
    }
}
