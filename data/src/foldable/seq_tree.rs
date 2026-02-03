// SPDX-FileCopyrightText: 2025-2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Helpers for sequences that need to fold like trees

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;

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
        // We will gradually traverse down to depth 0, where the leaves are placed.
        let depth = len.saturating_sub(1).checked_ilog(arity).unwrap_or(0);

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
        if self.current_depth == 0 {
            for idx in self.current_start..self.current_start + self.arity {
                if idx >= self.total_len {
                    break;
                }

                let item = (self.generator)(idx);
                builder.add(&item);
            }

            return builder.done();
        }

        let next_chunk_len = self.arity.pow(self.current_depth);

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

#[cfg(test)]
mod tests {
    use crate::foldable::Fold;
    use crate::foldable::Foldable;
    use crate::foldable::NodeFold;
    use crate::foldable::seq_tree::IndexableSeqAsTree;

    /// Simple tree data type for testing purposes
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TestTree {
        Leaf(usize),
        Node(Vec<Self>),
    }

    impl Foldable<TestFolder> for TestTree {
        fn fold(&self, _builder: TestFolder) -> TestTree {
            self.clone()
        }
    }

    /// Folder for [`TestTree`]
    struct TestFolder;

    impl Fold for TestFolder {
        type Folded = TestTree;

        type NodeFold = TestNodeFolder;

        fn into_node_fold(self) -> Self::NodeFold {
            TestNodeFolder {
                children: Vec::new(),
            }
        }
    }

    /// Node folder for [`TestTree`]
    struct TestNodeFolder {
        children: Vec<TestTree>,
    }

    impl NodeFold for TestNodeFolder {
        type Parent = TestFolder;

        fn add<F: Foldable<Self::Parent>>(&mut self, child: &F) {
            let folded_child = child.fold(TestFolder);
            self.children.push(folded_child);
        }

        fn done(self) -> TestTree {
            TestTree::Node(self.children)
        }
    }

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

    /// This test ensures that the Merkle tree layout produced by [`IndexableSeqAsTree`] is
    /// consistent with the previous Merkle tree layout used for sequences.
    #[test]
    fn consistency_with_previous_merkle_tree_layout() {
        proptest::proptest!(|(arity in 2usize..=32, max_len in 1..=1024usize)| {
            let driver = IndexableSeqAsTree::new(max_len, arity, &|i| TestTree::Leaf(i));
            let tree = driver.fold(TestFolder);

            let custom_tree =
                build_custom_merkle_tree(arity, (0..max_len).map(TestTree::Leaf).collect());

            assert_eq!(
                tree, custom_tree,
                "arity = {arity}, max_len = {max_len}\nleft = {tree:#?}\nright = {custom_tree:#?}"
            );
        });
    }

    /// Test length zero case.
    #[test]
    fn len_zero() {
        let driver = IndexableSeqAsTree::new(0, 2, &|i| TestTree::Leaf(i));
        let tree = driver.fold(TestFolder);

        assert_eq!(tree, TestTree::Node(vec![]));
    }

    /// Test length one case.
    #[test]
    fn len_one() {
        let driver = IndexableSeqAsTree::new(1, 2, &|i| TestTree::Leaf(i));
        let tree = driver.fold(TestFolder);

        assert_eq!(tree, TestTree::Leaf(0));
    }
}
