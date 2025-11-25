// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Helpers for sequences that need to fold like trees

use std::ops::Range;

use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::utils::next_power_of;

/// Driver for `Foldable` that lets you turn an indexable sequence into a tree-like structure where
/// the leaves are the items of the sequence
pub struct IndexableSeqAsTree<'a, L, G> {
    /// Range of indices for which the sequence has defined items
    defined_range: Range<usize>,

    /// Range of indices being currently processed
    range: Range<usize>,

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
    pub fn new(len: usize, arity: usize, generator: &'a G) -> Self {
        Self {
            defined_range: 0..len,
            // Extending the range to the next power of `arity` gives us a simple way to make sure
            // the leaves will always exist at the same depth. [`IndexableSeqAsTree::fold`] only
            // needs to divide the range by `arity` to get chunk sizes which will eventually be cut
            // down to the right size at the same recursion depth.
            range: 0..next_power_of(len, arity),
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
        if self.defined_range.len() == 1 {
            return (self.generator)(self.defined_range.start).fold(builder);
        }

        let mut builder = builder.into_node_fold();

        // When the range fits into the arity, we have reached the deepest level for this portion
        // of the sequence.
        if self.range.len() <= self.arity {
            for idx in self.range.clone() {
                if !self.defined_range.contains(&idx) {
                    continue;
                }

                let item = (self.generator)(idx);
                builder.add(&item);
            }

            return builder.done();
        }

        let chunk_len = self.range.len().div_ceil(self.arity);
        let chunk_starts = self
            .range
            .clone()
            .step_by(chunk_len)
            .take_while(|&start| start < self.defined_range.end);

        for start in chunk_starts {
            let end = start + chunk_len;

            builder.add(&IndexableSeqAsTree {
                defined_range: self.defined_range.clone(),
                range: start..end,
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
        if nodes.is_empty() {
            panic!()
        }

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
}
