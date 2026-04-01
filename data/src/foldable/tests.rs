// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

#![cfg(test)]

use bincode::Decode;

use crate::foldable::Fold;
use crate::foldable::FoldLeaf;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::foldable::Unfoldable;
use crate::serialisation::deserialise;

impl<A: Unfoldable, B: Unfoldable> Unfoldable for (A, B) {
    fn unfold<U: Unfold>(source: U) -> Result<Self, UnfoldError> {
        let mut u = source.into_node()?;

        let a = u.next_branch()?;
        let b = u.next_branch()?;

        u.done((a, b))
    }
}

impl<A: Unfoldable, B: Unfoldable, C: Unfoldable> Unfoldable for (A, B, C) {
    fn unfold<U: Unfold>(source: U) -> Result<Self, UnfoldError> {
        let mut u = source.into_node()?;

        let a = u.next_branch()?;
        let b = u.next_branch()?;
        let c = u.next_branch()?;

        u.done((a, b, c))
    }
}

/// Simple tree data type for testing purposes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestTree {
    Leaf(Vec<u8>),
    Node(Vec<Self>),
}

impl Foldable<TestFolder> for u8 {
    fn fold(&self, _builder: TestFolder) -> TestTree {
        TestTree::Leaf(vec![*self])
    }
}

impl Foldable<TestFolder> for TestTree {
    fn fold(&self, _builder: TestFolder) -> TestTree {
        self.clone()
    }
}

/// Folder for [`TestTree`]
pub struct TestFolder;

impl Fold for TestFolder {
    type Folded = TestTree;

    type NodeFold = TestNodeFolder;

    fn into_node_fold(self) -> Self::NodeFold {
        TestNodeFolder {
            children: Vec::new(),
        }
    }
}

impl FoldLeaf for TestFolder {
    fn fold_leaf_raw(self, bytes: &[u8]) -> TestTree {
        let mut v = vec![];
        v.extend_from_slice(bytes);
        TestTree::Leaf(v)
    }
}

/// Node folder for [`TestTree`]
pub struct TestNodeFolder {
    children: Vec<TestTree>,
}

impl NodeFold for TestNodeFolder {
    type Parent = TestFolder;

    fn add_labelled<F: Foldable<Self::Parent>>(&mut self, child: &F, _label: Option<&str>) {
        let folded_child = child.fold(TestFolder);
        self.children.push(folded_child);
    }

    fn done(self) -> TestTree {
        TestTree::Node(self.children)
    }
}

impl Unfold for TestTree {
    type NodeUnfold = Vec<TestTree>;

    fn into_node(self) -> Result<Vec<TestTree>, UnfoldError> {
        match self {
            TestTree::Leaf(_) => Err(UnfoldError::UnexpectedLeaf),
            TestTree::Node(children) => Ok(children),
        }
    }

    fn into_leaf_raw<const LEN: usize>(self) -> Result<Box<[u8; LEN]>, UnfoldError> {
        match self {
            TestTree::Leaf(bytes) => {
                if bytes.len() == LEN {
                    let mut arr: Box<[u8; LEN]> = vec![0; LEN].try_into().unwrap();
                    arr.copy_from_slice(&bytes[0..LEN]);
                    Ok(arr)
                } else {
                    Err(UnfoldError::UnexpectedLeafSize {
                        expected: LEN,
                        got: bytes.len(),
                    })
                }
            }
            TestTree::Node(_) => Err(UnfoldError::UnexpectedNode),
        }
    }

    fn into_leaf<T: Decode<()>>(self) -> Result<T, UnfoldError> {
        match self {
            TestTree::Leaf(bytes) => {
                let t = deserialise(&bytes[..])?;
                Ok(t)
            }
            TestTree::Node(_) => Err(UnfoldError::UnexpectedNode),
        }
    }
}

impl NodeUnfold for Vec<TestTree> {
    type Parent = TestTree;

    fn next_branch_with<T>(
        &mut self,
        unfolder: impl FnOnce(TestTree) -> Result<T, UnfoldError>,
    ) -> Result<T, UnfoldError> {
        if self.is_empty() {
            Err(UnfoldError::TooFewChildren)
        } else {
            let tree = self.remove(0);
            unfolder(tree)
        }
    }

    fn done<T>(self, t: T) -> Result<T, UnfoldError> {
        if self.is_empty() {
            Ok(t)
        } else {
            Err(UnfoldError::TooManyChildren(self.len()))
        }
    }
}

impl Unfoldable for u8 {
    fn unfold<U: Unfold>(source: U) -> Result<u8, UnfoldError> {
        source.into_leaf()
    }
}

#[test]
fn test_unfold() {
    type Data = (u8, (u8, u8, (u8, u8, u8)), (u8, u8));

    // Fold and unfold data of a given shape
    let data: Data = (1, (2, 3, (4, 5, 6)), (7, 8));
    let tree = data.fold(TestFolder);
    let unfolded = Data::unfold(tree).unwrap();
    assert_eq!(data, unfolded);

    // Incorrect shape: too many children
    let bad_data = (1, (2, 3, (4, 5, 6)), (7, 8, 9));
    let tree = bad_data.fold(TestFolder);
    let result = Data::unfold(tree);
    assert!(matches!(
        result.unwrap_err(),
        UnfoldError::TooManyChildren(1)
    ));

    // Incorrect shape: too few children
    let bad_data = (1, (2, 3, (4, 6)), (7, 8));
    let tree = bad_data.fold(TestFolder);
    let result = Data::unfold(tree);
    assert!(matches!(result.unwrap_err(), UnfoldError::TooFewChildren));

    // Incorrect shape: unexpected leaf
    let bad_data = (1, (2, 3, 4), (7, 8));
    let tree = bad_data.fold(TestFolder);
    let result = Data::unfold(tree);
    assert!(matches!(result.unwrap_err(), UnfoldError::UnexpectedLeaf));

    // Incorrect shape: unexpected node
    let bad_data = (1, ((2, 2), 3, (4, 5, 6)), (7, 8));
    let tree = bad_data.fold(TestFolder);
    let result = Data::unfold(tree);
    assert!(matches!(result.unwrap_err(), UnfoldError::UnexpectedNode));
}
