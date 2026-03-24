// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

#![cfg(test)]

use std::convert::Infallible;
use std::error;

use bincode::Decode;
use unwrap_infallible::UnwrapInfallible;

use crate::foldable::Fold;
use crate::foldable::FoldResult;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::foldable::NodeUnfold;
use crate::foldable::Unfold;
use crate::foldable::UnfoldError;
use crate::foldable::Unfoldable;
use crate::serialisation::deserialise;

impl<A: Unfoldable, B: Unfoldable> Unfoldable for (A, B) {
    fn unfold<U: Unfold>(source: U) -> Result<Self, U::Error> {
        let mut u = source.into_node()?;

        let a = u.next_branch()?;
        let b = u.next_branch()?;

        u.done((a, b))
    }
}

impl<A: Unfoldable, B: Unfoldable, C: Unfoldable> Unfoldable for (A, B, C) {
    fn unfold<U: Unfold>(source: U) -> Result<Self, U::Error> {
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
    fn fold(&self, _builder: TestFolder) -> FoldResult<TestFolder> {
        Ok(TestTree::Leaf(vec![*self]))
    }
}

impl Foldable<TestFolder> for TestTree {
    fn fold(&self, _builder: TestFolder) -> FoldResult<TestFolder> {
        Ok(self.clone())
    }
}

/// Folder for [`TestTree`]
pub struct TestFolder;

impl Fold for TestFolder {
    type Error = Infallible;

    type Folded = TestTree;

    type NodeFold = TestNodeFolder;

    fn into_node_fold(self) -> Self::NodeFold {
        TestNodeFolder {
            children: Vec::new(),
        }
    }
}

/// Node folder for [`TestTree`]
pub struct TestNodeFolder {
    children: Vec<TestTree>,
}

impl NodeFold for TestNodeFolder {
    type Parent = TestFolder;

    fn add<F: Foldable<Self::Parent>>(&mut self, child: &F) -> Result<(), Infallible> {
        let folded_child = child.fold(TestFolder)?;
        self.children.push(folded_child);
        Ok(())
    }

    fn done(self) -> FoldResult<TestFolder> {
        Ok(TestTree::Node(self.children))
    }
}

impl UnfoldError for String {
    fn custom<E: error::Error>(error: E) -> String {
        format!("{error:?}")
    }
}

impl Unfold for TestTree {
    type NodeUnfold = Vec<TestTree>;

    type Error = String;

    fn into_node(self) -> Result<Vec<TestTree>, String> {
        match self {
            TestTree::Leaf(_) => Err("Unexpected leaf".to_string()),
            TestTree::Node(children) => Ok(children),
        }
    }

    fn into_leaf_raw<const LEN: usize>(self) -> Result<Box<[u8; LEN]>, String> {
        match self {
            TestTree::Leaf(bytes) => {
                if bytes.len() == LEN {
                    let mut arr: Box<[u8; LEN]> = vec![0; LEN].try_into().unwrap();
                    arr.copy_from_slice(&bytes[0..LEN]);
                    Ok(arr)
                } else {
                    Err("Incorrect length in leaf".to_string())
                }
            }
            TestTree::Node(_) => Err("Unexpected node".to_string()),
        }
    }

    fn into_leaf<T: Decode<()>>(self) -> Result<T, String> {
        match self {
            TestTree::Leaf(bytes) => {
                deserialise(&bytes[..]).map_err(|_| "Deserialisation error".to_string())
            }
            TestTree::Node(_) => Err("Unexpected node".to_string()),
        }
    }
}

impl NodeUnfold for Vec<TestTree> {
    type Parent = TestTree;

    fn next_branch_with<T>(
        &mut self,
        unfolder: impl FnOnce(TestTree) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.is_empty() {
            Err("Too few children".to_string())
        } else {
            let tree = self.remove(0);
            unfolder(tree)
        }
    }

    fn done<T>(self, t: T) -> Result<T, String> {
        if self.is_empty() {
            Ok(t)
        } else {
            Err("Too many children".to_string())
        }
    }
}

impl Unfoldable for u8 {
    fn unfold<U: Unfold>(source: U) -> Result<u8, U::Error> {
        source.into_leaf()
    }
}

#[test]
fn test_unfold() {
    type Data = (u8, (u8, u8, (u8, u8, u8)), (u8, u8));

    // Fold and unfold data of a given shape
    let data: Data = (1, (2, 3, (4, 5, 6)), (7, 8));
    let tree = data.fold(TestFolder).unwrap_infallible();
    let unfolded = Data::unfold(tree).unwrap();
    assert_eq!(data, unfolded);

    // Incorrect shape: too many children
    let bad_data = (1, (2, 3, (4, 5, 6)), (7, 8, 9));
    let tree = bad_data.fold(TestFolder).unwrap_infallible();
    let result = Data::unfold(tree);
    assert_eq!(result.unwrap_err().as_str(), "Too many children");

    // Incorrect shape: too few children
    let bad_data = (1, (2, 3, (4, 6)), (7, 8));
    let tree = bad_data.fold(TestFolder).unwrap_infallible();
    let result = Data::unfold(tree);
    assert_eq!(result.unwrap_err().as_str(), "Too few children");

    // Incorrect shape: unexpected leaf
    let bad_data = (1, (2, 3, 4), (7, 8));
    let tree = bad_data.fold(TestFolder).unwrap_infallible();
    let result = Data::unfold(tree);
    assert_eq!(result.unwrap_err().as_str(), "Unexpected leaf");

    // Incorrect shape: unexpected node
    let bad_data = (1, ((2, 2), 3, (4, 5, 6)), (7, 8));
    let tree = bad_data.fold(TestFolder).unwrap_infallible();
    let result = Data::unfold(tree);
    assert_eq!(result.unwrap_err().as_str(), "Unexpected node");
}
