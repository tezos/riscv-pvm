// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;
use std::marker::PhantomData;

use bincode::Decode;

use super::deserialiser::Deserialiser;
use super::deserialiser::DeserialiserNode;
use super::deserialiser::Partial;
use super::deserialiser::ProofLayoutResult;
use super::deserialiser::Suspended;
use crate::state_backend::ProofLayoutError;
use crate::state_backend::ProofPart;
use crate::state_backend::ProofTree;
use crate::state_backend::proof_backend::proof::MerkleProofLeaf;
use crate::state_backend::proof_backend::proof::deserialiser;
use crate::state_backend::proof_backend::proof::deserialiser::ProofParseResult;
use crate::state_backend::proof_backend::proof::deserialiser::RunDeserialiser;
use crate::state_backend::proof_backend::tree::Tree;
use crate::storage::binary;

/// Deserialiser for [`Deserialiser`] which owns the data.
pub struct ProofTreeDeserialiser<'t>(ProofTree<'t>);

impl<'t> Deserialiser for ProofTreeDeserialiser<'t> {
    type Suspended<R> = OwnedParserComb<'t, R>;

    type DeserialiserNode<R> = OwnedBranchComb<R, Self>;

    fn into_leaf_raw<const LEN: usize>(
        self,
    ) -> ProofLayoutResult<Self::Suspended<Partial<Box<[u8; LEN]>>>> {
        self.deserialise_as_leaf()?
            .map_present_fallible(|data| {
                let data_len = data.len();
                let bytes: Box<[u8; LEN]> =
                    data.try_into()
                        .map_err(|_| ProofLayoutError::UnexpectedLeafSize {
                            expected: LEN,
                            got: data_len,
                        })?;
                Ok(bytes)
            })
            .map(|data| OwnedParserComb::new(Ok(data)))
    }

    fn into_leaf<T: Decode<()>>(self) -> ProofLayoutResult<Self::Suspended<Partial<(T, Vec<u8>)>>> {
        let leaf_data = self
            .deserialise_as_leaf()?
            .map_present_fallible(|data| Ok((binary::deserialise::<T>(data.as_ref())?, data)));

        Ok(OwnedParserComb::new(leaf_data))
    }

    fn into_node(self) -> ProofLayoutResult<Self::DeserialiserNode<Partial<()>>> {
        let branches = self.deserialise_as_node()?;
        Ok(OwnedBranchComb::new(branches))
    }
}

impl<'t> From<ProofTree<'t>> for ProofTreeDeserialiser<'t> {
    fn from(proof: ProofTree<'t>) -> Self {
        ProofTreeDeserialiser(proof)
    }
}

impl ProofTreeDeserialiser<'_> {
    /// Deserialise the proof as a leaf.
    pub fn deserialise_as_leaf(self) -> ProofLayoutResult<Partial<Vec<u8>>> {
        match self.0 {
            ProofPart::Absent => Ok(Partial::Absent),
            ProofPart::Present(Tree::Node(_)) => Err(ProofLayoutError::UnexpectedNode),
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Blind(hash))) => {
                Ok(Partial::Blinded(*hash))
            }
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Read(items))) => {
                Ok(Partial::Present(items.clone()))
            }
        }
    }

    /// Deserialise the proof as a node.
    pub fn deserialise_as_node(self) -> ProofLayoutResult<Partial<Vec<Self>>> {
        match self.0 {
            ProofPart::Absent => Ok(Partial::Absent),
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Blind(hash))) => {
                Ok(Partial::Blinded(*hash))
            }
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Read(_))) => {
                Err(ProofLayoutError::UnexpectedLeaf)
            }
            ProofPart::Present(Tree::Node(trees)) => Ok(Partial::Present(
                trees
                    .iter()
                    .map(ProofPart::Present)
                    .map(ProofTreeDeserialiser)
                    .collect(),
            )),
        }
    }
}

/// Suspended computation combinator for [`ProofTreeDeserialiser`] deserialiser.
pub struct OwnedParserComb<'t, R> {
    result: ProofParseResult<R>,
    _pd: PhantomData<fn(ProofTreeDeserialiser<'t>)>,
}

impl<R> OwnedParserComb<'_, R> {
    fn new(result: ProofParseResult<R>) -> Self {
        Self {
            result,
            _pd: PhantomData,
        }
    }
}

/// Branch deserialiser combinator for [`ProofTreeDeserialiser`] deserialiser.
pub struct OwnedBranchComb<R, B> {
    f: ProofParseResult<R>,
    node_data: Partial<VecDeque<B>>,
}

impl<B> OwnedBranchComb<Partial<()>, B> {
    /// Create a new [`OwnedBranchComb`] with the given branches,
    /// preserving the absent/blind/present information from the given [`Partial`].
    fn new(branches: Partial<Vec<B>>) -> Self {
        // Similar to `map_present` but for `&Partial<R>`.
        // This is done to preserve absent/blind/present information from node until calling `done()`.
        // See test_blind_node_parsing for an example.
        let f_comb = match &branches {
            Partial::Absent => Partial::Absent,
            Partial::Blinded(hash) => Partial::Blinded(*hash),
            Partial::Present(_) => Partial::Present(()),
        };

        Self {
            f: Ok(f_comb),
            node_data: branches.map_present(VecDeque::from),
        }
    }
}

impl<'t, R> DeserialiserNode<R> for OwnedBranchComb<R, ProofTreeDeserialiser<'t>> {
    type Parent = ProofTreeDeserialiser<'t>;

    fn next_branch<T>(
        mut self,
        branch_deserialiser: impl FnOnce(
            Self::Parent,
        ) -> ProofLayoutResult<
            <Self::Parent as Deserialiser>::Suspended<T>,
        >,
    ) -> ProofLayoutResult<<Self::Parent as Deserialiser>::DeserialiserNode<(R, T)>>
    where
        T: 'static,
        R: 'static,
    {
        let next_branch = match self.node_data {
            // If the node is absent or blinded, the branch to be deserialised as a tree is absent.
            Partial::Absent | Partial::Blinded(_) => ProofTreeDeserialiser(ProofTree::Absent),
            Partial::Present(ref mut branches) => {
                branches
                    .pop_front()
                    .ok_or(ProofLayoutError::BadNumberOfBranches {
                        expected: 1,
                        got: 0,
                    })?
            }
        };
        let br_comb = branch_deserialiser(next_branch)?;

        Ok(OwnedBranchComb {
            f: self.f.and_then(|res| Ok((res, br_comb.result?))),
            node_data: self.node_data,
        })
    }

    fn map<T>(
        self,
        f: impl FnOnce(R) -> T + 'static,
    ) -> <Self::Parent as Deserialiser>::DeserialiserNode<T>
    where
        T: 'static,
        R: 'static,
    {
        OwnedBranchComb {
            f: self.f.map(f),
            node_data: self.node_data,
        }
    }

    fn done(self) -> ProofLayoutResult<<Self::Parent as Deserialiser>::Suspended<R>> {
        if let Partial::Present(branches) = self.node_data {
            if !branches.is_empty() {
                let length = branches.len();
                return Err(ProofLayoutError::BadNumberOfBranches {
                    expected: 0,
                    got: length,
                });
            }
        }

        Ok(OwnedParserComb {
            result: self.f,
            _pd: PhantomData,
        })
    }
}

impl<'t, R> Suspended for OwnedParserComb<'t, R> {
    type Output = R;

    type Parent = ProofTreeDeserialiser<'t>;

    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T + 'static,
    ) -> <Self::Parent as Deserialiser>::Suspended<T>
    where
        Self::Output: 'static,
    {
        OwnedParserComb::new(self.result.map(f))
    }
}

impl<R> OwnedParserComb<'_, R> {
    pub fn into_result(self) -> ProofParseResult<R> {
        self.result
    }
}

impl<'t> RunDeserialiser for ProofTreeDeserialiser<'t> {
    type Data = ProofTree<'t>;

    fn run<R>(
        input_data: Self::Data,
        deser_fn: impl FnOnce(Self) -> ProofLayoutResult<<Self as Deserialiser>::Suspended<R>>,
    ) -> deserialiser::Result<R> {
        let proof_tree = ProofTreeDeserialiser(input_data);
        deser_fn(proof_tree)?
            .into_result()
            .map_err(deserialiser::Error::ParseProof)
    }
}
