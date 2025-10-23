// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;
use std::marker::PhantomData;

use bincode::Decode;

use super::deserialiser::Deserialiser;
use super::deserialiser::DeserialiserNode;
use super::deserialiser::Partial;
use super::deserialiser::Result;
use super::deserialiser::Suspended;
use crate::state_backend::AllocatedOf;
use crate::state_backend::OwnedProofPart;
use crate::state_backend::ProofError;
use crate::state_backend::ProofLayout;
use crate::state_backend::ProofPart;
use crate::state_backend::ProofTree;
use crate::state_backend::proof_backend::proof::MerkleProofLeaf;
use crate::state_backend::proof_backend::proof::deserialiser;
use crate::state_backend::proof_backend::tree::Tree;
use crate::state_backend::verify_backend::Verifier;
use crate::storage::binary;

/// Deserialiser for [`Deserialiser`] which owns the data.
pub struct ProofTreeDeserialiser<'t>(ProofTree<'t>);

impl<'t> Deserialiser for ProofTreeDeserialiser<'t> {
    type Suspended<R> = OwnedParserComb<'t, R>;

    type DeserialiserNode = OwnedBranchComb<Self>;

    fn into_leaf_raw<const LEN: usize>(self) -> Result<Self::Suspended<Partial<Box<[u8; LEN]>>>> {
        self.deserialise_as_leaf()?
            .map_present_fallible(|data| {
                let data_len = data.len();
                let bytes: Box<[u8; LEN]> =
                    data.try_into()
                        .map_err(|_| ProofError::UnexpectedLeafSize {
                            expected: LEN,
                            got: data_len,
                        })?;
                Ok(bytes)
            })
            .map(OwnedParserComb::new)
    }

    fn into_leaf<T: Decode<()>>(self) -> Result<Self::Suspended<Partial<T>>> {
        let result = self
            .deserialise_as_leaf()?
            .map_present_fallible(|data| binary::deserialise::<T>(data.as_ref()))?;
        Ok(OwnedParserComb::new(result))
    }

    fn into_node(self) -> Result<Self::DeserialiserNode> {
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
    pub fn deserialise_as_leaf(self) -> Result<Partial<Vec<u8>>> {
        match self.0 {
            ProofPart::Absent => Ok(Partial::Absent),
            ProofPart::Present(Tree::Node(_)) => Err(ProofError::UnexpectedNode),
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Blind(hash))) => {
                Ok(Partial::Blinded(*hash))
            }
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Read(items))) => {
                Ok(Partial::Present(items.clone()))
            }
        }
    }

    /// Deserialise the proof as a node.
    pub fn deserialise_as_node(self) -> Result<Partial<Vec<Self>>> {
        match self.0 {
            ProofPart::Absent => Ok(Partial::Absent),
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Blind(hash))) => {
                Ok(Partial::Blinded(*hash))
            }
            ProofPart::Present(Tree::Leaf(MerkleProofLeaf::Read(_))) => {
                Err(ProofError::UnexpectedLeaf)
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
    result: R,
    _pd: PhantomData<fn(ProofTreeDeserialiser<'t>)>,
}

impl<R> OwnedParserComb<'_, R> {
    fn new(result: R) -> Self {
        Self {
            result,
            _pd: PhantomData,
        }
    }

    /// Consume the combinator and return its result.
    pub fn into_result(self) -> R {
        self.result
    }
}

/// Branch deserialiser combinator for [`ProofTreeDeserialiser`] deserialiser.
pub struct OwnedBranchComb<B> {
    node_data: Partial<VecDeque<B>>,
}

impl<B> OwnedBranchComb<B> {
    /// Create a new [`OwnedBranchComb`] with the given branches,
    /// preserving the absent/blind/present information from the given [`Partial`].
    fn new(branches: Partial<Vec<B>>) -> Self {
        Self {
            node_data: branches.map_present(VecDeque::from),
        }
    }
}

impl<'t> DeserialiserNode for OwnedBranchComb<ProofTreeDeserialiser<'t>> {
    type Parent = ProofTreeDeserialiser<'t>;

    fn presence(&self) -> Partial<()> {
        match &self.node_data {
            Partial::Absent => Partial::Absent,
            Partial::Blinded(hash) => Partial::Blinded(*hash),
            Partial::Present(_) => Partial::Present(()),
        }
    }

    fn next_branch<T>(
        mut self,
        branch_deserialiser: impl FnOnce(
            Self::Parent,
        )
            -> Result<<Self::Parent as Deserialiser>::Suspended<T>>,
    ) -> Result<(Self, T)> {
        let next_branch = match self.node_data {
            // If the node is absent or blinded, the branch to be deserialised as a tree is absent.
            Partial::Absent | Partial::Blinded(_) => ProofTreeDeserialiser(ProofTree::Absent),
            Partial::Present(ref mut branches) => {
                branches
                    .pop_front()
                    .ok_or(ProofError::BadNumberOfBranches {
                        expected: 1,
                        got: 0,
                    })?
            }
        };

        let result = branch_deserialiser(next_branch)?.result;
        Ok((self, result))
    }

    fn done<T>(self, value: T) -> Result<<Self::Parent as Deserialiser>::Suspended<T>> {
        if let Partial::Present(branches) = self.node_data {
            if !branches.is_empty() {
                let length = branches.len();
                return Err(ProofError::BadNumberOfBranches {
                    expected: 0,
                    got: length,
                });
            }
        }

        Ok(OwnedParserComb {
            result: value,
            _pd: PhantomData,
        })
    }
}

impl<'t, R> Suspended for OwnedParserComb<'t, R> {
    type Output = R;

    type Parent = ProofTreeDeserialiser<'t>;

    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T,
    ) -> <Self::Parent as Deserialiser>::Suspended<T> {
        OwnedParserComb {
            result: f(self.result),
            _pd: PhantomData,
        }
    }
}

/// Given a [`ProofTree`] deserialise it into an allocated [`Verifier`] backend.
pub fn deserialise<L: ProofLayout>(
    proof: ProofTree,
) -> deserialiser::Result<(AllocatedOf<L, Verifier>, OwnedProofPart)> {
    let owned_proof = match proof {
        ProofPart::Absent => OwnedProofPart::Absent,
        ProofPart::Present(proof) => OwnedProofPart::Present(proof.clone()),
    };

    let context = ProofTreeDeserialiser::from(proof);
    let parser = L::into_verifier_alloc::<ProofTreeDeserialiser>(context)?;
    let result = parser.into_result();

    Ok((result, owned_proof))
}
