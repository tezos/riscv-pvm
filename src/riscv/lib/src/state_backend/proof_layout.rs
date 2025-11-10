// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;
use std::error;

use bincode::Decode;
use bincode::Encode;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof::DeserialiserError;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
use octez_riscv_data::merkle_proof::tag::InvalidTagError;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation;
use octez_riscv_data::tree::Tree;
use perfect_derive::perfect_derive;

use super::Array;
use super::Atom;
use super::DynArray;
use super::Layout;
use super::Many;
use super::RefVerifyAlloc;
use super::proof_backend::merkle::MERKLE_ARITY;
use super::proof_backend::merkle::MERKLE_LEAF_SIZE;
use super::proof_backend::proof::deserialiser::Result;
use super::verify_backend::PartialState;
use crate::array_utils::boxed_array;
use crate::state_backend::proof_backend::proof::NotEnoughBytesError;
use crate::state_backend::verify_backend::PageId;

/// Errors occurring when parsing the tag structure of a Merkle proof.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum TagError {
    #[error("Invalid tag encountered: {0}")]
    InvalidTag(#[from] InvalidTagError),

    #[error("Not enough bytes available")]
    NotEnoughBytes(#[from] NotEnoughBytesError),
}

/// Errors occurring when parsing a Merkle proof
#[derive(Debug, thiserror::Error)]
pub enum ProofError {
    #[error("Error during deserialisation: {0}")]
    Deserialise(#[from] DecodeError),

    #[error("Not enough bytes")]
    NotEnoughBytes(#[from] NotEnoughBytesError),

    #[error("Deserialising as a stream and not all bytes were consumed")]
    RemainingBytes,

    #[error("Error during tag deserialisation: {0}")]
    TagDeserialise(#[from] TagError),

    #[error("Proof tree is absent")]
    AbsentProof,

    #[error("A part of the proof required to parse further is absent")]
    DependentNodeIsAbsent,

    #[error("Encountered a node with a bad number of branches: expected {expected}, got {got}")]
    BadNumberOfBranches { expected: usize, got: usize },

    #[error("Expected a leaf of size {expected}, got {got}")]
    UnexpectedLeafSize { expected: usize, got: usize },

    #[error("Encountered a leaf where a node was expected")]
    UnexpectedLeaf,

    #[error("Encountered a node where a leaf was expected")]
    UnexpectedNode,

    #[error("Custom error: {0}")]
    Custom(Box<dyn std::error::Error>),
}

impl DeserialiserError for ProofError {
    fn custom<E: error::Error + 'static>(error: E) -> Self {
        // SAFETY: `ProofError` does not contain lifetimes, so unty-ing is safe.
        match unsafe { unty::unty(error) } {
            Ok(this) => this,
            Err(error) => Self::Custom(Box::new(error)),
        }
    }
}

/// Regions for the verifier backend for a specific layout.
pub type VerifyAlloc<L> = <L as Layout>::Allocated<Verify>;

/// Errors that may occur when hashing a state in [`Verify`] mode
#[derive(Debug, thiserror::Error)]
pub enum PartialHashError {
    /// The hash could not be computed because encoding a value to bytes failed. The byte
    /// representation is used as input to the hash function.
    #[error("Error while encoding a to-be-hashed value: {0}")]
    Encode(#[from] EncodeError),

    #[error("Error from proof: {0}")]
    FromProof(#[from] ProofError),

    /// Indicates that a hash could not be computed due to absent data,
    /// but from which it is possible to recover if the level at which
    /// it was raised is part of a blinded subtree and its hash is present
    /// in the proof.
    #[error("Potentially recoverable error")]
    PotentiallyRecoverable,

    /// Indicates that a hash could not be computed because the data being
    /// hashed is only partially available.
    #[error("Fatal error")]
    Fatal,
}

/// Part of a tree that may be absent
#[derive(Debug, PartialEq)]
#[perfect_derive(Clone, Copy)]
pub enum ProofPart<'a, T: ?Sized> {
    /// This part of the tree is absent.
    Absent,

    /// There is a proof for this part of the tree.
    Present(&'a T),
}

/// Part of a Merkle proof tree
pub type ProofTree<'a> = ProofPart<'a, MerkleProof>;

impl<'a> ProofTree<'a> {
    /// Interpret this part of the Merkle proof as a node with `LEN` branches.
    pub fn into_branches<const LEN: usize>(self) -> Result<Box<[Self; LEN]>> {
        let ProofTree::Present(proof) = self else {
            // The requested branches are not represented in the Merkle proof at all, not even
            // through a blinded node.
            return Ok(boxed_array![ProofTree::Absent; LEN]);
        };

        match proof {
            Tree::Node(branches) => {
                let branches: &[MerkleProof; LEN] =
                    branches.as_slice().try_into().map_err(|_| {
                        ProofError::BadNumberOfBranches {
                            got: branches.len(),
                            expected: LEN,
                        }
                    })?;
                Ok(branches
                    .iter()
                    .map(ProofTree::Present)
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| {
                        unreachable!(
                            "Converting a vector to an array of the same size always succeeds"
                        )
                    })
                    .unwrap())
            }

            Tree::Leaf(leaf) => match leaf {
                MerkleProofLeaf::Blind(_hash) => Ok(boxed_array![ProofTree::Absent; LEN]),
                _ => Err(ProofError::UnexpectedLeaf)?,
            },
        }
    }

    /// Interpret this part of the Merkle proof as a leaf.
    pub fn into_leaf(self) -> Result<ProofPart<'a, [u8]>> {
        if let ProofTree::Present(proof) = self {
            match proof {
                Tree::Node(_) => Err(ProofError::UnexpectedNode),
                Tree::Leaf(leaf) => match leaf {
                    MerkleProofLeaf::Blind(_) => Ok(ProofPart::Absent),
                    MerkleProofLeaf::Read(data) => Ok(ProofPart::Present(data.as_slice())),
                },
            }
        } else {
            Ok(ProofPart::Absent)
        }
    }

    /// For the purpose of computing the final hash of a state in `Verify` mode,
    /// interpret this part of a Merkle proof as a leaf and return its hash if
    /// it is a blinded leaf or hash the data if it is present.
    pub(crate) fn partial_hash_leaf(self) -> Result<Hash, PartialHashError> {
        let ProofTree::Present(proof) = self else {
            return Err(PartialHashError::PotentiallyRecoverable);
        };

        let Tree::Leaf(leaf) = proof else {
            return Err(ProofError::UnexpectedNode.into());
        };

        let hash = match leaf {
            MerkleProofLeaf::Blind(hash) => *hash,
            MerkleProofLeaf::Read(data) => Hash::blake3_hash_bytes(data),
        };

        Ok(hash)
    }

    /// For the purpose of computing the final hash of a state in `Verify` mode,
    /// if present, try to interpret this part of a Merkle proof as:
    /// - a node with `LEN` branches, in which case return the proof branches
    ///   and no proof hash
    /// - a blinded leaf which corresponds to a node with `LEN` children,
    ///   in which case return absent branches and the proof hash
    ///
    /// If the proof tree is absent, return absent branches and no proof hash.
    pub fn into_branches_with_hash<const LEN: usize>(
        self,
    ) -> Result<(Box<[ProofTree<'a>; LEN]>, Option<Hash>), PartialHashError> {
        let ProofTree::Present(proof) = self else {
            return Ok((boxed_array![ProofTree::Absent; LEN], None));
        };

        match proof {
            Tree::Node(branches) if branches.len() != LEN => Err(PartialHashError::FromProof(
                ProofError::BadNumberOfBranches {
                    got: branches.len(),
                    expected: LEN,
                },
            )),
            Tree::Node(branches) => Ok((
                branches
                    .iter()
                    .map(ProofTree::Present)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
                    .try_into()
                    .map_err(|_| PartialHashError::Fatal)?,
                None,
            )),
            Tree::Leaf(leaf) => match leaf {
                MerkleProofLeaf::Blind(hash) => {
                    Ok((boxed_array![ProofTree::Absent; LEN], Some(*hash)))
                }
                _ => Err(ProofError::UnexpectedLeaf)?,
            },
        }
    }
}

/// Similar to [`ProofPart`], but owns the underlying [`MerkleProof`].
#[derive(Clone)]
pub enum OwnedProofPart {
    /// This part of the tree is absent.
    Absent,
    /// There is a proof for this part of the tree.
    Present(MerkleProof),
}

impl OwnedProofPart {
    /// Obtain an [`OwnedProofPart`] from a [`Partial<T>`] considering it a leaf.
    pub fn leaf_from_partial<T>(partial: Partial<T>, f: impl FnOnce(T) -> Vec<u8>) -> Self {
        match partial {
            Partial::Absent => OwnedProofPart::Absent,
            Partial::Blinded(hash) => OwnedProofPart::Present(MerkleProof::leaf_blind(hash)),
            Partial::Present(data) => OwnedProofPart::Present(MerkleProof::leaf_read(f(data))),
        }
    }

    /// Obtain an [`OwnedProofPart`] from a [`Partial<Vec<MerkleProof>>`] considering it a node.
    pub fn node_from_partial(partial: Partial<Vec<MerkleProof>>) -> Self {
        match partial {
            Partial::Absent => OwnedProofPart::Absent,
            Partial::Blinded(hash) => OwnedProofPart::Present(MerkleProof::leaf_blind(hash)),
            Partial::Present(children) => OwnedProofPart::Present(MerkleProof::Node(children)),
        }
    }

    /// Construct a node from its child proofs. The `parent` parameter allows us to restruct the
    /// blinded state of the parent.
    pub fn node_from_children(
        parent: Partial<()>,
        children: impl IntoIterator<Item = Self>,
    ) -> Self {
        match parent {
            Partial::Absent => return OwnedProofPart::Absent,
            Partial::Blinded(hash) => {
                return OwnedProofPart::Present(MerkleProof::leaf_blind(hash));
            }
            Partial::Present(_) => {}
        }

        let mut partial_children = Vec::with_capacity(MERKLE_ARITY);

        for item in children {
            match item {
                OwnedProofPart::Absent => return OwnedProofPart::Absent,
                OwnedProofPart::Present(tree) => partial_children.push(tree),
            }
        }

        OwnedProofPart::Present(MerkleProof::Node(partial_children))
    }
}

/// [`Layouts`] which may be used in a Merkle proof
///
/// [`Layouts`]: crate::state_backend::Layout
pub trait ProofLayout: Layout {
    /// Compute the state hash of a partial state in `Verify` mode using its
    /// corresponding proof tree where data is missing.
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError>;
}

impl<T: ProofLayout> ProofLayout for Box<T> {
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        T::partial_state_hash(*state, proof)
    }
}

impl<T> ProofLayout for Atom<T>
where
    T: Encode + Decode<()> + 'static,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let region = state.into_region();
        match region.get_partial_region() {
            PartialState::Complete(region) => Ok(Hash::blake3_hash(region)?),
            PartialState::Absent => proof.partial_hash_leaf(),
            PartialState::Incomplete => Err(PartialHashError::Fatal),
        }
    }
}

impl<T, const LEN: usize> ProofLayout for Array<T, LEN>
where
    T: Encode + Decode<()> + 'static,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let region = state.into_region();
        match region.get_partial_region() {
            PartialState::Complete(region) => Ok(Hash::blake3_hash(region)?),
            PartialState::Absent => proof.partial_hash_leaf(),
            PartialState::Incomplete => Err(PartialHashError::Fatal),
        }
    }
}

impl ProofLayout for DynArray {
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let region = state.region_ref();

        let proof = match proof {
            ProofPart::Absent => {
                if !region.is_completely_absent() {
                    // The verifier contains data, but the proof is not in the right shape for us
                    // to insert back the data and obtain a root hash. This indicates an invalid
                    // proof.
                    return Err(PartialHashError::Fatal);
                }

                // The dynamic region is untouched, so re-hashing needs to resume higher up in the
                // proof tree.
                return Err(PartialHashError::PotentiallyRecoverable);
            }

            ProofPart::Present(proof) => proof,
        };

        let children = match proof {
            Tree::Leaf(MerkleProofLeaf::Blind(hash)) => {
                if !region.is_completely_absent() {
                    // The verifier contains data, but the proof is not in the right shape for us
                    // to insert back the data and obtain a root hash. This indicates an invalid
                    // proof.
                    //
                    // From a partial Merkle tree perspective, technically you can blind a node in
                    // the proof and then write all the data necessary to expand the node fully
                    // after the verification is done. In the context of a dynamic region that
                    // would entail setting the length and writing all pages. However, dynamic
                    // regions cannot be re-created hence you cannot "write" the length. Any
                    // modification requires at least the length node to be present. Hence,
                    // blinding the pages node can be paired with modifications, but you cannot
                    // blind the dynamic region's root node if there are modifications.
                    return Err(PartialHashError::Fatal);
                }

                // We already know that the dynamic region has been untouched at this point.
                return Ok(*hash);
            }

            Tree::Leaf(MerkleProofLeaf::Read(_)) => {
                return Err(PartialHashError::FromProof(ProofError::UnexpectedLeaf));
            }

            Tree::Node(children) => children.as_slice(),
        };

        let [length_tree, pages_tree] = children else {
            return Err(PartialHashError::FromProof(
                ProofError::BadNumberOfBranches {
                    expected: 2,
                    got: children.len(),
                },
            ));
        };

        let length = if let Tree::Leaf(MerkleProofLeaf::Read(data)) = length_tree {
            // The length data must be present if the node is present. Practically, if there is any
            // pages data to be dealt with we require the length. Or if there is no pages data,
            // then the only reason the parent node would be present is if the length was to be
            // read during verification.
            serialisation::deserialise::<u64>(data).map_err(ProofError::Deserialise)? as usize
        } else {
            return Err(PartialHashError::Fatal);
        };

        enum Event<'a> {
            Span(usize, usize, ProofTree<'a>),
            Node(Option<Hash>),
        }

        let mut queue = VecDeque::new();
        queue.push_back(Event::Span(0usize, length, ProofTree::Present(pages_tree)));

        let mut hashes: Vec<Result<Hash, PartialHashError>> = Vec::new();

        while let Some(event) = queue.pop_front() {
            match event {
                Event::Span(start, length, tree) => {
                    if length <= MERKLE_LEAF_SIZE.get() {
                        // TODO RV-463: Leaves smaller than `MERKLE_LEAF_SIZE` should also be accepted.
                        // The span's size if that of a leaf, obtain its hash if possible and push
                        // the result to the `hashes` stack.
                        match state
                            .region_ref()
                            .get_partial_page(PageId::from_address(start))
                        {
                            PartialState::Absent => hashes.push(tree.partial_hash_leaf()),
                            PartialState::Complete(data) => {
                                hashes.push(Ok(Hash::blake3_hash_bytes(data)))
                            }
                            PartialState::Incomplete => {
                                return Err(PartialHashError::Fatal);
                            }
                        };
                    } else {
                        // TODO RV-463: Nodes with fewer than `MERKLE_ARITY` children should also be accepted.
                        // The span's size is that of a node, produce `Event::Span` work items for each of its
                        // children and add them to the work queue, followed by an `Event::Node`.
                        let (branches, proof_hash) =
                            tree.into_branches_with_hash::<{ MERKLE_ARITY }>()?;

                        push_work_items_for_branches(
                            start,
                            length,
                            branches.as_ref(),
                            |branch_start, branch_length, branch| {
                                queue.push_back(Event::Span(branch_start, branch_length, branch));
                            },
                        );

                        queue.push_back(Event::Node(proof_hash));
                    }
                }
                Event::Node(proof_hash) => {
                    if hashes.is_empty() {
                        // The hashes which need to be combined have not yet been computed because
                        // their processing resulted in more `Event::Span` items. Push to the back
                        // of the work queue.
                        queue.push_back(Event::Node(proof_hash));
                        continue;
                    }
                    if hashes.len() < MERKLE_ARITY {
                        return Err(PartialHashError::Fatal);
                    };

                    // Take `MERKLE_ARITY` children hashes, compute their parent's hash, and
                    // push it to the `hashes` stack.
                    let node_hashes: Vec<_> = hashes.drain(hashes.len() - MERKLE_ARITY..).collect();
                    hashes.push(combine_partial_hashes(node_hashes, proof_hash))
                }
            }
        }

        // There must only be a single hash, the root hash. If there are more or less, that is an
        // error.
        let pages_hash = hashes
            .pop()
            .filter(|_| hashes.is_empty())
            .map_or(Err(PartialHashError::Fatal), |hash| hash)?;

        let length_hash = Hash::blake3_hash(length as u64)?;
        let root_hash = Hash::combine([length_hash, pages_hash]);

        Ok(root_hash)
    }
}

impl<A, B> ProofLayout for (A, B)
where
    A: ProofLayout,
    B: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<2>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C> ProofLayout for (A, B, C)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<3>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C, D> ProofLayout for (A, B, C, D)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
    D: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<4>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
            D::partial_state_hash(state.3, branches[3]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C, D, E> ProofLayout for (A, B, C, D, E)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
    D: ProofLayout,
    E: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<5>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
            D::partial_state_hash(state.3, branches[3]),
            E::partial_state_hash(state.4, branches[4]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C, D, E, F> ProofLayout for (A, B, C, D, E, F)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
    D: ProofLayout,
    E: ProofLayout,
    F: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<6>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
            D::partial_state_hash(state.3, branches[3]),
            E::partial_state_hash(state.4, branches[4]),
            F::partial_state_hash(state.5, branches[5]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<T, const LEN: usize> ProofLayout for [T; LEN]
where
    T: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<LEN>()?;
        let hashes = state
            .into_iter()
            .zip(branches.iter())
            .map(|(state, proof)| T::partial_state_hash(state, *proof))
            .collect::<Vec<Result<Hash, PartialHashError>>>();
        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<T, const LEN: usize> ProofLayout for Many<T, LEN>
where
    T: ProofLayout,
{
    fn partial_state_hash(
        state: RefVerifyAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        enum Event<'a> {
            Span(usize, usize, ProofTree<'a>),
            Node(Option<Hash>),
        }

        // `T::partial_state_hash` needs to take ownership of the elements of `state`.
        // Given that `T` is not `Copy`, in order to take ownership of arbitrary elements
        // of `state` we'd first need to duplicate it and wrap each element in a type
        // which supports taking ownership.
        // However, in practice, we compute the hash of each element sequentially, meaning
        // that we can simply iterate over the state directly when calling `T::partial_state_hash`.
        let mut state = state.into_iter();
        let mut next_vec_index = 0;

        let mut queue = VecDeque::new();
        queue.push_back(Event::Span(0usize, LEN, proof));

        let mut hashes: Vec<Result<Hash, PartialHashError>> = Vec::new();

        while let Some(event) = queue.pop_front() {
            match event {
                Event::Span(start, length, tree) => {
                    if length == 1 {
                        // Check that iterating over the state is equivalent to calling `state[start]`
                        debug_assert_eq!(start, next_vec_index);
                        next_vec_index += 1;
                        hashes.push(T::partial_state_hash(
                            state.next().ok_or(PartialHashError::Fatal)?,
                            tree,
                        ))
                    } else {
                        // TODO RV-463: Nodes with fewer than `MERKLE_ARITY` children should also be accepted.
                        // The span's size is that of a node, produce `Event::Span` work items for each of its
                        // children and add them to the work queue, followed by an `Event::Node`.
                        let (branches, proof_hash) =
                            tree.into_branches_with_hash::<{ MERKLE_ARITY }>()?;

                        push_work_items_for_branches(
                            start,
                            length,
                            branches.as_ref(),
                            |branch_start, branch_length, branch| {
                                queue.push_back(Event::Span(branch_start, branch_length, branch));
                            },
                        );

                        queue.push_back(Event::Node(proof_hash));
                    }
                }
                Event::Node(proof_hash) => {
                    if hashes.is_empty() {
                        // The hashes which need to be combined have not yet been computed because
                        // their processing resulted in more `Event::Span` items. Push to the back
                        // of the work queue.
                        queue.push_back(Event::Node(proof_hash));
                        continue;
                    }
                    if hashes.len() < MERKLE_ARITY {
                        return Err(PartialHashError::Fatal);
                    };

                    // Take `MERKLE_ARITY` children hashes, compute their parent's hash, and
                    // push it to the `hashes` stack.
                    let node_hashes: Vec<_> = hashes.drain(hashes.len() - MERKLE_ARITY..).collect();
                    hashes.push(combine_partial_hashes(node_hashes, proof_hash))
                }
            }
        }

        // Check that we iterated over all the elements of the state
        debug_assert_eq!(next_vec_index, LEN);

        // There must only be a single hash, the root hash. If there are more or less, that is an
        // error.
        hashes
            .pop()
            .filter(|_| hashes.is_empty())
            .map_or(Err(PartialHashError::Fatal), |hash| hash)
    }
}

/// Attempt to compute the partial hash of a node from its children's partial
/// hashes if they are present. If none of the children hashes can be computed
/// due to absent data, this node is either a blinded leaf in the proof, in which
/// case its hash can be recovered from the proof, or it is part of a blinded
/// subtree whose hash cannot be computed as this point.
pub fn combine_partial_hashes(
    hash_results: impl AsRef<[Result<Hash, PartialHashError>]>,
    proof_hash: Option<Hash>,
) -> Result<Hash, PartialHashError> {
    let hash_results = hash_results.as_ref();
    if hash_results.is_empty() {
        return Ok(Hash::combine::<Hash, _>([]));
    }

    // If the first result is a hash, all results need to be a hash in order to
    // compute the combined hash. If the first result is a potentially
    // recoverable error, all results need to to be potentially recoverable
    // errors in order to fall back on the proof hash. Anything else is a fatal error.
    let expect_ok = match hash_results[0] {
        Ok(_) => true,
        Err(PartialHashError::PotentiallyRecoverable) => false,
        _ => return Err(PartialHashError::Fatal),
    };

    let mut hashes = Vec::with_capacity(hash_results.len());
    let hash_results_len = hash_results.len();
    for r in hash_results {
        match r {
            Ok(hash) if expect_ok => hashes.push(*hash),
            Err(PartialHashError::PotentiallyRecoverable) if !expect_ok => (),
            _ => return Err(PartialHashError::Fatal),
        }
    }

    if expect_ok {
        debug_assert_eq!(hashes.len(), hash_results_len);
        return Ok(Hash::combine(hashes));
    };

    proof_hash.ok_or(PartialHashError::PotentiallyRecoverable)
}

fn work_merkle_params<const CHILDREN: usize>(
    mut branch_start: usize,
    mut length_left: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let branch_max_length = length_left.div_ceil(MERKLE_ARITY);

    (0..CHILDREN).map(move |_| {
        let this_branch_length = branch_max_length.min(length_left);

        let item = (branch_start, this_branch_length);

        branch_start = branch_start.saturating_add(this_branch_length);
        length_left = length_left.saturating_sub(this_branch_length);

        item
    })
}

fn push_work_items_for_branches<'a, const CHILDREN: usize>(
    branch_start: usize,
    length_left: usize,
    branches: &'_ [ProofTree<'a>; CHILDREN],
    mut push: impl FnMut(usize, usize, ProofTree<'a>),
) {
    let children = work_merkle_params::<CHILDREN>(branch_start, length_left);
    for (branch, (child_start, child_length)) in branches.iter().zip(children) {
        if child_length > 0 {
            push(child_start, child_length, *branch);
        }
    }
}

#[cfg(test)]
mod tests {
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Prove;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::*;
    use crate::state_backend::AllocatedOf;
    use crate::state_backend::Cells;
    use crate::state_backend::DynCells;
    use crate::state_backend::FnManagerIdent;
    use crate::state_backend::ManagerWrite;
    use crate::state_backend::proof_backend::ProofRegion;
    use crate::state_backend::proof_backend::ProofWrapper;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned;
    use crate::state_backend::verify_backend::handle_stepper_panics;

    const CELLS_SIZE: usize = 32;

    // When producing a proof from a state in `Prove` mode, values written during
    // the execution of the tick being proven should not be blinded, whereas
    // values which were not accessed should be blinded. When a proof contains
    // blinded values, it should be possible to compute the final hash of the
    // state in `Verify` mode constructed from this proof.
    #[test]
    fn test_proof_blinding() {
        type TestLayout = (Array<u64, CELLS_SIZE>, Array<u64, CELLS_SIZE>);

        proptest!(|(value_before: u64, value_after: u64, i in 0..CELLS_SIZE)| {
            // Bind `Prove` cells and write at one address
            let cells1 = [value_before; CELLS_SIZE];
            let mut proof_region1: ProofRegion<u64, CELLS_SIZE> = ProofRegion::bind(&cells1);
            Prove::region_write(&mut proof_region1, i, value_after);
            let proof_cells1: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(proof_region1);

            // Bind `Prove` cells and do not access them
            let cells2 = [value_before; CELLS_SIZE];
            let proof_region2: ProofRegion<u64, CELLS_SIZE> = ProofRegion::bind(&cells2);
            let proof_cells2: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(proof_region2);

            let proof_state = (proof_cells1, proof_cells2);

            let merkle_proof = merkle_tree_to_merkle_proof(MerkleTree::from_foldable(&proof_state));

            let verifier_state =
                deserialise_owned::deserialise::<AllocatedOf<TestLayout, Verify>>(
                    ProofTree::Present(&merkle_proof),
                ).unwrap();

            // The first component of the state was present in the proof, can be
            // fully read, and contains the initial state.
            prop_assert_eq!(verifier_state.0.0.read_all(), vec![value_before; CELLS_SIZE]);

            // The second component of the state is fully blinded: no values can
            // be read from the array.
            for i in 0..CELLS_SIZE {
                prop_assert!(handle_stepper_panics(|| verifier_state.0.1.read(i)).is_err());
            };

            let ref_verifier_state = (
                verifier_state.0.0.struct_ref::<FnManagerIdent>(),
                verifier_state.0.1.struct_ref::<FnManagerIdent>(),
            );
            prop_assert!(
                <TestLayout as ProofLayout>::partial_state_hash(ref_verifier_state, ProofTree::Present(&merkle_proof)).is_ok()
            );
        })
    }

    /// Test the proof generation and verification for a computation against a dynamic region.
    ///
    /// # Safety
    ///
    /// The `test_proof` and `test_verify` function must be the same function instantiated to
    /// different managers.
    ///
    /// Due to Rust's limitation on higher-ranked polymorphism, we can't accept
    /// a single function and instantiate it within the function body with the respective modes
    /// `Prove<_>` and `Verify`. One could work around this restriction by using a trait to
    /// simulate the rank-2-ness, but that means you can't provide closures as the implementation
    /// any more. If any of the given `test_proof` or `test_verify` capture an environment, this
    /// would no longer work.
    unsafe fn test_dyn_array_with_funs(
        len: usize,
        test_proof: impl FnOnce(&mut DynCells<Prove>),
        test_verify: impl FnOnce(&mut DynCells<Verify>),
    ) {
        let owned_cell = DynCells::new(len);

        // We require the initial hash to ensure that the generated proof, but also the
        // instantiated state from the proof match the "before" state.
        let init_hash = Hash::from_foldable(&owned_cell);

        // The `ProofWrapper` transformer ensures the resulting dynamic region (via `DynCells`) is
        // setup for proof generation. You can think of this as starting the recording for a proof.
        let mut proof_cell = owned_cell.struct_ref::<ProofWrapper>();

        test_proof(&mut proof_cell);

        // The post-hash is required to ensure that the verifier's final state matches the prover's
        // final state.
        let post_hash = Hash::from_foldable(&proof_cell);

        let tree = MerkleTree::from_foldable(&proof_cell);
        let proof_tree = merkle_tree_to_merkle_proof(tree);
        assert_eq!(proof_tree.root_hash(), init_hash);

        // Instantiating the verifier state allows us to replay the computation and verify it does
        // the right things.
        let (mut verify_cell, out_proof) = deserialise_owned::deserialise::<
            AllocatedOf<DynArray, Verify>,
        >(ProofTree::Present(&proof_tree))
        .unwrap();

        let OwnedProofPart::Present(out_proof) = out_proof else {
            panic!("Expected present proof");
        };
        assert_eq!(proof_tree, out_proof);

        let out_proof_tree = ProofTree::Present(&out_proof);

        // The initial verifier state must match that of the initial state against which we
        // produced the proof.
        let verifier_init_hash = {
            let state_ref = verify_cell.struct_ref::<FnManagerIdent>();
            DynArray::partial_state_hash(state_ref, out_proof_tree).unwrap()
        };
        assert_eq!(verifier_init_hash, init_hash);

        test_verify(&mut verify_cell);

        // Once we're doing replaying the computation on the verifier side, the final state must
        // match that of the prover's. If not, that means we produced a proof that results in a
        // transition that we did not intend to prove.
        let verifier_post_hash = {
            let state_ref = verify_cell.struct_ref::<FnManagerIdent>();
            DynArray::partial_state_hash(state_ref, out_proof_tree).unwrap()
        };
        assert_eq!(verifier_post_hash, post_hash);
    }

    /// Generate a test for dynamic regions using a given size and closure which operates on the
    /// [`DynCells`]. This effectively demonstrates that the actions performed by the given closure
    /// can be proven and verified correctly.
    macro_rules! test_dyn_array_with {
        ($len:literal, | $param:ident | { $($body:tt)* }) => {
            {
                let test_proof = |$param: &mut DynCells<Prove>| {
                    $($body)*
                };

                let test_verify = |$param: &mut DynCells<Verify>| {
                    $($body)*
                };

                // SAFETY: This function is intended to be used only in this macro.
                unsafe {
                    test_dyn_array_with_funs($len, test_proof, test_verify);
                }
            }
        };
    }

    #[test]
    fn test_dyn_array_proofs_nothing() {
        test_dyn_array_with!(65536, |_cell| {});
    }

    #[test]
    fn test_dyn_array_proofs_read() {
        proptest!(|(addr in 0..65528usize)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.read::<u64>(addr);
                }
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_write() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.write::<u64>(addr, val);
                }
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_len() {
        test_dyn_array_with!(65536, |cell| {
            cell.len();
        });
    }

    #[test]
    fn test_dyn_array_proofs_read_and_len() {
        proptest!(|(addr in 0..65528usize)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.read::<u64>(addr);
                }

                cell.len();
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_write_and_len() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.write::<u64>(addr, val);
                }

                cell.len();
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_read_and_write() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    let x = cell.read::<u64>(addr);
                    cell.write(addr, x.wrapping_add(val));
                }
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_read_and_write_and_len() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    let x = cell.read::<u64>(addr);
                    cell.write(addr, x.wrapping_add(val));
                }

                cell.len();
            });
        });
    }
}
